"""Python driver for the GPU TreeSA binary (built by `make` in the repository root into treesa_gpu/bin/).

    from treesa_gpu import optimize
    tree, info = optimize(ixs, iy, nchains=2048, niters=20, sc_target=30, mode="tc", init="greedy")

``tree`` is an omeco-format dict (``{"tensor_index": i}`` / ``{"args": [l, r], "eins": {...}}``) with
internal ``iy`` recomputed from the topology; ``info`` holds tc/sc (recomputed in Python with
``trees.complexity``), modeled GPU time, timings, throughput and the annealing trace.

The binary runs on the local GPU. Optional environment variables:
  TREESA_GPU_BINARY  path of the binary (default: treesa_gpu/bin/treesa_gpu inside this package;
                     e.g. bin/treesa_gpu_sm90 for an H100 built with `make ARCH=sm_90a OUT=...`)
  TREESA_GPU_SSH     run the binary on this host via ssh (the temp dir must be on a shared filesystem)
  TREESA_GPU_TMP     temp dir used with TREESA_GPU_SSH (default ~/.cache/treesa_gpu)
  TREESA_GPU_LOCK    wrap the run in ``flock <path>`` (to serialize with other GPU jobs)
"""
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time

from . import trees

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.environ.get("TREESA_GPU_BINARY", os.path.join(HERE, "bin", "treesa_gpu"))
SSH_HOST = os.environ.get("TREESA_GPU_SSH", "")
LOCK = os.environ.get("TREESA_GPU_LOCK", "")
SHARED_TMP = os.environ.get("TREESA_GPU_TMP", os.path.join(os.path.expanduser("~"), ".cache", "treesa_gpu"))

DEFAULT_GPU_MODEL = dict(trees.GPU_MODEL)
DEFAULT_H100_MODEL = dict(trees.H100_MODEL)
MODE_ALIASES = {"gpu2": "h100", "tc+gpu2": "tc+h100"}   # old mode names
H100_KEYS = ("launch", "perm_bw_bits", "cc_bit", "cc_cell", "tc_bit", "tc_pad_cell", "tc_out_bit", "small_rank",
            "small_cell")


# ---------------------------------------------------------------------------------------------
# instance / tree conversion
# ---------------------------------------------------------------------------------------------
def label_map(ixs, iy):
    """Map arbitrary hashable labels to 0..L-1 (order of first appearance)."""
    mp = {}
    for ix in list(ixs) + [iy]:
        for l in ix:
            if l not in mp:
                mp[l] = len(mp)
    return mp


def write_instance(path, ixs, iy):
    mp = label_map(ixs, iy)
    with open(path, "w") as f:
        f.write(f"{len(ixs)} {len(mp)}\n")
        for ix in ixs:
            f.write(" ".join([str(len(ix))] + [str(mp[l]) for l in ix]) + "\n")
        f.write(" ".join([str(len(iy))] + [str(mp[l]) for l in iy]) + "\n")
    return mp


def dict_to_children(d, m):
    """omeco dict -> list of (left, right) for internal nodes m..2m-2 (post-order ids, root last)."""
    ch = []

    def rec(t):
        if "tensor_index" in t:
            return t["tensor_index"]
        a = rec(t["args"][0])
        b = rec(t["args"][1])
        ch.append((a, b))
        return m + len(ch) - 1

    root = rec(d)
    assert root == 2 * m - 2 and len(ch) == m - 1, "tree must be binary over all m tensors"
    return ch


def children_to_dict(ch, m, ixs, iy):
    """Children list (node ids, root = 2m-2) -> omeco dict with topology-derived internal iy."""
    sys.setrecursionlimit(max(10000, 4 * m))

    def rec(v):
        if v < m:
            return {"tensor_index": v}
        l, r = ch[v - m]
        return {"args": [rec(l), rec(r)], "eins": {"ixs": [[], []], "iy": []}}

    return trees.recompute_outputs(rec(2 * m - 2), ixs, iy)


def check_tree(d, m):
    """Every leaf exactly once, binary."""
    ids = trees.leaf_ids(d)
    return sorted(ids) == list(range(m))


def greedy_tree(ixs, iy):
    import omeco
    sizes = {l: 2 for ix in ixs for l in ix}
    for l in iy:
        sizes[l] = 2
    return omeco.optimize_code(ixs, iy, sizes, omeco.GreedyMethod()).to_dict()


def julia_betas(start=0.01, step=0.05, stop=15.0):
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [start + i * step for i in range(n)]


# ---------------------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------------------
def optimize(ixs, iy, nchains=1024, betas=None, niters=50, sc_target=20.0, sc_weight=1.0, mode="tc",
             init="greedy", seed=1, scheme="indep", kernel="auto", trunc_frac=0.25, eval_every=1,
             gpu_model=None, h100_model=None, wpb=0, block=0, lpc=32, init_tree=None, stage2_betas=None,
             stage2_niters=None, slicing=False, slice_moves=1, slice_every=1, max_slices=0, init_slices=None,
             wide=False, verbose=False, timeout=3600, **old_names):
    """Run GPU TreeSA.  Returns (tree_dict, info).

    slicing: anneal a slice set S together with the tree (see README).  Then ``tree`` is the tree of
            the *sliced* network: its internal ``iy`` are recomputed on ``ixs`` with the sliced labels
            removed (``trees.recompute_outputs(tree, ixs_sliced, iy_sliced)``), and
            ``info["sliced"]`` lists the sliced labels (original label ids).  ``info`` reports per-slice
            tc/sc/modeled times recomputed with trees.py on the sliced network (``tc_py``, ``sc_py``,
            ``h100_time_py``) and totals over the 2^|S| slices (``tc_total_py``, ``h100_time_total_py``).
            ``runner_json(tree, info)`` gives the ``{"tree", "sliced"}`` dict for the H100 runner.
            sc_target is a feasibility target: the best chain with sc <= sc_target is returned, and if
            none reaches it the result is repaired greedily (Julia TreeSASlicer rule; ``info["n_repair"]``).

    mode:   "tc"  -> Julia/omeco energy (tc, sc with sc_target/sc_weight)
            "gpu" -> GPU-runtime-aware energy with a simple roofline model (trees.gpu_time; ``gpu_model``)
            "h100" -> modeled time of the booltn kernels measured on an H100 (trees.h100_time; constants from
                     ``h100_model``, default trees.H100_MODEL). Old name: "gpu2".
            "tc+gpu" / "tc+h100" -> anneal with "tc" on ``betas``, then continue the same population with "gpu"/"h100"
                     on ``stage2_betas`` (default 1:0.05:16, i.e. cold; ``stage2_niters`` defaults to
                     ``niters``).  Recommended for the GPU-time objective: "gpu" alone from a hot start
                     gets stuck (the launch-overhead floor and tile padding make the energy flat).
    init:   "greedy" (omeco GreedyMethod, replicated to all chains) | "random" (Julia random_exprtree
            per chain) | "tree" (use ``init_tree`` dict)
    scheme: "indep" (independent chains, best-of-N) | "pa" (population annealing, resampling after
            every evaluated level) | "trunc" (replace worst ``trunc_frac`` by copies of the best)
    kernel: "auto" (warp per chain; chains staged in shared memory when the population exceeds L2)
            | "warp" | "warpsm" | "seq" | "seqsm";  lpc = lanes per chain for warp kernels (32, 16, 8)
    """
    if old_names:   # names before the gpu2 -> h100 rename
        unknown = set(old_names) - {"gpu_model2"}
        if unknown:
            raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
        h100_model = h100_model or old_names["gpu_model2"]
    mode = MODE_ALIASES.get(mode, mode)
    if betas is None:
        betas = julia_betas()
    gm = dict(DEFAULT_GPU_MODEL)
    if gpu_model:
        gm.update(gpu_model)
    h100m = dict(DEFAULT_H100_MODEL)
    if h100_model:
        h100m.update(h100_model)
    m = len(ixs)
    if SSH_HOST:  # the remote host must see the temp files: use a directory on the shared home
        os.makedirs(SHARED_TMP, exist_ok=True)
    tmpdir = tempfile.mkdtemp(dir=SHARED_TMP if SSH_HOST else None)
    inst = os.path.join(tmpdir, "inst.txt")
    lmap = write_instance(inst, ixs, iy)
    inv = {v: k for k, v in lmap.items()}
    t_greedy = 0.0
    args = [BINARY, "-i", inst]
    if init in ("greedy", "tree"):
        t0 = time.time()
        d0 = greedy_tree(ixs, iy) if init == "greedy" else init_tree
        t_greedy = time.time() - t0
        tf = os.path.join(tmpdir, "tree.txt")
        with open(tf, "w") as f:
            for a, b in dict_to_children(d0, m):
                f.write(f"{a} {b}\n")
        args += ["--tree", tf]
    bf = os.path.join(tmpdir, "betas.txt")
    with open(bf, "w") as f:
        f.write("\n".join(repr(float(b)) for b in betas) + "\n")
    if mode in ("tc+gpu", "tc+h100"):
        b2 = stage2_betas if stage2_betas is not None else julia_betas(1.0, 0.05, 16.0)
        bf2 = os.path.join(tmpdir, "betas2.txt")
        with open(bf2, "w") as f:
            f.write("\n".join(repr(float(b)) for b in b2) + "\n")
        args += ["--stage2-betas", "file:" + bf2, "--stage2-niters", str(stage2_niters or niters),
                 "--stage2-mode", mode.split("+")[1]]
    out = os.path.join(tmpdir, "out.json")
    args += ["--betas", "file:" + bf, "--nchains", str(nchains), "--niters", str(niters),
             "--sc-target", str(sc_target), "--sc-weight", str(sc_weight), "--mode", mode if mode in ("gpu", "h100") else "tc",
             "--scheme", scheme, "--kernel", kernel, "--seed", str(seed), "--trunc-frac", str(trunc_frac),
             "--eval-every", str(eval_every), "--out", out,
             "--gm", ",".join(str(gm[k]) for k in ("peak_ops", "bw_bits", "m0", "n0", "k0", "launch")),
             "--h100-model", ",".join(str(h100m[k]) for k in H100_KEYS)]
    if slicing:
        args += ["--slicing", "--slice-moves", str(slice_moves), "--slice-every", str(slice_every)]
        if max_slices:
            args += ["--max-slices", str(max_slices)]
        if init_slices:
            args += ["--init-slices", ",".join(str(lmap[l]) for l in init_slices)]
    if wide:
        args += ["--wide"]
    if wpb:
        args += ["--wpb", str(wpb)]
    if lpc != 32:
        args += ["--lpc", str(lpc)]
    if block:
        args += ["--block", str(block)]
    if not verbose:
        args += ["--quiet"]
    cmd = " ".join(shlex.quote(a) for a in args)
    if LOCK:
        cmd = f"flock {LOCK} {cmd}"
    if SSH_HOST:
        cmd = f"ssh {SSH_HOST} {shlex.quote('bash -lc ' + shlex.quote('module load cuda/12.8.0 >/dev/null 2>&1; ' + cmd))}"
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    if verbose and p.stderr:
        print(p.stderr, file=sys.stderr)
    if p.returncode != 0 and not os.path.exists(BINARY) and not SSH_HOST:
        raise RuntimeError(f"treesa_gpu binary {BINARY} not found: build it with `make` in the treesa_gpu repository")
    if p.returncode != 0:
        raise RuntimeError(f"treesa_gpu failed ({p.returncode}): {p.stderr[-2000:]}")
    with open(out) as f:
        res = json.load(f)
    sliced = [inv[i] for i in res["sliced"]]
    sl = set(sliced)
    ixs_s = [[l for l in ix if l not in sl] for ix in ixs] if sl else ixs
    iy_s = [l for l in iy if l not in sl]
    tree = children_to_dict(res["tree"], m, ixs_s, iy_s)
    assert check_tree(tree, m), "output tree does not contain every leaf exactly once"
    tc, sc = trees.complexity(tree, ixs_s)
    info = dict(res)
    del info["tree"]
    t1 = trees.tree_gpu_time(tree, ixs_s, gm)
    t2 = trees.tree_h100_time(tree, ixs_s, h100m)
    info.update(sliced=sliced, tc_py=tc, sc_py=sc, gpu_time_py=t1, h100_time_py=t2, tc_total_py=tc + len(sliced),
                h100_time_total_py=t2 * 2.0 ** len(sliced), wall=wall, t_greedy=t_greedy, tmpdir=tmpdir)
    for fn in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, fn))
    os.rmdir(tmpdir)
    return tree, info


def runner_json(tree, info):
    """{"tree": dict with iy on the sliced ixs, "sliced": [original labels]} for the H100 runner."""
    return {"tree": tree, "sliced": list(info.get("sliced", []))}


if __name__ == "__main__":  # smoke test: python -m treesa_gpu.driver [n]  (random 3-SAT, alpha = 4.26)
    import random
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = random.Random(1)
    ixs, iy = [rng.sample(range(n), 3) for _ in range(round(4.26 * n))], []
    tree, info = optimize(ixs, iy, nchains=512, niters=10, sc_target=30, verbose=True)
    print({k: v for k, v in info.items() if k != "trace"})
