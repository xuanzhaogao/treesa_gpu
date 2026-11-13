"""Slicing comparison: omeco TreeSA + omeco slice_code(TreeSASlicer)  vs  greedy_slice (booltn/booltn/
slicer.py, kind="gpu") applied to GPU-TreeSA unsliced trees  vs  GPU TreeSA with in-annealer slicing.
Also evaluates the existing trees in booltn-gpu/trees/ for the same instance and sc target.

All numbers are recomputed with treesa_gpu.trees on the sliced network (ixs with sliced labels removed):
per-slice tc / sc, #sliced, total tc = per-slice tc + #sliced, modeled total H100 time =
tree_h100_time x 2^#sliced.  Trees of the in-annealer runs are saved to
booltn-gpu/trees/gpusa_slice/<name>_sc<T>_<tc|h100>.json as {"tree", "sliced"} (iy on the sliced ixs).

    python bench/slicing_compare.py [--instances ...] [--targets 30,34,36,37] > log
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import instances  # noqa: E402
import treesa_gpu as tg  # noqa: E402  (the package next to bench/)
from treesa_gpu import trees as tu  # noqa: E402

# trees/ of the booltn-gpu research repository (this package is its submodule); override with $BOOLTN_TREES
TREES = os.environ.get("BOOLTN_TREES", os.path.join(HERE, "..", "..", "trees"))
OUTDIR = os.path.join(TREES, "gpusa_slice")
RES = os.path.join(HERE, "results", "slicing_compare.jsonl")
HOST = os.uname().nodename.split(".")[0]
NAMES = {"sat60": "r3sat_n60_s1", "sat70": "r3sat_n70_s1", "sat80": "r3sat_n80_s1",
         "grid22": "grid_L22_a3_s1", "grid24": "grid_L24_a3_s1", "grid28": "grid_L28_a3_s1"}
GPU = dict(nchains=576, niters=20, scheme="pa", seed=1)
# stage-2 schedule for tc+h100: beta 3 -> 16 (300 levels); beta 1 was too hot with slice moves
STAGE2 = [3.0 + 13.0 * i / 299 for i in range(300)]


def evaluate(tree, sliced, ixs):
    sl = set(sliced)
    ixs_s = [[l for l in ix if l not in sl] for ix in ixs]
    d = tu.recompute_outputs(tree, ixs_s, [])
    assert sorted(tu.leaf_ids(d)) == list(range(len(ixs)))
    tc, sc = tu.complexity(d, ixs_s)
    t2 = tu.tree_h100_time(d, ixs_s)
    return d, dict(nsliced=len(sl), tc=tc, sc=sc, tc_total=tc + len(sl), h100_slice=t2, h100_total=t2 * 2.0 ** len(sl))


def omeco_worker(inst, T, ntrials=10):
    import omeco
    ixs, iy = instances.get(inst)
    sizes = {l: 2 for ix in ixs for l in ix}
    sf = omeco.ScoreFunction(sc_target=T)
    t0 = time.time()
    tree = omeco.optimize_code(ixs, iy, sizes, omeco.TreeSA(ntrials=ntrials, score=sf))
    t1 = time.time()
    s = omeco.slice_code(tree, ixs, sizes, omeco.TreeSASlicer(score=sf))
    t2 = time.time()
    return dict(tree=s.tree().to_dict(), sliced=list(s.slicing()), wall=t2 - t0, wall_treesa=t1 - t0, wall_slicer=t2 - t1)


def run_omeco(inst, T, ntrials=10, threads=32):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    p = subprocess.run([sys.executable, __file__, "--omeco-worker", json.dumps([inst, T, ntrials])], env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def emit(r):
    r = dict(r, host=HOST)
    print(json.dumps(r), flush=True)
    with open(RES, "a") as f:
        f.write(json.dumps(r) + "\n")


def save(inst, T, mode, tree, sliced, metrics):
    os.makedirs(OUTDIR, exist_ok=True)
    fn = os.path.join(OUTDIR, f"{NAMES[inst]}_sc{T}_{mode}.json")
    with open(fn, "w") as f:
        json.dump({"tree": tree, "sliced": sliced}, f)
    idx = os.path.join(OUTDIR, "index.json")
    meta = json.load(open(idx)) if os.path.exists(idx) else {}
    meta[os.path.basename(fn)] = metrics
    with open(idx, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omeco-worker")
    ap.add_argument("--instances", default="sat60,sat70,sat80,grid22,grid24,grid28")
    ap.add_argument("--targets", default="30,34,36,37")
    ap.add_argument("--methods", default="existing,omeco,greedy_tc,greedy_tch100,slice_tc,slice_tch100")
    ap.add_argument("--out", default=None, help="results jsonl (default bench/results/slicing_compare.jsonl)")
    ap.add_argument("--no-save", action="store_true", help="do not write trees to booltn-gpu/trees/gpusa_slice")
    ap.add_argument("--omeco-trials", type=int, default=10)
    ap.add_argument("--omeco-threads", type=int, default=32)
    ap.add_argument("--gpu-niters", type=int, default=20)
    a = ap.parse_args()
    if a.omeco_worker:
        inst, T, nt = json.loads(a.omeco_worker)
        print(json.dumps(omeco_worker(inst, T, nt)))
        return
    global RES
    if a.out:
        RES = a.out
    GPU["niters"] = a.gpu_niters
    suf = "" if a.gpu_niters == 20 else f" (niters={a.gpu_niters})"
    from booltn.slicer import greedy_slice  # the comparison needs the booltn package installed
    methods = a.methods.split(",")
    for inst in a.instances.split(","):
        ixs, iy = instances.get(inst)
        for T in map(int, a.targets.split(",")):
            base = dict(instance=inst, sc_target=T)
            if "existing" in methods:
                fn = os.path.join(TREES, f"{NAMES[inst]}_sc{T}.json")
                if os.path.exists(fn):
                    d = json.load(open(fn))
                    _, mt = evaluate(d["tree"], d["sliced"], ixs)
                    emit(dict(base, method="existing trees/", file=os.path.basename(fn), wall=None, **mt))
            if "omeco" in methods:
                r = run_omeco(inst, T, a.omeco_trials, a.omeco_threads)
                _, mt = evaluate(r["tree"], r["sliced"], ixs)
                name = "omeco TreeSA + slice_code" + ("" if a.omeco_trials == 10 else f" (ntrials={a.omeco_trials})")
                emit(dict(base, method=name, wall=r["wall"], wall_treesa=r["wall_treesa"], wall_slicer=r["wall_slicer"],
                          threads=a.omeco_threads, **mt))
            for key, mode in (("greedy_tc", "tc"), ("greedy_tch100", "tc+h100")):
                if key not in methods:
                    continue
                t0 = time.time()
                tree, info = tg.optimize(ixs, iy, sc_target=T, mode=mode, stage2_betas=STAGE2, **GPU)
                t1 = time.time()
                sliced, d = greedy_slice(tree, ixs, T, kind="gpu")
                t2 = time.time()
                _, mt = evaluate(d, sliced, ixs)
                emit(dict(base, method=f"GPU {mode} unsliced + greedy_slice(gpu){suf}", wall=t2 - t0, wall_gpu=t1 - t0,
                          wall_slicer=t2 - t1, **mt))
            for key, mode in (("slice_tc", "tc"), ("slice_tch100", "tc+h100")):
                if key not in methods:
                    continue
                t0 = time.time()
                tree, info = tg.optimize(ixs, iy, sc_target=T, mode=mode, slicing=True, slice_every=2,
                                         stage2_betas=STAGE2, **GPU)
                wall = time.time() - t0
                d, mt = evaluate(tree, info["sliced"], ixs)
                # kernel-internal numbers vs tree_utils recomputation
                assert d == tree
                ok = (abs(info["tc"] - mt["tc"]) < 1e-4 and info["sc"] == mt["sc"] and info["nsliced"] == mt["nsliced"]
                      and abs(info["h100_time_total"] / mt["h100_total"] - 1) < 1e-5)
                emit(dict(base, method=f"GPU {mode} + in-annealer slicing{suf}", wall=wall, n_repair=info["n_repair"],
                          internal_matches_tree_utils=ok, **mt))
                if not a.no_save:
                    save(inst, T, "tc" if mode == "tc" else "h100", tree, info["sliced"],
                         dict(mt, mode=mode, sc_target=T, gpu=GPU, n_repair=info["n_repair"], wall=wall))


if __name__ == "__main__":
    main()
