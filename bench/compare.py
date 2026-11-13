"""Quality / wallclock comparison: omeco CPU TreeSA vs GPU TreeSA (independent chains, population
annealing, truncation selection).  Everything runs on the same workstation (ccmlin078: 2x Xeon Gold
6234 = 16 cores / 32 threads, Quadro RTX 6000).  Results are appended as JSON lines.

    python bench/compare.py --suite omeco --instances sat60,sat100 --out bench/results/compare.jsonl
    python bench/compare.py --suite gpu   --instances sat60,sat100 --out bench/results/compare.jsonl

All runs use ScoreFunction(sc_target=30, sc_weight=1, rw_weight=0) and the Julia schedule
betas = 0.01:0.05:15 (300 levels); `niters` = sweeps per level.  Wallclock is end-to-end from Python
(GPU: includes process start, CUDA context, greedy init, I/O; omeco: includes its greedy init and
preprocessing).
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

SC_TARGET = 30.0


def omeco_worker(cfg):
    """Runs inside a fresh process so RAYON_NUM_THREADS takes effect."""
    import omeco
    import tree_utils as tu
    ixs, iy = instances.get(cfg["instance"])
    sizes = {l: 2 for ix in ixs for l in ix}
    opt = omeco.TreeSA(ntrials=cfg["ntrials"], niters=cfg["niters"], preprocess=cfg["preprocess"],
                       score=omeco.ScoreFunction(sc_target=SC_TARGET))
    t0 = time.time()
    tree = omeco.optimize_code(ixs, iy, sizes, opt)
    wall = time.time() - t0
    d = tu.recompute_outputs(tree.to_dict(), ixs, iy)
    tc, sc = tu.complexity(d, ixs)
    c = tree.complexity(ixs, sizes)
    return dict(wall=wall, tc=tc, sc=sc, tc_omeco=c.tc, sc_omeco=c.sc, gpu_time=tu.tree_gpu_time(d, ixs))


def run_omeco(inst, ntrials, niters, threads, preprocess=True):
    cfg = dict(instance=inst, ntrials=ntrials, niters=niters, preprocess=preprocess)
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    p = subprocess.run([sys.executable, __file__, "--omeco-worker", json.dumps(cfg)], env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r.update(method="omeco", threads=threads, **cfg)
    return r


def run_gpu(inst, nchains, niters, scheme, seed, mode="tc", init="greedy"):
    import treesa_gpu as tg
    ixs, iy = instances.get(inst)
    t0 = time.time()
    tree, info = tg.optimize(ixs, iy, nchains=nchains, niters=niters, sc_target=SC_TARGET, scheme=scheme,
                             seed=seed, mode=mode, init=init)
    wall = time.time() - t0
    return dict(method="gpu", instance=inst, nchains=nchains, niters=niters, scheme=scheme, seed=seed, mode=mode,
                init=init, wall=wall, tc=info["tc_py"], sc=info["sc_py"], gpu_time=info["gpu_time_py"],
                t_anneal=info["t_anneal"], kernel_s=info["kernel_s"], kernel=info["kernel"],
                proposals=info["proposals"], families_end=info["trace"][-1][6],
                stage1_gpu_time=info.get("stage1_gpu_time"), stage1_tc=info.get("stage1_tc"))


def emit(out, r):
    line = json.dumps(r)
    print(line, flush=True)
    with open(out, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omeco-worker")
    ap.add_argument("--suite", default="gpu")
    ap.add_argument("--instances", default="sat60,sat80,sat100,sat120,lat14,reg3_100,reg3_250")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "compare.jsonl"))
    a = ap.parse_args()
    if a.omeco_worker:
        print(json.dumps(omeco_worker(json.loads(a.omeco_worker))))
        return
    for inst in a.instances.split(","):
        if a.suite == "omeco":
            emit(a.out, run_omeco(inst, 10, 50, 32))               # omeco default
            emit(a.out, run_omeco(inst, 10, 50, 16))
            emit(a.out, run_omeco(inst, 10, 50, 32, preprocess=False))
            for nit in (5, 20, 50, 200):                         # quality vs time, all 32 threads busy
                emit(a.out, run_omeco(inst, 32, nit, 32))
        elif a.suite == "gpu":
            # equal-work scheme comparison (3 seeds) + quality-vs-time curves
            for nit in (5, 20, 50, 200, 800):
                for scheme in ("indep", "pa", "trunc"):
                    seeds = (1, 2, 3) if nit in (20, 50) else (1,)
                    for s in seeds:
                        emit(a.out, run_gpu(inst, 576, nit, scheme, s))
            for nit in (5, 20, 50, 200):
                for scheme in ("indep", "pa"):
                    emit(a.out, run_gpu(inst, 4608, nit, scheme, 1))
        elif a.suite == "gpumode":
            # GPU-runtime objective: tc only, gpu only, tc then gpu refinement
            for mode in ("tc", "gpu", "tc+gpu"):
                emit(a.out, run_gpu(inst, 576, 50, "pa", 1, mode=mode))


if __name__ == "__main__":
    main()
