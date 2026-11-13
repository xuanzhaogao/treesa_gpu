"""Extra seeds of GPU TreeSA (tc+gpu2, in-annealer slicing, PA 576 x 20) for the sc targets the H100 engine
runs (36, 37).  Candidate trees go to bench/results/candidates/; `--update-best` then replaces
booltn_gpu/trees/gpusa_slice/<name>_sc<T>_best.json when a candidate has a lower modeled total time.

    python bench/extra_seeds.py --seeds 2,3,4            # (on the GPU node, with TREESA_GPU_LOCK set)
    python bench/extra_seeds.py --update-best
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import slicing_compare as sc  # noqa: E402
import instances  # noqa: E402
import treesa_gpu as tg  # noqa: E402

CAND = os.path.join(HERE, "results", "candidates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2,3,4")
    ap.add_argument("--targets", default="36,37")
    ap.add_argument("--instances", default=",".join(sc.NAMES))
    ap.add_argument("--update-best", action="store_true")
    a = ap.parse_args()
    if a.update_best:
        idx_path = os.path.join(sc.OUTDIR, "index.json")
        index = json.load(open(idx_path))
        for fn in sorted(glob.glob(os.path.join(CAND, "*.json"))):
            if fn.endswith(".meta.json"):
                continue
            meta = json.load(open(fn[:-5] + ".meta.json"))
            name, T = meta["name"], meta["sc_target"]
            best = f"{name}_sc{T}_best.json"
            if meta["sc"] <= T and meta["gpu2_total"] < index[best]["gpu2_total"]:
                print(f"{best}: {index[best]['gpu2_total']:.4g} -> {meta['gpu2_total']:.4g} s (seed {meta['seed']})")
                shutil.copyfile(fn, os.path.join(sc.OUTDIR, best))
                index[best] = dict(meta, method=f"GPU tc+gpu2 + in-annealer slicing (seed {meta['seed']})")
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=1, sort_keys=True)
        return
    os.makedirs(CAND, exist_ok=True)
    for inst in a.instances.split(","):
        ixs, iy = instances.get(inst)
        for T in map(int, a.targets.split(",")):
            for seed in map(int, a.seeds.split(",")):
                gpu = dict(sc.GPU, seed=seed)
                t0 = time.time()
                tree, info = tg.optimize(ixs, iy, sc_target=T, mode="tc+gpu2", slicing=True, slice_every=2,
                                         stage2_betas=sc.STAGE2, **gpu)
                wall = time.time() - t0
                d, mt = sc.evaluate(tree, info["sliced"], ixs)
                base = os.path.join(CAND, f"{sc.NAMES[inst]}_sc{T}_gpu2_seed{seed}")
                with open(base + ".json", "w") as f:
                    json.dump({"tree": d, "sliced": info["sliced"]}, f)
                meta = dict(mt, name=sc.NAMES[inst], instance=inst, sc_target=T, seed=seed, mode="tc+gpu2", gpu=gpu,
                            wall=wall, host=sc.HOST)
                with open(base + ".meta.json", "w") as f:
                    json.dump(meta, f)
                print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
