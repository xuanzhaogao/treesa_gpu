"""Write booltn-gpu/trees/gpusa_slice/<name>_sc<T>_best.json: the best (lowest modeled total H100 time,
sc <= T) GPU-TreeSA tree per instance and sc target from bench/results/slicing_compare.jsonl.
In-annealer winners are copied from the saved files; greedy_slice winners are regenerated (GPU runs
are deterministic for a fixed seed) and checked to reproduce the benchmark's modeled time.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import slicing_compare as sc  # noqa: E402
import instances  # noqa: E402
import treesa_gpu as tg  # noqa: E402  (the package next to bench/)

rows = [json.loads(l) for l in open(sc.RES) if l.strip()]
mine = [r for r in rows if r["method"].startswith("GPU") and "niters=" not in r["method"]]
idx_path = os.path.join(sc.OUTDIR, "index.json")
index = json.load(open(idx_path))
for inst in ([] if "--cross-target" in sys.argv else sc.NAMES):
    for T in (30, 34, 36, 37):
        rr = [r for r in mine if r["instance"] == inst and r["sc_target"] == T and r["sc"] <= T]
        if not rr:
            continue
        b = min(rr, key=lambda r: r["gpu2_total"])
        dst = os.path.join(sc.OUTDIR, f"{sc.NAMES[inst]}_sc{T}_best.json")
        if "in-annealer" in b["method"]:
            mode = "tc" if b["method"].startswith("GPU tc +") else "gpu2"
            src = os.path.join(sc.OUTDIR, f"{sc.NAMES[inst]}_sc{T}_{mode}.json")
            shutil.copyfile(src, dst)
            meta = dict(index[os.path.basename(src)], source=os.path.basename(src))
        else:
            from slicer import greedy_slice
            ixs, iy = instances.get(inst)
            mode = "tc+gpu2" if "tc+gpu2" in b["method"] else "tc"
            tree, _ = tg.optimize(ixs, iy, sc_target=T, mode=mode, stage2_betas=sc.STAGE2, **sc.GPU)
            sliced, d = greedy_slice(tree, ixs, T, kind="gpu")
            _, mt = sc.evaluate(d, sliced, ixs)
            assert abs(mt["gpu2_total"] / b["gpu2_total"] - 1) < 1e-9, (inst, T, mt, b)
            with open(dst, "w") as f:
                json.dump({"tree": d, "sliced": sliced}, f)
            meta = dict(mt, mode=mode + " unsliced, then greedy_slice(gpu)", sc_target=T, gpu=sc.GPU)
        meta["method"] = b["method"]
        index[os.path.basename(dst)] = meta
        print(f"{os.path.basename(dst)}: {b['method']}: #sliced {b['nsliced']} sc {b['sc']} modeled {b['gpu2_total']:.4g} s")
if "--cross-target" not in sys.argv:
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)


def cross_target_pass():
    """<name>_sc<T>_best.json := best of all saved/candidate trees of that instance with sc <= T (a tree
    found for a smaller target is also feasible for a larger one)."""
    import glob
    cands = []
    for fn in glob.glob(os.path.join(sc.OUTDIR, "*.json")) + glob.glob(os.path.join(HERE, "results", "candidates", "*.json")):
        if fn.endswith("index.json") or fn.endswith(".meta.json"):
            continue
        b = os.path.basename(fn)
        name = b.rsplit("_sc", 1)[0]
        inst = {v: k for k, v in sc.NAMES.items()}[name]
        ixs, iy = instances.get(inst)
        d = json.load(open(fn))
        _, mt = sc.evaluate(d["tree"], d["sliced"], ixs)
        cands.append((name, mt, fn))
    index = json.load(open(idx_path))
    for name in sc.NAMES.values():
        for T in (30, 34, 36, 37):
            ok = [c for c in cands if c[0] == name and c[1]["sc"] <= T]
            if not ok:
                continue
            b = min(ok, key=lambda c: c[1]["gpu2_total"])
            dst = os.path.join(sc.OUTDIR, f"{name}_sc{T}_best.json")
            if b[1]["gpu2_total"] < index[os.path.basename(dst)]["gpu2_total"] * (1 - 1e-12):
                print(f"{os.path.basename(dst)}: {index[os.path.basename(dst)]['gpu2_total']:.4g} -> {b[1]['gpu2_total']:.4g} s "
                      f"(from {os.path.basename(b[2])})")
                if os.path.abspath(b[2]) != os.path.abspath(dst):
                    shutil.copyfile(b[2], dst)
                index[os.path.basename(dst)] = dict(b[1], source=os.path.basename(b[2]), sc_target=T,
                                                    method=f"best feasible tree (from {os.path.basename(b[2])})")
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)


if "--cross-target" in sys.argv:
    cross_target_pass()
