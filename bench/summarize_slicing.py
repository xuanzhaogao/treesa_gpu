"""Markdown tables from bench/results/slicing_compare.jsonl.

    python bench/summarize_slicing.py > bench/results/slicing_summary.md
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "slicing_compare.jsonl")
rows = [json.loads(l) for l in open(path) if l.strip()]
SHORT = {"existing trees/": "existing trees/",
         "omeco TreeSA + slice_code": "omeco TreeSA + slice_code",
         "GPU tc unsliced + greedy_slice(gpu)": "GPU tc → greedy_slice",
         "GPU tc+gpu2 unsliced + greedy_slice(gpu)": "GPU tc+gpu2 → greedy_slice",
         "GPU tc + in-annealer slicing": "GPU tc, in-annealer slicing",
         "GPU tc+gpu2 + in-annealer slicing": "GPU tc+gpu2, in-annealer slicing"}

print("## Slicing comparison (per-slice tc / sc / #sliced / total tc / modeled total H100 time / wall)\n")
print("Modeled total time = tree_gpu_time_v2(tree on sliced ixs) × 2^#sliced.  Best modeled time per row in bold.\n")
groups = collections.OrderedDict()
for r in rows:
    groups.setdefault((r["instance"], r["sc_target"]), []).append(r)
print("| instance | sc_target | method | #sliced | tc/slice | sc | total tc | modeled time (s) | wall (s) |")
print("|---|---|---|---|---|---|---|---|---|")
for (inst, T), rr in groups.items():
    best = min(r["gpu2_total"] for r in rr if r["sc"] <= T)
    for r in rr:
        t = f"{r['gpu2_total']:.3g}"
        if r["gpu2_total"] == best:
            t = f"**{t}**"
        wall = "-" if r["wall"] is None else f"{r['wall']:.1f}"
        flag = "" if r["sc"] <= T else " (sc > target!)"
        print(f"| {inst} | {T} | {SHORT.get(r['method'], r['method'])} | {r['nsliced']} | {r['tc']:.2f} | {r['sc']}{flag} | "
              f"{r['tc_total']:.2f} | {t} | {wall} |")

print("\n## Modeled time relative to the best method (geometric mean over instances × targets)\n")
ratios = collections.defaultdict(list)
wins = collections.Counter()
for (inst, T), rr in groups.items():
    best = min(r["gpu2_total"] for r in rr if r["sc"] <= T)
    bm = [r["method"] for r in rr if r["gpu2_total"] == best]
    for m in bm:
        wins[m] += 1
    if not any(r["method"].startswith("omeco") for r in rr):
        continue
    for r in rr:
        if r["method"] != "existing trees/":
            ratios[r["method"]].append(r["gpu2_total"] / best)
import math  # noqa: E402
print("| method | geo-mean (modeled time / best) | max | #best |")
print("|---|---|---|---|")
for m, v in ratios.items():
    g = math.exp(sum(math.log(x) for x in v) / len(v))
    print(f"| {SHORT.get(m, m)} | {g:.2f} | {max(v):.2f} | {wins[m]} |")
checks = [r.get("internal_matches_tree_utils") for r in rows if "internal_matches_tree_utils" in r]
print(f"\nKernel-internal (per-slice tc, sc, #sliced, modeled total) == tree_utils recomputation: "
      f"{sum(1 for c in checks if c)}/{len(checks)} runs.  Greedy repairs needed: "
      f"{sum(1 for r in rows if r.get('n_repair'))} runs.")

# ---- cross-machine table (optional extra result files) -------------------------------------------
extra = {}
for tag, fn in (("h100", "slicing_compare_h100.jsonl"), ("w7273", "slicing_compare_worker7273.jsonl")):
    p = os.path.join(os.path.dirname(path), fn)
    if os.path.exists(p):
        extra[tag] = [json.loads(l) for l in open(p) if l.strip()]
if extra:
    def find(rs, inst, T, method):
        for r in rs:
            if r["instance"] == inst and r["sc_target"] == T and r["method"] == method:
                return r
        return None
    print("\n## Same comparison on other machines (modeled time s / wall s)\n")
    print("GPU columns: in-annealer slicing tc+gpu2 (identical trees on both GPUs, only wall time differs). "
          "omeco columns: TreeSA + slice_code, ccmlin078 = 32 threads (2x Xeon Gold 6234), worker7273 = 96 threads "
          "(AMD EPYC 9474F).\n")
    print("| instance | T | GPU RTX 6000 | GPU H100 | omeco ccmlin078 ntrials=10 | omeco worker7273 ntrials=10 | omeco worker7273 ntrials=96 |")
    print("|---|---|---|---|---|---|---|")
    M = "GPU tc+gpu2 + in-annealer slicing"
    for (inst, T), rr in groups.items():
        cells = []
        for rs, meth in ((rows, M), (extra.get("h100", []), M), (rows, "omeco TreeSA + slice_code"),
                         (extra.get("w7273", []), "omeco TreeSA + slice_code"),
                         (extra.get("w7273", []), "omeco TreeSA + slice_code (ntrials=96)")):
            r = find(rs, inst, T, meth)
            cells.append("-" if r is None else f"{r['gpu2_total']:.3g} / {r['wall']:.1f}")
        print(f"| {inst} | {T} | " + " | ".join(cells) + " |")
    h = extra.get("h100", [])
    same = sum(1 for r in h if (q := find(rows, r["instance"], r["sc_target"], r["method"])) is not None
               and abs(q["gpu2_total"] - r["gpu2_total"]) <= 1e-9 * q["gpu2_total"] and q["nsliced"] == r["nsliced"])
    comp = sum(1 for r in h if find(rows, r["instance"], r["sc_target"], r["method"]) is not None)
    print(f"\nH100 runs reproducing the RTX 6000 result exactly (same #sliced and modeled time): {same}/{comp}.")
