"""Markdown tables from bench/results/compare.jsonl.

    python bench/summarize.py [bench/results/compare.jsonl] > bench/results/summary.md
"""
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "compare.jsonl")
rows = [json.loads(l) for l in open(path) if l.strip()]
insts = list(dict.fromkeys(r["instance"] for r in rows))


def label(r):
    if r["method"] == "omeco":
        pre = "" if r["preprocess"] else ", no preprocess"
        return f"omeco ntrials={r['ntrials']} niters={r['niters']} ({r['threads']} thr{pre})"
    return f"GPU {r['scheme']} {r['nchains']}x niters={r['niters']}" + (f" [{r['mode']}]" if r["mode"] != "tc" else "")


def fmt(r):
    return f"{r['tc']:.2f} / {r['sc']} / {r['wall']:.2f}s"


def pick(inst, **kw):
    return [r for r in rows if r["instance"] == inst and all(r.get(k) == v for k, v in kw.items())]


print("## Headline: omeco CPU TreeSA vs GPU TreeSA (tc / sc / wallclock)\n")
cols = [("omeco default, 32 thr", dict(method="omeco", ntrials=10, niters=50, threads=32, preprocess=True)),
        ("omeco default, 16 thr", dict(method="omeco", ntrials=10, niters=50, threads=16, preprocess=True)),
        ("omeco 32 trials x niters=200", dict(method="omeco", ntrials=32, niters=200)),
        ("GPU indep 576 x 50", dict(method="gpu", scheme="indep", nchains=576, niters=50, seed=1, mode="tc")),
        ("GPU PA 576 x 50", dict(method="gpu", scheme="pa", nchains=576, niters=50, seed=1, mode="tc")),
        ("GPU PA 4608 x 50", dict(method="gpu", scheme="pa", nchains=4608, niters=50, seed=1, mode="tc")),
        ("GPU PA 576 x 800", dict(method="gpu", scheme="pa", nchains=576, niters=800, seed=1, mode="tc"))]
print("| instance | " + " | ".join(c for c, _ in cols) + " |")
print("|---" * (len(cols) + 1) + "|")
for inst in insts:
    cells = []
    for _, kw in cols:
        rr = pick(inst, **kw)
        cells.append(fmt(rr[0]) if rr else "-")
    print(f"| {inst} | " + " | ".join(cells) + " |")

print("\n## Matched wallclock: best tc (sc, config) reached within the wallclock of an omeco run\n")
print("Budget = end-to-end wallclock of the omeco run; GPU rows use seed 1 only (no best-of-seeds).\n")
print("| instance | budget | omeco (any config) | GPU indep | GPU PA | GPU trunc |")
print("|---|---|---|---|---|---|")
for inst in insts:
    for bname, kw in (("omeco default", dict(method="omeco", ntrials=10, niters=50, threads=32, preprocess=True)),
                      ("omeco 32x200", dict(method="omeco", ntrials=32, niters=200))):
        ref = pick(inst, **kw)
        if not ref:
            continue
        B = ref[0]["wall"]
        cells = []
        fams = [dict(method="omeco"), dict(method="gpu", scheme="indep"), dict(method="gpu", scheme="pa"),
                dict(method="gpu", scheme="trunc")]
        for fk in fams:
            cand = [r for r in pick(inst, **fk) if r["wall"] <= B * 1.05 and r.get("seed", 1) == 1
                    and r.get("mode", "tc") == "tc"]
            if not cand:
                cells.append("-")
                continue
            r = min(cand, key=lambda r: (r["tc"], r["sc"]))
            cfg = (f"{r['ntrials']}x{r['niters']}" if r["method"] == "omeco" else f"{r['nchains']}x{r['niters']}")
            cells.append(f"**{r['tc']:.2f}** ({r['sc']}, {cfg}, {r['wall']:.1f}s)")
        print(f"| {inst} | {bname} {B:.1f}s | " + " | ".join(cells) + " |")

print("\n## Population schemes at equal work (576 chains, 300 levels; tc mean ± sd / best over seeds 1-3)\n")
print("| instance | niters | indep | PA | trunc (25%) |")
print("|---|---|---|---|---|")
for inst in insts:
    for nit in (20, 50):
        cells = []
        for scheme in ("indep", "pa", "trunc"):
            tcs = [r["tc"] for r in pick(inst, method="gpu", scheme=scheme, nchains=576, niters=nit, mode="tc")]
            if not tcs:
                cells.append("-")
            elif len(tcs) == 1:
                cells.append(f"{tcs[0]:.2f}")
            else:
                cells.append(f"{statistics.mean(tcs):.2f} ± {statistics.stdev(tcs):.2f} / {min(tcs):.2f}")
        print(f"| {inst} | {nit} | " + " | ".join(cells) + " |")

print("\n## Quality vs wallclock (best tc reached by each configuration, sorted by wallclock)\n")
for inst in insts:
    print(f"\n### {inst}\n")
    print("| wallclock (s) | method | tc | sc |")
    print("|---|---|---|---|")
    rr = [r for r in rows if r["instance"] == inst and r.get("mode", "tc") == "tc" and r.get("seed", 1) == 1]
    seen = set()  # identical configs re-run by another suite: keep the first (results are deterministic)
    rr = [r for r in rr if not (label(r) in seen or seen.add(label(r)))]
    for r in sorted(rr, key=lambda r: r["wall"]):
        print(f"| {r['wall']:.2f} | {label(r)} | {r['tc']:.2f} | {r['sc']} |")

if any(r["method"] == "gpu" and r["mode"] in ("gpu", "tc+gpu") for r in rows):
    print("\n## GPU-runtime objective (mode ii): modeled time sum_t (s) / tc / sc, PA 576 x 50\n")
    print("| instance | omeco default tree | GPU [tc] | GPU [gpu] | GPU [tc+gpu] |")
    print("|---|---|---|---|---|")
    for inst in insts:
        cells = []
        o = pick(inst, method="omeco", ntrials=10, niters=50, threads=32, preprocess=True)
        cells.append(f"{o[0]['gpu_time']:.3g} / {o[0]['tc']:.1f} / {o[0]['sc']}" if o else "-")
        for mode in ("tc", "gpu", "tc+gpu"):
            rr = pick(inst, method="gpu", scheme="pa", nchains=576, niters=50, seed=1, mode=mode)
            cells.append(f"{rr[-1]['gpu_time']:.3g} / {rr[-1]['tc']:.1f} / {rr[-1]['sc']}" if rr else "-")
        print(f"| {inst} | " + " | ".join(cells) + " |")
