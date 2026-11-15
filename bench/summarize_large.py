"""Table of bench/results/large_compare_{gpu,omeco}.jsonl -> bench/results/large_summary.md (tc / sc / wall)."""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = []
for f in ("large_compare_gpu.jsonl", "large_compare_omeco.jsonl"):
    p = os.path.join(R, f)
    if os.path.exists(p):
        rows += [json.loads(l) for l in open(p)]


def label(r):
    m = r["method"]
    if m == "gpu":
        s = f"GPU {r['scheme']} {r['nchains']}x{r['niters']}"
        if r.get("init") == "sweep":
            s += f" from sweep, beta0 {r.get('beta0')}"
        return s
    if m == "omeco_treesa":
        return f"omeco TreeSA {r['ntrials']}x{r['niters']}"
    return {"omeco_greedy": "omeco greedy", "sweep": "row-major sweep"}[m]


inst = []
by = collections.defaultdict(dict)
meta = {}
for r in rows:
    if "error" in r:
        continue
    if r["instance"] not in meta:
        inst.append(r["instance"]); meta[r["instance"]] = (r["m"], r["labels"])
    by[r["instance"]][label(r)] = r
methods = []
for r in rows:
    if "error" not in r and label(r) not in methods:
        methods.append(label(r))
order = ["row-major sweep", "omeco greedy"] + [m for m in methods if m.startswith("omeco TreeSA")] + \
        [m for m in methods if m.startswith("GPU")]
out = ["# Large networks: GPU TreeSA (Quadro RTX 6000) vs omeco (12 threads, workergpu165 CPU)", "",
       "Cells: tc (log2) / sc / wall s; pure-tc objective (sc_target 10000); single runs, GPU seed 1. Best tc per instance in bold. GPU wall includes process start and greedy init.", "",
       "| method | " + " | ".join(f"{i} (m={meta[i][0]}, {meta[i][1]} labels)" for i in inst) + " |",
       "|---|" + "---:|" * len(inst)]
best = {i: min(r["tc"] for r in by[i].values() if r["method"] != "sweep" or True) for i in inst}
for m in order:
    cells = []
    for i in inst:
        r = by[i].get(m)
        if r is None:
            cells.append("–")
            continue
        tc = f"{r['tc']:.1f}"
        if abs(r["tc"] - best[i]) < 1e-9:
            tc = f"**{tc}**"
        cells.append(f"{tc} / {r['sc']} / {r['wall']:.0f}")
    out.append(f"| {m} | " + " | ".join(cells) + " |")
open(os.path.join(R, "large_summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# RTX 6000 vs H200 wall time for the same GPU runs (identical trees: the annealer is deterministic per seed)
h = os.path.join(R, "large_compare_h200.jsonl")
if os.path.exists(h):
    H = [json.loads(l) for l in open(h)]
    key = lambda r: (r["instance"], r["nchains"], r["niters"], r["scheme"], r.get("init"))
    rtx = {key(r): r for r in rows if r["method"] == "gpu" and "error" not in r}
    t = ["", "## Same runs on an H200 (workergpu318) vs the RTX 6000", "",
         "| instance | config | tc RTX / H200 | wall RTX 6000 | wall H200 | speed-up |", "|---|---|---|---:|---:|---:|"]
    for r in sorted(H, key=lambda r: (r["nchains"], r["niters"], r["instance"])):
        q = rtx.get(key(r))
        if q is None or "error" in r:
            continue
        t.append(f"| {r['instance']} | {label(r)[4:]} | {q['tc']:.2f} / {r['tc']:.2f} | {q['wall']:.0f} s | {r['wall']:.0f} s | "
                 f"{q['wall'] / r['wall']:.2f}x |")
    open(os.path.join(R, "large_summary.md"), "a").write("\n".join(t) + "\n")
    print("\n".join(t))
