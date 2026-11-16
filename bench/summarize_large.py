"""Tables of bench/results/large_compare_*.jsonl -> bench/results/large_summary.md (tc / sc / wall).

All GPU runs on the Quadro RTX 6000 unless stated.  large_compare_gpu.jsonl: original kernels.
large_compare_gpu_v3.jsonl: new kernels (cooperative visits, chunk summaries, one block per chain) with the final
automatic choices ("GPU v2" rows).  large_compare_gpu_v2.jsonl: an intermediate build of the new kernels (16 warps
per chain, cooperative visits also up to 512 labels); used only for configurations not in v3 (72 x 500).
large_compare_h100_v1 / _v2.jsonl: original / intermediate new kernels on the H100 PCIe.  The trees depend only on the
seed, so every build and GPU must give the same tc."""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
key = lambda r: (r["instance"], r["nchains"], r["niters"], r["scheme"], r.get("init"))


def load(f):
    p = os.path.join(R, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


rows = []
for f in ("large_compare_gpu.jsonl", "large_compare_omeco.jsonl"):
    rows += load(f)
v3 = [dict(r, v2=True) for r in load("large_compare_gpu_v3.jsonl") if "error" not in r]
have = {key(r) for r in v3}
v2mid = [dict(r, v2=True, mid=True) for r in load("large_compare_gpu_v2.jsonl")
         if "error" not in r and r.get("method") == "gpu" and key(r) not in have]
rows += v3 + v2mid


def label(r):
    m = r["method"]
    if m == "gpu":
        s = f"GPU{' v2' if r.get('v2') else ''} {r['scheme']} {r['nchains']}x{r['niters']}{'†' if r.get('mid') else ''}"
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
        [m for m in methods if m.startswith("GPU") and not m.startswith("GPU v2")] + \
        [m for m in methods if m.startswith("GPU v2")]
out = ["# Large networks: GPU TreeSA (Quadro RTX 6000) vs omeco (12 threads, workergpu165 CPU)", "",
       "Cells: tc (log2) / sc / wall s; pure-tc objective (sc_target 10000); single runs, GPU seed 1. Best tc per instance in bold. GPU wall includes process start and greedy init. "
       "GPU: original kernels; GPU v2: new kernels (same trees). †: intermediate build of the new kernels (16 warps per chain).", "",
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
    rtx = {key(r): r for r in rows if r["method"] == "gpu" and "error" not in r and not r.get("v2")}
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

# new kernels vs the original ones on the same GPU, same configurations and seeds (trees must be identical)
old = {key(r): r for r in rows if r["method"] == "gpu" and "error" not in r and not r.get("v2")}
new = [r for r in rows if r.get("v2") and "error" not in r]
if new:
    t = ["", "## New kernels vs the original ones (RTX 6000, same configurations and seeds)", "",
         "Cooperative visits (G lanes per node when fewer than 32 nodes are ready; above 512 labels), chunk summaries "
         "(>= 128 words) and one thread block of up to 8 warps per chain (auto when the chains do not fill the GPU).  "
         "The trees must be identical.", "",
         "| instance | config | kernel | tc old / new | wall old | wall new | speed-up | kernel s old / new |",
         "|---|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(new, key=lambda r: (r["nchains"], r["niters"], r["instance"])):
        q = old.get(key(r))
        if q is None:
            continue
        t.append(f"| {r['instance']} | {label(r)[7:]} | {r.get('kernel')} | {q['tc']:.2f} / {r['tc']:.2f}"
                 f"{'' if abs(q['tc'] - r['tc']) < 1e-9 else ' **differ**'} | {q['wall']:.0f} s | {r['wall']:.0f} s | "
                 f"{q['wall'] / r['wall']:.2f}x | {q.get('kernel_s') or 0:.1f} / {r.get('kernel_s') or 0:.1f} |")
    open(os.path.join(R, "large_summary.md"), "a").write("\n".join(t) + "\n")
    print("\n".join(t))

# H100 PCIe: original kernels (v1) vs the intermediate new build (v2: 16 warps per chain), and new-kernel-only runs
h1 = {key(r): r for r in load("large_compare_h100_v1.jsonl") if "error" not in r}
h2 = [r for r in load("large_compare_h100_v2.jsonl") if "error" not in r]
if h2:
    rtx_new = {key(r): r for r in rows if r.get("v2") and "error" not in r}
    t = ["", "## H100 PCIe (workergpu165): original vs new kernels (intermediate build, 16 warps per chain)", "",
         "| instance | config | tc | H100 old | H100 new | speed-up | RTX 6000 new |", "|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(h2, key=lambda r: (r["nchains"], r["niters"], r.get("init") or "", r["instance"])):
        q, x = h1.get(key(r)), rtx_new.get(key(r))
        bad = (q and abs(q["tc"] - r["tc"]) > 1e-9) or (x and abs(x["tc"] - r["tc"]) > 1e-9)
        w_old = f"{q['wall']:.0f} s" if q else "–"
        sp = f"{q['wall'] / r['wall']:.2f}x" if q else "–"
        w_rtx = f"{x['wall']:.0f} s" if x else "–"
        t.append(f"| {r['instance']} | {label(r)[4:]} | {r['tc']:.2f}{' **differ**' if bad else ''} | {w_old} | "
                 f"{r['wall']:.0f} s | {sp} | {w_rtx} |")
    open(os.path.join(R, "large_summary.md"), "a").write("\n".join(t) + "\n")
    print("\n".join(t))
