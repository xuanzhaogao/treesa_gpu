"""Modeled H100 time (tree_utils.gpu_time_v2) of trees optimized for tc vs for the v2 model.

    python bench/gpu2_study.py > bench/results/gpu2_study.md

3-SAT, alpha = 4.26, n = 50..70 (instance seed 1), sc_target = 30, sc_weight = 1, greedy init,
Julia schedule (300 levels), PA with 576 chains x niters 50.  For every tree we report the total
modeled time sum_t, how it splits over the three kernel paths of the model (batched small
contractions, CUDA-core path `cc`, 1-bit tensor-core path `tc`), how many contractions take the
tensor-core path with K >= 9, and the dominant contraction.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import instances  # noqa: E402
import treesa_gpu as tg  # noqa: E402
import tree_utils as tu  # noqa: E402

MODEL = tu.GPU_MODEL_V2
SC_TARGET = 30.0


def path_of(B, M, N, K, model=MODEL):
    """Which branch of gpu_time_v2 a contraction takes: 'small', 'cc' or 'tc' (mirrors tree_utils)."""
    if B + M + N <= model["small_rank"]:
        return "small"
    if K >= 7 and max(M, N) >= 5:
        a, b, c = 2.0 ** (B + M + K), 2.0 ** (B + N + K), 2.0 ** (B + M + N)
        moved = a + b + c
        cc = model["cc_bit"] * moved + model["cc_cell"] * 2.0 ** (B + M + N + K)
        tcc = (model["tc_bit"] * moved + model["tc_pad_cell"] * 2.0 ** (B + max(M, 6) + max(N, 6) + max(K, 8))
               + model["tc_out_bit"] * c)
        return "tc" if tcc < cc else "cc"
    return "cc"


def stats(tree, ixs):
    cs = tu.contractions(tree, ixs)
    tot = {"small": 0.0, "cc": 0.0, "tc": 0.0}
    n_tc_k9 = 0
    t_tc_k9 = 0.0
    top = None
    for (B, M, N, K) in cs:
        p = path_of(B, M, N, K)
        t = tu.gpu_time_v2(B, M, N, K, MODEL, small=(p == "small"))
        tot[p] += t
        if p == "tc" and K >= 9:
            n_tc_k9 += 1
            t_tc_k9 += t
        if top is None or t > top[0]:
            top = (t, (B, M, N, K), p)
    T = sum(tot.values())
    tc, sc = tu.complexity(tree, ixs)
    return dict(T=T, tc=tc, sc=sc, frac=tot, n_tc_k9=n_tc_k9, f_tc_k9=t_tc_k9 / T, top=top)


def row(name, label, st, wall):
    f = st["frac"]
    T = st["T"]
    t, bmnk, p = st["top"]
    return (f"| {name} | {label} | {T:.3e} | {st['tc']:.1f} | {st['sc']} | {f['small'] / T:.2f} / {f['cc'] / T:.2f} / "
            f"{f['tc'] / T:.2f} | {st['n_tc_k9']} ({st['f_tc_k9']:.2f}) | {bmnk} {p} {t / T:.2f} | {wall:.1f} |")


def main():
    import omeco
    print("| instance | optimizer | modeled Σt v2 (s) | tc | sc | time share small / cc / tc-core | "
          "#tensor-core contractions with K≥9 (time share) | dominant (B,M,N,K) path share | wall (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in (50, 55, 60, 65, 70):
        ixs, iy = instances.sat(n, seed=1)
        name = f"sat{n}"
        sizes = {l: 2 for ix in ixs for l in ix}
        t0 = time.time()
        o = omeco.optimize_code(ixs, iy, sizes, omeco.TreeSA(score=omeco.ScoreFunction(sc_target=SC_TARGET)))
        w = time.time() - t0
        od = tu.recompute_outputs(o.to_dict(), ixs, iy)
        print(row(name, "omeco TreeSA default (tc)", stats(od, ixs), w), flush=True)
        for mode, extra in (("tc", {}), ("gpu2", {}), ("tc+gpu2", {}),
                            ("gpu2", dict(betas=[1.0 + 0.05 * i for i in range(301)], label="gpu2, cold β=1..16"))):
            label = extra.pop("label", f"GPU PA 576x50 [{mode}]")
            t0 = time.time()
            tree, info = tg.optimize(ixs, iy, nchains=576, niters=50, sc_target=SC_TARGET, scheme="pa", mode=mode,
                                     seed=1, **extra)
            w = time.time() - t0
            st = stats(tree, ixs)
            assert abs(st["T"] - info["gpu_time2_py"]) <= 1e-9 * st["T"]
            print(row(name, label if "label" not in extra else label, st, w), flush=True)


if __name__ == "__main__":
    main()
