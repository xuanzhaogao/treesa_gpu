"""Python-level validation of the GPU TreeSA wrapper.

    python tests/test_python.py

(0) trees.complexity agrees with omeco's own tree.complexity on omeco trees,
(1) dict <-> children conversion round-trips,
(2) GPU output trees are valid (binary, every leaf exactly once, internal iy from topology),
(3) tc / sc / modeled time reported by the binary agree with treesa_gpu.trees on the returned dict,
for all schemes, inits and energy modes.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import instances  # noqa: E402
import treesa_gpu as tg  # noqa: E402  (the package next to bench/)
from treesa_gpu import trees as tu  # noqa: E402
import omeco  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("ok   " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


for name in ("sat40", "sat60", "reg3_100", "lat10"):
    ixs, iy = instances.get(name)
    sizes = {l: 2 for ix in ixs for l in ix}
    m = len(ixs)
    # (0)
    for opt in (omeco.GreedyMethod(), omeco.TreeSA(ntrials=2, niters=5, score=omeco.ScoreFunction(sc_target=30))):
        t = omeco.optimize_code(ixs, iy, sizes, opt)
        c = t.complexity(ixs, sizes)
        d = tu.recompute_outputs(t.to_dict(), ixs, iy)
        tc, sc = tu.complexity(d, ixs)
        check(abs(tc - c.tc) < 1e-9 and sc == c.sc, f"{name}: trees == omeco complexity ({type(opt).__name__}: {tc:.3f}/{sc})")
        # (1)
        ch = tg.dict_to_children(d, m)
        d2 = tg.children_to_dict(ch, m, ixs, iy)
        check(tg.dict_to_children(d2, m) == ch and tu.complexity(d2, ixs) == (tc, sc), f"{name}: dict<->children round trip")
    # (2), (3)
    for scheme in ("indep", "pa", "trunc"):
        for init in ("greedy", "random"):
            for mode in ("tc", "gpu", "tc+gpu", "h100", "tc+h100"):
                if mode != "tc" and (scheme != "pa" or init != "greedy"):
                    continue
                tree, info = tg.optimize(ixs, iy, nchains=128, niters=4, sc_target=30, scheme=scheme, init=init,
                                         mode=mode, stage2_betas=[1.0 + 0.1 * i for i in range(40)])
                leaves = sorted(tu.leaf_ids(tree))
                ok_tree = leaves == list(range(m)) and info["valid"] and info["labels_ok"]
                ok_iy = tu.recompute_outputs(tree, ixs, iy) == tree
                ok_c = abs(info["tc"] - info["tc_py"]) < 1e-4 and info["sc"] == info["sc_py"]
                ok_t = abs(math.log(info["gpu_time"]) - math.log(info["gpu_time_py"])) < 1e-5 and \
                    abs(math.log(info["h100_time"]) - math.log(info["h100_time_py"])) < 1e-5
                check(ok_tree and ok_iy and ok_c and ok_t,
                      f"{name} {scheme:5s} {init:6s} {mode:6s}: valid={ok_tree} iy={ok_iy} tc {info['tc']:.4f}=={info['tc_py']:.4f} "
                      f"sc {info['sc']}=={info['sc_py']} time {info['gpu_time']:.4e}=={info['gpu_time_py']:.4e} "
                      f"time2 {info['h100_time']:.4e}=={info['h100_time_py']:.4e}")

# (4) slicing: returned tree is the tree of the sliced network (iy on ixs without sliced labels), the
# binary's per-slice tc / sc / #sliced and modeled total time equal treesa_gpu.trees on the sliced network,
# and the {"tree", "sliced"} runner json round-trips.  Includes a > 512-label instance (wide path).
import json  # noqa: E402
for name, T in (("sat60", 30), ("reg3_250", 24), ("grid24", 34)):
    ixs, iy = instances.get(name)
    m = len(ixs)
    labels = {l for ix in ixs for l in ix}
    for mode in ("tc", "tc+h100"):
        tree, info = tg.optimize(ixs, iy, nchains=128, niters=3, sc_target=T, scheme="pa", mode=mode, slicing=True,
                                 stage2_betas=[3.0 + 0.5 * i for i in range(26)])
        sl = set(info["sliced"])
        ixs_s = [[l for l in ix if l not in sl] for ix in ixs]
        ok_tree = sorted(tu.leaf_ids(tree)) == list(range(m)) and info["valid"] and info["labels_ok"] and sl <= labels
        ok_iy = tu.recompute_outputs(tree, ixs_s, []) == tree
        tc, sc = tu.complexity(tree, ixs_s)
        T2 = tu.tree_h100_time(tree, ixs_s) * 2.0 ** len(sl)
        ok_c = abs(info["tc"] - tc) < 1e-4 and info["sc"] == sc and info["nsliced"] == len(sl) and sc <= T
        ok_t = abs(math.log(info["h100_time_total"]) - math.log(T2)) < 1e-5
        rj = json.loads(json.dumps(tg.runner_json(tree, info)))
        ok_j = rj["sliced"] == info["sliced"] and rj["tree"] == tree
        check(ok_tree and ok_iy and ok_c and ok_t and ok_j,
              f"slicing {name} {mode:7s} T={T}: valid={ok_tree} iy_sliced={ok_iy} |S|={len(sl)} tc/slice {info['tc']:.3f}=={tc:.3f} "
              f"sc {info['sc']}=={sc}<={T} total time {info['h100_time_total']:.4e}=={T2:.4e} json={ok_j} repair={info['n_repair']}")

print("ALL PYTHON TESTS PASSED" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
