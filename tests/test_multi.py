"""optimize_multi: independent runs on several GPUs (or several runs on one GPU: gpus=[0, 0, 0]).
Checks that every run returns a valid tree and that the best one is selected.
    python tests/test_multi.py [gpu ids, default 0,0,0]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import treesa_gpu as tg  # noqa: E402

gpus = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "0", "0"]
rng = random.Random(1)
ixs = [rng.sample(range(80), 3) for _ in range(340)]
tree, best, infos = tg.optimize_multi(ixs, [], gpus=gpus, nchains=256, niters=10, sc_target=30, mode="tc+h100")
times = [i["h100_time_total_py"] for i in infos]
ok = all(i["valid"] for i in infos) and best["h100_time_total_py"] == min(times) and len(set(times)) > 1
print("runs (seed: tc_total, modeled H100 s):", [(s, round(i["tc_total_py"], 2), f"{t:.3g}")
                                                 for s, i, t in zip(range(1, len(gpus) + 1), infos, times)])
print("best:", f"{best['h100_time_total_py']:.3g}", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
