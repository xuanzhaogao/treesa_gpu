"""Kernel-mapping throughput benchmark: proposals/s (and node visits/s) of the four kernels
vs number of chains, on several instances.  Full Julia schedule (300 betas), greedy init.

    python bench/throughput.py [--niters 4] > bench/results/throughput.txt
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
BIN = os.environ.get("TREESA_GPU_BINARY", os.path.join(ROOT, "build", "treesa_gpu"))


BETAS = "0.01:0.05:15"


def run(inst, kernel, nc, niters, extra=()):
    if kernel.endswith("+wide"):  # force the wide (streaming) label path
        kernel = kernel[:-5]
        extra = tuple(extra) + ("--wide",)
    if "/" in kernel:  # e.g. "warp/16" = warp kernel with 16 lanes per chain
        kernel, lpc = kernel.split("/")
        extra = tuple(extra) + ("--lpc", lpc)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"tp_{os.getpid()}.json")
    cmd = [BIN, "-i", f"{ROOT}/tests/data/{inst}.txt", "--tree", f"{ROOT}/tests/data/{inst}.tree",
           "--nchains", str(nc), "--niters", str(niters), "--betas", BETAS, "--sc-target", "30",
           "--kernel", kernel, "--eval-every", "50", "--out", out, "--quiet", *extra]
    if os.environ.get("TREESA_GPU_LOCK"):  # serialize with other GPU users (H100)
        cmd = ["flock", os.environ["TREESA_GPU_LOCK"]] + cmd
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0,):
        return None
    with open(out) as f:
        r = json.load(f)
    os.remove(out)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niters", type=int, default=4)
    ap.add_argument("--instances", default="sat60,sat100,lat14,reg3_250")
    ap.add_argument("--kernels", default="seq,seqsm,warp,warpsm")
    ap.add_argument("--nchains", default="72,576,2304,9216")
    ap.add_argument("--betas", default="0.01:0.05:15")
    a = ap.parse_args()
    global BETAS
    BETAS = a.betas
    run("sat100", "warp", 576, 2)  # warm-up (clocks)
    print(f"{'instance':9s} {'nw':>2s} {'kernel':10s} {'nchains':>7s} {'kernel_s':>8s} {'visits/s':>10s} {'proposals/s':>11s} "
          f"{'sweeps/s/chain':>14s} {'lane_occ':>8s}")
    for inst in a.instances.split(","):
        with open(f"{ROOT}/tests/data/{inst}.txt") as f:
            m = int(f.readline().split()[0])
        for k in a.kernels.split(","):  # "warp/16" etc. selects lanes per chain
            for nc in map(int, a.nchains.split(",")):
                r = run(inst, k, nc, a.niters)
                if r is None:
                    print(f"{inst:9s} {'':>2s} {k:10s} {nc:7d}  (does not fit / failed)")
                    continue
                ks = r["kernel_s"]
                pps = r["proposals"] / ks
                sweeps = r["total_sweeps"]
                vps = sweeps * nc * (m - 1) / ks  # node visits = internal nodes per sweep
                occ = r["node_visits"] / (r["lpc"] * r["warp_steps"]) if r["warp_steps"] else float("nan")
                print(f"{inst:9s} {r['nw']:2d} {k:10s} {nc:7d} {ks:8.3f} {vps:10.3e} {pps:11.3e} "
                      f"{sweeps / ks:14.1f} {occ:8.3f}", flush=True)


if __name__ == "__main__":
    main()
