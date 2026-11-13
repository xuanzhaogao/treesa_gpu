"""Benchmark instances: random 3-SAT, omeco 3-regular graphs, lattice-local 3-SAT."""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "py"))
from satgen import random_ksat, sat_einsum, grid_ksat  # noqa: E402

GRAPHS = os.path.join(HERE, "..", "..", "..", "omeco", "benchmarks", "graphs")


def sat(n, seed=1, alpha=4.26, k=3):
    ixs, iy, _ = sat_einsum(random_ksat(n, round(alpha * n), k, seed), n)
    return ixs, iy


def reg3(n):
    """omeco benchmark 3-regular graph: one tensor per vertex, one label per edge."""
    with open(os.path.join(GRAPHS, f"reg3_{n}.json")) as f:
        d = json.load(f)
    if "ixs" in d:
        return [list(ix) for ix in d["ixs"]], list(d["iy"])
    # edge-list format (reg3_220): vertices -> tensors, edges -> labels
    nv = d["vertices"]
    off = min(min(p) for p in d["edge_list"])  # 1-based in the json
    ixs = [[] for _ in range(nv)]
    for e, (a, b) in enumerate(d["edge_list"]):
        ixs[a - off].append(e)
        ixs[b - off].append(e)
    return [ix for ix in ixs if ix], []


def lattice_sat(L, seed=1, alpha=4.0, radius=1, k=3):
    """3-SAT on an L x L grid of variables; each clause = a random variable plus k-1 distinct
    others within Chebyshev distance `radius` (open boundary).  Treewidth ~ O(L)."""
    rng = random.Random(seed)
    n = L * L
    ixs = []
    for _ in range(round(alpha * n)):
        x, y = rng.randrange(L), rng.randrange(L)
        nb = [(x + dx) * L + (y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
              if (dx or dy) and 0 <= x + dx < L and 0 <= y + dy < L]
        ixs.append([x * L + y] + rng.sample(nb, k - 1))
    return ixs, []


def grid(L, alpha=3.0, seed=1, radius=1, k=3):
    """booltn_gpu/py/satgen.grid_ksat (local 3-SAT on an L x L grid), as requested for slicing."""
    clauses, n = grid_ksat(L, alpha, k, radius, seed)
    ixs, iy, _ = sat_einsum(clauses, n)
    return ixs, iy


INSTANCES = {
    "sat40": lambda: sat(40, 1),
    "sat60": lambda: sat(60, 1),
    "sat80": lambda: sat(80, 1),
    "sat100": lambda: sat(100, 1),
    "sat120": lambda: sat(120, 1),
    "reg3_100": lambda: reg3(100),
    "reg3_250": lambda: reg3(250),
    "lat10": lambda: lattice_sat(10, 1),
    "lat14": lambda: lattice_sat(14, 1),
    "sat70": lambda: sat(70, 1),
    "sat2000": lambda: sat(2000, 1),
}
for _L in (22, 24, 28, 32, 36, 40, 45):
    INSTANCES[f"grid{_L}"] = (lambda L: (lambda: grid(L)))(_L)


def get(name):
    return INSTANCES[name]()
