"""Benchmark instances: random 3-SAT, omeco 3-regular graphs, lattice-local 3-SAT."""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


# --- the two generators below are copies of booltn.satgen (same RNG calls -> identical instances) ---
def random_ksat(n, m, k, seed=0):
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        vs = rng.sample(range(1, n + 1), k)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return clauses


def grid_ksat(L, alpha, k=3, radius=1, seed=0):
    rng = random.Random(seed)
    n = L * L
    clauses = []
    for _ in range(round(alpha * n)):
        cx, cy = rng.randrange(L), rng.randrange(L)
        cand = [(x, y) for x in range(max(0, cx - radius), min(L, cx + radius + 1))
                for y in range(max(0, cy - radius), min(L, cy + radius + 1))]
        pts = rng.sample(cand, k)
        clauses.append([(x * L + y + 1) * (1 if rng.random() < 0.5 else -1) for x, y in pts])
    return clauses, n


def sat_einsum(clauses, n):
    """One tensor per clause, labels = 0-based variables; scalar output."""
    ixs = [[abs(l) - 1 for l in c] for c in clauses]
    return ixs, [], {l: 2 for ix in ixs for l in ix}

# omeco benchmark graphs (github.com/GiggleLiu/omeco, benchmarks/graphs); override with $OMECO_GRAPHS
GRAPHS = os.environ.get("OMECO_GRAPHS", os.path.join(HERE, "..", "..", "..", "omeco", "benchmarks", "graphs"))


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
    """booltn.satgen.grid_ksat (local 3-SAT on an L x L grid), as requested for slicing."""
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
