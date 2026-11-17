"""Spectral initial trees: linear orderings of the tensors from Laplacian eigenvectors, as path ("sweep") trees.

The annealer improves trees by local rotations, which cannot easily change a tree's global structure; on large
networks the starting tree therefore matters (64x64 lattice: tc 89-91 from omeco's greedy tree, 78.6 from the
row-major sweep).  A row-major sweep is a linear ordering with small cuts.  For any network, the Fiedler vector of
the tensor graph's Laplacian (and its neighbours in the spectrum) gives orderings with small cuts without knowing
any geometry: on the 32/48/64 lattices the best spectral direction gives path trees within 0.1-2.3 log2 of the
row-major sweep.

Tensor graph: one vertex per tensor; every label shared by k tensors adds weight 1/(k-1) to each of the k(k-1)/2
pairs (clique expansion of the hyperedge, total weight k/2).  Eigenvectors: dense ``numpy.linalg.eigh`` up to
``dense_max`` tensors, else block Krylov (block size 4) with full reorthogonalization.  Candidate orderings sort the tensors by
cos(a) v1 + sin(a) v2 for angles a in [0, pi) (on square lattices v1 and v2 are degenerate, and the solver returns
an arbitrary mix of the x- and y-waves, i.e. a diagonal ordering with twice the cut), and by random unit
combinations of v1..v3, each also reversed.  Every candidate path tree is scored exactly (``path_cost``, O(sum of
tensor sizes)); the best ``k`` distinct ones are returned, so that different chains can start from different trees.
"""
import math
import numpy as np


def tensor_graph(ixs):
    """Weighted clique expansion as (n, rows, cols, w) with each unordered pair once (rows < cols)."""
    n = len(ixs)
    occ = {}
    for t, ix in enumerate(ixs):
        for l in ix:
            occ.setdefault(l, []).append(t)
    acc = {}
    for ts in occ.values():
        k = len(ts)
        if k < 2:
            continue
        w = 1.0 / (k - 1)
        for i in range(k):
            a = ts[i]
            for j in range(i + 1, k):
                b = ts[j]
                key = (a, b) if a < b else (b, a)
                acc[key] = acc.get(key, 0.0) + w
    if not acc:
        return n, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    keys = np.array(list(acc.keys()), dtype=np.int64)
    return n, keys[:, 0], keys[:, 1], np.array(list(acc.values()))


def _lap_matvec(n, rows, cols, w, deg):
    def mv(x):
        y = deg * x
        np.subtract.at(y, rows, w * x[cols])
        np.subtract.at(y, cols, w * x[rows])
        return y
    return mv


def laplacian_vectors(n, rows, cols, w, k=3, dense_max=3000, seed=0):
    """Eigenvectors 1..k (skipping the lowest) of the graph Laplacian, as columns of an (n, k) array."""
    k = min(k, max(1, n - 2))
    deg = np.zeros(n)
    np.add.at(deg, rows, w)
    np.add.at(deg, cols, w)
    if n <= dense_max:
        L = np.zeros((n, n))
        L[rows, cols] -= w
        L[cols, rows] -= w
        L[np.arange(n), np.arange(n)] += deg
        _, V = np.linalg.eigh(L)
        return V[:, 1:k + 1]
    # block Krylov (block size 4, so that degenerate pairs such as the x- and y-waves of a square lattice are both
    # captured; single-vector Lanczos finds one direction per eigenvalue), full reorthogonalization, Rayleigh-Ritz
    mv = _lap_matvec(n, rows, cols, w, deg)
    bs = 4
    steps = max(2, min(n // bs, 200))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, bs))
    Q = np.zeros((n, 0))
    LQ = np.zeros((n, 0))
    for _ in range(steps):
        for _rep in range(2):
            X -= Q @ (Q.T @ X)
        X, R = np.linalg.qr(X)
        keep = np.abs(np.diag(R)) > 1e-10
        X = X[:, keep]
        if X.shape[1] == 0:
            break
        LX = np.column_stack([mv(X[:, j]) for j in range(X.shape[1])])
        Q = np.hstack([Q, X])
        LQ = np.hstack([LQ, LX])
        X = LX.copy()
    T = Q.T @ LQ
    _, S = np.linalg.eigh((T + T.T) / 2)
    return Q @ S[:, 1:k + 1]


def path_cost(order, ixs, iy=()):
    """(tc, sc) in log2 of the path tree contracting the tensors in ``order`` left to right (same numbers as
    trees.complexity on the path tree).  Step i contracts the prefix (labels alive across the cut) with tensor i."""
    m = len(order)
    pos = np.empty(m, dtype=np.int64)
    pos[np.asarray(order)] = np.arange(m)
    first, last = {}, {}
    for t, ix in enumerate(ixs):
        p = pos[t]
        for l in ix:
            if l not in first or p < first[l]:
                first[l] = p
            if l not in last or p > last[l]:
                last[l] = p
    labs = list(first)
    f = np.array([first[l] for l in labs], dtype=np.int64)
    la = np.array([last[l] for l in labs], dtype=np.int64)
    iys = set(iy)
    if iys:
        la = np.where(np.array([l in iys for l in labs]), m, la)
    diff = np.zeros(m + 2, dtype=np.int64)
    np.add.at(diff, f + 1, 1)
    np.add.at(diff, la + 1, -1)
    alive = np.cumsum(diff)[:m + 1]            # alive[i]: labels of the prefix 0..i-1 needed after it (i >= 1)
    newc = np.bincount(f, minlength=m)         # labels first seen at position i
    c = alive[1:m] + newc[1:m]                 # step i = 1..m-1: |L(prefix_{i-1}) u L(t_i)|
    c[0] = len(set(ixs[order[0]]) | set(ixs[order[1]])) if m > 1 else 0   # step 1: the first tensor is a leaf
    out = alive[2:m + 1] if m > 1 else np.zeros(0, np.int64)                # output of step i (prefix 0..i)
    sc = int(max(out.max() if len(out) else 0, max((len(ix) for ix in ixs), default=0)))
    cm = int(c.max()) if len(c) else 0
    tc = cm + math.log2(float(np.sum(np.exp2(c - cm)))) if len(c) else 0.0
    return tc, sc


def path_tree(order):
    t = {"tensor_index": int(order[0])}
    for i in order[1:]:
        t = {"args": [t, {"tensor_index": int(i)}]}
    return t


def spectral_orders(ixs, iy=(), k=8, nangles=24, nrandom=24, seed=0, dense_max=3000):
    """The k best distinct spectral orderings: list of (tc, sc, order), best first."""
    n = len(ixs)
    if n < 3:
        return [(0.0, 0, list(range(n)))]
    n_, rows, cols, w = tensor_graph(ixs)
    V = laplacian_vectors(n, rows, cols, w, k=3, dense_max=dense_max, seed=seed)
    d = V.shape[1]
    rng = np.random.default_rng(seed)
    dirs = []
    if d >= 2:
        for a in range(nangles):
            th = math.pi * a / nangles
            v = np.zeros(d); v[0], v[1] = math.cos(th), math.sin(th)
            dirs.append(v)
    else:
        dirs.append(np.ones(1))
    for _ in range(nrandom if d >= 3 else 0):
        v = rng.standard_normal(d)
        dirs.append(v / np.linalg.norm(v))
    cands, seen = [], set()

    def score(v):
        x = V @ v
        o = np.argsort(x, kind="stable")
        best = None
        for oo in (o, o[::-1]):
            key = hash(oo.tobytes())
            if key in seen:
                continue
            seen.add(key)
            tc, sc = path_cost(oo.tolist(), ixs, iy)
            cands.append((tc, sc, oo.tolist(), v))
            best = tc if best is None else min(best, tc)
        return best

    for v in dirs:
        score(v)
    # refine the angle around the best few directions in the (v1, v2) plane: on square lattices the best ordering is
    # one wave with a small tilt (which orders the sites within a column), easy to miss on a coarse grid
    if d >= 2:
        cands.sort(key=lambda c: (c[0], c[1]))
        starts = []
        for c in cands:
            th = math.atan2(c[3][1], c[3][0]) if len(c[3]) >= 2 else 0.0
            if all(abs(th - t0) > 1e-9 for t0 in starts):
                starts.append(th)
            if len(starts) == 3:
                break
        for th0 in starts:
            th, step = th0, math.pi / nangles / 2
            cur = min(c[0] for c in cands if len(c[3]) >= 2 and abs(math.atan2(c[3][1], c[3][0]) - th0) < 1e-9)
            for _ in range(8):
                improved = False
                for dth in (-step, step):
                    v = np.zeros(d); v[0], v[1] = math.cos(th + dth), math.sin(th + dth)
                    t = score(v)
                    if t is not None and t < cur - 1e-9:
                        cur, th, improved = t, th + dth, True
                        break
                if not improved:
                    step /= 2
    cands.sort(key=lambda c: (c[0], c[1]))
    return [(c[0], c[1], c[2]) for c in cands[:k]]
