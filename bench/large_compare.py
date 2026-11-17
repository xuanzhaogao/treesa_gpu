"""Large networks: GPU TreeSA vs omeco (greedy, TreeSA) on networks with 1000-8000 tensors / labels.

Instances (all indices of dimension 2):
  sqL      L x L square-lattice graph network (one tensor per vertex, one label per edge); a row-major
           sweep gives sc = L + 1, tc = log2 sum of its steps (the 'sweep' row), the optimum is close to it
  reg3_N   random 3-regular graph network, N vertices (seed 1)
  gridL    lattice-local 3-SAT, L x L variables, 3 L^2 clauses (bench/instances.grid)
  satN     random 3-SAT, alpha 4.26
Objective: pure tc (sc_target 10000, so the sc term never acts); both optimizers get the same score.

  python bench/large_compare.py --method gpu   --instances sq32,sq48 --out bench/results/large_compare.jsonl
  RAYON_NUM_THREADS=12 python bench/large_compare.py --method omeco --instances sq32 --out ...
"""
import argparse, json, math, os, random, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.setrecursionlimit(1000000)
import instances  # noqa: E402

SC_TARGET = 10000.0


def sq(L):
    lab = {}
    ixs = [[] for _ in range(L * L)]
    for x in range(L):
        for y in range(L):
            v = x * L + y
            for u in ((x + 1) * L + y if x + 1 < L else None, x * L + y + 1 if y + 1 < L else None):
                if u is not None:
                    e = len(lab); lab[(v, u)] = e
                    ixs[v].append(e); ixs[u].append(e)
    return ixs, []


def reg3(n, seed=1):
    import networkx as nx
    g = nx.random_regular_graph(3, n, seed=seed)
    ixs = [[] for _ in range(n)]
    for e, (a, b) in enumerate(g.edges()):
        ixs[a].append(e); ixs[b].append(e)
    return ixs, []


def get(name):
    if name.startswith("sq"):
        return sq(int(name[2:]))
    if name.startswith("reg3_") and int(name[5:]) >= 500:
        return reg3(int(name[5:]))
    if name.startswith("grid"):
        return instances.grid(int(name[4:]))
    if name.startswith("sat"):
        return instances.sat(int(name[3:]))
    return instances.get(name)


def path_tree(order):
    t = {"tensor_index": order[0]}
    for i in order[1:]:
        t = {"args": [t, {"tensor_index": i}]}
    return t


def stats(d, ixs, iy):
    from treesa_gpu import trees as tu
    d = tu.recompute_outputs(d, ixs, iy)
    tc, sc = tu.complexity(d, ixs)
    return round(tc, 3), sc


def base(name, ixs):
    return dict(instance=name, m=len(ixs), labels=len({l for ix in ixs for l in ix}), host=socket.gethostname().split(".")[0])


def run(method, name, cfg):
    ixs, iy = get(name)
    rec = base(name, ixs)
    rec.update(method=method, **cfg)
    if method == "sweep":
        tc, sc = stats(path_tree(list(range(len(ixs)))), ixs, iy)
        rec.update(tc=tc, sc=sc, wall=0.0)
        return rec
    if method in ("omeco_greedy", "omeco_treesa"):
        import omeco
        sizes = {l: 2 for ix in ixs for l in ix}
        opt = omeco.GreedyMethod() if method == "omeco_greedy" else omeco.TreeSA(
            ntrials=cfg["ntrials"], niters=cfg["niters"], score=omeco.ScoreFunction(sc_target=SC_TARGET))
        t0 = time.time()
        tree = omeco.optimize_code(ixs, iy, sizes, opt)
        rec["wall"] = round(time.time() - t0, 2)
        rec["threads"] = os.environ.get("RAYON_NUM_THREADS")
        rec["tc"], rec["sc"] = stats(tree.to_dict(), ixs, iy)
        return rec
    import treesa_gpu as tg
    kw = dict(init="greedy")                # the configurations without "init" are from greedy (as before init="auto")
    if cfg.get("init") == "sweep":          # refine the row-major sweep (sq lattices) from a colder start
        kw = dict(init="tree", init_tree=path_tree(list(range(len(ixs)))))
    if cfg.get("init") == "spectral":       # path trees along spectral orderings (treesa_gpu.spectral), betas from 1
        kw = dict(init="spectral", spectral_k=cfg.get("spectral_k", 4))
    for k in ("order", "kernel", "par_follow"):    # sweep order / kernel (default: the package defaults)
        if k in cfg:
            kw[k] = cfg[k]
    if cfg.get("beta0") is not None:
        b0 = cfg["beta0"]
        kw["betas"] = [b0 + 0.05 * i for i in range(int(round((15 - b0) / 0.05)) + 1)]
    t0 = time.time()
    tree, info = tg.optimize(ixs, iy, nchains=cfg["nchains"], niters=cfg["niters"], sc_target=SC_TARGET,
                             scheme=cfg["scheme"], seed=cfg.get("seed", 1), mode="tc", timeout=7200, **kw)
    rec["wall"] = round(time.time() - t0, 2)
    rec.update(tc=round(info["tc_py"], 3), sc=info["sc_py"], kernel_s=info.get("kernel_s"), kernel=info.get("kernel"),
               t_init=round(info.get("t_greedy") or 0.0, 2), init_tcs=info.get("init_tcs"),
               t_anneal=info.get("t_anneal"), order=info.get("order"), gpu=subprocess.run(
                   ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                    os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                   capture_output=True, text=True).stdout.strip())
    return rec


CONFIGS = {
    "sweep": [{}],
    "omeco_greedy": [{}],
    "omeco_treesa": [dict(ntrials=10, niters=50)],              # omeco default
    "gpu": [dict(nchains=576, niters=n, scheme=s) for n in (5, 20) for s in ("pa", "indep")],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="gpu | omeco (greedy + TreeSA) | sweep")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--niters", default=None, help="gpu: comma list overriding 5,20")
    ap.add_argument("--configs", default=None, help='JSON list of configs, e.g. [{"nchains":72,"niters":50,"scheme":"indep"}]')
    a = ap.parse_args()
    methods = {"gpu": ["gpu"], "omeco": ["omeco_greedy", "omeco_treesa"], "omeco_treesa": ["omeco_treesa"],
               "sweep": ["sweep"]}[a.method]
    for name in a.instances.split(","):
        for meth in methods:
            cfgs = json.loads(a.configs) if a.configs else CONFIGS[meth]
            if meth == "gpu" and a.niters:
                cfgs = [dict(nchains=576, niters=int(n), scheme=s) for n in a.niters.split(",") for s in ("pa", "indep")]
            if meth == "sweep" and not name.startswith("sq"):
                continue
            for cfg in cfgs:
                try:
                    rec = run(meth, name, cfg)
                except Exception as e:  # noqa: BLE001
                    rec = dict(base(name, get(name)[0]), method=meth, **cfg, error=repr(e)[-400:])
                line = json.dumps(rec)
                print(line, flush=True)
                with open(a.out, "a") as f:
                    f.write(line + "\n")
