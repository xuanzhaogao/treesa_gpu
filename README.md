# treesa_gpu

GPU contraction-order optimizer for tensor networks whose indices all have dimension 2 (e.g. the clause
networks of k-SAT). It is a CUDA port of TreeSA, the simulated-annealing optimizer of
[OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl) /
[omeco](https://github.com/GiggleLiu/omeco). It runs thousands of annealing chains at once, and also parallelizes inside each chain: every sweep visits large
sets of non-adjacent tree nodes concurrently (their moves commute), so a sweep takes about 7 parallel phases
instead of one step per tree level.

## Performance

Compared with omeco's TreeSA (Rust, multithreaded). Tables and raw data: [`docs/REPORT.md`](docs/REPORT.md).

| what | result |
|---|---|
| Annealing throughput (Quadro RTX 6000) | 9.2·10^9 node visits/s on random 3-SAT n = 100: **about 300× omeco on all 16 CPU cores** |
| Tree quality at equal wall time (random 3-SAT n = 60–120) | tc lower by **1.8–11.5 log2 units**, i.e. 3.4×–2900× fewer operations, within omeco's default run time |
| Against omeco's heavy setting (32 trials × 200 sweeps, 10–25 s) | better trees in **1.1–2.5 s** |
| Contraction time on an H100 ([booltn](https://github.com/xuanzhaogao/booltn)), GPU-time energy `tc+h100` | trees contract **4–64× faster** than omeco trees (grid 3-SAT L = 22: 1.24 s → 19 ms) |
| Slicing inside the annealer | modeled total time **1.5–341× lower** than omeco TreeSA + `slice_code` (geometric mean 7.2×); random 3-SAT n = 70 measured **41.7 s** vs about 13 min projected for the omeco tree |
| Large networks (1000–8000 tensors, RTX 6000 vs omeco on 12 threads) | **spectral start (default)**: chains start from path trees along spectral orderings of the tensor graph; ahead of omeco on 7 of 8 networks in less wall time than omeco's runs, by 1.8–31 log2 units (random 3-SAT n = 500: 318.7 in 9 s vs 327.7 in 82 s; 3-regular n = 2000: 245.4 vs 276.4 at ~40 s; grid 3-SAT L = 45: 83.0 in 39 s vs 90.7 in 187 s; 64×64 lattice: 78.6 in 18 s vs 91.1–93.4). Exception: grid 3-SAT L = 32 (61.8 vs omeco's 60.4). Parallel sweep order (default): sweeps **2.4–9.5× faster** than the pre-order kernels, **11–17×** when refining a path-like tree (64×64 lattice from the row-major sweep, 72 chains × 20: 22.4 s → 1.6 s). Spending the time on longer chains, tc at matched time drops on 4 of the 6 networks tested with several seeds and is level on the 48×48 and 64×64 lattices (random 3-SAT n = 500: 340.3 → 329.0 in ~4 s; grid 3-SAT L = 32: 62.5 → 59.6 in ~20 s). From greedy the GPU is ahead of omeco at every budget on 6 of 8 networks by 1.3–20 log2 units (n = 500: 319.7 in 90 s vs 327.7 in 82 s; 3-regular n = 2000: 254.5 in 90 s vs 268.1 in 146 s); grid 3-SAT L = 32 only at short budgets; the 48×48 lattice is level from greedy, and 61.7 in 2 s from the row-major sweep |
| Scale | any number of labels (tested to 8064, 64×64 lattice), up to 32767 tensors |

Where it does not win:
* Easy instances, because each call costs about 0.3 s of process start.
* Large networks (thousands of labels): label sets are dense bitsets over all labels, so each visit costs
  O(#labels); hot runs on the largest lattice are memory-bandwidth-bound. The annealer's local moves rarely change a
  tree's global structure, so the start tree decides a lot (64×64 lattice: 89–91 from greedy, 78.6 from the spectral
  start). See the report's "Large networks" section.
* With slicing, the default start stays greedy (`init="auto"`): from a path start far above the sc target the slicer
  cuts labels early and keeps them (grid 3-SAT L = 24, T = 36: 4× worse modeled time; other instances 1.3–2× better).
* The parallel order is a different visiting order than Julia's: per sweep it is up to 2.5 log2 units worse on
  2 of 5 networks (it wins at matched time). `order="preorder"` reproduces the Julia sweep bit for bit.
* Grid slicing runs, which take as long as omeco on 96 cores; the GPU wins on contraction time, not on
  optimizer wall time.
* `mode="h100"` on its own, which gets stuck; use `"tc+h100"`.

## Features

* **Same algorithm, more chains.** The Julia move set is ported exactly; a CPU reference port matches
  the GPU kernels bit for bit. At equal trial counts it reproduces omeco's quality.
* **Spectral initial trees** (`init="spectral"`, the default without slicing): path trees along orderings of the
  tensors by Laplacian eigenvectors of the tensor graph (best of many directions, each scored exactly); they recover
  the row-major sweep on lattices without knowing the geometry, and help on random SAT and regular graphs too
  (random 3-SAT n = 100: 68.7 vs 70.5 from greedy). `treesa_gpu.spectral`; numpy only.
* **Parallel sweeps inside a chain** (`order="par"`, default): moves at nodes that are not parent and child
  touch disjoint data and commute, so each phase visits an independent set of nodes chosen by random priorities;
  O(log n) phases per sweep. Validated bit for bit against a CPU reference of the same order.
  `order="preorder"` gives the Julia sweep.
* **Population schemes:** independent chains (best of N), population annealing, or truncation selection.
* **GPU-time energies.** Besides omeco's tc/sc score, the annealer can minimize the modeled contraction
  time of the booltn kernels on an H100.
* **Slicing inside the annealer** (`slicing=True`). The slice set is annealed together with the tree.

The H100 measurements of these trees are in the research repository [booltn-gpu](https://github.com/xuanzhaogao/booltn-gpu),
which includes this package as a submodule.

## Install

```bash
module load cuda/12.8.0          # in a login shell
make                             # -> treesa_gpu/bin/treesa_gpu for the local GPU (ARCH=native)
pip install -e .                 # pure Python + numpy; optional: omeco for init="greedy"
```

Cross-builds: `make ARCH=sm_90a OUT=treesa_gpu/bin/treesa_gpu_sm90` (H100) or `ARCH=sm_75`
(RTX 6000). Pick a binary at run time with `TREESA_GPU_BINARY`. The package does not depend on `booltn`.

## Use

```python
import treesa_gpu

# ixs: one label list per tensor; iy: output labels (labels may be any hashable values)
tree, info = treesa_gpu.optimize(ixs, iy, nchains=1024, niters=20, sc_target=30, scheme="pa", init="greedy")
print(info["tc_py"], info["sc_py"], info["wall"])        # tree is an omeco-format dict

# GPU-time objective plus slicing, for the booltn H100 engine:
tree, info = treesa_gpu.optimize(ixs, iy, nchains=576, niters=20, sc_target=36, scheme="pa",
                                 mode="tc+h100", slicing=True, slice_every=2,
                                 stage2_betas=[3 + 13 * i / 299 for i in range(300)])
json.dump(treesa_gpu.runner_json(tree, info), open("tree.json", "w"))   # {"tree", "sliced"} for booltn
```

Main options of `optimize` (all of them are documented in its docstring):

| option | values |
|---|---|
| `mode` | `"tc"`: omeco score (tc, plus an sc penalty above `sc_target`). `"gpu"` / `"h100"`: modeled GPU time. **`"tc+h100"`**: anneal with `tc`, then continue the same population cold with `h100` (recommended for wall time) |
| `scheme` | `"indep"` (best of N), `"pa"` (population annealing), `"trunc"` (replace the worst 25 %) |
| `init` | `"greedy"` (omeco GreedyMethod), `"random"`, `"tree"` (`init_tree=`) |
| `slicing`, `slice_every`, `max_slices` | in-annealer slicing; `info["sliced"]` lists the sliced labels |
| `gpu_model`, `h100_model` | constants of the GPU-time models (defaults: `treesa_gpu.trees.GPU_MODEL{,_V2}`, fitted for booltn on an H100 PCIe) |

Environment: `TREESA_GPU_BINARY` selects the binary. `TREESA_GPU_SSH=host` runs it on another node over
ssh; the temp dir `TREESA_GPU_TMP` must then be on a shared filesystem. `TREESA_GPU_LOCK=/tmp/x.lock`
wraps each run in `flock`, which matters on a shared GPU.

## Layout

| path | content |
|---|---|
| `csrc/` | CUDA kernels and host driver (`treesa_gpu.cu`), shared code (`common.h`), CPU reference port (`cpu_ref.h`) |
| `treesa_gpu/driver.py` | `optimize`, `runner_json`, conversion between omeco dicts and the binary's format |
| `treesa_gpu/trees.py` | tc/sc and both GPU-time energies in Python (the tests check the binary against them) |
| `tests/` | `run_checks.sh <binary>` (kernel validation matrix), `test_python.py` (wrapper validation), `data/` |
| `bench/` | throughput, omeco comparison, h100 study, slicing comparison; raw data and summaries in `bench/results/` |
| `docs/REPORT.md` | design, validation and all result tables |

## Tests

```bash
tests/run_checks.sh treesa_gpu/bin/treesa_gpu   # GPU kernels == CPU reference, labels, energies (~5 min)
python tests/test_python.py                     # wrapper: conversions, tc/sc, modeled times, slicing
```

## Limitations

These are the main ones; the full list is in the report.

* Only the TreeSA core is ported. There is no PathDecomp, no omeco `preprocess`, and no fixed slices.
* `mode="gpu"` or `"h100"` started from a hot schedule gets stuck. Use `"tc+h100"`.
* Each call starts a process, which costs about 0.3 s. On easy instances omeco is just as fast.
* Population annealing lowers the mean tc on random SAT. On rugged instances where one lucky chain
  matters (reg3_250), use `"indep"`.
