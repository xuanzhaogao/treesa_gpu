# treesa_gpu

GPU contraction-order optimizer for tensor networks whose indices all have dimension 2 (e.g. the clause
networks of k-SAT). It is a CUDA port of TreeSA, the simulated-annealing optimizer of
[OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl) /
[omeco](https://github.com/GiggleLiu/omeco). It runs thousands of annealing chains at once.

## Performance

Compared with omeco's TreeSA (Rust, multithreaded). Tables and raw data: [`docs/REPORT.md`](docs/REPORT.md).

| what | result |
|---|---|
| Annealing throughput (Quadro RTX 6000) | 9.2·10^9 node visits/s on random 3-SAT n = 100: **about 300× omeco on all 16 CPU cores** |
| Tree quality at equal wall time (random 3-SAT n = 60–120) | tc lower by **1.8–11.5 log2 units**, i.e. 3.4×–2900× fewer operations, within omeco's default run time |
| Against omeco's heavy setting (32 trials × 200 sweeps, 10–25 s) | better trees in **1.1–2.5 s** |
| Contraction time on an H100 ([booltn](https://github.com/xuanzhaogao/booltn)), GPU-time energy `tc+h100` | trees contract **4–64× faster** than omeco trees (grid 3-SAT L = 22: 1.24 s → 19 ms) |
| Slicing inside the annealer | modeled total time **1.5–341× lower** than omeco TreeSA + `slice_code` (geometric mean 7.2×); random 3-SAT n = 70 measured **41.7 s** vs about 13 min projected for the omeco tree |
| Scale | any number of labels (tested to 2025, grid L = 45), up to 32767 tensors |

Where it does not win:
* Easy instances, because each call costs about 0.3 s of process start.
* Grid slicing runs, which take as long as omeco on 96 cores; the GPU wins on contraction time, not on
  optimizer wall time.
* `mode="h100"` on its own, which gets stuck; use `"tc+h100"`.

## Features

* **Same algorithm, more chains.** The Julia move set is ported exactly; a CPU reference port matches
  the GPU kernels bit for bit. At equal trial counts it reproduces omeco's quality.
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
pip install -e .                 # pure Python; optional: omeco for init="greedy"
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
tests/run_checks.sh treesa_gpu/bin/treesa_gpu   # GPU kernels == CPU reference, labels, energies (~1 min)
python tests/test_python.py                     # wrapper: conversions, tc/sc, modeled times, slicing
```

## Limitations

These are the main ones; the full list is in the report.

* Only the TreeSA core is ported. There is no PathDecomp, no omeco `preprocess`, and no fixed slices.
* `mode="gpu"` or `"h100"` started from a hot schedule gets stuck. Use `"tc+h100"`.
* Each call starts a process, which costs about 0.3 s. On easy instances omeco is just as fast.
* Population annealing lowers the mean tc on random SAT. On rugged instances where one lucky chain
  matters (reg3_250), use `"indep"`.
