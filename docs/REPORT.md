# treesa_gpu — design and results report

A CUDA port of the TreeSA contraction-order optimizer (`OMEinsumContractionOrders.jl/src/treesa.jl`)
specialized to tensor networks whose indices all have dimension 2 (k-SAT networks, `log2 size = popcount`).
It runs hundreds to thousands of annealing chains concurrently, adds population schemes on top of
best-of-N, has GPU-runtime-aware energies, supports any number of labels (tested to 2025; e.g. grid
3-SAT up to L = 45) and can anneal a **slice set together with the tree** (in-annealer slicing).

Developed and validated on **ccmlin078**: Quadro RTX 6000 (Turing, sm_75, 72 SMs, 24 GB, 6 MB L2,
64 KB smem/block) and 2x Xeon Gold 6234 (16 cores / 32 threads).  Later measurements also on the
**H100 PCIe** of workergpu165 (`make ARCH=sm_90a OUT=treesa_gpu/bin/treesa_gpu_sm90`; every run under
`flock /tmp/booltn_gpu.lock`) and omeco baselines on **worker7273** (AMD EPYC 9474F, 96 cores).  Each
table states its machine.  The code uses no Hopper-only features.

## Quick start

See [`../README.md`](../README.md) (build, install, Python API). This file is the full design and
results report; tables were produced by the scripts in `bench/` (raw data in `bench/results/`).

## Design

### Algorithm (exact port of Julia `optimize_tree_sa!`, TreeDecomp, `rw_weight = 0`)
For each β in `βs`, `niters` sweeps; a sweep is the pre-order recursion of `optimize_subtree!`: at an
internal node pick a random applicable rule (`ruleset`: none if both children are leaves; rules 1:2 if
only the left child is internal; 3:4 if only the right one is; 1:4 otherwise), compute
`tcsc_diff`/`abcacb` (new intermediate `ac = (a ∪ c) ∩ (b ∪ d)`, d = this node's output — correct for
hyperedges), `dE = (max(sc, sc+dsc) > sc_target ? sc_weight : 0)*dsc + dtc` with
`sc = max(|d|, |pivot child|)` (Julia `_sc`), accept if `rand() < exp(-β dE)`, then recurse into the
node's *new* children.  Node costs follow Julia `tcscrw`: `sc = |z|`, `tc = |z| + |x ∩ y \ z|`, and
`dtc = log2(2^tc_l1 + 2^tc_r1) - log2(2^tc_l0 + 2^tc_r0)` over the two contractions the rule changes.
All four rules are the same operation in slot form: swap the moved grandchild M with the uncle O
(keeping slots), and give the pivot child the labels `X = (K|O)&(M|d)`.

### Data structures
* Node ids: leaves `0..m-1` (= input tensor ids), internal nodes `m..2m-2`, root fixed at `2m-2`
  (rules never change which node is the root, as in Julia).
* Per chain (SoA, 32-bit words): `labels[m-1][W]` (output-label bitset of each internal node,
  `W = ceil(nlabels/32)` rounded to 1,2,4,8,12,16) and `children[m-1]` (`left | right<<16`).
  Leaf label sets are the same for all chains and live in one shared read-only table.
  `log2 size = popcount`, so every complexity is a handful of `AND`/`ANDN` + `__popc`.
  sat100: 8.5 KB per chain.
* No parent pointers: the sweep uses an explicit stack (sequential kernel) or a work list.
* Initial trees are built on the host (OpenMP): Julia `random_exprtree` (random bipartitions) per
  chain, or one given tree (omeco `GreedyMethod`) replicated; internal labels are derived from the
  topology by occurrence counting (correct for hyperedges and open labels).

### Counter-based RNG → order-independent, bit-reproducible sweeps
Every proposal at internal node `v` in global sweep `s` of chain slot `c` draws
`Philox4x32-10(counter = {v, s, c, 0}, key = seed)` (word 0 → rule, word 1 → Metropolis `u`).
Each node is visited exactly once per sweep, so the random numbers do not depend on the order in
which nodes are visited.  After a node is processed its two subtrees never interact again in that
sweep (a move at `u` reads/writes only `u`, its children and grandchildren, all inside `u`'s subtree),
so **any** topological order of the pre-order recursion gives the *identical* result.  This makes
the parallel-frontier kernel provably equivalent to the sequential sweep and lets the tests compare
GPU kernels and a CPU reference bit for bit.

### Kernels (`--kernel`, measured below)
| kernel | mapping | storage |
|---|---|---|
| `seq` (b) | one thread per chain, sequential pre-order with explicit stack | global |
| `seqsm` (c) | one thread per chain, a block's chains staged in shared memory per launch | smem |
| `warp` (a') | **one warp (or 16/8 lanes, `--lpc`) per chain; lanes process up to 32 ready nodes concurrently** (work list of nodes whose parent is done, in smem) | global |
| `warpsm` | as `warp`, chain staged in shared memory for the launch (`niters` sweeps) | smem |
| `auto` | `warpsm` if 8 chains fit per block, or if the population exceeds L2 and ≥4 fit; else `warp` | |

The literal option (a) "lanes own bitset words" was not built: here `W` is 2–12 words, so ≥ 20 of 32
lanes would idle, while lanes-own-nodes keeps lanes busy with independent work and also cuts the
per-chain latency (the critical path is the tree depth, not the number of nodes).  Each lane keeps
its node's 5 label sets (`K, M, P, O, d`) in registers and uses vector loads (`uint4`).

### Population schemes (`--scheme`)
After every β level the per-chain energy `E = log2(Σ_v 2^cost(v)) + sc_weight·max(0, sc − sc_target)`
is computed on the GPU (one warp per chain).
* `indep`: independent chains, best-of-N (the omeco/Julia `ntrials` semantics).
* `pa`: population annealing — systematic resampling with weights `exp(-(β_{k+1} − β_k)(E_i − E_min))`
  before moving to the next temperature; copies get fresh random streams because the Philox key
  includes the slot index.
* `trunc`: truncation selection — the worst 25% of chains are overwritten by copies of the best 25%.
The best tree seen at any level boundary (omeco score `2^tc + sc_weight·max(0, 2^sc − 2^sc_target)`)
is kept on the host and returned.

### Energy modes (`--mode`)
* `tc` (i): Julia/omeco energy as above.
* `gpu` (ii): each pairwise contraction (x,y)→z costs `t(B,M,N,K)` of `trees.gpu_time` with
  `B=|x∩y∩z|, M=|(x\y)∩z|, N=|(y\x)∩z|, K=|(x∩y)\z|` (popcounts); constants are runtime parameters
  (`gpu_model=` / `--gm peak,bw,m0,n0,k0,launch`).  Everything is evaluated in log2 domain (no
  overflow: `ops` reaches 2^130 on these instances), `dtc = log2(t_l1+t_r1) − log2(t_l0+t_r0)`.
  The sc term is unchanged.  The reported score is the total modeled time Σt over the tree.
* `gpu2`: the measured H100 model `trees.gpu_time_v2` (batched small contractions for output
  rank ≤ 14; otherwise `3·launch + permutes + min(cc, tc-core)`), mirrored in log2 domain with all
  nine constants as runtime parameters (`gpu_model2=` / `--gm2`).
* `tc+gpu` / `tc+gpu2`: anneal in `tc` on `βs`, then continue the *same population* in `gpu`/`gpu2`
  on a cold schedule (`stage2_betas`, default `1:0.05:16`).  The stage-1 best tree is kept as the
  incumbent, so the result is never worse than the `tc` tree in the new objective.  See results:
  `gpu`/`gpu2` alone do not work well.

### Wide label sets (> 512 labels)
Instances with ≤ 512 labels use the register fast path (`W` ∈ {1,2,4,8,12,16} words, the five label
sets of a move in registers).  Larger ones (or `--wide`) use a streaming path: `NW` = ceil(labels/32)
rounded up to a multiple of 4, and a move streams the five label sets in 16-byte chunks, accumulating
the popcount counters `|d|, |P|, |X|` and (B,M,N,K) of the four affected contractions; the accepted
`X` is written in a second pass.  Both paths feed the same `energy_from_counts` (identical float code),
so the wide path is bit-identical to the fast path and to the CPU reference (tested).  Limits: node
ids are 16-bit (≤ 32767 tensors); per-chain memory ≈ (m−1)(NW+1)·4 B (grid L=45: 1.6 MB/chain);
the work list of the warp kernel lives in shared memory (2 bytes/node).

### Slicing (`slicing=True`)
* Every chain carries a slice set S (an `NW`-word bitset after the tree).  Stored node labels stay
  *unsliced*: the labels of the sliced network are exactly `labels & ~S` (slicing removes the label from
  every tensor), so a slice change needs no relabelling and every cost is a popcount of masked sets —
  this is Julia's `Slicer` (sliced labels get log2 size 0).  Tree moves use the masked sets in
  `dtc`/`dsc` (the `2^|S|` factor cancels in local moves).
* Slice moves: after every `slice_every` sweeps the chain's warp makes `slice_moves` proposals —
  add a random label of a randomly chosen largest internal tensor (p = 1/2), remove a random sliced
  label (1/4) or swap (1/4) — and accepts them with Metropolis at the current β on the total energy
  `E = |S| + log2 Σ_v cost_v(masked) + sc_weight·max(0, sc − sc_target)`  (cost = 2^tc for `tc`,
  modeled seconds for `gpu`/`gpu2`), evaluated over the whole tree by the warp (3 tree passes per
  proposal; the candidate is picked in node-id order with ballots).  RNG keys `{0x10000+2j, sweep,
  chain}` keep it reproducible.  The population schemes copy S with the tree.
* Selection: feasibility first (sc ≤ sc_target), then the total cost; if the best chain still has
  sc > sc_target, the host repairs it with the Julia TreeSASlicer rule (slice the label occurring most
  often among the largest tensors) — `info["n_repair"]` (0 in all runs below).
* Staging: `mode="tc+gpu2", slicing=True` keeps S through both stages.  With slice moves the default
  stage-2 start β = 1 is too hot (the population leaves the stage-1 optimum and never comes back);
  β₂ = 3 → 16 works best on sat60/sat70/grid24 (see results).
* Output (`optimize(..., slicing=True)`): the tree dict has internal `iy` recomputed on the **sliced**
  ixs (`recompute_outputs(tree, ixs_sliced, iy_sliced)`), `info["sliced"]` lists the sliced labels with
  their original ids, and `runner_json(tree, info)` gives `{"tree", "sliced"}` for the H100 runner.
  Per-slice tc/sc and totals (`tc_total_py = tc + |S|`, `gpu_time2_total_py = Σt × 2^|S|`) are recomputed
  with `treesa_gpu.trees` on the sliced network.

## Validation (`tests/`)

* `tests/run_checks.sh` — CLI `--check` mode, 48 chains × 10 β-levels × 4 sweeps per configuration:
  **A** 6 instances (sat40, sat100, reg3_100, reg3_250, lat10, lat14) × 6 kernel variants (`seq, warp,
  seqsm, warpsm, warp --lpc 16, warpsm --lpc 8`) × 3 energy modes (`tc, gpu, gpu2`) × {random, greedy}
  init; **B** the wide label path (`--wide` on sat40/sat100/reg3_250, automatic on grid24 (576 labels)
  and grid45 (2025 labels)) × {seq, warp} × {tc, gpu2}; **C** fixed random slice sets (8 labels per
  chain) on sat100/reg3_250/grid24 × {seq, warp, warpsm} × {tc, gpu2}; **D** slice moves on
  (sat100/grid24/grid28 × {warp, warpsm} × {tc, gpu2}).  Per configuration:
  1. every chain is a valid tree and the **incrementally maintained labels equal a from-scratch
     recomputation from the topology** (occurrence counting) for every internal node;
  2. GPU per-chain energy kernel == host double-precision recomputation (criterion |Δlog2| < 1e-3,
     observed ≤ 6.6e-5 (float accumulation); sc exact);
  3. the other kernel family (sequential vs warp-parallel) gives **bit-identical** trees and labels;
  4. a straightforward CPU port of the Julia recursion (`csrc/cpu_ref.h`: `std::vector<int>` label
     sets, literal `ruleset`/`abcacb`/`tcscrw`/`update_tree!`) gives **bit-identical** trees.
  (3) and (4) run for fixed slice sets (C: the reference uses Julia's slicer semantics, sliced log2
  size 0) and are skipped for D (only the warp kernels implement slice moves).
  Result (`tests/check_results_sm75.txt`): **248/248 passed**, 18 skipped (chains that do not fit in
  Turing's 64 KB shared memory for the smem kernels).
* `tests/test_python.py` — `trees.complexity` == omeco `tree.complexity` on omeco greedy/TreeSA
  trees; dict↔children round trip; for all schemes/inits/modes the returned dict contains every
  leaf exactly once, its internal `iy` equals `recompute_outputs`, and the binary's tc / sc /
  modeled times (`gpu_time`, `gpu_time_v2`) equal `treesa_gpu.trees` on the returned dict; with slicing
  (sat60, reg3_250, grid24 × {tc, tc+gpu2}) the tree's `iy` equal `recompute_outputs(tree, ixs_sliced,
  [])`, per-slice tc / sc / #sliced and `gpu_time_v2 × 2^|S|` equal the binary's numbers, sc ≤ target,
  and `runner_json` round-trips.  **62/62 passed** (`tests/test_python_results.txt`).
* Cross-GPU determinism: the H100 (sm_90a build) reproduced all 96 GPU results of the slicing
  comparison bit-for-bit (same #sliced, same modeled time) — trees do not depend on the GPU.
* All 72 saved trees in `trees/gpusa_slice/` (repository root) were re-validated independently (leaves,
  `iy` on sliced ixs, sc ≤ target, tc and modeled time equal to `index.json`).

## Kernel throughput (RTX 6000)

`bench/throughput.py` (full 300-level schedule, 4 sweeps/level, greedy init, sc_target 30; kernel
time only).  *Node visits* = sweeps × chains × (m−1) internal nodes (the unit of omeco's sweep);
*proposals* = visits at nodes with an applicable rule.  Full table: `bench/results/throughput_rtx6000.txt`.

| instance (m, W) | kernel | chains | node visits/s | sweeps/s per chain | lane occupancy |
|---|---|---|---|---|---|
| sat60 (255, 2) | warpsm | 9216 | **1.58e10** | 6.7e3 | 0.37 |
| sat60 | warpsm | 576 | 9.2e9 | 6.3e4 | 0.37 |
| sat60 | warp | 576 | 6.2e9 | 4.2e4 | 0.37 |
| sat60 | seqsm / seq | 2304 | 1.9e9 / 1.4e9 | 3.2e3 / 2.4e3 | – |
| sat100 (426, 4) | warpsm | 9216 | **9.2e9** | 2.3e3 | 0.48 |
| sat100 | warp | 576 | 7.8e9 | 3.2e4 | 0.48 |
| sat100 | warp | 72 | 1.4e9 | **4.5e4** | 0.48 |
| sat100 | warp | 9216 | 2.1e9 | 5.4e2 | 0.48 |
| sat100 | warp --lpc 16 / 8 | 576 | 5.2e9 / 3.4e9 | 2.1e4 / 1.4e4 | 0.71 / 0.87 |
| sat100 | seq / seqsm | 2304 | 1.0e9 / 5.8e8 | 1.1e3 / 5.9e2 | – |
| lat14 (784, 8) | warp | 576 | 2.6e9 | 5.8e3 | 0.45 |
| lat14 | warpsm (2 chains/block) | 576 | 1.1e9 | 2.5e3 | 0.45 |
| reg3_250 (250, 12) | warp | 9216 | 1.9e9 | 8.2e2 | 0.19 |
| reg3_250 | warp | 576 | 1.5e9 | 1.0e4 | 0.19 |

CPU reference point (same box, omeco TreeSA, 1 thread, `preprocess=False`, 3000 sweeps):
sat100 1.9e6, lat14 2.5e6, reg3_250 2.4e6 node visits/s per core (4.4e3 / 3.3e3 / 9.7e3 sweeps/s).
So one GPU chain on the `warp` kernel sweeps ~10× faster than one CPU core (sat100: 4.5e4 vs 4.4e3
sweeps/s), and the whole GPU does 9.2e9 / 1.9e6 ≈ 4800× one core in node visits on sat100, i.e.
≈ 300× all 16 cores even assuming perfect CPU scaling.
Single-thread CPU reference port (`cpu_ref.h`, label vectors, not optimized): ~1e6 proposals/s.

### Throughput vs label width (`bench/results/throughput_vs_W_rtx6000.txt`)

`warp` kernel (global memory), greedy init, full 300-level schedule, 2 sweeps/level, kernel time only.
`+wide` = the streaming path forced on an instance that also fits the register path.

| instance | m | labels | NW | path | visits/s, 72 chains | visits/s, 576 chains | sweeps/s/chain (72) |
|---|---|---|---|---|---|---|---|
| sat100 | 426 | 100 | 4 | fast / wide | 8.6e8 / 7.1e8 | 5.3e9 / 4.3e9 | 2.8e4 / 2.3e4 |
| lat14 | 784 | 196 | 8 | fast / wide | 5.2e8 / 5.4e8 | 2.1e9 / 2.4e9 | 9.3e3 / 9.7e3 |
| reg3_250 | 250 | 372 | 12 | fast / wide | 2.8e8 / 2.1e8 | 1.4e9 / 1.0e9 | 1.6e4 / 1.2e4 |
| grid22 | 1452 | 484 | 16 | fast / wide | 4.7e8 / 3.7e8 | 1.14e9 / 1.10e9 | 4.5e3 / 3.5e3 |
| grid24 | 1728 | 576 | 20 | wide | 3.0e8 | 7.6e8 | 2.4e3 |
| grid28 | 2352 | 784 | 28 | wide | 2.4e8 | 4.9e8 | 1.4e3 |
| grid32 | 3072 | 1024 | 32 | wide | 2.8e8 | 6.5e8 | 1.3e3 |
| grid45 | 6075 | 2025 | 64 | wide | 1.5e8 | 2.5e8 | 343 |
| sat2000 | 8520 | 2000 | 64 | wide | 1.6e8 | 2.3e8 | 263 |

* The streaming path costs 0–25 % vs the register path at equal width (it is even slightly faster on
  lat14), so instances above 512 labels lose nothing to the generalization itself.
* Beyond L2-resident sizes the kernel is memory bound: a visit streams 5 label sets of 4·NW bytes, so
  visits/s fall roughly ∝ 1/NW (grid45 at 576 chains: 2.5e8 visits/s × ~1.3 KB ≈ 330 GB/s, about half
  of the RTX 6000's 672 GB/s).  For scale: omeco does 1.9–2.5e6 visits/s per core on the small
  instances (not measured on the wide ones), i.e. ~90–130× less than the GPU at NW = 64.
* Lane occupancy grows with m (0.6–0.94 for m ≥ 1452): big trees have wide frontiers.

Same on the **H100 PCIe** (workergpu165, sm_90a build, every run under `flock`;
`bench/results/throughput_vs_W_h100.txt`), visits/s:

| instance | NW | warp, 576 chains | warp, 2304 | warpsm, 576 / 2304 | warp +wide, 576 / 2304 |
|---|---|---|---|---|---|
| sat100 | 4 | 6.3e9 | 1.46e10 | 8.0e9 / 1.55e10 | 5.2e9 / 1.01e10 |
| lat14 | 8 | 4.1e9 | 9.7e9 | 4.9e9 / 6.0e9 | 3.2e9 / 6.1e9 |
| reg3_250 | 12 | 1.5e9 | 3.4e9 | 1.9e9 / 2.9e9 | 1.3e9 / 2.5e9 |
| grid22 | 16 | 2.9e9 | 5.2e9 | 0.85e9 / 0.96e9 | 2.2e9 / 4.2e9 |
| grid24 | 20 | 1.9e9 | 3.2e9 | – | – |
| grid28 | 28 | 1.5e9 | 2.4e9 | – | – |
| grid32 | 32 | 1.7e9 | 2.5e9 | – | – |
| grid45 | 64 | 0.91e9 | 0.96e9 | – | – |
| sat2000 | 64 | 0.55e9 | 0.67e9 | – | – |

The H100 is 2.4–3.6× the RTX 6000 on the wide instances at 576 chains (grid45: 9.1e8 vs 2.5e8), and the
end-to-end slicing runs on the grids were 2.1–2.5× faster (identical results).

What the measurements say about the mapping:
* **Lanes-own-nodes (warp) ≫ thread-per-chain (seq)** by 2–10× at equal chain count: the sequential kernel is a chain of
  dependent memory accesses per node; the warp kernel overlaps up to 32 independent nodes, so a
  sweep costs ~(depth + nodes/32) steps instead of `nodes` steps.
* **Shared memory helps only while chains are small** (Turing: 64 KB/block): sat60/sat100 gain up to
  4×, lat14 (28 KB/chain) and reg3_250 (13 KB/chain + 12 KB leaf table) lose.  Global-memory `warp`
  collapses once the population exceeds the 6 MB L2 (sat100: 7.8e9 → 2.0e9 visits/s from 576 to
  2304 chains).  `--kernel auto` encodes this.
* **Packing 2–4 chains per warp does not help** (`--lpc 16/8`): occupancy rises (0.48 → 0.87) but
  the chains' control flow diverges, so the warp serializes them; throughput drops.  Lane occupancy
  is limited by the frontier width (balanced SAT trees ~0.4–0.5, deep reg3 trees ~0.19).
* `seqsm` loses because only 1–20 chains (one per thread) fit in a 64 KB block, i.e. less than one
  warp per SM.
* Indicative H100 numbers (measured before moving to the workstation, older build, very short
  runs, sat100, 2 sweeps/level): `warp` 1.1e9 / 4.1e9 / 1.05e10 / 1.33e10 proposals/s at
  132 / 528 / 2112 / 8448 chains vs `seq` 6.2e7 / 2.4e8 / 9.1e8 / 2.0e9 — same picture, ~2× the RTX 6000.

## Results (all on ccmlin078: RTX 6000 vs 2x Xeon Gold 6234)

Setup: instances from `bench/instances.py` — random 3-SAT at α = 4.26 (`satN`: N variables,
m = 4.26 N clauses, seed 1), omeco's `reg3_100`/`reg3_250` (graph TN, no hyperedges) and `lat14`
(14×14 lattice-local 3-SAT, α = 4, m = 784, tc ≈ 2^29).  Score `ScoreFunction(sc_target=30,
sc_weight=1)`, Julia schedule β = 0.01:0.05:15 (300 levels), greedy initial tree, mode `tc`.
omeco = `omeco.optimize_code(..., TreeSA(ntrials, niters, score))` (its default `preprocess=True`
unless noted) with `RAYON_NUM_THREADS` = 32 or 16.  Wallclock is end-to-end from Python (GPU
includes process start + CUDA context ≈ 0.25–0.35 s, omeco greedy, I/O).  GPU runs use
`kernel="auto"`, which picked `warpsm` for sat60/sat80/reg3_100 (576 and 4608 chains) and
sat100/sat120 (4608), and `warp` otherwise.  Raw data:
`bench/results/compare.jsonl`; all tables incl. per-instance quality-vs-time:
`bench/results/summary.md`.  tc and sc below are log2.

**Sanity check: same algorithm, same statistics.**  At equal trial counts the GPU reproduces omeco's
quality: sat100 GPU 10 chains × niters 50 → tc 69.2–74.8 over 6 seeds (omeco 10×50: 72.93; omeco
best-of-1/2/3 trials: 79.07/78.07/72.93); reg3_250 GPU 1 chain → 46.1–50.0, 10 chains → 44.8–47.0,
32 chains → 42.1–45.3 (omeco 10×50: 47.00, 32×20..200: 44.05–47.00).  All GPU gains below therefore
come from *more chains / longer schedules per second* and from the population schemes.

### Headline (tc / sc / wallclock, single run, GPU seed 1)

| instance | omeco default, 32 thr | omeco default, 16 thr | omeco 32 trials x niters=200 | GPU indep 576 x 50 | GPU PA 576 x 50 | GPU PA 4608 x 50 | GPU PA 576 x 800 |
|---|---|---|---|---|---|---|---|
| sat60 | 44.56 / 40 / 1.72s | 44.56 / 40 / 1.72s | 44.36 / 41 / 10.26s | 43.62 / 40 / 0.51s | 43.43 / 40 / 0.51s | 43.02 / 40 / 1.48s | 43.06 / 40 / 3.37s |
| sat80 | 59.12 / 54 / 2.41s | 59.12 / 54 / 2.42s | 56.19 / 52 / 14.89s | 56.84 / 53 / 0.60s | 55.40 / 50 / 0.60s | 55.28 / 51 / 2.52s | 54.59 / 51 / 4.33s |
| sat100 | 72.93 / 69 / 3.24s | 72.93 / 69 / 3.16s | 69.42 / 64 / 19.36s | 70.27 / 66 / 0.79s | 69.33 / 65 / 0.83s | 66.50 / 63 / 3.65s | 66.82 / 63 / 7.32s |
| sat120 | 94.17 / 89 / 3.91s | 94.17 / 89 / 3.94s | 87.34 / 80 / 24.77s | 87.33 / 82 / 0.89s | 83.92 / 79 / 0.91s | 81.46 / 77 / 4.70s | 81.83 / 77 / 8.62s |
| lat14 | 28.91 / 22 / 4.08s | 28.91 / 22 / 4.09s | 28.71 / 23 / 24.59s | 28.71 / 23 / 3.41s | 29.09 / 23 / 3.22s | 29.09 / 23 / 36.31s | 28.70 / 23 / 49.27s |
| reg3_100 | 17.16 / 11 / 0.52s | 17.16 / 11 / 0.51s | 17.16 / 11 / 3.44s | 17.16 / 11 / 0.62s | 17.16 / 11 / 0.62s | 17.16 / 11 / 2.88s | 17.16 / 11 / 5.31s |
| reg3_250 | 47.00 / 33 / 1.54s | 47.00 / 33 / 1.44s | 46.05 / 32 / 9.25s | 40.30 / 34 / 1.87s | 40.19 / 34 / 2.00s | 40.11 / 34 / 11.32s | 40.40 / 33 / 24.34s |

### Matched wallclock: best tc reached within the time of an omeco run

| instance | budget | omeco (any config) | GPU indep | GPU PA | GPU trunc |
|---|---|---|---|---|---|
| sat60 | omeco default 1.7s | **44.56** (40, 10x50, 1.7s) | **43.06** (40, 576x200, 1.1s) | **42.78** (39, 576x200, 1.1s) | **43.50** (40, 576x50, 0.5s) |
| sat60 | omeco 32x200 10.3s | **44.36** (41, 32x200, 10.3s) | **42.88** (39, 576x800, 3.4s) | **42.78** (39, 576x200, 1.1s) | **43.50** (40, 576x50, 0.5s) |
| sat80 | omeco default 2.4s | **59.12** (54, 10x50, 2.4s) | **55.53** (51, 576x200, 1.4s) | **54.74** (51, 576x200, 1.4s) | **55.29** (50, 576x200, 1.4s) |
| sat80 | omeco 32x200 14.9s | **56.19** (52, 32x200, 14.9s) | **54.95** (51, 576x800, 4.4s) | **54.59** (51, 576x800, 4.3s) | **55.29** (50, 576x200, 1.4s) |
| sat100 | omeco default 3.2s | **72.93** (69, 10x50, 3.2s) | **68.06** (64, 576x200, 2.1s) | **66.51** (62, 576x200, 2.1s) | **67.23** (62, 576x200, 2.1s) |
| sat100 | omeco 32x200 19.4s | **69.42** (64, 32x200, 19.4s) | **67.06** (62, 576x800, 7.4s) | **66.50** (63, 4608x50, 3.6s) | **67.23** (62, 576x200, 2.1s) |
| sat120 | omeco default 3.9s | **94.17** (89, 10x50, 3.9s) | **83.24** (78, 576x200, 2.5s) | **82.65** (78, 576x200, 2.5s) | **85.46** (81, 576x200, 2.5s) |
| sat120 | omeco 32x200 24.8s | **87.34** (80, 32x200, 24.8s) | **82.62** (78, 576x800, 8.9s) | **81.46** (77, 4608x50, 4.7s) | **83.06** (76, 576x800, 8.5s) |
| lat14 | omeco default 4.1s | **28.74** (23, 32x20, 2.5s) | **28.71** (23, 576x50, 3.4s) | **28.70** (23, 576x20, 1.5s) | **28.70** (23, 576x50, 3.1s) |
| lat14 | omeco 32x200 24.6s | **28.71** (23, 32x200, 24.6s) | **28.70** (23, 576x200, 12.7s) | **28.70** (23, 4608x20, 14.6s) | **28.70** (23, 576x50, 3.1s) |
| reg3_100 | omeco default 0.5s | **17.16** (11, 32x20, 0.4s) | **17.16** (11, 576x5, 0.3s) | **17.16** (11, 576x20, 0.4s) | **17.16** (11, 576x20, 0.4s) |
| reg3_100 | omeco 32x200 3.4s | **17.16** (11, 32x200, 3.4s) | **17.16** (11, 576x5, 0.3s) | **17.16** (11, 576x20, 0.4s) | **17.16** (11, 576x20, 0.4s) |
| reg3_250 | omeco default 1.5s | **44.05** (33, 32x20, 1.0s) | **40.46** (34, 576x20, 0.9s) | **42.73** (34, 4608x5, 1.4s) | **43.81** (36, 576x5, 0.5s) |
| reg3_250 | omeco 32x200 9.2s | **44.05** (33, 32x20, 1.0s) | **40.30** (34, 576x50, 1.9s) | **40.19** (34, 576x50, 2.0s) | **40.08** (34, 576x50, 2.1s) |

**Where the GPU wins:** on random 3-SAT, where quality is limited by annealing effort.  Within
omeco-default's wallclock the GPU finds trees with tc lower by 1.8 (sat60), 4.4 (sat80), 6.4
(sat100) and 11.5 (sat120) log2 units, i.e. 3.4× to 2900× fewer operations; it also beats omeco's
heavy setting (32 trials × niters 200, 10–25 s) within 1.1–2.5 s.  On reg3_250 (rugged, heavy-tailed
outcome distribution) the advantage is purely best-of-576 (indep ≈ PA).

### Population schemes at equal work (576 chains, same schedule; tc mean ± sd / best over 3 seeds)

| instance | niters | indep | PA | trunc (25%) |
|---|---|---|---|---|
| sat60 | 20 | 44.47 ± 0.24 / 44.19 | 44.26 ± 0.52 / 43.70 | 44.27 ± 0.95 / 43.26 |
| sat60 | 50 | 43.64 ± 0.13 / 43.53 | 43.31 ± 0.16 / 43.11 | 43.97 ± 1.03 / 43.24 |
| sat80 | 20 | 57.68 ± 0.39 / 57.42 | 56.37 ± 0.70 / 55.89 | 56.73 ± 0.10 / 56.64 |
| sat80 | 50 | 56.70 ± 0.32 / 56.33 | 55.27 ± 0.15 / 55.11 | 55.87 ± 0.44 / 55.41 |
| sat100 | 20 | 72.96 ± 0.22 / 72.76 | 71.06 ± 1.00 / 70.06 | 70.98 ± 0.21 / 70.74 |
| sat100 | 50 | 69.91 ± 0.49 / 69.35 | 69.21 ± 0.57 / 68.42 | 68.20 ± 0.96 / 67.21 |
| sat120 | 20 | 92.38 ± 1.99 / 90.42 | 87.16 ± 1.24 / 85.72 | 90.42 ± 3.99 / 87.82 |
| sat120 | 50 | 88.04 ± 0.71 / 87.33 | 84.41 ± 0.56 / 83.92 | 84.95 ± 2.46 / 82.14 |
| lat14 | 20 | 28.72 ± 0.00 / 28.71 | 28.74 ± 0.06 / 28.70 | 29.02 ± 0.28 / 28.70 |
| lat14 | 50 | 28.71 ± 0.00 / 28.71 | 29.00 ± 0.19 / 28.70 | 28.85 ± 0.21 / 28.70 |
| reg3_100 | 20 | 17.16 ± 0.00 / 17.16 | 17.16 ± 0.00 / 17.16 | 17.16 ± 0.00 / 17.16 |
| reg3_100 | 50 | 17.16 ± 0.00 / 17.16 | 17.16 ± 0.00 / 17.16 | 17.16 ± 0.00 / 17.16 |
| reg3_250 | 20 | 41.55 ± 1.60 / 40.46 | 42.28 ± 1.60 / 40.43 | 42.39 ± 1.99 / 40.29 |
| reg3_250 | 50 | 40.87 ± 0.85 / 40.30 | 41.41 ± 2.42 / 40.19 | 42.03 ± 2.70 / 40.08 |

* **PA and truncation beat independent chains at equal work on random SAT**: sat120 −5.2 (niters 20)
  and −3.6 (niters 50) log2 units for PA, sat80 −1.3/−1.4, sat100 −1.9/−0.7.  At niters 5 the effect
  is larger still (sat100: indep 83.1, PA 76.3, trunc 73.8; sat120: 104.6 / 94.5 / 98.3).
* **Not on lat14 / reg3_250**: lat14 is saturated (all ≈ 28.7) and PA's selection occasionally
  freezes a slightly worse tree (29.0 ± 0.2); on reg3_250, where the best result is a rare lucky
  chain, collapsing diversity hurts the mean (indep 40.9 ± 0.9 vs PA 41.4 ± 2.4 vs trunc 42.0 ± 2.7).
* Diversity: at the end of a PA (or trunc) run the whole population typically descends from a
  single initial chain (median 1 family; 77 of 97 PA runs ≤ 2; `families` column of the trace).
  Exception: reg3_100, where many chains reach the same optimal energy, keeps 33–1093 families.
* Beyond ~200 sweeps/level × 576 chains, more effort barely helps on these instances
  (sat100: PA 576×200 66.51, 4608×50 66.50, 576×800 66.82, 4608×200 67.08).

### Quality vs wallclock, sat120 (seed 1)

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.43 | GPU pa 576x niters=5 | 94.54 | 90 |
| 0.44 | GPU indep 576x niters=5 | 104.59 | 101 |
| 0.46 | GPU trunc 576x niters=5 | 98.33 | 93 |
| 0.59 | GPU pa 576x niters=20 | 87.86 | 83 |
| 0.61 | GPU indep 576x niters=20 | 90.42 | 85 |
| 0.61 | GPU trunc 576x niters=20 | 88.41 | 84 |
| 0.85 | omeco ntrials=32 niters=5 (32 thr) | 107.59 | 104 |
| 0.89 | GPU indep 576x niters=50 | 87.33 | 82 |
| 0.90 | GPU trunc 576x niters=50 | 86.01 | 76 |
| 0.91 | GPU pa 576x niters=50 | 83.92 | 79 |
| 0.93 | GPU indep 4608x niters=5 | 103.36 | 100 |
| 1.02 | GPU pa 4608x niters=5 | 90.84 | 86 |
| 2.13 | GPU indep 4608x niters=20 | 90.42 | 85 |
| 2.26 | GPU pa 4608x niters=20 | 84.88 | 79 |
| 2.48 | GPU indep 576x niters=200 | 83.24 | 78 |
| 2.49 | GPU trunc 576x niters=200 | 85.46 | 81 |
| 2.54 | GPU pa 576x niters=200 | 82.65 | 78 |
| 2.66 | omeco ntrials=32 niters=20 (32 thr) | 94.33 | 91 |
| 3.91 | omeco ntrials=10 niters=50 (32 thr) | 94.17 | 89 |
| 3.93 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 95.13 | 90 |
| 3.94 | omeco ntrials=10 niters=50 (16 thr) | 94.17 | 89 |
| 4.53 | GPU indep 4608x niters=50 | 86.50 | 82 |
| 4.70 | GPU pa 4608x niters=50 | 81.46 | 77 |
| 6.22 | omeco ntrials=32 niters=50 (32 thr) | 92.29 | 88 |
| 8.54 | GPU trunc 576x niters=800 | 83.06 | 76 |
| 8.62 | GPU pa 576x niters=800 | 81.83 | 77 |
| 8.88 | GPU indep 576x niters=800 | 82.62 | 78 |
| 16.47 | GPU pa 4608x niters=200 | 81.72 | 77 |
| 16.57 | GPU indep 4608x niters=200 | 83.24 | 78 |
| 24.77 | omeco ntrials=32 niters=200 (32 thr) | 87.34 | 80 |

### GPU-runtime objective (mode ii; modeled Σt in seconds / tc / sc; PA 576 × 50)

| instance | omeco default tree | GPU [tc] | GPU [gpu] | GPU [tc+gpu] |
|---|---|---|---|---|
| sat60 | 1.12 / 44.6 / 40 | 1.25 / 43.4 / 40 | 424 / 55.6 / 51 | 0.278 / 45.4 / 40 |
| sat80 | 1.09e+04 / 59.1 / 54 | 2.2e+03 / 55.4 / 50 | 5.28e+07 / 72.6 / 67 | 336 / 55.8 / 50 |
| sat100 | 1.09e+09 / 72.9 / 69 | 1.61e+08 / 69.3 / 65 | 1.94e+14 / 95.1 / 89 | 2.69e+06 / 72.1 / 63 |
| sat120 | 1.21e+14 / 94.2 / 89 | 3.45e+12 / 83.9 / 79 | 6.24e+19 / 114.3 / 108 | 9.89e+10 / 88.1 / 78 |
| lat14 | 0.00318 / 28.9 / 22 | 0.00326 / 29.1 / 23 | 0.531 / 48.9 / 40 | 0.00323 / 35.9 / 27 |
| reg3_100 | 0.000396 / 17.2 / 11 | 0.000396 / 17.2 / 11 | - | - |
| reg3_250 | 0.0497 / 47.0 / 33 | 0.078 / 40.2 / 34 | 0.0209 / 46.0 / 35 | 0.00405 / 43.9 / 32 |

* `tc+gpu` (tc anneal, then the same population annealed in `gpu` mode on β = 1:0.05:16) lowers the
  modeled time vs the tc-optimized tree by 4.5× (sat60), 6.5× (sat80), 60× (sat100), 35× (sat120),
  19× (reg3_250), at a modest tc increase (+0.4 to +4.1); vs omeco's default tree it is 4×–1200×
  faster in the model.  It costs ~2× the wallclock (two schedules).
* `gpu` alone from a hot start fails badly on SAT (2.5–7 orders of magnitude worse than `tc`): once
  the tree is randomized, the energy is nearly flat — every contraction below ~2^35.5 ops costs the
  4 µs launch floor and the m0/n0/k0 padding hides M/N/K differences — so the annealer has no gradient back.
  Starting `gpu` from a cold tree works; hence `tc+gpu`.
* lat14 / reg3_100 are launch-floor dominated (783 contractions × 4 µs = 3.1 ms of 3.2 ms), so the
  objective cannot be improved there.

### Measured H100 model v2 (`gpu2`): tc-optimized vs gpu2-optimized trees

`bench/gpu2_study.py`: 3-SAT α = 4.26, n = 50..70 (instance seed 1), sc_target 30, PA 576 × 50,
greedy init, GPU seed 1.  "time share" splits Σt over the model's three paths (batched small /
CUDA-core `cc` / 1-bit tensor-core `tc`); the next column counts contractions that take the
tensor-core path with K ≥ 9 (and their share of Σt).

| instance | optimizer | modeled Σt v2 (s) | tc | sc | time share small / cc / tc-core | #tensor-core contractions with K≥9 (time share) | dominant (B,M,N,K) path share | wall (s) |
|---|---|---|---|---|---|---|---|---|
| sat50 | omeco TreeSA default (tc) | 5.248e-03 | 36.2 | 32 | 0.00 / 0.70 / 0.30 | 0 (0.00) | (22, 3, 7, 3) cc 0.45 | 1.4 |
| sat50 | GPU PA 576x50 [tc] | 6.153e-03 | 35.8 | 33 | 0.00 / 1.00 / 0.00 | 0 (0.00) | (23, 1, 5, 5) cc 0.53 | 0.7 |
| sat50 | GPU PA 576x50 [gpu2] | 9.927e-03 | 37.3 | 33 | 0.00 / 0.86 / 0.14 | 0 (0.00) | (22, 8, 2, 3) cc 0.37 | 1.1 |
| sat50 | GPU PA 576x50 [tc+gpu2] | 4.846e-03 | 36.7 | 32 | 0.00 / 0.71 / 0.29 | 1 (0.29) | (21, 5, 6, 4) cc 0.54 | 1.2 |
| sat50 | gpu2, cold β=1..16 | 1.701e-02 | 39.0 | 34 | 0.00 / 0.66 / 0.34 | 1 (0.02) | (22, 5, 7, 4) cc 0.58 | 1.0 |
| sat55 | omeco TreeSA default (tc) | 2.768e-01 | 41.3 | 38 | 0.00 / 1.00 / 0.00 | 0 (0.00) | (30, 2, 3, 5) cc 0.55 | 1.6 |
| sat55 | GPU PA 576x50 [tc] | 1.568e-01 | 40.0 | 37 | 0.00 / 1.00 / 0.00 | 0 (0.00) | (27, 8, 2, 2) cc 0.40 | 0.6 |
| sat55 | GPU PA 576x50 [gpu2] | 2.901e-01 | 44.1 | 39 | 0.00 / 0.27 / 0.73 | 1 (0.00) | (23, 8, 5, 8) tc 0.66 | 1.1 |
| sat55 | GPU PA 576x50 [tc+gpu2] | 5.281e-02 | 43.1 | 36 | 0.00 / 0.07 / 0.93 | 1 (0.01) | (18, 9, 9, 7) tc 0.52 | 1.3 |
| sat55 | gpu2, cold β=1..16 | 2.514e-01 | 44.2 | 38 | 0.00 / 0.24 / 0.76 | 2 (0.17) | (24, 7, 6, 7) tc 0.60 | 1.1 |
| sat60 | omeco TreeSA default (tc) | 1.745e+00 | 44.6 | 40 | 0.00 / 0.90 / 0.10 | 1 (0.00) | (30, 4, 5, 5) cc 0.51 | 1.9 |
| sat60 | GPU PA 576x50 [tc] | 1.265e+00 | 43.4 | 40 | 0.00 / 0.98 / 0.02 | 1 (0.02) | (30, 4, 2, 6) cc 0.41 | 0.7 |
| sat60 | GPU PA 576x50 [gpu2] | 6.097e+00 | 46.9 | 42 | 0.00 / 0.78 / 0.22 | 2 (0.22) | (32, 6, 4, 4) cc 0.55 | 1.2 |
| sat60 | GPU PA 576x50 [tc+gpu2] | 9.884e-01 | 45.1 | 40 | 0.00 / 0.59 / 0.41 | 1 (0.04) | (27, 7, 6, 4) cc 0.55 | 1.4 |
| sat60 | gpu2, cold β=1..16 | 7.302e+00 | 52.0 | 43 | 0.00 / 0.06 / 0.94 | 3 (0.57) | (21, 10, 12, 9) tc 0.55 | 1.2 |
| sat65 | omeco TreeSA default (tc) | 6.783e+01 | 50.2 | 46 | 0.00 / 1.00 / 0.00 | 1 (0.00) | (29, 11, 4, 6) cc 0.67 | 2.1 |
| sat65 | GPU PA 576x50 [tc] | 1.139e+01 | 46.5 | 43 | 0.00 / 0.99 / 0.01 | 1 (0.01) | (33, 2, 8, 2) cc 0.35 | 0.7 |
| sat65 | GPU PA 576x50 [gpu2] | 6.946e+01 | 52.3 | 46 | 0.00 / 0.35 / 0.65 | 1 (0.01) | (31, 6, 8, 7) tc 0.48 | 1.4 |
| sat65 | GPU PA 576x50 [tc+gpu2] | 5.105e+00 | 49.2 | 42 | 0.00 / 0.16 / 0.84 | 0 (0.00) | (27, 7, 8, 7) tc 0.58 | 1.6 |
| sat65 | gpu2, cold β=1..16 | 9.904e+01 | 53.2 | 47 | 0.00 / 0.40 / 0.60 | 3 (0.11) | (26, 13, 6, 8) tc 0.49 | 1.4 |
| sat70 | omeco TreeSA default (tc) | 4.504e+02 | 52.9 | 49 | 0.00 / 0.59 / 0.41 | 2 (0.02) | (33, 9, 3, 7) tc 0.38 | 2.2 |
| sat70 | GPU PA 576x50 [tc] | 6.700e+01 | 49.6 | 46 | 0.00 / 1.00 / 0.00 | 1 (0.00) | (33, 3, 9, 4) cc 0.53 | 0.8 |
| sat70 | GPU PA 576x50 [gpu2] | 1.154e+03 | 57.1 | 50 | 0.00 / 0.12 / 0.88 | 1 (0.00) | (34, 7, 9, 7) tc 0.59 | 1.5 |
| sat70 | GPU PA 576x50 [tc+gpu2] | 6.623e+01 | 50.7 | 45 | 0.00 / 0.80 / 0.20 | 2 (0.02) | (35, 5, 5, 5) cc 0.70 | 1.7 |
| sat70 | gpu2, cold β=1..16 | 9.127e+02 | 57.0 | 50 | 0.00 / 0.10 / 0.90 | 2 (0.01) | (32, 10, 8, 7) tc 0.54 | 1.4 |

Stage-2 schedule check (`bench/results/gpu2_stage2_schedules.md`; Σt in s, 2 GPU seeds):

| instance | seed | tc only | tc+gpu2 β2=0.3..16 | β2=1..16 | β2=3..16 | β2=1..16, niters 200 |
|---|---|---|---|---|---|---|
| sat60 | 1 | 1.26 | 1.26 (stage1 1.26, K≥9 tc: 1) | 0.988 (stage1 1.26, K≥9 tc: 1) | 0.802 (stage1 1.26, K≥9 tc: 1) | 0.756 (stage1 1.26, K≥9 tc: 2) |
| sat60 | 2 | 1.03 | 1.03 (stage1 1.03, K≥9 tc: 1) | 0.463 (stage1 1.03, K≥9 tc: 0) | 0.491 (stage1 1.03, K≥9 tc: 1) | 0.499 (stage1 1.03, K≥9 tc: 1) |
| sat65 | 1 | 11.4 | 8.53 (stage1 11.4, K≥9 tc: 2) | 5.1 (stage1 11.4, K≥9 tc: 0) | 6.54 (stage1 11.4, K≥9 tc: 1) | 4.74 (stage1 11.4, K≥9 tc: 2) |
| sat65 | 2 | 18.2 | 16.8 (stage1 18.2, K≥9 tc: 1) | 5.93 (stage1 18.2, K≥9 tc: 0) | 7.44 (stage1 18.2, K≥9 tc: 2) | 6.72 (stage1 18.2, K≥9 tc: 0) |
| sat70 | 1 | 67 | 67 (stage1 67, K≥9 tc: 1) | 66.2 (stage1 67, K≥9 tc: 2) | 65.7 (stage1 67, K≥9 tc: 2) | 64.5 (stage1 67, K≥9 tc: 1) |
| sat70 | 2 | 64.4 | 64.4 (stage1 64.4, K≥9 tc: 1) | 43.5 (stage1 64.4, K≥9 tc: 1) | 42.4 (stage1 64.4, K≥9 tc: 1) | 41.6 (stage1 64.4, K≥9 tc: 1) |

* **`tc+gpu2` lowers the modeled time vs the tc-optimized tree by 1.0–3.1×** (β₂ starting at 1–3;
  a 4× longer stage 2 adds little; β₂ from 0.3 is too hot and returns the stage-1 incumbent), and
  vs omeco's default tree by 1.1–13×, mostly thanks to the stronger tc annealing.
* **`gpu2` alone is worse than plain `tc`** (e.g. sat70: 1154 s vs 67 s; also from a cold start),
  the same flat-landscape problem as `gpu`.
* **The hoped-for many large-K tensor-core contractions do not appear**: `tc+gpu2` does move time
  onto the tensor-core path (sat55: 93 %, sat65: 84 %), but via contractions with K = 7–8;
  contractions with K ≥ 9 on the tensor-core path stay at 0–2 per tree under every schedule.  The
  dominant contractions of these SAT trees are batch-heavy (B = 21–35, K ≤ 9), consistent with
  hyperedges: a variable can only be summed (K) once all its ~13 clauses are inside the subtree.

## Slicing results (`bench/slicing_compare.py`, raw data `bench/results/slicing_compare*.jsonl`)

Instances: random 3-SAT n = 60, 70, 80 (α = 4.26, seed 1) and `booltn.satgen.grid_ksat(L, 3.0, 3, 1, seed=1)` for
L = 22, 24, 28 (1452 / 1728 / 2352 clauses, 484 / 576 / 784 labels — the latter two use the wide path).
sc_target T ∈ {30, 34, 36, 37}.  Every number is recomputed with `treesa_gpu.trees` on the sliced network:
**modeled time = `tree_gpu_time_v2` per slice × 2^#sliced** (the measured H100 model).  Methods:
* `existing trees/` — the trees already in `trees/` (repository root) for the same instance/target (where present);
* omeco `TreeSA()` (default: 10 trials × 300 β × 50 sweeps) + `slice_code(TreeSASlicer())`, both with
  `ScoreFunction(sc_target=T)`;
* GPU TreeSA unsliced (PA, 576 chains × 20 sweeps/level, `sc_target=T`) followed by
  `booltn.slicer.greedy_slice(tree, ixs, T, kind="gpu")`, for modes `tc` and `tc+gpu2`;
* GPU TreeSA with in-annealer slicing, same budget, `slice_every=2`, modes `tc` and `tc+gpu2`
  (stage 2: β = 3 → 16).
Single runs, GPU seed 1.  Cells: modeled total time in s (#sliced); best per row in bold.

| instance | T | existing `trees/` | omeco TreeSA + slice_code | GPU tc → greedy_slice | GPU tc+gpu2 → greedy_slice | GPU tc, in-annealer slicing | **GPU tc+gpu2, in-annealer slicing** |
|---|---|---|---|---|---|---|---|
| sat60 | 30 | 9.64 (13) | 1.8 (10) | 2.39 (10) | 1.2 (10) | 0.781 (9) | **0.514 (9)** |
| sat60 | 34 | 3.12 (8) | 4.66 (8) | 2.32 (7) | 1.83 (6) | 0.82 (5) | **0.685 (5)** |
| sat60 | 36 | 1.68 (4) | 2.32 (5) | 1.33 (4) | 0.784 (4) | 0.83 (6) | **0.453 (3)** |
| sat60 | 37 | - | 3.35 (4) | 1.31 (3) | 1.36 (2) | 0.83 (5) | **0.39 (2)** |
| sat70 | 30 | 714 (19) | 744 (19) | 225 (17) | 45.7 (15) | 118 (18) | **33.7 (15)** |
| sat70 | 34 | 507 (15) | 278 (14) | 213 (14) | 60.2 (12) | 57.2 (12) | **29.3 (11)** |
| sat70 | 36 | 711 (13) | 205 (12) | 136 (10) | 57.6 (10) | 52 (13) | **23.5 (9)** |
| sat70 | 37 | - | 339 (12) | 89.3 (9) | 51.2 (8) | 38.6 (9) | **25.6 (8)** |
| sat80 | 30 | - | 2.56e+04 (24) | 1.01e+04 (23) | 7.06e+03 (22) | 5.62e+03 (21) | **2.87e+03 (21)** |
| sat80 | 34 | - | 1.78e+04 (20) | 6.44e+03 (18) | 2.53e+03 (17) | 3.31e+03 (17) | **1.7e+03 (17)** |
| sat80 | 36 | - | 3.55e+04 (19) | 1.53e+04 (17) | 3.27e+03 (16) | 6.23e+03 (22) | **1.24e+03 (14)** |
| sat80 | 37 | - | 1.85e+04 (17) | 1.64e+04 (17) | 7.81e+03 (17) | 3.94e+05 (31) | **1.52e+03 (14)** |
| grid22 | 30 | - | 0.109 (4) | 0.197 (4) | 0.03 (3) | 1.83 (8) | **0.0285 (3)** |
| grid22 | 34 | - | 0.0999 (0) | 0.0857 (0) | **0.014 (0)** | 1.44 (4) | 0.0149 (0) |
| grid22 | 36 | - | 0.171 (0) | 0.602 (1) | **0.0137 (0)** | 0.998 (2) | 0.0142 (0) |
| grid22 | 37 | - | 0.247 (0) | 0.563 (0) | **0.0137 (0)** | 0.826 (1) | 0.0143 (0) |
| grid24 | 30 | - | 234 (14) | 402 (15) | **15 (11)** | 126 (13) | 20.9 (12) |
| grid24 | 34 | - | 5.94 (5) | 1.73 (3) | 0.8 (2) | 11.1 (8) | **0.7 (2)** |
| grid24 | 36 | - | 1.37 (0) | 2.97 (2) | 1.1 (1) | 2.97 (2) | **0.723 (1)** |
| grid24 | 37 | 1.67 (0) | 1.44 (0) | 3.84 (2) | **0.62 (0)** | 2.25 (1) | 0.98 (0) |
| grid28 | 30 | - | 6.9e+06 (28) | 5.01e+06 (28) | 9.11e+05 (26) | 2.71e+05 (24) | **1.64e+04 (21)** |
| grid28 | 34 | - | 2.61e+05 (20) | 7.16e+04 (18) | 7.56e+03 (16) | 6.64e+03 (14) | **652 (13)** |
| grid28 | 36 | - | 1.1e+04 (14) | 2.73e+03 (11) | 1.05e+04 (15) | 1.24e+04 (15) | **776 (11)** |
| grid28 | 37 | - | 6.47e+04 (15) | 9.49e+03 (13) | 2.06e+03 (12) | 5.47e+03 (12) | **686 (10)** |

* **`tc+gpu2` with in-annealer slicing is the best method in 19/24 cases** and within 1.58× of the best
  in all; geometric mean 1.04× the best (`bench/results/slicing_summary.md`).  Against the best of
  three omeco variants (ccmlin078 32 threads; worker7273 96 threads with ntrials 10 and 96) it is
  better in every case, by 1.5× (min), 5.8× (median), 341× (max), 7.2× (geometric mean).
* The remaining 5 wins are greedy_slice on the *unsliced* tc+gpu2 GPU tree (grid22 T = 34–37, where all
  GPU gpu2 trees need 0 slices and differ by < 5 %, and grid24 T = 37: 0.62 vs 0.98 s).
* In-annealer slicing with the `tc` energy finds the lowest *total tc* (e.g. sat70 T=30: 48.8 vs
  49.6–52.1 for the others, grid28 T=30: 64.3 vs 65.0–71.9) but often over-slices (sc below target): in the H100 model every
  slice re-pays the launch/permute overheads, so for wall time the `gpu2` energy must drive slicing.
* Greedy repair (Julia rule) was never needed (sc ≤ T reached by the annealer in all 48 runs), and the
  kernel's internal per-slice tc / sc / #sliced / modeled time equal the `treesa_gpu.trees` recomputation
  in 48/48 runs.

**Wall time by machine** (same runs; GPU = in-annealer `tc+gpu2`; H100 runs used `flock`, the trees are
bit-identical to the RTX 6000 ones; omeco cells: modeled time / wall):

| instance | T | RTX 6000 wall | H100 wall | omeco ccmlin078 (32 thr) modeled / wall | omeco worker7273 (96 thr) ntrials=10 | ntrials=96 |
|---|---|---|---|---|---|---|
| sat60 | 30 | 1.0 s | 1.0 s | 1.8 / 2.5 s | 1.93 / 1.6 s | 2.47 / 1.9 s |
| sat60 | 37 | 1.1 s | 1.0 s | 3.35 / 2.1 s | 2.18 / 1.3 s | 1.59 / 1.6 s |
| sat70 | 30 | 1.3 s | 1.1 s | 744 / 3.9 s | 355 / 2.6 s | 363 / 2.7 s |
| sat70 | 37 | 1.3 s | 1.1 s | 339 / 3.1 s | 451 / 2.1 s | 99.3 / 2.1 s |
| sat80 | 30 | 1.3 s | 1.2 s | 2.56e+04 / 4.9 s | 2.85e+04 / 3.5 s | 1.8e+04 / 3.5 s |
| sat80 | 37 | 1.4 s | 1.2 s | 1.85e+04 / 4.1 s | 2.28e+04 / 2.9 s | 6.04e+03 / 2.9 s |
| grid22 | 30 | 15.4 s | 7.1 s | 0.109 / 10.3 s | 0.0451 / 6.6 s | 0.145 / 7.4 s |
| grid22 | 37 | 15.2 s | 7.2 s | 0.247 / 9.3 s | 0.247 / 5.9 s | 1.12 / 6.3 s |
| grid24 | 30 | 25.1 s | 11.6 s | 234 / 14.1 s | 50.6 / 9.5 s | 240 / 10.0 s |
| grid24 | 37 | 24.7 s | 11.6 s | 1.44 / 11.4 s | 1.44 / 7.3 s | 10.5 / 10.6 s |
| grid28 | 30 | 51.3 s | 20.7 s | 6.9e+06 / 26.8 s | 4.5e+06 / 18.7 s | 6.41e+08 / 20.5 s |
| grid28 | 37 | 51.2 s | 20.8 s | 6.47e+04 / 25.7 s | 1.08e+05 / 15.2 s | 1.81e+06 / 18.9 s |

* On the SAT instances the GPU run is ~1–1.4 s end-to-end (mostly process start, greedy init and I/O)
  and better than every omeco variant.  On the grids the GPU (20 sweeps/level, two stages) takes about
  as long as omeco on the 96-core node or longer (H100 7–21 s, RTX 6000 15–51 s, omeco 6–21 s) — but
  finds 1.5–340× lower modeled time than the best omeco variant.

**Matched wall time on the grids** (H100, in-annealer `tc+gpu2`, shorter schedules; modeled time / wall):

| instance | T | GPU H100 niters=5 | niters=10 | niters=20 | omeco worker7273 96 thr, ntrials=10 | ntrials=96 |
|---|---|---|---|---|---|---|
| grid22 | 30 | 0.0299 / 2.5 s | 0.03 / 3.8 s | 0.0285 / 7.1 s | 0.0451 / 6.6 s | 0.145 / 7.4 s |
| grid22 | 36 | 0.0351 / 2.3 s | 0.0268 / 3.8 s | 0.0142 / 7.1 s | 0.171 / 5.8 s | 0.633 / 6.4 s |
| grid22 | 37 | 0.0252 / 2.3 s | 0.0237 / 3.8 s | 0.0143 / 7.2 s | 0.247 / 5.9 s | 1.12 / 6.3 s |
| grid24 | 30 | 7.59 / 3.5 s | 22.7 / 5.9 s | 20.9 / 11.6 s | 50.6 / 9.5 s | 240 / 10.0 s |
| grid24 | 36 | 2.52 / 3.4 s | 4.5 / 6.0 s | 0.723 / 11.9 s | 1.37 / 7.1 s | 1.23 / 9.7 s |
| grid24 | 37 | 5.95 / 3.4 s | 1.03 / 6.8 s | 0.98 / 11.6 s | 1.44 / 7.3 s | 10.5 / 10.6 s |
| grid28 | 30 | 2.6e+05 / 6.0 s | 1.79e+03 / 11.0 s | 1.64e+04 / 20.7 s | 4.5e+06 / 18.7 s | 6.41e+08 / 20.5 s |
| grid28 | 36 | 2.38e+03 / 8.4 s | 427 / 10.8 s | 776 / 21.0 s | 7.78e+03 / 15.4 s | 5.25e+04 / 19.6 s |
| grid28 | 37 | 1.9e+03 / 5.6 s | 482 / 10.6 s | 686 / 20.8 s | 1.08e+05 / 15.2 s | 1.81e+06 / 18.9 s |

* At ~1/3 of omeco's wall time (niters 5: 2.3–8.4 s) the GPU is already better on grid22 (all T),
  grid24 T = 30 and grid28 (all T: 1.9e3–2.6e5 vs 7.8e3–6.4e8 s); grid24 T = 37 at niters 10 (1.03 s
  modeled in 6.8 s vs omeco 1.44 s in 7.3 s).  **grid24 T = 36 is the exception**: omeco on 96 cores
  (1.37 s modeled in 7.1 s) beats the GPU at matched time (2.52 s in 3.4 s, 4.5 s in 6.0 s); the GPU
  needs niters 20 (11.9 s) to win (0.72 s).
* Single runs are noisy: quality is not monotone in niters (grid28 T = 30: 2.6e5 → 1.8e3 → 1.6e4 s).

### Saved trees for the H100 runner (`trees/gpusa_slice/` (repository root))

`<name>_sc<T>_tc.json` / `_gpu2.json`: in-annealer slicing runs of the comparison (seed 1);
`<name>_sc<T>_best.json`: the best feasible (sc ≤ T) tree for that instance and target among all my
runs — the comparison runs, extra seeds 2–4 of `tc+gpu2` in-annealer slicing for T = 36, 37
(`bench/extra_seeds.py`; seed spread is large, e.g. grid28 T=37: 444–1977 s), and trees found for a
smaller T (feasible for a larger one).  Format `{"tree", "sliced"}`: internal `iy` computed on the ixs
with the sliced labels removed (`recompute_outputs(tree, ixs_sliced, [])`), `sliced` = original label
ids (0-based variables of `sat_einsum`).  Metrics and provenance in `index.json`; all 72 files
re-validated with `treesa_gpu.trees`.  Names follow the existing files (`r3sat_n60_s1` = `random_ksat(60,
256, 3, seed=1)`, `grid_L24_a3_s1` = `grid_ksat(24, 3.0, 3, 1, seed=1)`; verified to be the same instances
by recomputing the existing trees' `iy`).

| file in `trees/gpusa_slice/` | #sliced | sc | tc/slice | total tc | modeled H100 time (s) | existing `trees/<name>_sc<T>.json` (s) |
|---|---|---|---|---|---|---|
| grid_L22_a3_s1_sc30_best | 3 | 30 | 42.89 | 45.89 | 0.02847 | - |
| grid_L22_a3_s1_sc34_best | 0 | 33 | 45.41 | 45.41 | 0.01301 | - |
| grid_L22_a3_s1_sc36_best | 0 | 33 | 45.41 | 45.41 | 0.01301 | - |
| grid_L22_a3_s1_sc37_best | 0 | 33 | 45.41 | 45.41 | 0.01301 | - |
| grid_L24_a3_s1_sc30_best | 11 | 30 | 44.72 | 55.72 | 15 | - |
| grid_L24_a3_s1_sc34_best | 2 | 34 | 50.29 | 52.29 | 0.7 | - |
| grid_L24_a3_s1_sc36_best | 0 | 36 | 52.16 | 52.16 | 0.5916 | - |
| grid_L24_a3_s1_sc37_best | 0 | 36 | 52.16 | 52.16 | 0.5916 | 1.666 |
| grid_L28_a3_s1_sc30_best | 21 | 30 | 44.01 | 65.01 | 1.636e+04 | - |
| grid_L28_a3_s1_sc34_best | 13 | 34 | 48.17 | 61.17 | 652.5 | - |
| grid_L28_a3_s1_sc36_best | 11 | 36 | 50.46 | 61.46 | 585 | - |
| grid_L28_a3_s1_sc37_best | 9 | 37 | 51.99 | 60.99 | 443.5 | - |
| r3sat_n60_s1_sc30_best | 9 | 30 | 33.83 | 42.83 | 0.5136 | 9.644 |
| r3sat_n60_s1_sc34_best | 9 | 30 | 33.83 | 42.83 | 0.5136 | 3.119 |
| r3sat_n60_s1_sc36_best | 3 | 36 | 40.74 | 43.74 | 0.4532 | 1.685 |
| r3sat_n60_s1_sc37_best | 2 | 37 | 43.08 | 45.08 | 0.3904 | - |
| r3sat_n70_s1_sc30_best | 15 | 30 | 34.55 | 49.55 | 33.65 | 713.8 |
| r3sat_n70_s1_sc34_best | 11 | 34 | 39.19 | 50.19 | 29.34 | 506.7 |
| r3sat_n70_s1_sc36_best | 9 | 36 | 40.37 | 49.37 | 23.52 | 710.6 |
| r3sat_n70_s1_sc37_best | 9 | 36 | 40.37 | 49.37 | 23.52 | - |
| r3sat_n80_s1_sc30_best | 21 | 30 | 33.88 | 54.88 | 2867 | - |
| r3sat_n80_s1_sc34_best | 17 | 34 | 38.16 | 55.16 | 1701 | - |
| r3sat_n80_s1_sc36_best | 14 | 36 | 42.24 | 56.24 | 1015 | - |
| r3sat_n80_s1_sc37_best | 14 | 36 | 42.24 | 56.24 | 1015 | - |

### Where the GPU does not win (honest list)
* **Grid instances with slicing, wall time**: the GPU slicing run (20 sweeps/level, two stages) takes
  as long as or longer than omeco TreeSA + slice_code on 96 cores (H100 7–21 s vs 6–21 s; RTX 6000
  15–51 s) — it wins on modeled time, not on optimizer wall time; at matched wall time omeco wins
  grid24 T = 36.
* **Slicing with the `tc` energy** is worse than every `gpu2`-aware method for the H100 objective
  (over-slicing), and `tc+gpu2` with the default stage-2 start β = 1 returned the stage-1 tree.
* **Easy / saturated instances** (reg3_100, lat14): every method reaches the same tc; the GPU's
  ~0.3 s process + context start-up makes it no faster than omeco (reg3_100: 0.3 s vs 0.4 s; lat14
  equal quality, omeco default's tree even has sc 22 vs 23).
* **Large-m, wide-label instances** run slower per chain (lat14, m = 784, W = 8: 5.8e3 sweeps/s/chain
  at 576 chains; `PA 4608×50` takes 36 s for no gain).
* **Per-chain speed at large populations**: at 9216 chains a sat100 chain advances at 2.3e3
  sweeps/s, slower than one CPU core (4.4e3); with ≤ 2304 chains a GPU chain is faster than a core
  (8.4e3 at 2304, 3.2e4 at 576, 4.5e4 at 72 sweeps/s).  Long single trajectories are not what the GPU is good at.
* **PA hurts on rugged instances where best-of-N matters** (reg3_250, above); use `indep` there.
* **`mode="gpu"` / `"gpu2"` from scratch** (above); and `tc+gpu2` gains only 1.0–3.1× over plain `tc`
  trees in the measured model.
* The timings are single runs; omeco has no Python seed, so its run-to-run variance is unmeasured.

## Known limitations

* Only the TreeSA core (+ in-annealer slicing): TreeDecomp, `rw_weight = 0`; no PathDecomp, omeco's
  `preprocess` simplifier, waist surgery or fine-tune rounds; no fixed (user-given, never removed) slices.  omeco's default `TreeSA()` *does* run `preprocess`;
  here it changed the result only slightly (sat120 94.17 with vs 95.13 without, lat14 28.91 vs 28.83,
  reg3_250 47.00 vs 47.01, identical elsewhere).
* All index dimensions are 2; no repeated label inside one tensor; ≤ 32767 tensors (16-bit node ids).
  Any number of labels (tested to 2025); above 512 the streaming path is memory-bound, and visits/s
  fall ~∝ 1/NW (grid45 on the H100: 9e8 visits/s, ~260 sweeps/s per chain at 576 chains).
* Per-chain memory is ~(m−1)·(NW+1)·4 bytes (grid45: 1.6 MB); the shared-memory kernels need the
  chain (and the leaf table) to fit in one block's shared memory, which on Turing (64 KB) excludes
  lat14 / reg3_250-size chains in groups ≥ 4 and all wide instances.  The warp kernel's per-chain work
  list (2 B/node) is in shared memory too, so m ≤ ~32k.
* Slicing: slice moves are Metropolis moves on a heuristic global energy (not detailed-balanced with the
  tree moves), proposals only add labels of the *largest* internal tensors, and each proposal costs 3
  full-tree passes (`slice_every=2` adds ~50 % run time).  The default stage-2 start β = 1 is too hot
  when slicing; use β₂ from ~3 (as in the benchmarks).  In-annealer slicing with the `tc` energy
  over-slices for the H100 objective; use `tc+gpu2`.  Results are single seeds and noisy (see the
  matched-time table); sc_target is enforced by selection + repair, not by the annealing itself.
* omeco baselines are not reproducible across machines (same config, different results on ccmlin078
  and worker7273, e.g. sat70 T=30: 744 vs 355 s modeled), so treat single omeco numbers as one sample.
* Lane occupancy of the warp kernel is bounded by the width of the "ready" frontier: ~0.4–0.5 on
  SAT trees, ~0.19 on deep (path-like) trees such as reg3_250; packing several chains per warp does
  not help (divergence).
* Population annealing uses the global energy `log2 Σ 2^tc + sc penalty`, while the Metropolis moves
  use Julia's local surrogate `dtc` (not a Δ of that global energy), so PA here is a heuristic
  selection scheme, not an unbiased sampler.  It loses diversity: by the end, the population
  typically descends from a single initial chain.
* `mode="gpu"`/`"gpu2"` on their own get stuck; use `mode="tc+gpu"`/`"tc+gpu2"`.  The GPU cost model is
  taken as given from `trees.gpu_time` / `gpu_time_v2`; v1 heavily penalizes batched (B-heavy) small-M/N/K
  contractions through the tile padding `m0, n0, k0`.
* The Python wrapper starts a process per call (CUDA context + file I/O ≈ 0.25–0.35 s), which
  dominates for small instances.  The best tree is tracked at β-level boundaries only.
* Wallclock comparisons are single runs (workstation lightly loaded; the H100 node is shared, all
  H100 runs under `flock`); omeco's Python API has no seed, so its variance across seeds is not
  measured (GPU variance is, for 3 seeds, in the population-scheme table).

## Files

| path | content |
|---|---|
| `csrc/common.h` | host/device shared code: Philox4x32-10, GPU cost models v1/v2, `MoveCounts`/`acc_counts`/`energy_from_counts` (abcacb + dE from popcount counters), rule selection, Metropolis |
| `csrc/cpu_ref.h` | CPU reference: literal port of Julia `optimize_subtree!` with vector label sets and Julia-`Slicer` masking |
| `csrc/treesa_gpu.cu` | kernels (`k_seq`, `k_warp` with slice moves, `k_energy`, `k_copy`; register fast path W ≤ 16 and wide streaming path W = 0), host init (random/greedy trees, topology labels, slice sets), annealing driver with population schemes, stage 2, slicing selection + greedy repair, `--check` validation mode, CLI |
| `treesa_gpu/driver.py`, `treesa_gpu/trees.py` | Python API `optimize(..., slicing=...)`, `runner_json`, instance writer, omeco-dict ↔ tree conversion |
| `Makefile` | `make` (local GPU, `ARCH=native`) / `make ARCH=sm_90a OUT=treesa_gpu/bin/treesa_gpu_sm90` |
| `tests/run_checks.sh`, `tests/check_results_sm75.txt` | kernel validation matrix and its output |
| `tests/test_python.py` | Python-level validation |
| `tests/data/` | test instances (`m nlabels / k l1..lk / niy ..`) and omeco greedy trees (`left right` per internal node) |
| `bench/instances.py` | random 3-SAT (copy of `booltn.satgen`), omeco `reg3_*` graphs, lattice-local 3-SAT generator |
| `bench/throughput.py` | kernel-mapping throughput benchmark |
| `bench/compare.py`, `bench/summarize.py` | omeco vs GPU quality/wallclock benchmark and tables |
| `bench/gpu2_study.py` | tc vs gpu2 (measured H100 model) study → `bench/results/gpu2_study.md` |
| `bench/slicing_compare.py`, `bench/summarize_slicing.py` | slicing comparison (omeco + slice_code / greedy_slice / in-annealer) on any host, tables → `bench/results/slicing_summary.md` |
| `bench/save_best.py`, `bench/extra_seeds.py` | best trees per instance/target → `trees/gpusa_slice/` (repository root) |
| `trees/gpusa_slice/` (repository root) | `<name>_sc<T>_{tc,gpu2,best}.json` = `{"tree", "sliced"}` (tree iy on the sliced ixs, sliced = original labels), metrics in `index.json` |
| `bench/results/` | raw results (`compare.jsonl`, `slicing_compare{,_h100,_worker7273}.jsonl`, `throughput_rtx6000.txt`, `throughput_vs_W_{rtx6000,h100}.txt`, `gpu2_*.md`, logs) and summaries |

## How to run

```bash
module load cuda/12.8.0; cd treesa_gpu; make
source ../.venv/bin/activate
tests/run_checks.sh                 # kernel validation (~1 min)
python tests/test_python.py         # Python-level validation
python bench/throughput.py          # kernel throughput table
python bench/compare.py --suite omeco   # CPU baselines  (RAYON_NUM_THREADS set per run)
python bench/compare.py --suite gpu     # GPU configurations
python bench/compare.py --suite gpumode # GPU-runtime objective
python bench/summarize.py > bench/results/summary.md
python bench/gpu2_study.py > bench/results/gpu2_study.md  # measured H100 model v2
python bench/slicing_compare.py                            # slicing comparison (saves trees)
python bench/summarize_slicing.py > bench/results/slicing_summary.md
python bench/save_best.py                                  # *_best.json in booltn-gpu/trees/gpusa_slice
# on the H100 (shared GPU; every binary call wrapped in flock):
#   make ARCH=sm_90a OUT=treesa_gpu/bin/treesa_gpu_sm90
#   TREESA_GPU_BINARY=$PWD/treesa_gpu/bin/treesa_gpu_sm90 TREESA_GPU_LOCK=/tmp/booltn_gpu.lock python bench/...
# CLI example (sliced, H100 objective):
treesa_gpu/bin/treesa_gpu -i tests/data/grid24.txt --tree tests/data/grid24.tree --nchains 576 --niters 20 \
    --sc-target 36 --scheme pa --slicing --slice-every 2 --stage2-betas 3:0.0434:16 --stage2-mode gpu2 \
    --out result.json
# Python: tree, info = treesa_gpu.optimize(ixs, iy, nchains=576, niters=20, sc_target=36, scheme="pa",
#             mode="tc+gpu2", slicing=True, slice_every=2, stage2_betas=[3 + 13*i/299 for i in range(300)])
#         json.dump(treesa_gpu.runner_json(tree, info), open("tree.json", "w"))
```


