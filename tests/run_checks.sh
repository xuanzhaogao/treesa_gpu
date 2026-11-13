#!/bin/bash
# Validation matrix for the GPU TreeSA kernels (check mode).  Usage: tests/run_checks.sh [binary]
# For each configuration: (1) maintained labels == recomputation from topology, (2) GPU energy ==
# host energy (on the sliced network), (3) other kernel family bit-identical, (4) CPU Julia-port
# reference bit-identical.  (3)/(4) are skipped when slice moves are on (only the warp kernels
# implement them; the reference has no slice moves).
BIN=${1:-treesa_gpu/bin/treesa_gpu}
cd "$(dirname "$0")/.."
fail=0
run() {  # name, args...
  local name=$1; shift
  local out
  out=$($BIN --check --nchains 48 --niters 4 --betas 0.2:0.7:7 --sc-target 24 --ref-chains 4 --seed 7 "$@" 2>&1)
  if echo "$out" | grep -q "do not fit in shared memory"; then echo "$name: SKIP (chains do not fit in shared memory on this GPU)"; return; fi
  local st=$(echo "$out" | grep -E "^CHECK" || echo "CHECK ERROR")
  echo "$name: $st | $(echo "$out" | grep -oE 'chains differing: [0-9]+/[0-9]+|[0-9]+/[0-9]+ chains differ|label mismatch vs recomputation: [0-9]+|max \|dlog2cost\| = [0-9.e+-]+|[0-9]+/[0-9]+ slice moves accepted' | tr '\n' ' ')"
  echo "$st" | grep -q PASSED || { fail=1; echo "$out"; }
}
# A: register fast path, all kernels, all energy modes, random/greedy init, no slicing
for inst in sat40 sat100 reg3_100 reg3_250 lat10 lat14; do
  for kernel in seq warp seqsm warpsm warp/16 warpsm/8; do
    for mode in tc gpu h100; do
      for init in random greedy; do
        extra=()
        [ $init = greedy ] && extra+=(--tree tests/data/$inst.tree)
        [ $kernel = seqsm ] && extra+=(--block 1)
        k=${kernel%/*}; [ "$k" != "$kernel" ] && extra+=(--lpc ${kernel#*/})
        run "A $inst $kernel $mode $init" -i tests/data/$inst.txt --mode $mode --kernel $k "${extra[@]}"
      done
    done
  done
done
# B: wide (streaming) label path: forced on small instances, automatic for > 512 labels
for inst in sat40 sat100 reg3_250 grid24 grid45; do
  for kernel in seq warp; do
    for mode in tc h100; do
      run "B wide $inst $kernel $mode" -i tests/data/$inst.txt --tree tests/data/$inst.tree --mode $mode --kernel $kernel --wide
    done
  done
done
# C: fixed random slice sets (masking), compared with the Julia reference with sliced log2 size 0
for inst in sat100 reg3_250 grid24; do
  for kernel in seq warp warpsm; do
    for mode in tc h100; do
      run "C slices $inst $kernel $mode" -i tests/data/$inst.txt --tree tests/data/$inst.tree --mode $mode --kernel $kernel --random-slices 8
    done
  done
done
# D: slice moves on (labels and sliced energies must stay consistent)
for inst in sat100 grid24 grid28; do
  for kernel in warp warpsm; do
    for mode in tc h100; do
      run "D slicemoves $inst $kernel $mode" -i tests/data/$inst.txt --tree tests/data/$inst.tree --mode $mode --kernel $kernel --slicing --sc-target 20
    done
  done
done
[ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
