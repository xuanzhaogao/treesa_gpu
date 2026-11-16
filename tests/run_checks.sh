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
# E: lanes sharing one node visit (warp kernels, adaptive G = 32/nextpow2(ready nodes)).  Small instances with one
# word per lane (--coop-min-units 1; forced on with --coop-max 32 up to 512 labels, where it is off by default), and
# a 46x46 lattice (4140 labels, wide path) from the row-major sweep, a caterpillar tree with one ready node per
# step, so G = 32
for inst in sat100 reg3_250 grid24; do
  for mode in tc h100; do
    run "E coop $inst warp $mode" -i tests/data/$inst.txt --tree tests/data/$inst.tree --mode $mode --kernel warp --coop-min-units 1 --coop-max 32
  done
done
for kernel in warp warpsm; do
  for mode in tc gpu h100; do
    run "E coop sq46 sweep $kernel $mode" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode $mode --kernel $kernel
  done
done
run "E coop sq46 sweep slices" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode tc --kernel warp --random-slices 8
run "E coop sq46 sweep sums off" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode tc --kernel warp --no-sum
run "E coop grid45 sums forced" -i tests/data/grid45.txt --tree tests/data/grid45.tree --mode h100 --kernel warp --sums
run "E coop sq46 sweep slicemoves" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode h100 --kernel warp --slicing --sc-target 20
# F: one thread block per chain (--kernel block, wpc warps sharing a work list), against the sequential kernel
# and the CPU reference
for inst in sat100 reg3_250 lat14 grid24 grid45; do
  for wpc in 2 4 16; do
    for mode in tc h100; do
      run "F block $inst wpc$wpc $mode" -i tests/data/$inst.txt --tree tests/data/$inst.tree --mode $mode --kernel block --wpc $wpc
    done
  done
done
for mode in tc gpu h100; do
  run "F block sq46 sweep wpc8 $mode" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode $mode --kernel block --wpc 8
done
run "F block sat100 random init wpc4" -i tests/data/sat100.txt --mode tc --kernel block --wpc 4
run "F block reg3_250 coop forced wpc4" -i tests/data/reg3_250.txt --tree tests/data/reg3_250.tree --mode h100 --kernel block --wpc 4 --coop-max 32 --coop-min-units 1
run "F block grid24 slices wpc4" -i tests/data/grid24.txt --tree tests/data/grid24.tree --mode h100 --kernel block --wpc 4 --random-slices 8
run "F block grid28 slicemoves wpc4" -i tests/data/grid28.txt --tree tests/data/grid28.tree --mode h100 --kernel block --wpc 4 --slicing --sc-target 20
# G: parallel sweep order (--kernel par: independent sets of non-adjacent nodes per phase; options --par-band,
# --par-follow), against the CPU reference of the same order (no other kernel implements it, so no cross-kernel check)
for inst in sat100 reg3_250 lat14 grid24 grid45; do
  for wpc in 1 4 8; do
    for mode in tc h100; do
      run "G par $inst wpc$wpc $mode" -i tests/data/$inst.txt --tree tests/data/$inst.tree --mode $mode --kernel par --wpc $wpc
    done
  done
done
for mode in tc gpu h100; do
  run "G par sq46 sweep wpc8 $mode" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode $mode --kernel par --wpc 8
done
run "G par sat100 random init wpc2" -i tests/data/sat100.txt --mode tc --kernel par --wpc 2
run "G par reg3_250 coop forced wpc4" -i tests/data/reg3_250.txt --tree tests/data/reg3_250.tree --mode h100 --kernel par --wpc 4 --coop-max 32 --coop-min-units 1 --coop-full 8
run "G par grid45 coop-full 32 wpc2" -i tests/data/grid45.txt --tree tests/data/grid45.tree --mode tc --kernel par --wpc 2 --coop-full 32
run "G par grid24 wide wpc4" -i tests/data/grid24.txt --tree tests/data/grid24.tree --mode h100 --kernel par --wpc 4 --wide
run "G par grid24 slices wpc4" -i tests/data/grid24.txt --tree tests/data/grid24.tree --mode h100 --kernel par --wpc 4 --random-slices 8
run "G par sq46 sweep sums off" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode tc --kernel par --wpc 8 --no-sum
run "G par sat100 band4 wpc4" -i tests/data/sat100.txt --tree tests/data/sat100.tree --mode tc --kernel par --wpc 4 --par-band 4
run "G par grid45 band16 follow wpc8" -i tests/data/grid45.txt --tree tests/data/grid45.tree --mode h100 --kernel par --wpc 8 --par-band 16 --par-follow
run "G par sq46 sweep band8 wpc8" -i tests/data/sq46.txt --tree tests/data/sq46.tree --mode tc --kernel par --wpc 8 --par-band 8
run "G par reg3_250 follow wpc2" -i tests/data/reg3_250.txt --tree tests/data/reg3_250.tree --mode tc --kernel par --wpc 2 --par-follow
run "G par grid45 sums forced coop-chunks 1 wpc8" -i tests/data/grid45.txt --tree tests/data/grid45.tree --mode tc --kernel par --wpc 8 --sums --coop-chunks 1
run "G par grid28 slicemoves wpc4" -i tests/data/grid28.txt --tree tests/data/grid28.tree --mode h100 --kernel par --wpc 4 --slicing --sc-target 20
[ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
