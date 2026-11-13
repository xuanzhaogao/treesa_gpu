// treesa_gpu.cu -- massively parallel TreeSA contraction-order optimizer (with slicing) for
// tensor networks whose indices all have dimension 2 (boolean / k-SAT tensor networks).
//
// Algorithm = OMEinsumContractionOrders.jl/src/treesa.jl (TreeDecomp, rw_weight=0):
//   for β in βs, for _ in 1:niters:  one pre-order sweep over the tree; at every internal
//   node a random applicable rule (1-4) is proposed, accepted with Metropolis on
//   dE = (max(sc, sc+dsc) > sc_target ? sc_weight : 0)*dsc + dtc, and then the sweep
//   recurses into the node's (possibly new) children.
// Slicing (optional): every chain carries a slice set S; sliced labels have log2 size 0 (Julia
//   `Slicer`), i.e. all costs are popcounts of label sets masked by ~S.  After every
//   `slice_every` sweeps the warp proposes `slice_moves` changes of S (add a label of a largest
//   tensor / remove / swap), accepted with Metropolis on the total energy
//   E = |S| + log2(sum_v cost_v) + sc_weight*max(0, sc - sc_target)   (full-tree evaluation).
//
// Data layout (per chain, 32-bit words, NW = words per label set):
//   labels[nint][NW]  output-label bitset (unsliced) of internal node v at (v-m)*NW
//   children[nint]    left | right << 16   (node ids: leaves 0..m-1, internal m..2m-2)
//   S[NW]             slice set, at word offset s_off
// Leaf labels are identical for all chains: one shared read-only table leaf[m][NW].  The root
// is always node 2m-2 (rules never change the root node, as in Julia).
//
// Label width: W > 0 (1,2,4,8,12,16 words, up to 512 labels) uses a register fast path with the
// five label sets of a move held in registers; W == 0 is the wide path (any NW multiple of 4)
// streaming the label sets in 16-byte chunks.  Both accumulate the same popcount counters and
// evaluate the move with the same float code (common.h), so they are bit-identical.
//
// Kernels (all bit-identical thanks to the counter-based RNG, see common.h):
//   seq     one thread per chain, sequential pre-order sweep with an explicit stack.
//   seqsm   as seq, but the chains of a block are staged in shared memory for each launch.
//   warp    one warp per chain; lanes process up to 32 *independent* ready nodes at a time
//           (after a node is processed its two subtrees never interact again in that sweep).
//   warpsm  as warp, with the chain staged in shared memory.
// Slice moves are implemented in the warp kernels (one warp per chain) only.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <functional>
#include <omp.h>

#include "common.h"
#include "cpu_ref.h"

float h_lse_tab[64];
static int g_smem_optin = 232448;  // max dynamic smem per block (queried at startup)

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);  \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__host__ __device__ static inline int round4(int x) { return (x + 3) / 4 * 4; }

// JSON number (non-finite values, e.g. modeled times beyond 2^1024 s, become null)
static std::string jnum(double x) {
  if (!std::isfinite(x)) return "null";
  char b[64];
  snprintf(b, sizeof b, "%.6e", x);
  return b;
}

// =====================================================================================
// Device helpers
// =====================================================================================
template <int W>
__device__ __forceinline__ void ldlab(const uint32_t* p, uint32_t* o) {
  if constexpr (W % 4 == 0) {
#pragma unroll
    for (int i = 0; i < W / 4; i++) {
      uint4 t = reinterpret_cast<const uint4*>(p)[i];
      o[4 * i] = t.x; o[4 * i + 1] = t.y; o[4 * i + 2] = t.z; o[4 * i + 3] = t.w;
    }
  } else if constexpr (W % 2 == 0) {
#pragma unroll
    for (int i = 0; i < W / 2; i++) {
      uint2 t = reinterpret_cast<const uint2*>(p)[i];
      o[2 * i] = t.x; o[2 * i + 1] = t.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < W; i++) o[i] = p[i];
  }
}

template <int W>
__device__ __forceinline__ void stlab(uint32_t* p, const uint32_t* x) {
  if constexpr (W % 4 == 0) {
#pragma unroll
    for (int i = 0; i < W / 4; i++)
      reinterpret_cast<uint4*>(p)[i] = make_uint4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
  } else if constexpr (W % 2 == 0) {
#pragma unroll
    for (int i = 0; i < W / 2; i++) reinterpret_cast<uint2*>(p)[i] = make_uint2(x[2 * i], x[2 * i + 1]);
  } else {
#pragma unroll
    for (int i = 0; i < W; i++) p[i] = x[i];
  }
}

__device__ __forceinline__ uint4 ld4(const uint32_t* p) { return *reinterpret_cast<const uint4*>(p); }

// position of the n-th (0-based) set bit of x (n < popc(x))
__device__ __forceinline__ int nth_bit(uint32_t x, int n) {
  for (int k = 0; k < n; k++) x &= x - 1;
  return __ffs(x) - 1;
}

// n-th set bit (label id) of (A & ~S) over NW words; S may be null.  Warp-uniform.
__device__ __forceinline__ int nth_label(const uint32_t* A, const uint32_t* S, int NW, int n) {
  for (int w = 0; w < NW; w++) {
    uint32_t a = A[w] & (S ? ~S[w] : 0xFFFFFFFFu);
    int c = __popc(a);
    if (n < c) return w * 32 + nth_bit(a, n);
    n -= c;
  }
  return -1;
}

// One TreeSA proposal at internal node v of one chain (lab/ch/Sm point at that chain's arrays,
// which may live in global or shared memory; Sm == nullptr: no slicing).  Returns v's (possibly
// updated) children word.
//
// Rule r in 0..3 (Julia 1..4).  With P the pivot child (left for rules 0,1; right for 2,3),
// O the other child, M the grandchild under P that moves up and K the one that stays:
//   rule 0: (K,M),O -> (K,O),M     rule 1: (M,K),O -> (O,K),M
//   rule 2: O,(M,K) -> M,(O,K)     rule 3: O,(K,M) -> M,(K,O)
// i.e. every rule swaps M and O, keeping slots, and P gets labels X = (K|O)&(M|d).
// Stored labels are unsliced; costs use (set & ~S) (and (X & ~S) == X computed from masked sets).
template <int W, int MODE>
__device__ __forceinline__ uint32_t visit(uint32_t* lab, uint32_t* ch, const uint32_t* __restrict__ leaf,
                                          const uint32_t* Sm, const Params& P, int v, float beta, uint32_t sweep,
                                          uint32_t chain, int& nprop, int& nacc) {
  const int m = P.m;
  const int NW = (W > 0) ? W : P.nw;
  uint32_t chv = ch[v - m];
  int l = (int)(chv & 0xFFFFu), r = (int)(chv >> 16);
  bool li = l >= m, ri = r >= m;
  if (!li && !ri) return chv;
  U4 rnd = philox4x32(U4{(uint32_t)v, sweep, chain, 0u}, P.key0, P.key1);
  int rule = pick_rule(li, ri, rnd.x);
  int slotP = rule >> 1;
  int slotM = ((rule ^ (rule >> 1)) & 1) ^ 1;
  int Pn = slotP ? r : l, On = slotP ? l : r;
  uint32_t chp = ch[Pn - m];
  int Mn = slotM ? (int)(chp >> 16) : (int)(chp & 0xFFFFu);
  int Kn = slotM ? (int)(chp & 0xFFFFu) : (int)(chp >> 16);
  uint32_t* pP = lab + (size_t)(Pn - m) * NW;
  const uint32_t* pD = lab + (size_t)(v - m) * NW;
  const uint32_t* pO = On >= m ? lab + (size_t)(On - m) * NW : leaf + (size_t)On * NW;
  const uint32_t* pK = Kn >= m ? lab + (size_t)(Kn - m) * NW : leaf + (size_t)Kn * NW;
  const uint32_t* pM = Mn >= m ? lab + (size_t)(Mn - m) * NW : leaf + (size_t)Mn * NW;
  MoveCounts cnt = {};
  bool acc;
  if constexpr (W > 0) {
    // fast path: whole label sets in registers
    uint32_t D[W], Pl[W], O[W], K[W], M[W], X[W];
    ldlab<W>(pD, D);
    ldlab<W>(pP, Pl);
    ldlab<W>(pO, O);
    ldlab<W>(pK, K);
    ldlab<W>(pM, M);
#pragma unroll
    for (int i = 0; i < W; i++) {
      X[i] = (K[i] | O[i]) & (M[i] | D[i]);
      uint32_t nm = Sm ? ~Sm[i] : 0xFFFFFFFFu;
      acc_counts<MODE>(cnt, K[i] & nm, M[i] & nm, Pl[i] & nm, O[i] & nm, D[i] & nm, X[i] & nm);
    }
    float dE = energy_from_counts<MODE>(cnt, P);
    nprop++;
    acc = metropolis(rnd.y, beta, dE);
    if (acc) stlab<W>(pP, X);
  } else {
    // wide path: stream the five label sets in 16-byte chunks
    for (int w = 0; w < NW; w += 4) {
      uint4 d = ld4(pD + w), p = ld4(pP + w), o = ld4(pO + w), k = ld4(pK + w), mm = ld4(pM + w);
      uint4 s = Sm ? ld4(Sm + w) : make_uint4(0u, 0u, 0u, 0u);
#define TSA_ACC(c)                                                                          \
  {                                                                                         \
    uint32_t nm = ~s.c, x = (k.c | o.c) & (mm.c | d.c);                                     \
    acc_counts<MODE>(cnt, k.c & nm, mm.c & nm, p.c & nm, o.c & nm, d.c & nm, x & nm);       \
  }
      TSA_ACC(x) TSA_ACC(y) TSA_ACC(z) TSA_ACC(w)
#undef TSA_ACC
    }
    float dE = energy_from_counts<MODE>(cnt, P);
    nprop++;
    acc = metropolis(rnd.y, beta, dE);
    if (acc) {
      for (int w = 0; w < NW; w += 4) {
        uint4 d = ld4(pD + w), o = ld4(pO + w), k = ld4(pK + w), mm = ld4(pM + w);
        *reinterpret_cast<uint4*>(pP + w) = make_uint4((k.x | o.x) & (mm.x | d.x), (k.y | o.y) & (mm.y | d.y),
                                                       (k.z | o.z) & (mm.z | d.z), (k.w | o.w) & (mm.w | d.w));
      }
    }
  }
  if (acc) {
    nacc++;
    uint32_t nchp = slotM ? ((chp & 0xFFFFu) | ((uint32_t)On << 16)) : ((chp & 0xFFFF0000u) | (uint32_t)On);
    uint32_t nchv = slotP ? ((chv & 0xFFFF0000u) | (uint32_t)Mn) : ((chv & 0xFFFFu) | ((uint32_t)Mn << 16));
    ch[Pn - m] = nchp;
    ch[v - m] = nchv;
    chv = nchv;
  }
  return chv;
}

// -------------------------------------------------------------------------------------
// Whole-chain evaluation (warp-cooperative: all 32 lanes call it).  Costs use label sets
// masked by ~S (S may be null).  Returns lcost = log2 sum_v 2^cost(v) (per slice), sc (max
// tensor size incl. leaves), sc_int / ncand (max internal size and how many internal nodes
// have it), ns = |S|.
// -------------------------------------------------------------------------------------
struct EvalOut {
  float lcost;
  int sc, sc_int, ncand, ns;
};

template <int W, int MODE>
__device__ EvalOut eval_chain(const Params& P, const uint32_t* lab, const uint32_t* ch, const uint32_t* S,
                              const uint32_t* leaf, int lane) {
  const int NW = (W > 0) ? W : P.nw, m = P.m;
  const unsigned FULL = 0xFFFFFFFFu;
  float acc = -INFINITY;
  int scl = 0, cntl = 0;
  for (int i = lane; i < P.nint; i += 32) {
    uint32_t cw = ch[i];
    int l = (int)(cw & 0xFFFFu), r = (int)(cw >> 16);
    const uint32_t* x = l >= m ? lab + (size_t)(l - m) * NW : leaf + (size_t)l * NW;
    const uint32_t* y = r >= m ? lab + (size_t)(r - m) * NW : leaf + (size_t)r * NW;
    const uint32_t* z = lab + (size_t)i * NW;
    int B = 0, M = 0, N = 0, K = 0, Z = 0;
    for (int w = 0; w < NW; w++) {
      uint32_t nm = S ? ~S[w] : 0xFFFFFFFFu;
      uint32_t zz = z[w] & nm;
      count3(x[w] & nm, y[w] & nm, zz, B, M, N, K);
      Z += popc32(zz);
    }
    acc = lse_f(acc, node_cost<MODE>(B, M, N, K, Z, P));
    if (Z > scl) { scl = Z; cntl = 1; }
    else if (Z == scl) cntl++;
  }
  int scmax = scl;
  for (int o = 16; o > 0; o >>= 1) {
    acc = lse_f(acc, __shfl_xor_sync(FULL, acc, o));
    scmax = max(scmax, __shfl_xor_sync(FULL, scmax, o));
  }
  int cnt = (scl == scmax) ? cntl : 0, ns = 0;
  if (S)
    for (int w = lane; w < NW; w += 32) ns += __popc(S[w]);
  for (int o = 16; o > 0; o >>= 1) {
    cnt += __shfl_xor_sync(FULL, cnt, o);
    ns += __shfl_xor_sync(FULL, ns, o);
  }
  int sc = max(scmax, P.leaf_sc);
  if (S && P.leaf_sc > scmax) {  // a leaf may be the largest tensor: masked leaf sizes
    int ll = 0;
    for (int i = lane; i < m; i += 32) {
      int Z = 0;
      for (int w = 0; w < NW; w++) Z += __popc(leaf[(size_t)i * NW + w] & ~S[w]);
      ll = max(ll, Z);
    }
    for (int o = 16; o > 0; o >>= 1) ll = max(ll, __shfl_xor_sync(FULL, ll, o));
    sc = max(scmax, ll);
  }
  return EvalOut{acc, sc, scmax, cnt, ns};
}

__device__ __forceinline__ float slice_energy(const EvalOut& e, const Params& P) {
  float pen = (float)e.sc > P.sc_target ? P.sc_weight * ((float)e.sc - P.sc_target) : 0.0f;
  return (float)e.ns + e.lcost + pen;
}

// Slice moves of one chain (warp-cooperative).  Proposal: add a random label of a randomly
// chosen largest internal tensor (prob 1/2), remove a random sliced label (1/4) or swap
// (remove + add, 1/4); add only when |S| = 0, no add when |S| = max_slices.  Metropolis on
// the total energy E = |S| + log2(sum cost) + sc_weight*max(0, sc - sc_target).
template <int W, int MODE>
__device__ void slice_step(const Params& P, const uint32_t* lab, const uint32_t* ch, uint32_t* S, const uint32_t* leaf,
                           float beta, uint32_t sweep, uint32_t chain, int lane, int& nsp, int& nsa) {
  const int NW = (W > 0) ? W : P.nw;
  const unsigned FULL = 0xFFFFFFFFu;
  EvalOut cur = eval_chain<W, MODE>(P, lab, ch, S, leaf, lane);
  float Ecur = slice_energy(cur, P);
  for (int j = 0; j < P.slice_moves; j++) {
    U4 r1 = philox4x32(U4{0x10000u + 2u * j, sweep, chain, 0u}, P.key0, P.key1);
    U4 r2 = philox4x32(U4{0x10001u + 2u * j, sweep, chain, 0u}, P.key0, P.key1);
    int type;  // 0 add, 1 remove, 2 swap
    if (cur.ns == 0) type = 0;
    else if (cur.ns >= P.max_slices) type = (r1.x >> 31) ? 1 : 2;
    else type = r1.x < 0x80000000u ? 0 : (r1.x < 0xC0000000u ? 1 : 2);
    int ladd = -1, lrem = -1;
    if (type != 1 && cur.ncand > 0 && cur.sc_int > 0) {
      // the (r1.y % ncand)-th internal node of maximal masked size, in node-id order
      int rr = (int)(r1.y % (uint32_t)cur.ncand), node = -1;
      for (int base = 0; base < P.nint && node < 0; base += 32) {
        int i = base + lane, Z = -1;
        if (i < P.nint) {
          Z = 0;
          const uint32_t* z = lab + (size_t)i * NW;
          for (int w = 0; w < NW; w++) Z += __popc(z[w] & ~S[w]);
        }
        unsigned b = __ballot_sync(FULL, Z == cur.sc_int);
        int c = __popc(b);
        if (rr < c) node = base + nth_bit(b, rr);
        else rr -= c;
      }
      if (node >= 0) ladd = nth_label(lab + (size_t)node * NW, S, NW, (int)(r1.z % (uint32_t)cur.sc_int));
    }
    if (type != 0) lrem = nth_label(S, nullptr, NW, (int)(r1.w % (uint32_t)cur.ns));
    if (type == 2 && ladd < 0) type = 1;
    if (type == 0 && ladd < 0) continue;
    __syncwarp();
    if (lane == 0) {
      if (type != 1) S[ladd >> 5] |= 1u << (ladd & 31);
      if (type != 0) S[lrem >> 5] &= ~(1u << (lrem & 31));
    }
    __syncwarp();
    EvalOut nxt = eval_chain<W, MODE>(P, lab, ch, S, leaf, lane);
    float Enew = slice_energy(nxt, P);
    nsp++;
    if (metropolis(r2.x, beta, Enew - Ecur)) {
      nsa++;
      cur = nxt;
      Ecur = Enew;
    } else {
      __syncwarp();
      if (lane == 0) {
        if (type != 1) S[ladd >> 5] &= ~(1u << (ladd & 31));
        if (type != 0) S[lrem >> 5] |= 1u << (lrem & 31);
      }
      __syncwarp();
    }
  }
}

// -------------------------------------------------------------------------------------
// Kernel "seq"/"seqsm": one thread per chain, literal pre-order recursion with a stack of
// pending right children (stack in global memory, interleaved across chains).  Supports a
// fixed slice set (masking) but no slice moves.
// -------------------------------------------------------------------------------------
template <int W, int MODE, bool SMEM>
__global__ void __launch_bounds__(256) k_seq(Params P, uint32_t* __restrict__ rec, const uint32_t* __restrict__ leaf,
                                             int nchains, float beta, uint32_t sweep0, int nsweeps,
                                             uint16_t* __restrict__ stk, unsigned long long* stats) {
  extern __shared__ uint4 smem4[];
  const int NW = (W > 0) ? W : P.nw;
  const int c0 = blockIdx.x * blockDim.x;
  const int c = c0 + threadIdx.x;
  const int CW = P.chain_words;
  uint32_t* base;
  const uint32_t* lf = leaf;
  int leafw = round4(P.m * NW);
  if constexpr (SMEM) {
    // stage leaf table + this block's chains in shared memory
    uint32_t* sm = reinterpret_cast<uint32_t*>(smem4);
    for (int i = threadIdx.x; i < P.m * NW; i += blockDim.x) sm[i] = leaf[i];
    int nc = min((int)blockDim.x, nchains - c0);
    uint4* dst = reinterpret_cast<uint4*>(sm + leafw);
    const uint4* src = reinterpret_cast<const uint4*>(rec + (size_t)c0 * CW);
    for (int i = threadIdx.x; i < nc * CW / 4; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    lf = sm;
    base = sm + leafw + (size_t)threadIdx.x * CW;
  } else {
    base = rec + (size_t)c * CW;
  }
  int np = 0, na = 0;
  if (c < nchains) {
    uint32_t* lab = base;
    uint32_t* ch = base + (size_t)P.nint * NW;
    const uint32_t* Sm = P.slicing ? base + P.s_off : nullptr;
    const int m = P.m, root = 2 * m - 2;
    for (int s = 0; s < nsweeps; s++) {
      int v = root, sp = 0;
      while (true) {
        uint32_t cw = visit<W, MODE>(lab, ch, lf, Sm, P, v, beta, sweep0 + s, (uint32_t)c, np, na);
        int l = (int)(cw & 0xFFFFu), r = (int)(cw >> 16);
        bool li = l >= m, ri = r >= m;
        if (li) {
          if (ri) stk[(size_t)(sp++) * nchains + c] = (uint16_t)r;
          v = l;
        } else if (ri) {
          v = r;
        } else {
          if (sp == 0) break;
          v = stk[(size_t)(--sp) * nchains + c];
        }
      }
    }
  }
  if constexpr (SMEM) {
    __syncthreads();
    uint32_t* sm = reinterpret_cast<uint32_t*>(smem4);
    int nc = min((int)blockDim.x, nchains - c0);
    const uint4* src = reinterpret_cast<const uint4*>(sm + leafw);
    uint4* dst = reinterpret_cast<uint4*>(rec + (size_t)c0 * CW);
    for (int i = threadIdx.x; i < nc * CW / 4; i += blockDim.x) dst[i] = src[i];
  }
  if (c < nchains) {
    atomicAdd(&stats[0], (unsigned long long)np);
    atomicAdd(&stats[1], (unsigned long long)na);
  }
}

// -------------------------------------------------------------------------------------
// Kernel "warp"/"warpsm": a group of LPC lanes (LPC = 32: one warp; 16 or 8: 2 or 4 chains per
// warp) per chain.  A per-chain work list (shared memory) holds internal nodes whose parent has
// already been processed in the current sweep.  Each step pops up to LPC of them, the lanes
// process them concurrently (their neighbourhoods are disjoint: each touches only its own
// children/grandchildren, inside its own subtree), and the internal children are pushed back.
// Equivalent to the sequential pre-order sweep because the RNG is keyed by node id, not by
// visit order.  With LPC == 32 and P.slice_moves > 0, slice moves follow every `slice_every`
// (global) sweeps.
// -------------------------------------------------------------------------------------
template <int W, int MODE, bool SMEM, int LPC>
__global__ void __launch_bounds__(256) k_warp(Params P, uint32_t* __restrict__ rec, const uint32_t* __restrict__ leaf,
                                              int nchains, float beta, uint32_t sweep0, int nsweeps,
                                              unsigned long long* stats) {
  static_assert(LPC == 32 || LPC == 16 || LPC == 8, "LPC");
  constexpr int CPW = 32 / LPC;  // chains per warp
  extern __shared__ uint4 smem4[];
  uint32_t* sm = reinterpret_cast<uint32_t*>(smem4);
  const int NW = (W > 0) ? W : P.nw;
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const int g = lane / LPC, gl = lane % LPC;     // group in warp, lane in group
  const int slot = wib * CPW + g, nslot = wpb * CPW;
  const int c = blockIdx.x * nslot + slot;
  const unsigned gmask = (LPC == 32) ? 0xFFFFFFFFu : (((1u << LPC) - 1u) << (g * LPC));
  const int CW = P.chain_words;
  const int m = P.m, root = 2 * m - 2;
  const int wlw = ((P.nint + 7) / 8) * 4;  // work-list words per chain (uint16 entries, 16B aligned)
  int leafw = 0;
  const uint32_t* lf = leaf;
  if constexpr (SMEM) {
    leafw = round4(m * NW);
    for (int i = threadIdx.x; i < m * NW; i += blockDim.x) sm[i] = leaf[i];
    __syncthreads();
    lf = sm;
  }
  if (c >= nchains) return;  // whole groups exit; no block-level barriers below this point
  uint16_t* wl = reinterpret_cast<uint16_t*>(sm + leafw + (size_t)slot * wlw);
  uint32_t* base;
  if constexpr (SMEM) {
    base = sm + leafw + (size_t)nslot * wlw + (size_t)slot * CW;
    const uint4* src = reinterpret_cast<const uint4*>(rec + (size_t)c * CW);
    uint4* dst = reinterpret_cast<uint4*>(base);
    for (int i = gl; i < CW / 4; i += LPC) dst[i] = src[i];
    __syncwarp(gmask);
  } else {
    base = rec + (size_t)c * CW;
  }
  uint32_t* lab = base;
  uint32_t* ch = base + (size_t)P.nint * NW;
  uint32_t* S = base + P.s_off;
  const uint32_t* Sm = P.slicing ? S : nullptr;
  int np = 0, na = 0, nst = 0, nv = 0, nsp = 0, nsa = 0;  // proposals, accepts, steps, visits, slice props/accepts
  const unsigned lt = (1u << gl) - 1u;
  const int shift = g * LPC;
  for (int s = 0; s < nsweeps; s++) {
    if (gl == 0) wl[0] = (uint16_t)root;
    int sp = 1;
    __syncwarp(gmask);
    while (sp > 0) {
      int k = min(LPC, sp), b0 = sp - k;
      nst++;
      nv += gl < k;
      int v = gl < k ? (int)wl[b0 + gl] : 0;
      __syncwarp(gmask);
      uint32_t cw = 0;
      if (gl < k) cw = visit<W, MODE>(lab, ch, lf, Sm, P, v, beta, sweep0 + s, (uint32_t)c, np, na);
      bool c1 = gl < k && (int)(cw & 0xFFFFu) >= m;
      bool c2 = gl < k && (int)(cw >> 16) >= m;
      unsigned b1 = __ballot_sync(gmask, c1) >> shift, b2 = __ballot_sync(gmask, c2) >> shift;
      if constexpr (LPC < 32) { b1 &= (1u << LPC) - 1u; b2 &= (1u << LPC) - 1u; }
      int pos = b0 + __popc(b1 & lt) + __popc(b2 & lt);
      if (c2) wl[pos++] = (uint16_t)(cw >> 16);
      if (c1) wl[pos] = (uint16_t)(cw & 0xFFFFu);
      sp = b0 + __popc(b1) + __popc(b2);
      __syncwarp(gmask);
    }
    if constexpr (LPC == 32) {
      if (P.slice_moves > 0 && (sweep0 + s + 1) % (uint32_t)P.slice_every == 0)
        slice_step<W, MODE>(P, lab, ch, S, lf, beta, sweep0 + s, (uint32_t)c, lane, nsp, nsa);
    }
  }
  if constexpr (SMEM) {
    __syncwarp(gmask);
    const uint4* src = reinterpret_cast<const uint4*>(base);
    uint4* dst = reinterpret_cast<uint4*>(rec + (size_t)c * CW);
    for (int i = gl; i < CW / 4; i += LPC) dst[i] = src[i];
  }
  // group-reduce counters
  for (int o = LPC / 2; o > 0; o >>= 1) {
    np += __shfl_xor_sync(gmask, np, o);
    na += __shfl_xor_sync(gmask, na, o);
    nv += __shfl_xor_sync(gmask, nv, o);
  }
  if (gl == 0) {
    atomicAdd(&stats[0], (unsigned long long)np);
    atomicAdd(&stats[1], (unsigned long long)na);
    atomicAdd(&stats[2], (unsigned long long)nst);
    atomicAdd(&stats[3], (unsigned long long)nv);
    atomicAdd(&stats[4], (unsigned long long)nsp);
    atomicAdd(&stats[5], (unsigned long long)nsa);
  }
}

// Per-chain global energy (one warp per chain): per-slice lcost, sc, |S|.
template <int W, int MODE>
__global__ void k_energy(Params P, const uint32_t* __restrict__ rec, const uint32_t* __restrict__ leaf, int nchains,
                         float* lcost, int* scout, int* nsout) {
  int c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (c >= nchains) return;
  const uint32_t* lab = rec + (size_t)c * P.chain_words;
  const uint32_t* ch = lab + (size_t)P.nint * ((W > 0) ? W : P.nw);
  const uint32_t* S = P.slicing ? lab + P.s_off : nullptr;
  EvalOut e = eval_chain<W, MODE>(P, lab, ch, S, leaf, lane);
  if (lane == 0) {
    lcost[c] = e.lcost;
    scout[c] = e.sc;
    nsout[c] = e.ns;
  }
}

// Population resampling: dst chain c <- src chain anc[c].
__global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ anc,
                       int n4) {
  const uint4* s = src + (size_t)anc[blockIdx.x] * n4;
  uint4* d = dst + (size_t)blockIdx.x * n4;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) d[i] = s[i];
}

// =====================================================================================
// Host side
// =====================================================================================
struct Inst {
  int m = 0, nlab = 0;
  std::vector<std::vector<int>> ixs;
  std::vector<int> iy;
  std::vector<int> total;   // occurrences of each label over all input tensors
  std::vector<char> in_iy;  // label is an open (output) label
};

static Inst read_instance(const char* path) {
  Inst I;
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  if (fscanf(f, "%d %d", &I.m, &I.nlab) != 2) { fprintf(stderr, "bad header\n"); exit(1); }
  I.ixs.resize(I.m);
  for (int i = 0; i < I.m; i++) {
    int k;
    if (fscanf(f, "%d", &k) != 1) { fprintf(stderr, "bad tensor line\n"); exit(1); }
    for (int j = 0; j < k; j++) {
      int l;
      if (fscanf(f, "%d", &l) != 1) exit(1);
      I.ixs[i].push_back(l);
    }
  }
  int ny;
  if (fscanf(f, "%d", &ny) != 1) ny = 0;
  for (int j = 0; j < ny; j++) {
    int l;
    if (fscanf(f, "%d", &l) != 1) exit(1);
    I.iy.push_back(l);
  }
  fclose(f);
  I.total.assign(I.nlab, 0);
  I.in_iy.assign(I.nlab, 0);
  for (auto& ix : I.ixs) {
    std::sort(ix.begin(), ix.end());
    if (std::adjacent_find(ix.begin(), ix.end()) != ix.end()) {
      fprintf(stderr, "repeated label inside one tensor is not supported\n");
      exit(1);
    }
    for (int l : ix) {
      if (l < 0 || l >= I.nlab) { fprintf(stderr, "label out of range\n"); exit(1); }
      I.total[l]++;
    }
  }
  for (int l : I.iy) I.in_iy[l] = 1;
  return I;
}

// Read an initial tree: m-1 lines "left right" for internal nodes m..2m-2 (root = 2m-2).
static std::vector<uint32_t> read_tree(const char* path, int m) {
  std::vector<uint32_t> ch(m - 1);
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  for (int i = 0; i < m - 1; i++) {
    int l, r;
    if (fscanf(f, "%d %d", &l, &r) != 2) { fprintf(stderr, "bad tree file\n"); exit(1); }
    ch[i] = (uint32_t)l | ((uint32_t)r << 16);
  }
  fclose(f);
  return ch;
}

// Julia `random_exprtree` (TreeDecomp): recursive random bipartition.  Internal nodes are
// numbered in post-order starting at m, so the root is 2m-2.
static std::vector<uint32_t> random_tree(int m, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<uint32_t> ch(m - 1);
  int next = m;
  std::function<int(const std::vector<int>&)> build = [&](const std::vector<int>& xs) -> int {
    int n = (int)xs.size();
    if (n == 1) return xs[0];
    std::vector<char> mask(n);
    int cnt = 0;
    for (int i = 0; i < n; i++) { mask[i] = (char)(rng() & 1); cnt += mask[i]; }
    if (cnt == 0 || cnt == n) { int i = (int)(rng() % n); mask[i] = !mask[i]; }
    std::vector<int> a, b;
    for (int i = 0; i < n; i++) (mask[i] ? a : b).push_back(xs[i]);
    int l = build(a), r = build(b);
    int v = next++;
    ch[v - m] = (uint32_t)l | ((uint32_t)r << 16);
    return v;
  };
  std::vector<int> all(m);
  std::iota(all.begin(), all.end(), 0);
  build(all);
  return ch;
}

// Pre-order list of internal nodes; returns false if the structure is not a valid binary
// tree over all 2m-1 nodes rooted at 2m-2 (every node reached exactly once).
static bool preorder(int m, const uint32_t* ch, std::vector<int>& order) {
  order.clear();
  std::vector<char> seen(2 * m - 1, 0);
  std::vector<int> st{2 * m - 2};
  int nseen = 0;
  while (!st.empty()) {
    int v = st.back();
    st.pop_back();
    if (v < 0 || v >= 2 * m - 1 || seen[v]) return false;
    seen[v] = 1;
    nseen++;
    if (v >= m) {
      order.push_back(v);
      st.push_back((int)(ch[v - m] >> 16));
      st.push_back((int)(ch[v - m] & 0xFFFFu));
    }
  }
  return nseen == 2 * m - 1;
}

// Internal-node output labels derived from topology: label l is an output of node v iff it
// occurs in v's subtree and (also occurs outside it, or is an open label).  Writes lab[nint*NW].
static bool labels_from_topology(const Inst& I, int NW, const uint32_t* ch, uint32_t* lab) {
  const int m = I.m, L = I.nlab;
  std::vector<int> order;
  if (!preorder(m, ch, order)) return false;
  std::vector<uint16_t> cnt((size_t)(m - 1) * L, 0);
  for (int k = (int)order.size() - 1; k >= 0; k--) {
    int v = order[k];
    uint16_t* cv = &cnt[(size_t)(v - m) * L];
    for (int side = 0; side < 2; side++) {
      int u = side ? (int)(ch[v - m] >> 16) : (int)(ch[v - m] & 0xFFFFu);
      if (u < m) {
        for (int l : I.ixs[u]) cv[l]++;
      } else {
        const uint16_t* cu = &cnt[(size_t)(u - m) * L];
        for (int l = 0; l < L; l++) cv[l] += cu[l];
      }
    }
    uint32_t* z = lab + (size_t)(v - m) * NW;
    for (int w = 0; w < NW; w++) z[w] = 0;
    for (int l = 0; l < L; l++)
      if (cv[l] > 0 && (cv[l] < I.total[l] || I.in_iy[l])) z[l >> 5] |= 1u << (l & 31);
  }
  return true;
}

struct HostEnergy {
  double tc = 0;     // log2 sum_v 2^tc(v)            (per slice)
  int sc = 0;        // max tensor size (log2), leaves included (per slice)
  double time = 0;   // sum_v t(B,M,N,K)  GPU model v1, seconds (per slice)
  double time2 = 0;  // sum_v t(B,M,N,K)  measured H100 model, seconds (per slice)
  int ns = 0;        // |S|
};

static double gpu_time_d(int B, int M, int N, int K, const Params& P) {
  double ops = std::exp2((double)(1 + B + std::max(M, P.m0) + std::max(N, P.n0) + std::max(K, P.k0)));
  double bits = std::exp2((double)(B + M + K)) + std::exp2((double)(B + N + K)) + std::exp2((double)(B + M + N));
  return std::max(ops / std::exp2((double)P.lpeak), bits / std::exp2((double)P.lbw)) + std::exp2((double)P.llaunch);
}

// Double-precision mirror of tree_utils.h100_time (constants recovered from the log2 params).
static double h100_time_d(int B, int M, int N, int K, const Params& P) {
  auto e = [](float l) { return std::exp2((double)l); };
  if (B + M + N <= P.v2_small_rank) return e(P.v2_lsmall_cell) * std::exp2((double)(B + M + N + K));
  double a = std::exp2((double)(B + M + K)), b = std::exp2((double)(B + N + K)), c = std::exp2((double)(B + M + N));
  double moved = a + b + c;
  double perm = e(P.v2_lperm2) * (a + b);
  double best = e(P.v2_lcc_bit) * moved + e(P.v2_lcc_cell) * std::exp2((double)(B + M + N + K));
  if (K >= 7 && std::max(M, N) >= 5) {
    double tc = e(P.v2_ltc_bit) * moved +
                e(P.v2_ltc_pad) * std::exp2((double)(B + std::max(M, 6) + std::max(N, 6) + std::max(K, 8))) +
                e(P.v2_ltc_out) * c;
    best = std::min(best, tc);
  }
  return e(P.v2_l3launch) + perm + best;
}

// Complexities of one chain on the sliced network (label sets masked by ~S; S may be null).
static HostEnergy host_energy(const Inst& I, int NW, const Params& P, const uint32_t* lab, const uint32_t* ch,
                              const std::vector<uint32_t>& leaf, const uint32_t* S) {
  HostEnergy E;
  std::vector<uint32_t> nm(NW, 0xFFFFFFFFu);
  if (S)
    for (int w = 0; w < NW; w++) { nm[w] = ~S[w]; E.ns += popc32(S[w]); }
  for (int i = 0; i < I.m; i++) {
    int z = 0;
    for (int w = 0; w < NW; w++) z += popc32(leaf[(size_t)i * NW + w] & nm[w]);
    E.sc = std::max(E.sc, z);
  }
  double mx = -1e300;
  std::vector<double> tcs;
  for (int i = 0; i < I.m - 1; i++) {
    int l = (int)(ch[i] & 0xFFFFu), r = (int)(ch[i] >> 16);
    const uint32_t* x = l >= I.m ? lab + (size_t)(l - I.m) * NW : &leaf[(size_t)l * NW];
    const uint32_t* y = r >= I.m ? lab + (size_t)(r - I.m) * NW : &leaf[(size_t)r * NW];
    const uint32_t* z = lab + (size_t)i * NW;
    int B = 0, M = 0, N = 0, K = 0, sz = 0;
    for (int w = 0; w < NW; w++) {
      uint32_t a = x[w] & nm[w], b = y[w] & nm[w], c = z[w] & nm[w];
      B += popc32(a & b & c);
      M += popc32(a & ~b & c);
      N += popc32(~a & b & c);
      K += popc32(a & b & ~c);
      sz += popc32(c);
    }
    double t = B + M + N + K;
    tcs.push_back(t);
    mx = std::max(mx, t);
    E.sc = std::max(E.sc, sz);
    E.time += gpu_time_d(B, M, N, K, P);
    E.time2 += h100_time_d(B, M, N, K, P);
  }
  double s = 0;
  for (double t : tcs) s += std::exp2(t - mx);
  E.tc = mx + std::log2(s);
  return E;
}

// Julia TreeSASlicer rule as a final repair: while sc > sc_target, slice the label that occurs
// most often among the largest (masked) tensors (internal nodes and leaves; ties -> lowest id).
static int greedy_repair(const Inst& I, int NW, const Params& P, const uint32_t* lab, const std::vector<uint32_t>& leaf,
                         uint32_t* S, double sc_target) {
  int added = 0;
  while (true) {
    int scmax = 0;
    std::vector<int> sizes(2 * I.m - 1);
    auto set_of = [&](int v) -> const uint32_t* { return v < I.m ? &leaf[(size_t)v * NW] : lab + (size_t)(v - I.m) * NW; };
    for (int v = 0; v < 2 * I.m - 1; v++) {
      const uint32_t* a = set_of(v);
      int z = 0;
      for (int w = 0; w < NW; w++) z += popc32(a[w] & ~S[w]);
      sizes[v] = z;
      scmax = std::max(scmax, z);
    }
    if (scmax <= sc_target || scmax == 0) return added;
    std::vector<int> freq(I.nlab, 0);
    for (int v = 0; v < 2 * I.m - 1; v++)
      if (sizes[v] == scmax) {
        const uint32_t* a = set_of(v);
        for (int l = 0; l < I.nlab; l++)
          if ((a[l >> 5] & ~S[l >> 5]) >> (l & 31) & 1u) freq[l]++;
      }
    int best = (int)(std::max_element(freq.begin(), freq.end()) - freq.begin());
    S[best >> 5] |= 1u << (best & 31);
    added++;
  }
}

// =====================================================================================
// Options / driver
// =====================================================================================
struct Opts {
  std::string inst, tree, out, kernel = "auto", scheme = "indep", betas = "0.01:0.05:15";
  std::string betas2;  // optional stage 2: continue the population in mode2 on this schedule
  std::string init_slices;  // comma-separated label ids sliced initially in every chain
  int niters2 = -1;
  int nchains = 1024, niters = 50, block = 0, wpb = 0, lpc = 32, eval_every = 1, ref_chains = 8, mode = 0;
  bool check = false, quiet = false, cross = true, wide = false, slicing = false;
  int slice_moves = 1, slice_every = 1, max_slices = 0, random_slices = 0;
  double sc_target = 20.0, sc_weight = 1.0, trunc_frac = 0.25;
  uint64_t seed = 1;
  double gm[6] = {11.9e15, 1.6e13, 6, 6, 8, 4e-6};  // peak ops/s, bw bits/s, m0, n0, k0, launch s
  // H100 model (trees.H100_MODEL): launch, perm_bw_bits, cc_bit, cc_cell, tc_bit, tc_pad_cell,
  // tc_out_bit, small_rank, small_cell
  double h100m[9] = {4e-6, 8.0e12, 6.25e-14, 1.97e-14, 2.78e-14, 7.6e-17, 2.14e-13, 14, 1e-12};
  int mode2 = 1;  // energy mode of stage 2 (1 = gpu, 2 = h100)
  int max_levels = -1;
};

static std::vector<float> parse_betas(const std::string& s) {
  std::vector<float> b;
  if (s.rfind("file:", 0) == 0) {
    FILE* f = fopen(s.c_str() + 5, "r");
    if (!f) { fprintf(stderr, "cannot open betas file\n"); exit(1); }
    double x;
    while (fscanf(f, "%lf", &x) == 1) b.push_back((float)x);
    fclose(f);
    return b;
  }
  double a, st, e;
  if (sscanf(s.c_str(), "%lf:%lf:%lf", &a, &st, &e) == 3) {
    // Julia range a:st:e (inclusive, computed as a + i*st)
    long n = (long)std::floor((e - a) / st + 1e-9) + 1;
    for (long i = 0; i < n; i++) b.push_back((float)(a + i * st));
    return b;
  }
  // comma separated list
  const char* p = s.c_str();
  while (*p) {
    char* q;
    double x = strtod(p, &q);
    if (q == p) break;
    b.push_back((float)x);
    p = q;
    while (*p == ',' || *p == ' ') p++;
  }
  return b;
}

static void usage() {
  fprintf(stderr,
          "usage: treesa_gpu -i instance.txt [--tree init_tree.txt] [--nchains N] [--niters K]\n"
          "  [--betas a:step:b | --betas file:PATH | --betas b1,b2,..] [--mode tc|gpu|h100]\n"
          "  [--gm peak,bw,m0,n0,k0,launch] [--h100-model launch,perm_bw,cc_bit,cc_cell,tc_bit,tc_pad,tc_out,small_rank,small_cell]\n"
          "  [--stage2-betas SPEC [--stage2-niters K] [--stage2-mode gpu|h100]]  (continue the population)\n"
          "  [--sc-target x] [--sc-weight y] [--scheme indep|pa|trunc] [--trunc-frac f]\n"
          "  [--slicing [--slice-moves k] [--slice-every s] [--max-slices n] [--init-slices l1,l2,..]]\n"
          "  [--kernel auto|seq|seqsm|warp|warpsm] [--lpc 32|16|8] [--wide] [--block B] [--wpb W] [--seed s]\n"
          "  [--eval-every k] [--out result.json] [--check [--ref-chains R] [--random-slices k]] [--quiet]\n");
  exit(1);
}

static Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
    if (a == "-i") o.inst = nx();
    else if (a == "--tree") o.tree = nx();
    else if (a == "--out") o.out = nx();
    else if (a == "--nchains") o.nchains = std::stoi(nx());
    else if (a == "--niters") o.niters = std::stoi(nx());
    else if (a == "--betas") o.betas = nx();
    else if (a == "--stage2-betas") o.betas2 = nx();
    else if (a == "--stage2-niters") o.niters2 = std::stoi(nx());
    else if (a == "--mode") { std::string s = nx(); o.mode = (s == "gpu") ? 1 : (s == "h100" || s == "gpu2") ? 2 : 0; }  // gpu2: old name
    else if (a == "--stage2-mode") { std::string s = nx(); o.mode2 = (s == "h100" || s == "gpu2") ? 2 : 1; }
    else if (a == "--h100-model" || a == "--gm2") {
      std::string s = nx();
      if (sscanf(s.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &o.h100m[0], &o.h100m[1], &o.h100m[2], &o.h100m[3],
                 &o.h100m[4], &o.h100m[5], &o.h100m[6], &o.h100m[7], &o.h100m[8]) != 9)
        usage();
    }
    else if (a == "--sc-target") o.sc_target = std::stod(nx());
    else if (a == "--sc-weight") o.sc_weight = std::stod(nx());
    else if (a == "--scheme") o.scheme = nx();
    else if (a == "--trunc-frac") o.trunc_frac = std::stod(nx());
    else if (a == "--kernel") o.kernel = nx();
    else if (a == "--block") o.block = std::stoi(nx());
    else if (a == "--wpb") o.wpb = std::stoi(nx());
    else if (a == "--lpc") o.lpc = std::stoi(nx());
    else if (a == "--seed") o.seed = std::stoull(nx());
    else if (a == "--eval-every") o.eval_every = std::stoi(nx());
    else if (a == "--ref-chains") o.ref_chains = std::stoi(nx());
    else if (a == "--max-levels") o.max_levels = std::stoi(nx());
    else if (a == "--slicing") o.slicing = true;
    else if (a == "--slice-moves") o.slice_moves = std::stoi(nx());
    else if (a == "--slice-every") o.slice_every = std::stoi(nx());
    else if (a == "--max-slices") o.max_slices = std::stoi(nx());
    else if (a == "--init-slices") { o.init_slices = nx(); o.slicing = true; }
    else if (a == "--random-slices") { o.random_slices = std::stoi(nx()); o.slicing = true; }
    else if (a == "--wide") o.wide = true;
    else if (a == "--check") o.check = true;
    else if (a == "--no-cross") o.cross = false;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--gm") {
      std::string s = nx();
      if (sscanf(s.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf", &o.gm[0], &o.gm[1], &o.gm[2], &o.gm[3], &o.gm[4], &o.gm[5]) != 6)
        usage();
    } else usage();
  }
  if (o.inst.empty()) usage();
  return o;
}

static uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Everything needed to run one label-width instantiation (W > 0 fast path, W == 0 wide path).
template <int W>
struct Runner {
  const Inst& I;
  const Opts& O;
  Params P;
  int NW;
  std::vector<uint32_t> h_leaf;
  int nchains;
  uint32_t *d_rec = nullptr, *d_rec2 = nullptr, *d_leaf = nullptr;
  uint16_t* d_stk = nullptr;
  unsigned long long* d_stats = nullptr;
  float* d_lcost = nullptr;
  int *d_sc = nullptr, *d_ns = nullptr;
  int* d_anc = nullptr;
  std::string kernel;
  int mode;              // current energy mode (0 tc, 1 gpu, 2 h100); may switch for stage 2
  double kernel_ms = 0;  // accumulated sweep-kernel time

  Runner(const Inst& I_, const Opts& O_) : I(I_), O(O_) {
    nchains = O.nchains;
    kernel = O.kernel;
    mode = O.mode;
    NW = W > 0 ? W : round4((I.nlab + 31) / 32);
    P.m = I.m;
    P.nint = I.m - 1;
    P.nw = NW;
    P.s_off = round4(P.nint * NW + P.nint);
    P.chain_words = P.s_off + round4(NW);
    P.leaf_sc = 0;
    for (auto& ix : I.ixs) P.leaf_sc = std::max(P.leaf_sc, (int)ix.size());
    P.slicing = O.slicing ? 1 : 0;
    P.slice_moves = (O.slicing && O.random_slices == 0) ? O.slice_moves : 0;
    P.slice_every = std::max(1, O.slice_every);
    P.max_slices = O.max_slices > 0 ? O.max_slices : I.nlab;
    P.sc_target = (float)O.sc_target;
    P.sc_weight = (float)O.sc_weight;
    uint64_t k = splitmix64(O.seed);
    P.key0 = (uint32_t)k;
    P.key1 = (uint32_t)(k >> 32);
    P.lpeak = (float)std::log2(O.gm[0]);
    P.lbw = (float)std::log2(O.gm[1]);
    P.m0 = (int)O.gm[2];
    P.n0 = (int)O.gm[3];
    P.k0 = (int)O.gm[4];
    P.llaunch = (float)std::log2(O.gm[5]);
    P.v2_l3launch = (float)std::log2(3 * O.h100m[0]);
    P.v2_lperm2 = (float)std::log2(2 / O.h100m[1]);
    P.v2_lcc_bit = (float)std::log2(O.h100m[2]);
    P.v2_lcc_cell = (float)std::log2(O.h100m[3]);
    P.v2_ltc_bit = (float)std::log2(O.h100m[4]);
    P.v2_ltc_pad = (float)std::log2(O.h100m[5]);
    P.v2_ltc_out = (float)std::log2(O.h100m[6]);
    P.v2_small_rank = (int)O.h100m[7];
    P.v2_lsmall_cell = (float)std::log2(O.h100m[8]);
    h_leaf.assign((size_t)I.m * NW, 0);
    for (int i = 0; i < I.m; i++)
      for (int l : I.ixs[i]) h_leaf[(size_t)i * NW + (l >> 5)] |= 1u << (l & 31);
    if (kernel == "auto") {
      // Measured on the RTX 6000 (bench/results/throughput_rtx6000.txt): staging chains in shared
      // memory (warpsm) wins when a full block of 8 chains fits, or when the population no longer
      // fits in L2 and at least 4 chains fit per block; otherwise the warp kernel on global memory.
      int l2 = 0;
      CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
      size_t foot = chain_bytes() * (size_t)nchains;
      size_t per_warp = (size_t)(((P.nint + 7) / 8) * 4) * 4 + chain_bytes();
      size_t leafb = (size_t)round4(P.m * NW) * 4;
      size_t fit = leafb < (size_t)g_smem_optin ? ((size_t)g_smem_optin - leafb) / per_warp : 0;
      kernel = (fit >= 8 || (foot > (size_t)l2 && fit >= 4)) ? "warpsm" : "warp";
    }
    if (P.slice_moves > 0 && (kernel == "seq" || kernel == "seqsm" || O.lpc != 32)) {
      fprintf(stderr, "slice moves need the warp kernels with --lpc 32\n");
      exit(1);
    }
  }

  size_t chain_bytes() const { return (size_t)P.chain_words * 4; }
  const uint32_t* chS(const uint32_t* base) const { return P.slicing ? base + P.s_off : nullptr; }

  void alloc() {
    CK(cudaMalloc(&d_rec, chain_bytes() * nchains));
    if (O.scheme != "indep") CK(cudaMalloc(&d_rec2, chain_bytes() * nchains));
    CK(cudaMalloc(&d_leaf, h_leaf.size() * 4));
    CK(cudaMemcpy(d_leaf, h_leaf.data(), h_leaf.size() * 4, cudaMemcpyHostToDevice));
    if (kernel == "seq" || kernel == "seqsm" || O.check) CK(cudaMalloc(&d_stk, (size_t)P.nint * nchains * sizeof(uint16_t)));
    CK(cudaMalloc(&d_stats, 6 * sizeof(unsigned long long)));
    CK(cudaMemset(d_stats, 0, 6 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_lcost, nchains * sizeof(float)));
    CK(cudaMalloc(&d_sc, nchains * sizeof(int)));
    CK(cudaMalloc(&d_ns, nchains * sizeof(int)));
    CK(cudaMalloc(&d_anc, nchains * sizeof(int)));
  }
  void release() {
    cudaFree(d_rec); if (d_rec2) cudaFree(d_rec2); cudaFree(d_leaf); if (d_stk) cudaFree(d_stk); cudaFree(d_stats);
    cudaFree(d_lcost); cudaFree(d_sc); cudaFree(d_ns); cudaFree(d_anc);
  }

  // Build initial chains on the host (tree + topology-derived labels + slice set) and upload.
  void init_chains(std::vector<uint32_t>& host) {
    host.assign((size_t)P.chain_words * nchains, 0);
    std::vector<uint32_t> tree0;
    if (!O.tree.empty()) tree0 = read_tree(O.tree.c_str(), I.m);
    std::vector<uint32_t> S0(NW, 0);
    if (!O.init_slices.empty()) {
      const char* p = O.init_slices.c_str();
      while (*p) {
        char* q;
        long l = strtol(p, &q, 10);
        if (q == p) break;
        if (l < 0 || l >= I.nlab) { fprintf(stderr, "bad --init-slices label\n"); exit(1); }
        S0[l >> 5] |= 1u << (l & 31);
        p = q;
        while (*p == ',' || *p == ' ') p++;
      }
    }
    bool ok = true;
#pragma omp parallel for schedule(dynamic, 16)
    for (int c = 0; c < nchains; c++) {
      uint32_t* base = &host[(size_t)c * P.chain_words];
      if (!O.tree.empty() && c > 0) continue;  // replicated below
      std::vector<uint32_t> ch = O.tree.empty() ? random_tree(I.m, splitmix64(O.seed * 1000003ULL + c)) : tree0;
      std::copy(ch.begin(), ch.end(), base + (size_t)P.nint * NW);
      if (!labels_from_topology(I, NW, ch.data(), base)) ok = false;
    }
    if (!ok) { fprintf(stderr, "invalid initial tree\n"); exit(1); }
    if (!O.tree.empty())
      for (int c = 1; c < nchains; c++)
        std::copy(host.begin(), host.begin() + P.chain_words, host.begin() + (size_t)c * P.chain_words);
    for (int c = 0; c < nchains; c++) {
      uint32_t* S = &host[(size_t)c * P.chain_words + P.s_off];
      std::copy(S0.begin(), S0.end(), S);
      if (O.random_slices > 0) {  // check mode: a different fixed random slice set per chain
        std::mt19937_64 rng(splitmix64(O.seed * 7919ULL + c));
        for (int k = 0; k < O.random_slices; k++) {
          int l = (int)(rng() % I.nlab);
          S[l >> 5] |= 1u << (l & 31);
        }
      }
    }
    CK(cudaMemcpy(d_rec, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  }

  // Launch the warp-family kernel: wpb warps per block, 32/LPC chains per warp.
  template <int MODE, int LPC>
  void launch_warp(uint32_t* rec, float beta, uint32_t sweep0, int nsweeps, bool sm) {
    constexpr int CPW = 32 / LPC;
    int wlw = ((P.nint + 7) / 8) * 4;
    size_t per_chain = (size_t)wlw * 4 + (sm ? chain_bytes() : 0);
    size_t leafb = sm ? (size_t)round4(P.m * NW) * 4 : 0;
    // warps per block: as many as fit in shared memory (work lists [+ chains]), at most 8 (smem) / 4
    int wpb = O.wpb > 0 ? O.wpb
                        : (int)std::min<size_t>(sm ? 8 : 4, ((size_t)g_smem_optin - std::min<size_t>(leafb, g_smem_optin)) /
                                                                (per_chain * CPW));
    if (wpb < 1 || leafb + (size_t)wpb * CPW * per_chain > (size_t)g_smem_optin) {
      fprintf(stderr, "chains do not fit in shared memory (try --kernel warp / --lpc 32)\n");
      exit(1);
    }
    size_t smem = leafb + (size_t)wpb * CPW * per_chain;
    static bool attr = false;
    if (!attr) {
      CK(cudaFuncSetAttribute(k_warp<W, MODE, true, LPC>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem_optin));
      CK(cudaFuncSetAttribute(k_warp<W, MODE, false, LPC>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem_optin));
      attr = true;
    }
    int grid = (nchains + wpb * CPW - 1) / (wpb * CPW);
    if (sm) k_warp<W, MODE, true, LPC><<<grid, 32 * wpb, smem>>>(P, rec, d_leaf, nchains, beta, sweep0, nsweeps, d_stats);
    else k_warp<W, MODE, false, LPC><<<grid, 32 * wpb, smem>>>(P, rec, d_leaf, nchains, beta, sweep0, nsweeps, d_stats);
  }

  // Launch nsweeps sweeps of every chain at inverse temperature beta.
  template <int MODE>
  void sweeps_mode(uint32_t* rec, float beta, uint32_t sweep0, int nsweeps) {
    if (kernel == "seq" || kernel == "seqsm") {
      int B = O.block > 0 ? O.block : 64;
      size_t leafb = (size_t)round4(P.m * NW) * 4;
      if (kernel == "seqsm" && O.block <= 0)  // as many chains per block as fit in shared memory (<= 32)
        B = (int)std::max<size_t>(1, std::min<size_t>(32, (g_smem_optin - leafb) / chain_bytes()));
      if (kernel == "seq") {
        k_seq<W, MODE, false><<<(nchains + B - 1) / B, B>>>(P, rec, d_leaf, nchains, beta, sweep0, nsweeps, d_stk, d_stats);
      } else {
        size_t smem = leafb + (size_t)B * chain_bytes();
        static bool attr = false;
        if (!attr) { CK(cudaFuncSetAttribute(k_seq<W, MODE, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem_optin)); attr = true; }
        if (smem > (size_t)g_smem_optin) { fprintf(stderr, "seqsm: %zu bytes smem needed for %d chains/block; lower --block\n", smem, B); exit(1); }
        k_seq<W, MODE, true><<<(nchains + B - 1) / B, B, smem>>>(P, rec, d_leaf, nchains, beta, sweep0, nsweeps, d_stk, d_stats);
      }
    } else {
      bool sm = kernel == "warpsm";
      if constexpr (W == 0) {  // wide path: one warp per chain only
        launch_warp<MODE, 32>(rec, beta, sweep0, nsweeps, sm);
      } else {
        if (O.lpc == 32) launch_warp<MODE, 32>(rec, beta, sweep0, nsweeps, sm);
        else if (O.lpc == 16) launch_warp<MODE, 16>(rec, beta, sweep0, nsweeps, sm);
        else launch_warp<MODE, 8>(rec, beta, sweep0, nsweeps, sm);
      }
    }
    CK(cudaGetLastError());
  }
  void sweeps(uint32_t* rec, float beta, uint32_t sweep0, int nsweeps) {
    if (mode == 0) sweeps_mode<0>(rec, beta, sweep0, nsweeps);
    else if (mode == 1) sweeps_mode<1>(rec, beta, sweep0, nsweeps);
    else sweeps_mode<2>(rec, beta, sweep0, nsweeps);
  }
  void energy(uint32_t* rec, std::vector<float>& lc, std::vector<int>& sc, std::vector<int>& ns) {
    int B = 128, grid = (nchains * 32 + B - 1) / B;  // one warp per chain
    if (mode == 0) k_energy<W, 0><<<grid, B>>>(P, rec, d_leaf, nchains, d_lcost, d_sc, d_ns);
    else if (mode == 1) k_energy<W, 1><<<grid, B>>>(P, rec, d_leaf, nchains, d_lcost, d_sc, d_ns);
    else k_energy<W, 2><<<grid, B>>>(P, rec, d_leaf, nchains, d_lcost, d_sc, d_ns);
    CK(cudaGetLastError());
    lc.resize(nchains);
    sc.resize(nchains);
    ns.resize(nchains);
    CK(cudaMemcpy(lc.data(), d_lcost, nchains * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sc.data(), d_sc, nchains * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ns.data(), d_ns, nchains * 4, cudaMemcpyDeviceToHost));
  }
  // Selection score (lower is better).
  //   no slicing, mode 0: omeco ScoreFunction (tc_weight=1): 2^tc + sc_weight*max(0, 2^sc - 2^sc_target)
  //   no slicing, mode 1/2: the log-domain energy below
  //   slicing: feasibility first (sc <= sc_target), then total cost |S| + log2(per-slice cost)
  // All scores are in log2 domain (no overflow for tc > 1024).
  double score(float lc, int sc, int ns) const {
    if (O.slicing) return (double)ns + (double)lc + 1000.0 * std::max(0.0, (double)sc - O.sc_target);
    if (mode == 0) {  // log2(2^tc + w*(2^sc - 2^sc_target)) when sc > sc_target, else tc
      double tc = (double)lc;
      if ((double)sc <= O.sc_target || O.sc_weight <= 0) return tc;
      double pen = std::log2(O.sc_weight) + (double)sc + std::log2(1.0 - std::exp2(O.sc_target - (double)sc));
      double mx = std::max(tc, pen), mn = std::min(tc, pen);
      return mx + std::log2(1.0 + std::exp2(mn - mx));
    }
    return energyE(lc, sc, ns);
  }
  // Log-domain energy used for population resampling:
  //   |S| + log2(per-slice cost) + sc_weight*max(0, sc - sc_target).
  double energyE(float lc, int sc, int ns) const {
    return (double)ns + (double)lc + O.sc_weight * std::max(0.0, (double)sc - O.sc_target);
  }

  // ---------------------------------------------------------------------------------
  int run_anneal() {
    std::vector<float> betas = parse_betas(O.betas);
    if (O.max_levels > 0 && (int)betas.size() > O.max_levels) betas.resize(O.max_levels);
    // level schedule: (beta, mode, sweeps); stage 2 (if any) continues the population in mode2
    std::vector<int> lmode(betas.size(), O.mode), lniters(betas.size(), O.niters);
    if (!O.betas2.empty()) {
      std::vector<float> b2 = parse_betas(O.betas2);
      for (float b : b2) { betas.push_back(b); lmode.push_back(O.mode2); lniters.push_back(O.niters2 > 0 ? O.niters2 : O.niters); }
    }
    HostEnergy H1;
    bool staged = false;
    uint32_t sweep_ctr = 0;
    double t_start = now_s();
    alloc();
    std::vector<uint32_t> host;
    init_chains(host);
    CK(cudaDeviceSynchronize());
    double t_init = now_s() - t_start;
    double t0 = now_s();
    std::vector<float> lc;
    std::vector<int> sc, ns;
    double best_score = INFINITY;
    int best_chain = -1;
    std::vector<uint32_t> best(P.chain_words);
    std::vector<int> family(nchains), nfam(nchains);
    std::iota(family.begin(), family.end(), 0);
    std::string trace;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::mt19937_64 hrng(splitmix64(O.seed + 777));
    float beta_last_resample = betas.empty() ? 0.f : betas[0];
    int nresample = 0;
    for (size_t lev = 0; lev < betas.size(); lev++) {
      if (lmode[lev] != mode) {
        // stage switch: record the stage-1 best and keep it as the incumbent, scored in the new
        // energy (so stage 2 can only improve the returned tree in the new objective)
        H1 = host_energy(I, NW, P, best.data(), best.data() + (size_t)P.nint * NW, h_leaf, chS(best.data()));
        staged = true;
        mode = lmode[lev];
        best_score = score((float)std::log2(mode == 1 ? H1.time : H1.time2), H1.sc, H1.ns);
        beta_last_resample = betas[lev];
      }
      CK(cudaEventRecord(e0));
      sweeps(d_rec, betas[lev], sweep_ctr, lniters[lev]);
      sweep_ctr += (uint32_t)lniters[lev];
      CK(cudaEventRecord(e1));
      bool last = lev + 1 == betas.size();
      bool switching = !last && lmode[lev + 1] != mode;  // no resampling across an energy change
      if ((int)(lev % O.eval_every) == O.eval_every - 1 || last || switching) {
        energy(d_rec, lc, sc, ns);  // synchronizes
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        kernel_ms += ms;
        int bi = 0;
        double bs = INFINITY, meanE = 0, meanNs = 0;
        for (int c = 0; c < nchains; c++) {
          double s = score(lc[c], sc[c], ns[c]);
          if (s < bs) { bs = s; bi = c; }
          meanE += energyE(lc[c], sc[c], ns[c]);
          meanNs += ns[c];
        }
        meanE /= nchains;
        meanNs /= nchains;
        if (bs < best_score) {
          best_score = bs;
          best_chain = bi;
          CK(cudaMemcpy(best.data(), d_rec + (size_t)bi * P.chain_words, chain_bytes(), cudaMemcpyDeviceToHost));
        }
        // count distinct ancestral families (population diversity)
        std::fill(nfam.begin(), nfam.end(), 0);
        int nf = 0;
        for (int c = 0; c < nchains; c++) if (nfam[family[c]]++ == 0) nf++;
        char buf[320];
        snprintf(buf, sizeof buf, "%s[%.4f,%zu,%.4f,%.6f,%d,%.6f,%d,%d,%d,%.3f]", trace.empty() ? "" : ",", now_s() - t0,
                 lev, betas[lev], (double)lc[bi], sc[bi], meanE, nf, mode, ns[bi], meanNs);
        trace += buf;
        if (!O.quiet && (lev % 25 == 0 || last))
          fprintf(stderr, "lev %4zu beta %7.3f  best(l2cost/slice %.3f sc %d |S| %d)  meanE %.3f  mean|S| %.2f  families %d  t=%.2fs\n",
                  lev, betas[lev], (double)lc[bi], sc[bi], ns[bi], meanE, meanNs, nf, now_s() - t0);
        // ---- population step ----
        if (!last && !switching && O.scheme == "pa") {
          // population annealing: resample with weights exp(-(beta_next - beta_prev) * E)
          double db = betas[lev + 1] - beta_last_resample;
          beta_last_resample = betas[lev + 1];
          std::vector<double> w(nchains);
          double Emin = INFINITY;
          for (int c = 0; c < nchains; c++) Emin = std::min(Emin, energyE(lc[c], sc[c], ns[c]));
          double wsum = 0;
          for (int c = 0; c < nchains; c++) { w[c] = std::exp(-db * (energyE(lc[c], sc[c], ns[c]) - Emin)); wsum += w[c]; }
          // systematic resampling
          std::vector<int> anc(nchains);
          double u = std::uniform_real_distribution<double>(0, 1)(hrng), acc = 0;
          int j = 0;
          for (int c = 0; c < nchains; c++) {
            acc += w[c] / wsum * nchains;
            while (j < nchains && j + u < acc) anc[j++] = c;
          }
          while (j < nchains) anc[j++] = nchains - 1;
          resample(anc, family);
          nresample++;
        } else if (!last && !switching && O.scheme == "trunc") {
          // truncation selection: worst trunc_frac of chains replaced by copies of the best ones
          std::vector<int> idx(nchains);
          std::iota(idx.begin(), idx.end(), 0);
          std::sort(idx.begin(), idx.end(),
                    [&](int a, int b) { return energyE(lc[a], sc[a], ns[a]) < energyE(lc[b], sc[b], ns[b]); });
          std::vector<int> anc(nchains);
          std::iota(anc.begin(), anc.end(), 0);
          int nrep = (int)(O.trunc_frac * nchains);
          for (int k = 0; k < nrep; k++) anc[idx[nchains - 1 - k]] = idx[k];
          resample(anc, family);
          nresample++;
        }
      } else {
        // keep event timing accurate without a host sync on every level
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        kernel_ms += ms;
      }
    }
    CK(cudaDeviceSynchronize());
    double t_anneal = now_s() - t0;
    unsigned long long stats[6];
    CK(cudaMemcpy(stats, d_stats, sizeof stats, cudaMemcpyDeviceToHost));

    // Final best tree: verify on host from scratch; with slicing, repair sc > sc_target greedily.
    uint32_t* bch = best.data() + (size_t)P.nint * NW;
    std::vector<uint32_t> lab2((size_t)P.nint * NW);
    bool valid = labels_from_topology(I, NW, bch, lab2.data());
    bool labels_ok = valid && std::equal(lab2.begin(), lab2.end(), best.begin());
    int n_repair = 0;
    if (O.slicing && valid) n_repair = greedy_repair(I, NW, P, lab2.data(), h_leaf, best.data() + P.s_off, O.sc_target);
    HostEnergy HE = host_energy(I, NW, P, lab2.data(), bch, h_leaf, chS(best.data()));
    if (!O.quiet)
      fprintf(stderr, "done: best tc %.4f sc %d |S| %d (repair +%d) time %.4e s time2 %.4e s (per slice) | anneal %.2fs init %.2fs "
              "kernel %.2fs | %.3e proposals/s | lane occupancy %.3f | slice moves %llu/%llu accepted\n",
              HE.tc, HE.sc, HE.ns, n_repair, HE.time, HE.time2, t_anneal, t_init, kernel_ms / 1e3,
              stats[0] / (kernel_ms / 1e3), stats[2] ? stats[3] / ((double)O.lpc * stats[2]) : 0.0, stats[5], stats[4]);
    FILE* f = O.out.empty() ? stdout : fopen(O.out.c_str(), "w");
    fprintf(f, "{\"tc\": %.6f, \"sc\": %d, \"gpu_time\": %s, \"h100_time\": %s, \"valid\": %s, \"labels_ok\": %s, \"best_chain\": %d,\n",
            HE.tc, HE.sc, jnum(HE.time).c_str(), jnum(HE.time2).c_str(), valid ? "true" : "false", labels_ok ? "true" : "false", best_chain);
    fprintf(f, " \"slicing\": %s, \"nsliced\": %d, \"n_repair\": %d, \"tc_total\": %.6f, \"gpu_time_total\": %s, \"h100_time_total\": %s, \"sliced\": [",
            O.slicing ? "true" : "false", HE.ns, n_repair, HE.tc + HE.ns, jnum(HE.time * std::exp2((double)HE.ns)).c_str(),
            jnum(HE.time2 * std::exp2((double)HE.ns)).c_str());
    {
      int k = 0;
      const uint32_t* S = best.data() + P.s_off;
      for (int l = 0; l < I.nlab; l++)
        if (O.slicing && (S[l >> 5] >> (l & 31) & 1u)) fprintf(f, "%s%d", k++ ? "," : "", l);
    }
    fprintf(f, "],\n \"t_anneal\": %.6f, \"t_init\": %.6f, \"kernel_s\": %.6f, \"proposals\": %llu, \"accepts\": %llu, \"warp_steps\": %llu, \"node_visits\": %llu, \"slice_proposals\": %llu, \"slice_accepts\": %llu,\n",
            t_anneal, t_init, kernel_ms / 1e3, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]);
    fprintf(f, " \"nchains\": %d, \"niters\": %d, \"nlevels\": %zu, \"total_sweeps\": %u, \"W\": %d, \"nw\": %d, \"lpc\": %d, \"kernel\": \"%s\", \"scheme\": \"%s\", \"mode\": %d, \"nresample\": %d,\n",
            nchains, O.niters, betas.size(), sweep_ctr, W, NW, O.lpc, kernel.c_str(), O.scheme.c_str(), O.mode, nresample);
    if (staged)
      fprintf(f, " \"stage1_tc\": %.6f, \"stage1_sc\": %d, \"stage1_nsliced\": %d, \"stage1_gpu_time\": %s, \"stage1_h100_time\": %s, \"stage1_h100_time_total\": %s,\n",
              H1.tc, H1.sc, H1.ns, jnum(H1.time).c_str(), jnum(H1.time2).c_str(), jnum(H1.time2 * std::exp2((double)H1.ns)).c_str());
    fprintf(f, " \"trace_cols\": [\"t\", \"level\", \"beta\", \"best_lcost_per_slice\", \"best_sc\", \"mean_E\", \"families\", \"mode\", \"best_nsliced\", \"mean_nsliced\"],\n");
    fprintf(f, " \"trace\": [%s],\n \"tree\": [", trace.c_str());
    for (int i = 0; i < P.nint; i++) fprintf(f, "%s[%u,%u]", i ? "," : "", bch[i] & 0xFFFFu, bch[i] >> 16);
    fprintf(f, "]}\n");
    if (f != stdout) fclose(f);
    release();
    return valid && labels_ok ? 0 : 2;
  }

  void resample(const std::vector<int>& anc, std::vector<int>& family) {
    CK(cudaMemcpy(d_anc, anc.data(), nchains * 4, cudaMemcpyHostToDevice));
    k_copy<<<nchains, 128>>>((const uint4*)d_rec, (uint4*)d_rec2, d_anc, P.chain_words / 4);
    CK(cudaGetLastError());
    std::swap(d_rec, d_rec2);
    std::vector<int> nfm(nchains);
    for (int c = 0; c < nchains; c++) nfm[c] = family[anc[c]];
    family.swap(nfm);
  }

  // ---------------------------------------------------------------------------------
  // Validation: (1) all chains valid and maintained labels == labels recomputed from
  // topology, (2) GPU energy kernel == host energy (on the sliced network), (3) the other
  // kernel family gives a bit-identical result, (4) the CPU Julia-port reference gives
  // bit-identical trees.  (3) and (4) need a fixed slice set (no slice moves).
  int run_check() {
    std::vector<float> betas = parse_betas(O.betas);
    if (O.max_levels > 0 && (int)betas.size() > O.max_levels) betas.resize(O.max_levels);
    alloc();
    std::vector<uint32_t> init;
    init_chains(init);
    for (size_t lev = 0; lev < betas.size(); lev++) sweeps(d_rec, betas[lev], (uint32_t)(lev * O.niters), O.niters);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> fin((size_t)P.chain_words * nchains);
    CK(cudaMemcpy(fin.data(), d_rec, fin.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long stats[6];
    CK(cudaMemcpy(stats, d_stats, sizeof stats, cudaMemcpyDeviceToHost));
    printf("check: %d chains, %zu levels x %d sweeps, kernel %s, mode %d, W=%d (nw %d), slicing %d (moves %d): "
           "%llu proposals, %llu accepted, %llu/%llu slice moves accepted\n",
           nchains, betas.size(), O.niters, kernel.c_str(), O.mode, W, NW, P.slicing, P.slice_moves, stats[0], stats[1],
           stats[5], stats[4]);
    // (1) labels
    int bad_tree = 0, bad_lab = 0, changed = 0, Schanged = 0;
    std::vector<uint32_t> lab2((size_t)P.nint * NW);
    for (int c = 0; c < nchains; c++) {
      const uint32_t* base = &fin[(size_t)c * P.chain_words];
      const uint32_t* b0 = &init[(size_t)c * P.chain_words];
      if (!labels_from_topology(I, NW, base + (size_t)P.nint * NW, lab2.data())) { bad_tree++; continue; }
      if (!std::equal(lab2.begin(), lab2.end(), base)) bad_lab++;
      if (!std::equal(base, base + P.s_off, b0)) changed++;
      if (!std::equal(base + P.s_off, base + P.s_off + NW, b0 + P.s_off)) Schanged++;
    }
    printf("  [1] invalid trees: %d, chains with label mismatch vs recomputation: %d (chains changed by SA: %d/%d, slice set changed: %d)\n",
           bad_tree, bad_lab, changed, nchains, Schanged);
    // (2) energies (per slice, on the sliced network)
    std::vector<float> lc;
    std::vector<int> sc, ns;
    energy(d_rec, lc, sc, ns);
    double maxd = 0;
    int scbad = 0;
    for (int c = 0; c < nchains; c++) {
      const uint32_t* base = &fin[(size_t)c * P.chain_words];
      HostEnergy HE = host_energy(I, NW, P, base, base + (size_t)P.nint * NW, h_leaf, chS(base));
      double ref = O.mode == 0 ? HE.tc : std::log2(O.mode == 1 ? HE.time : HE.time2);
      maxd = std::max(maxd, std::fabs(ref - (double)lc[c]));
      if (HE.sc != sc[c] || HE.ns != ns[c]) scbad++;
    }
    printf("  [2] GPU energy kernel vs host double: max |dlog2cost| = %.3g, sc/|S| mismatches %d\n", maxd, scbad);
    // (3) the other kernel family from the same initial state
    int xdiff = -1;
    if (O.cross && P.slice_moves == 0) {
      std::string other = (kernel == "seq" || kernel == "seqsm") ? "warp" : "seq";
      std::string saved = kernel;
      kernel = other;
      CK(cudaMemcpy(d_rec, init.data(), init.size() * 4, cudaMemcpyHostToDevice));
      for (size_t lev = 0; lev < betas.size(); lev++) sweeps(d_rec, betas[lev], (uint32_t)(lev * O.niters), O.niters);
      std::vector<uint32_t> fin2(fin.size());
      CK(cudaMemcpy(fin2.data(), d_rec, fin2.size() * 4, cudaMemcpyDeviceToHost));
      kernel = saved;
      xdiff = 0;
      for (int c = 0; c < nchains; c++)
        if (!std::equal(fin.begin() + (size_t)c * P.chain_words, fin.begin() + (size_t)(c + 1) * P.chain_words,
                        fin2.begin() + (size_t)c * P.chain_words))
          xdiff++;
      printf("  [3] kernel %s vs %s: chains differing: %d/%d\n", saved.c_str(), other.c_str(), xdiff, nchains);
    }
    // (4) CPU reference (fixed slice set)
    int R = P.slice_moves == 0 ? std::min(O.ref_chains, nchains) : 0, refdiff = 0;
    long long rp = 0, ra = 0;
    double tref = now_s();
    for (int c = 0; c < R; c++) {
      RefTree T;
      T.m = I.m;
      T.left.assign(2 * I.m - 1, -1);
      T.right.assign(2 * I.m - 1, -1);
      T.labels.resize(2 * I.m - 1);
      T.sliced.assign(I.nlab, 0);
      const uint32_t* b0 = &init[(size_t)c * P.chain_words];
      if (P.slicing)
        for (int l = 0; l < I.nlab; l++) T.sliced[l] = (char)(b0[P.s_off + (l >> 5)] >> (l & 31) & 1u);
      for (int i = 0; i < I.m; i++) T.labels[i] = I.ixs[i];
      for (int i = 0; i < P.nint; i++) {
        uint32_t cw = b0[(size_t)P.nint * NW + i];
        T.left[I.m + i] = (int)(cw & 0xFFFFu);
        T.right[I.m + i] = (int)(cw >> 16);
        for (int l = 0; l < I.nlab; l++)
          if (b0[(size_t)i * NW + (l >> 5)] >> (l & 31) & 1u) T.labels[I.m + i].push_back(l);
      }
      for (size_t lev = 0; lev < betas.size(); lev++)
        for (int it = 0; it < O.niters; it++) {
          uint32_t sw = (uint32_t)(lev * O.niters + it);
          if (O.mode == 0) ref_optimize_subtree<0>(T, 2 * I.m - 2, betas[lev], sw, (uint32_t)c, P, rp, ra);
          else if (O.mode == 1) ref_optimize_subtree<1>(T, 2 * I.m - 2, betas[lev], sw, (uint32_t)c, P, rp, ra);
          else ref_optimize_subtree<2>(T, 2 * I.m - 2, betas[lev], sw, (uint32_t)c, P, rp, ra);
        }
      const uint32_t* bf = &fin[(size_t)c * P.chain_words];
      bool same = true;
      std::vector<uint32_t> z(NW);
      for (int i = 0; i < P.nint && same; i++) {
        uint32_t cw = bf[(size_t)P.nint * NW + i];
        if (T.left[I.m + i] != (int)(cw & 0xFFFFu) || T.right[I.m + i] != (int)(cw >> 16)) same = false;
        std::fill(z.begin(), z.end(), 0u);
        for (int l : T.labels[I.m + i]) z[l >> 5] |= 1u << (l & 31);
        for (int w = 0; w < NW; w++)
          if (z[w] != bf[(size_t)i * NW + w]) same = false;
      }
      if (!same) refdiff++;
    }
    tref = now_s() - tref;
    if (R > 0)
      printf("  [4] CPU Julia-port reference: %d/%d chains differ from GPU (ref did %lld proposals, %lld accepts, %.2fs => %.3g proposals/s single core)\n",
             refdiff, R, rp, ra, tref, rp / tref);
    bool ok = bad_tree == 0 && bad_lab == 0 && scbad == 0 && maxd < 1e-3 && xdiff <= 0 && refdiff == 0;
    printf("CHECK %s\n", ok ? "PASSED" : "FAILED");
    release();
    return ok ? 0 : 3;
  }
};

template <int W>
static int run(const Inst& I, const Opts& O) {
  Runner<W> R(I, O);
  return O.check ? R.run_check() : R.run_anneal();
}

int main(int argc, char** argv) {
  Opts O = parse(argc, argv);
  Inst I = read_instance(O.inst.c_str());
  if (I.m < 3) { fprintf(stderr, "need at least 3 tensors\n"); return 1; }
  if (2 * I.m - 1 > 65535) { fprintf(stderr, "too many tensors (node ids are 16-bit)\n"); return 1; }
  for (int d = 0; d < 64; d++) h_lse_tab[d] = (float)std::log2(1.0 + std::exp2(-(double)d));
  CK(cudaMemcpyToSymbol(c_lse_tab, h_lse_tab, sizeof h_lse_tab));
  CK(cudaDeviceGetAttribute(&g_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  int Wn = (I.nlab + 31) / 32;
  if (O.wide || Wn > 16) return run<0>(I, O);  // wide path: any number of labels
  if (Wn <= 1) return run<1>(I, O);
  if (Wn <= 2) return run<2>(I, O);
  if (Wn <= 4) return run<4>(I, O);
  if (Wn <= 8) return run<8>(I, O);
  if (Wn <= 12) return run<12>(I, O);
  return run<16>(I, O);
}
