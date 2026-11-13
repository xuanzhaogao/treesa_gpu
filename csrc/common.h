// common.h -- host/device shared pieces of GPU TreeSA for dimension-2 tensor networks.
//
// Everything here is __host__ __device__ so that the CPU reference implementation
// (cpu_ref.h) uses *bit-identical* arithmetic and random numbers to the GPU kernels.
#pragma once
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

// ----------------------------------------------------------------------------
// Counter-based RNG: Philox4x32-10 (Salmon et al. 2011).
// Every proposal at internal node v during global sweep s of chain c uses
// philox({v, s, c, 0}, seed).  Because each node is visited exactly once per
// sweep, the random numbers do NOT depend on the order in which independent
// subtrees are processed, so the sequential kernel, the warp-parallel frontier
// kernel and the CPU reference produce identical trajectories.
// ----------------------------------------------------------------------------
struct U4 { uint32_t x, y, z, w; };

HD void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
#ifdef __CUDA_ARCH__
  hi = __umulhi(a, b);
  lo = a * b;
#else
  uint64_t p = (uint64_t)a * (uint64_t)b;
  hi = (uint32_t)(p >> 32);
  lo = (uint32_t)p;
#endif
}

HD U4 philox4x32(U4 c, uint32_t k0, uint32_t k1) {
#pragma unroll
  for (int i = 0; i < 10; i++) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, c.x, hi0, lo0);
    mulhilo32(0xCD9E8D57u, c.z, hi1, lo1);
    c = U4{hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c;
}

HD int popc32(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __popc(x);
#else
  return __builtin_popcount(x);
#endif
}

// ----------------------------------------------------------------------------
// Parameters shared by all kernels.
// ----------------------------------------------------------------------------
struct Params {
  int m;            // number of leaves (= input tensors); leaves are nodes 0..m-1
  int nint;         // number of internal nodes = m-1; nodes m..2m-2; root = 2m-2
  int nw;           // 32-bit words per label set (compile-time W for the fast path, runtime here)
  int s_off;        // word offset of the slice set S (nw words) inside a chain record
  int chain_words;  // words per chain: [labels nint*nw][children nint][pad][S nw][pad]
  int leaf_sc;      // max leaf tensor size (log2, unsliced), contributes to sc
  int slicing;      // 1: costs use label sets masked by ~S (S may be nonempty)
  int slice_moves;  // >0: number of slice proposals after every `slice_every` sweeps (warp kernel)
  int slice_every;
  int max_slices;   // cap on |S|
  float sc_target, sc_weight;
  uint32_t key0, key1;  // Philox key (from seed)
  // GPU-runtime cost model (mode 1), all in log2 units:
  //   t = max(2^(1+B+max(M,m0)+max(N,n0)+max(K,k0)) / peak,
  //           (2^(B+M+K)+2^(B+N+K)+2^(B+M+N)) / bw) + launch
  float lpeak, lbw, llaunch;
  int m0, n0, k0;
  // Measured H100 model (mode 2 = "h100", trees.h100_time), log2 of the constants:
  //   small (B+M+N <= small_rank):  t = small_cell * 2^(B+M+N+K)
  //   else: t = 3*launch + 2(a+b)/perm_bw + min(cc, tc),  a=2^(B+M+K), b=2^(B+N+K), c=2^(B+M+N)
  //         cc = cc_bit*(a+b+c) + cc_cell*2^(B+M+N+K)
  //         tc = tc_bit*(a+b+c) + tc_pad*2^(B+max(M,6)+max(N,6)+max(K,8)) + tc_out*c
  //              (only if K >= 7 and max(M,N) >= 5)
  float v2_l3launch, v2_lperm2, v2_lcc_bit, v2_lcc_cell, v2_ltc_bit, v2_ltc_pad, v2_ltc_out, v2_lsmall_cell;
  int v2_small_rank;
};

// log2(1 + 2^-d) for integer d >= 0 (d >= 63 -> ~0).  Mode 0 (tc) costs are
// integers, so log2(2^a + 2^b) = max(a,b) + LSE_TAB[|a-b|] is evaluated with the
// same float table on host and device (deterministic).
#ifdef __CUDACC__
__constant__ float c_lse_tab[64];
#endif
extern float h_lse_tab[64];

HD float lse_tab(int d) {
  d = d < 63 ? d : 63;
#ifdef __CUDA_ARCH__
  return c_lse_tab[d];
#else
  return h_lse_tab[d];
#endif
}

// log2(2^a + 2^b) for integers.
HD float lse_i(int a, int b) {
  int mx = a > b ? a : b;
  int d = a > b ? a - b : b - a;
  return (float)mx + lse_tab(d);
}

// log2(2^a + 2^b) for floats (mode 1).  a or b may be -inf.
HD float lse_f(float a, float b) {
  float mx = a > b ? a : b;
  float mn = a > b ? b : a;
  if (mx == -INFINITY) return mx;
  return mx + log2f(1.0f + exp2f(mn - mx));
}

// log2 of modeled GPU time of one pairwise contraction with label counts B,M,N,K.
HD float gpu_lt(int B, int M, int N, int K, const Params& P) {
  int Me = M > P.m0 ? M : P.m0, Ne = N > P.n0 ? N : P.n0, Ke = K > P.k0 ? K : P.k0;
  float lops = (float)(1 + B + Me + Ne + Ke) - P.lpeak;
  float lbits = lse_f(lse_f((float)(B + M + K), (float)(B + N + K)), (float)(B + M + N)) - P.lbw;
  float lt = lops > lbits ? lops : lbits;
  return lse_f(lt, P.llaunch);
}

// log2 of modeled time, measured H100 model (see Params).
HD float h100_lt(int B, int M, int N, int K, const Params& P) {
  if (B + M + N <= P.v2_small_rank) return P.v2_lsmall_cell + (float)(B + M + N + K);
  float la = (float)(B + M + K), lb = (float)(B + N + K), lc = (float)(B + M + N);
  float lab = lse_f(la, lb), lmoved = lse_f(lab, lc);
  float perm = P.v2_lperm2 + lab;  // 2(a+b)/perm_bw
  float best = lse_f(P.v2_lcc_bit + lmoved, P.v2_lcc_cell + (float)(B + M + N + K));
  int MN = M > N ? M : N;
  if (K >= 7 && MN >= 5) {
    int Me = M > 6 ? M : 6, Ne = N > 6 ? N : 6, Ke = K > 8 ? K : 8;
    float tc = lse_f(lse_f(P.v2_ltc_bit + lmoved, P.v2_ltc_pad + (float)(B + Me + Ne + Ke)), P.v2_ltc_out + lc);
    best = tc < best ? tc : best;
  }
  return lse_f(lse_f(P.v2_l3launch, perm), best);
}

// Label counts of one pairwise contraction (x, y) -> z (after masking by ~S):
//   B=|x&y&z|, M=|x&~y&z|, N=|~x&y&z|, K=|x&y&~z|, Z=|z|.
// Cost (log2) of the contraction:
//   MODE 0:  tc = |z| + |x & y & ~z|                 (Julia `tcscrw`)
//   MODE 1:  log2 t(B, M, N, K) of the GPU model v1
//   MODE 2:  log2 t(B, M, N, K) of the measured H100 model
template <int MODE>
HD float node_cost(int B, int M, int N, int K, int Z, const Params& P) {
  if (MODE == 0) return (float)(Z + K);
  return MODE == 1 ? gpu_lt(B, M, N, K, P) : h100_lt(B, M, N, K, P);
}

// Accumulators for one TreeSA proposal: sizes of d, P, X and the counts of the four
// contractions involved:  0: (K,M)->P   1: (P,O)->d   (before)    2: (K,O)->X   3: (X,M)->d (after).
struct MoveCounts {
  int sd, sp, sx;
  int B[4], M[4], N[4], K[4];
};

HD void count3(uint32_t x, uint32_t y, uint32_t z, int& B, int& M, int& N, int& K) {
  B += popc32(x & y & z);
  M += popc32(x & ~y & z);
  N += popc32(~x & y & z);
  K += popc32(x & y & ~z);
}

// One 32-bit word of the five (already masked) label sets K, M, P, O, d and X.
template <int MODE>
HD void acc_counts(MoveCounts& c, uint32_t k, uint32_t m, uint32_t p, uint32_t o, uint32_t d, uint32_t x) {
  c.sd += popc32(d);
  c.sp += popc32(p);
  c.sx += popc32(x);
  if (MODE == 0) {  // only K (= |x&y&~z|) is needed for tc
    c.K[0] += popc32(k & m & ~p);
    c.K[1] += popc32(p & o & ~d);
    c.K[2] += popc32(k & o & ~x);
    c.K[3] += popc32(x & m & ~d);
  } else {
    count3(k, m, p, c.B[0], c.M[0], c.N[0], c.K[0]);
    count3(p, o, d, c.B[1], c.M[1], c.N[1], c.K[1]);
    count3(k, o, x, c.B[2], c.M[2], c.N[2], c.K[2]);
    count3(x, m, d, c.B[3], c.M[3], c.N[3], c.K[3]);
  }
}

// dE of the move (Julia `tcsc_diff` + energy):
//   dtc = log2(2^c2 + 2^c3) - log2(2^c0 + 2^c1)   (c_i = log2 cost of contraction i)
//   dE  = (max(sc0, sc1) > sc_target ? sc_weight : 0) * (sc1 - sc0) + dtc,
//   sc0 = max(|P|, |d|) (Julia `_sc`), sc1 = max(|X|, |d|).
template <int MODE>
HD float energy_from_counts(const MoveCounts& c, const Params& P) {
  float dtc;
  if (MODE == 0) {
    float tc0 = lse_i(c.sp + c.K[0], c.sd + c.K[1]);
    float tc1 = lse_i(c.sx + c.K[2], c.sd + c.K[3]);
    dtc = tc1 - tc0;
  } else {
    float t0 = lse_f(node_cost<MODE>(c.B[0], c.M[0], c.N[0], c.K[0], c.sp, P),
                     node_cost<MODE>(c.B[1], c.M[1], c.N[1], c.K[1], c.sd, P));
    float t1 = lse_f(node_cost<MODE>(c.B[2], c.M[2], c.N[2], c.K[2], c.sx, P),
                     node_cost<MODE>(c.B[3], c.M[3], c.N[3], c.K[3], c.sd, P));
    dtc = t1 - t0;
  }
  int sc0 = c.sp > c.sd ? c.sp : c.sd;
  int sc1 = c.sx > c.sd ? c.sx : c.sd;
  int dsc = sc1 - sc0;
  int scmax = sc0 > sc1 ? sc0 : sc1;
  float w = ((float)scmax > P.sc_target) ? P.sc_weight : 0.0f;
  return w * (float)dsc + dtc;
}

// Metropolis acceptance, shared so host and device agree:  u < exp(-beta*dE),
// u = top 24 bits of a Philox word in [0,1).
HD bool metropolis(uint32_t r, float beta, float dE) {
  float u = (float)(r >> 8) * (1.0f / 16777216.0f);
  return u < expf(-beta * dE);
}

// Rule selection, identical to Julia `ruleset(::TreeDecomp, tree)` + `rand(rst)`.
// li/ri: left/right child is internal.  Returns 0..3 (= Julia rules 1..4), or -1.
HD int pick_rule(bool li, bool ri, uint32_t r) {
  if (li && ri) return (int)(r >> 30);
  if (li) return (int)(r >> 31);       // right is leaf: rules 1:2
  if (ri) return 2 + (int)(r >> 31);   // left is leaf: rules 3:4
  return -1;
}
