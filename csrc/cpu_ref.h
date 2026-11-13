// cpu_ref.h -- straightforward CPU port of OMEinsumContractionOrders.jl `optimize_subtree!`
// (TreeDecomp, rw_weight = 0), used to validate the GPU kernels.
//
// Labels are plain std::vector<int> (no bitsets), the tree is a pointer-style node array,
// and the recursion is the literal Julia recursion.  Slicing follows Julia's `Slicer`: a sliced
// label keeps appearing in the label sets but has log2 size 0, i.e. it is skipped in every size
// and cost (`sl[l] != 0`).  Only the float arithmetic of the energy (lse table / gpu models) and
// the RNG are shared with the GPU code, so that trajectories can be compared bit-for-bit.
#pragma once
#include <vector>
#include <algorithm>
#include "common.h"

struct RefTree {
  int m = 0;                              // leaves 0..m-1, internal m..2m-2
  std::vector<int> left, right;           // -1 for leaves
  std::vector<std::vector<int>> labels;   // output labels of every node (unsliced)
  std::vector<char> sliced;               // sliced[l] = 1 -> label l has log2 size 0
  bool isleaf(int v) const { return left[v] < 0; }
};

static inline bool has(const std::vector<int>& s, int l) { return std::find(s.begin(), s.end(), l) != s.end(); }

// log2 size of a label set (sliced labels have size 0)
static inline int ref_size(const std::vector<int>& s, const std::vector<char>& sl) {
  int n = 0;
  for (int l : s) n += !sl[l];
  return n;
}

// Julia `tcscrw` without rw:  tc = |l3| + |{l in l1 : l in l2 && l notin l3}|, sc = |l3|  (log2 sizes)
static inline void ref_tcsc(const std::vector<int>& l1, const std::vector<int>& l2, const std::vector<int>& l3,
                            const std::vector<char>& sl, int& tc, int& sc) {
  sc = ref_size(l3, sl);
  tc = sc;
  for (int l : l1)
    if (!sl[l] && has(l2, l) && !has(l3, l)) tc += 1;
}

// GPU-model cost of (x,y)->z from vector label sets (MODE 1: model v1, MODE 2: measured model v2).
template <int MODE>
static inline float ref_gpu_cost(const std::vector<int>& x, const std::vector<int>& y, const std::vector<int>& z,
                                 const std::vector<char>& sl, const Params& P) {
  int B = 0, M = 0, N = 0, K = 0;
  for (int l : z) {
    if (sl[l]) continue;
    bool ix = has(x, l), iy = has(y, l);
    if (ix && iy) B++;
    else if (ix) M++;
    else if (iy) N++;
  }
  for (int l : x)
    if (!sl[l] && has(y, l) && !has(z, l)) K++;
  return MODE == 1 ? gpu_lt(B, M, N, K, P) : gpu2_lt(B, M, N, K, P);
}

// Julia `abcacb(a, b, ab, c, d)`: returns dE pieces; `ac` = new labels of the (a,c) node.
template <int MODE>
static inline void ref_abcacb(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& ab,
                              const std::vector<int>& c, const std::vector<int>& d, const std::vector<char>& sl,
                              const Params& P, float& dtc, int& sc0, int& sc1, std::vector<int>& ac) {
  ac.clear();
  for (int l : a)
    if (has(b, l) || has(d, l)) ac.push_back(l);
  for (int l : c)
    if (!has(a, l) && (has(b, l) || has(d, l))) ac.push_back(l);
  if (MODE == 0) {
    int tcl, scl, tcr, scr;
    ref_tcsc(a, b, ab, sl, tcl, scl);
    ref_tcsc(ab, c, d, sl, tcr, scr);
    float tc0 = lse_i(tcl, tcr);
    sc0 = std::max(scl, scr);
    ref_tcsc(a, c, ac, sl, tcl, scl);
    ref_tcsc(ac, b, d, sl, tcr, scr);
    float tc1 = lse_i(tcl, tcr);
    sc1 = std::max(scl, scr);
    dtc = tc1 - tc0;
  } else {
    float t0 = lse_f(ref_gpu_cost<MODE>(a, b, ab, sl, P), ref_gpu_cost<MODE>(ab, c, d, sl, P));
    float t1 = lse_f(ref_gpu_cost<MODE>(a, c, ac, sl, P), ref_gpu_cost<MODE>(ac, b, d, sl, P));
    sc0 = std::max(ref_size(ab, sl), ref_size(d, sl));
    sc1 = std::max(ref_size(ac, sl), ref_size(d, sl));
    dtc = t1 - t0;
  }
}

// Julia `optimize_subtree!(tree, β, ...)` for one node, then recursion into the (new) children.
template <int MODE>
static void ref_optimize_subtree(RefTree& T, int v, float beta, uint32_t sweep, uint32_t chain, const Params& P,
                                 long long& nprop, long long& nacc) {
  if (T.isleaf(v)) return;
  int L = T.left[v], R = T.right[v];
  bool li = !T.isleaf(L), ri = !T.isleaf(R);
  if (!li && !ri) return;  // empty ruleset -> return without recursing
  U4 rnd = philox4x32(U4{(uint32_t)v, sweep, chain, 0u}, P.key0, P.key1);
  int rule = pick_rule(li, ri, rnd.x) + 1;  // Julia numbering 1..4
  const std::vector<int>* a;
  const std::vector<int>* b;
  const std::vector<int>* ab;
  const std::vector<int>* c;
  if (rule == 1) { a = &T.labels[T.left[L]];  b = &T.labels[T.right[L]]; ab = &T.labels[L]; c = &T.labels[R]; }
  else if (rule == 2) { a = &T.labels[T.right[L]]; b = &T.labels[T.left[L]]; ab = &T.labels[L]; c = &T.labels[R]; }
  else if (rule == 3) { a = &T.labels[T.right[R]]; b = &T.labels[T.left[R]]; ab = &T.labels[R]; c = &T.labels[L]; }
  else { a = &T.labels[T.left[R]]; b = &T.labels[T.right[R]]; ab = &T.labels[R]; c = &T.labels[L]; }
  float dtc;
  int sc0, sc1;
  std::vector<int> subout;
  ref_abcacb<MODE>(*a, *b, *ab, *c, T.labels[v], T.sliced, P, dtc, sc0, sc1, subout);
  int dsc = sc1 - sc0;
  // Julia: sc = _sc(tree, rule) = max(|labels(tree)|, |labels(pivot child)|)
  int sc = std::max(ref_size(T.labels[v], T.sliced), ref_size(T.labels[(rule <= 2) ? L : R], T.sliced));
  float w = ((float)std::max(sc, sc + dsc) > P.sc_target) ? P.sc_weight : 0.0f;
  float dE = w * (float)dsc + dtc;
  nprop++;
  if (metropolis(rnd.y, beta, dE)) {  // Julia `update_tree!`
    nacc++;
    if (rule == 1) { int bb = T.right[L], cc = R; T.right[L] = cc; T.right[v] = bb; T.labels[L] = subout; }
    else if (rule == 2) { int aa = T.left[L], cc = R; T.left[L] = cc; T.right[v] = aa; T.labels[L] = subout; }
    else if (rule == 3) { int aa = L, bb = T.left[R]; T.left[v] = bb; T.left[R] = aa; T.labels[R] = subout; }
    else { int aa = L, cc = T.right[R]; T.left[v] = cc; T.right[R] = aa; T.labels[R] = subout; }
  }
  int nl = T.left[v], nr = T.right[v];  // siblings(tree) after the update
  ref_optimize_subtree<MODE>(T, nl, beta, sweep, chain, P, nprop, nacc);
  ref_optimize_subtree<MODE>(T, nr, beta, sweep, chain, P, nprop, nacc);
}
