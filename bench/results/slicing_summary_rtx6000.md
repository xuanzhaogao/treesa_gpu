## Slicing comparison (per-slice tc / sc / #sliced / total tc / modeled total H100 time / wall)

Modeled total time = tree_h100_time(tree on sliced ixs) × 2^#sliced.  Best modeled time per row in bold.

| instance | sc_target | method | #sliced | tc/slice | sc | total tc | modeled time (s) | wall (s) |
|---|---|---|---|---|---|---|---|---|
| sat60 | 30 | existing trees/ | 13 | 32.44 | 29 | 45.44 | 9.64 | - |
| sat60 | 30 | omeco TreeSA + slice_code | 10 | 33.86 | 30 | 43.86 | 1.8 | 2.5 |
| sat60 | 30 | GPU tc → greedy_slice | 10 | 33.71 | 30 | 43.71 | 2.39 | 1.2 |
| sat60 | 30 | GPU tc+h100 → greedy_slice | 10 | 34.67 | 30 | 44.67 | 1.2 | 1.4 |
| sat60 | 30 | GPU tc, in-annealer slicing | 9 | 33.77 | 30 | 42.77 | 0.781 | 0.5 |
| sat60 | 30 | GPU tc+h100, in-annealer slicing | 9 | 33.83 | 30 | 42.83 | **0.514** | 1.0 |
| sat60 | 34 | existing trees/ | 8 | 36.70 | 34 | 44.70 | 3.12 | - |
| sat60 | 34 | omeco TreeSA + slice_code | 8 | 37.03 | 34 | 45.03 | 4.66 | 2.4 |
| sat60 | 34 | GPU tc → greedy_slice | 7 | 37.61 | 34 | 44.61 | 2.32 | 1.0 |
| sat60 | 34 | GPU tc+h100 → greedy_slice | 6 | 39.35 | 34 | 45.35 | 1.83 | 1.3 |
| sat60 | 34 | GPU tc, in-annealer slicing | 5 | 37.79 | 34 | 42.79 | 0.82 | 0.5 |
| sat60 | 34 | GPU tc+h100, in-annealer slicing | 5 | 38.88 | 34 | 43.88 | **0.685** | 1.1 |
| sat60 | 36 | existing trees/ | 4 | 40.04 | 36 | 44.04 | 1.68 | - |
| sat60 | 36 | omeco TreeSA + slice_code | 5 | 39.61 | 36 | 44.61 | 2.32 | 2.1 |
| sat60 | 36 | GPU tc → greedy_slice | 4 | 39.67 | 36 | 43.67 | 1.33 | 0.8 |
| sat60 | 36 | GPU tc+h100 → greedy_slice | 4 | 41.24 | 36 | 45.24 | 0.784 | 1.1 |
| sat60 | 36 | GPU tc, in-annealer slicing | 6 | 36.79 | 33 | 42.79 | 0.83 | 0.5 |
| sat60 | 36 | GPU tc+h100, in-annealer slicing | 3 | 40.74 | 36 | 43.74 | **0.453** | 1.1 |
| sat60 | 37 | omeco TreeSA + slice_code | 4 | 40.67 | 37 | 44.67 | 3.35 | 2.1 |
| sat60 | 37 | GPU tc → greedy_slice | 3 | 40.66 | 37 | 43.66 | 1.31 | 0.8 |
| sat60 | 37 | GPU tc+h100 → greedy_slice | 2 | 43.18 | 37 | 45.18 | 1.36 | 0.9 |
| sat60 | 37 | GPU tc, in-annealer slicing | 5 | 37.81 | 34 | 42.81 | 0.83 | 0.5 |
| sat60 | 37 | GPU tc+h100, in-annealer slicing | 2 | 43.08 | 37 | 45.08 | **0.39** | 1.1 |
| sat70 | 30 | existing trees/ | 19 | 33.03 | 30 | 52.03 | 714 | - |
| sat70 | 30 | omeco TreeSA + slice_code | 19 | 33.06 | 30 | 52.06 | 744 | 3.9 |
| sat70 | 30 | GPU tc → greedy_slice | 17 | 33.82 | 30 | 50.82 | 225 | 1.9 |
| sat70 | 30 | GPU tc+h100 → greedy_slice | 15 | 36.21 | 30 | 51.21 | 45.7 | 2.0 |
| sat70 | 30 | GPU tc, in-annealer slicing | 18 | 30.80 | 27 | 48.80 | 118 | 0.5 |
| sat70 | 30 | GPU tc+h100, in-annealer slicing | 15 | 34.55 | 30 | 49.55 | **33.7** | 1.3 |
| sat70 | 34 | existing trees/ | 15 | 36.36 | 33 | 51.36 | 507 | - |
| sat70 | 34 | omeco TreeSA + slice_code | 14 | 37.44 | 34 | 51.44 | 278 | 3.3 |
| sat70 | 34 | GPU tc → greedy_slice | 14 | 36.86 | 34 | 50.86 | 213 | 1.7 |
| sat70 | 34 | GPU tc+h100 → greedy_slice | 12 | 40.14 | 34 | 52.14 | 60.2 | 1.9 |
| sat70 | 34 | GPU tc, in-annealer slicing | 12 | 36.88 | 34 | 48.88 | 57.2 | 0.5 |
| sat70 | 34 | GPU tc+h100, in-annealer slicing | 11 | 39.19 | 34 | 50.19 | **29.3** | 1.3 |
| sat70 | 36 | existing trees/ | 13 | 40.27 | 36 | 53.27 | 711 | - |
| sat70 | 36 | omeco TreeSA + slice_code | 12 | 38.78 | 36 | 50.78 | 205 | 3.1 |
| sat70 | 36 | GPU tc → greedy_slice | 10 | 39.93 | 36 | 49.93 | 136 | 1.4 |
| sat70 | 36 | GPU tc+h100 → greedy_slice | 10 | 41.23 | 36 | 51.23 | 57.6 | 1.7 |
| sat70 | 36 | GPU tc, in-annealer slicing | 13 | 35.80 | 33 | 48.80 | 52 | 0.5 |
| sat70 | 36 | GPU tc+h100, in-annealer slicing | 9 | 40.37 | 36 | 49.37 | **23.5** | 1.3 |
| sat70 | 37 | omeco TreeSA + slice_code | 12 | 40.63 | 37 | 52.63 | 339 | 3.1 |
| sat70 | 37 | GPU tc → greedy_slice | 9 | 40.57 | 37 | 49.57 | 89.3 | 1.3 |
| sat70 | 37 | GPU tc+h100 → greedy_slice | 8 | 44.12 | 37 | 52.12 | 51.2 | 1.6 |
| sat70 | 37 | GPU tc, in-annealer slicing | 9 | 39.64 | 36 | 48.64 | 38.6 | 0.5 |
| sat70 | 37 | GPU tc+h100, in-annealer slicing | 8 | 43.07 | 37 | 51.07 | **25.6** | 1.3 |
| sat80 | 30 | omeco TreeSA + slice_code | 24 | 33.14 | 30 | 57.14 | 2.56e+04 | 4.9 |
| sat80 | 30 | GPU tc → greedy_slice | 23 | 33.07 | 30 | 56.07 | 1.01e+04 | 2.7 |
| sat80 | 30 | GPU tc+h100 → greedy_slice | 22 | 36.15 | 30 | 58.15 | 7.06e+03 | 3.1 |
| sat80 | 30 | GPU tc, in-annealer slicing | 21 | 33.61 | 30 | 54.61 | 5.62e+03 | 0.6 |
| sat80 | 30 | GPU tc+h100, in-annealer slicing | 21 | 33.88 | 30 | 54.88 | **2.87e+03** | 1.3 |
| sat80 | 34 | omeco TreeSA + slice_code | 20 | 37.55 | 34 | 57.55 | 1.78e+04 | 4.7 |
| sat80 | 34 | GPU tc → greedy_slice | 18 | 37.89 | 34 | 55.89 | 6.44e+03 | 2.2 |
| sat80 | 34 | GPU tc+h100 → greedy_slice | 17 | 42.03 | 34 | 59.03 | 2.53e+03 | 2.6 |
| sat80 | 34 | GPU tc, in-annealer slicing | 17 | 37.62 | 34 | 54.62 | 3.31e+03 | 0.6 |
| sat80 | 34 | GPU tc+h100, in-annealer slicing | 17 | 38.16 | 34 | 55.16 | **1.7e+03** | 1.4 |
| sat80 | 36 | omeco TreeSA + slice_code | 19 | 39.71 | 36 | 58.71 | 3.55e+04 | 4.3 |
| sat80 | 36 | GPU tc → greedy_slice | 17 | 40.10 | 36 | 57.10 | 1.53e+04 | 2.2 |
| sat80 | 36 | GPU tc+h100 → greedy_slice | 16 | 42.14 | 36 | 58.14 | 3.27e+03 | 2.5 |
| sat80 | 36 | GPU tc, in-annealer slicing | 22 | 32.61 | 29 | 54.61 | 6.23e+03 | 0.6 |
| sat80 | 36 | GPU tc+h100, in-annealer slicing | 14 | 42.16 | 36 | 56.16 | **1.24e+03** | 1.4 |
| sat80 | 37 | omeco TreeSA + slice_code | 17 | 41.23 | 37 | 58.23 | 1.85e+04 | 4.1 |
| sat80 | 37 | GPU tc → greedy_slice | 17 | 39.63 | 37 | 56.63 | 1.64e+04 | 2.1 |
| sat80 | 37 | GPU tc+h100 → greedy_slice | 17 | 39.64 | 37 | 56.64 | 7.81e+03 | 2.5 |
| sat80 | 37 | GPU tc, in-annealer slicing | 31 | 23.59 | 20 | 54.59 | 3.94e+05 | 0.5 |
| sat80 | 37 | GPU tc+h100, in-annealer slicing | 14 | 41.64 | 37 | 55.64 | **1.52e+03** | 1.4 |
| grid22 | 30 | omeco TreeSA + slice_code | 4 | 41.31 | 30 | 45.31 | 0.109 | 10.3 |
| grid22 | 30 | GPU tc → greedy_slice | 4 | 41.27 | 30 | 45.27 | 0.197 | 6.8 |
| grid22 | 30 | GPU tc+h100 → greedy_slice | 3 | 42.89 | 30 | 45.89 | 0.03 | 10.9 |
| grid22 | 30 | GPU tc, in-annealer slicing | 8 | 38.27 | 30 | 46.27 | 1.83 | 7.4 |
| grid22 | 30 | GPU tc+h100, in-annealer slicing | 3 | 42.89 | 30 | 45.89 | **0.0285** | 15.4 |
| grid22 | 34 | omeco TreeSA + slice_code | 0 | 43.55 | 34 | 43.55 | 0.0999 | 9.3 |
| grid22 | 34 | GPU tc → greedy_slice | 0 | 45.15 | 34 | 45.15 | 0.0857 | 5.2 |
| grid22 | 34 | GPU tc+h100 → greedy_slice | 0 | 45.57 | 33 | 45.57 | **0.014** | 9.9 |
| grid22 | 34 | GPU tc, in-annealer slicing | 4 | 41.94 | 34 | 45.94 | 1.44 | 7.4 |
| grid22 | 34 | GPU tc+h100, in-annealer slicing | 0 | 45.78 | 33 | 45.78 | 0.0149 | 15.3 |
| grid22 | 36 | omeco TreeSA + slice_code | 0 | 43.23 | 34 | 43.23 | 0.171 | 9.1 |
| grid22 | 36 | GPU tc → greedy_slice | 1 | 44.48 | 36 | 45.48 | 0.602 | 5.4 |
| grid22 | 36 | GPU tc+h100 → greedy_slice | 0 | 45.60 | 33 | 45.60 | **0.0137** | 9.6 |
| grid22 | 36 | GPU tc, in-annealer slicing | 2 | 44.20 | 36 | 46.20 | 0.998 | 7.4 |
| grid22 | 36 | GPU tc+h100, in-annealer slicing | 0 | 45.74 | 33 | 45.74 | 0.0142 | 15.2 |
| grid22 | 37 | omeco TreeSA + slice_code | 0 | 43.36 | 36 | 43.36 | 0.247 | 9.3 |
| grid22 | 37 | GPU tc → greedy_slice | 0 | 45.28 | 37 | 45.28 | 0.563 | 5.0 |
| grid22 | 37 | GPU tc+h100 → greedy_slice | 0 | 45.57 | 33 | 45.57 | **0.0137** | 9.7 |
| grid22 | 37 | GPU tc, in-annealer slicing | 1 | 44.36 | 37 | 45.36 | 0.826 | 7.3 |
| grid22 | 37 | GPU tc+h100, in-annealer slicing | 0 | 45.67 | 33 | 45.67 | 0.0143 | 15.2 |
| grid24 | 30 | omeco TreeSA + slice_code | 14 | 41.91 | 30 | 55.91 | 234 | 14.1 |
| grid24 | 30 | GPU tc → greedy_slice | 15 | 40.83 | 30 | 55.83 | 402 | 16.7 |
| grid24 | 30 | GPU tc+h100 → greedy_slice | 11 | 44.72 | 30 | 55.72 | **15** | 21.6 |
| grid24 | 30 | GPU tc, in-annealer slicing | 13 | 40.10 | 30 | 53.10 | 126 | 12.6 |
| grid24 | 30 | GPU tc+h100, in-annealer slicing | 12 | 42.88 | 30 | 54.88 | 20.9 | 25.1 |
| grid24 | 34 | omeco TreeSA + slice_code | 5 | 48.34 | 34 | 53.34 | 5.94 | 13.5 |
| grid24 | 34 | GPU tc → greedy_slice | 3 | 49.70 | 34 | 52.70 | 1.73 | 10.3 |
| grid24 | 34 | GPU tc+h100 → greedy_slice | 2 | 50.67 | 34 | 52.67 | 0.8 | 17.1 |
| grid24 | 34 | GPU tc, in-annealer slicing | 8 | 42.43 | 34 | 50.43 | 11.1 | 12.3 |
| grid24 | 34 | GPU tc+h100, in-annealer slicing | 2 | 50.29 | 34 | 52.29 | **0.7** | 24.8 |

## Modeled time relative to the best method (geometric mean over instances × targets)

| method | geo-mean (modeled time / best) | max | #best |
|---|---|---|---|
| omeco TreeSA + slice_code | 9.78 | 28.68 | 0 |
| GPU tc → greedy_slice | 6.93 | 43.86 | 0 |
| GPU tc+h100 → greedy_slice | 1.78 | 5.13 | 4 |
| GPU tc, in-annealer slicing | 7.26 | 259.03 | 0 |
| GPU tc+h100, in-annealer slicing | 1.03 | 1.39 | 14 |

Kernel-internal (per-slice tc, sc, #sliced, modeled total) == tree_utils recomputation: 36/36 runs.  Greedy repairs needed: 0 runs.
