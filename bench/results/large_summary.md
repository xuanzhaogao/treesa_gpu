# Large networks: GPU TreeSA (Quadro RTX 6000) vs omeco (12 threads, workergpu165 CPU)

Cells: tc (log2) / sc / wall s; pure-tc objective (sc_target 10000); single runs, GPU seed 1. Best tc per instance in bold. GPU wall includes process start and greedy init. GPU: original kernels; GPU v2: new kernels (same trees). †: intermediate build of the new kernels (16 warps per chain).

| method | sq32 (m=1024, 1984 labels) | sq48 (m=2304, 4512 labels) | sq64 (m=4096, 8064 labels) | reg3_1000 (m=1000, 1500 labels) | grid32 (m=3072, 1024 labels) | sat500 (m=2130, 500 labels) | grid45 (m=6075, 2025 labels) | reg3_2000 (m=2000, 3000 labels) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| row-major sweep | 44.9 / 33 / 0 | 62.1 / 49 / 0 | 78.9 / 65 / 0 | – | – | – | – | – |
| omeco greedy | 61.0 / 47 / 0 | 88.5 / 63 / 0 | 117.0 / 81 / 0 | 285.0 / 221 / 0 | 115.0 / 85 / 0 | 493.0 / 481 / 1 | 160.0 / 111 / 0 | 604.0 / 446 / 0 |
| omeco TreeSA 10x50 | 46.1 / 37 / 8 | 66.2 / 58 / 25 | 91.1 / 74 / 56 | 134.8 / 128 / 11 | 61.6 / 55 / 21 | 345.2 / 341 / 21 | 103.8 / 98 / 47 | 276.4 / 270 / 39 |
| omeco TreeSA 10x200 | 47.0 / 39 / 36 | 63.4 / 50 / 102 | 93.4 / 84 / 249 | 132.1 / 125 / 42 | **60.4** / 53 / 85 | 327.7 / 324 / 82 | 90.7 / 83 / 187 | 268.1 / 262 / 146 |
| GPU pa 576x5 | 46.6 / 38 / 5 | 75.0 / 56 / 26 | 100.8 / 81 / 105 | 149.5 / 143 / 4 | 69.2 / 56 / 5 | 409.1 / 404 / 4 | 103.0 / 86 / 27 | 328.8 / 323 / 13 |
| GPU indep 576x5 | 45.8 / 34 / 6 | 75.0 / 58 / 28 | 96.2 / 88 / 116 | 142.6 / 136 / 4 | 65.1 / 58 / 5 | 417.7 / 414 / 4 | 99.0 / 86 / 26 | 324.9 / 320 / 15 |
| GPU pa 576x20 | 45.0 / 33 / 18 | 75.0 / 58 / 97 | 101.0 / 79 / 412 | 139.1 / 132 / 13 | 65.1 / 51 / 16 | 357.4 / 352 / 11 | 96.6 / 83 / 91 | 290.5 / 284 / 48 |
| GPU indep 576x20 | 44.9 / 33 / 24 | 65.2 / 52 / 111 | 91.2 / 77 / 464 | 135.3 / 128 / 13 | 63.4 / 54 / 18 | 377.5 / 374 / 12 | 93.2 / 79 / 96 | 278.4 / 272 / 52 |
| GPU indep 72x50 | 45.6 / 34 / 42 | 66.4 / 53 / 216 | 89.0 / 75 / 801 | 131.3 / 125 / 24 | 61.5 / 54 / 25 | 341.0 / 336 / 6 | 95.1 / 83 / 92 | 270.9 / 264 / 85 |
| GPU indep 144x50 | 45.3 / 36 / 46 | 66.4 / 53 / 222 | 88.9 / 80 / 853 | 131.3 / 125 / 25 | 61.5 / 54 / 27 | 339.6 / 335 / 7 | 94.9 / 87 / 97 | 270.9 / 264 / 89 |
| GPU indep 72x150 | 45.2 / 34 / 142 | – | – | 130.4 / 123 / 71 | 62.3 / 54 / 82 | 329.8 / 326 / 17 | 90.9 / 83 / 303 | 261.1 / 253 / 262 |
| GPU indep 576x5 from sweep, beta0 5 | 44.3 / 33 / 5 | 61.7 / 49 / 24 | 78.7 / 65 / 89 | – | – | – | – | – |
| GPU indep 72x20 from sweep, beta0 5 | **44.2** / 33 / 13 | **61.7** / 49 / 67 | – | – | – | – | – | – |
| GPU v2 indep 72x50 | 45.6 / 34 / 24 | 66.4 / 53 / 47 | 89.0 / 75 / 87 | 131.3 / 125 / 17 | 61.5 / 54 / 20 | 341.0 / 336 / 5 | 95.1 / 83 / 42 | 270.9 / 264 / 37 |
| GPU v2 indep 576x20 | 44.9 / 33 / 13 | 65.2 / 52 / 34 | 91.2 / 77 / 111 | 135.3 / 128 / 9 | 63.4 / 54 / 16 | 377.5 / 374 / 12 | 93.2 / 79 / 88 | 278.4 / 272 / 23 |
| GPU v2 indep 72x150 | 45.2 / 34 / 81 | 66.3 / 51 / 167 | 86.7 / 76 / 284 | 130.4 / 123 / 49 | 62.3 / 54 / 67 | 329.8 / 326 / 16 | 90.9 / 83 / 140 | 261.1 / 253 / 112 |
| GPU v2 indep 576x5 from sweep, beta0 5 | 44.3 / 33 / 3 | 61.7 / 49 / 6 | 78.7 / 65 / 17 | – | – | – | – | – |
| GPU v2 indep 72x20 from sweep, beta0 5 | **44.2** / 33 / 8 | **61.7** / 49 / 13 | **78.6** / 65 / 24 | – | – | – | – | – |
| GPU v2 indep 72x500† | 44.6 / 35 / 289 | 68.5 / 58 / 627 | 85.2 / 74 / 1210 | **127.6** / 120 / 179 | 60.7 / 52 / 231 | **325.3** / 320 / 83 | **85.0** / 77 / 596 | **255.3** / 248 / 412 |

## Same runs on an H200 (workergpu318) vs the RTX 6000

| instance | config | tc RTX / H200 | wall RTX 6000 | wall H200 | speed-up |
|---|---|---|---:|---:|---:|
| grid32 | indep 72x50 | 61.51 / 61.51 | 25 s | 23 s | 1.11x |
| grid45 | indep 72x50 | 95.06 / 95.06 | 92 s | 92 s | 1.00x |
| reg3_1000 | indep 72x50 | 131.35 / 131.35 | 24 s | 22 s | 1.12x |
| reg3_2000 | indep 72x50 | 270.88 / 270.88 | 85 s | 83 s | 1.02x |
| sat500 | indep 72x50 | 341.02 / 341.02 | 6 s | 5 s | 1.11x |
| sq32 | indep 72x50 | 45.63 / 45.63 | 42 s | 39 s | 1.07x |
| sq48 | indep 72x50 | 66.38 / 66.38 | 216 s | 210 s | 1.03x |
| sq64 | indep 72x50 | 89.03 / 89.03 | 801 s | 727 s | 1.10x |
| grid32 | indep 72x150 | 62.29 / 62.29 | 82 s | 73 s | 1.12x |
| grid45 | indep 72x150 | 90.88 / 90.88 | 303 s | 302 s | 1.00x |
| reg3_1000 | indep 72x150 | 130.41 / 130.41 | 71 s | 62 s | 1.15x |
| reg3_2000 | indep 72x150 | 261.07 / 261.07 | 262 s | 252 s | 1.04x |
| sat500 | indep 72x150 | 329.79 / 329.79 | 17 s | 13 s | 1.31x |
| sq32 | indep 576x5 from sweep, beta0 5 | 44.29 / 44.29 | 5 s | 7 s | 0.76x |
| sq48 | indep 576x5 from sweep, beta0 5 | 61.73 / 61.73 | 24 s | 24 s | 1.03x |
| sq64 | indep 576x5 from sweep, beta0 5 | 78.67 / 78.67 | 89 s | 84 s | 1.06x |
| grid32 | indep 576x20 | 63.37 / 63.37 | 18 s | 12 s | 1.55x |
| grid45 | indep 576x20 | 93.18 / 93.18 | 96 s | 42 s | 2.29x |
| reg3_1000 | indep 576x20 | 135.26 / 135.26 | 13 s | 12 s | 1.09x |
| reg3_2000 | indep 576x20 | 278.39 / 278.39 | 52 s | 42 s | 1.24x |
| sat500 | indep 576x20 | 377.54 / 377.54 | 12 s | 3 s | 3.72x |
| sq32 | indep 576x20 | 44.91 / 44.91 | 24 s | 21 s | 1.15x |
| sq48 | indep 576x20 | 65.21 / 65.21 | 111 s | 91 s | 1.21x |
| sq64 | indep 576x20 | 91.20 / 91.20 | 464 s | 286 s | 1.62x |

## New kernels vs the original ones (RTX 6000, same configurations and seeds)

Cooperative visits (G lanes per node when fewer than 32 nodes are ready; above 512 labels), chunk summaries (>= 128 words) and one thread block of up to 8 warps per chain (auto when the chains do not fill the GPU).  The trees must be identical.

| instance | config | kernel | tc old / new | wall old | wall new | speed-up | kernel s old / new |
|---|---|---|---|---:|---:|---:|---:|
| sq32 | indep 72x20 from sweep, beta0 5 | block | 44.24 / 44.24 | 13 s | 8 s | 1.67x | 12.4 / 7.3 |
| sq48 | indep 72x20 from sweep, beta0 5 | block | 61.69 / 61.69 | 67 s | 13 s | 5.06x | 65.9 / 12.3 |
| grid32 | indep 72x50 | block | 61.51 / 61.51 | 25 s | 20 s | 1.23x | 24.4 / 19.7 |
| grid45 | indep 72x50 | block | 95.06 / 95.06 | 92 s | 42 s | 2.21x | 90.0 / 39.8 |
| reg3_1000 | indep 72x50 | block | 131.35 / 131.35 | 24 s | 17 s | 1.44x | 23.5 / 16.2 |
| reg3_2000 | indep 72x50 | block | 270.88 / 270.88 | 85 s | 37 s | 2.30x | 84.0 / 35.9 |
| sat500 | indep 72x50 | block | 341.02 / 341.02 | 6 s | 5 s | 1.12x | 4.7 / 4.2 |
| sq32 | indep 72x50 | block | 45.63 / 45.63 | 42 s | 24 s | 1.75x | 41.3 / 23.3 |
| sq48 | indep 72x50 | block | 66.38 / 66.38 | 216 s | 47 s | 4.63x | 214.6 / 45.4 |
| sq64 | indep 72x50 | block | 89.03 / 89.03 | 801 s | 87 s | 9.17x | 797.8 / 84.5 |
| grid32 | indep 72x150 | block | 62.29 / 62.29 | 82 s | 67 s | 1.22x | 81.0 / 66.2 |
| grid45 | indep 72x150 | block | 90.88 / 90.88 | 303 s | 140 s | 2.17x | 301.2 / 138.2 |
| reg3_1000 | indep 72x150 | block | 130.41 / 130.41 | 71 s | 49 s | 1.45x | 70.6 / 48.6 |
| reg3_2000 | indep 72x150 | block | 261.07 / 261.07 | 262 s | 112 s | 2.33x | 260.8 / 111.4 |
| sat500 | indep 72x150 | block | 329.79 / 329.79 | 17 s | 16 s | 1.02x | 15.6 / 15.2 |
| sq32 | indep 72x150 | block | 45.23 / 45.23 | 142 s | 81 s | 1.76x | 141.3 / 80.2 |
| sq32 | indep 576x5 from sweep, beta0 5 | block | 44.29 / 44.29 | 5 s | 3 s | 1.75x | 4.1 / 2.1 |
| sq48 | indep 576x5 from sweep, beta0 5 | block | 61.73 / 61.73 | 24 s | 6 s | 3.94x | 21.9 / 3.8 |
| sq64 | indep 576x5 from sweep, beta0 5 | block | 78.67 / 78.67 | 89 s | 17 s | 5.09x | 83.3 / 12.1 |
| grid32 | indep 576x20 | block | 63.37 / 63.37 | 18 s | 16 s | 1.15x | 16.7 / 14.4 |
| grid45 | indep 576x20 | block | 93.18 / 93.18 | 96 s | 88 s | 1.10x | 91.7 / 83.6 |
| reg3_1000 | indep 576x20 | block | 135.26 / 135.26 | 13 s | 9 s | 1.44x | 12.7 / 8.6 |
| reg3_2000 | indep 576x20 | block | 278.39 / 278.39 | 52 s | 23 s | 2.20x | 49.4 / 21.2 |
| sat500 | indep 576x20 | block | 377.54 / 377.54 | 12 s | 12 s | 0.98x | 10.3 / 10.8 |
| sq32 | indep 576x20 | block | 44.91 / 44.91 | 24 s | 13 s | 1.88x | 22.8 / 11.7 |
| sq48 | indep 576x20 | block | 65.21 / 65.21 | 111 s | 34 s | 3.27x | 107.2 / 30.3 |
| sq64 | indep 576x20 | block | 91.20 / 91.20 | 464 s | 111 s | 4.18x | 450.5 / 97.8 |

## H100 PCIe (workergpu165): original vs new kernels (intermediate build, 16 warps per chain)

| instance | config | tc | H100 old | H100 new | speed-up | RTX 6000 new |
|---|---|---|---:|---:|---:|---:|
| sq32 | indep 72x20 from sweep, beta0 5 | 44.24 | – | 5 s | – | 8 s |
| sq48 | indep 72x20 from sweep, beta0 5 | 61.69 | – | 16 s | – | 13 s |
| sq64 | indep 72x20 from sweep, beta0 5 | 78.65 | – | 31 s | – | 24 s |
| grid32 | indep 72x50 | 61.51 | 25 s | 15 s | 1.62x | 20 s |
| grid45 | indep 72x50 | 95.06 | 101 s | 30 s | 3.34x | 42 s |
| reg3_1000 | indep 72x50 | 131.35 | 24 s | 13 s | 1.86x | 17 s |
| reg3_2000 | indep 72x50 | 270.88 | 92 s | 28 s | 3.30x | 37 s |
| sat500 | indep 72x50 | 341.02 | 5 s | 6 s | 0.82x | 5 s |
| sq32 | indep 72x50 | 45.63 | 43 s | 16 s | 2.70x | 24 s |
| sq48 | indep 72x50 | 66.38 | 237 s | 55 s | 4.32x | 47 s |
| sq64 | indep 72x50 | 89.03 | 821 s | 103 s | 7.99x | 87 s |
| sq32 | indep 72x100 from sweep, beta0 5 | 44.12 | – | 25 s | – | – |
| sq48 | indep 72x100 from sweep, beta0 5 | 61.61 | – | 76 s | – | – |
| sq64 | indep 72x100 from sweep, beta0 5 | 78.58 | – | 150 s | – | – |
| grid32 | indep 72x150 | 62.29 | – | 49 s | – | 67 s |
| grid45 | indep 72x150 | 90.88 | – | 99 s | – | 140 s |
| reg3_1000 | indep 72x150 | 130.41 | – | 36 s | – | 49 s |
| reg3_2000 | indep 72x150 | 261.07 | – | 84 s | – | 112 s |
| sat500 | indep 72x150 | 329.79 | – | 15 s | – | 16 s |
| sq32 | indep 72x150 | 45.23 | – | 52 s | – | 81 s |
| sq48 | indep 72x150 | 66.35 | – | 199 s | – | 167 s |
| sq64 | indep 72x150 | 86.71 | – | 341 s | – | 284 s |
| grid32 | indep 72x500 | 60.75 | – | 153 s | – | 231 s |
| grid45 | indep 72x500 | 84.98 | – | 380 s | – | 596 s |
| reg3_1000 | indep 72x500 | 127.64 | – | 121 s | – | 179 s |
| reg3_2000 | indep 72x500 | 255.30 | – | 287 s | – | 412 s |
| sat500 | indep 72x500 | 325.28 | – | 52 s | – | 83 s |
| sq32 | indep 72x500 | 44.60 | – | 170 s | – | 289 s |
| sq48 | indep 72x500 | 68.48 | – | 663 s | – | 627 s |
| sq64 | indep 72x500 | 85.19 | – | 1312 s | – | 1210 s |
