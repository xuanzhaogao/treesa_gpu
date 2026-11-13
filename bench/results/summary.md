## Headline: omeco CPU TreeSA vs GPU TreeSA (tc / sc / wallclock)

| instance | omeco default, 32 thr | omeco default, 16 thr | omeco 32 trials x niters=200 | GPU indep 576 x 50 | GPU PA 576 x 50 | GPU PA 4608 x 50 | GPU PA 576 x 800 |
|---|---|---|---|---|---|---|---|
| sat60 | 44.56 / 40 / 1.72s | 44.56 / 40 / 1.72s | 44.36 / 41 / 10.26s | 43.62 / 40 / 0.51s | 43.43 / 40 / 0.51s | 43.02 / 40 / 1.48s | 43.06 / 40 / 3.37s |
| sat80 | 59.12 / 54 / 2.41s | 59.12 / 54 / 2.42s | 56.19 / 52 / 14.89s | 56.84 / 53 / 0.60s | 55.40 / 50 / 0.60s | 55.28 / 51 / 2.52s | 54.59 / 51 / 4.33s |
| sat100 | 72.93 / 69 / 3.24s | 72.93 / 69 / 3.16s | 69.42 / 64 / 19.36s | 70.27 / 66 / 0.79s | 69.33 / 65 / 0.83s | 66.50 / 63 / 3.65s | 66.82 / 63 / 7.32s |
| sat120 | 94.17 / 89 / 3.91s | 94.17 / 89 / 3.94s | 87.34 / 80 / 24.77s | 87.33 / 82 / 0.89s | 83.92 / 79 / 0.91s | 81.46 / 77 / 4.70s | 81.83 / 77 / 8.62s |
| lat14 | 28.91 / 22 / 4.08s | 28.91 / 22 / 4.09s | 28.71 / 23 / 24.59s | 28.71 / 23 / 3.41s | 29.09 / 23 / 3.22s | 29.09 / 23 / 36.31s | 28.70 / 23 / 49.27s |
| reg3_100 | 17.16 / 11 / 0.52s | 17.16 / 11 / 0.51s | 17.16 / 11 / 3.44s | 17.16 / 11 / 0.62s | 17.16 / 11 / 0.62s | 17.16 / 11 / 2.88s | 17.16 / 11 / 5.31s |
| reg3_250 | 47.00 / 33 / 1.54s | 47.00 / 33 / 1.44s | 46.05 / 32 / 9.25s | 40.30 / 34 / 1.87s | 40.19 / 34 / 2.00s | 40.11 / 34 / 11.32s | 40.40 / 33 / 24.34s |

## Matched wallclock: best tc (sc, config) reached within the wallclock of an omeco run

Budget = end-to-end wallclock of the omeco run; GPU rows use seed 1 only (no best-of-seeds).

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

## Population schemes at equal work (576 chains, 300 levels; tc mean ± sd / best over seeds 1-3)

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

## Quality vs wallclock (best tc reached by each configuration, sorted by wallclock)


### sat60

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.38 | GPU pa 576x niters=5 | 45.27 | 41 |
| 0.41 | GPU trunc 576x niters=5 | 46.78 | 44 |
| 0.43 | GPU trunc 576x niters=20 | 44.39 | 41 |
| 0.45 | omeco ntrials=32 niters=5 (32 thr) | 49.43 | 46 |
| 0.46 | GPU pa 576x niters=20 | 43.70 | 40 |
| 0.46 | GPU indep 576x niters=20 | 44.65 | 41 |
| 0.47 | GPU indep 4608x niters=5 | 47.89 | 44 |
| 0.47 | GPU indep 576x niters=5 | 48.30 | 43 |
| 0.51 | GPU pa 4608x niters=5 | 44.54 | 41 |
| 0.51 | GPU indep 576x niters=50 | 43.62 | 40 |
| 0.51 | GPU pa 576x niters=50 | 43.43 | 40 |
| 0.53 | GPU trunc 576x niters=50 | 43.50 | 40 |
| 0.78 | GPU indep 4608x niters=20 | 44.01 | 41 |
| 0.85 | GPU pa 4608x niters=20 | 43.14 | 40 |
| 1.10 | GPU pa 576x niters=200 | 42.78 | 39 |
| 1.10 | GPU trunc 576x niters=200 | 45.06 | 38 |
| 1.11 | GPU indep 576x niters=200 | 43.06 | 40 |
| 1.20 | omeco ntrials=32 niters=20 (32 thr) | 46.50 | 42 |
| 1.48 | GPU pa 4608x niters=50 | 43.02 | 40 |
| 1.48 | GPU indep 4608x niters=50 | 43.30 | 40 |
| 1.70 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 44.56 | 40 |
| 1.72 | omeco ntrials=10 niters=50 (32 thr) | 44.56 | 40 |
| 1.72 | omeco ntrials=10 niters=50 (16 thr) | 44.56 | 40 |
| 2.67 | omeco ntrials=32 niters=50 (32 thr) | 44.56 | 40 |
| 3.29 | GPU trunc 576x niters=800 | 44.20 | 39 |
| 3.37 | GPU pa 576x niters=800 | 43.06 | 40 |
| 3.42 | GPU indep 576x niters=800 | 42.88 | 39 |
| 4.62 | GPU indep 4608x niters=200 | 42.96 | 39 |
| 4.66 | GPU pa 4608x niters=200 | 42.82 | 39 |
| 10.26 | omeco ntrials=32 niters=200 (32 thr) | 44.36 | 41 |

### sat80

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.36 | GPU indep 576x niters=5 | 63.85 | 61 |
| 0.37 | GPU pa 576x niters=5 | 60.41 | 57 |
| 0.38 | GPU trunc 576x niters=5 | 57.63 | 54 |
| 0.44 | GPU indep 576x niters=20 | 57.42 | 54 |
| 0.44 | GPU pa 576x niters=20 | 56.05 | 53 |
| 0.45 | GPU trunc 576x niters=20 | 56.83 | 54 |
| 0.50 | omeco ntrials=32 niters=5 (32 thr) | 67.26 | 63 |
| 0.60 | GPU pa 576x niters=50 | 55.40 | 50 |
| 0.60 | GPU indep 576x niters=50 | 56.84 | 53 |
| 0.63 | GPU trunc 576x niters=50 | 55.92 | 52 |
| 0.64 | GPU indep 4608x niters=5 | 62.95 | 59 |
| 0.73 | GPU pa 4608x niters=5 | 59.21 | 55 |
| 1.23 | GPU indep 4608x niters=20 | 57.42 | 54 |
| 1.31 | GPU pa 4608x niters=20 | 55.83 | 53 |
| 1.35 | GPU pa 576x niters=200 | 54.74 | 51 |
| 1.36 | GPU indep 576x niters=200 | 55.53 | 51 |
| 1.39 | GPU trunc 576x niters=200 | 55.29 | 50 |
| 1.58 | omeco ntrials=32 niters=20 (32 thr) | 60.74 | 57 |
| 2.38 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 59.12 | 54 |
| 2.41 | omeco ntrials=10 niters=50 (32 thr) | 59.12 | 54 |
| 2.42 | omeco ntrials=10 niters=50 (16 thr) | 59.12 | 54 |
| 2.44 | GPU indep 4608x niters=50 | 55.82 | 51 |
| 2.52 | GPU pa 4608x niters=50 | 55.28 | 51 |
| 3.76 | omeco ntrials=32 niters=50 (32 thr) | 58.83 | 54 |
| 4.18 | GPU trunc 576x niters=800 | 55.94 | 52 |
| 4.33 | GPU pa 576x niters=800 | 54.59 | 51 |
| 4.41 | GPU indep 576x niters=800 | 54.95 | 51 |
| 8.41 | GPU indep 4608x niters=200 | 55.09 | 51 |
| 8.49 | GPU pa 4608x niters=200 | 54.71 | 51 |
| 14.89 | omeco ntrials=32 niters=200 (32 thr) | 56.19 | 52 |

### sat100

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.40 | GPU indep 576x niters=5 | 83.08 | 81 |
| 0.40 | GPU pa 576x niters=5 | 76.28 | 72 |
| 0.42 | GPU trunc 576x niters=5 | 73.76 | 70 |
| 0.53 | GPU pa 576x niters=20 | 71.09 | 64 |
| 0.55 | GPU trunc 576x niters=20 | 70.74 | 67 |
| 0.55 | GPU indep 576x niters=20 | 73.19 | 70 |
| 0.66 | omeco ntrials=32 niters=5 (32 thr) | 86.63 | 83 |
| 0.79 | GPU indep 576x niters=50 | 70.27 | 66 |
| 0.80 | GPU indep 4608x niters=5 | 82.22 | 78 |
| 0.81 | GPU trunc 576x niters=50 | 69.14 | 65 |
| 0.83 | GPU pa 576x niters=50 | 69.33 | 65 |
| 0.88 | GPU pa 4608x niters=5 | 75.11 | 70 |
| 1.71 | GPU indep 4608x niters=20 | 72.15 | 68 |
| 1.77 | GPU pa 4608x niters=20 | 69.18 | 65 |
| 2.10 | GPU trunc 576x niters=200 | 67.23 | 62 |
| 2.11 | GPU indep 576x niters=200 | 68.06 | 64 |
| 2.14 | GPU pa 576x niters=200 | 66.51 | 62 |
| 2.15 | omeco ntrials=32 niters=20 (32 thr) | 75.12 | 72 |
| 3.13 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 72.93 | 69 |
| 3.16 | omeco ntrials=10 niters=50 (16 thr) | 72.93 | 69 |
| 3.24 | omeco ntrials=10 niters=50 (32 thr) | 72.93 | 69 |
| 3.53 | GPU indep 4608x niters=50 | 68.91 | 65 |
| 3.65 | GPU pa 4608x niters=50 | 66.50 | 63 |
| 4.96 | omeco ntrials=32 niters=50 (32 thr) | 72.93 | 69 |
| 7.28 | GPU trunc 576x niters=800 | 68.10 | 62 |
| 7.32 | GPU pa 576x niters=800 | 66.82 | 63 |
| 7.44 | GPU indep 576x niters=800 | 67.06 | 62 |
| 12.70 | GPU indep 4608x niters=200 | 67.26 | 62 |
| 12.72 | GPU pa 4608x niters=200 | 67.08 | 63 |
| 19.36 | omeco ntrials=32 niters=200 (32 thr) | 69.42 | 64 |

### sat120

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

### lat14

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.63 | GPU trunc 576x niters=5 | 31.50 | 24 |
| 0.63 | GPU indep 576x niters=5 | 28.80 | 23 |
| 0.64 | GPU pa 576x niters=5 | 29.50 | 23 |
| 0.74 | omeco ntrials=32 niters=5 (32 thr) | 28.87 | 23 |
| 1.43 | GPU trunc 576x niters=20 | 28.70 | 23 |
| 1.55 | GPU pa 576x niters=20 | 28.70 | 23 |
| 1.57 | GPU indep 576x niters=20 | 28.72 | 23 |
| 2.54 | omeco ntrials=32 niters=20 (32 thr) | 28.74 | 23 |
| 3.14 | GPU trunc 576x niters=50 | 28.70 | 23 |
| 3.22 | GPU pa 576x niters=50 | 29.09 | 23 |
| 3.41 | GPU indep 576x niters=50 | 28.71 | 23 |
| 3.74 | GPU indep 4608x niters=5 | 28.75 | 23 |
| 3.77 | GPU pa 4608x niters=5 | 28.74 | 23 |
| 4.08 | omeco ntrials=10 niters=50 (32 thr) | 28.91 | 22 |
| 4.09 | omeco ntrials=10 niters=50 (16 thr) | 28.91 | 22 |
| 4.94 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 28.83 | 23 |
| 6.31 | omeco ntrials=32 niters=50 (32 thr) | 28.82 | 23 |
| 12.51 | GPU trunc 576x niters=200 | 28.70 | 23 |
| 12.65 | GPU pa 576x niters=200 | 28.70 | 23 |
| 12.66 | GPU indep 576x niters=200 | 28.70 | 23 |
| 14.63 | GPU pa 4608x niters=20 | 28.70 | 23 |
| 14.95 | GPU indep 4608x niters=20 | 28.71 | 23 |
| 24.59 | omeco ntrials=32 niters=200 (32 thr) | 28.71 | 23 |
| 36.31 | GPU pa 4608x niters=50 | 29.09 | 23 |
| 39.52 | GPU indep 4608x niters=50 | 28.70 | 23 |
| 48.89 | GPU trunc 576x niters=800 | 28.70 | 23 |
| 49.27 | GPU pa 576x niters=800 | 28.70 | 23 |
| 50.11 | GPU indep 576x niters=800 | 28.70 | 23 |
| 156.98 | GPU pa 4608x niters=200 | 28.70 | 23 |
| 157.51 | GPU indep 4608x niters=200 | 28.70 | 23 |

### reg3_100

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.11 | omeco ntrials=32 niters=5 (32 thr) | 17.16 | 11 |
| 0.32 | GPU indep 576x niters=5 | 17.16 | 11 |
| 0.32 | GPU pa 576x niters=5 | 17.16 | 11 |
| 0.33 | GPU trunc 576x niters=5 | 17.16 | 11 |
| 0.35 | omeco ntrials=32 niters=20 (32 thr) | 17.16 | 11 |
| 0.42 | GPU indep 576x niters=20 | 17.16 | 11 |
| 0.42 | GPU pa 576x niters=20 | 17.16 | 11 |
| 0.43 | GPU trunc 576x niters=20 | 17.16 | 11 |
| 0.51 | omeco ntrials=10 niters=50 (16 thr) | 17.16 | 11 |
| 0.52 | omeco ntrials=10 niters=50 (32 thr) | 17.16 | 11 |
| 0.58 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 17.16 | 11 |
| 0.59 | GPU indep 4608x niters=5 | 17.16 | 11 |
| 0.62 | GPU indep 576x niters=50 | 17.16 | 11 |
| 0.62 | GPU pa 576x niters=50 | 17.16 | 11 |
| 0.63 | GPU trunc 576x niters=50 | 17.16 | 11 |
| 0.64 | GPU pa 4608x niters=5 | 17.16 | 11 |
| 0.85 | omeco ntrials=32 niters=50 (32 thr) | 17.16 | 11 |
| 1.35 | GPU indep 4608x niters=20 | 17.16 | 11 |
| 1.40 | GPU pa 4608x niters=20 | 17.16 | 11 |
| 1.58 | GPU trunc 576x niters=200 | 17.16 | 11 |
| 1.60 | GPU indep 576x niters=200 | 17.16 | 11 |
| 1.60 | GPU pa 576x niters=200 | 17.16 | 11 |
| 2.84 | GPU indep 4608x niters=50 | 17.16 | 11 |
| 2.88 | GPU pa 4608x niters=50 | 17.16 | 11 |
| 3.44 | omeco ntrials=32 niters=200 (32 thr) | 17.16 | 11 |
| 5.30 | GPU indep 576x niters=800 | 17.16 | 11 |
| 5.30 | GPU trunc 576x niters=800 | 17.16 | 11 |
| 5.31 | GPU pa 576x niters=800 | 17.16 | 11 |
| 10.15 | GPU indep 4608x niters=200 | 17.16 | 11 |
| 10.23 | GPU pa 4608x niters=200 | 17.16 | 11 |

### reg3_250

| wallclock (s) | method | tc | sc |
|---|---|---|---|
| 0.27 | omeco ntrials=32 niters=5 (32 thr) | 47.32 | 35 |
| 0.46 | GPU indep 576x niters=5 | 43.76 | 35 |
| 0.47 | GPU pa 576x niters=5 | 42.92 | 36 |
| 0.47 | GPU trunc 576x niters=5 | 43.81 | 36 |
| 0.88 | GPU pa 576x niters=20 | 43.27 | 34 |
| 0.92 | GPU indep 576x niters=20 | 40.46 | 34 |
| 0.92 | GPU trunc 576x niters=20 | 44.24 | 35 |
| 0.95 | omeco ntrials=32 niters=20 (32 thr) | 44.05 | 33 |
| 1.38 | GPU indep 4608x niters=5 | 42.95 | 35 |
| 1.44 | GPU pa 4608x niters=5 | 42.73 | 34 |
| 1.44 | omeco ntrials=10 niters=50 (16 thr) | 47.00 | 33 |
| 1.54 | omeco ntrials=10 niters=50 (32 thr, no preprocess) | 47.01 | 33 |
| 1.54 | omeco ntrials=10 niters=50 (32 thr) | 47.00 | 33 |
| 1.87 | GPU indep 576x niters=50 | 40.30 | 34 |
| 2.00 | GPU pa 576x niters=50 | 40.19 | 34 |
| 2.07 | GPU trunc 576x niters=50 | 40.08 | 34 |
| 2.34 | omeco ntrials=32 niters=50 (32 thr) | 47.00 | 33 |
| 4.43 | GPU indep 4608x niters=20 | 40.46 | 34 |
| 4.55 | GPU pa 4608x niters=20 | 41.10 | 34 |
| 5.86 | GPU trunc 576x niters=200 | 42.44 | 34 |
| 5.96 | GPU pa 576x niters=200 | 43.48 | 33 |
| 6.38 | GPU indep 576x niters=200 | 40.32 | 34 |
| 9.25 | omeco ntrials=32 niters=200 (32 thr) | 46.05 | 32 |
| 10.61 | GPU indep 4608x niters=50 | 40.25 | 33 |
| 11.32 | GPU pa 4608x niters=50 | 40.11 | 34 |
| 22.56 | GPU trunc 576x niters=800 | 43.22 | 33 |
| 24.22 | GPU indep 576x niters=800 | 40.49 | 34 |
| 24.34 | GPU pa 576x niters=800 | 40.40 | 33 |
| 41.44 | GPU indep 4608x niters=200 | 40.32 | 34 |
| 44.44 | GPU pa 4608x niters=200 | 40.01 | 34 |

## GPU-runtime objective (mode ii): modeled time sum_t (s) / tc / sc, PA 576 x 50

| instance | omeco default tree | GPU [tc] | GPU [gpu] | GPU [tc+gpu] |
|---|---|---|---|---|
| sat60 | 1.12 / 44.6 / 40 | 1.25 / 43.4 / 40 | 424 / 55.6 / 51 | 0.278 / 45.4 / 40 |
| sat80 | 1.09e+04 / 59.1 / 54 | 2.2e+03 / 55.4 / 50 | 5.28e+07 / 72.6 / 67 | 336 / 55.8 / 50 |
| sat100 | 1.09e+09 / 72.9 / 69 | 1.61e+08 / 69.3 / 65 | 1.94e+14 / 95.1 / 89 | 2.69e+06 / 72.1 / 63 |
| sat120 | 1.21e+14 / 94.2 / 89 | 3.45e+12 / 83.9 / 79 | 6.24e+19 / 114.3 / 108 | 9.89e+10 / 88.1 / 78 |
| lat14 | 0.00318 / 28.9 / 22 | 0.00326 / 29.1 / 23 | 0.531 / 48.9 / 40 | 0.00323 / 35.9 / 27 |
| reg3_100 | 0.000396 / 17.2 / 11 | 0.000396 / 17.2 / 11 | - | - |
| reg3_250 | 0.0497 / 47.0 / 33 | 0.078 / 40.2 / 34 | 0.0209 / 46.0 / 35 | 0.00405 / 43.9 / 32 |
