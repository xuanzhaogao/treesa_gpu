# Build the GPU TreeSA binary (module load cuda/12.8.0). No Hopper-only features are used.
#   make                                              -> treesa_gpu/bin/treesa_gpu  for the local GPU (ARCH=native)
#   make ARCH=sm_75  OUT=treesa_gpu/bin/treesa_gpu_sm75   e.g. Quadro RTX 6000 (workstation)
#   make ARCH=sm_90a OUT=treesa_gpu/bin/treesa_gpu_sm90   H100 (select with TREESA_GPU_BINARY)
#   tests/run_checks.sh <binary>                      validation matrix (see tests/)
NVCC ?= nvcc
ARCH ?= native
OUT ?= treesa_gpu/bin/treesa_gpu
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -fopenmp,-O3 -Xptxas -v

$(OUT): csrc/treesa_gpu.cu csrc/common.h csrc/cpu_ref.h
	@mkdir -p $(dir $(OUT)) build
	$(NVCC) $(NVFLAGS) -o $(OUT) csrc/treesa_gpu.cu -lgomp 2> build/ptxas_$(subst /,_,$(ARCH)).log || (cat build/ptxas_$(subst /,_,$(ARCH)).log; false)

clean:
	rm -rf treesa_gpu/bin build

.PHONY: clean
