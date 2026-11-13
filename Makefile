# Build the GPU TreeSA binary (module load cuda/12.8.0).
#   make                      -> build/treesa_gpu      for the local Quadro RTX 6000 (sm_75)
#   make ARCH=sm_90a OUT=build/treesa_gpu_sm90         for the H100 (no Hopper-only features are used)
NVCC ?= nvcc
ARCH ?= sm_75
OUT ?= build/treesa_gpu
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -fopenmp,-O3 -Xptxas -v
$(OUT): src/treesa_gpu.cu src/common.h src/cpu_ref.h
	$(NVCC) $(NVFLAGS) -o $(OUT) src/treesa_gpu.cu -lgomp 2> build/ptxas_$(ARCH).log || (cat build/ptxas_$(ARCH).log; false)
clean:
	rm -f build/treesa_gpu build/treesa_gpu_sm90
