CC=invcc
CUDA_FLAGS=-arch=sm_37 -lcudart
OPT_FLAGS=-O3
OMP_FLAGS=-Xcompiler -openmp

all: dgemm_gpu_shared_uvm dgemm_gpu_shared


dgemm_gpu_shared: dgemm_gpu_shared.cu
	$(CC) $(OPT_FLAGS) $(OMP_FLAGS) $(CUDA_FLAGS) $? -o $@

dgemm_gpu_shared_uvm: dgemm_gpu_shared_uvm.cu
	$(CC) $(OPT_FLAGS) $(OMP_FLAGS) $(CUDA_FLAGS) $? -o $@ 
clean:
	rm dgemm_gpu_shared dgemm_gpu_shared_uvm
