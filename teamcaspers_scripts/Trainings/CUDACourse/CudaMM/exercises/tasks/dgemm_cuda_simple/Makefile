CC=invcc
CUDA_FLAGS=-arch=sm_37 -lcudart
OPT_FLAGS=-O3
OMP_FLAGS=-Xcompiler -openmp

all: dgemm_cpu_simple dgemm_gpu_simple dgemm_gpu_simple_uvm


dgemm_cpu_simple: dgemm_cpu_simple.cu
	$(CC) $(OPT_FLAGS) $(CUDA_FLAGS) $(OMP_FLAGS) $? -o $@ 

dgemm_gpu_simple: dgemm_gpu_simple.cu
	$(CC) $(OPT_FLAGS) $(CUDA_FLAGS) $(OMP_FLAGS) $? -o $@ 

dgemm_gpu_simple_uvm: dgemm_gpu_simple_uvm.cu
	$(CC) $(OPT_FLAGS) $(CUDA_FLAGS) $(OMP_FLAGS) $? -o $@

clean:
	rm -f dgemm_cpu_simple 
	rm -f dgemm_gpu_simple 
	rm -f dgemm_gpu_simple_uvm
