NVCC=nvcc
NVCC_FLAGS= -G -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35
BINARIES= task1-cuda-gdb task0-cuda-memcheck
all: $(BINARIES)

task1-cuda-gdb: task1-cuda-gdb.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -o task1-cuda-gdb task1-cuda-gdb.cu
	
task0-cuda-memcheck: task0-cuda-memcheck.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -o task0-cuda-memcheck task0-cuda-memcheck.cu
	
clean:
	rm $(BINARIES)

