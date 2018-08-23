CC = invcc
OPT_FLAGS=-O3

CUBLASFLAGS = -lcublas 
NVCCFLAGS = -Xcompiler -openmp -arch=sm_37 

BINARIES = task1a task1b task2

all: $(BINARIES)

task1a: task1a.cu
	$(CC) $(OPT_FLAGS) $(NVCCFLAGS) $? -o $@

task1b: task1b.cu
	$(CC) $(OPT_FLAGS) $(NVCCFLAGS) $? -o $@

task2: task2.cu
	$(CC) $(OPT_FLAGS) $(CUBLASFLAGS) $(NVCCFLAGS) $? -o $@ 

.phony: clean

clean:
	rm $(BINARIES)
