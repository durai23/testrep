CC = invcc
OPT_FLAGS=-O3

CUBLASFLAGS = -lcublas -arch=sm_37 
NVCCFLAGS = -Xcompiler -openmp

BINARIES = task1a task1b task2

all: $(BINARIES)

task1a: task1a.cu Makefile
	$(CC) $(OPT_FLAGS) $? -o $@

task1b: task1b.cu Makefile
	$(CC) $(OPT_FLAGS) $? -o $@

task2: task2.cu Makefile
	$(CC) $(OPT_FLAGS) $(CUBLASFLAGS) $(NVCCFLAGS) $? -o $@ 

.phony: clean

clean:
	rm $(BINARIES)
