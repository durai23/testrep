MPICC=mpicc
CC=gcc
NVCC=nvcc

CFLAGS=-std=c99 -O3 -march=native
OMPCFLAGS=-DOMP_MEMLOCALTIY -fopenmp
CUDACFLAGS=-I$(CUDAROOT)/include
MPICFLAGS=-DUSE_MPI

NVCCFLAGS=-O3 -arch=sm_20 -Xcompiler -march=native -lineinfo

OMPLDFLAGS=-fopenmp -lm
CUDALDFLAGS=-L$(CUDAROOT)/lib64 -lcudart

BINDIR=.
JACOBI_MPICUDA=$(BINDIR)/jacobi_mpi+cuda
BINARIES=$(JACOBI_MPICUDA)

all: $(BINARIES)

jacobi_cuda_kernel.o: jacobi_cuda_kernel.cu Makefile
	$(NVCC) $(NVCCFLAGS) -c jacobi_cuda_kernel.cu -o jacobi_cuda_kernel.o
	
jacobi_mpi+cuda.o: jacobi_cuda.c Makefile
	$(MPICC) $(CFLAGS) $(OMPCFLAGS) $(MPICFLAGS) $(CUDACFLAGS) -c jacobi_cuda.c -o jacobi_mpi+cuda.o
	
$(JACOBI_MPICUDA): jacobi_mpi+cuda.o jacobi_cuda_kernel.o Makefile
	$(MPICC) jacobi_mpi+cuda.o jacobi_cuda_kernel.o -o $(JACOBI_MPICUDA) $(OMPLDFLAGS) $(CUDALDFLAGS) 
	
clean:
	rm *.o $(BINARIES)
