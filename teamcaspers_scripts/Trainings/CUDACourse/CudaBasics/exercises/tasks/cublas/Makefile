CC = icc
NVCC = nvcc
MPICC = mpicc

MKLFLAGS = -qopenmp -I${MKLROOT}/include\
-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a\
${MKLROOT}/lib/intel64/libmkl_core.a\
${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl
CUBLASFLAGS = -I${CUDAROOT}/include -L${CUDAROOT}/lib64 -lcublas -lcudart
CFLAGS = -g -debug inline_debug_info -O3
NVFLAGS = --cudart=static -gencode arch=compute_20,code=compute_20 -gencode arch=compute_35,code=compute_35 -std=c++11

all: dgemm dgemm_um
	
dgemm: dgemm.cpp
	$(CC) -o dgemm dgemm.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
dgemm_um: dgemm_um.cpp
	$(CC) -o dgemm_um dgemm_um.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
.phony: clean

clean:
	rm -f *.o dgemm dgemm_um
