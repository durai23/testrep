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

all: dgemm ddot mandelbrot
	
dgemm: dgemm_mkl.cpp
	$(CC) -o dgemm dgemm_mkl.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
ddot: ddot.cpp
	$(CC) -o ddot ddot.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
mandelbrot: mandelbrot.cu
	$(NVCC) -o mandelbrot mandelbrot.cu $(NVFLAGS)

.phony: clean

clean:
	rm -f *.o dgemm ddot

