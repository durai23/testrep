CC = icc
MPICC = mpicc

MKLFLAGS = -w -I${MKLINCLUDE}\
-L"${MKLPATH}"\
"${MKLPATH}"/libmkl_intel_lp64.a -Wl,--start-group\
"${MKLPATH}"/libmkl_intel_thread.a\
"${MKLPATH}"/libmkl_core.a -Wl,--end-group\
-L"${MKLPATH}" -liomp5 -lpthread -lm
CUBLASFLAGS = -I${CUDAROOT}/include -L${CUDAROOT}/lib64 -lcublas -lcudart
CFLAGS = -g -debug inline_debug_info -O3

all: dgemm
	
dgemm: dgemm.cpp
	$(CC) -o dgemm dgemm.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
.phony: clean

clean:
	rm -f *.o dgemm
