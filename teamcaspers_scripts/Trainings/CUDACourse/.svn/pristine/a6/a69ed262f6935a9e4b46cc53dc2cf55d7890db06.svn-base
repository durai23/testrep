CC=invcc
#NO_CACHE_FLAGS=-Xptxas -dlcm=cs
CUDA_FLAGS=-arch=sm_37 -lcudart -lcublas
OPT_FLAGS=-O3
OMP_FLAGS=-Xcompiler -fopenmp


all: dgemm_tiled dgemm_tiled_streams dgemm_tiled_streams_multigpu


dgemm_tiled: dgemm_tiled.cu
	$(CC) $(OPT_FLAGS) $(CUDA_FLAGS) $(OMP_FLAGS) $? -o $@ 

dgemm_tiled_streams: dgemm_tiled_streams.cu
	$(CC) $(OPT_FLAGS) $(CUDA_FLAGS) $(OMP_FLAGS) $? -o $@ 

dgemm_tiled_streams_multigpu: dgemm_tiled_streams_multigpu.cu
	$(CC) $(OPT_FLAGS) $(CUDA_FLAGS) $(OMP_FLAGS) $? -o $@

clean:
	rm -f dgemm_tiled 
	rm -f dgemm_tiled_streams 
	rm -f dgemm_tiled_streams_multigpu
