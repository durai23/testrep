Modify the provided MPI+CUDA Jacobi to utilize CUDA-aware MPI

Follow TODOs in 
 * CUDA-aware_MPI/exercises/tasks/jacobi_cuda.c
	- Initialize CUDA before MPI_Init (call cudaSetDevice)
	- Pass device pointers directly to MPI 
Solution in
 * CUDA-aware_MPI/exercises/solutions/jacobi_cuda.c
 
Build instruction
  cd CUDA-aware_MPI/exercises/tasks
  make
  
