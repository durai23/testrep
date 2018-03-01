/*
 * dgemm_gpu_simple.cu
 *
 * compile with: make dgemm_gpu_simple
 *
 * Matrices are stored as arrays in row-major order: 
 * A[row][col] = A[row * N + col]
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 16 


// Declaration of helper functions (see bottom of file for details)
void checkError (const char* action);
float getGflops (int, float);


//Matrix multiplication kernel called by matrixMulOnDevice()
__global__ void dgemm_gpu_simple (double *a, double *b, double *c, int n) { 

  
  double Cvalue = 0.0; 

  // TODO
  // Get global threadId in x and y direction
  int row = ...
  int col = ...

    
  // TODO
  // Each thread computes one element of C by accumulating results into Cvalue
  // Matrices are stored in row-major order: 
  // A[row][col] = A[row * N + col]
  // Remember to check bounds !
  if ( row < n && col < n) {
    ...
  }

}


/*
 *  Matrix multiplication host function called by main() 
 */

void matrixMulOnDevice(const double *a, const double* b, double *c, int n) { 
  
  double *d_a;             // matirx A in device memory
  double *d_b;             // matirx B in device memory
  double *d_c;             // matirx C in device memory
  size_t size;
  float time;

  // Define grid and block layout for kernel execution
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
  // TODO
  // determine number of blocks for x and y dimension
  dim3 gridDim( ... , ... ); 
  
  // Define events for timing
  cudaEvent_t start, stop; 
  
  cudaEventCreate(&start); 
  cudaEventCreate(&stop); 

 
  
  // TODO
  // Allocate memory for d_a, d_b and d_c on device
  size = n * n * sizeof (double);
  ...
  checkError("allocating device memory for A");
  ...
  checkError("allocating device memory for B");
  ...
  checkError("allocating device memory for C");

  // start measurment
  cudaEventRecord(start, 0);

  // TODO
  // Copy data for a and b from host to device
  ...
  checkError("copying data of A from host to device");
  ...
  checkError("copying data of B from host to device");
  
  printf("\nExecution grid: %d, %d; block:%d, %d\n", gridDim.x, gridDim.y , BLOCK_SIZE, BLOCK_SIZE);
    
  
  // TODO
  // Invoke kernel 
  dgemm_gpu_simple<<< ..., ... >>>( ... ); 

  
  // TODO
  // Read restults from device memory to C 
  ...
  checkError("copying results from device to host");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkError("kernel execution");

  // Get elapsed time for kernel execution
  cudaEventElapsedTime(&time, start, stop); 
  cudaEventDestroy(start); 
  cudaEventDestroy(stop);

  printf ("\nExecution Time: %f ms (dim C: %d * %d)", time, n, n);
  printf ("\nThis corresponds to: %4.4f GFLOPS", getGflops(n, time));
  
  // Free device memory 
  cudaFree(d_a);
  checkError("Freeing d_a");
  cudaFree(d_b);
  checkError("Freeing d_b");
  cudaFree(d_c);
  checkError("Freeing d_c");

} 


/*
 *  Main program
 */
int main (int argc, char* argv[]) {

  int n = 1024; // dimension of square matrices
  double *a, *b, *c;
  int row, col;
  double absError, maxAbsError = 0.0, sumAbsError = 0.0;
  
  if (argc > 1) {
       n = atoi(argv[1]);
  }

  // show banner
  printf ("\n\n     Matrix-Multiplication \n");
  printf (    "     ==========================================\n");
  printf (  "\n     Simple DGEMM implemantation on GPU");  

  // echo device data
  int idevice = 0;
  cudaSetDevice(idevice);
  cudaDeviceProp dprops;
  cudaGetDeviceProperties( &dprops, idevice );
  printf ("\n     Device name = %s, with compute capability %d.%d \n", 
	  dprops.name, dprops.major, dprops.minor);
  printf (  "\n     Matrix size %d x %d", n, n);
  
  // Allocate memory for matrices on host
  assert ( a = (double*) malloc (n * n * sizeof(double)) );
  assert ( b = (double*) malloc (n * n * sizeof(double)) );
  assert ( c = (double*) malloc (n * n * sizeof(double)) );
  
  // Init matrices A and B
  #pragma omp parallel for private (row, col)
  for ( row = 0; row < n; row++ ) {
    for ( col = 0; col < n; col++ ) {
      a[row * n + col] = (row == col) ? 1.0 : 0.0;
      b[row * n + col] = row * n + col;
    }
  }

  // Execute matrix multiplication (on device and on host for reference
  matrixMulOnDevice (a, b, c, n);
  
  // Compare results
  for ( row = 0; row < n; ++row){
    for ( col = 0; col < n; ++col) {

      absError = fabs ( c[row * n + col] - b[row * n + col]);
      sumAbsError += absError;

      if (absError > maxAbsError)
	maxAbsError = absError;
    }
  }

  // Free memory on host
  free (a);
  free (b);
  free (c);
  
  printf ("\nmaxAbsError: %4.4f, sumAbsError: %4.4f", maxAbsError, sumAbsError);
  if (maxAbsError < 2.0e-5)
    printf ("\n\nProgram terminated SUCCESSFULLY.\n\n");

  return 0;
}



/*
 *  Some helper functions
 */

// Simple error checking function for CUDA actions
void checkError (const char* action) {
  
  cudaError_t error;
  error = cudaGetLastError(); 

  if (error != cudaSuccess) {
    printf ("\nError while '%s': %s\nprogram terminated ...\n\n", action, cudaGetErrorString(error));
    exit (EXIT_SUCCESS);
  }
}

// get compute performance
float getGflops (int width, float time) {

	float gf = (2.0e-6 * width * width* width / time);

	return gf;
}