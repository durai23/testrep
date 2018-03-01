// include libraries
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "cublas_v2.h"
#include "cuda.h"

#define nstreams 4    

int main () {

  // banner
  printf ("\n\n     GPU Computing Advanced Workshop Exercise\n");
  printf (    "     ==========================================\n");
  printf (  "\n     Tiled Matrix-Matrix Multiplication\n");
  printf (    "     Using NVIDIA cuBLAS Library with Streams\n");

  // echo device data
  int idevice = 0;
  cudaSetDevice(idevice);
  cudaDeviceProp dprops;
  cudaGetDeviceProperties( &dprops, idevice );
  printf ("\n     Device name = %s, with compute capability %d.%d \n", 
	  dprops.name, dprops.major, dprops.minor);

  // define parameters
  int n = 4096;        // matrix dimension - all matrices being multiplied will be square
  
  // allocate arrays
  double *a;
  double *b;
  double *c;
  a = (double *) malloc ( n*n*sizeof(double) );
  b = (double *) malloc ( n*n*sizeof(double) );
  c = (double *) malloc ( n*n*sizeof(double) );
  
  // initialize data
  #pragma omp parallel for
  for ( int row = 0; row<n; row++ ) {
    for ( int col = 0; col<n; col++ ) {
      // data in row-major format
      a[row*n+col] = sin( 0.01*col ) + cos( 0.013*row );
      b[row*n+col] = sin( 0.017*col ) + cos( 0.03*row );
      c[row*n+col] = 0.0;
    }
  }

  // create communcations arrays
  int m = 1024;   // tile size - tiles will be square, n must be divisible by m
  double *pa;
  double *pb;
  double *pc;
  cudaMallocHost ( &pa, m*m*sizeof(double)*nstreams );
  cudaMallocHost ( &pb, m*m*sizeof(double)*nstreams );
  cudaMallocHost ( &pc, m*m*sizeof(double)*nstreams );
	  
  // create a handle to cuBlas
  cublasHandle_t cublasHandle;
  cublasCreate( &cublasHandle );

  // allocate space on device - 3 tiles for a, b, c
  double *d_a;
  double *d_b;
  double *d_c;
  cudaMalloc ( &d_a, nstreams*m*m*sizeof(double) );
  cudaMalloc ( &d_b, nstreams*m*m*sizeof(double) );
  cudaMalloc ( &d_c, nstreams*m*m*sizeof(double) );

  int offset = m*m;
  int ntiles = n/m;

  cudaStream_t myStreams[nstreams];
  for ( int i=0; i<nstreams; i++ ) {
    cudaStreamCreate( &myStreams[i] );
  }

  cudaEvent_t bufferfilled[nstreams];
  for ( int i=0; i<nstreams; i++ ) {
    cudaEventCreate ( &bufferfilled[i] );
  }

  // record start time
  cudaEvent_t t_start;
  cudaEvent_t t_end;
  cudaEventCreate (&t_start);
  cudaEventCreate (&t_end);
  cudaEventRecord (t_start,0);

  // caches for indices of previous tiles in streams
  int prowtile[nstreams];
  int pcoltile[nstreams];

  // PERFORM MULTIPLICATION
  {

    double alpha = 1.0;
    double beta = 1.0; 

    int ibuff = 0;
    int itile = 0;

    // loop over inner tile dimension
    for ( int iktile = 0; iktile < ntiles; iktile++ ) {
  
      // loop over row tiles
      for ( int irowtile = 0; irowtile < ntiles; irowtile++ ) {

        // loop over column tiles
        for ( int icoltile = 0; icoltile < ntiles; icoltile++ ) {
	  
	  if ( itile >= nstreams ) {

	    // block the host until this stream's buffers are available
	    // (that is, all previous operations in this stream have completed)
	    cudaEventSynchronize ( bufferfilled[ibuff] );

	    // copy result in pinned buffer back to global matrix
            # pragma omp parallel for
	    for ( int i=0; i<m; i++ ) {
	      for ( int j=0; j<m; j++ ) {
		c[(prowtile[ibuff]*m+i)*n+pcoltile[ibuff]*m+j] = pc[ibuff*offset+i*m+j];
	      }
	    }
	  } 

	  // copy next tile to pinned buffer on host
          # pragma omp parallel for
	  for ( int i=0; i<m; i++ ) {
	    for ( int j=0; j<m; j++ ) {
	      pa[ibuff*offset+i*m+j] = a[(irowtile*m+i)*n+iktile*m+j];
	      pb[ibuff*offset+i*m+j] = b[(iktile*m+i)*n+icoltile*m+j];
	      // TODO - copy correct tile into correct place in pc[] buffer
	    }
	  }

	  // copy tile data to device
	  cudaMemcpyAsync ( &d_a[ibuff*offset], &pa[ibuff*offset], m*m*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
	  cudaMemcpyAsync ( &d_b[ibuff*offset], &pb[ibuff*offset], m*m*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );
	  cudaMemcpyAsync ( &d_c[ibuff*offset], &pc[ibuff*offset], m*m*sizeof(double), cudaMemcpyHostToDevice, myStreams[ibuff] );

	  // tell cuBLAS which stream to use
	  // TODO: set correct stream
	  cublasSetStream( cublasHandle, ... );

	  // perform dgemm
	  cublasDgemm ( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, m, m, &alpha, &d_a[ibuff*offset], m, &d_b[ibuff*offset], m, &beta, &d_c[ibuff*offset], m ); 
	  prowtile[ibuff] = irowtile;
	  pcoltile[ibuff] = icoltile;

	  // copy result back to host
	  // TODO: set correct stream
	  cudaMemcpyAsync ( &pc[ibuff*offset], &d_c[ibuff*offset], m*m*sizeof(double), cudaMemcpyDeviceToHost, ... );

	  // this event will signal when the D2H copy of the result has completed
	  //TODO: add correct arguments
	  cudaEventRecord ( ... );
	  
	  // update buffer / stream
	  ibuff++;
	  ibuff = ibuff%nstreams;
	  itile++;

	}
      }
    }

    // catch the last nstream result tiles
    for ( itile=0; itile < nstreams; itile ++ ) {

      // block the host until this stream's buffers are available
      // TODO: add synchronization operation

      // copy result in pinned buffer back to source 
      # pragma omp parallel for
      for ( int i=0; i<m; i++ ) {
	for ( int j=0; j<m; j++ ) {
	  c[(prowtile[itile]*m+i)*n+pcoltile[itile]*m+j] = pc[itile*offset+i*m+j];
	}
      }
	    
    }

  }

  // record end time
  cudaEventRecord (t_end,0);
  cudaEventSynchronize(t_end);
  float et;
  cudaEventElapsedTime (&et, t_start, t_end);
    
  // report results
  printf("\n     reference (768,768) = %4.4f \n", c[768*n+768] );
  printf("\n     elapsedTime        = %4.4f seconds\n", (double)et/1000.);     // cudaEventElapsedTime is in milliseconds
  printf(  "     gigaflops achieved = %4.4f Gflops/s\n\n\n", 2.0e-6*n*n*n/et); // 2( * and + ) *n (inner dimension)*n^2(result size)/(time in ms.)

  // clean up
  cublasDestroy ( cublasHandle );
  cudaEventDestroy ( t_start  );
  cudaEventDestroy ( t_end );

  cudaFreeHost ( pa );
  cudaFreeHost ( pb );
  cudaFreeHost ( pc );

  cudaFree ( d_a );
  cudaFree ( d_b );
  cudaFree ( d_c );

}
