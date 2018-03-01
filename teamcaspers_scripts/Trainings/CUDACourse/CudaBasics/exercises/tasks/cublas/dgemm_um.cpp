#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_vsl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "errcheck.inc"

int main(int argc, char** argv){
    double* A;
    double* B;
    double* C;
    
    double alpha = 1.0;
    double beta = 0.0;
    int i;        
    struct timeval t1,t2, t3, t4;
    
    const int SEED = 1;
    const int METHOD = 0;
    const int BRNG = VSL_BRNG_MCG31;
    VSLStreamStatePtr stream;
    int errcode;
    
    cublasStatus_t status;
    cublasHandle_t handle;
    
    double a=0.0, b= 1.0; // Uniform distribution between 0 and 1
    
    errcode = vslNewStream(&stream, BRNG, SEED);
    
    int width = 100;
    if (argc > 1){
        width = atoi(argv[1]);
    }
    /* Allocate memory for A, B, and C */
    if (cudaMallocManaged(&A, width * width * sizeof(double)) != cudaSuccess){
        fprintf(stderr, "!!!! device memory alocation error (allocate A)\n");
        return EXIT_FAILURE;
    }
    if (cudaMallocManaged(&B, width * width * sizeof(double)) != cudaSuccess){
        fprintf(stderr, "!!!! device memory alocation error (allocate B)\n");
        return EXIT_FAILURE;
    }
    if (cudaMallocManaged(&C, width * width * sizeof(double)) != cudaSuccess){
        fprintf(stderr, "!!!! device memory alocation error (allocate C)\n");
        return EXIT_FAILURE;
    }
    /* Generate width * width random numbers between 0 and 1 to fill matrices A and B. */
    errcode = vdRngUniform(METHOD, stream, width * width, A, a, b);
    CheckVslError(errcode);
    errcode = vdRngUniform(METHOD, stream, width * width, B, a, b);
    CheckVslError(errcode);
    
    /* Now prepare the call to CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }
    gettimeofday(&t3, NULL);

    
    /* Perform calculation */
    status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, width, width, width, &alpha, A,
        width, B, width, &beta, C, width);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize(); 
    gettimeofday(&t4, NULL);
    printf("Call to cublasDGEMM took %lf\n",(double) (t4.tv_sec - t3.tv_sec) + 1e-6 * (t4.tv_usec -
        t3.tv_usec));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }
    
    return 0;
}