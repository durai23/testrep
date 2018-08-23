CUBLAS
======

In this exercise, we use the BLAS3 routine DGEMM in CUBLAS to perform  a matrix-matrix
multiplication using double precision. GEMM is defined as

.. math:: C=\alpha A B + \beta C.

where A,B, and C are matrices and :math:`\alpha ` and :math:`\beta ` are scalars. 
In our case, we only want to multiply *A* and *B* so we set :math:`\beta=0`.

The program (`dgemm.cpp`) allocates space for two square matrices *A* and *B* and fills them with 
random numbers. The data is then copied to the GPU where the DGEMM routine is executed and the result 
is returned back to the host.

Todo
----

Add copy of matrix B to GPU (see todo in source). 

Why are there so many parameters in ``cublasSetVector()``? What do they mean? 

Extra credit
____________

Replace random number generator with call to CURAND and remove unneccessary code.

