Instructions
============

In this exercise, we'll scale a vector (array) of single precision numbers by a scalar. You'll learn 
how to allocate memory on the GPU and transfer data to and from the GPU.

Data transfer with unified memory
---------------------------------

For the first exercise, you need to log on to JUROPA 3 (see cheat sheet). Your home directory is the
same as on JUDGE. Take a look at the file ``scale_vector_um.cu``. It contains a number of todos. To
compile and run this exercise you need to use CUDA 6: ``module load cuda/6.0_rc``.

For this exercise, you'll use ``cudaMallocManaged`` to allocate memory:

cudaMallocManaged(T** devPtr, size_t size, unsigned int flags)

Like most CUDA functions, ``cudaMallocManaged`` returns ``cudaError_t``.

See the source for more todos. Use

nvcc -o scale_vector_um scale_vector_um.cu

To run your code submit your job to the batch queue using ``sbatch scale_vector_um.sh``.

Data transfer without unified memory
------------------------------------

Unified memory requires a chip with capability 3.5 (Kepler and up). The Fermi type GPUs installed on
JUDGE can't be used this way. See ``scale_vector.cu`` for things that need to be done and the slides
for the API. You compile the code in the same way as you did ``scale_vector_um.cu``:

nvcc -o scale_vector scale_vector.cu
