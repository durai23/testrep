import numpy as np
cimport numpy as np

cdef extern from "mandelbrot.h":
    int init(double complex* q, double xmin, double xmax, double ymin, double ymax, int xres, int yres)
    void calculate(int* iterations, double complex* q, int n, int maxiter)

def init_mandelbrot(xmin, xmax, ymin, ymax, xres, yres):
    
    cdef np.ndarray[np.complex128_t, ndim = 1] q

    q = np.zeros(xres * yres, dtype=np.complex128)
    init(<np.complex128_t*> q.data, xmin, xmax, ymin, ymax, xres, yres)
    return q
    
def calculate_mandelbrot(np.ndarray[np.complex128_t, ndim = 1] q, maxiter):
    cdef np.ndarray[np.int32_t, ndim = 1] iterations
    iterations = np.zeros(len(q), dtype=np.int32)
    calculate(<np.npy_intp*> iterations.data, <np.complex128_t*> q.data, len(q), maxiter)
    return iterations

    
