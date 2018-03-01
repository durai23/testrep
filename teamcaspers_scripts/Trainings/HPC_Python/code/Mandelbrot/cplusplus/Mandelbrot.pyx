import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "mandelbrot.h":
    cdef cppclass Mandelbrot:
        Mandelbrot(double, double, double, double, int, int, int)
        void calculate()
        vector[int] iterations

cdef class PyMandelbrot:
    cdef Mandelbrot *thisptr

    def __cinit__(self, double xmin, double xmax, double ymin, double ymax, int xres, int yres, int maxiter):
        self.thisptr = new Mandelbrot(xmin, xmax, ymin, ymax, xres, yres, maxiter)

    def __dealloc__(self):
        del self.thisptr

    def calculate(self):
        self.thisptr.calculate()

    def getIterations(self):
       cdef np.ndarray[np.int_t, ndim=1] iteration = np.zeros(self.thisptr.iterations.size(),dtype=np.int)
       cdef int i
       cdef int iterMax
       for i in range(0, self.thisptr.iterations.size()):
           iteration[i] = self.thisptr.iterations[i]
           if iterMax < iteration[i]:
               iterMax = iteration[i]
       return iteration
