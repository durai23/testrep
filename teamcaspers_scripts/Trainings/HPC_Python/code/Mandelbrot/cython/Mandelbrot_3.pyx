#!/usr/bin/env python
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
import sys
import time

class Mandelbrot(object):
    
    def __init__(self, xmin = -2.13, xmax = 0.77, ymin = -1.3, ymax = 1.3, xres = 1200, yres = 1200,
            maxiter = 1000): 
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xres = xres
        self.yres = yres
        self.maxiter = maxiter
        self.update_ranges()
        
    def update_ranges(self):
        xx = np.arange(self.xmin, self.xmax, (self.xmax - self.xmin) / self.xres)
        yy = np.arange(self.ymin, self.ymax, (self.ymax - self.ymin) / self.yres) * 1j
        self.q = np.ravel(xx + yy[:, np.newaxis]).astype(np.complex128)
#    @profile 
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute(self):
        cdef np.ndarray[np.complex128_t, ndim = 1] q = self.q
        cdef np.ndarray[np.int32_t, ndim = 1] r = np.ones(len(self.q), dtype=np.int32) *\
	    self.maxiter
        cdef float zr, zi
        cdef float cr, ci
        cdef int j
        cdef int i
        cdef int n = len(self.q)
        cdef int m = self.maxiter
        for j in prange(n, nogil=True, schedule='dynamic'):
            zr = zi = 0           
            cr = q[j].real
            ci = q[j].imag
            for i in range(m):
                zr, zi = (zr * zr - zi * zi + cr), (2 * zr * zi) + ci
                if (zr * zr + zi * zi) > 4:
                    r[j] = i
                    break
        self.r = r
