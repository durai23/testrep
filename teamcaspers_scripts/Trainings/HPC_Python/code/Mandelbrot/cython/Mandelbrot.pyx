#!/usr/bin/env python
import numpy as np
import sys
import time
from pylab import *
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
        self.q = np.ravel(xx + yy[:, np.newaxis]).astype(np.complex64)
#    @profile    
    def compute(self):
        z = np.zeros_like(self.q)
        self.r = np.zeros(len(self.q))
        for i in range(self.maxiter):
            z = z * z + self.q
            done = np.abs(z) > 2
            self.r[done] = i
            z[done] = 0.0 + 0.0j
            self.q[done] = 0.0 + 0.0j

