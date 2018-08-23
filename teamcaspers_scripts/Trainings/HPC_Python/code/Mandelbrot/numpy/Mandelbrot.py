#!/usr/bin/env python
import numpy as np
import sys
import time
from pylab import *
class Mandelbrot(object):
    
    def __init__(self, xmin = -2.13, xmax = 0.77, ymin = -1.3, ymax = 1.3, xres = 1200, yres = 1200, maxiter = 100): 
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xres = xres
        self.yres = yres
        self.maxiter = maxiter
        self.update_ranges()
#    @profile        
    def update_ranges(self):
        xx = np.linspace(self.xmin, self.xmax, self.xres)
        yy = np.linspace(self.ymin, self.ymax, self.yres)
	X, Y = np.meshgrid(xx, yy)
        self.q = X + Y * 1j
#    @profile    
    def compute(self):
        z = np.zeros_like(self.q)
        self.r = np.zeros(self.q.shape)
        for i in range(self.maxiter):
            z = z * z + self.q
            done = np.abs(z) > 2
            self.r[done] = i
            z[done] = 0.0 + 0.0j
            self.q[done] = 0.0 + 0.0j

def main(argv):
    xres = 750
    yres = 500
    mandelbrot = Mandelbrot(xmin=-2.0, xmax=1.0, ymin=-1.0, ymax=1.0, xres=xres, yres=yres, maxiter = 20)
    t0 = time.time()
    mandelbrot.compute()
    t1 = time.time()
    print "Calculation took %.2f s" % (t1 - t0)
    imshow(mandelbrot.r)
    show()
    
    
if __name__ == "__main__":
    main(sys.argv)
