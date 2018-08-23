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
        
    def update_ranges(self):
	dx = (self.xmax - self.xmin) / (self.xres - 1)
	dy = (self.ymax - self.ymin) / (self.yres - 1)
        self.q = [self.xmin + i * dx + (self.ymin + j * dy) * 1j for i in xrange(0, self.xres) for j in
			xrange(0, self.yres)]

#    @profile    
    def compute(self):
	self.r = [0] * len(self.q)
	for j, c in enumerate(self.q):
            z = 0 + 0j
            for i in xrange(self.maxiter):
                z = z * z + c
		if abs(z) > 2.0:
		    self.r[j] = i
		    break
                

def main(argv):
    xres = 80
    yres = 60
    mandelbrot = Mandelbrot(xres = xres, yres = yres, maxiter =10)
    t0 = time.time()
    mandelbrot.compute()
    t1 = time.time()
    print "Calculation took %.2f s" % (t1 - t0)
    r = array(mandelbrot.r).reshape(xres,yres).transpose()
    imshow(r)
    show()
    
    
if __name__ == "__main__":
    main(sys.argv)
