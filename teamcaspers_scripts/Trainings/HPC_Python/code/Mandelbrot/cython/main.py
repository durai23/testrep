#!/usr/bin/env python
import pyximport
pyximport.install()
import numpy as np
#from mayavi import mlab
import sys
import time

from Mandelbrot_3 import Mandelbrot



def main(argv):
    xres = yres = 1200 if len(sys.argv) == 1 else int(sys.argv[1])
    mandelbrot = Mandelbrot(xres = xres, yres = yres, maxiter=1000)
    t0 = time.time()
    mandelbrot.compute()
    t1 = time.time()
    print "Calculation took %.2f s" % (t1 - t0)
    r = np.array(mandelbrot.r).reshape(xres,yres).transpose()
    #mlab.surf(r, warp_scale="auto")
#    mlab.surf(r)
#    mlab.show()

if __name__ == "__main__":
    main(sys.argv)
