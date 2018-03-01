from mandelbrot_f import mandelbrot
import time
import sys
import matplotlib.pyplot as plt
if len(sys.argv) > 1:
    xres = yres = int(sys.argv[1])
if len(sys.argv) > 2:
    yres = int(sys.argv[2])
else:
    xres = yres = 1200
maxiter = 20
mandelbrot.init(-2, 1.0, -1.0, 1.0, xres, yres,maxiter)
t0 = time.time()
mandelbrot.calculate()
t1 = time.time()
print "Calculation took %.2f s" % (t1 - t0)
plt.imshow(mandelbrot.iterations.reshape(xres, yres).transpose(), interpolation='nearest')
plt.show()
