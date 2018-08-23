import mandelbrot
import time
import sys
import matplotlib.pyplot as plt
if len(sys.argv) > 1:
    xres = yres = int(sys.argv[1])
else:
    xres = yres = 1200
q = mandelbrot.init_mandelbrot(-2.13, 0.77, -1.3, 1.3, xres, yres)
t0 = time.time()
i = mandelbrot.calculate_mandelbrot(q, 1000)
t1 = time.time()
print "Calculation took %.2f s" % (t1 - t0)
plt.imshow(i.reshape(xres, yres), interpolation='nearest')
plt.show()
