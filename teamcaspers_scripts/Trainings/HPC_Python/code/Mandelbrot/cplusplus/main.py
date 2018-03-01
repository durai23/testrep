import mandelbrot
import matplotlib.pyplot as plt
import time

m = mandelbrot.PyMandelbrot(-2.13, 0.77, -1.3, 1.3, 640, 480, 20)
t0 = time.time()
m.calculate()
t1 = time.time()
print "Calculation took %.2f s" % (t1 - t0)
plt.imshow(m.getIterations().reshape(640, 480).transpose(), interpolation = 'nearest')
plt.show()
