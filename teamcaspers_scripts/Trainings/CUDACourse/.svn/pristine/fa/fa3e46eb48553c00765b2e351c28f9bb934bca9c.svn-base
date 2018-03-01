import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np

datafile = "mandelbrot_bench.dat"
posFlops = 11

data = np.array([[l.split()[posFlops - 3], l.split()[posFlops]] for l in open("mandelbrot_bench.dat")])
#plt.loglog(data[2::3,0], data[2::3,1], 'o-', label="CPU (OpenCL)")
plt.loglog(data[::1,0], data[::1,1], 'o-', label="GPU (CUDA)")
#plt.loglog(data[1::3,0], data[1::3,1], 'o-', label="GPU (OpenCL)")
plt.axis([60, 10000, 10**-4, 1])
plt.legend(loc='upper left')
plt.xlabel("Width of Image")
plt.ylabel("t [s]")
plt.title("Mandelbrot Benchmark")
plt.savefig("mandelbrot_bench.png")
plt.show()
