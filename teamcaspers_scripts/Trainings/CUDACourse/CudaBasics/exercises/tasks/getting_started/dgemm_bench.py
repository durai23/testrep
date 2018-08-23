import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np

datafile = "dgemm_bench.dat"
posFlops = 4
n = np.array([2**i for i in xrange(5, 15, 1 )])
GFLOP = 2. * n ** 3 / 10 ** 9
data = np.array([float(l.split()[posFlops]) for l in file(datafile)])
plt.plot(n[:len(data[::2])], GFLOP[:len(data[::2])]/data[::2], 'o-', label="CPU")
plt.plot(n[:len(data[1::2])], GFLOP[:len(data[1::2])]/data[1::2],'s-',  label="GPU")
plt.legend(loc='upper left')
plt.xlabel("Size of Square Matrix")
plt.ylabel("GFLOP/s")
plt.title("DGEMM Benchmark")
plt.savefig("dgemm_bench.png")
plt.show()
