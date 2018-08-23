import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import numpy as np

datafile = "ddot_bench.dat"
posFlops = 4
n = np.array([10**i for i in xrange(3, 8 )])
data = np.array([float(l.split()[posFlops]) for l in file(datafile)])
plt.loglog(n, data[::2], 'o-', label="CPU")
plt.loglog(n, data[1::2],'s-',  label="GPU")
plt.legend(loc='upper left')
plt.xlabel("Length of Vector")
plt.ylabel("t[s]")
plt.title("DDot Benchmark")
plt.savefig("ddot_bench.png")
plt.show()
