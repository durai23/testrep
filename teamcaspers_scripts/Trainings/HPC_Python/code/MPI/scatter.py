#!/usr/bin/env python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
myRank = comm.Get_rank()
size = comm.Get_size()

n = 10000
a = np.empty(n / size, dtype=np.float64)
localTotal = None
a_all = None

if myRank == 0:
    a_all = np.random.random(10000)
    localTotal = np.sum(a_all)

comm.Scatter(a_all,a)
myres = np.sum(a)
total = np.zeros(1)
comm.Reduce(myres, total )
if myRank == 0:
    if np.allclose(total, localTotal):
        print "Result is OK: %f" % total
    else:
        print "Local and MPI result differ by %f" % (localTotal - total)
