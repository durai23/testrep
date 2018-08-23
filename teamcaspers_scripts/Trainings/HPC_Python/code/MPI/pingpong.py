#!/usr/bin/env python
import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
myRank = comm.Get_rank()
numberOfRanks = comm.Get_size()

maxPingPong = 10000
pingPongCount = np.zeros(1, dtype=np.int)

assert numberOfRanks % 2 == 0, "Number of ranks must be even!"

t0 = time.time()
while pingPongCount < maxPingPong:
    if (myRank) % 2 == 0:
        if pingPongCount % 2 == 0:
#            print "[%d] is sending %d" % (myRank, pingPongCount)
            pingPongCount += 1
            comm.Send(pingPongCount, dest = myRank + 1)
        else:
            comm.Recv(pingPongCount, source = myRank + 1)
    else:
        if pingPongCount % 2 == 1:
#            print "[%d] is sending %d" % (myRank, pingPongCount)
            pingPongCount += 1
            comm.Send(pingPongCount, dest = myRank - 1)
        else:
            comm.Recv(pingPongCount, source = myRank - 1)
t1 = time.time()

if myRank == 0:
    print "The average roundtrip time was %.3f ms." % ((t1 - t0) * 2 * 1000 / maxPingPong)
