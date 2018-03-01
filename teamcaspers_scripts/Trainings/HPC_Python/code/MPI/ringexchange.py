#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
myRank = comm.Get_rank()
numberOfRanks = comm.Get_size()

youAreIt = np.empty(1, dtype=np.int)

if myRank == 0:
    youAreIt = np.ones(1, dtype=np.int)

for i in xrange(4):
    comm.Isend(youAreIt, dest = (myRank + 1) % numberOfRanks)
    youAreIt[0] = 0
    comm.Recv(youAreIt, source = (myRank - 1) % numberOfRanks)
    if youAreIt[0]:
       print "Rank %d: I am it!" % myRank 
    else:
       print "Rank %d: I am not it!" % myRank 
      


