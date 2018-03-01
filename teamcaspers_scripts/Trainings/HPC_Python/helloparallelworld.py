from __future__ import print_function
from mpi4py import MPI
import numpy as np

rank = MPI.COMM_WORLD.Get_rank()
numberOfRanks = MPI.COMM_WORLD.Get_size()

a = np.zeros(1)
if rank == 0:
    print("There are %d MPI ranks." % numberOfRanks)
    a = np.random.random(1)

print("I'm rank %d." % rank)
if rank == 0:
    MPI.COMM_WORLD.Send(a, dest=1)
elif rank == 1:
    MPI.COMM_WORLD.Recv(a, source=1)
    