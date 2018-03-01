from __future__ import print_function
from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numberOfRanks = comm.Get_size()

N = 100000
Nnode = int(N / numberOfRanks) + 1

# compute numbers and scatter to workers
if rank == 0:
    print('\nlocal min and max values')
    sys.stdout.flush()
    send_buffer = np.random.normal(size=(N))
else:
    send_buffer = np.zeros(Nnode)

# initialize buffer for received data
receive_buffer = np.zeros(Nnode)

# scatter the data 
comm.Scatter(send_buffer, receive_buffer, root = 0)

# compute minimal and maximal values
gauss_min = np.array(np.min(receive_buffer))
gauss_max = np.array(np.max(receive_buffer))

print(rank, ':', gauss_min, gauss_max)


# collect min and max values from all ranks
all_gauss_min = np.zeros(numberOfRanks)
all_gauss_max = np.zeros(numberOfRanks)

comm.Gather(gauss_min, all_gauss_min)
comm.Gather(gauss_max, all_gauss_max)

if rank == 0:
    print('\nlocal min and max values collected')
    print(all_gauss_min)
    print(all_gauss_max)

    
# compute the global min and max values

global_gauss_min = np.zeros(1)
global_gauss_max = np.zeros(1)

comm.Reduce(gauss_min, global_gauss_min, op=MPI.MIN, root=0)
comm.Reduce(gauss_max, global_gauss_max, op=MPI.MAX, root=0)

if rank == 0:
    print('\nglobal min and max values')
    print('reduced:  ', global_gauss_min, global_gauss_max)
    print('computed: ', np.min(send_buffer), np.max(send_buffer))
    print('')
    