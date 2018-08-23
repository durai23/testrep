#!/usr/bin/env python
import time
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from math import sqrt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
itemsPerRank = 100000
numberOfItems = itemsPerRank * size

a = range(rank * itemsPerRank, rank * itemsPerRank +itemsPerRank)
b = np.array(a)

# Gather Python objects
t0 = time.time()
a = comm.gather(a) # Note, it's an assignment!
t1 = time.time()

b_all = None
if rank == 0:
    b_all = np.empty(b.size * size)
    
t2 = time.time()
comm.Gather(b, b_all)    # Note, must provide the receive buffer, but it can be 'None' on all but
                         # the receiving rank.
t3 = time.time()

if rank == 0:
    #print a
    print "Gathering the data using gather took %.6s." % (t1 - t0)
    print "Gathering the data using Gather took %.6s." % (t3 - t2)
    print "Gather is %.1f times faster than gather!" % ((t1 - t0) / (t3 - t2))

# Now we gather some 2D data. We start with horizontal stripes.
c = np.ones((100/size, 100), dtype = int) * rank
c_all = None
if rank == 0:
    c_all = np.empty(c.size * size, dtype = int)
comm.Gather(c, c_all)
if rank == 0:
    print c_all.shape
    c_all = c_all.reshape(100, 100)
    plt.imshow(c_all)
    plt.show()

# Now for something a little harder. We want to use n by n blocks for each rank. Collect them and 
# put them back together.
c = np.ones((128 / sqrt(size), 128 / sqrt(size)), dtype = int) * rank
if rank == 0:
    c_all = np.empty((size, c.size), dtype = int)
comm.Gather(c, c_all)
if rank == 0:
    print c_all.shape
    plt.imshow(c_all.reshape(128, 128))
    plt.show()

    # OK, we are not quite there, yet.

    c_blocked = np.empty((128, 128), dtype = int)
    blockwidth = int(128 / sqrt(size))
    count = 0
    for i in range(0, 128, blockwidth):
        for j in range(0, 128, blockwidth):
            c_blocked[i: i + blockwidth, j: j + blockwidth] = \
                c_all[count].reshape(blockwidth, blockwidth)
            count += 1
    plt.imshow(c_blocked)
    plt.colorbar()
    plt.show()
    
# Now, we do some more shape manipulation. We want to distribute our data and put it back 
# columnwise.
height = 16
width = height
blockwidth = width / size

c = np.ones((height, blockwidth), dtype = int) * rank
print c.size
if rank == 0:
    c_all = np.empty((size, c.size), dtype = int)
comm.Gather(c, c_all)

if rank == 0:
    print c_all
    result = np.zeros((height, width), dtype = int)
    for i in range(size):
        result[:,i::size] = c_all[i].reshape(height, blockwidth)
    print result
    plt.imshow(result)
    plt.show()


        


