# coding: utf-8

from __future__ import print_function, division

from mpi4py import MPI
from subprocess import call
import sys
import math

# Get the global communicator
comm = MPI.COMM_WORLD
# Obtain the rank of each process
rank = comm.Get_rank()
# Get the total number of processes
world_size = comm.Get_size()

# read the name of the tasks file from the command line
tasks_file_name = sys.argv[1]
if not tasks_file_name:
    if rank == 0:
        print('ERROR: no task file specified!')
    exit(1)

# pars optional command line parameter to indicate to be verbose
if len(sys.argv) > 2:
    verbose = sys.argv[2] == '-v'
else:
    verbose = False

# read input file
with open(tasks_file_name, 'rt') as f:
    tasks = [s for s in f.readlines()]

# determine number of tasks
total_tasks = len(tasks)
# determin number of tasks per process
tasks_per_process = int(math.ceil(int(total_tasks) / world_size))
# calculate which tasks to be computed by this process
start = rank * tasks_per_process
end = min(start + tasks_per_process, total_tasks)
# execute all tasks
for i in range(start, end):
    if verbose:
        print('Worker {0:04d} : {1}'.format(rank, tasks[i]))
    status = call(tasks[i], shell=True)
    if verbose:
        print('Worker {0:04d} : Status {1}'.format(rank, str(status)))
