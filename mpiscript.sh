#!/bin/bash
#SBATCH -J mpitest
#SBATCH -N 4
#SBATCH --ntasks-per-node=24
#SBATCH -o mpitest-%j.out
#SBATCH -e mpitest-%j.err
#SBATCH --partition=batch
#SBATCH --time=00:30:00
# run MPI application below (with srun)
srun -N 4 --ntasks-per-node=24 ./mpi-prog
