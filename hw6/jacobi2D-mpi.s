#!/bin/bash

module purge
module load openmpi/gnu/2.0.2

cd $SCRATCH/nyu-hpc-19/hw6
rm jacobi2D-mpi
make

RUNDIR=$SCRATCH/nyu-hpc-19/hw6/jacobi-run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
cd $RUNDIR

mpirun $SCRATCH/nyu-hpc-19/hw6/jacobi2D-mpi $N 1000
