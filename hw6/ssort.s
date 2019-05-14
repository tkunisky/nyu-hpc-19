#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8

module purge
module load openmpi/gnu/2.0.2

cd $SCRATCH/nyu-hpc-19/hw6
rm ssort
make

RUNDIR=$SCRATCH/nyu-hpc-19/hw6/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
cd $RUNDIR

mpirun $SCRATCH/nyu-hpc-19/hw6/ssort 10000
mpirun $SCRATCH/nyu-hpc-19/hw6/ssort 100000
mpirun $SCRATCH/nyu-hpc-19/hw6/ssort 1000000
