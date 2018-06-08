#!/bin/bash
#SBATCH -n 24
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 5
#SBATCH --output=example_%j.out
#SBATCH --error=example_%j.err
#SBATCH --mem-per-cpu=2400


# Module loading
module load Lmod/6.5
source $LMOD_PROFILE
ml intel/2017b Python/2.7.14

# Hostname list and environment variables for mpirun
echo $SLURM_JOB_NODELIST
rm hostnames.txt
scontrol show hostnames $SLURM_JOB_NODELIST > hostnames.txt
export I_MPI_PERHOST=$SLURM_NTASKS_PER_NODE

# Parameters for dmlc-tracker and mpirun
NUM_WORKERS=$SLURM_NTASKS
WORKER_MEMORY=$((2400*$SLURM_CPUS_PER_TASK))
WORKER_CORES=$SLURM_CPUS_PER_TASK 
HOST_FILE=hostnames.txt

# Execution with dmlc-submit
/usr/bin/time dmlc-core/tracker/dmlc-submit --cluster mpi --num-workers $NUM_WORKERS --worker-memory ${WORKER_MEMORY}m --worker-cores $WORKER_CORES --host-file $HOST_FILE python xgboost_distributed.py   

