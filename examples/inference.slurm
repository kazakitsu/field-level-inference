#!/bin/bash

#SBATCH --job-name=field_level
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kakitsu@ias.edu
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=1

hostname; pwd; date

module load cudatoolkit/11.7
source /home/kakitsu/python/jax04_env39/bin/activate

cd /home/kakitsu/field_level/field_level

nvidia-smi

echo
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

date

python run_inference.py ./examples/params_gauss_example.dat

date
