#!/bin/bash

#SBATCH --output=/dev/null

# set the number of nodesT
#SBATCH --nodes=1

# set the number of cpus per node.
#SBATCH --mincpus=4

# set max wallclock time for the entire fitting job (1 days)
#SBATCH --time=0-20:00:00

# set name of job
#SBATCH --job-name=flu-prediction

# Enable Anaconda Python 3.5
module purge
module load anaconda3

python3 fluSight_step_beta_all_free_11272022.py 8 7 1 
