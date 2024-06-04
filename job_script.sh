#!/bin/bash

#SBATCH --output=/dev/null

# set the number of nodesT
#SBATCH --nodes=1

# set the number of cpus per node.
#SBATCH --mincpus=4

# set max wallclock time for the entire fitting job (1 days)
#SBATCH --time=0-20:00:00

# set name of job
#SBATCH --job-name=pf-flu-prediction

cd /projects/math_cheny/filter_forecast || exit

# Install python packages
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt

module load R/4.2.3
export PATH=$PATH:/home/yc424/R/4.2.3/

python3 cluster_single_loc_test.py