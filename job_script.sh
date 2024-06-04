#!/bin/bash

#SBATCH --job-name=pf-flu-prediction
#SBATCH --output=/scratch/apa235
#SBATCH --nodes=1
#SBATCH --mincpus=4
#SBATCH --time=0-20:00:00

# added echo statements for debugging

echo "Working..."
cd /projects/math_cheny/filter_forecast
echo "Changed directory"

# Install python packages
module load python3
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
echo "Installed Python packages"

module load R/4.2.3
export PATH=$PATH:/home/yc424/R/4.2.3/
echo "Loaded R"

python3 cluster_single_loc_test.py
echo "Completed job."