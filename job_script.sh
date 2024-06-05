#!/bin/bash

#SBATCH --job-name=pf-flu-prediction-test
#SBATCH --output=/scratch/apa235/test_output.txt
#SBATCH --nodes=1
#SBATCH --mincpus=4
#SBATCH --time=1:00:00
#SBATCH --chdir=/projects/math_cheny/filter_forecast/

# added echo statements for debugging

echo -e "Starting up...\n"
# Install python packages
module load anaconda3/2024.02
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt
echo -e "\n   Installed Python packages\n"

module load R/4.2.3
echo -e "\n   Loaded R\n"

echo -e "\n Running the Python script... \n"
python3 parallel_test.py
echo -e "\n   Completed job.\n"