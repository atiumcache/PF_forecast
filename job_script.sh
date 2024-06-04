cd /projects/math_cheny/filter_forecast

# Install python packages
python3 -m ensurepip
python3 -m pip install -r ./requirements.txt

module load R/4.2.3
export PATH=$PATH:/home/yc424/R/4.2.3/

python3 cluster_single_loc_test.py