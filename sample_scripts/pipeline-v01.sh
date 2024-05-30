##Step 1. Download data
####Download the latest hospitalization data file manually
#https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh
cd /projects/math_cheny/FluSight/Data
rm rows.csv?accessType=DOWNLOAD
wget https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD

##Step 2. Generate the shell job script
#generate the qsub commands for Monsoon
cd /projects/math_cheny/FluSight/2023_2024
rm run_qsub_all_no_d.sh
rm *MCMC_day250_2x_MSA*

date=$(date '+%Y-%m-%d')
sed '$d' MCMC_test_vac.sh > temp_1st.sh
add=0
for (( counter=0; counter<54; counter++ ))
do
#for (( add=0; add<1; add++ ))
#do
echo "python3 fluSight_mcmc_N4_v03.py $counter 4 $add" > temp_file"$counter"_$add.txt
cat temp_1st.sh temp_file"$counter"_$add.txt > ${date}_State${counter}_$add.sh
echo "qsub ${date}_State${counter}_$add.sh" >> run_qsub_all_no_d.sh

#done
done

rm temp_*

## step 3. National ensemble
cd /projects/math_cheny/FluSight/2023_2024
module load anaconda3
python3 national_combine_quantiles.py

##Step 4. Compute rate trend pmf
python3 hospital_rate_trend_pmf_Wednesday.py

##Step 5. Calibration
rm run_qsub_all.sh
rm *MCMC_day250_2x_MSA*
i=1
date=$(date '+%Y-%m-%d')
sed '$d' MCMC_test_vac.sh > temp_1st.sh
for (( counter=0; counter<54; counter++ ))
do
echo "python3 calibration_weekly.py $counter " > temp_file"$counter".txt
cat temp_1st.sh temp_file"$counter".txt > ${date}_State${counter}_run_${i}.sh
echo "qsub ${date}_State${counter}_run_${i}.sh" >> run_qsub_all.sh

done

rm temp_*


##Step 6 QC the results
#After the jobs executed in Monsoon, make plots.
#This steps generate quality plot, check if the results are reasonable by comparing the actual results and the prediction percentiles.
cd /projects/math_cheny/FluSight/2023_2024
module load anaconda3
python3 plot_UQ_weekly.py
python3 plot_UQ_weekly-calibrated.py
python3 plot_UQ_daily_v03.py
python3 plot_trace_loglikelihood_states.py

##Step 7. Generate the percentile result for Github.
cd /projects/math_cheny/FluSight/2023_2024/mcmc_N4
cp *calibrated.txt monsoon_percentiles/
cp *rate-trend.txt monsoon_percentiles/

##Step 8. Copy the quantile data to local drive and generate the final csv table
cd /projects/math_cheny/FluSight/2023_2024
module load R/4.2.3
export PATH=$PATH:/home/yc424/R/4.2.3/
 #Avery, please add your path after you install the R packages, do not delete this line

Rscript convert-Monsoon-ouput-to-FluSight-Calib_Wednesday.R


