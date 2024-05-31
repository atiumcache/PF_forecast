cd /projects/math_cheny/COVID_trend_detection/SMC_EPI

rm run_qsub_all.sh
for (( counter=0; counter<50; counter++ ))
do
rm job_state"$counter".sh

echo python main.py "$counter" > job_state_temp"$counter".sh
cat temp.sh job_state_temp"$counter".sh > job_state"$counter".sh
rm job_state_temp"$counter".sh
dos2unix job_state"$counter".sh
echo qsub job_state"$counter".sh >> run_qsub_all.sh
done
