# Particle Filter --- Forecasting Hospitalizations
The goal of this project is to predict new flu cases using a particle 
filter and trend forecasting. We model the state of our system using a SIRH 
model, and infer the transmission rate using a particle filter.

This repository implements an automated pipeline to:
- Collect new hospitalization data.
- Run the hospitalization data through a particle filter to estimate the transmission rate.
- Forecast future transmission rates. 
- Use the forecasted transmission rates to predict future hospitalizations. 

We utilize a bash script (`./job_script.sh`) to automate and parallelize 
most of this process on a HPC cluster. 

### Collect New Hospitalization Data
Download the new hospitalization reports and split the data into 
state-level data using the `filter_forecast.process_new_data` script.

### Run the Particle Filter
For each location, we run the particle filter on the state-level data. This outputs an inferred transmission rate `β`. 
The particle filter is implemented in the `filter_forecast.particle_filter` package.  

### Forecasting
The `./r_scripts/beta_trend_forecast.R` script uses changepoint detection and 
trend 
forecasting to predict the transmission rate `β` for 28 days into the future.

Then, `filter_forecast.hosp_forecast` uses the predicted `β` to forecast hospitalization rates 28 days into the future.

### Determine Accuracy
Use Weighted Interval Scores (WIS) to determine the accuracy of our forecasts. This is performed in the [Flu Forecast Accuracy repository](https://github.com/atiumcache/flu-forecast-accuracy). We also compare this method with 
MCMC forecasting.

More information on WIS can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7880475/

## Implementation Details
See `./forecast_all_states.py` for the logic that runs the full pipeline on the 
HPC cluster. This script is called from the `./job_script.sh` shell script.

## Particle Filter Credits
Particle filter code derived from:   

C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.

D. Calvetti, A. Hoover, J. Rose, and E. Somersalo. Bayesian particle filter algorithm for learning epidemic dynamics. Inverse Problems, 2021



