# Particle Filter --- Forecasting and Automation
The goal of this project is to predict new flu cases using a particle filter and trend forecasting. 

We will set up an automated pipeline to:
- Collect new hospitalization data.
- Run the hospitalization data through a particle filter to predict the transmission rate.
- Use the transmission rate to predict future hospitalizations. 
- Determine forecast accuracy by comparing the predictions against observed data.
- Finally, compare accuracy results between particle filter and MCMC predictions. 

We will utilize a bash script to automate and parallelize most of this process on a HPC cluster. 

### Collect New Hospitalization Data
Download the new hospitalization reports and split the data into state-level data using the `process_new_data.py` script.

### Run the Particle Filter
For each location, we run the particle filter `particle_filter.py` on the state-level data. This outputs an inferred transmission rate `β`. 

### Forecasting
The `prog3_cpt-PLT-prd_v0-4.R` script uses trend forecasting to predict the transmission rate `β` up to 28 days into the future.

Then, `LSODA_forecast.py` uses the predicted `β` to forecast hospitalization rates 28 days into the future.

### Determine Accuracy
Use Weighted Interval Scores (WIS) to determine the accuracy of our forecasts. 

More information on WIS can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7880475/

### Compare Against MCMC Forecasting
We compare the WIS accuracy between particle filter and MCMC methods.

## Implementation Details

See `cluster_main.py` for the logic that runs the program on the HPC cluster.


## Particle Filter Credits
Particle filter code derived from:   

C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.

D. Calvetti, A. Hoover, J. Rose, and E. Somersalo. Bayesian particle filter algorithm for learning epidemic dynamics. Inverse Problems, 2021



