---
title: Overview
---

The goal of this project is to predict new flu cases using a particle 
filter and trend forecasting. We model the state of our system using a SIRH 
model, and infer the transmission rate using a particle filter.

This repository implements an automated pipeline to:
- Collect new hospitalization data.
- Run the hospitalization data through a particle filter to estimate the transmission rate.
- Forecast future transmission rates. 
- Use the forecasted transmission rates to predict future hospitalizations. 

We utilize a bash script (`./job_script.sh`) to automate and parallelize 
most of this process on an HPC cluster. 

## Sections
Our prediction pipeline is split into distinct sections:
- [Gather/Process Data](/PF_forecast/process_data)
- [Log PF](/PF_forecast/log_pf)
- [Trend Forecasting](/PF_forecast/trend_forecasting)
- [Hospitalization Predicting](/PF_forecast/hosp_predictions)

### Parallel Processing
We employ an HPC cluster to run multiple pipelines in parallel. We can run 
each location's pipeline separately, but that location's pipeline must be 
run sequentially (we can't predict hospitalizations before we have a 
predicted $\beta$).