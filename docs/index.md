---
title: Overview
---

Our prediction pipeline is split into 3 distinct sections:
- [Log PF](/PF_forecast/log_pf)
- [Trend Forecasting](/PF_forecast/trend_forecasting)
- [Hospitalization Predicting](/PF_forecast/hosp_predictions)

We employ a HPC cluster to run multiple pipelines in parallel. We can run 
each location's pipeline separately, but that location's pipeline must be 
run sequentially (we can't predict hospitalizations before we have a 
predicted \beta). 

Each pipeline/location runs on a node on the cluster. 

