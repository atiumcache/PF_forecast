# Particle Filter --- Forecasting and Automation
The goal of this project is to predict new flu cases using particle filtering and trend forecasting. 

We will set up an automated pipeline to:
- Collect new hospitalization data.
- Run the hospitalization data through a particle filter to predict the transmission rate.
- Use the transmission rate to predict future hospitalizations. 
- Determine forecast accuracy by comparing the predictions against observed data.
- Finally, compare accuracy results between particle filter and MCMC predictions. 

We will utilize a bash script to automate this process on NAU's Monsoon HPC. 

### Collect New Hospitalization Data
Download the new hospitalization reports and split the data into state-level data using the `process_new_data.py` script.

### Run the Particle Filter
For each state, we run the particle filter `main.py` on the state-level data. This outputs a predicted transmission rate `β`. 

### Forecasting
Using the `prog3_cpt-PLT-prd_v0-4.R` script, we use the predicted transmission rate `β` to forecast hospitalization rates 28 days into the future.

### Determine Accuracy
Use Weighted Interval Scores (WIS) to determine the accuracy of our forecasts. More information on WIS can be found here:

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7880475/

### Compare Against MCMC Forecasting
We compare the WIS accuracy between particle filter and MCMC methods.

## Particle Filter Information
Particle filter code derived from:   

C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.

D. Calvetti, A. Hoover, J. Rose, and E. Somersalo. Bayesian particle filter algorithm for learning epidemic dynamics. Inverse Problems, 2021

The code is structured in a class hierarchy, abstract base classes (ABC) are used to template implementations of the basic algorithm. 

Abstract/Algorithm: 
This ABC is the template for the entrypoint of the algorithm, it has 5 fields, 3 of which all also ABC's templating subfunctionality required for running the algorithm. 

Abstract/Perturb: 
This ABC is one of the fields of Algorithm. It templates a class which perturbs the particles according to some distribution. This class templates only one method, "randomly_perturb", and one field, 
hyperparameters, which is a dictionary of hyperparameters for the perturbation. 

Abstract/Resampler: 
This ABC is one of the fields of Algorithm, it performs the resampling step of the particle filter and templates two functions, "compute_weights" to compute the particle weights using the likelihood function, and "resample" to perform the actual resampling. 




