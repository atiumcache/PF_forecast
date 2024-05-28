import numpy as np
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import pandas as pd


def main(state_abbrev):

    '''Read in the necessary csv files'''

    predicted_beta = pd.read_csv('./datasets/Out_prog3/out_logit-beta_trj_rnorm.csv').to_numpy()
    predicted_beta = np.delete(predicted_beta,0,1)

    observations = pd.read_csv(f'./datasets/{state_abbrev}_FLU_HOSPITALIZATIONS.csv').to_numpy()
    observations = np.delete(observations,0,1)

    estimated_state = pd.read_csv('./datasets/ESTIMATED_STATE.csv').to_numpy()
    estimated_state = np.delete(estimated_state,0,1)

    pf_beta = pd.read_csv('./datasets/average_beta.csv').to_numpy()
    pf_beta = np.delete(pf_beta,0,1).squeeze()


    endpoint = 80

    '''Setup the time span up to the final data point and the forecast'''
    t_span = [0,endpoint]
    forecast_span = [endpoint,endpoint+26]


    '''Plots the beta estimate from Jaechouls code '''
    for i in range(10):
        plt.plot(predicted_beta[i,:])
    plt.show()


    '''Model definition for the integrator'''
    def RHS_H(t,state,param):
        #params has all the parameters – beta, gamma
        #state is a numpy array

            S,I,R,H,new_H = state #unpack the state variables
            N = S + I + R + H #compute the total population 

            new_H = (1/param['D'])*(param['gamma']) * I

            '''The state transitions of the ODE model is below'''
            dS = -param['beta'](int(t))*(S*I)/N + (1/param['L'])*R 
            dI = param['beta'](int(t))*S*I/N-(1/param['D'])*I
            dR = (1/param['hosp']) * H + ((1/param['D'])*(1-(param['gamma']))*I)-(1/param['L'])*R 
            dH = (1/param['D'])*(param['gamma']) * I - (1/param['hosp']) * H 

            return np.array([dS,dI,dR,dH,new_H])

    '''Functional form of beta to use for integration'''

    def beta(t):  
        if(t < t_span[1]): 
            return pf_beta[t]
        else:
            return predicted_beta[5,t-forecast_span[0]]

    par={
    "beta":beta,
    "gamma":0.06,
    "hosp":10,
    "L":90,
    "D":10}

    '''Solve the system through the forecast time'''
    forecast = solve_ivp(fun=lambda t,z: RHS_H(t,z,par), 
                            t_span=[forecast_span[0],forecast_span[1]],
                            y0=np.concatenate((estimated_state[forecast_span[0]],observations[forecast_span[0]])),
                            t_eval = np.linspace(forecast_span[0],forecast_span[1],forecast_span[1] - forecast_span[0]),
                            method='RK45').y


    '''Plotting'''

    labels = ['Real: S','real: I','Real: R','Real: H']
    for i in range(4):
        plt.plot(estimated_state[t_span[0]:forecast_span[1],i],label = labels[i])
        plt.plot(np.arange(forecast_span[0],forecast_span[1]),forecast[i,:])
        plt.legend()
        plt.show()


    plt.title("New Hospitalizatons")
    plt.plot(np.arange(forecast_span[0],forecast_span[1]-1),np.diff(forecast[4,:]))
    plt.plot(observations[t_span[0]:forecast_span[1]])
    plt.legend()


