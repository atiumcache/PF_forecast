'''Stochastic analog to the euler solver for Alex and Kayodes SIRH model'''
from utilities.Utils import Particle,Context
from Abstract.Integrator import Integrator

from typing import List
import numpy as np




class PoissonSolver(Integrator): 
    '''This class uses the tau-leaping method to compute the next state of the particle and the observations, 
    i.e. a poisson stochastic propagation model'''
    def propagate(self, particleArray: List[Particle],ctx:Context) -> List[Particle]:
        '''Implementation of the one step propagation function from t to t+1'''

        tau = 1

        for j,particle in enumerate(particleArray): 

            particleArray[j].observation = np.array([0 for _ in range(len(particleArray[j].observation))])
            
            for _ in range(int(1/tau)):

                state,hospitalized,new_hospitalized = self.RHS(particleArray[j],ctx,tau)
                particleArray[j].state = state
                particleArray[j].observation += [hospitalized,new_hospitalized]


        return particleArray
    
    def RHS(self,particle:Particle,ctx:Context,tau:float):
        S,I,R,H = particle.state
        N = S+I+R+H
        new_susceptibles = ctx.rng.poisson(((1/particle.param['L'])*R) * tau)
        new_infected = ctx.rng.poisson((((particle.param['beta'] * S * I)/N)) * tau)
        new_recovered_from_H = ctx.rng.poisson(((1/particle.param['hosp']) * H) * tau)
        new_recovered_from_I = ctx.rng.poisson((((1/particle.param['D']) * (1-particle.param['gamma']))*I)*tau) 
        new_hospitalized = ctx.rng.poisson(((1/particle.param['D']) * particle.param['gamma'] * I)*tau)
            
        state = np.zeros_like(particle.state)
        '''Enforces the minimum will be 0'''
        state[0] = max(0.,S-new_infected + new_susceptibles)
        state[1] = max(0.,I + new_infected - (new_hospitalized + new_recovered_from_I))
        state[2] = max(0.,R + new_recovered_from_H + new_recovered_from_I - new_susceptibles)
        state[3] = max(0.,H + new_hospitalized - new_recovered_from_H)

        return state,int(state[3]),new_hospitalized
    
