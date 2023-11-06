from utilities.Utils import Particle,Context
from Abstract.Integrator import Integrator
from typing import List
import numpy as np


class EulerSolver(Integrator): 
    '''Uses the SIRSH model with a basic euler integrator to obtain the predictions for the state'''
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]: 

        dt = 0.01
        for particle in particleArray: 
            particle.observation = 0

        for _ in range(1/dt): 

            for j,particle in enumerate(particleArray): 
                '''This loop runs over the particleArray, performing the integration in RHS for each one'''
                d_RHS,sim_obv =self.RHS_H(particle)

                particleArray[j].state += d_RHS*dt
                particleArray[j].observation += np.array([sim_obv])

        return particleArray
    

    def RHS_H(self,particle:Particle):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = particle.state #unpack the state variables
        N = S + I + R + H #compute the total population

        new_H = ((1/particle.param['D'])*particle.param['gamma']) * I #our observation value for the particle  

        '''The state transitions of the ODE model is below'''
        dS = -particle.param['beta']*(S*I)/N + (1/particle.param['L'])*R 
        dI = particle.param['beta']*S*I/N-(1/particle.param['D'])*I
        dR = (1/particle.param['hosp']) * H + ((1/particle.param['D'])*(1-(particle.param['gamma']))*I)-(1/particle.param['L'])*R 
        dH = (1/particle.param['D'])*(particle.param['gamma']) * I - (1/particle.param['hosp']) * H 

        return np.array([dS,dI,dR,dH]),new_H
    
