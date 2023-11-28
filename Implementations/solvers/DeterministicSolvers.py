from utilities.Utils import Particle,Context
from Abstract.Integrator import Integrator
from scipy.integrate import solve_ivp,odeint
from typing import List
import numpy as np

class EulerSolver(Integrator): 
    '''Uses the SIRSH model with a basic euler integrator to obtain the predictions for the state'''
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]: 

        dt = 1
        for particle in particleArray: 
            particle.observation = 0

        for j,_ in enumerate(particleArray): 
            for _ in range(int(1/dt)): 
                '''This loop runs over the particleArray, performing the integration in RHS for each one'''
                d_RHS,sim_obv =self.RHS_H(particleArray[j])

                particleArray[j].state += d_RHS*dt
                if(np.any(np.isnan(particleArray[j].state))): 
                    print(f"NaN state at particle: {j}")
                particleArray[j].observation += np.array([sim_obv])


        return particleArray
    

    def RHS_H(self,particle:Particle):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = particle.state #unpack the state variables
        N = S + I + R + H #compute the total population

        new_H = ((1/particle.param['D'])*particle.param['gamma']) * I #our observation value for the particle  
        new_I = particle.param['beta']*S*I/N

        '''The state transitions of the ODE model is below'''
        dS = -particle.param['beta']*(S*I)/N + (1/particle.param['L'])*R 
        dI = particle.param['beta']*S*I/N-(1/particle.param['D'])*I
        dR = (1/particle.param['hosp']) * H + ((1/particle.param['D'])*(1-(particle.param['gamma']))*I)-(1/particle.param['L'])*R 
        dH = (1/particle.param['D'])*(particle.param['gamma']) * I - (1/particle.param['hosp']) * H 

        return np.array([dS,dI,dR,dH]),new_I
    
class Rk45Solver(Integrator): 

    '''Runge Kutta algorithm for computing the t->t+1 transition'''
    def __init__(self) -> None:
        super().__init__()

    '''Elements of particleArray are of Particle class in utilities/Utils.py'''
    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]: 


        for i,particle in enumerate(particleArray): 

            y0 = np.concatenate((particle.state,particle.observation))  # Initial state of the system
            
            t_span = (0.0, 1.0)
            result_solve_ivp = solve_ivp(RHS_H, t_span, y0, args=(particle.param,))

            particleArray[i].state = np.array(result_solve_ivp.y[0:ctx.state_size,-1])
            particleArray[i].observation = np.array([result_solve_ivp.y[-1,-1]])


            if(np.any(np.isnan(particleArray[i].state))): 
                    print(f"NaN state at particle: {i}")


        return particleArray

class EulerSolver_SEAIRH(Integrator):

    '''Uses the SIRSH model with a basic euler integrator to obtain the predictions for the state'''

    def __init__(self) -> None:
        super().__init__()

 

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''

    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]:

 

        dt = 1/100

        for particle in particleArray:

            particle.observation = 0

 

       

        for j,_ in enumerate(particleArray):

            for _ in range(int(1/dt)):

                '''This loop runs over the particleArray, performing the integration in RHS for each one'''

                d_RHS,sim_obv =self.RHS(particleArray[j])

                particleArray[j].state += d_RHS*dt
                if(np.any(np.isnan(particleArray[j].state))): 
                    print(f"NaN state at particle: {j}")
                particleArray[j].observation += np.array([sim_obv])

 

        return particleArray

   

 

    def RHS(t,particle:Particle):

    #params has all the parameters – beta, gamma

    #state is a numpy array

 

        S,E,A,I,H,R, D = particle.state #unpack the state variables

        N = S + E + A + I + R + H + D  #compute the total population

        kL=0.25

        fA=0.44

        fH=0.054

        fR=0.79

        cA=0.26

        cI=0.12

        cH=0.17

 
        '''The state transitions of the ODE model is below'''

        dS = -particle.param['beta']*(S*I)/N

        dE = particle.param['beta']*S*I/N-kL*E

        dA = kL*fA*E-cA*A

        dI = kL*(1-fA)*E-cI*I # compare the I compartment to the reported case number, this model works for case number comparison, may not work for hospitalization number

        dH = cI*fH*I-cH*fR * H

        dR = cA*A+cI*(1-fH)*I+cH*fR*H

        dD = cH*(1-fR)*H

        new_I = kL*(1-fA)*E
        return np.array([dS,dE,dA,dI,dH,dR,dD]),new_I

class EulerSolver_SIR(Integrator): 
    '''Uses the SIRSH model with a basic euler integrator to obtain the predictions for the state'''
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]: 

        dt = 1
        for particle in particleArray: 
            particle.observation = 0

        for j,_ in enumerate(particleArray): 
            for _ in range(int(1/dt)): 
                '''This loop runs over the particleArray, performing the integration in RHS for each one'''
                d_RHS,sim_obv =self.RHS(particleArray[j])

                particleArray[j].state += d_RHS*dt
                if(np.any(np.isnan(particleArray[j].state))): 
                    print(f"NaN state at particle: {j}")
                particleArray[j].observation = particleArray[j].state[1]

        return particleArray
    
    def RHS(self,particle:Particle):
        
        S,I,R = particle.state
        N = S + I + R

        new_I = particle.param['beta']*S*I/N - particle.param['gamma'] * I

        dS = -particle.param['beta']*S * I/N
        dI = particle.param['beta']*S*I/N - particle.param['gamma'] * I
        dR = particle.param['gamma'] * I
 
        return np.array([dS,dI,dR]),new_I
    

    

    
