from Abstract.Perturb import Perturb
from typing import List,Dict
import numpy as np
from utilities.Utils import Context,Particle,ESTIMATION

'''Multivariate normal perturbations to all parameters and state variables after log transform'''
class MultivariatePerturbations(Perturb): 
    def __init__ (self,hyper_params:Dict)-> None: 
        '''The perturber has special hyperparameters which tell randomly_perturb how much to move the parameters and state'''
        super().__init__(hyper_params=hyper_params)


    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        '''Implementations of this method will take a list of particles and perturb it according to a user defined distribution'''
        static_param_mat = []
        for particle in particleArray: 
            static_params = []
            variable_params = []
            for _,(key,val) in enumerate(particle.param.items()):
                if(key in ctx.estimated_params): 
                    if(ctx.estimated_params[key] == ESTIMATION.STATIC): 
                        static_params.append(np.array([val]))
                    elif(ctx.estimated_params[key] == ESTIMATION.VARIABLE): 
                        variable_params.append(val)
            static_param_mat.append(static_params)

        static_param_mat = np.array(static_param_mat).squeeze(axis=2)
        log_mean  = ctx.weights*np.sum(np.log(static_param_mat))
        


            

        
   