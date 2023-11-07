from Abstract.Perturb import Perturb
from typing import List,Dict
from ordered_set import OrderedSet
import numpy as np
from utilities.Utils import Context,Particle,ESTIMATION

'''Multivariate normal perturbations to all parameters and state variables after log transform'''
class MultivariatePerturbations(Perturb): 
    def __init__ (self,hyper_params:Dict)-> None: 
        '''The perturber has special hyperparameters which tell randomly_perturb how much to move the parameters and state'''
        super().__init__(hyper_params=hyper_params)


    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        '''Implementations of this method will take a list of particles and perturb it according to a user defined distribution'''

        '''All this craziness is just used to extract the static and time-varying parameters from each particle and put them in lists'''
        static_param_mat = []
        static_names = []

        var_param_mat = []
        var_names = []

        for particle in particleArray: 
            static_params = []
            variable_params = []

            for _,(key,val) in enumerate(particle.param.items()):

                if(key in ctx.estimated_params): 

                    if(ctx.estimated_params[key] == ESTIMATION.STATIC):
                        static_names.append(key)
                        static_params.append(np.array([val]))

                    elif(ctx.estimated_params[key] == ESTIMATION.VARIABLE): 
                        var_names.append(key)
                        variable_params.append(val)

            static_param_mat.append(static_params)
            var_param_mat.append(variable_params)

        var_names = OrderedSet(var_names)
        static_names = OrderedSet(static_names)

        static_param_mat = np.log(np.array(static_param_mat).squeeze(axis=2))
        #var_param_mat = np.log(np.array(var_param_mat).squeeze(axis=2))

        '''Computes the log_mean as defined in Calvetti et.al. '''
        log_mean = 0
        for i,param_vec in enumerate(static_param_mat): 
            log_mean += ctx.weights[i] * (param_vec)

        '''Computes the covariance of the logarithms of the particles'''
        cov = 0
        for i,param_vec in enumerate(static_param_mat): 
            cov += ctx.weights[i] * (param_vec-log_mean) * (param_vec - log_mean).T

        '''Holds the hyperparameter a, defined in terms of h'''
        a = np.sqrt(1-self.hyperparameters["h"]**2)

        
        '''TODO for some reason the multivariate normal distribution isn't working in the 1-dimension case, need to investigate'''
        for i in range(len(particleArray)): 
            new_statics = ctx.rng.normal(a * static_param_mat[i] + (1-a)*log_mean,(self.hyperparameters["h"]**2)*cov)

            '''puts the perturbed static parameters back in the particle field'''
            for j,static in enumerate(new_statics): 
                particleArray[i].param[static_names[j]] = np.exp(static)

        
   