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
                        static_params.append(val)

                    elif(ctx.estimated_params[key] == ESTIMATION.VARIABLE): 
                        var_names.append(key)
                        variable_params.append(val)

            static_param_mat.append(static_params)
            var_param_mat.append(variable_params)

        var_names = OrderedSet(var_names)
        static_names = OrderedSet(static_names)

        static_param_mat = np.log(np.array(static_param_mat))

        var_param_mat = np.log(np.array(var_param_mat))


        '''Computes the log_mean as defined in Calvetti et.al. '''
        log_mean = 0
        for i,param_vec in enumerate(static_param_mat): 
            log_mean += ctx.weights[i] * (param_vec)

        '''Computes the covariance of the logarithms of the particles'''
        cov = 0
        for i,param_vec in enumerate(static_param_mat): 
            cov += ctx.weights[i] * np.outer(param_vec-log_mean,param_vec-log_mean)

        '''Holds the hyperparameter a, defined in terms of h'''
        a = np.sqrt(1-(self.hyperparameters["h"])**2)

        # '''TODO for some reason the multivariate normal distribution isn't working in the 1-dimension case, need to investigate'''
        for i in range(len(particleArray)): 
            new_statics = ctx.rng.multivariate_normal(a * static_param_mat[i] + (1-a)*log_mean,(self.hyperparameters["h"]**2)*cov)
        #     new_statics = ctx.rng.normal(static_param_mat[i])

            '''puts the perturbed static parameters back in the particle field'''
            for j,static in enumerate(new_statics): 
                particleArray[i].param[static_names[j]] = np.exp(static)


        '''Perturb the variable parameters '''


        state1 = np.array([self.hyperparameters['sigma1']/ctx.population]) ** 2
        otherstates = np.array([self.hyperparameters['sigma1']**2 for _ in range(ctx.state_size-1)])
        param_variance = np.array([self.hyperparameters['sigma2']**2 for _ in range(len(var_names))])

        C = np.concatenate((state1,otherstates,param_variance))

        C = np.diag(C).astype(float)


        '''Main perturbation loop'''
        for i in range(len(particleArray)): 
            log_state = np.log(particleArray[i].state)
            td_vec = (np.concatenate((log_state,var_param_mat[i])))
            perturbed = np.exp(ctx.rng.multivariate_normal(td_vec,C))

            '''Normalization'''
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population


            particleArray[i].state = perturbed[:ctx.state_size]
            for j,name in enumerate(var_names): 
                particleArray[i].param[name] = perturbed[ctx.state_size+j:ctx.state_size+j+1][0]

        return particleArray


        
   