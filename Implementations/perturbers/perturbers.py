from Abstract.Perturb import Perturb
from typing import List,Dict
import numpy as np
from utilities.Utils import Context,Particle

'''Multivariate normal perturbations to all parameters and state variables after log transform'''
class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self,ctx:Context,particleArray:List[Particle]):
        '''Randomly perturbs the parameters and state'''

        '''Constructs the diagonal variance-covariance matrix using the perturbation hyperparameters'''
        C = np.diag([(self.hyperparameters['sigma1']/ctx.population) ** 2,
                     self.hyperparameters['sigma1'] ** 2,
                     self.hyperparameters['sigma1'] ** 2,
                     self.hyperparameters['sigma1'] **2,
                     self.hyperparameters['sigma2'] ** 2]).astype(float)
        
        
        A = np.linalg.cholesky(C) #cholesky decomposition or SVD decomposition needs to be performed manually
        for i,_ in enumerate(particleArray): 

            #variation of the state and parameters

            '''concatenate the state and beta to get array equal to the mean of our multivariate normal implementation'''
            perturbed = np.log(np.concatenate((particleArray[i].state,[particleArray[i].param['beta']])))

            perturbed = np.exp(multivariate_normal(perturbed,A)) #
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population


            particleArray[i].state = perturbed[0:ctx.state_size]
            particleArray[i].param['beta'] = perturbed[-1]

            particleArray[i].dispersion = np.exp(ctx.rng.normal(np.log(particleArray[i].dispersion)))
            

        return particleArray
   