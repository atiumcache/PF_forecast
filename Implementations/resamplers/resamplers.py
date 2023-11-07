from utilities.Utils import Particle,Context
from Abstract.Resampler import Resampler
from utilities.Utils import log_norm,jacob
from scipy.stats import nbinom,poisson
from typing import List
import numpy as np
from numpy.typing import NDArray

def log_likelihood_NB(observation,particle_observation:NDArray[np.int_],std:float)->NDArray: 
    '''a wrapper for the logpmf of the negative binomial distribution, modified to use the parameterization aobut a known mean 
    and standard deviation, r and p solved accordingly'''
    return nbinom.pmf(observation, n = (particle_observation)**2 / (std**2  - particle_observation), p = particle_observation / std**2)

def log_likelihood_poisson(observation,particle_observation):
    return poisson.pmf(observation,particle_observation)

class NBinomResample(Resampler): 
    def __init__(self) -> None:
        super().__init__(log_likelihood_NB)

    '''weights[i] += (self.likelihood(observation=observation[j],
                                               particle_observation=particle.observation[j],
                                               std=particle.param['std']))'''
    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros
        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(observation,particleArray[i].observation,std=particleArray[i].param['std'])
            '''iterate over the particles and call the likelihood function for each one '''


        #weights = np.array(self.likelihood(np.round(observation),[particle.observation for particle in particleArray],[particle.dispersion for particle in particleArray]))

        '''This loop sets all weights that are out of bounds to a very small non-zero number'''
        for j in range(len(particleArray)):  
            if(weights[j] == 0):
                weights[j] = 10**-300 
            elif(np.isnan(weights[j])):
                weights[j] = 10**-300
            elif(np.isinf(weights[j])):
                weights[j] = 10**-300

        weights = weights/np.sum(weights)#normalize the weights

        
        return np.squeeze(weights)
    
    def resample(self, ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        '''This is a basic resampling method, more advanced methods like systematic resampling need to override this'''    
    
        indexes = np.arange(ctx.particle_count) #create a cumulative ndarray from 0 to particle_count

        #The numpy resampling algorithm, see jupyter notebnook resampling.ipynb for more details
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=ctx.weights)

        particleCopy = particleArray.copy()#copy the particle array refs to ensure we don't overwrite particles

        #this loop reindexes the particles by rebuilding the particles
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),
                                        particleCopy[new_particle_indexes[i]].state.copy(),
                                        particleCopy[new_particle_indexes[i]].observation)


        

        return particleArray



#TODO Sit on this until after we test the static parameter estimatiion
class LogNBinomResample(Resampler): 
    '''Resampler using a negative binomial likelihood function with estimated variance and log resampling step from 
    C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.'''
    def __init__(self) -> None:
        super().__init__(log_likelihood_NB)

    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        p_obvs = np.array([particle.observation for particle in particleArray])
        weights = np.zeros(len(p_obvs))
        for i,particle in enumerate(particleArray):
            for j in range(len(particle.observation)): 
                weights[i] += (self.likelihood(observation=observation[j],
                                               particle_observations=particle.observation[j],
                                               var=particle.param['std']))

        weights = weights-np.max(weights) #normalize the weights wrt their maximum, improves numerical stability
        weights = log_norm(weights) #normalize the weights using the jacobian logarithm
        
        return weights
    
    def resample(self, ctx: Context,particleArray:List[Particle],weights:NDArray) -> List[Particle]:
        '''The actual resampling algorithm, the log variant of systematic resampling'''
        ctx.weights = weights
        log_cdf = jacob(weights)
        
        i = 0
        indices = np.zeros(ctx.particle_count)
        u = ctx.rng.uniform(0,1/ctx.particle_count)
        for j in range(0,ctx.particle_count): 
            r = np.log(u + 1/ctx.particle_count * j)
            while r > log_cdf[i]: 
                i += 1
            indices[j] = i

        indices=indices.astype(int)
        particleCopy = particleArray.copy()
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[indices[i]].param.copy(),particleCopy[indices[i]].state.copy(),particleCopy[indices[i]].observation)

        return particleArray



    
