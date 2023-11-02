from utilities.Utils import Particle,Context
from Abstract.Resampler import Resampler
from utilities.Utils import log_norm,jacob
from scipy.stats import nbinom
from typing import List
import numpy as np
from numpy.typing import NDArray

def log_likelihood_NB(observation,particle_observation:NDArray[np.int_],var:float)->NDArray: 
    '''a wrapper for the logpmf of the negative binomial distribution, modified to use the parameterization aobut a known mean 
    and variance, r and p solved accordingly'''
    return nbinom.logpmf(observation, n = (particle_observation)**2 / (var  - particle_observation), p = particle_observation / var)


class LogNBinomResample(Resampler): 
  
    def __init__(self) -> None:
        super().__init__(log_likelihood_NB)

    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> float:
        p_obvs = np.array([particle.observation for particle in particleArray])
        weights = np.zeros(len(p_obvs))
        for i,particle in enumerate(particleArray):
            for j in range(len(particle.observation)): 
                weights[i] += (self.likelihood(observation=observation[j],particle_observations=particle.observation[j]))

        weights = weights-np.max(weights)
        weights = log_norm(weights)
        
        return weights
    
    def resample(self, ctx: Context,particleArray:List[Particle],weights:NDArray) -> List[Particle]:
        
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



    
