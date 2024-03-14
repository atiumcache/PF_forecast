from utilities.Utils import Particle,Context,SMCContext
from Abstract.Resampler import Resampler
from utilities.Utils import log_norm,jacob
from scipy.stats import nbinom,poisson
from scipy.special import loggamma
from typing import List
import numpy as np
import math
from numpy.typing import NDArray

''''''
def likelihood_NB(observation,particle_observation:NDArray[np.int_],var:float)->NDArray: 
    '''a wrapper for the pmf of the negative binomial distribution, modified to use the parameterization aobut a known mean 
    and variance, r and p solved accordingly'''
    X = nbinom.pmf(observation, n = (particle_observation)**2 / (var  - particle_observation), p = particle_observation / var)

    return X

def likelihood_NB_R(observation,particle_observation:NDArray[np.int_],R:float)->NDArray: 
    X = nbinom.pmf(observation, n = R, p = R/(particle_observation+R))
    return X

def likelihood_poisson(observation,particle_observation,var):
    return poisson.pmf(observation,particle_observation)

def likelihood_NB_r(observation,particle_observation:NDArray[np.int_],R:float)->NDArray: 

    prob = np.array([R/(R+particle_observation)])
    prob[prob<=1e-10] = 1e-10
    prob[prob>=1-1e-10] = 1-1e-10
    v1 = prob[observation>=0] #do not include the days if observation is negative
    v2 = observation[observation>=0]
    x = loggamma(v2+R)-loggamma(v2+1)-loggamma(R)+R*np.log(v1)+v2*np.log(1-v1)

    return np.exp(x)

def log_likelihood_NB(observation:NDArray[np.int_],particle_observation:NDArray[np.int_],R:float)->int: 
    
    prob = np.array([R/(R+particle_observation)])
    prob[prob<=1e-10] = 1e-10
    prob[prob>=1-1e-10] = 1-1e-10
    v1 = prob[observation>=0] #do not include the days if observation is negative
    v2 = observation[observation>=0]
    x = loggamma(v2+R)-loggamma(v2+1)-loggamma(R)+R*np.log(v1)+v2*np.log(1-v1)

    return x


class NBinomResample(Resampler): 
    def __init__(self) -> None:
        super().__init__(likelihood=likelihood_NB_r)

    '''weights[i] += (self.likelihood(observation=observation[j],
                                               particle_observation=particle.observation[j],
                                               std=particle.param['std']))'''
    def compute_prior_weights(self, ctx:Context,observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros

        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])
            '''iterate over the particles and call the likelihood function for each one '''

        
        #max_particle = particleArray[np.argmax(weights)]

        # R = (max_particle.observation)**2 / (  - max_particle.observation)
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
    
    def compute_pos_weights(self,observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros

        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])
            '''iterate over the particles and call the likelihood function for each one '''

        
        #max_particle = particleArray[np.argmax(weights)]

        # R = (max_particle.observation)**2 / (  - max_particle.observation)
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
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=ctx.prior_weights)

        # Add new indices as a column in the sankey matrix
        ctx.sankey_indices.append(new_particle_indexes)

        particleCopy = particleArray.copy()#copy the particle array refs to ensure we don't overwrite particles

        #this loop reindexes the particles by rebuilding the particles
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),
                                        particleCopy[new_particle_indexes[i]].state.copy(),
                                        particleCopy[new_particle_indexes[i]].observation)

        return particleArray

class NBinomResampleR(Resampler):
    def __init__(self) -> None:
        super().__init__(likelihood=likelihood_NB_r)

    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros
        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,R=1/(particleArray[i].param['R']))
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

class LogNBinomResample(Resampler): 
    '''Resampler using a negative binomial likelihood function with estimated variance and log resampling step from 
    C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.'''
    def __init__(self) -> None:
        super().__init__(log_likelihood_NB)

    def compute_prior_weights(self, ctx:Context,observation: NDArray[np.int_], particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))

        for i,particle in enumerate(particleArray):
            LL = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])
                
            weights[i] = LL

            if(math.isnan(weights[i])): 
                print(f"real obv: {np.round(observation)}")
                print(f"particle obv: {np.round(particle.observation)}")

        

        #weights = weights-np.max(weights) #normalize the weights wrt their maximum, improves numerical stability
        weights = log_norm(weights) #normalize the weights using the jacobian logarithm
        
        return weights
    
    def compute_pos_weights(self, observation: NDArray[np.int_], particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))

        for i,particle in enumerate(particleArray):


            LL = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])
                
            weights[i] = LL

            if(math.isnan(weights[i])): 
                print(f"real obv: {np.round(observation)}")
                print(f"particle obv: {np.round(particle.observation)}")

        

        #weights = weights-np.max(weights) #normalize the weights wrt their maximum, improves numerical stability
        weights = log_norm(weights) #normalize the weights using the jacobian logarithm
        
        return weights
    
    def resample(self, ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        '''The actual resampling algorithm, the log variant of systematic resampling'''
        log_cdf = jacob(ctx.prior_weights)
        
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

class PoissonResample(Resampler):
    def __init__(self) -> None:
        super().__init__(likelihood=likelihood_poisson)

    def compute_prior_weights(self, ctx:Context, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros
        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,var = 0)
            '''iterate over the particles and call the likelihood function for each one '''

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
    
    def compute_pos_weights(self, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros
        for i in range(len(particleArray)): 
            obv = particleArray[i].param['eta'] * particleArray[i].state[1]
            weights[i] = self.likelihood(np.round(observation),obv,var =0)
            '''iterate over the particles and call the likelihood function for each one '''

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
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=ctx.prior_weights)



        particleCopy = particleArray.copy()#copy the particle array refs to ensure we don't overwrite particles

        #this loop reindexes the particles by rebuilding the particles
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),
                                        particleCopy[new_particle_indexes[i]].state.copy(),
                                        particleCopy[new_particle_indexes[i]].observation)


        

        return particleArray

class SMCResample:
    def __init__(self) -> None:
        self.likelihood=likelihood_poisson

    def compute_weights(self,ctx:SMCContext,observation:NDArray,particleArray:List[Particle]) ->NDArray:
        for i,particle in enumerate(particleArray):
            ctx.weights[i] = self.likelihood(observation,particle.observation,0) 

    
    def resample(self, ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        '''This is a basic resampling method, more advanced methods like systematic resampling need to override this'''    

        indexes = np.arange(ctx.particle_count) #create a cumulative ndarray from 0 to particle_count

        #The numpy resampling algorithm, see jupyter notebnook resampling.ipynb for more details
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=ctx.prior_weights)



        particleCopy = particleArray.copy()#copy the particle array refs to ensure we don't overwrite particles

        #this loop reindexes the particles by rebuilding the particles
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),
                                        particleCopy[new_particle_indexes[i]].state.copy(),
                                        particleCopy[new_particle_indexes[i]].observation)


        

        return particleArray

