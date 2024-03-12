from utilities.Utils import Particle,Context,SMCContext
from Abstract.Resampler import Resampler
from utilities.Utils import log_norm,jacob
from typing import List
import numpy as np
import math
from numpy.typing import NDArray
from utilities.likelihood_functions import *

'''TODO Convert the Poisson and NBinomResample to systematic resampling, or give an option to the user to choose which scheme they prefer. '''

class NBinomResample(Resampler): 
    
    def __init__(self) -> None:
        """The resampler for the negative binomal distribution, supports a variety of likelihood functions which are listed in resamplers/likelihood_functions."""
        super().__init__(likelihood=likelihood_NB_r)

    def compute_prior_weights(self, ctx:Context,observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        """Computes the prior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            ctx: The Algorithm's Context, in case metadata is needed. 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """


        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros

        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])
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
    
    def compute_pos_weights(self,observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:
        """Computes the posterior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """

        weights = np.zeros(len(particleArray))#initialize weights as an array of zeros

        for i in range(len(particleArray)): 
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])
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
        """Takes in the context and the weights computed from compute weights and performs the resampling. 
        
        Args: 
            ctx: The Algorithm's Context, holds the weights needed for resampling. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the 
            self.particles list in the Algorithm is updated via assignment.  
        """ 

        indexes = np.arange(ctx.particle_count) #create a cumulative ndarray from 0 to particle_count

        #The numpy multinomial resampling algorithm, see jupyter notebnook resampling.ipynb for more details
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=ctx.prior_weights)

        particleCopy = particleArray.copy()#copy the particle array refs to ensure we don't overwrite particles

        '''this loop reindexes the particles by rebuilding the particles, I threw in the copies just to be safe, we need to make sure this is a deep copy, 
        just copying the references will screw things up.'''

        '''TODO 
        Investigate whether copy is necessary for final loop. 
        maybe benchmark. 
        '''
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),
                                        particleCopy[new_particle_indexes[i]].state.copy(),
                                        particleCopy[new_particle_indexes[i]].observation)


        

        return particleArray

class LogNBinomResample(Resampler): 
    def __init__(self) -> None:
        """Resampler using a negative binomial likelihood function with estimated variance and log resampling step from 
        C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018."""
        super().__init__(log_likelihood_NB)

    def compute_prior_weights(self, ctx:Context,observation: NDArray[np.int_], particleArray:List[Particle]) -> NDArray[np.float64]:
        """Computes the prior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            ctx: The Algorithm's Context, in case metadata is needed. 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """
        weights = np.zeros(len(particleArray))

        for i,particle in enumerate(particleArray):
            weights[i] = self.likelihood(np.round(observation),particleArray[i].observation,R=particleArray[i].param['R'])

            if(math.isnan(weights[i])): 
                '''This is for debugging, hopefully these warnings won't pop up in practice.'''
                print(f"real obv: {np.round(observation)}")
                print(f"particle obv: {np.round(particle.observation)}")

        

        #weights = weights-np.max(weights) #normalize the weights wrt their maximum, improves numerical stability
        weights = log_norm(weights) #normalize the log-weights using the jacobian logarithm
        
        return weights
    
    def compute_pos_weights(self, observation: NDArray[np.int_], particleArray:List[Particle]) -> NDArray[np.float64]:
        """Computes the posterior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """
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
        """The actual resampling algorithm, the log variant of systematic resampling. 
        
        Args: 
            ctx: The Algorithm's Context, holds the weights needed for resampling. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the 
            self.particles list in the Algorithm is updated via assignment.  

        
        The algorithm proceeds as follows, 

        1. Generate the log-CDF via the jacobian logarithm. Currently uses the prior_weights, calls out to the jacob function in Utils which returns the whole vector of partial sums. 
            
        2. Resample using the standard systematic algorithm in the log domain. Note the value r is logged compared to the standard implementation, otherwise any systematic resampling literature
        describes the algorithm. 
        """
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
        """Poisson likelihood based resampler."""
        super().__init__(likelihood=likelihood_poisson)

    def compute_prior_weights(self, ctx:Context, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float64]:

        """Computes the prior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            ctx: The Algorithm's Context, in case metadata is needed. 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """
        
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
        """Computes the posterior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """
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
        """Takes in the context and the weights computed from compute weights and performs the resampling. 
        
        Args: 
            ctx: The Algorithm's Context, holds the weights needed for resampling. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the 
            self.particles list in the Algorithm is updated via assignment.  
        """   

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

