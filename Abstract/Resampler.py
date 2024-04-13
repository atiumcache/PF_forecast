from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from typing import List,Callable
from utilities.Utils import Context,Particle

class Resampler(ABC): 
    """Abstract class templating all the resampling functions necessary for running the algorithm, the Calvetti algorithm defines two different resampling steps, hence the computation
    of the prior and the posterior weights."""

    likelihood:Callable #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 

    def __init__(self,likelihood) -> None:
        """Enforces the likelihood is a function, derived classes will call this constructor directly, hence not an abstract function.
        
        Args: 
            likelihood: A function that returns a valid probability and takes a series of parameters. 

        Returns: 
            None
        
        Raises: 
            Exception: If likelihood is not a function raises an exception. 
        
        """
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self.likelihood = likelihood

    @abstractmethod
    def compute_weights(self,ctx:Context, observation:NDArray,particleArray:List[Particle])->NDArray[float_]: 
        """Computes the prior weights of the particles given an observation at time t from the time series. 
        
        Args: 
            ctx: The Algorithm's Context, in case metadata is needed. 
            observation: An array of observations for the current time point, count data. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            A numpy array of the normalized weights. 
            
        """
        pass

    @abstractmethod
    def resample(self,ctx:Context,particleArray:List[Particle]) ->List[Particle]:
        """Takes in the context and the weights computed from compute weights and performs the resampling. 
        
        Args: 
            ctx: The Algorithm's Context, holds the weights needed for resampling. 
            particleArray: A list of particles, the Algorithm's self.particles list. 

        Returns: 
            Outputs the updated particle list. Note that as python lists are mutable and therefore passed by reference
            we could forego the return, however I've found that for consistentcy purposes it's better to ensure the 
            self.particles list in the Algorithm is updated via assignment.  
        """
        pass




    