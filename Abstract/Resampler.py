from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from typing import List,Callable
from utilities.Utils import Context,Particle

class Resampler(ABC): 

    likelihood:Callable #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 

    def __init__(self,likelihood) -> None:
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self.likelihood = likelihood

    @abstractmethod
    def compute_weights(self,observation:NDArray,particleArray:List[Particle])->NDArray[float_]: 
        '''Computes the weights of the particles given an observation at time t from the time series'''
        pass

    @abstractmethod
    def resample(self,ctx:Context,particleArray:List[Particle]) ->List[Particle]:
        '''Takes in the context and the weights computed from compute weights and performs the resampling'''
        pass




    