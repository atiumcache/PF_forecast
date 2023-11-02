from abc import ABC,abstractmethod
from typing import Tuple,List
from numpy import int_
from utilities.Utils import Particle,Context

class Integrator(ABC): 

    
    @abstractmethod
    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]: 
        '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
        pass
    '''Note: observations are forced to be integers, if using a deterministic integrator round off values before returning them from propagate
    python may do this for you via downcasting, but it's very up in the air, best to ensure data is correct yourself'''

    