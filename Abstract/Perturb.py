from abc import ABC,abstractmethod
from typing import Dict,List
from utilities.Utils import Particle,Context

class Perturb(ABC): 
    hyperparameters: Dict #A dictionary of perturbation parameters

    def __init__ (self,hyper_params:Dict)-> None: 
        '''The perturber has special hyperparameters which tell randomly_perturb how much to move the parameters and state'''
        self.hyperparameters = hyper_params 


    
    @abstractmethod
    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        '''Implementations of this method will take a list of particles and perturb it according to a user defined distribution'''
        pass


