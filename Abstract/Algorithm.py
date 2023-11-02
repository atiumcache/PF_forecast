from abc import ABC,abstractmethod
from typing import List,Dict,Callable
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import Particle,Context



'''Abstract base class for the algorithm. All algorithmic implementations derive from here'''
class Algorithm(ABC): 

    integrator: Integrator #the underlying integrator 
    perturb: Perturb #A class defining a perturbation scheme
    resampler: Resampler #A class defining a resampling scheme
    particles: List[Particle] #a list of particles for the algorithm to operate on
    ctx: Context #meta data about the algorithm


    def __init__(self,integrator:Integrator,perturb:Perturb,resampler:Resampler,ctx:Context)->None:
        self.integrator = integrator
        self.perturb = perturb
        self.resampler = resampler
        self.particles = []
        self.ctx = ctx
        self.output = None



    '''Abstract Methods''' 
    @abstractmethod
    def initialize(self,params:Dict[str,float],priors:Dict[str,Callable])->None: #method to initialize all fields of the estimation, implementation defined
        '''Method to initialize the state and the parameters based on their flags and priors'''
        pass

        
    @abstractmethod
    def run(self) ->None:
        '''No base implementation, defined in child class implementations'''
        pass
        

    '''Callables'''

    def print_particles(self): 
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")




    



    







    

