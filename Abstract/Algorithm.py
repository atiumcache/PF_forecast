from abc import ABC,abstractmethod
from typing import List,Dict,Callable
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import Particle,Context




class Algorithm(ABC): 
    """Abstract base class for the algorithm. All algorithmic implementations derive from here."""
    integrator: Integrator #the underlying integrator 
    perturb: Perturb #A class defining a perturbation scheme
    resampler: Resampler #A class defining a resampling scheme
    particles: List[Particle] #a list of particles for the algorithm to operate on
    ctx: Context #meta data about the algorithm


    def __init__(self,integrator:Integrator,perturb:Perturb,resampler:Resampler,ctx:Context)->None:
        """Note that the particle list is initialized to empty, in initialize method will set up the list before running the chosen algorithm."""
        self.integrator = integrator
        self.perturb = perturb
        self.resampler = resampler
        self.particles = []
        self.ctx = ctx
        self.output = None



    '''Abstract Methods''' 
    @abstractmethod
    def initialize(self,params:Dict[str,float],priors:Dict[str,Callable])->None:
        """Function to initialize the state and the parameters based on their flags and priors
        
        Args: 
            params: A dictionary of named parameters with either a fixed value or a flag indicating the parameter is to be estimated.(Uses the ESTIMATION enum)
            priors: A dictionary of named functions which are sampled to form a prior on the named parameters. 

        Returns: 
            None
        """
        pass

        
    @abstractmethod
    def run(self,data_path:str,runtime:int) ->None:
        """The function to run the algorithm for a specified number of iterations.
        
        Args: 
            data_path: A valid file path pointing to a csv of time series data to run the algorithm against. Currently the left column must be enumerated and the right column is
            the count data. 

            runtime: An integer value indicating the number of iterations for which to run the algorithm. One iteration corresponds to one day, as such this value must be less than
            the length of the time series in data_path. 
        
        
        """
        pass
        

    '''Callables'''

    def print_particles(self): 
        """Prints the particles in self.particles for debugging."""
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")

            




    



    







    

