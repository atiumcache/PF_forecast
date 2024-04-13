from dataclasses import dataclass,field
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from numpy import random
from typing import Dict,List
from functools import wraps
from time import perf_counter

class Clock: 
    '''Internal clock for keeping track of the time the algorithm is at in the observation data'''

    time: int
    def __init__(self) -> None:
        self.time = 0
    
    def tick(self):
        self.time +=1

    def reset(self): 
        self.time = 0

class ESTIMATION(Enum): 
    '''Enum which flags a parameter for estimation'''
    STATIC = -1
    VARIABLE = -2


@dataclass
class Particle: 
    '''The basic particle class'''
    param: Dict #a dictionary of parameters pertaining to the model of interest
    state: NDArray #underlying simulated state information
    observation: NDArray #the current observation of the particle



@dataclass
class Context: 
    '''Meta data about the algorithm'''
    weights: NDArray[np.float64]
    particle_count: int = field(default = 1000) 
    clock: Clock = field(default_factory=lambda: Clock())
    rng:random.Generator = field(default_factory=lambda: np.random.default_rng())
    seed_size: float = field(default = 0.01) #estimate of initial percentage of infected out of the total population
    state_size: int = field(default = 4) #number of state variables in the model
    seed_loc: List[int] = field(default_factory=lambda: list()) #zero indexed seed location 
    sankey_indices: List[int] = field(default_factory=lambda: list()) # a 2D matrix that stores particle indices at each step of the algo
    population: int = field(default=100_000) #estimate of the total population 
    estimated_params: Dict[str,int] = field(default_factory=lambda: dict()) #number of estimated parameters in the model 
    forward_estimation: int = 7 #The number of subsequent states to be considered in the likelihood function
    run_sankey: bool = False # Set False to turn off Sankey code


def timing(f):
    '''Decorator for timing function calls '''
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


def quantiles(items:List)->List: 
        '''Returns 23 quantiles of the List passed in'''
        qtlMark = 1.00*np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])
        return list(np.quantile(items, qtlMark))


def jacob(δ:NDArray[np.float_])->NDArray[np.float_]:
    """The jacobian logarithm, used in log likelihood normalization and resampling processes
    δ will be an array of values. 
    
    Args: 
        δ: An array of values to sum

    Returns: 
        The vector of partial sums of δ.          
    
    """
    n = len(δ)
    Δ = np.zeros(n)
    Δ[0] = δ[0]
    for i in range(1,n):
        Δ[i] = max(δ[i],Δ[i-1]) + np.log(1 + np.exp(-1*np.abs(δ[i] - Δ[i-1])))
    return(Δ)


def log_norm(log_weights:NDArray): 
    '''normalizes the probability space using the jacobian logarithm as defined in jacob() '''
    norm = (jacob(log_weights)[-1])
    log_weights -= norm
    return log_weights













