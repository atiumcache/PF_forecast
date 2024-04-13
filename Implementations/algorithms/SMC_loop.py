
from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import *


from utilities.Utils import Context

class SMC: 

    integrator:Integrator #The integration scheme 
    resampler: Resampler #The resampling scheme
    ctx: SMCContext #The context object

    '''Main particle filtering algorithm as described in Calvetti et. al. '''
    def __init__(self, integrator: Integrator, resampler: Resampler,ctx:SMCContext,params:NDArray) -> None:
        '''Constructor passes back to the parent, nothing fancy here'''

        self.integrator = integrator
        self.resampler = resampler
        self.ctx = ctx


    def run(self,data:np.array):
        pass

  







        