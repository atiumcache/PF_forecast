
from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import *
from typing import Dict,Callable
import pandas as pd
from utilities.sankey import visualize_particles
import matplotlib.pyplot as plt

from utilities.Utils import Context

@dataclass
class Stats:
    log_likelihood : float
    state: np.array
    beta: np.array


class Bootstrap(Algorithm): 
    '''Main particle filtering algorithm as described in Calvetti et. al. '''
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler,ctx:Context) -> None:
        '''Constructor passes back to the parent, nothing fancy here'''
        super().__init__(integrator, perturb, resampler,ctx)
        
    def initialize(self,params:Dict[str,int],priors:Dict[str,Callable]) -> None:

        self.particles.clear()
        self.ctx.clock.reset()

        '''Initialize the parameters and their flags for estimation'''
        for _,(key,val) in enumerate(params.items()):
            if(val == ESTIMATION.VARIABLE): 
                self.ctx.estimated_params[key] = ESTIMATION.VARIABLE
            elif(val == ESTIMATION.PMCMC):
                self.ctx.estimated_params[key] = ESTIMATION.PMCMC


        for _ in range(self.ctx.particle_count): 
            '''Setup the particles at t = 0, important we make a copy of the params dictionary before using it
            to setup each particle.'''

            p_params = params.copy()
            '''Call the priors to generate values for the estimated params and set their values in the new params.'''
            for _,(key,val) in enumerate(self.ctx.estimated_params.items()):
                if(p_params[key] == ESTIMATION.VARIABLE): 
                    p_params[key] = priors[key]()
                elif(p_params[key] == ESTIMATION.PMCMC): 
                    p_params[key] = priors[key]


            seeds = []
            '''Generate seeds from U(0,seed_size*pop) in the length of the seed loc array'''
            for _ in range(len(self.ctx.seed_loc)):
                seeds.append(self.ctx.rng.uniform(0,self.ctx.seed_size))

            state = np.concatenate((np.array([self.ctx.population],dtype=np.float_),[0 for _ in range(self.ctx.state_size-1)])) 
            for i in range(len(seeds)):
                state[self.ctx.seed_loc[i]] += seeds[i]
                state[0] -= seeds[i]

            self.particles.append(Particle(param=p_params,state=state.copy(),observation=np.array([0 for _ in range(self.ctx.estimation_size)])))   


    def run(self,data_path:str,runtime:int) ->Stats:
        '''The algorithms main run method, takes the time series data as a parameter and returns an output object encapsulating parameter and state values'''

        data1 = pd.read_csv(data_path).to_numpy()
        data1 = np.delete(data1,0,1)

        LL = 0
        mean_state = []
        beta = []

        "Initialize labels and first column of sankey matrix"
        if self.ctx.run_sankey == True:
            self.ctx.sankey_indices.append(np.arange(self.ctx.particle_count)) 

        while(self.ctx.clock.time < runtime): 
 

            self.particles = self.integrator.propagate(self.particles,self.ctx)
        
            obv = data1[self.ctx.clock.time]

            self.ctx.weights = self.resampler.compute_weights(self.ctx,obv,self.particles)

            self.particles = self.resampler.resample(self.ctx,self.particles)

            self.particles = self.perturb.randomly_perturb(self.ctx,self.particles) 

            LL += np.mean(self.ctx.weights)
            mean_state.append(np.average([particle.state for particle in self.particles],weights=self.ctx.weights,axis=0))
            beta.append(np.average([particle.param['beta'] for particle in self.particles],weights=self.ctx.weights,axis=0))

            #print(f"Iteration: {self.ctx.clock.time}")
            self.ctx.clock.tick()


        beta = np.array(beta)

        mean_state = np.array(mean_state)

        # plt.yscale('log')
        # for i in range(mean_state.shape[1]): 
        #     plt.plot(mean_state[:,i])
        # plt.show()

        # plt.plot(beta)
        # plt.show()

        # sankey test code
        if self.ctx.run_sankey == True:
            visualize_particles(self.ctx.particle_count, self.ctx.sankey_indices)

        return Stats(log_likelihood=LL,state = mean_state,beta=beta)
       



        
