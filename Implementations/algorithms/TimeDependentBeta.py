
from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import *
from typing import Dict,Callable
import pandas as pd
import matplotlib.pyplot as plt

from utilities.Utils import Context

class TimeDependentAlgo(Algorithm): 
    '''Main particle filtering algorithm as described in Calvetti et. al. '''
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler,ctx:Context) -> None:
        '''Constructor passes back to the parent, nothing fancy here'''
        super().__init__(integrator, perturb, resampler,ctx)
        
    def initialize(self,params:Dict[str,int],priors:Dict[str,Callable]) -> None:
        '''Initialize the parameters and their flags for estimation'''
        for _,(key,val) in enumerate(params.items()):
            if(val == ESTIMATION.STATIC): 
                self.ctx.estimated_params[key] = ESTIMATION.STATIC
            elif(val == ESTIMATION.VARIABLE): 
                self.ctx.estimated_params[key] = ESTIMATION.VARIABLE

        for _ in range(self.ctx.particle_count): 
            '''Setup the particles at t = 0'''
            p_params = params.copy()
            for _,(key,val) in enumerate(self.ctx.estimated_params.items()):
                if((p_params[key] == ESTIMATION.STATIC) or (p_params[key] == ESTIMATION.VARIABLE)): 
                    p_params[key] = priors[key]()

            initial_infected = self.ctx.rng.uniform(0,self.ctx.seed_size*self.ctx.population)
            state = np.concatenate((np.array([self.ctx.population-initial_infected,initial_infected]),[0 for _ in range(self.ctx.state_size-2)])) 
            
            self.particles.append(Particle(param=p_params,state=state.copy(),observation=np.array([0])))    


    @timing
    def run(self,data_path:str) ->None:
        '''The algorithms main run method, takes the time series data as a parameter and returns an output object encapsulating parameter and state values'''

        data = pd.read_csv(data_path).to_numpy()
        data = np.delete(data,0,1)

        beta = []
        D = []
        R = []
        state = []
        LL = []
        ESS = []

        while(self.ctx.clock.time < len(data)): 
            self.integrator.propagate(self.particles,self.ctx)

            self.ctx.weights = (self.resampler.compute_weights(data[self.ctx.clock.time],self.particles))

            self.particles = self.resampler.resample(self.ctx,self.particles)
            self.particles = self.perturb.randomly_perturb(self.ctx,self.particles) 

            particle_max = self.particles[np.argmax(self.ctx.weights)]


            LL.append(np.log((max(self.ctx.weights))))

            beta.append(np.mean([particle.param['beta'] for particle in self.particles]))
            R.append(particle_max.param['R'])
            D.append(particle_max.param['D'])

            ESS.append(1/np.sum(self.ctx.weights **2))


            state.append(np.mean([particle.observation for particle in self.particles],axis=0))

            print(f"Iteration: {self.ctx.clock.time}")
            self.ctx.clock.tick()

        rowN = 3
        N = 5

        fig = plt.clf()
        fig = plt.figure()
        fig.set_size_inches(10,5)
        ax = [plt.subplot(2,rowN,i+1) for i in range(N)]

        ax[0].plot(beta,label='Beta')
        ax[0].title.set_text('Beta')

        ax[1].plot(state,label='New Hospitalizations')
        ax[1].plot(data[:140])
        ax[1].title.set_text('New Hospitalizations')

        ax[2].plot(R,label='NB(r,p)')
        ax[2].title.set_text('NB(r,p)')

        ax[3].plot(LL,label='Log Likelihood')
        total_LL = np.sum(LL)
        ax[3].title.set_text(f'Log Likelihood = {total_LL}')

        ax[4].plot(ESS,label='Effective Sample Size')
        ax[4].title.set_text('Effective Sample Size')

        fig.tight_layout()
        h = self.perturb.hyperparameters['h']
        fig.savefig(f'figuresBigRh{h}.png',dpi=300)

                





        