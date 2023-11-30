
from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import *
from typing import Dict,Callable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

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
            state = np.concatenate((np.array([self.ctx.population-initial_infected]),[0 for _ in range(self.ctx.state_size-1)])) 
            state[self.ctx.seed_loc] = initial_infected

            self.particles.append(Particle(param=p_params,state=state.copy(),observation=np.array([0])))    


    @timing
    def run(self,data_path:str,runtime:int) ->None:
        '''The algorithms main run method, takes the time series data as a parameter and returns an output object encapsulating parameter and state values'''

        data = pd.read_csv(data_path).to_numpy()
        data = np.delete(data,0,1)


        beta = []
        D = []
        R = []
        state = []
        LL = []
        ESS = []
        gamma = []
        R_quantiles = []
        state_quantiles = []
        beta_quantiles = []

        while(self.ctx.clock.time < runtime): 
            self.integrator.propagate(self.particles,self.ctx)

            # for particle in self.particles: 
            #     particle.param['beta'] = particle.param['beta'] * np.exp(self.ctx.rng.standard_normal() * 0.01)

            self.ctx.weights = (self.resampler.compute_weights(data[self.ctx.clock.time],self.particles))

            self.particles = self.resampler.resample(self.ctx,self.particles)
            self.particles = self.perturb.randomly_perturb(self.ctx,self.particles) 

            particle_max = self.particles[np.argmax(self.ctx.weights)]


            LL.append(np.log((max(self.ctx.weights))))

            beta.append(np.mean([particle.param['beta'] for particle in self.particles]))
            R.append(particle_max.param['hosp'])
            R_quantiles.append(quantiles([particle.param['gamma'] for particle in self.particles]))
            state_quantiles.append(quantiles([particle.observation for particle in self.particles]))
            beta_quantiles.append(quantiles([particle.param['beta'] for particle in self.particles]))
            D.append(particle_max.param['D'])
            ESS.append(1/np.sum(self.ctx.weights **2))
            print(f"{([particle_max.param['gamma']])}")

            state.append(particle_max.observation)

            print(f"Iteration: {self.ctx.clock.time}")
            self.ctx.clock.tick()

        rowN = 3
        N = 6

        R_quantiles = np.array(R_quantiles)
        state_quantiles = np.array(state_quantiles)
        beta_quantiles = np.array(beta_quantiles)

        colors = cm.plasma(np.linspace(0, 1, 12)) # type: ignore

        fig = plt.clf()
        fig = plt.figure()
        fig.set_size_inches(10,5)
        ax = [plt.subplot(2,rowN,i+1) for i in range(N)]

        #ax[0].plot(beta,label='Beta',zorder=12)
        for i in range(11):
            ax[0].fill_between(np.arange(self.ctx.clock.time), beta_quantiles[:,i], beta_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        ax[0].title.set_text('Beta')

        #ax[1].plot(state,label='New Hospitalizations')
        for i in range(11):
            ax[1].fill_between(np.arange(self.ctx.clock.time), state_quantiles[:,i], state_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        ax[1].scatter(np.arange(self.ctx.clock.time),data,s=0.5,zorder=12)
        ax[1].title.set_text('New Hospitalizations')

        ax[2].plot(R,label='hosp')
        ax[2].title.set_text('hosp')

        ax[3].plot(LL,label='Log Likelihood')
        total_LL = np.sum(LL)
        ax[3].title.set_text(f'Log Likelihood = {total_LL}')

        ax[4].plot(ESS,label='Effective Sample Size')
        ax[4].title.set_text('Effective Sample Size')

        
        for i in range(11):
            ax[5].fill_between(np.arange(self.ctx.clock.time), R_quantiles[:,i], R_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)

        fig.tight_layout()
        h = self.perturb.hyperparameters['h']
        fig.savefig(f'figuresh{h}.png',dpi=300)

                





        