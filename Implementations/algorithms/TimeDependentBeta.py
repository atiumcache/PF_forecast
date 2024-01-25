
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

            self.particles.append(Particle(param=p_params,state=state.copy(),observation=np.array([0 for _ in range(self.ctx.forward_estimation)])))    

    def forward_propagator(): 
        '''This function simulates the 7 days data to be '''


    @timing
    def run(self,data_path:str,runtime:int) ->None:
        '''The algorithms main run method, takes the time series data as a parameter and returns an output object encapsulating parameter and state values'''

        data1 = pd.read_csv(data_path).to_numpy()
        data1 = np.delete(data1,0,1)



        '''Arrays to hold all the output data'''
        eta_quantiles = []

        state = []
        LL = []
        ESS = []
        gamma_quantiles = []

        state_quantiles = []
        beta_quantiles = []
        beta = []
        eta = []
        observations = []
        gamma = []

        while(self.ctx.clock.time < runtime): 

            #one step propagation 
            self.integrator.propagate(self.particles,self.ctx)
        
            obv = data1[self.ctx.clock.time:self.ctx.clock.time+(self.ctx.forward_estimation)]
            self.ctx.weights = (self.resampler.compute_weights(obv,self.particles))
            self.particles = self.resampler.resample(self.ctx,self.particles)
            self.particles = self.perturb.randomly_perturb(self.ctx,self.particles) 



            particle_max = self.particles[np.argmax(self.ctx.weights)]
            print(particle_max.observation)
            observations.append(particle_max.observation)
            print(f"{data1[self.ctx.clock.time]}")

            LL.append(((max(self.ctx.weights))))

            #state_quantiles.append(quantiles([particle.observation[1] for particle in self.particles]))
            beta_quantiles.append(quantiles([particle.param['beta'] for particle in self.particles]))
            beta.append(np.mean([particle.param['beta'] for particle in self.particles]))
            ESS.append(1/np.sum(self.ctx.weights **2))

            state.append(np.mean([particle.state for particle in self.particles],axis=0))
            eta_quantiles.append(quantiles([particle.param['eta'] for particle in self.particles]))
            eta.append(np.mean([particle.param['eta'] for particle in self.particles]))
            gamma_quantiles.append(quantiles([particle.param['gamma'] for particle in self.particles]))
            gamma.append(np.mean([particle.param['gamma'] for particle in self.particles]))

            print(f"Iteration: {self.ctx.clock.time}")
            self.ctx.clock.tick()

        pd.DataFrame(beta).to_csv('../datasets/average_beta.csv')
        pd.DataFrame(eta).to_csv('../datasets/average_eta.csv')
        pd.DataFrame(beta).to_csv('../datasets/average_gamma.csv')


        rowN = 3
        N = 6

        labels = ['Susceptible','Exposed','Infected','Recovered']
        state = np.array(state)
        plt.yscale('log')
        for i in range(self.ctx.state_size): 
                plt.plot(state[:,i],label=labels[i])
        #plt.plot(observations)


        plt.xlabel('Time(Days)')
        plt.legend()
        plt.show()



        pd.DataFrame(state).to_csv('../datasets/ESTIMATED_STATE.csv')            

        state_quantiles = np.array(state_quantiles)
        beta_quantiles = np.array(beta_quantiles)
        eta_quantiles = np.array(eta_quantiles)
        gamma_quantiles = np.array(gamma_quantiles)

        colors = cm.plasma(np.linspace(0, 1, 12)) # type: ignore

        # for i in range(11):
        #     plt.fill_between(np.arange(self.ctx.clock.time), beta_quantiles[:,i], beta_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        # plt.show()

        # plt.title("Mean of Beta")
        # plt.plot(beta)
        # plt.show()

        # plt.title("Mean of Eta")
        # plt.plot(eta)
        # plt.show()

        # plt.title("Mean of Gamma")
        # plt.plot(gamma)
        # plt.show()

        # for i in range(11):
        #     plt.fill_between(np.arange(self.ctx.clock.time), eta_quantiles[:,i], eta_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        
        # plt.title("Distribution of Eta")
        # plt.show()

        # plt.title("Distribution of Gamma")
        # for i in range(11):
        #     plt.fill_between(np.arange(self.ctx.clock.time), gamma_quantiles[:,i], gamma_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        # plt.show()


        # fig = plt.figure()
        # fig.set_size_inches(10,5)
        # ax = [plt.subplot(2,rowN,i+1) for i in range(N)]
        # fig.subplots_adjust(hspace=0)
        
        # for i in range(N): 
        #     ax[i].set_xlabel('Days since 3/16/2020')

        #ax[0].plot(beta,label='Beta',zorder=12)
        # for i in range(11):
        #     ax[0].fill_between(np.arange(self.ctx.clock.time), beta_quantiles[:,i], beta_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        # ax[0].title.set_text('Beta')

        
        # labels = ['Susceptible','Exposed','Asymptomatic','Infected','Hospitalized','Recovered','Dead']
        # ax[1].title.set_text('Latent State')
        # state = np.array(state)
        # ax[1].set_yscale('log')
        # for i in range(1,self.ctx.state_size): 
        #     ax[1].plot(state[:,i],label=labels[i],linewidth=0.5)


        # for i in range(11):
        #     ax[2].fill_between(np.arange(self.ctx.clock.time), eta_quantiles[:,i], eta_quantiles[:,22-i], facecolor=colors[11 - i], zorder=i)
        # ax[2].scatter(np.arange(self.ctx.clock.time),data1[:self.ctx.clock.time],s=0.5,zorder=12)
        # ax[2].title.set_text('Total Infected')


        # ax[3].plot(LL,label='Log Likelihood')
        # total_LL = np.sum(LL)
        # ax[3].title.set_text(f'Log Likelihood = {round(total_LL,2)}')

        # ax[4].plot(ESS,label='Effective Sample Size')
        # ax[4].title.set_text('Effective Sample Size')

        
        #fig.tight_layout()

        #fig.savefig(f'{data_path[11:13]}_figuresR{R_val}_forward{self.ctx.forward_estimation}_var{var}.png',dpi=300)

                





        