
from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import *
from typing import Dict,Callable

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
            
            self.particles.append(Particle(param=p_params,state=np.array([0]),observation=np.array([0])))


        


        

        
            

        
    @timing
    def run(self) ->None:
        pass

    #     '''field initializations for Output'''
    #     self.output = Output(observation_data=info.observation_data)
    #     self.output_flags = info.output_flags

    #     '''Create a list to store the dispersion param-hacked in, theres probably a better solution here'''
    #     dispersion = []; 

    #     while self.context.clock.time < len(info.observation_data): 
    #         '''This loop contains all the algorithm steps '''
            
    #         self.particles = self.integrator.propagate(self.particles,self.context)
            
    #         '''hacked in solution for forecasting, more robust solution could be useful'''
    #         if(self.context.clock.time < len(info.observation_data)): 
    #             weights = self.resampler.compute_weights(info.observation_data[self.context.clock.time],self.particles)
    #             self.particles = self.resampler.resample(weights=weights,ctx=self.context,particleArray=self.particles)
    #             self.particles = self.perturb.randomly_perturb(ctx=self.context,particleArray=self.particles)
            
    #         dispersion.append(np.mean([particle.dispersion for particle in self.particles])) 
            
    #         '''output updates, not part of the main algorithm'''
    #         self.output.beta_qtls[:,self.context.clock.time] = quantiles([particle.param['beta'] for _,particle in enumerate(self.particles)])
    #         self.output.observation_qtls[:,self.context.clock.time] = quantiles([particle.observation for _,particle in enumerate(self.particles)])
    #         self.output.average_beta[self.context.clock.time] = np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)])
    #         self.output.average_state[self.context.clock.time,:]=np.mean([particle.state for particle in self.particles],axis=0)

    #         self.context.clock.tick()
    #         print(f"iteration: {self.context.clock.time}")


    #     plt.show()

    #     plt.plot(np.arange(self.output.time_series),dispersion)
    #     plt.show()

    #     self.clean_up()
    #     return self.output
    


    