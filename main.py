from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResampleR,LogNBinomResample
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.solvers.DeterministicSolvers import EulerSolver
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

algo = TimeDependentAlgo(integrator = EulerSolver(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1,"sigma1":0.01,"sigma2":0.1}),
                        resampler = NBinomResampleR(),
                        ctx=Context(population=7_000_000,
                                    state_size = 4,
                                    weights=np.zeros(1000),
                                    rng=np.random.default_rng(),
                        particle_count=1000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":0.1,
"std":ESTIMATION.VARIABLE,
"R":ESTIMATION.STATIC,
"hosp":5.3,
"L":90,
"D":ESTIMATION.STATIC}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,1.),
          "D":partial(algo.ctx.rng.normal,10,3),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "gamma":partial(algo.ctx.rng.normal,0.18687750960396116,0.1),
          "L":partial(algo.ctx.rng.uniform,1,75),
          "R":partial(algo.ctx.rng.uniform,1,100)
          })
data = pd.read_csv('./datasets/FLU_HOSPITALIZATIONS.csv').to_numpy()
data = np.delete(data,0,1)


print("\n")
#algo.print_particles()
beta = []
D = []
L= []
R = []
hosp = []
state = []
LL = []
ESS = []
for t in range(100): 
     print(f"iteration: {t}")
     algo.particles = algo.integrator.propagate(algo.particles,algo.ctx)   
     algo.ctx.weights = (algo.resampler.compute_weights(data[t],algo.particles))

     particle_max = algo.particles[np.argmax(algo.ctx.weights)]


     LL.append(np.log((max(algo.ctx.weights))))

     #print(algo.ctx.weights)
     algo.particles = algo.resampler.resample(algo.ctx,algo.particles)
     algo.particles = algo.perturb.randomly_perturb(algo.ctx,algo.particles) 
    #  print(np.mean([particle.observation for particle in algo.particles]))
    #  print(data[t])
     beta.append(np.mean([particle.param['beta'] for particle in algo.particles]))
     R.append(particle_max.param['R'])

     ESS.append(1/np.sum(algo.ctx.weights **2))

     print(f"real: {data[t]}")

     state.append(np.mean([particle.observation for particle in algo.particles],axis=0))


print(f"log likelihood: {np.sum(LL)}")


rowN = 3
N = 5

fig = plt.clf()
fig = plt.figure()
fig.set_size_inches(10,5)
ax = [plt.subplot(2,rowN,i+1) for i in range(N)]

ax[0].plot(beta,label='Beta')
ax[0].title.set_text('Beta')

ax[1].plot(state,label='New Hospitalizations')
ax[1].plot(data[:100])
ax[1].title.set_text('New Hospitalizations')

ax[2].plot(R,label='NB(r,p)')
ax[2].title.set_text('NB(r,p)')

ax[3].plot(LL,label='Log Likelihood')
total_LL = np.sum(LL)
ax[3].title.set_text(f'Log Likelihood = {total_LL}')

ax[4].plot(ESS,label='Effective Sample Size')
ax[4].title.set_text('Effective Sample Size')

fig.tight_layout()
h = algo.perturb.hyperparameters['h']
fig.savefig(f'figuresBigRh{h}.png',dpi=300)






    
