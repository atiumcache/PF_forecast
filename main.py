from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample
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
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1,"sigma1":0.1,"sigma2":0.1}),
                        resampler = NBinomResample(),
                        ctx=Context(population=7_000_000,
                                    state_size = 4,
                                    weights=np.zeros(1000),
                                    rng=np.random.default_rng(),
                        particle_count=1000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":0.1,
"std":ESTIMATION.VARIABLE,
"hosp":5.3,
"L":90,
"D":ESTIMATION.STATIC}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,1.),
          "D":partial(algo.ctx.rng.normal,10,3),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "gamma":partial(algo.ctx.rng.normal,0.18687750960396116,0.1),
          "L":partial(algo.ctx.rng.uniform,1,75),
          "R":partial(algo.ctx.rng.normal,100000,100)
          })
data = pd.read_csv('./datasets/COVID_ADMISSIONS_CA.csv').to_numpy()
data = np.delete(data,0,1)


print("\n")
#algo.print_particles()
beta = []
D = []
L= []
hosp = []
state = []
LL = 0
for t in range(200): 
     print(f"iteration: {t}")
     algo.particles = algo.integrator.propagate(algo.particles,algo.ctx)   
     algo.ctx.weights = (algo.resampler.compute_weights(data[t],algo.particles))


     LL += np.log((max(algo.ctx.weights)))

     #print(algo.ctx.weights)
     algo.particles = algo.resampler.resample(algo.ctx,algo.particles)
     algo.particles = algo.perturb.randomly_perturb(algo.ctx,algo.particles) 
    #  print(np.mean([particle.observation for particle in algo.particles]))
    #  print(data[t])
     beta.append(np.mean([particle.param['beta'] for particle in algo.particles]))
     D.append(np.mean([particle.param['std'] for particle in algo.particles]))


     # #algo.print_particles()
     state.append(np.mean([particle.state for particle in algo.particles],axis=0))
     #print(state[-1])
#     print(algo.ctx.weights)

print(f"log likelihood: {LL}")

plt.plot(beta)
plt.show()

plt.yscale('log')
plt.plot(state)
plt.show()

plt.title("Estimate of R over time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.plot(D)
plt.show()


plt.show()


    
