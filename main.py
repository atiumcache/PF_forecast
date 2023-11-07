from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.solvers.DeterministicSolvers import EulerSolver
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

algo = TimeDependentAlgo(integrator = PoissonSolver(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1,"sigma1":0.01,"sigma2":0.1}),
                        resampler = NBinomResample(),
                        ctx=Context(population=7_000_000,
                                    state_size = 4,
                                    weights=np.zeros(5),
                                    rng=np.random.default_rng(5),
                        particle_count=5))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":ESTIMATION.STATIC,
"std":ESTIMATION.VARIABLE, 
"hosp":5.3,
"L":90.0,
"D":10.0}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,1.),
          "gamma":partial(algo.ctx.rng.uniform,0.,1.),
          "std":partial(algo.ctx.rng.uniform,20.,30.)
          })
data = pd.read_csv('./datasets/FLU_HOSPITALIZATIONS.csv').to_numpy()
data = np.delete(data,0,1)


print("\n")

beta = []
for t in range(len(data)): 
    #print(f"iteration: {t}")
    algo.integrator.propagate(algo.particles,algo.ctx)   
    weights = (algo.resampler.compute_weights(data[t],algo.particles))
    algo.ctx.weights = weights
    algo.resampler.resample(algo.ctx,algo.particles)
    algo.perturb.randomly_perturb(algo.ctx,algo.particles) 
    #print(np.mean([particle.param['beta'] for particle in algo.particles]))
    
