from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample,NBinomResampleR
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
                        resampler = NBinomResampleR(),
                        ctx=Context(population=7_000_000,
                                    state_size = 4,
                                    weights=np.zeros(10000),
                                    rng=np.random.default_rng(),
                        particle_count=10000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":0.1,
"std":ESTIMATION.VARIABLE,
"R":ESTIMATION.STATIC,
"hosp":5.3,
"L":90,
"D":10}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,1.),
          "D":partial(algo.ctx.rng.uniform,1,1000),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "gamma":partial(algo.ctx.rng.normal,0.18687750960396116,0.1),
          "L":partial(algo.ctx.rng.uniform,1,75),
          "R":partial(algo.ctx.rng.uniform,10000,1000000)
          })
data = pd.read_csv('./datasets/FLU_HOSPITALIZATIONS.csv').to_numpy()
data = np.delete(data,0,1)

algo.run('./datasets/FLU_HOSPITALIZATIONS.csv')











    
