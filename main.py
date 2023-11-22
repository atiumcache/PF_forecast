from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample,NBinomResampleR
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.solvers.DeterministicSolvers import EulerSolver,Rk45Solver,EulerSolver_SEAIRH
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

algo = TimeDependentAlgo(integrator = EulerSolver_SEAIRH(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1,"sigma1":0.1,"sigma2":0.1}),
                        resampler = NBinomResample(),
                        ctx=Context(population=7_000_000,
                                    state_size = 7,
                                    weights=np.zeros(1000),
                                    rng=np.random.default_rng(),
                        particle_count=1000))

algo.initialize(params={
"beta":0.1,
"gamma":0.1,
"std":ESTIMATION.VARIABLE,
"R":0,
"hosp":15,
"L":90,
"D":7}
,priors={"beta":partial(algo.ctx.rng.uniform,0.07,0.15),
          "D":partial(algo.ctx.rng.uniform,1,20),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "hosp":partial(algo.ctx.rng.normal,17.21147833,5),
          "L":partial(algo.ctx.rng.uniform,1,75),
          "R":partial(algo.ctx.rng.uniform,0.,1.)
          })


algo.run('./datasets/JHU_COVID_CASES.csv',500)











    
