from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample,NBinomResampleR
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.solvers.DeterministicSolvers import EulerSolver,Rk45Solver,EulerSolver_SEAIRH,EulerSolver_SIR
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

algo = TimeDependentAlgo(integrator = EulerSolver_SIR(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1,"sigma1":0.01,"sigma2":0.05}),
                        resampler = NBinomResample(),
                        ctx=Context(population=7_000_000,
                                    state_size = 3,
                                    weights=np.zeros(1000),
                                    seed_loc=1,
                                    seed_size=0.05,
                                    rng=np.random.default_rng(),
                        particle_count=1000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":ESTIMATION.STATIC,
"std":ESTIMATION.VARIABLE,
"R":0,
"hosp":15,
"L":90,
"D":7}
,priors={"beta":partial(algo.ctx.rng.uniform,0.0,1.0),
          "D":partial(algo.ctx.rng.uniform,1,20),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "hosp":partial(algo.ctx.rng.normal,17.21147833,5),
          "L":partial(algo.ctx.rng.uniform,1,75),
          "gamma":partial(algo.ctx.rng.uniform,0.1,0.2)
          })

#algo.print_particles()
algo.run('./datasets/SIR_SIM_DATA.csv',100)











    
