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

algo = TimeDependentAlgo(integrator = EulerSolver_SEAIRH(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.1,"sigma1":0.1,"sigma2":0.05}),
                        resampler = NBinomResampleR(),
                        ctx=Context(population=7_000_000,
                                    state_size = 7,
                                    weights=np.zeros(10000),
                                    seed_loc=1,
                                    rng=np.random.default_rng(),
                        particle_count=10000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":0,
"eta":0,
"std":10,
"R":1/10,
"hosp":15,
"L":90,
"D":7}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,0.15), 
          "D":partial(algo.ctx.rng.uniform,1,20),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "hosp":partial(algo.ctx.rng.normal,17.21147833,5),
          "gamma":partial(algo.ctx.rng.uniform,0.1,0.7),
          "eta":partial(algo.ctx.rng.uniform,0.3,0.5),
          "R":partial(algo.ctx.rng.uniform,0.1,0.9)
          })

#algo.print_particles()
algo.run('./datasets/JHU_CONVOLVED_Arizona.csv',879)











    
