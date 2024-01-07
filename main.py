from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample,NBinomResampleR
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.solvers.DeterministicSolvers import EulerSolver,LSODASolver,LSODASolverSEIARHD
from Implementations.perturbers.perturbers import MultivariatePerturbations,DynamicPerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

state = "Arizona"

algo = TimeDependentAlgo(integrator = LSODASolverSEIARHD(),
                        perturb = DynamicPerturbations(hyper_params={"h":1.,"sigma1":0.0001,"sigma2":0.1,"k":0.01}),
                        resampler = NBinomResample(),
                        ctx=Context(population=7_000_000,
                                    state_size = 7,
                                    weights=np.zeros(10000),
                                    seed_loc=3,
                                    seed_size=0.0001,
                                    forward_estimation=1,
                                    rng=np.random.default_rng(),
                        particle_count=10000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":0.1,
"eta":0.1,
"std":10,
"R":1,
"hosp":15,
"L":90,
"D":10}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,0.2), 
          "D":partial(algo.ctx.rng.uniform,5,15),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "hosp":partial(algo.ctx.rng.normal,17.21147833,5),
          "gamma":partial(algo.ctx.rng.uniform,0.1,0.7),
          "eta":partial(algo.ctx.rng.uniform,0.3,0.5),
          "R":partial(algo.ctx.rng.uniform,0.1,0.9), 
          })

#algo.print_particles()
algo.run(f'./datasets/JHU_DATA_Arizona.csv',100)








    
