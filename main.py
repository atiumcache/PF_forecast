from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample,NBinomResampleR
from Implementations.solvers.StochasticSolvers import PoissonSolver
from Implementations.solvers.DeterministicSolvers import EulerSolver,LSODASolver,LSODASolverSEIARHD,LSODACalvettiSolver
from Implementations.perturbers.perturbers import MultivariatePerturbations,DynamicPerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

state = "AZ"

algo = TimeDependentAlgo(integrator = LSODACalvettiSolver(),
                        perturb = MultivariatePerturbations(hyper_params={"h":1.,"sigma1":0.01,"sigma2":0.1,"k":0.01}),
                        resampler = LogNBinomResample(),
                        ctx=Context(population=100_000,
                                    state_size = 4,
                                    weights=np.zeros(10000),
                                    seed_loc=1,
                                    seed_size=0.0001,
                                    forward_estimation=1,
                                    rng=np.random.default_rng(),
                        particle_count=10000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":ESTIMATION.STATIC,
"mu":0.004,
"q":0.1,
"eta":ESTIMATION.STATIC,
"std":10,
"R":50,
"hosp":15,
"L":90,
"D":10}
,priors={"beta":partial(algo.ctx.rng.uniform,0.2,1.), 
          "D":partial(algo.ctx.rng.uniform,5,15),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "hosp":partial(algo.ctx.rng.normal,17.21147833,5),
          "gamma":partial(algo.ctx.rng.uniform,0.1,0.7),
          "eta":partial(algo.ctx.rng.uniform,0.3,0.5),
          "R":partial(algo.ctx.rng.uniform,30,50), 
          })

algo.run(f'./datasets/calvetti_sim_data_protocol_A.csv',119)








    
