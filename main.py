from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import NBinomResample,LogNBinomResample,NBinomResampleR,PoissonResample
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
                        resampler = PoissonResample(),
                        ctx=Context(population=100_000,
                                    state_size = 4,
                                    prior_weights=np.zeros(1000),
                                    pos_weights=np.zeros(1000),
                                    weight_ratio=np.ones(1000),
                                    seed_loc=[1,2],
                                    seed_size=0.0001,
                                    forward_estimation=1,
                                    rng=np.random.default_rng(),
                        particle_count=1000))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":1/14,
"mu":0.004,
"q":0.1,
"eta":ESTIMATION.STATIC,
"std":10,
"R":50,
"hosp":15,
"L":90,
"D":10}
,priors={"beta":partial(algo.ctx.rng.uniform,0.1,0.6), 
          "gamma":partial(algo.ctx.rng.uniform,1/28,1/7),
          })

algo.print_particles()
#algo.run(f'./datasets/calvetti_sim_data_protocol_A.csv',119)








    
