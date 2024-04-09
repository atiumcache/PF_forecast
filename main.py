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
                                    particle_count=1000, 
                                    run_sankey=True))

algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":1/14,
"mu":0.004,
"q":0.1,
"eta":1/7,
"std":10,
"R":50,
"hosp":15,
"L":90,
"D":10}
,priors={"beta":partial(algo.ctx.rng.uniform,0.1,0.6), 
          "D":partial(algo.ctx.rng.uniform,5,15),
          "std":partial(algo.ctx.rng.uniform,20.,30.),
          "hosp":partial(algo.ctx.rng.normal,17.21147833,5),
          "gamma":partial(algo.ctx.rng.uniform,1/28,1/7),
          "eta":partial(algo.ctx.rng.uniform,1/15,1/3),
          "R":partial(algo.ctx.rng.uniform,30,50), 
          })

algo.run(f'./datasets/calvetti_sim_data_protocol_A.csv',119)








    
