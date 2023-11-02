from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from utilities.Utils import Context,ESTIMATION
from functools import partial
import numpy as np

algo = TimeDependentAlgo(integrator = None,perturb = None,resampler = None,ctx=Context(population=39_000_000,particle_count=5))
algo.initialize(params={
"beta":ESTIMATION.VARIABLE,
"gamma":ESTIMATION.STATIC,
"variance":ESTIMATION.VARIABLE, 
"hosp":5.3,
"L":90.0,
"D":10.0}
,priors={"beta":partial(algo.ctx.rng.uniform,0.,1.),
          "gamma":partial(algo.ctx.rng.uniform,0.,1.),
          "variance":partial(algo.ctx.rng.uniform,20.,30.)
          })

            
algo.print_particles()
