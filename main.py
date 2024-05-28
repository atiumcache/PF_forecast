from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import PoissonResample,NBinomResample
from Implementations.solvers.DeterministicSolvers import LSODACalvettiSolver,LSODASolver,LSODASolverSEIARHD,EulerSolver
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import pandas as pd
import numpy as np
from argparse import ArgumentParser


def process_args():
    """
    Processes command line arguments.

    Returns:
        Namespace: Contains the parsed command line arguments.
    """    
    parser = ArgumentParser(
                    prog='SMC_EPI',
                    description='Runs a particle filter over the given data.')
    parser.add_argument('filepath', help="absolute path for hospitalization data")
    parser.add_argument('state_code', help="state location code from 'locations.csv'")
    parser.add_argument('runtime', help='time steps', type=int)
    return parser.parse_args()


def get_population(state_code):
    """
    Returns a state's population.

    Args:
        state_code (string): location code corresponding to state

    Returns:
        int: population of given state
    """    
    # state_code = str(state_code).zfill(2) # Standardize state code

    df = pd.read_csv('./datasets/state_populations.csv')
    
    # Query the DataFrame to get the population
    try: 
        population = df.loc[df['state_code'] == int(state_code), 'population'].values
        return population[0]
    except:
        return None
    



def main():
    args = process_args()
    state_population = get_population(args.state_code)

    algo = TimeDependentAlgo(integrator = LSODASolver(),
                        perturb = MultivariatePerturbations(hyper_params={"h":0.5,"sigma1":0.1,"sigma2":0.05}),
                        resampler = NBinomResample(),
                        ctx=Context(population=state_population,
                                    state_size = 4,
                                    weights=np.ones(1000),
                                    seed_loc=[1],
                                    seed_size=0.005,
                                    forward_estimation=1,
                                    rng=np.random.default_rng(),
                                    particle_count=10))

    algo.initialize(params={
    "beta":ESTIMATION.VARIABLE,
    "gamma":0.06,
    "mu":0.004,
    "q":0.1,
    "eta":0.1,
    "std":10,
    "R":50, 
    "hosp":10,
    "L":90,
    "D":10,
    }
    ,priors={"beta":partial(algo.ctx.rng.uniform,0.1,0.15), 
            "D":partial(algo.ctx.rng.uniform,0,15),
            })

    algo.run(args.filepath, args.runtime)


if __name__ == "__main__":
    main()






    
