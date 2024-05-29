from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import PoissonResample,NBinomResample
from Implementations.solvers.DeterministicSolvers import LSODACalvettiSolver,LSODASolver,LSODASolverSEIARHD,EulerSolver
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from datetime import date


def process_args():
    """
    Processes command line arguments.

    :return Namespace: Contains the parsed command line arguments.
    """    
    parser = ArgumentParser(
                    prog='SMC_EPI',
                    description='Runs a particle filter over the given data.')
    parser.add_argument('filepath', help="file path for hospitalization data", type=str)
    parser.add_argument('state_code', help="state location code from 'locations.csv'")
    parser.add_argument('start_date', help='day to forecast from. ISO 8601 format.', type=str)
    return parser.parse_args()


def get_population(state_code: str) -> int:
    """
    Returns a state's population.

    :param state_code: Location code corresponding to state. 
    :return: population.
    """       
    df = pd.read_csv('./datasets/state_populations.csv')
    
    # Query the DataFrame to get the population
    try: 
        population = df.loc[df['state_code'] == int(state_code), 'population'].values
        return population[0]
    except:
        return None
    

def get_previous_80_rows(df: pd.DataFrame, target_date: pd.DateTime) -> pd.DataFrame:
    """
    Returns a data frame containing 80 rows of a state's hospitalization data.
    Data runs from input date to 80 days prior. 

    :param df: A single state's hospitalization data. 
    :param date: Date object in ISO 8601 format. 
    :return: The filtered df with 80 rows. 
    """    
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values(by='date')
    date_index = df_sorted[df_sorted['date'] == target_date].index[0]
    start_index = max(date_index - 80, 0)
    result_df = df_sorted.iloc[start_index:date_index+1]
    result_df = result_df.drop(columns=['state', 'date'], axis=1)

    return result_df
    

def initialize_algo(state_population):
    """Returns an algorithm object, given a state's population."""
    algorithm = TimeDependentAlgo(integrator = LSODASolver(),
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
    
    algorithm.initialize(params={
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
    ,priors={"beta":partial(algorithm.ctx.rng.uniform,0.1,0.15), 
            "D":partial(algorithm.ctx.rng.uniform,0,15),
            })
    
    return algorithm


def main():
    args = process_args()
    state_pop = get_population(args.state_code)
    hosp_csv_path = './datasets/hosp_data/hosp_' + args.state_code + '.csv'
    state_data = pd.read_csv(hosp_csv_path)

    start_date = pd.to_datetime(args.start_date)
    state_data = get_previous_80_rows(state_data, start_date)

    # Run the particle filter for 80 days prior to start date
    algo = initialize_algo(state_pop)
    algo.run(state_data, 80)
    
    
if __name__ == "__main__":
    main()






    
