import pandas as pd
from datetime import date
from filter_forecast.state import State
from filter_forecast.helpers import process_args, get_population, get_previous_80_rows
from filter_forecast.algo_init import initialize_algo

def main():
    args = process_args()
    
    state = State(args.state_code)

    start_date = pd.to_datetime(args.start_date)
    
    filtered_state_data = get_previous_80_rows(state.hosp_data, start_date)

    # Run the particle filter for 80 days prior to start date
    algo = initialize_algo(state.population)
    algo.run(filtered_state_data, 80)
    
if __name__ == "__main__":
    main()
