import pandas as pd
import os

def main():
    '''Load new hospitalization data.'''
    full_data_file = './COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv'
    destination_folder = './datasets/hosp_data/'
    os.makedirs(destination_folder, exist_ok=True)

    '''Import locations'''
    locations = pd.read_csv('./datasets/locations.csv',skiprows=0) 
    locations = locations.drop([0]) #skip first row (national ID)
    location_to_state = {}

    '''Map locations codes to state abbreviations.'''
    for index, row in locations.iterrows():
        location_number = row['location']
        abbreviation = row['abbreviation']
        location_to_state.update({location_number: abbreviation})

    '''Extract hospitalization data'''
    global full_hosp_data
    full_hosp_data = pd.read_csv(full_data_file) 
    full_hosp_data = full_hosp_data[['date','state','previous_day_admission_influenza_confirmed']]
    full_hosp_data['date'] = pd.to_datetime(full_hosp_data['date'])

    '''Output each state's data to a separate csv file'''
    for state_code, abbreviation in location_to_state.items():
        state_data = full_hosp_data[full_hosp_data['state'] == abbreviation].sort_values(by=['date'])
        output_filepath = destination_folder + 'hosp_' + state_code + '.csv'
        state_data.to_csv(output_filepath, index=True)


if __name__ == '__main__':
    main()