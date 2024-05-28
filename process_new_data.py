import pandas as pd


def main():
    location_to_state = {}

    full_data_file = './COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv'
    destination_folder = './datasets/hosp_data/'

    '''Import locations'''
    locations = pd.read_csv('./locations.csv',skiprows=0) 
    locations = locations.drop([0]) #skip first row (national ID)
    print("Number of Locations:", len(locations))

    '''Map locations codes to state abbreviations.'''
    for index, row in locations.iterrows():
        location_number = row['location']
        abbreviation = row['abbreviation']
        location_to_state.update({location_number: abbreviation})

    '''Extract hospitalization data'''
    global full_hosp_data
    full_hosp_data = pd.read_csv(full_data_file) 
    full_hosp_data = full_hosp_data[['date','state','previous_day_admission_influenza_confirmed']].sort_values(['state','date'])
    full_hosp_data['date'] = pd.to_datetime(full_hosp_data['date'])


if __name__ == '__main__':
    main()