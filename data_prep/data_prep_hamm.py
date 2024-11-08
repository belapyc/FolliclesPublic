import logging
import math

import pandas as pd
import numpy as np
import pickle
from data_model.cycle_model.Profile import Profile
from data_model.cycle_model.Cycle import Cycle
from tqdm import tqdm

# Excluding patients who have more than one cycle as this is currently not supported
excluded_ids = ['6356-IVF', '18342-IVF', '20154-ICSI', '32406-ICSI', '40948-ICSI', '50840-ICSI', '54430-ICSI',
                '58478-ICSI',
                '65252-ICSI', '6956-ICSI']
# Last ones excluded because starting days were after 16
LOWEST_DAY = 1
HIGHEST_DAY = 20


def profiles_to_objects_lists(row, simple_format = False):
    '''
    Convert array of follicles to objects
    Args:
        row: a row in a dataframe containing list of integers representing follicle
        sizes

    Returns:

    '''
    if row['follicles'] is not None:
        if len(row['follicles']) > 3:
            if simple_format:
                # Process follicles only
                temp_profile = Profile(row['follicles'], row['Cycle Number'], int(row['Day of Scan']),
                                       drug_dosage=None,
                                       afc_count=None)
                row['{}_profile_object'.format('follicles')] = temp_profile
            else:
                if math.isnan(row['afc_r']) or math.isnan(row['afc_l']):
                    afc_count = None
                else:
                    afc_count = int(row['afc_r']) + int(row['afc_l'])

                if math.isnan(row['drug_dosage']):
                    drug_dosage = None
                else:
                    drug_dosage = int(row['drug_dosage'])

                temp_profile = Profile(row['follicles'], row['key'], int(row['day']),
                                       drug_dosage=drug_dosage,
                                       afc_count=afc_count)

                row['{}_profile_object'.format('follicles')] = temp_profile
        else:
            row['{}_profile_object'.format('follicles')] = pd.NA
    else:
        row['{}_profile_object'.format('follicles')] = pd.NA
    return row


def gen_day_of_trigger_df(df):
    explor_data = df.copy(deep=True)

    explor_data['date_of_trigger'] = explor_data['date_of_trigger'].replace(' ', np.NaN)

    explor_data['start_date'] = explor_data['start_date'].replace(' ', np.NaN)

    explor_data.dropna(subset=['date_of_trigger', 'date_of_trigger'], inplace=True)

    explor_data['date_of_trigger'] = pd.to_datetime(explor_data['date_of_trigger'], dayfirst=True)

    explor_data['start_date'] = pd.to_datetime(explor_data['start_date'], dayfirst=True)

    explor_data['day_of_trigger'] = (explor_data['date_of_trigger'] - explor_data['start_date']).dt.days + 1

    explor_data = explor_data[['key', 'day_of_trigger', 'afc_r', 'afc_l', 'age_at_start_of_treatment', 'suppressant_protocol',
                               'weight', 'drug_dosage']]

    return explor_data

def add_drug_dosage(patients, df_dosage):
    no_dosages = 0
    for patient in patients:
        dosages = df_dosage[df_dosage['key'] == patient.key][['drug_dosage', 'day_drug_taken']]
        # print(dosages)
        # Check if any are nan and raise error
        if dosages.isnull().values.any():
            no_dosages += 1

        # Drop nan values
        dosages = dosages.dropna()

        # Convert to list of tuples
        tuple_dosages = [tuple(x) for x in dosages.to_numpy()]

        normalised_dosages = []

        # normalize by weight
        if patient.weight != 0:
            for dosage in tuple_dosages:
                # dosage[0] = dosage[0] / patient.weight
                normalised_dosages.append((int(dosage[0]) / int(patient.weight), dosage[1]))

        patient.dosages = normalised_dosages

        # add dosage to profile
        for profile in patient.profiles.values():
            for dosage in normalised_dosages:
                if profile.day == dosage[1]:
                    profile.drug_dosage_per_weight = dosage[0]
                    break

    print('No dosages: {}'.format(no_dosages))


def add_cycle_features(patients, explor_data):
    num_no_trigger = 0
    num_no_afc = 0
    num_no_age = 0
    num_no_suppressant = 0
    for patient in patients:

        first_row = explor_data[explor_data['key'] == patient.key].iloc[0]

        if 'day_of_trigger' in first_row:
            trigger_day = explor_data[explor_data['key'] == patient.key].iloc[0]['day_of_trigger']
            patient.trigger_day = trigger_day
        else:
            num_no_trigger += 1
            # print('No trigger cycle: {}'.format(patient.key))

        if 'afc_r' in first_row and 'afc_l' in first_row:
            afc_count = explor_data[explor_data['key'] == patient.key].iloc[0]['afc_r'] + \
                        explor_data[explor_data['key'] == patient.key].iloc[0]['afc_l']
            patient.afc_count = afc_count
        else:
            # print('No AFC cycle: {}'.format(patient.key))
            num_no_afc += 1

        if 'age_at_start_of_treatment' in first_row:
            age = int(explor_data[explor_data['key'] == patient.key].iloc[0]['age_at_start_of_treatment'])
            patient.age = age
        else:
            num_no_age += 1
            # print('No age cycle: {}'.format(patient.key))

        if 'suppressant_protocol' in first_row and not pd.isna(first_row['suppressant_protocol']):
            suppressant_protocol = explor_data[explor_data['key'] == patient.key].iloc[0]['suppressant_protocol']
            patient.suppressant_protocol = suppressant_protocol
        else:
            # print('No suppressant cycle: {}'.format(patient.key))
            num_no_suppressant += 1

        if 'weight' in first_row:
            weight = int(explor_data[explor_data['key'] == patient.key].iloc[0]['weight'])
            patient.weight = weight
        else:
            num_no_weight += 1

        os_init_dose = first_row['drug_dosage']
        if not pd.isnull(os_init_dose):
            # print(os_init_dose)
            patient.os_initial_dose = os_init_dose






def str_to_int(row):
    '''
    Convert a string array row into integer
    Args:
        row:

    Returns:

    '''
    # profiles_df['all_follicles_str'] = profiles_df['all_follicles'].str.split(', ')
    list_r = [int(val) for val in row['follicles_rs']]
    list_l = [int(val) for val in row['follicles_ls']]
    row['follicles'] = list_r + list_l

    return row


def profiles_to_patients(ids, profiles, pad_prof=True, pad_median=False):
    patients = []
    logging.info('Creating cycles objects')
    for id in tqdm(ids):
        if id in excluded_ids:
            continue

        list_forid = [x for x in profiles if x.id == id]
        list_forid.sort(key=lambda x: x.day)

        profiles_dict = {}
        for profile in list_forid:
            profiles_dict[str(int(profile.day))] = profile
        if len(list_forid) < 2:
            continue
        temp_patient = Cycle(id, profiles_dict, pad_prof=pad_prof, pad_median=pad_median)
        temp_patient.clinic = 'Hamm'
        patients.append(temp_patient)
    return patients


def prep_data(data_path="../Export_290922_Anon.csv", return_object=False, pad_prof=True, include_groups=['drug_dosage', 'suppressant_protocol']):
    '''

    Args:
        data_path: path to the csv file containing the data
        return_object:  if True, returns a list of PatientU objects
        pad_prof:
        include_groups:

    Returns:

    '''
    df = pd.read_csv(data_path, encoding='cp1252')

    day_of_trigger_df = gen_day_of_trigger_df(df)
    df_dosages = df[['key', 'drug_dosage', 'day_drug_taken']]

    columns_to_use = ['key', 'day', 'list_of_fiolicle_sizes_left_ov', 'list_of_follicle_sizes_right_o', 'afc_r',
                      'afc_l', 'age_at_start_of_treatment', 'weight']
    columns_to_use.extend(include_groups)
    print(columns_to_use)
    df = df[columns_to_use]

    print('Number of cycles: {}'.format(len(df['key'].unique())))
    df = df.dropna(subset=['key', 'day', 'list_of_fiolicle_sizes_left_ov', 'list_of_follicle_sizes_right_o'])
    # print('Number of cycles: {}'.format(len(df['key'].unique())))

    data_new = df[(df['day'] < HIGHEST_DAY) & (df['day'] >= LOWEST_DAY)]

    # Filtering patients with more than 1 scan
    check_ = data_new[['key', 'day']].groupby("key").count().reset_index()
    ids = check_[check_['day'] != 1]['key']
    data_new = data_new[data_new['key'].isin(ids)]

    # Split string array columns
    data_new['follicles_ls'] = data_new['list_of_fiolicle_sizes_left_ov'].str.split(', ')
    data_new['follicles_rs'] = data_new['list_of_follicle_sizes_right_o'].str.split(', ')

    data_new = data_new.apply(str_to_int, axis=1)

    # Remove unnecessary columns
    data_new.drop(['list_of_fiolicle_sizes_left_ov', 'list_of_follicle_sizes_right_o', 'follicles_ls', 'follicles_rs'],
                  axis=1, inplace=True)

    data_new = data_new.apply(profiles_to_objects_lists, axis=1)

    long_df = data_new.dropna(subset=['follicles_profile_object'])

    # Get a list of profile objects from dataframe
    profiles = long_df['follicles_profile_object'].tolist()

    ids = long_df['key'].unique()

    patients = profiles_to_patients(ids, profiles, pad_prof=pad_prof)

    add_cycle_features(patients, day_of_trigger_df)
    add_drug_dosage(patients, df_dosages)

    with open('../all_patients_padded_clean_dict.pickle', 'wb') as handle:
        pickle.dump(patients, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_object:
        return patients
