import pickle

import pandas as pd

from data_model.cycle_model.Profile import Profile
from data_prep.data_prep_hamm import profiles_to_patients
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def profiles_to_objects_lists_tfp(row, drop_not_trigger_scan=False):
    if len(row['Tracking Scans']) > 1:
        profiles = []
        for scan in row['Tracking Scans']:

            follicles = scan[3]
            day = scan[0]
            # drop follicles below 5
            follicles = [follicle for follicle in follicles if follicle >= 5]
            if len(follicles) > 3:
                profiles.append(Profile(follicles, row['Cycle Number'], day))

        trigger_day_follicles = row['DoT Follicles']
        # drop follicles below 5

        if isinstance(trigger_day_follicles, list) and len(trigger_day_follicles) > 3:
            raise Exception("DoT Follicles is not a list")
            # print('list')
            trigger_day_follicles = [follicle for follicle in trigger_day_follicles if follicle >= 5]
            if len(trigger_day_follicles) > 3:
                trigger_day_profile = Profile(trigger_day_follicles, row['Cycle Number'], row['Trigger Day'])
                profiles.append(trigger_day_profile)
        else:
            if drop_not_trigger_scan:
                row['follicles_profile_object'] = None
                return row
        row['follicles_profile_object'] = profiles
        return row
    else:
        row['follicles_profile_object'] = None
        return row


def add_trigger_day(patients, df):
    for patient in patients:
        trigger_day = df.loc[df['Cycle Number'] == patient.key]['Trigger Day'].values[0]
        patient.trigger_day = trigger_day
    # filter patients whose trigger day is below 1 or above 25
    patients = [patient for patient in patients if 1 <= patient.trigger_day <= 20]
    return patients


def clean_profiles(profiles):

    # Check that all profiles have at least 3 follicles
    profiles = [profile for profile in profiles if len(profile.follicles) > 3]
    # Check that profile data is between 1 and 20
    profiles = [profile for profile in profiles if 1 <= profile.day <= 20]
    # replace all follicles below 5 with 5
    for profile in profiles: # TODO: check if this is needed -- can remove
        profile.follicles = [5 if follicle < 5 else follicle for follicle in profile.follicles]
    return profiles


def prep_data_tfp(data_path="../../AllClinics_27092023_DoT_Artsiom.pkl", drop_not_trigger_scan=False, df=None):
    if df is None:
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
    df['Trigger Day'] = (df['Trigger Date'] - df['Treatment Start Date']).dt.days

    # drop rows with no tracking scans
    df = df.dropna(subset=['Tracking Scans'])
    # df = df.dropna(subset=['DoT Follicles'])
    df = df.dropna(subset=['Trigger Day'])

    df = df.apply(lambda x: profiles_to_objects_lists_tfp(x, drop_not_trigger_scan), axis=1)

    long_df = df.dropna(subset=['follicles_profile_object'])

    # Get a list of profile objects from dataframe
    profiles = long_df['follicles_profile_object'].tolist()
    # flatten profiles
    profiles = [item for sublist in profiles for item in sublist]

    ids = long_df['Cycle Number'].unique()

    profiles = clean_profiles(profiles)

    patients = profiles_to_patients(ids, profiles, pad_prof=False, pad_median=False)

    patients = add_trigger_day(patients, df)

    return patients


def get_train_test_split_tfp():
    cycles = prep_data_tfp('AllClinics_27092023_DoT_Artsiom.pkl', return_object=True)
    # %%
    # Load pickled data
    with open('AllClinics_27092023_DoT_Artsiom.pkl', 'rb') as f:
        data = pickle.load(f)

    cycles = clean_cycles(cycles)

    cycles = add_features_to_cycles(cycles, data, tqdm_disable=False)

    make_bins_profiles(cycles)
    # %%
    high_responders = [patient for patient in cycles if patient.responder_class == "HIGH"]
    normal_responders = [patient for patient in cycles if patient.responder_class == "NORMAL"]
    low_responders = [patient for patient in cycles if patient.responder_class == "LOW"]

    # print number of patients in each group
    print("Number of high responders: {}".format(len(high_responders)))
    print("Number of normal responders: {}".format(len(normal_responders)))
    print("Number of low responders: {}".format(len(low_responders)))

    train_normal, test_normal = train_test_split(normal_responders, test_size=0.3, random_state=0)
    train_patients = train_normal
    test_patients = test_normal + high_responders + low_responders

    return train_patients, test_patients


def clean_cycles(cycles):
    # remove patients whose trigger day is below 0 or above 25
    cycles = [cycle for cycle in cycles if 0 < cycle.trigger_day <= 25]
    # remove patients who have less than 2 profiles
    cycles = [cycle for cycle in cycles if len(cycle.profiles) >= 2]

    # remove patients who have profiles on days before 0 or after 25
    clean_patients = []
    for cycle in cycles:
        add_flag = True
        for scan in list(cycle.profiles.keys()):
            if int(scan) > 25 or int(scan) < 3:
                # print(patient.key, scan)
                # if patient.key == 'NUR-70583':
                #     raise Exception("NUR-70583 with scan {}".format(scan))
                # patients.remove(patient)
                add_flag = False
                break
        if add_flag:
            clean_patients.append(cycle)

    return cycles


def add_features_to_cycles(cycles, data, tqdm_disable=False):
    for cycle in tqdm(cycles, disable=tqdm_disable):
        row = data[data['Cycle Number'] == cycle.key]
        if row.empty:
            continue
        age = row['Age at Egg Collection'].values[0]
        if not pd.isnull(age):
            cycle.age = age

        weight = row['Weight_kg'].values[0]
        if not pd.isnull(weight) and 30 <= weight <= 120:
            cycle.weight = weight

        amh = row['amh_value'].values[0]
        if not pd.isnull(amh):
            cycle.amh = amh

        afc = row['AFC_result'].values[0]
        if not pd.isnull(afc) and 4 <= afc:
            cycle.afc_count = afc

        responder = row['responder'].values[0]
        if not responder == 'UNKNOWN' and not pd.isnull(responder):
            cycle.responder_class = responder

        clinic = row['Clinic'].values[0]
        if not pd.isnull(clinic):
            cycle.clinic = clinic

        patient_id = row['PatientIdentifier'].values[0]
        if not pd.isnull(patient_id):
            cycle.patient_id = patient_id

        os_init_dose = row['os_init_dose'].values[0]
        if not pd.isnull(os_init_dose) and 75 <= os_init_dose <= 900:
            # print(os_init_dose)
            cycle.os_initial_dose = os_init_dose
            change = row['os_pct_chng_dose'].values[0]
            if not pd.isnull(change) and 75 <= os_init_dose <= 900:
                cycle.init_dose_change = change
        # else:
        #     menopur_dose = row['meno_init_dose'].values[0]
        #     if not pd.isnull(menopur_dose):
        #         cycle.os_initial_dose = menopur_dose
        #         change = row['meno_pct_chng_dose'].values[0]
        #         if not pd.isnull(change):
        #             cycle.init_dose_change = change

        protocol_map = row['protocol_map'].values[0]
        if not pd.isnull(protocol_map):
            cycle.suppressant_protocol = protocol_map



    return cycles


def split_cycles_by_age(cycles):
    # Split cycles into 3 groups by age
    cycles_35 = [cycle for cycle in cycles if cycle.age >= 35]
    cycles_30 = [cycle for cycle in cycles if 30 <= cycle.age < 35]
    cycles_25 = [cycle for cycle in cycles if cycle.age < 30]

    return cycles_35, cycles_30, cycles_25


def split_cycles_by_protocol(cycles):
    # Split cycles into 3 groups by age
    cycles_LONG = [cycle for cycle in cycles if cycle.suppressant_protocol == 'LONG']
    cycles_SHORT = [cycle for cycle in cycles if cycle.suppressant_protocol == 'SHORT']

    return cycles_LONG, cycles_SHORT

def split_cycles_by_weight(cycles):
    # Split cycles into 3 groups by weight
    cycles_80 = [cycle for cycle in cycles if cycle.weight >= 80]
    cycles_70 = [cycle for cycle in cycles if 60 <= cycle.weight < 80]
    cycles_60 = [cycle for cycle in cycles if cycle.weight < 60]

    return cycles_60, cycles_70, cycles_80


def split_cycles_by_amh(cycles):
    # Split cycles into 3 groups by amh 0, 10,25,100
    cycles_10 = [cycle for cycle in cycles if cycle.amh <= 10]
    cycles_25 = [cycle for cycle in cycles if 25 >= cycle.amh > 10]
    cycles_100 = [cycle for cycle in cycles if cycle.amh > 25]
    return cycles_10, cycles_25, cycles_100


def split_cycles_by_afc(cycles):
    # Split cycles into 3 groups by amh 0, 10, 20, 30
    cycles_10 = [cycle for cycle in cycles if cycle.afc_count <= 10]
    cycles_20 = [cycle for cycle in cycles if 20 >= cycle.afc_count > 10]
    cycles_30 = [cycle for cycle in cycles if cycle.afc_count > 20]
    return cycles_10, cycles_20, cycles_30

def split_cycles_by_os_init_dose(cycles):
    # Split cycles into 3 groups by initial dose 0, 150, 300
    cycles_0_150 = [cycle for cycle in cycles if cycle.os_initial_dose <= 150]
    cycles_150_300 = [cycle for cycle in cycles if 300 >= cycle.os_initial_dose > 150]
    cycles_300 = [cycle for cycle in cycles if cycle.os_initial_dose > 300]
    return cycles_0_150, cycles_150_300, cycles_300

#############
# Dictionaries
#############

def split_cycles_by_group_conditions_dict(cycles, groups_conditions):
    # Split cycles into 3 groups by weight
    cycles_dict = {
        group: [cycle for cycle in cycles if groups_conditions[group](cycle)]
        for group in groups_conditions.keys()
    }

    return cycles_dict

def split_cycles_by_supp_protocol_dict(cycles):
    # Split cycles into 3 groups by weight
    protocol_groups_conditions = {'LONG': lambda suppressant_protocol: suppressant_protocol == 'LONG',
                                'SHORT': lambda suppressant_protocol: suppressant_protocol == 'SHORT'}
    protocol_cycles_dict = {
        group: [cycle for cycle in cycles if protocol_groups_conditions[group](cycle.suppressant_protocol)]
        for group in protocol_groups_conditions.keys()
    }

    return protocol_cycles_dict
def split_cycles_by_weight_dict(cycles):
    # Split cycles into 3 groups by weight
    weight_groups_conditions = {'<60': lambda weight: weight < 60,
                                '60-80': lambda weight: 60 <= weight < 80,
                                '>80': lambda weight: weight >= 80}
    weight_cycles_dict = {
        group: [cycle for cycle in cycles if weight_groups_conditions[group](cycle.weight)]
        for group in weight_groups_conditions.keys()
    }

    return weight_cycles_dict


def split_cycles_by_age_dict(cycles):
    # Split cycles into 3 groups by age
    age_groups_conditions = {'<30': lambda age: age < 30,
                             '30-35': lambda age: 30 <= age < 35,
                             '>35': lambda age: age >= 35}
    age_cycles_dict = {
        group: [cycle for cycle in cycles if age_groups_conditions[group](cycle.age)]
        for group in age_groups_conditions.keys()
    }

    return age_cycles_dict


def split_cycles_by_amh_dict(cycles):
    # Split cycles into 3 groups by amh 0, 10,25,100
    amh_groups_conditions = {'<10': lambda amh: amh <= 10,
                             '10-25': lambda amh: 10 < amh <= 25,
                             '>25': lambda amh: amh > 25}
    amh_cycles_dict = {
        group: [cycle for cycle in cycles if amh_groups_conditions[group](cycle.amh)]
        for group in amh_groups_conditions.keys()
    }

    return amh_cycles_dict


def split_cycles_by_afc_dict(cycles):
    # Split cycles into 3 groups by amh 0, 10, 20, 30
    afc_groups_conditions = {'<10': lambda afc: afc <= 10,
                             '10-20': lambda afc: 10 < afc <= 20,
                             '>20': lambda afc: afc > 20}
    afc_cycles_dict = {
        group: [cycle for cycle in cycles if afc_groups_conditions[group](cycle.afc_count)]
        for group in afc_groups_conditions.keys()
    }

    return afc_cycles_dict


def split_cycles_by_os_init_dose_dict(cycles):
    # Split cycles into 3 groups by initial dose 0, 150, 300
    os_init_dose_groups_conditions = {'<150': lambda os_init_dose: os_init_dose <= 150,
                                      '150-300': lambda os_init_dose: 150 < os_init_dose <= 300,
                                      '>300': lambda os_init_dose: os_init_dose > 300}
    os_init_dose_cycles_dict = {
        group: [cycle for cycle in cycles if os_init_dose_groups_conditions[group](cycle.os_initial_dose)]
        for group in os_init_dose_groups_conditions.keys()
    }

    return os_init_dose_cycles_dict


def split_cycles_by_clinic(cycles):
    clinics = ['BELF', 'BOST', 'CMM', 'GCRM', 'NURT', 'OXFD', 'SIMP', 'THVF', 'VITR', 'WESX', 'WFC']
    clinic_cycles_dict = {
        clinic: [cycle for cycle in cycles if cycle.clinic == clinic]
        for clinic in clinics
    }

    return clinic_cycles_dict

#############
# Rest
#############

def make_bins_profiles(cycles):
    # check that all follicles are above 5
    for cycle in cycles:
        for profile in cycle.profiles.values():
            profile.make_bins()
