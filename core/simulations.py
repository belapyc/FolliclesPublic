import logging

import numpy as np
import pickle
from data_model.cycle_model.Profile import Profile
from core.models.histostep import _simulate_follicles_bin_based, \
    _simulate_follicles_follicle_based
from core.sim_helper import _simulate_updated_days_of_trigger, _simulate_days_of_trigger, \
    _get_prev_profiles, _gen_new_prof_from_candidates, _reinit_sim_get_first_day
from core.helper_core import get_next_day
import random

LAST_DAY_TO_PREDICT = 18


def load_data(stats_path='dict_histogram_stats.pickle', ):
    with open(stats_path, 'rb') as handle:
        dict_histo_matrix = pickle.load(handle)

    return dict_histo_matrix,


def _simulate_follicles_constant_growth(follicles, growth_rate=1.35):
    # print('Simulating follicles with constant growth')
    # Simulate follicles for profile
    simulated_follicles = []
    for follicle in follicles:
        new_follicle = follicle + growth_rate
        simulated_follicles.append(new_follicle)
    simulated_follicles.sort()
    return simulated_follicles


def combined_simulation(profile, combined_follicles_dict, dict_histo_matrix, predict_using_perc_growth, n_sim, mean_lag,
                        day, dayy, use_drug_dosage):
    simulated_follicles_fol = _simulate_follicles_follicle_based(profile.follicles, combined_follicles_dict,
                                                                 n_sim, profile.drug_dosage_per_weight, use_drug_dosage)
    simulated_follicles_bin = _simulate_follicles_bin_based(day, dayy, dict_histo_matrix, predict_using_perc_growth,
                                                            profile,
                                                            n_sim, mean_lag)
    # Average the two
    simulated_follicles = []
    for idx in range(len(simulated_follicles_fol)):
        simulated_follicles.append((simulated_follicles_fol[idx] + simulated_follicles_bin[idx]) / 2)

    return simulated_follicles


# TODO change name
def simulate_patient(patient, dict_histo_matrix,
                     bin_based=True,
                     predict_on_actual_data=False,
                     predict_using_perc_growth=False,
                     start_sim_day=0,
                     n_sim=1,
                     mean_lag=1,
                     is_update=False,
                     drug_dosage=0,
                     use_drug_dosage=False,
                     use_constant_growth=False,
                     combined=False,
                     combined_follicles_dict=None):
    np.random.seed(0)
    random.seed(0)

    first_day = _reinit_sim_get_first_day(patient, is_update)

    for i in range(start_sim_day):
        first_day = get_next_day(first_day, patient.profiles)

    if first_day == 'Not found':
        raise Exception(
            'First day not found for patient {} with profile days {}'.format(patient.key, patient.profiles.keys()))

    for i in range(int(patient.profiles[first_day].day), LAST_DAY_TO_PREDICT):

        day = str(i)
        dayy = str(i + 1)

        # Use actual profile if it is the first profile or flag predict on actual is set to true
        if (int(i) == int(first_day)) or (str(i) in patient.profiles.keys() and predict_on_actual_data):
            profile = patient.profiles[str(i)]
        else:
            if is_update:
                profile = list(patient.updated_simulated_profiles.values())[-1]
            else:
                profile = list(patient.simulated_profiles.values())[-1]

        if bin_based:
            logging.info('Simulating day {} for patient {} using bin based simulation'.format(day, patient.key))
            simulated_follicles = _simulate_follicles_bin_based(day, dayy, dict_histo_matrix, predict_using_perc_growth,
                                                                profile,
                                                                n_sim, mean_lag)

        elif use_constant_growth:
            logging.info('Simulating day {} for patient {} using constant growth'.format(day, patient.key))
            simulated_follicles = _simulate_follicles_constant_growth(profile.follicles)

        elif combined:
            logging.info('Simulating day {} for patient {} using combined methods'.format(day, patient.key))

            # Average the two
            simulated_follicles = combined_simulation(profile=profile,combined_follicles_dict=combined_follicles_dict,
                                                        dict_histo_matrix=dict_histo_matrix,
                                                        predict_using_perc_growth=predict_using_perc_growth,
                                                        n_sim=n_sim, mean_lag=mean_lag,
                                                        day=day, dayy=dayy,
                                                        use_drug_dosage=use_drug_dosage)


        else:
            logging.info('Simulating day {} for patient {} using follicle based simulation'.format(day, patient.key))
            simulated_follicles = _simulate_follicles_follicle_based(profile.follicles, dict_histo_matrix,
                                                                     n_sim, profile.drug_dosage_per_weight,
                                                                     use_drug_dosage)

        simulated_profile = Profile(simulated_follicles, patient.key, i + 1)

        if is_update:
            patient.updated_simulated_profiles[str(i + 1)] = simulated_profile
            _simulate_updated_days_of_trigger(patient)
        else:
            logging.info(
                'Simulated profile for day: ' + str(i + 1) + ' with ' + str(simulated_profile.follicles) + ' follicles')
            patient.simulated_profiles[str(i + 1)] = simulated_profile
            _simulate_days_of_trigger(patient)



def simulate_patient_weighted(patient, dict_histo_matrix,
                              use_percentage_increase=True,
                              pred_on_actual_data=False,
                              time_based_weights=True,
                              n_sim=1,
                              return_profile=False,
                              predict_until = LAST_DAY_TO_PREDICT):
    np.random.seed(0)
    random.seed(0)

    first_day = _reinit_sim_get_first_day(patient)

    for i in range(int(patient.profiles[first_day].day) + 1, predict_until):
        day_to_simulate = str(i)
        previous_days = [str(x) for x in range(int(patient.profiles[first_day].day), i)]
        len_prev_days = len(previous_days)

        previous_profiles = _get_prev_profiles(first_day, len_prev_days, patient, pred_on_actual_data)

        candidate_profiles = []

        # Create candidate profiles
        for prev_profile in previous_profiles:
            simulated_follicles = _simulate_follicles_bin_based(int(prev_profile.day),
                                                                day_to_simulate,
                                                                dict_histo_matrix,
                                                                use_percentage_increase,
                                                                prev_profile,
                                                                n_sim=n_sim)

            # TODO: we assume that all bins have the same number of samples 'top' naming
            # Generate weights. If time based, use the number of days since the previous day
            if str(int(prev_profile.day)) not in dict_histo_matrix.keys():
                # for now follicle does not grow
                day_diff_weight: float = 1 / (int(day_to_simulate) - int(prev_profile.day))
                candidate_profiles.append((prev_profile.follicles, 1, day_diff_weight))
                print('Day X {} not in histo matrix'.format(day_to_simulate))
                continue
            if str(day_to_simulate) not in dict_histo_matrix[str(int(prev_profile.day))]['top'].keys():
                day_diff_weight: float = 1 / (int(day_to_simulate) - int(prev_profile.day))
                candidate_profiles.append((prev_profile.follicles, 1, day_diff_weight))
                print('Day Y {} not in histo matrix'.format(day_to_simulate))
                continue

            num_samples = dict_histo_matrix[str(int(prev_profile.day))]['top'][day_to_simulate]['num_samples']
            day_diff_weight: float = 1 / (int(day_to_simulate) - int(prev_profile.day))

            candidate_profiles.append((simulated_follicles, num_samples, day_diff_weight))

        # TODO: change here to sort and simulate days_of_trigger
        if len(candidate_profiles) == 1:
            simulated_profile = Profile(candidate_profiles[0][0], patient.key, int(day_to_simulate))
            patient.simulated_profiles[day_to_simulate] = simulated_profile
            continue

        new_profile = _gen_new_prof_from_candidates(candidate_profiles, time_based_weights)
        new_profile.sort()

        # Create new profile and add to simulated profiles
        simulated_profile = Profile(new_profile, patient.key, int(day_to_simulate))
        if return_profile:
            return simulated_profile
        patient.simulated_profiles[day_to_simulate] = simulated_profile
        _simulate_days_of_trigger(patient)
