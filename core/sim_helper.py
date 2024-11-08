import logging

import numpy as np

from core.helper_core import get_next_day

############################
# Helper functions
############################
def _simulate_updated_days_of_trigger(patient):
    max_12_19 = 0
    max_12_19_day = 0
    first_3_more = 0

    for x, sim_profile in enumerate(patient.updated_simulated_profiles.values()):
        sim_12_19: int = len([follicle for follicle in sim_profile.follicles if follicle >= 12 if follicle < 19])
        if sim_12_19 > max_12_19:
            max_12_19 = sim_12_19
            max_12_19_day = sim_profile.day
        if len([follicle for follicle in sim_profile.follicles if follicle > 18]) >= 3:
            if first_3_more == 0:
                first_3_more = sim_profile.day

    patient.updated_simulated_trigger_days['max_12_19'] = int(max_12_19_day)
    patient.updated_simulated_trigger_days['first_3_more'] = int(first_3_more)


def _simulate_days_of_trigger(patient):
    max_12_19 = 0
    max_12_19_day = 0
    first_3_more = 0
    first_2_more = 0

    for x, sim_profile in enumerate(patient.simulated_profiles.values()):
        sim_12_19: int = len([follicle for follicle in sim_profile.follicles if follicle >= 12 if follicle < 19])
        if sim_12_19 > max_12_19:
            max_12_19 = sim_12_19
            max_12_19_day = sim_profile.day
        if len([follicle for follicle in sim_profile.follicles if follicle > 18]) >= 2:
            if first_2_more == 0:
                first_2_more = sim_profile.day
        if len([follicle for follicle in sim_profile.follicles if follicle > 18]) >= 3:
            if first_3_more == 0:
                first_3_more = sim_profile.day

    patient.simulated_trigger_days['max_12_19'] = int(max_12_19_day)
    patient.simulated_trigger_days['first_3_more'] = int(first_3_more)
    patient.simulated_trigger_days['first_2_more'] = int(first_3_more)


def _get_prev_profiles(first_day, len_prev_days, patient, pred_on_actual_data):
    previous_profiles = [patient.profiles[str(int(first_day))]]
    # Append previous profile use actual if exists and flag is true
    for prev_simulated_profile in list(patient.simulated_profiles.values())[:len_prev_days - 1]:
        if (int(prev_simulated_profile.day) in patient.profiles) and pred_on_actual_data:
            previous_profiles.append(patient.profiles[int(prev_simulated_profile.day)])
        else:
            previous_profiles.append(prev_simulated_profile)
    return previous_profiles


def _gen_new_prof_from_candidates(candidate_profiles, time_based_weights):
    new_profile = []
    # Generate new profile using weighted average of follicle predictions
    for idx in range(len(candidate_profiles[0][0])):
        list_of_candidates = [x[0][idx] for x in candidate_profiles]
        if time_based_weights:
            list_of_weights = [x[1] * x[2] for x in candidate_profiles]
        else:
            list_of_weights = [x[1] for x in candidate_profiles]
        weighted_averaged_follicle = np.average(list_of_candidates, weights=list_of_weights)
        new_profile.append(weighted_averaged_follicle)
    return new_profile


def _reinit_sim_get_first_day(patient, is_update=False):
    if not is_update:
        logging.info('Re-initiating simulation for patient {}'.format(patient.key))
        patient.simulated_profiles = {}
        patient.simulated_trigger_days = {'max_12_19': None, 'first_3_more': None}
    patient.updated_simulated_profiles = {}
    patient.updated_simulated_trigger_days = {'max_12_19': None, 'first_3_more': None}

    first_day = get_next_day(0, patient.profiles)
    return first_day
