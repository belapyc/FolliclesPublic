import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_model.cycle_model.Profile import Profile
from core.models.ensemble_simulations import simulate_profiles_follicles
from core.sim_helper import _reinit_sim_get_first_day


def get_amh_model(models, patient):
    patient_amh = patient.amh
    if patient_amh is None:
        return None
    if patient_amh <= 10:
        return models['amh_0_10']
    elif 10 < patient_amh <= 25:
        return models['amh_10_25']
    elif patient_amh > 25:
        return models['amh_25_inf']


def get_age_model(models, patient):
    patient_age = patient.age
    if patient_age is None:
        return None
    if patient_age < 30:
        return models['age_0_30']
    elif patient_age < 35:
        return models['age_30_35']
    else:
        return models['age_35_inf']

def get_weight_model(models, patient):
    patient_weight = patient.weight
    if patient_weight is None:
        return None
    if patient_weight < 60:
        return models['weight_0_60']
    elif patient_weight < 80:
        return models['weight_60_80']
    else:
        return models['weight_80_inf']

def get_afc_model(models, patient):
    patient_afc = patient.afc_count
    if patient_afc is None:
        return None
    if patient_afc < 10:
        return models['afc_0_10']
    elif patient_afc < 20:
        return models['afc_10_20']
    else:
        return models['afc_20_inf']

def get_os_init_dose_model(models, patient):
    patient_os_init_dose = patient.os_initial_dose
    if patient_os_init_dose is None:
        return None
    if patient_os_init_dose < 150:
        return models['init_dose_0_150']
    elif patient_os_init_dose < 300:
        return models['init_dose_150_300']
    else:
        return models['init_dose_300_inf']

def cycles_to_df(cycles, check_responder=False, more_than_11=False, use_2_scans=False):
    values = []
    nums_11plus = []
    for cycle in cycles:
        if use_2_scans and (len(cycle.profiles) < 3):
            continue

        values_dict = {}

        values_dict['responder'] = cycle.responder_class
        values_dict['amh'] = cycle.amh
        values_dict['age'] = cycle.age
        values_dict['weight'] = cycle.weight
        values_dict['afc'] = cycle.afc_count

        profiles = list(cycle.profiles.values())
        num_11more = len([follicle for follicle in profiles[-1].follicles if (follicle >= 11 or more_than_11)])
        nums_11plus.append(num_11more)

        if check_responder:
            # print(int(profiles[-1].day))
            # print(int(cycle.trigger_day))
            if int(cycle.trigger_day) != int(profiles[-1].day):
                continue
            if num_11more > 18:
                values_dict['responder'] = 'HYPER'
            elif num_11more <= 3:
                # print('LOW')
                values_dict['responder'] = 'HYPO'
            else:
                # normal_hamm.append(cycle)
                values_dict['responder'] = 'NORMAL'
        # print([profile.day for profile in profiles])
        first_profile = profiles[0]
        first_profile_follicles = first_profile.follicles
        values_dict['total_num'] = len(first_profile_follicles)
        values_dict['top_bin_median'] = np.median(first_profile.bins['top'].follicles)
        values_dict['upper_bin_median'] = np.median(first_profile.bins['upper'].follicles)
        values_dict['lower_bin_median'] = np.median(first_profile.bins['lower'].follicles)
        values_dict['bottom_bin_median'] = np.median(first_profile.bins['bottom'].follicles)
        if use_2_scans:
            second_profile = profiles[1]
            second_profile_follicles = second_profile.follicles
            values_dict['total_num_second'] = len(second_profile_follicles)
            values_dict['top_bin_median_second'] = np.median(second_profile.bins['top'].follicles)
            values_dict['upper_bin_median_second'] = np.median(second_profile.bins['upper'].follicles)
            values_dict['lower_bin_median_second'] = np.median(second_profile.bins['lower'].follicles)
            values_dict['bottom_bin_median_second'] = np.median(second_profile.bins['bottom'].follicles)
        values_dict['trigger_day'] = cycle.trigger_day
        values.append(values_dict)
    df = pd.DataFrame(values)
    return df


def calc_95_ci(yyPredProb, yyTrue):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    # print(yyPredProb)

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(yyPredProb), len(yyPredProb))
        # print(indices)
        # print(new_df['response'].index)
        if len(np.unique(yyTrue[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(yyTrue[indices], yyPredProb[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))
    return confidence_lower, confidence_upper


# get confidence intervals of scores using bootstrap
from sklearn.utils import resample


def calc_scores_ci(scores):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    # print(yyPredProb)

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(scores), len(scores))
        # print(indices)
        # print(new_df['response'].index)

        score = np.mean(np.array(scores)[indices])

        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))
    return confidence_lower, confidence_upper