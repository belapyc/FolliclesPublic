import logging
import math

import numpy as np
from scipy.stats import mannwhitneyu


def histosection(follicles_act, follicles_sim, bins_spread=4):
    if bins_spread == 4:
        bins = (5, 12, 15, 20, 26)
    else:
        bins = bins_spread
        # bins = (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 100)
        # bins = [x+5 for x in range(20)]

    A, _ = np.histogram(follicles_act, bins=bins)
    P, _ = np.histogram(follicles_sim, bins=bins)

    denom = np.sum(P)
    tmp = np.minimum(A, P)
    numerator = np.sum(tmp)

    return numerator / denom


def calc_kelsey_similarities(patients):
    similarities = []
    for patient in patients:
        for profile_key in list(patient.profiles.keys())[1:]:
            if int(profile_key) >= 18:
                continue

            follicles_act = patient.profiles[profile_key].follicles
            follicles_sim = patient.simulated_profiles[str(profile_key)].follicles
            similarity = histosection(follicles_act, follicles_sim)
            similarities.append(similarity)
    return np.mean(similarities)


def calc_p_value(patients):
    p_vals = []
    for patient in patients:
        for profile_key in list(patient.profiles.keys())[1:]:
            if int(profile_key) >= 18:
                continue
            # print(profile_key)
            follicles_act = patient.profiles[profile_key].follicles
            # print(patient.simulated_profiles.keys())
            follicles_sim = patient.simulated_profiles[str(profile_key)].follicles
            stat, p_value = mannwhitneyu(follicles_act, follicles_sim)
            p_vals.append(p_value)
    return np.mean(p_vals)


def calculate_exact_match(patients, max_12_19=True):
    num_correct = 0
    for i, patient in enumerate(patients):
        if max_12_19 and patient.simulated_trigger_days['max_12_19'] is not None:
            if patient.simulated_trigger_days['max_12_19'] == int(patient.trigger_day):
                num_correct += 1
                continue
        if patient.simulated_trigger_days['first_3_more'] is not None:
            if patient.simulated_trigger_days['first_3_more'] == int(patient.trigger_day):
                num_correct += 1
                continue

    return num_correct


def calculate_close_match(patients, max_12_19=True, first_2_more=False, first_3_more=True):
    num_close = 0
    miss_list = []
    for i, patient in enumerate(patients):

        trigger_daysub1 = int(patient.trigger_day) - 1
        trigger_dayadd1 = int(patient.trigger_day) + 1

        if first_2_more:
            if patient.simulated_trigger_days['first_2_more'] is not None:
                if trigger_daysub1 <= patient.simulated_trigger_days['first_2_more'] <= trigger_dayadd1:
                    num_close += 1
                    continue
            continue

        if max_12_19 and patient.simulated_trigger_days['max_12_19'] is not None:
            if trigger_daysub1 <= patient.simulated_trigger_days['max_12_19'] <= trigger_dayadd1:
                num_close += 1
                continue
        if first_3_more and patient.simulated_trigger_days['first_3_more'] is not None:
            if trigger_daysub1 <= patient.simulated_trigger_days['first_3_more'] <= trigger_dayadd1:
                num_close += 1
                continue
        miss_list.append(patient)

    return num_close, miss_list


def calculate_close_match_actual(patients):
    num_close = 0
    miss_list = []
    for i, patient in enumerate(patients):

        trigger_daysub1 = int(patient.trigger_day) - 1
        trigger_dayadd1 = int(patient.trigger_day) + 1

        if patient.simulated_trigger_days['max_12_19'] is not None:
            if trigger_daysub1 <= patient.simulated_trigger_days['max_12_19'] <= trigger_dayadd1:
                num_close += 1
                continue
        if patient.simulated_trigger_days['first_3_more'] is not None:
            if trigger_daysub1 <= patient.simulated_trigger_days['first_3_more'] <= trigger_dayadd1:
                num_close += 1
                continue
        miss_list.append(patient)

    return num_close, miss_list


def hamming_distance(profile_x, profile_y):
    return sum([1 if x != y else 0 for x, y in zip(profile_x, profile_y)]) / len(profile_x)


# def close_follicles(profile_x, profile_y):
#     return sum([1 if abs(x - y) <= 1 else 0 for x, y in zip(profile_x, profile_y)]) / len(profile_x)

def same_follicles(profile_x, profile_y):
    fol_x = profile_x.copy()
    fol_y = profile_y.copy()

    total_same = 0
    total = len(fol_x)
    for idx, x in enumerate(fol_x):
        for idy, y in enumerate(fol_y):
            if x == y:
                del fol_x[idx]
                del fol_y[idy]
                total_same += 1
    if total == 0:
        raise Exception("No follicles in profile")
    return total_same / total


def close_follicles(profile_x, profile_y):
    fol_x = profile_x.copy()
    fol_y = profile_y.copy()

    total_close = 0
    total = len(fol_x)
    for idx, x in enumerate(fol_x):
        for idy, y in enumerate(fol_y):
            if x == y:
                del fol_x[idx]
                del fol_y[idy]
                total_close += 1
                break

    for idx, x in enumerate(fol_x):
        for idy, y in enumerate(fol_y):
            if abs(x - y) <= 1:
                del fol_x[idx]
                del fol_y[idy]
                total_close += 1
                break

    return total_close / total


def close_follicles_ordered(profile_x, profile_y, proximity=1):
    fol_x = profile_x.copy()
    fol_y = profile_y.copy()

    total_close = 0
    total = len(fol_x)

    for idx, (x, y) in enumerate(zip(fol_x, fol_y)):
        if abs(x - y) <= proximity:
            total_close += 1

    return total_close / total


def calculate_average_follicle_accuracy(patient, ordered=True, proximity=1):
    percentages = []
    for profile in list(patient.profiles.keys())[1:]:
        if int(profile) >= 18:
            break
        print(patient.simulated_profiles)
        sim_profile = patient.simulated_profiles[profile].follicles
        actual_profile = patient.profiles[profile].follicles
        if ordered:
            percentage_close = close_follicles_ordered(actual_profile, sim_profile, proximity=proximity)
        else:
            percentage_close = close_follicles(actual_profile, sim_profile)
        percentages.append(percentage_close)
    if len(percentages) == 0:
        # raise Exception("Length is 0")
        print(patient.key)
        return None

    if math.isnan(np.mean(percentages)):
        raise Exception("No follicles in profile3")

    if np.mean(percentages) is None:
        print(percentages)
        raise Exception("No follicles in profile4")

    logging.info('Patient: %s, Average follicle accuracy: %s', patient.key, np.mean(percentages))

    return np.mean(percentages)


def calc_test_patients_average_accuracy(patients, ordered=True, proximity=1):
    '''
     Calculates the average follicle accuracy for a list of patients
     Accurate if simulated follicle is within proximity of the actual follicle

    Args:
        patients:  A list of patients
        ordered:  Whether follicles are ordered or not
        proximity:  The proximity of follicles to be considered close

    Returns:

    '''
    percentages = []
    for patient in patients:
        average_acc = calculate_average_follicle_accuracy(patient, ordered=ordered, proximity=proximity)
        if average_acc is None:
            continue
        percentages.append(average_acc)

    return np.mean(percentages)


# Calculation of differences in follicle sizes between actual and simulated follicles
def calculate_diff_in_follicles(patient):
    '''
        Calculates average differences in follicles for patient
    Args:
        patient:  A patient
    Returns:
    '''
    differences = []
    percentages = []
    for profile in list(patient.profiles.keys())[1:]:
        if int(profile) >= 18:
            break
        sim_profile = patient.simulated_profiles[profile].follicles
        actual_profile = patient.profiles[profile].follicles
        # Debugging
        print(sim_profile)
        print(actual_profile)
        print(np.array(sim_profile) - np.array(actual_profile))
        print(np.mean(np.array(sim_profile) - np.array(actual_profile)))
        print()
        print('------------------')
        percentages.append(
            np.mean(np.abs((np.array(sim_profile) - np.array(actual_profile)) / np.array(actual_profile))))
        differences.append(np.mean(np.array(sim_profile) - np.array(actual_profile)))

    return np.mean(percentages), np.mean(differences)


def calc_test_patients_accuracy(patients):
    percentages = []
    differences = []
    for patient in patients:
        percent, diff = calculate_diff_in_follicles(patient)
        if not math.isnan(percent):
            percentages.append(percent)

        if not math.isnan(diff):
            differences.append(diff)

    average_diff = np.mean(differences)
    average_percent = np.mean(percentages)

    return average_percent, average_diff


def calc_1219_for_patients(patients):
    percentages = []
    for patient in patients:
        percentages.append(calc_1219_accuracy_for_patient(patient))
    return np.mean(percentages)


def calc_1219_accuracy_for_patient(patient):
    number_profiles_correct = 0
    for profile in list(patient.profiles.keys())[1:]:
        if int(profile) >= 18:
            break
        sim_profile = patient.simulated_profiles[profile].follicles
        actual_profile = patient.profiles[profile].follicles

        sim_12to19 = [follicle for follicle in sim_profile if 12 <= follicle <= 19]
        actual_12to19 = [follicle for follicle in actual_profile if 12 <= follicle <= 19]

        if len(sim_12to19) == len(actual_12to19):
            number_profiles_correct += 1

    return number_profiles_correct / len(list(patient.profiles.keys())[1:])


def calc_1219_diff_for_patients(patients):
    differences = []
    for patient in patients:
        differences.append(calc_1219_diff_for_patient(patient))

    return np.mean(differences)


def calc_1219_diff_for_patient(patient):
    diffs = []
    for profile in list(patient.profiles.keys())[1:]:
        if int(profile) >= 18:
            break
        sim_profile = patient.simulated_profiles[profile].follicles
        actual_profile = patient.profiles[profile].follicles

        sim_12to19 = [follicle for follicle in sim_profile if 12 <= follicle <= 19]
        actual_12to19 = [follicle for follicle in actual_profile if 12 <= follicle <= 19]

        diff = abs(len(sim_12to19) - len(actual_12to19))
        diffs.append(diff)

    return np.mean(diffs)
