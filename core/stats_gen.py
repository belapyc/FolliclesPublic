import logging
import pickle
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_patients_file(file_path):
    with open(file_path, 'rb') as handle:
        patients = pickle.load(handle)
    return patients


def bin_stats_gen(binx, biny):
    '''
    Generate stats for given bins.

    Args:
        binx:
        biny:

    Returns:
        percent_grew : percentage of follicles which increased in size
        amounts_grew : list of amounts by which the follicles grew

    '''
    num_grew = 0
    amounts_grew = []
    percentages_grew = []
    for i, follicle in enumerate(binx):
        if biny[i] > binx[i]:
            num_grew += 1
            size_grew = biny[i] - binx[i]
            amounts_grew.append(size_grew)
            try:
                percentages_grew.append((biny[i] / binx[i]))
            except ZeroDivisionError:
                raise Exception('Zero division error in bin_stats_gen with binx: {} and biny: {}'.format(binx, biny))
    percent_grew = [num_grew / len(binx)]
    return percent_grew, amounts_grew, percentages_grew


def add_field_to_dict(dict_stats, bin_name, dayX, dayY, field_name, field_value):
    '''
    Adds a field to the dictionary of stats.
    Args:
        dict_stats:
        bin_name:
        dayX:
        dayY:
        field_name:
        field_value:

    Returns:

    '''
    dayX = str(int(dayX))
    dayY = str(int(dayY))

    if dayX not in dict_stats.keys():
        dict_stats[dayX] = {}
    if bin_name not in dict_stats[dayX].keys():
        dict_stats[dayX][bin_name] = {}
    if dayY not in dict_stats[dayX][bin_name].keys():
        dict_stats[dayX][bin_name][dayY] = {}
        dict_stats[dayX][bin_name][dayY]['num_samples'] = 0

    if field_name in dict_stats[dayX][bin_name][dayY].keys():
        dict_stats[dayX][bin_name][dayY][field_name].extend(field_value)
        # TODO check if this is correct. Does num samples correspond to the number of all samples or fields?
        dict_stats[dayX][bin_name][dayY]['num_samples'] = dict_stats[dayX][bin_name][dayY]['num_samples'] + 1
    else:
        dict_stats[dayX][bin_name][dayY][field_name] = field_value
        dict_stats[dayX][bin_name][dayY]['num_samples'] = dict_stats[dayX][bin_name][dayY]['num_samples'] + 1


def generate_hist_and_mean(dict_amounts_grew, gaussian_filter=False, sigma=1):
    '''
    Generates histogram of sizes grew for a particular pair of bins
    Args:
        dict_amounts_grew:

    Returns:

    '''
    for keyDx in dict_amounts_grew:
        for keyB in dict_amounts_grew[keyDx]:
            for keyDy in dict_amounts_grew[keyDx][keyB]:
                dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_amounts_grew'] = np.histogram(
                    dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['amounts'])

                if gaussian_filter:
                    counts = dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_amounts_grew'][0]
                    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
                    dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_amounts_grew'] = (
                        smoothed_counts, dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_amounts_grew'][1])

                dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_perc_growth'] = np.histogram(
                    dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['growth_in_perc'])

                if gaussian_filter:
                    counts = dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_perc_growth'][0]
                    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
                    dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_perc_growth'] = (
                        smoothed_counts, dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_perc_growth'][1])

                dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_number_grew'] = np.histogram(
                    dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['percent_grew'])

                if gaussian_filter:
                    counts = dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_number_grew'][0]
                    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
                    dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_number_grew'] = (
                        smoothed_counts, dict_amounts_grew[str(keyDx)][str(keyB)][keyDy]['hist_number_grew'][1])


def generate_stats_bin_based(output_file_path='dict_histogram_stats.pickle',
                             train_patients=None,
                             patients_file_path='all_patients_padded_clean_dict.pickle',
                             number_of_testing_samples=1000,
                             return_object=False,
                             save_object=False,
                             bins=['top', 'upper', 'lower', 'bottom'],
                             add_changes_to_patients=False,
                             tqdm_disable=False,
                             gaussian_filter=False):
    if train_patients is None:
        patients = read_patients_file(patients_file_path)
        train_patients = patients[:len(patients) - number_of_testing_samples]

    start_time = time.time()
    dict_amounts_grew = {}

    logger.info("Generating stats for bin based analysis")
    for patient in tqdm(train_patients, disable=tqdm_disable):

        for idx, profilex in enumerate(list(patient.profiles.values())):
            for idy, profiley in enumerate(list(patient.profiles.values())[idx + 1:]):

                if profiley.day <= profilex.day:
                    raise Exception('{} with day x: {} and day y: {}'.format(patient.key, profilex.day, profiley.day))

                if profilex.day > 18:
                    break
                # if profiley.day >= 19:
                #     break
                total_amount_grew = []
                for bin_name in bins:
                    percent_grew, amounts_grew, growth_in_perc = bin_stats_gen(profilex.bins[bin_name].follicles,
                                                                               profiley.bins[bin_name].follicles)

                    add_field_to_dict(dict_amounts_grew, bin_name, profilex.day, profiley.day, 'amounts', amounts_grew)
                    add_field_to_dict(dict_amounts_grew, bin_name, profilex.day, profiley.day, 'percent_grew',
                                      percent_grew)
                    add_field_to_dict(dict_amounts_grew, bin_name, profilex.day, profiley.day, 'growth_in_perc',
                                      growth_in_perc)
                    total_amount_grew.extend(amounts_grew)

                if len(total_amount_grew) == 0 or add_changes_to_patients:
                    continue
                if profilex.day in patient.changes.keys():
                    patient.changes[profilex.day][profiley.day] = total_amount_grew
                else:
                    patient.changes[profilex.day] = {profiley.day: total_amount_grew}

    generate_hist_and_mean(dict_amounts_grew=dict_amounts_grew, gaussian_filter=gaussian_filter)

    logger.info(f"Time to generate bin based stats: {time.time() - start_time}")

    if save_object:
        with open(output_file_path, 'wb') as handle:
            pickle.dump(dict_amounts_grew, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_object:
        return dict_amounts_grew, train_patients


def gen_changes_for_patient(patient, bins):
    for idx, profilex in enumerate(list(patient.profiles.values())):
        for idy, profiley in enumerate(list(patient.profiles.values())[idx + 1:]):
            total_amount_grew = []
            for bin_name in bins:
                percent_grew, amounts_grew, growth_in_perc = bin_stats_gen(profilex.bins[bin_name].follicles,
                                                                           profiley.bins[bin_name].follicles)
                total_amount_grew.extend(amounts_grew)
            if len(total_amount_grew) == 0:
                continue
            if profilex.day in patient.changes.keys():
                patient.changes[profilex.day][profiley.day] = total_amount_grew
            else:
                patient.changes[profilex.day] = {profiley.day: total_amount_grew}


def get_suitable_patients(patientx, dayx, dayy, patients, threshold=0.05):
    '''
    Returns patients that have changes histogram similar to patientx based
    on Kolmogorov-Smirnov test
    "Indeed, the p-value is lower than our threshold of 0.05,
    so we reject the null hypothesis in favor of the default
    “two-sided” alternative: the data were not drawn from the same distribution."
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
    Args:
        patientx: patient to compare to
        dayx: first day
        dayy: second day
        patients: list of patients

    Returns:
        list of patients that have similar changes histogram
    '''
    res = []
    suitable_patients = []
    for patient in patients:
        if dayx in patient.changes.keys():
            if dayy in patient.changes[dayx].keys():
                a = patient.changes[dayx][dayy]
                b = patientx.changes[dayx][dayy]
                res.append(stats.ks_2samp(a, b)[1])
                if stats.ks_2samp(a, b)[1] > threshold:
                    suitable_patients.append(patient)
    return suitable_patients
