import time

import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from core.stats_gen import logger


def gen_hists_follicle_based(follicles_dict, gaussian_filter=False, sigma=1):
    for follicle in follicles_dict:
        follicles_dict[follicle]['hist_absolute_growth'] = np.histogram(follicles_dict[follicle]['absolute_growth'])
        if gaussian_filter:
            counts = follicles_dict[follicle]['hist_absolute_growth'][0]
            smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
            follicles_dict[follicle]['hist_absolute_growth'] = (smoothed_counts,
                                                                follicles_dict[follicle]['hist_absolute_growth'][1])

        follicles_dict[follicle]['chance_of_growth'] = len(follicles_dict[follicle]["next_follicles"]) / \
                                                       follicles_dict[follicle]["num_total"]
        follicles_dict[follicle]['diff_absolute_median'] = np.median(
            np.mean(follicles_dict[follicle]['absolute_growth']))
        follicles_dict[follicle]['diff_absolute_mean'] = np.mean(np.mean(follicles_dict[follicle]['absolute_growth']))

        for dosage in follicles_dict[follicle]['chance_of_growth_per_dosage']:
            grew = follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage]['grew']
            did_not_grow = follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage]['did_not_grow']
            try:
                follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage]['chance_of_growth'] = grew / (
                        grew + did_not_grow)
            except:
                if grew + did_not_grow == 0:
                    follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage]['chance_of_growth'] = None


def add_to_chance_per_dosage(follicles_dict, dosage, follicle, did_grew):
    dosages_per_weight_bins = {'<1.5': 1.5, '1.5-1.75': 1.75, '1.75-2': 2, '2-2.25': 2.25, '2.25-2.5': 2.5,
                               '2.5-3.5': 3.5, '3.5-5': 5, '5-7.5': 7.5, '7.5-10': 10}
    if 'chance_of_growth_per_dosage' not in follicles_dict[follicle]:
        follicles_dict[follicle]['chance_of_growth_per_dosage'] = {}
        for dosage_bin in dosages_per_weight_bins:
            follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage_bin] = {'grew': 0, 'did_not_grow': 0}

    try:
        dosage = int(dosage)
    except:
        # logger.warning("Dosage is not a number: {}".format(dosage))
        return
    if dosage == 0:
        return
    for dosage_bin in dosages_per_weight_bins:
        if dosage < dosages_per_weight_bins[dosage_bin]:
            # print(dosage)
            # print(dosages_per_weight_bins[dosage_bin])
            follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage_bin]['grew'] += 1 if did_grew else 0
            follicles_dict[follicle]['chance_of_growth_per_dosage'][dosage_bin]['did_not_grow'] += 0 if did_grew else 1
            return


def generate_stats_follicle_based(train_patients=None, n_sim=1, tqdm_disable=False, include_negative_change=False,
                                  gaussian_filter=False, include_2_uniform=False):
    '''
    Generates stats for follicle based analysis. Independent of day.
    Args:
        train_patients:  list of patients
        n_sim:  number of simulations
        tqdm_disable:  disable tqdm progress bar
        include_negative_change:  include negative changes in the stats

    Returns:

    '''
    start_time = time.time()
    follicles_dict = {}
    logger.info("Generating stats for follicle based analysis")
    neg_growth_count = 0
    neg_follicle_second = []
    total_count = 0

    chance_of_growth_per_dosage = {}

    for patient in tqdm(train_patients, disable=tqdm_disable):
        list_of_profiles = list(patient.profiles.values())
        for idx, profile in enumerate(list_of_profiles):

            # Check if we are at the last profile, if so, skip
            if idx + 1 == len(patient.profiles.values()):
                continue

            # Get the follicles for the current and next profile
            day_x: int = int(profile.day)
            day_y: int = int(list_of_profiles[idx + 1].day)
            follicles_x: list[int] = list(reversed(profile.follicles))
            follicles_y: list[int] = list(reversed(list_of_profiles[idx + 1].follicles))

            # Check if the days are consecutive, if not, skip
            divide_by = 1
            if include_2_uniform:
                if day_y - day_x != 1 and day_y - day_x != 2:
                    continue
                if day_y - day_x == 2:
                    divide_by = 2
            else:
                if day_y - day_x != 1:
                    continue

            # Iterate through the follicles in the current profile
            for idk, follicle in enumerate(follicles_x):
                total_count += 1
                if idk + 1 == len(follicles_x):
                    continue
                if idk + 1 == len(follicles_y):
                    break
                if follicles_y[idk] - follicle < 0:
                    neg_growth_count += 1
                    neg_follicle_second.append(follicles_y[idk])

                if follicle in follicles_dict:

                    follicles_dict[follicle]['num_total'] += 1
                    if follicles_y[idk] - follicle > 0 or include_negative_change:
                        follicles_dict[follicle]['next_follicles'].append(follicles_y[idk])
                        follicles_dict[follicle]['absolute_growth'].append(abs(follicles_y[idk] - follicle)/divide_by)
                        add_to_chance_per_dosage(follicles_dict, profile.drug_dosage_per_weight, follicle, True)
                    else:
                        add_to_chance_per_dosage(follicles_dict, profile.drug_dosage_per_weight, follicle, False)

                else:
                    if follicles_y[idk] - follicle > 0 or include_negative_change:
                        follicles_dict[follicle] = {'next_follicles': [follicles_y[idk]],
                                                    'absolute_growth': [abs(follicles_y[idk] - follicle)/divide_by]}
                        follicles_dict[follicle]['num_total'] = 1
                        add_to_chance_per_dosage(follicles_dict, profile.drug_dosage_per_weight, follicle, True)
                    else:
                        follicles_dict[follicle] = {'num_total': 1,
                                                    'next_follicles': [],
                                                    'absolute_growth': []}
                        add_to_chance_per_dosage(follicles_dict, profile.drug_dosage_per_weight, follicle, False)

    # follicles_dict['chance_of_growth_per_dosage'] = chance_of_growth_per_dosage
    gen_hists_follicle_based(follicles_dict, gaussian_filter=gaussian_filter)

    follicles_dict['total_cycles_used'] = len(train_patients)
    follicles_dict['total_follicles_used'] = total_count
    follicles_dict['total_negative_growth_count'] = neg_growth_count
    follicles_dict['total_negative_growth_follicle_second'] = neg_follicle_second

    logger.info(f"Time to generate follicle based stats: {time.time() - start_time}")
    return follicles_dict


def gen_hists_follicle_based_with_days(follicles_dict):
    for follicle in follicles_dict:
        for day in follicles_dict[follicle]:
            follicles_dict[follicle][day]['hist_absolute_growth'] = np.histogram(
                follicles_dict[follicle][day]['absolute_growth'])
            follicles_dict[follicle][day]['chance_of_growth'] = len(follicles_dict[follicle][day]["next_follicles"]) / \
                                                                follicles_dict[follicle][day]["num_total"]
            follicles_dict[follicle][day]['diff_absolute_mean'] = np.mean(
                follicles_dict[follicle][day]['absolute_growth'])


def generate_stats_follicle_based_with_days(train_patients=None, n_sim=1, tqdm_disable=False):
    start_time = time.time()
    # follicles_dict = {}
    logger.info("Generating stats for follicle based analysis with days")
    follicles_dict = {}
    for patient in tqdm(train_patients, disable=tqdm_disable):
        list_of_profiles = list(patient.profiles.values())
        for idx, profile in enumerate(list_of_profiles):

            if idx + 1 == len(patient.profiles.values()):
                continue

            day_x: int = int(profile.day)
            day_y: int = int(list_of_profiles[idx + 1].day)
            follicles_x: list[int] = list(reversed(profile.follicles))
            follicles_y: list[int] = list(reversed(list_of_profiles[idx + 1].follicles))

            if day_y - day_x != 1 and day_y - day_x != 2:
                continue

            for idk, follicle in enumerate(follicles_x):
                if idk + 1 == len(follicles_x):
                    continue
                if idk + 1 == len(follicles_y):
                    break
                if follicle in follicles_dict:
                    if day_x in follicles_dict[follicle]:
                        follicles_dict[follicle][day_x]['num_total'] += 1
                        if follicles_y[idk] - follicle > 0:
                            follicles_dict[follicle][day_x]['next_follicles'].append(follicles_y[idk])
                            follicles_dict[follicle][day_x]['absolute_growth'].append(abs(follicles_y[idk] - follicle) if day_y - day_x == 1 else abs(follicles_y[idk] - follicle)/2)
                    else:
                        follicles_dict[follicle][day_x] = {'num_total': 1,
                                                           'next_follicles': [],
                                                           'absolute_growth': []}
                        if follicles_y[idk] - follicle > 0:
                            follicles_dict[follicle][day_x]['next_follicles'].append(follicles_y[idk])
                            follicles_dict[follicle][day_x]['absolute_growth'].append(abs(follicles_y[idk] - follicle) if day_y - day_x == 1 else abs(follicles_y[idk] - follicle)/2)

                else:
                    follicles_dict[follicle] = {}
                    if follicles_y[idk] - follicle > 0:
                        follicles_dict[follicle][day_x] = {'next_follicles': [follicles_y[idk]],
                                                           'absolute_growth': [abs(follicles_y[idk] - follicle) if day_y - day_x == 1 else abs(follicles_y[idk] - follicle)/2]}
                        follicles_dict[follicle][day_x]['num_total'] = 1
                    else:
                        follicles_dict[follicle][day_x] = {'num_total': 1,
                                                           'next_follicles': [],
                                                           'absolute_growth': []}
    gen_hists_follicle_based_with_days(follicles_dict)

    logger.info(f"Time to generate follicle based stats: {time.time() - start_time}")
    return follicles_dict

def generate_stats_days(train_patients=None, n_sim=1, tqdm_disable=False):
    start_time = time.time()
    # follicles_dict = {}
    logger.info("Generating stats for follicle based analysis with days")
    follicles_dict = {}
    for patient in tqdm(train_patients, disable=tqdm_disable):
        list_of_profiles = list(patient.profiles.values())
        for idx, profile in enumerate(list_of_profiles):

            if idx + 1 == len(patient.profiles.values()):
                continue

            day_x: int = int(profile.day)
            day_y: int = int(list_of_profiles[idx + 1].day)
            follicles_x: list[int] = list(reversed(profile.follicles))
            follicles_y: list[int] = list(reversed(list_of_profiles[idx + 1].follicles))

            # if day_y - day_x != 1:
            #     continue

            for idk, follicle in enumerate(follicles_x):
                if idk + 1 == len(follicles_x):
                    continue
                if idk + 1 == len(follicles_y):
                    break
                if day_x in follicles_dict:
                    if day_y in follicles_dict[day_x]:
                        follicles_dict[day_x][day_y]['num_total'] += 1
                        if follicles_y[idk] - follicle > 0:
                            follicles_dict[day_x][day_y]['next_follicles'].append(follicles_y[idk])
                            follicles_dict[day_x][day_y]['absolute_growth'].append(abs(follicles_y[idk] - follicle))
                    else:
                        follicles_dict[day_x][day_y] = {'num_total': 1,
                                                           'next_follicles': [],
                                                           'absolute_growth': []}
                        if follicles_y[idk] - follicle > 0:
                            follicles_dict[day_x][day_y]['next_follicles'].append(follicles_y[idk])
                            follicles_dict[day_x][day_y]['absolute_growth'].append(abs(follicles_y[idk] - follicle))

                else:
                    follicles_dict[day_x] = {}
                    if follicles_y[idk] - follicle > 0:
                        follicles_dict[day_x][day_y] = {'next_follicles': [follicles_y[idk]],
                                                           'absolute_growth': [abs(follicles_y[idk] - follicle)]}
                        follicles_dict[day_x][day_y]['num_total'] = 1
                    else:
                        follicles_dict[day_x][day_y] = {'num_total': 1,
                                                           'next_follicles': [],
                                                           'absolute_growth': []}
    gen_hists_follicle_based_with_days(follicles_dict)

    logger.info(f"Time to generate follicle based stats: {time.time() - start_time}")
    return follicles_dict