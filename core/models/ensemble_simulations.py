import random

import numpy as np

from core.models.histostep import _simulate_follicles_bin_based, \
    _simulate_follicles_follicle_based


def simulate_profiles_follicles(profile,
                        model,
                        dayx,
                        dayy,
                        predict_using_perc_growth=False,
                        n_sim=1,
                        is_weighted=False,
                        use_bin_act_only=False,
                        prev_act_day=None,
                        prev_act_profile=None,
                        mean_lag=1,
                        bin_only=False,
                        follicle_only=False,
                        return_probas=False):
    '''
    Simulates a profile follicles using bin based and follicle based simulation
    Args:
        profile:
        dict_bin_stats:
        dict_follicle_stats:
        predict_on_actual_data:
        predict_using_perc_growth:

    Returns:
        Simulated Profile
    '''
    np.random.seed(0)
    random.seed(0)

    follicles_dict, bins_dict = model

    simulated_follicles_fol = _simulate_follicles_follicle_based(profile.follicles, follicles_dict, n_sim=n_sim,
                                                                 mean_lag=mean_lag,
                                                                 return_probas=return_probas)

    if use_bin_act_only:
        simulated_follicles_bin = _simulate_follicles_bin_based(prev_act_day, dayy, bins_dict, predict_using_perc_growth,
                                                                prev_act_profile, n_sim=n_sim,
                                                                mean_lag=mean_lag,
                                                                 return_probas=return_probas)
    else:
        simulated_follicles_bin = _simulate_follicles_bin_based(dayx, dayy, bins_dict, predict_using_perc_growth,
                                                                profile, n_sim=n_sim,
                                                                mean_lag=mean_lag,
                                                                 return_probas=return_probas)

    # print('Simulated follicles follicle based: ' + str(simulated_follicles_fol))
    # print('Simulated follicles bin based: ' + str(simulated_follicles_bin))
    # print('Length follicles follicle based: ' + str(len(simulated_follicles_fol)))
    # print('Length follicles bin based: ' + str(len(simulated_follicles_bin)))
    if bin_only:
        return simulated_follicles_bin
    if follicle_only:
        return simulated_follicles_fol
    # Average the two
    simulated_follicles = []
    for idx in range(len(simulated_follicles_fol) if len(simulated_follicles_fol) < len(simulated_follicles_bin) else len(simulated_follicles_bin)):
        # print('Follicles follicle based: ' + str(simulated_follicles_fol[idx]))
        # print('Follicles bin based: ' + str(simulated_follicles_bin[idx]))
        # print('Average: ' + str(int((simulated_follicles_fol[idx] + simulated_follicles_bin[idx]) / 2)))
        simulated_follicles.append((simulated_follicles_fol[idx] + simulated_follicles_bin[idx]) / 2)

    return simulated_follicles

def simulate_cycle_ensemble(cycle, dayx, dayy,
                     models,
                     predict_on_actual_data=False,
                     predict_using_perc_growth=False
                     ):
    '''
    Simulates a cycle using bin based and follicle based simulation
    Args:
        cycle:
        dayx:
        dayy:
        models:
        predict_on_actual_data:
        predict_using_perc_growth:

    Returns:

    '''

