import cProfile
import os
import pickle
import pstats
import random
import sys
# import tkinter as tk
import warnings
from copy import deepcopy
from datetime import datetime
# from tkinter import simpledialog

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from modelling.helper import cycles_to_df_for_binpred, per_follicle_diffs_cycles, \
    plot_per_follicle_diffs
from modelling.helper  import df_to_long_xy
from modelling.histostep_model import HistostepModel
from modelling.histostep_model_v2 import HistostepModel2

np.random.seed(0)
random.seed(0)


def cycles_to_df(cycles, bin_size):
    df = cycles_to_df_for_binpred(cycles, bin_size=bin_size)
    df = df.drop(columns=['afc', 'amh'])
    return df


def data_prep_for_RF(cycles=None, train_patients=None, test_patients=None):
    if cycles:
        train_patients, test_patients = train_test_split(cycles, test_size=0.3, random_state=0)

    df_train_cleaner = cycles_to_df(train_patients, bin_size=3)
    df_test_cleaner = cycles_to_df(test_patients, bin_size=3)

    X_train, y_train = df_to_long_xy(df_train_cleaner, bin_size=3)
    X_test, y_test = df_to_long_xy(df_test_cleaner, bin_size=3)

    train_test_X_y = {}

    for bin in y_train.keys():
        train_test_X_y[bin] = {
            'X_train': X_train,
            'y_train': y_train[bin],
            'X_test': X_test,
            'y_test': y_test[bin]
        }
    return train_test_X_y, train_patients, test_patients


def load_all_cycles():
    with open('processed_data/cycles_tfp_trigscan_cutoff_amh_APR24.pkl', 'rb') as f:
        cycles_tfp = pickle.load(f)

    # with open('data/cycles_hamm_3DEC23.pkl', 'rb') as f:
    #     cycles_hamm = pickle.load(f)

    all_cycles = cycles_tfp  # + cycles_hamm

    for cycle in all_cycles:
        if cycle.weight is None:
            cycle.weight = -1

    return all_cycles


def produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='None', use_clinic=True):
    pass
    # save_figure_path = exp_folder + '/diffs_per_clinic' + '/' + name + '_all.png'
    # plot_per_follicle_diffs(results_dict['diffs_all'], save_figure_path)
    # for key in results_dict.keys():
    #     if 'diffs' in key and use_clinic:
    #         save_figure_path = exp_folder + '/diffs_per_clinic' + '/' + name + '_' + key + '.png'
    #         plot_per_follicle_diffs(results_dict[key], clinic=key.split('_')[0],
    #                                 save_path=save_figure_path)


def test_train_validation(all_cycles, best_params_dict, only_last=False, bins_results=4,
                          bin_only=False, follicle_only=False, use_rf=False, gaussian_filter=False, drop_scans=False,
                          drop_n = None, keep_12=False, drop_all_but_1=False):


    train_patients, test_patients = train_test_split(all_cycles, test_size=0.3, random_state=0)

    if drop_scans:
        test_patients = drop_scans_in_between(test_patients, num_scan_to_drop=drop_n, keep_12=keep_12, drop_all_but_1=drop_all_but_1)

    results_dict = {}
    if use_rf:
        train_test_X_y, train_patients, test_patients = data_prep_for_RF(train_patients=train_patients,
                                                                         test_patients=test_patients)

        models_dict = {}
        for bin in train_test_X_y.keys():
            print(bin)
            models_dict[bin] = RandomForestRegressor(n_estimators=round(best_params_dict[bin]['n_estimators']),
                                                     max_depth=round(best_params_dict[bin]['max_depth']),
                                                     min_samples_leaf=best_params_dict[bin]['min_samples_leaf'],
                                                     min_samples_split=best_params_dict[bin]['min_samples_split'],
                                                     random_state=42,
                                                     n_jobs=-1)

            models_dict[bin].fit(train_test_X_y[bin]['X_train'], train_test_X_y[bin]['y_train'])

        models_dict['bins'] = train_test_X_y.keys()
    histomodel = HistostepModel2()
    histomodel.train(train_patients, gaussian_filter=gaussian_filter)
    # remove protocol from predictors in histomodel.predictors
    histomodel.predictors = [x for x in histomodel.predictors if x.name != 'protocol']

    # only use overall predictor
    # histomodel.predictors = [x for x in histomodel.predictors if x.name == 'overall']
    if use_rf:
        simulated_corrected, _, _ = histomodel.predict(test_patients, use_bin_act_only=True, rf_ensemble=models_dict,
                                                       add_missing_dates=True, bin_only=bin_only,
                                                       follicle_only=follicle_only)
    else:
        simulated_corrected, _ = histomodel.predict(test_patients, use_bin_act_only=True, add_missing_dates=False,
                                                    bin_only=bin_only, follicle_only=follicle_only)

    scores = HistostepModel().check_score(simulated_corrected, only_first=False, only_trigger=False,
                                          only_last=only_last, bins_spread=bins_results)
    # get last real and simulated

    diffs = per_follicle_diffs_cycles(test_patients, only_last_scan=True)
    results_dict['diffs_all'] = diffs
    results_dict['all'] = scores[0]
    results_dict['all_cycles'] = scores[1]

    return results_dict, diffs


def cv_clinic(all_cycles, best_params_dict, only_last=False, bins_results=4, drop_scans=False,
                          drop_n = None, keep_12=False, drop_all_but_1=False):
    # perform leave one out using cycle.clinic
    clinics = [cycle.clinic for cycle in all_cycles]
    print('Cycles with clinic: {}'.format(len(clinics)))
    unique_clinics = list(set(clinics))
    unique_clinics.sort()
    results_dist = {}
    for clinic in unique_clinics:
        results_dist[clinic] = []
        results_dist[clinic + '_cycles'] = []
        train = [cycle for cycle in all_cycles if cycle.clinic != clinic]
        test = [cycle for cycle in all_cycles if cycle.clinic == clinic]
        if drop_scans:
            train = drop_scans_in_between(train, num_scan_to_drop=drop_n, keep_12=keep_12, drop_all_but_1=drop_all_but_1)
            test = drop_scans_in_between(test, num_scan_to_drop=drop_n, keep_12=keep_12, drop_all_but_1=drop_all_but_1)
        print('Clinic to leave out: {}'.format(clinic))
        print('Length of train: {}'.format(len(train)))
        print('Length of test: {}'.format(len(test)))
        print('Average number of scans in train: {}'.format(np.mean([len(cycle.profiles) for cycle in train])))
        print('Average number of scans in test: {}'.format(np.mean([len(cycle.profiles) for cycle in test])))


        train_test_X_y, train_patients, test_patients = data_prep_for_RF(train_patients=train, test_patients=test)

        models_dict = {}
        for bin in train_test_X_y.keys():
            print(bin)
            models_dict[bin] = RandomForestRegressor(n_estimators=round(best_params_dict[bin]['n_estimators']),
                                                     max_depth=round(best_params_dict[bin]['max_depth']),
                                                     min_samples_leaf=best_params_dict[bin]['min_samples_leaf'],
                                                     min_samples_split=best_params_dict[bin]['min_samples_split'],
                                                     random_state=42,
                                                     n_jobs=-1)

            models_dict[bin].fit(train_test_X_y[bin]['X_train'], train_test_X_y[bin]['y_train'])
        models_dict['bins'] = train_test_X_y.keys()
        histomodel = HistostepModel2()
        histomodel.train(train, gaussian_filter=True)

        # remove protocol from predictors in histomodel.predictors
        histomodel.predictors = [x for x in histomodel.predictors if x.name != 'protocol']

        simulated_corrected, _, _ = histomodel.predict(test, use_bin_act_only=True, rf_ensemble=models_dict,
                                                    add_missing_dates=True)
        scores = HistostepModel().check_score(simulated_corrected, only_first=False, only_trigger=False,
                                              only_last=only_last, bins_spread=bins_results)
        diffs_per_clinic = per_follicle_diffs_cycles(test, only_last_scan=True)

        results_dist[clinic].extend(scores[0])
        results_dist[clinic + '_cycles'].extend(scores[1])
        results_dist[clinic + '_diffs'] = diffs_per_clinic
        print(clinic)
        print(np.mean(scores[0]))

    # diffs = per_follicle_diffs_cycles(all_cycles)
    # results_dist['diffs_all'] = diffs

    return results_dist, diffs_per_clinic#, diffs


def drop_scans_in_between(cycles, num_scan_to_drop=None, keep_12=False, drop_all_but_1=False):
    # Remove all but the first and last scan
    cycles_tfp_1final_scan = deepcopy(cycles)
    for p in cycles_tfp_1final_scan:

        if num_scan_to_drop is not None:
            p.profiles.pop(list(p.profiles.keys())[num_scan_to_drop])
            continue
        # pop everything but the first and last scan
        if keep_12:
            for i in range(len(p.profiles) - 3):
                p.profiles.pop(list(p.profiles.keys())[-2])

            if len(p.profiles) != 3:
                raise 'ERROR in drop'
            continue

        for i in range(len(p.profiles) - 2):
            p.profiles.pop(list(p.profiles.keys())[1])
        if drop_all_but_1:
            p.profiles.pop(list(p.profiles.keys())[1])
            continue
        if len(p.profiles) != 2:
            raise 'ERROR'
    return cycles_tfp_1final_scan


#####################
# All Scans
#####################
def cv_clinic_next(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    # perform leave one out using cycle.clinic
    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, only_last=False, best_params_dict=best_params_dict,
                                    bins_results=bins_results)

    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='allscans_next')

    # print(results_dist)
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_next_' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_next(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, only_last=False, best_params_dict=best_params_dict,
                                                bins_results=bins_results)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='allscans_next', use_clinic=False)

    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_next_' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def cv_clinic_first_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    # perform leave one out using cycle.clinic
    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results, drop_scans=True)

    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='allscans_first_last')

    # print(results_dist)
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_first_final_' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_first_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    # do not use cycles with less than 3 scans
    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) >= 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    # test train validation
    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True, bins_results=bins_results, drop_scans=True)

    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='allscans_first_last', use_clinic=False)

    # print(results_dist)
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_first_final_' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def cv_clinic_12_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) >= 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results, drop_scans=True, keep_12=True)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='allscans_12_last')

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_cv_allscans_12_last1' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_12_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    # filter cycles to have at least 3 scans
    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) >= 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True,
                                                bins_results=bins_results, drop_scans=True, keep_12=True)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='allscans_12_last', use_clinic=False)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_allscans_12_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


#####################
# 4 Scans
#####################
def cv_clinic_4_last_drop_1(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results,
                                    drop_scans=True, drop_n=1)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scans_drop1_last')
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_4scans_drop1_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_4_last_drop_1(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True,
                                                bins_results=bins_results, drop_scans=True, drop_n=1)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scans_drop1_last', use_clinic=False)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_4scans_drop1_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def cv_clinic_4_last_drop_2(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results,
                                    drop_scans=True, drop_n=2)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scans_drop2_last')
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_4scans_drop2_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_4_last_drop_2(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True,
                                                bins_results=bins_results, drop_scans=True, drop_n=2)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scans_drop2_last', use_clinic=False)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_4scans_drop2_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def cv_clinic_4_last_drop_12(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results,
                                    drop_scans=True)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scan_drop12_last')

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_4scans_drop12_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_4_last_drop_12(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True,
                                                bins_results=bins_results, drop_scans=True)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scan_drop12_last', use_clinic=False)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_4scans_drop12_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def cv_clinic_4_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results)

    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scans_last')

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_4scans_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_4_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 4]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True, bins_results=bins_results)

    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='4scans_last', use_clinic=False)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_4scans_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


#####################
# 3 Scans
#####################

def cv_clinic_3_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='3scans_last')
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_3scans_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_3_last(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True, bins_results=bins_results)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='3scans_last', use_clinic=False)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_3scans_last' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def cv_clinic_3_drop_1(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = cv_clinic(all_cycles, best_params_dict, only_last=True, bins_results=bins_results,
                                    drop_scans=True, drop_n=1)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='3scans_drop1')

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_3scans_drop1' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def tt_3_drop_1(exp_folder, bins_results=4):
    all_cycles = load_all_cycles()

    all_cycles = [cycle for cycle in all_cycles if len(cycle.profiles) == 3]

    with open('best_params_all_scans.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)

    results_dict, diffs = test_train_validation(all_cycles, best_params_dict, only_last=True,
                                                bins_results=bins_results, drop_scans=True, drop_n=1)
    produce_figures_for_diffs_per_clinic(results_dict, exp_folder, name='3scans_drop1', use_clinic=False)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(exp_folder + '/scores_tt_3scans_drop1' + timestamp + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


def run_all_tt(exp_folder, bins_spread):
    print('Running all tt')
    print()
    tt_4_last(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_4_last done')
    tt_4_last_drop_1(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_4_last_drop_1 done')
    tt_4_last_drop_2(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_4_last_drop_2 done')
    tt_4_last_drop_12(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_4_last_drop_12 done')
    tt_3_last(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_3_last done')
    tt_3_drop_1(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_3_drop_1 done')
    tt_next(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_next done')
    tt_first_last(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_first_last done')
    tt_12_last(exp_folder=exp_folder, bins_results=bins_spread)
    print('tt_12_last done')


def run_all_cv(exp_folder, bins_spread):
    print('Running all cv')
    # cv_clinic_4_last(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_4_last done')
    # cv_clinic_4_last_drop_1(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_4_last_drop_1 done')
    # cv_clinic_4_last_drop_2(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_4_last_drop_2 done')
    # cv_clinic_4_last_drop_12(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_4_last_drop_12 done')
    # cv_clinic_3_last(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_3_last done')
    # cv_clinic_3_drop_1(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_3_drop_1 done')
    # cv_clinic_next(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_next done')
    # cv_clinic_first_last(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_first_last done')
    # cv_clinic_12_last(exp_folder=exp_folder, bins_results=bins_spread)
    # print('cv_clinic_12_last done')


def run():
    print(len(sys.argv))
    if len(sys.argv) == 3:
        exp_name = sys.argv[1]
        bins_spread = sys.argv[2]
    else:
        exp_name = "diss_combined_tt"
        bins_spread = 3
    # else:
    #     ROOT = tk.Tk()
    #
    #     ROOT.withdraw()
    #
    #     exp_name = simpledialog.askstring(title="Experiment name",
    #                                       prompt="Enter experiment name:")
    #     bins_spread = simpledialog.askinteger(title="Bins spread",
    #                                           prompt="Enter bins spread:")

    bins_spread_str = 'bins_spread_{}'.format(bins_spread)
    if int(bins_spread) == 3:
        bins_spread = (5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 100)
    elif int(bins_spread) == 2:
        bins_spread = (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 100)
    elif int(bins_spread) == 1:
        bins_spread = [x + 5 for x in range(20)]
    else:
        bins_spread = 4

    exp_folder = 'exp_outputs/{}_{}_{}'.format(exp_name, bins_spread_str,
                                               datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    os.mkdir(exp_folder)
    os.mkdir(exp_folder + '/diffs_per_clinic')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    profiler = cProfile.Profile()
    profiler.enable()
    func = run_all_tt
    func(exp_folder, bins_spread)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats(
        'experiment_{}_{}.prof'.format(func.__name__, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


if __name__ == '__main__':
    run()
