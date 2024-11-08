from itertools import cycle

import numpy as np
import pandas as pd
from tqdm import tqdm
import math

from data_model.cycle_model.Profile import Profile
from core.models.ensemble_simulations import simulate_profiles_follicles
from core.models.models_helper import get_afc_model, get_amh_model, get_age_model, \
    get_weight_model, get_os_init_dose_model
from core.sim_helper import _reinit_sim_get_first_day
from core.stats_gen import generate_stats_bin_based
from core.stats_gen_follicle import generate_stats_follicle_based
from data_prep.data_prep_tfp import split_cycles_by_afc, split_cycles_by_os_init_dose
from data_prep.data_prep_tfp import split_cycles_by_weight
from data_prep.data_prep_tfp import split_cycles_by_amh
from data_prep.data_prep_tfp import split_cycles_by_age
from helpers.results_generation import histosection
from modelling.RF_ensemble import RFEnsemble
from modelling.helper import calculate_lag_profile, calculate_lag, average_histo_rf_preds, \
    average_histo_rf_preds_bins


class HistostepModel:

    def __init__(self):
        self.models = None

    # def train(self, train_patients, gaussian_filter=False):
    #     # Split cycles based on age into 3 groups
    #
    #     cycles_old, cycles_middle, cycles_young = split_cycles_by_age(train_patients)
    #     # Split cycles based on amh into 3 groups
    #
    #     cycles_low_amh, cycles_middle_amh, cycles_high_amh = split_cycles_by_amh(
    #         [cycle for cycle in train_patients if cycle.amh is not None])
    #     # Split cycles based on weight into 3 groups
    #
    #     cycles_low_weight, cycles_middle_weight, cycles_high_weight = split_cycles_by_weight(train_patients)
    #     # Split cycles based on afc into 3 groups
    #
    #     cycles_low_afc, cycles_middle_afc, cycles_high_afc = split_cycles_by_afc(
    #         [cycle for cycle in train_patients if cycle.afc_count is not None])
    #
    #     # Split cycles based on initial dose into 3 groups
    #     cycles_low_init_dose, cycles_middle_init_dose, cycles_high_init_dose = split_cycles_by_os_init_dose(
    #         [cycle for cycle in train_patients if cycle.os_initial_dose is not None]
    #     )
    #     # Generate stats for all groups
    #
    #     bins = ['top', 'upper', 'lower', 'bottom']
    #
    #     dicts_follicles_per_age = {'age_35_inf': None, 'age_30_35': None, 'age_0_30': None}
    #     dicts_hist_grew_per_age = {'age_35_inf': None, 'age_30_35': None, 'age_0_30': None}
    #
    #     dicts_follicles_per_amh = {'amh_0_10': None, 'amh_10_25': None, 'amh_25_inf': None}
    #     dicts_hist_grew_per_amh = {'amh_0_10': None, 'amh_10_25': None, 'amh_25_inf': None}
    #
    #     dicts_follicles_per_weight = {'weight_0_60': None, 'weight_60_80': None, 'weight_80_inf': None}
    #     dicts_hist_grew_per_weight = {'weight_0_60': None, 'weight_60_80': None, 'weight_80_inf': None}
    #
    #     dicts_follicles_per_afc = {'afc_0_10': None, 'afc_10_20': None, 'afc_20_inf': None}
    #     dicts_hist_grew_per_afc = {'afc_0_10': None, 'afc_10_20': None, 'afc_20_inf': None}
    #
    #     dicts_follicles_per_init_dose = {'init_dose_0_150': None, 'init_dose_150_300': None, 'init_dose_300_inf': None}
    #     dicts_hist_grew_per_init_dose = {'init_dose_0_150': None, 'init_dose_150_300': None, 'init_dose_300_inf': None}
    #
    #     dicts_follicles_per_age['age_35_inf'] = generate_stats_follicle_based(train_patients=cycles_old,
    #                                                                           gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_age['age_30_35'] = generate_stats_follicle_based(train_patients=cycles_middle,
    #                                                                          gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_age['age_0_30'] = generate_stats_follicle_based(train_patients=cycles_young,
    #                                                                         gaussian_filter=gaussian_filter)
    #
    #     dicts_hist_grew_per_age['age_35_inf'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                         train_patients=cycles_old,
    #                                                                         gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_age['age_30_35'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                        train_patients=cycles_middle,
    #                                                                        gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_age['age_0_30'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                       train_patients=cycles_young,
    #                                                                       gaussian_filter=gaussian_filter)
    #
    #     dicts_follicles_per_amh['amh_0_10'] = generate_stats_follicle_based(train_patients=cycles_low_amh,
    #                                                                         gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_amh['amh_10_25'] = generate_stats_follicle_based(train_patients=cycles_middle_amh,
    #                                                                          gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_amh['amh_25_inf'] = generate_stats_follicle_based(train_patients=cycles_high_amh,
    #                                                                           gaussian_filter=gaussian_filter)
    #
    #     dicts_hist_grew_per_amh['amh_0_10'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                       train_patients=cycles_low_amh,
    #                                                                       gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_amh['amh_10_25'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                        train_patients=cycles_middle_amh,
    #                                                                        gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_amh['amh_25_inf'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                         train_patients=cycles_high_amh,
    #                                                                         gaussian_filter=gaussian_filter)
    #
    #     dicts_follicles_per_weight['weight_0_60'] = generate_stats_follicle_based(train_patients=cycles_low_weight,
    #                                                                               gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_weight['weight_60_80'] = generate_stats_follicle_based(train_patients=cycles_middle_weight,
    #                                                                                gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_weight['weight_80_inf'] = generate_stats_follicle_based(train_patients=cycles_high_weight,
    #                                                                                 gaussian_filter=gaussian_filter)
    #
    #     dicts_hist_grew_per_weight['weight_0_60'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                             train_patients=cycles_low_weight,
    #                                                                             gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_weight['weight_60_80'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                              train_patients=cycles_middle_weight,
    #                                                                              gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_weight['weight_80_inf'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                               train_patients=cycles_high_weight,
    #                                                                               gaussian_filter=gaussian_filter)
    #
    #     dicts_follicles_per_afc['afc_0_10'] = generate_stats_follicle_based(train_patients=cycles_low_afc,
    #                                                                         gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_afc['afc_10_20'] = generate_stats_follicle_based(train_patients=cycles_middle_afc,
    #                                                                          gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_afc['afc_20_inf'] = generate_stats_follicle_based(train_patients=cycles_high_afc,
    #                                                                           gaussian_filter=gaussian_filter)
    #
    #     dicts_hist_grew_per_afc['afc_0_10'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                       train_patients=cycles_low_afc,
    #                                                                       gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_afc['afc_10_20'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                        train_patients=cycles_middle_afc,
    #                                                                        gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_afc['afc_20_inf'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                         train_patients=cycles_high_afc,
    #                                                                         gaussian_filter=gaussian_filter)
    #
    #     dicts_follicles_per_init_dose['init_dose_0_150'] = generate_stats_follicle_based(
    #         train_patients=cycles_low_init_dose,
    #         gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_init_dose['init_dose_150_300'] = generate_stats_follicle_based(
    #         train_patients=cycles_middle_init_dose,
    #         gaussian_filter=gaussian_filter)
    #     dicts_follicles_per_init_dose['init_dose_300_inf'] = generate_stats_follicle_based(
    #         train_patients=cycles_high_init_dose,
    #         gaussian_filter=gaussian_filter)
    #
    #     dicts_hist_grew_per_init_dose['init_dose_0_150'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                                    train_patients=cycles_low_init_dose,
    #                                                                                    gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_init_dose['init_dose_150_300'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                                      train_patients=cycles_middle_init_dose,
    #                                                                                      gaussian_filter=gaussian_filter)
    #     dicts_hist_grew_per_init_dose['init_dose_300_inf'], _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                                                      train_patients=cycles_high_init_dose,
    #                                                                                      gaussian_filter=gaussian_filter)
    #
    #     follicles_dict_base = generate_stats_follicle_based(train_patients=train_patients,
    #                                                         gaussian_filter=gaussian_filter)
    #
    #     bin_dict_base, _ = generate_stats_bin_based(return_object=True, bins=bins,
    #                                                 number_of_testing_samples=len(train_patients),
    #                                                 train_patients=train_patients,
    #                                                 gaussian_filter=gaussian_filter)
    #     models = {
    #         'age_35_inf': (dicts_follicles_per_age['age_35_inf'], dicts_hist_grew_per_age['age_35_inf']),
    #         'age_30_35': (dicts_follicles_per_age['age_30_35'], dicts_hist_grew_per_age['age_30_35']),
    #         'age_0_30': (dicts_follicles_per_age['age_0_30'], dicts_hist_grew_per_age['age_0_30']),
    #         'amh_0_10': (dicts_follicles_per_amh['amh_0_10'], dicts_hist_grew_per_amh['amh_0_10']),
    #         'amh_10_25': (dicts_follicles_per_amh['amh_10_25'], dicts_hist_grew_per_amh['amh_10_25']),
    #         'amh_25_inf': (dicts_follicles_per_amh['amh_25_inf'], dicts_hist_grew_per_amh['amh_25_inf']),
    #         'weight_0_60': (dicts_follicles_per_weight['weight_0_60'], dicts_hist_grew_per_weight['weight_0_60']),
    #         'weight_60_80': (dicts_follicles_per_weight['weight_60_80'], dicts_hist_grew_per_weight['weight_60_80']),
    #         'weight_80_inf': (dicts_follicles_per_weight['weight_80_inf'], dicts_hist_grew_per_weight['weight_80_inf']),
    #         'afc_0_10': (dicts_follicles_per_afc['afc_0_10'], dicts_hist_grew_per_afc['afc_0_10']),
    #         'afc_10_20': (dicts_follicles_per_afc['afc_10_20'], dicts_hist_grew_per_afc['afc_10_20']),
    #         'afc_20_inf': (dicts_follicles_per_afc['afc_20_inf'], dicts_hist_grew_per_afc['afc_20_inf']),
    #         'init_dose_0_150': (dicts_follicles_per_init_dose['init_dose_0_150'],
    #                             dicts_hist_grew_per_init_dose['init_dose_0_150']),
    #         'init_dose_150_300': (dicts_follicles_per_init_dose['init_dose_150_300'],
    #                                 dicts_hist_grew_per_init_dose['init_dose_150_300']),
    #         'init_dose_300_inf': (dicts_follicles_per_init_dose['init_dose_300_inf'],
    #                                 dicts_hist_grew_per_init_dose['init_dose_300_inf']),
    #         'base': (follicles_dict_base, bin_dict_base)
    #     }
    #
    #     self.models = models

    def check_score(self, test_patients, cut_off_5s=False, only_first=False, only_trigger=False, only_last=False, bins_spread=4):
        len(test_patients)
        # Check average Histosection score for high, normal and low responders
        kelsey_scores_high = []
        kelsey_scores_normal = []
        kelsey_scores_low = []
        score_patient_tuples = []
        for patient in test_patients:
            if patient.responder_class == "HIGH":
                kelsey_scores_high.append(
                    self.histosection_score_per_patient(patient, cut_off_5s=False, only_first=only_first,
                                                        only_trigger=only_trigger, only_last=only_last, bins_spread=bins_spread))
            elif patient.responder_class == "LOW":
                kelsey_scores_low.append(
                    self.histosection_score_per_patient(patient, cut_off_5s=False, only_first=only_first,
                                                        only_trigger=only_trigger, only_last=only_last, bins_spread=bins_spread))
            else:
                kelsey_scores_normal.append(
                    self.histosection_score_per_patient(patient, cut_off_5s=False, only_first=only_first,
                                                        only_trigger=only_trigger, only_last=only_last, bins_spread=bins_spread))
            score_patient_tuples.append(
                (self.histosection_score_per_patient(patient, cut_off_5s=False, only_first=only_first,
                                                     only_trigger=only_trigger, only_last=only_last, bins_spread=bins_spread), patient))

        # remove nan values
        kelsey_scores_high = [score for score in kelsey_scores_high if not math.isnan(score)]
        kelsey_scores_normal = [score for score in kelsey_scores_normal if not math.isnan(score)]
        kelsey_scores_low = [score for score in kelsey_scores_low if not math.isnan(score)]

        len(kelsey_scores_high), len(kelsey_scores_normal), len(kelsey_scores_low)
        print(np.mean(kelsey_scores_high), np.mean(kelsey_scores_normal), np.mean(kelsey_scores_low))
        return kelsey_scores_high + kelsey_scores_normal + kelsey_scores_low, score_patient_tuples

    # def predict(self, cycles, rf, num_scans_to_use=None, use_bin_act_only=False, bin_only=False,
    #             use_mean_lag=False, rf_ensemble=None):
    #     '''
    #     Predict follicle growth for a patient
    #     Args:
    #         cycle: cycle to predict
    #         models: histostep models
    #         rf: random forest model
    #         num_scans_to_use: number of scans to use for prediction
    #     '''
    #     simulated_patients = []
    #     skipped = 0
    #     if rf_ensemble is not None:
    #         #
    #         rf_ensemble = RFEnsemble(rf_ensemble)
    #         data_for_rf = rf_ensemble.prepare_data_for_pred(cycles)
    #         rf_preds = rf_ensemble.predict(data_for_rf)
    #         rf_preds['future_day'] = rf_preds['day'] + rf_preds['delta_future_day']
    #     for cycle in tqdm(cycles):
    #         cycle.rf_bins_preds = {}
    #         if rf_ensemble is not None:
    #             preds_for_cycle = rf_preds[rf_preds['id'] == cycle.key]
    #         cycle = self.predict_cycle(cycle, rf, num_scans_to_use=num_scans_to_use,
    #                                    use_bin_act_only=use_bin_act_only,
    #                                    use_mean_lag=use_mean_lag,
    #                                    bin_only=bin_only,
    #                                    rf_ensemble=rf_ensemble,
    #                                    rf_preds=preds_for_cycle if rf_ensemble is not None else None)
    #
    #         if cycle is None:
    #             skipped += 1
    #             continue
    #         simulated_patients.append(cycle)
    #     return simulated_patients, skipped

    @staticmethod
    def get_df_for_rf(cycle, dayx, dayy, bins):
        '''
        Get a dataframe for random forest prediction
        Args:
            profiles: profiles to predict
            dayx: dayx
            dayy: dayy

        Returns:
            dataframe
        '''
        afc = cycle.afc_count
        amh = cycle.amh
        age = cycle.age
        weight = cycle.weight
        profiles = cycle.profiles
        profile_days = list(profiles.keys())
        if dayx in profiles.keys():
            follicles = profiles[dayx].follicles
        # else get the last profile
        else:
            for i in range(int(dayx) - 1, 0, -1):
                if str(i) in profiles.keys():
                    follicles = profiles[str(i)].follicles
                    dayx = str(i)
                    break
        day_before = 0
        # Find profile before dayx
        for i in range(int(dayx) - 1, 0, -1):
            if str(i) in profiles.keys():
                # follicles = profiles[str(i)].follicles
                day_before = str(i)
                break

        # print('Day before: ' + str(day_before))
        # print('Day x: ' + str(dayx))
        # if day_before == 0:
        #     for bin_start in range(len(bins) - 1):
        #         follicles_features_dict_binned[bins_str[bin_start]] = 0

        # follicles_features_dict_binned = {}
        bins_str = [str(bins[i]) + '-' + str(bins[i + 1]) for i in range(len(bins) - 1)]
        follicles_features_dict_binned = dict(zip(bins_str, [0] * len(bins)))
        for bin_start in range(len(bins) - 1):
            follicles_features_dict_binned[bins_str[bin_start]] = len(
                [follicle for follicle in follicles if bins[bin_start] <= follicle < bins[bin_start + 1]])

            follicles_features_dict_binned['prev_' + bins_str[bin_start]] = len(
                [follicle for follicle in profiles[day_before].follicles
                 if bins[bin_start] <= follicle < bins[bin_start + 1]]) \
                if str(day_before) in profiles.keys() else 0

            # print(follicles_features_dict_binned)
            # print(bins)
        for bin_name in bins_str:
            follicles_features_dict_binned['delta_' + bin_name] = (follicles_features_dict_binned[bin_name] -
                                                                   follicles_features_dict_binned[
                                                                       'prev_' + bin_name]) \
                if str(day_before) in profiles.keys() else 0

        # print(day_before)
        # print(follicles_features_dict_binned)
        follicles_df = pd.DataFrame(follicles_features_dict_binned, index=[0])
        follicles_df['id'] = id
        follicles_df['day'] = dayx
        follicles_df['afc'] = afc
        follicles_df['amh'] = amh
        follicles_df['age'] = age
        follicles_df['weight'] = weight
        follicles_df['scan_num'] = profile_days.index(dayx) + 1
        follicles_df['delta_future_day'] = int(dayy) - int(dayx)
        follicles_df['delta_prev_day'] = int(dayx) - int(day_before) if day_before != 0 else 0
        follicles_df = follicles_df.drop(['id', 'afc', 'amh'], axis=1)
        # print('Before')
        follicles_df = follicles_df[['5-12', '12-15', '15-20', '20-26', 'day', 'age', 'weight', 'scan_num',
                                     'prev_5-12', 'prev_12-15', 'prev_15-20', 'prev_20-26',
                                     'delta_future_day', 'delta_prev_day', 'delta_5-12', 'delta_12-15',
                                     'delta_15-20', 'delta_20-26']]
        # print('After')
        # print(follicles_df)

        return follicles_df

    def predict_cycle(self, cycle, rf, num_scans_to_use=None, use_bin_act_only=False,
                      use_mean_lag=False,
                      bin_only=False,
                      rf_ensemble=None,
                      rf_preds=None,
                      ):
        '''
        Predict follicle growth for a patient
        Args:
            cycle:
            models:

        Returns:

        '''
        models = self.models
        cycle.histo_preds = {}

        values_dict = {}
        profiles = list(cycle.profiles.values())
        first_profile = profiles[0]
        first_profile_follicles = first_profile.follicles
        values_dict['total_num'] = len(first_profile_follicles)
        values_dict['top_bin_median'] = np.median(first_profile.bins['top'].follicles)
        values_dict['upper_bin_median'] = np.median(first_profile.bins['upper'].follicles)
        values_dict['lower_bin_median'] = np.median(first_profile.bins['lower'].follicles)
        values_dict['bottom_bin_median'] = np.median(first_profile.bins['bottom'].follicles)

        if rf is not None and rf.predict(pd.DataFrame(values_dict, index=[0])):
            return None

        # cycle.simulated_profiles = {}
        scans_used_counter = 0
        first_day = _reinit_sim_get_first_day(cycle, False)

        for i in range(int(cycle.profiles[first_day].day), 18):
            mean_lag = 1

            day = str(i)
            dayy = str(i + 1)

            if (rf_preds is not None) and (dayy in cycle.profiles.keys()):
                return_probas = False

                # get row for current day
                prediction = rf_preds[rf_preds['future_day'] == int(dayy)]

                if len(prediction) == 0:
                    raise Exception('No prediction for day ' + str(dayy))
                # print(rf_preds['future_day'])
                # print(prediction)

                bin_predictions = {}
                bins = rf_ensemble.models['bins']
                bins_str = [str(bins[i]) + '-' + str(bins[i + 1]) for i in range(len(bins) - 1)]
                for bin in bins_str:
                    bin_predictions[bin] = prediction[bin + '_pred'].iloc[0]
            else:
                return_probas = False

            # if current day is the first day or we have a profile for this day use actual data
            if (int(i) == int(first_day)) or (str(i) in cycle.profiles.keys()):

                prev_profile = cycle.profiles[str(i)]

                prev_day = str(i)
                if num_scans_to_use is None:
                    profile = cycle.profiles[str(i)]
                elif num_scans_to_use is not None and scans_used_counter < num_scans_to_use:
                    scans_used_counter += 1
                    profile = cycle.profiles[str(i)]
                else:
                    profile = list(cycle.simulated_profiles.values())[-1]
            # else use simulated data
            else:
                profile = list(cycle.simulated_profiles.values())[-1]

            if use_mean_lag and int(i) != int(first_day):
                mean_lag = mean_lag * calculate_lag(cycle.simulated_profiles[str(i)].follicles, profile.follicles)
                # print(mean_lag)

            n_sim = 1
            if cycle.amh is None:
                simulated_amh = simulate_profiles_follicles(profile, models['base'],
                                                            dayx=day,
                                                            dayy=dayy,
                                                            n_sim=n_sim,
                                                            prev_act_day=prev_day,
                                                            prev_act_profile=prev_profile,
                                                            use_bin_act_only=use_bin_act_only,
                                                            mean_lag=mean_lag,
                                                            bin_only=bin_only,
                                                            return_probas=return_probas)
            else:
                simulated_amh = simulate_profiles_follicles(profile, get_amh_model(models, cycle),
                                                            dayx=day,
                                                            dayy=dayy,
                                                            n_sim=n_sim,
                                                            prev_act_day=prev_day,
                                                            prev_act_profile=prev_profile,
                                                            use_bin_act_only=use_bin_act_only,
                                                            mean_lag=mean_lag,
                                                            bin_only=bin_only,
                                                            return_probas=return_probas)

            if cycle.afc_count is None:
                simulated_afc = simulate_profiles_follicles(profile, models['base'],
                                                            dayx=day,
                                                            dayy=dayy,
                                                            n_sim=n_sim,
                                                            prev_act_day=prev_day,
                                                            prev_act_profile=prev_profile,
                                                            use_bin_act_only=use_bin_act_only,
                                                            mean_lag=mean_lag,
                                                            bin_only=bin_only,
                                                            return_probas=return_probas)
            else:
                simulated_afc = simulate_profiles_follicles(profile, get_afc_model(models, cycle),
                                                            dayx=day,
                                                            dayy=dayy,
                                                            n_sim=n_sim,
                                                            prev_act_day=prev_day,
                                                            prev_act_profile=prev_profile,
                                                            use_bin_act_only=use_bin_act_only,
                                                            mean_lag=mean_lag,
                                                            bin_only=bin_only,
                                                            return_probas=return_probas)

            if cycle.os_initial_dose is None:
                simulated_init_dose = simulate_profiles_follicles(profile, models['base'],
                                                               dayx=day,
                                                               dayy=dayy,
                                                               n_sim=n_sim,
                                                               prev_act_day=prev_day,
                                                               prev_act_profile=prev_profile,
                                                               use_bin_act_only=use_bin_act_only,
                                                               mean_lag=mean_lag,
                                                               bin_only=bin_only,
                                                               return_probas=return_probas)
            else:
                simulated_init_dose = simulate_profiles_follicles(profile, get_os_init_dose_model(models, cycle),
                                                               dayx=day,
                                                               dayy=dayy,
                                                               n_sim=n_sim,
                                                               prev_act_day=prev_day,
                                                               prev_act_profile=prev_profile,
                                                               use_bin_act_only=use_bin_act_only,
                                                               mean_lag=mean_lag,
                                                               bin_only=bin_only,
                                                               return_probas=return_probas)

            simulated_age = simulate_profiles_follicles(profile, get_age_model(models, cycle),
                                                        dayx=day,
                                                        dayy=dayy,
                                                        n_sim=n_sim,
                                                        prev_act_day=prev_day,
                                                        prev_act_profile=prev_profile,
                                                        use_bin_act_only=use_bin_act_only,
                                                        mean_lag=mean_lag,
                                                        bin_only=bin_only,
                                                        return_probas=return_probas)

            simulated_weight = simulate_profiles_follicles(profile, get_weight_model(models, cycle),
                                                           dayx=day,
                                                           dayy=dayy,
                                                           n_sim=n_sim,
                                                           prev_act_day=prev_day,
                                                           prev_act_profile=prev_profile,
                                                           use_bin_act_only=use_bin_act_only,
                                                           mean_lag=mean_lag,
                                                           bin_only=bin_only,
                                                           return_probas=return_probas)

            # Get the average of the 3 models
            simulated_follicles = []
            for k in range(len(profile.follicles)):
                # print('Length of follicles: ' + str(len(profile.follicles)))
                # print('Simulated follicles: ' + str(simulated_amh[k]) + ' ' + str(simulated_age[k]) + ' ' + str(
                #     simulated_weight[k]) + ' ' + str(simulated_afc[k]))
                simulated_follicles.append(
                    round((simulated_amh[k] + simulated_age[k] + simulated_weight[k] + simulated_afc[k] +
                           simulated_init_dose[k]) / 5))

            if str(dayy) in cycle.profiles.keys() and rf_ensemble is not None:
                # print(simulated_follicles)
                # Check simulated follicles bin counts

                # print('Simulated follicles bin counts: ' + str(simulated_follicles_bin_counts))
                # print('Bin predictions: ' + str(bin_predictions))
                cycle.rf_bins_preds[str(i + 1)] = bin_predictions
                cycle.histo_preds[str(i + 1)] = simulated_follicles

                # print(bin_predictions)
                corrected_simulated_follicles = average_histo_rf_preds_bins(simulated_follicles, bin_predictions)
                # print('Corrected simulated follicles: ' + str(corrected_simulated_follicles))
                simulated_follicles = corrected_simulated_follicles

                # get actual follicles bin counts
            while len(simulated_follicles) < 4:
                simulated_follicles.append(5)
                simulated_follicles.sort()
            simulated_profile = Profile(simulated_follicles, cycle.key, i + 1)

            cycle.simulated_profiles[str(i + 1)] = simulated_profile

        return cycle

    def histosection_score_per_patient(self, patient, cut_off_5s=False, only_first=False, only_trigger=False,
                                       gaussian_filter=False, only_last=False, bins_spread=4):
        score = []
        scans = []
        if cut_off_5s:
            # find the profile with the most follicles of size 5
            max_5s = 0
            for scan in patient.profiles.keys():
                number_of_5s = len([follicle for follicle in patient.profiles[scan].follicles if follicle == 5])
                if number_of_5s > max_5s:
                    max_5s = number_of_5s
                    max_5s_scan = scan
            # cut off n=max_5s follicles from all profiles
            for scan in patient.profiles.keys():
                scans.append(patient.profiles[scan].follicles[max_5s:])

        cut_off = 0
        if cut_off_5s:
            cut_off = max_5s

        if only_first:
            scan = list(patient.profiles.keys())[1]
            if int(scan) > 18 or int(scan) <= 5:
                return np.nan
            act_follicles = patient.profiles[scan].follicles[cut_off:]
            sim_follicles = patient.simulated_profiles[str(scan)].follicles[cut_off:]
            score = histosection(act_follicles, sim_follicles, bins_spread=bins_spread)
            return score

        if only_trigger:
            trigger_day = str(int(patient.trigger_day))
            if trigger_day in patient.profiles.keys():
                scan = int(trigger_day)
                if int(scan) > 18 or int(scan) <= 5:
                    return np.nan
                if str(scan) in patient.simulated_profiles.keys():
                    act_follicles = patient.profiles[str(scan)].follicles[cut_off:]
                    sim_follicles = patient.simulated_profiles[str(scan)].follicles[cut_off:]
                    score = histosection(act_follicles, sim_follicles, bins_spread=bins_spread)
                    return score
                else:
                    return np.nan
            else:
                return np.nan

        if only_last:
            scan = list(patient.profiles.keys())[-1]
            if int(scan) > 18 or int(scan) <= 5:
                return np.nan
            act_follicles = patient.profiles[scan].follicles[cut_off:]
            sim_follicles = patient.simulated_profiles[str(scan)].follicles[cut_off:]
            score = histosection(act_follicles, sim_follicles, bins_spread=bins_spread)
            return score

        for scan in list(patient.profiles.keys())[1:]:
            if int(scan) > 18 or int(scan) <= 5:
                continue
            act_follicles = patient.profiles[scan].follicles[cut_off:]
            sim_follicles = patient.simulated_profiles[scan].follicles[cut_off:]
            if gaussian_filter:
                sim_follicles = gaussian_filter1d(sim_follicles, sigma=2)
                sim_follicles = [round(follicle) for follicle in sim_follicles]
            score.append(histosection(act_follicles, sim_follicles, bins_spread=bins_spread))
        return np.mean(score)
