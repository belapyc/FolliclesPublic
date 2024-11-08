import numpy as np
from tqdm import tqdm

from data_model.cycle_model.Profile import Profile
from core.models.ensemble_simulations import simulate_profiles_follicles
from core.sim_helper import _reinit_sim_get_first_day
from modelling.Predictor import Predictor
from modelling.RF_ensemble import RFEnsemble
from modelling.helper import calculate_lag, average_histo_rf_preds_bins, add_rows_for_pred


class HistostepModel2:

    def __init__(self):
        self.predictors = []
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def train(self, train_patients, gaussian_filter=False, include_2_uniform=False):
        self.predictors = []
        bins = ['top', 'upper', 'lower', 'bottom']

        predictor_age = Predictor(name='age', groups_conditions={'<35': lambda cycle: cycle.age < 35,
                                                                 '>=35': lambda cycle: cycle.age >= 35},
                                  field_name_in_cycle='age')

        predictor_weight = Predictor(name='weight', groups_conditions={'<60': lambda cycle: cycle.weight < 60,
                                                                       '60-80': lambda cycle: 60 <= cycle.weight < 80,
                                                                       '>80': lambda cycle: cycle.weight >= 80},
                                     field_name_in_cycle='weight')

        predictor_afc = Predictor(name='afc', groups_conditions={'<10': lambda cycle: cycle.afc_count < 10,
                                                                 '10-20': lambda cycle: 10 <= cycle.afc_count < 20,
                                                                 '>20': lambda cycle: cycle.afc_count >= 20},
                                  field_name_in_cycle='afc_count')

        predictor_amh = Predictor(name='amh', groups_conditions={'<10': lambda cycle: cycle.amh < 10,
                                                                 '10-25': lambda cycle: 10 <= cycle.amh < 25,
                                                                 '>25': lambda cycle: cycle.amh >= 25},
                                  field_name_in_cycle='amh')

        predictor_init_dose = Predictor(name='init_dose',
                                        groups_conditions={'<150': lambda
                                            cycle: cycle.os_initial_dose < 150 and cycle.init_dose_change == 0,
                                                           '150-300': lambda
                                                               cycle: 150 <= cycle.os_initial_dose < 300 and cycle.init_dose_change == 0,
                                                           '>300': lambda
                                                               cycle: cycle.os_initial_dose >= 300 and cycle.init_dose_change == 0},
                                        field_name_in_cycle='os_initial_dose')

        predictor_protocol = Predictor(name='protocol',
                                       groups_conditions={
                                           'LONG': lambda cycle: cycle.suppressant_protocol == 'LONG',
                                           'SHORT': lambda cycle: cycle.suppressant_protocol == 'SHORT'},
                                       field_name_in_cycle='suppressant_protocol')

        clinic = ['BELF', 'BOST', 'CMM', 'GCRM', 'NURT', 'OXFD', 'SIMP', 'THVF', 'VITR', 'WESX', 'WFC']
        predictor_clinic = Predictor(name='clinic',
                                     groups_conditions={
                                         'BELF': lambda cycle: cycle.clinic == 'BELF',
                                         'BOST': lambda cycle: cycle.clinic == 'BOST',
                                         'CMM': lambda cycle: cycle.clinic == 'CMM',
                                         'GCRM': lambda cycle: cycle.clinic == 'GCRM',
                                         'NURT': lambda cycle: cycle.clinic == 'NURT',
                                         'OXFD': lambda cycle: cycle.clinic == 'OXFD',
                                         'SIMP': lambda cycle: cycle.clinic == 'SIMP',
                                         'THVF': lambda cycle: cycle.clinic == 'THVF',
                                         'VITR': lambda cycle: cycle.clinic == 'VITR',
                                         'WESX': lambda cycle: cycle.clinic == 'WESX',
                                         'WFC': lambda cycle: cycle.clinic == 'WFC'
                                     },
                                     field_name_in_cycle='clinic')

        predictor_overall = Predictor(name='overall', groups_conditions={'overall': lambda cycle: True},
                                      field_name_in_cycle=None)

        print('Generating stats for predictors')
        print('Predictor: age')
        predictor_age.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                     include_2_uniform=include_2_uniform)
        print('Predictor: weight')
        predictor_weight.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                        include_2_uniform=include_2_uniform)
        print('Predictor: afc')
        predictor_afc.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                     include_2_uniform=include_2_uniform)
        print('Predictor: init_dose')
        predictor_init_dose.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                           include_2_uniform=include_2_uniform)
        # print('Predictor: clinic')
        # predictor_clinic.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
        #                                 include_2_uniform=include_2_uniform)
        print('Predictor: protocol')
        predictor_protocol.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                          include_2_uniform=include_2_uniform)
        print('Predictor: overall')
        predictor_overall.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                         split_by_groups=False, include_2_uniform=include_2_uniform)
        print('Predictor: amh')
        predictor_amh.generate_stats(cycles=train_patients, gaussian_filter=gaussian_filter, bins=bins,
                                     include_2_uniform=include_2_uniform)

        self.predictors.append(predictor_age)
        self.predictors.append(predictor_weight)
        self.predictors.append(predictor_afc)
        self.predictors.append(predictor_init_dose)
        # self.predictors.append(predictor_clinic)
        self.predictors.append(predictor_amh)
        self.predictors.append(predictor_protocol)
        self.predictors.append(predictor_overall)

    def predict(self, cycles, num_scans_to_use=None, use_bin_act_only=False, bin_only=False, follicle_only=False,
                use_mean_lag=False, rf_ensemble=None, add_missing_dates=False):
        '''
        Predict follicle growth for a patient
        Args:
            cycle: cycle to predict
            models: histostep models
            rf: random forest model
            num_scans_to_use: number of scans to use for prediction
        '''
        simulated_patients = []
        skipped = 0
        if rf_ensemble is not None:
            # Predict bin counts using random forest
            rf_ensemble = RFEnsemble(rf_ensemble)
            data_for_rf = rf_ensemble.prepare_data_for_pred(cycles, num_scans_to_use=num_scans_to_use)

            # print('Data for rf ')
            # print(data_for_rf)

            if add_missing_dates:
                data_for_rf = add_rows_for_pred(data_for_rf)
            print('Data for rf ')
            print(data_for_rf)
            rf_preds = rf_ensemble.predict(data_for_rf)

            rf_preds['future_day'] = rf_preds['day'] + rf_preds['delta_future_day']
            print('RF predictions ')
            print(rf_preds)

        for cycle in tqdm(cycles):
            cycle.rf_bins_preds = {}
            if rf_ensemble is not None:
                preds_for_cycle = rf_preds[rf_preds['id'] == cycle.key]
            cycle = self.predict_cycle(cycle, num_scans_to_use=num_scans_to_use,
                                       use_bin_act_only=use_bin_act_only,
                                       use_mean_lag=use_mean_lag,
                                       bin_only=bin_only,
                                       follicle_only=follicle_only,
                                       rf_ensemble=rf_ensemble,
                                       rf_preds=preds_for_cycle if rf_ensemble is not None else None,
                                       predictor_weights_dict=self.weights)

            if cycle is None:
                skipped += 1
                continue
            simulated_patients.append(cycle)
        if rf_ensemble is not None:
            return simulated_patients, skipped, data_for_rf
        else:
            return simulated_patients, skipped

    def _predict_profile_using_overall(self, profile, dayx, dayy, n_sim=1, is_weighted=False, use_bin_act_only=False,
                                       prev_act_day=None, prev_act_profile=None, mean_lag=1, bin_only=False,
                                       return_probas=False, follicle_only=False):
        overall_predictor = [predictor for predictor in self.predictors if predictor.name == 'overall'][0]
        follicles_stats = overall_predictor.dicts_follicles_stats['overall']
        bins_stats = overall_predictor.dicts_bins_stats['overall']
        simulated_follicles = simulate_profiles_follicles(profile, (follicles_stats, bins_stats),
                                                          dayx=dayx, dayy=dayy, n_sim=n_sim,
                                                          use_bin_act_only=use_bin_act_only,
                                                          bin_only=bin_only,
                                                          follicle_only=follicle_only,
                                                          prev_act_day=prev_act_day,
                                                          prev_act_profile=prev_act_profile,
                                                          mean_lag=mean_lag,
                                                          return_probas=False)
        return simulated_follicles

    @staticmethod
    def _get_profiles_prev_current(cycle, i, first_day, num_scans_to_use, scans_used_counter, prev_profile=None):
        if int(i) == int(first_day):
            prev_profile = cycle.profiles[str(i)]
            profile = cycle.profiles[str(i)]
            prev_day = str(i)
            return prev_profile, prev_day, profile, scans_used_counter
        # print('Scans ', cycle.profiles.keys())
        # print('Lenght of scans ', [len(cycle.profiles[key].follicles) for key in cycle.profiles.keys()])
        # print('Length of simulated scans ', [len(cycle.simulated_profiles[key].follicles) for key in cycle.simulated_profiles.keys()])
        # print('Num to use ', num_scans_to_use)
        # print('Num used ', scans_used_counter)
        # print('prev_profile day ',prev_profile.day)
        prev_day = str(prev_profile.day)
        if num_scans_to_use is None:
            # print('Using actual data')
            profile = cycle.profiles[str(i)]
            prev_profile = cycle.profiles[str(i)]
            prev_day = str(i)
        elif num_scans_to_use is not None and scans_used_counter < num_scans_to_use:
            # print('Using actual data')
            scans_used_counter += 1
            profile = cycle.profiles[str(i)]
            prev_profile = cycle.profiles[str(i)]
            prev_day = str(i)
        elif num_scans_to_use is not None and scans_used_counter >= num_scans_to_use:
            # print('Using simulated data')
            # print('Using simulated data')
            profile = list(cycle.simulated_profiles.values())[-1]
        # print('profile day ',prev_profile.day)
        #
        # print('Actual follicles ', prev_profile.follicles, len(prev_profile.follicles))
        # print('Simulated follicles ', profile.follicles, len(profile.follicles))
        # print('#' * 10)
        return prev_profile, prev_day, profile, scans_used_counter

    def get_predictor_follicles_dict(self, predictor_name):
        return [predictor for predictor in self.predictors if predictor.name == predictor_name][0]

    def predict_cycle(self, cycle, num_scans_to_use=None, use_bin_act_only=False, bin_only=False, follicle_only=False,
                      use_mean_lag=False, rf_ensemble=None, rf_preds=None, predictor_weights_dict=None):
        '''
            Predict follicle growth for a patient
            Args:
                cycle: cycle to predict
                models: histostep models
                rf_ensemble: random forest model
                num_scans_to_use: number of scans to use for prediction
            '''
        cycle.histo_preds = {}
        scans_used_counter = 0
        first_day = _reinit_sim_get_first_day(cycle, False)
        prev_profile = None

        for i in range(int(cycle.profiles[first_day].day), 18):
            mean_lag = 1

            day = str(i)
            dayy = str(i + 1)

            if rf_preds is not None:
                # get row for current day
                prediction = rf_preds[rf_preds['future_day'] == int(dayy)]

                bin_predictions = {}
                bins = rf_ensemble.models['bins']
                # bins_str = [str(bins[i]) + '-' + str(bins[i + 1]) for i in range(len(bins) - 1)]
                bins_str = bins
                # print(prediction)
                for bin in bins_str:
                    bin_predictions[bin] = prediction[bin + '_pred'].iloc[0]

            if (int(i) == int(first_day)) or (str(i) in cycle.profiles.keys()):
                # print(day)
                # print(dayy)

                prev_profile, prev_day, profile, scans_used_counter = (
                    self._get_profiles_prev_current(cycle, i, first_day, num_scans_to_use, scans_used_counter,
                                                    prev_profile))
                # print('Num scans to use ', num_scans_to_use)
                # print('Scans used ', scans_used_counter)
                # print('previous day ', prev_profile.day if prev_profile is not None else None)
                # print('prev_day ', prev_day if prev_day is not None else None)
                # print('#' * 10)
            # else use simulated data
            else:
                profile = list(cycle.simulated_profiles.values())[-1]

            if use_mean_lag and int(i) != int(first_day):
                mean_lag = mean_lag * calculate_lag(cycle.simulated_profiles[str(i)].follicles, profile.follicles)
                # print(mean_lag)

            n_sim = 1
            predictor_simulations = {}
            predictor_weights = []
            for predictor in self.predictors:
                if predictor.field_name_in_cycle in cycle.__dict__.keys() or predictor.name == 'overall':
                    if predictor.name == 'overall':
                        field_value = None
                    else:
                        field_value = cycle.__dict__[predictor.field_name_in_cycle]

                    init_dose_flag = False
                    if predictor.name == 'init_dose':
                        if cycle.init_dose_change != 0:
                            init_dose_flag = True
                    # print('Profile before prediction ', profile.follicles)
                    # print('Prev act profile before prediction ', prev_profile.follicles)
                    if field_value is None or field_value == 'None' or init_dose_flag:
                        simulated_follicles = self._predict_profile_using_overall(profile, dayx=day, dayy=dayy,
                                                                                  n_sim=n_sim, is_weighted=False,
                                                                                  use_bin_act_only=use_bin_act_only,
                                                                                  prev_act_day=prev_day,
                                                                                  prev_act_profile=prev_profile,
                                                                                  mean_lag=mean_lag,
                                                                                  bin_only=bin_only,
                                                                                  follicle_only=follicle_only,
                                                                                  return_probas=False)
                        predictor_simulations[predictor.name + '_ov'] = simulated_follicles
                        if predictor_weights_dict is not None:
                            predictor_weights.append(predictor_weights_dict[predictor.name])
                        continue
                    cycle_group = predictor.get_cycle_group(cycle)
                    follicles_stats = predictor.dicts_follicles_stats[cycle_group]
                    bins_stats = predictor.dicts_bins_stats[cycle_group]
                    simulated_follicles = simulate_profiles_follicles(profile, (follicles_stats, bins_stats),
                                                                      dayx=day, dayy=dayy, n_sim=n_sim,
                                                                      use_bin_act_only=use_bin_act_only,
                                                                      bin_only=bin_only,
                                                                      follicle_only=follicle_only,
                                                                      prev_act_day=prev_day,
                                                                      prev_act_profile=prev_profile,
                                                                      mean_lag=mean_lag,
                                                                      return_probas=False)
                    predictor_simulations[predictor.name] = simulated_follicles
                    if predictor_weights_dict is not None:
                        predictor_weights.append(predictor_weights_dict[predictor.name])
                    # print('Length of simulated follicles ', len(simulated_follicles))
                    # print('Length of actual follicles ', len(prev_profile.follicles))
                    # # if len(profile.follicles) == 6:
                    #     print('Actual follicles ', profile.follicles)
                    # print('#' * 10)

            simulated_follicles = self._average_simulations(predictor_simulations, len(profile.follicles),
                                                            weights=predictor_weights if len(
                                                                predictor_weights) > 0 else None)

            if rf_ensemble is not None:
                simulated_follicles = self._combine_histo_rf(cycle, bin_predictions, simulated_follicles, i)

            # if profile is less than 4, add 5 to the profile
            self.pad_with_five(simulated_follicles)

            simulated_profile = Profile(simulated_follicles, cycle.key, i + 1)

            cycle.simulated_profiles[str(i + 1)] = simulated_profile

        return cycle

    @staticmethod
    def _average_simulations(predictor_simulations, profile_len, weights=None):
        # print('Predictor simulations ', predictor_simulations)
        # print('Profile len ', profile_len)
        simulated_follicles = []
        for follicle_idx in range(profile_len):
            follicle_simulations = []
            for predictor_name in predictor_simulations.keys():
                if len(predictor_simulations[predictor_name]) < profile_len:
                    # append with median value
                    median_value = np.median(predictor_simulations[predictor_name])
                    predictor_simulations[predictor_name] = np.append(predictor_simulations[predictor_name],
                                                                      [median_value] * (profile_len - len(
                                                                          predictor_simulations[predictor_name])))
                follicle_simulations.append(predictor_simulations[predictor_name][follicle_idx])
            follicle_simulations = np.array(follicle_simulations)
            # print('length of follicle simulations ', len(follicle_simulations))
            # print('predictor simulations ', len(predictor_simulations))
            # print('length of weights ', len(weights))
            # print('weights ', weights)

            follicle_simulations = np.average(follicle_simulations, axis=0, weights=weights)
            simulated_follicles.append(follicle_simulations)
        return simulated_follicles

    @staticmethod
    def _combine_histo_rf(cycle, bin_predictions, simulated_follicles, i):
        # print('Combining histo and rf')
        # print('i ', i)
        cycle.rf_bins_preds[str(i + 1)] = bin_predictions
        cycle.histo_preds[str(i + 1)] = simulated_follicles
        corrected_simulated_follicles = average_histo_rf_preds_bins(simulated_follicles, bin_predictions)
        return corrected_simulated_follicles

    @staticmethod
    def pad_with_five(follicles):
        while len(follicles) < 4:
            follicles.append(5)
            follicles.sort()
