import json
from json import JSONEncoder

import jsonpickle
import numpy as np

from .Profile import Profile


class Cycle:

    def __init__(self, key, profiles, pad_prof=False, pad_median=False):
        self.key = key
        self.patient_id = None
        self.profiles = profiles
        self.rf_bins_preds = {}
        self.simulated_trigger_days = {'max_12_19': None, 'first_3_more': None}
        self.changes = {}
        self.afc_count = None
        self.age = None
        self.weight = None
        self.os_initial_dose = None
        self.init_dose_change = None
        self.clinic = None
        self.amh = None
        self.suppressant_protocol = None
        self.trigger_day = None
        self.dosages = None
        self.responder_class = None


        self.min_length = self.__get_min_length()

        self.max_length = self.__get_max_length()
        self.follicles_removed = 0
        # self.profiles.sort()
        if pad_prof:
            self.pad_profiles()
        else:
            if pad_median:
                self.pad_profiles_with_median()
            else:
                self.follicles_removed = self.cut_profiles()

    def __get_min_length(self):
        min_length = 100
        for profile in self.profiles.values():
            if len(profile.follicles) < min_length:
                min_length = len(profile.follicles)

        return min_length
    def __get_max_length(self):
        max_length = 0
        for profile in self.profiles.values():
            if len(profile.follicles) > max_length:
                max_length = len(profile.follicles)

        return max_length

    def pad_profiles_with_median(self):
        for key in self.profiles.keys():
            if len(self.profiles[key].follicles) < self.max_length:
                diff = self.max_length - len(self.profiles[key].follicles)
                self.profiles[key].follicles.sort()
                new_follicles = ([round(np.median(self.profiles[key].follicles))] * diff) + self.profiles[key].follicles
                extended_prof = Profile(new_follicles, self.key, self.profiles[key].day,
                                        drug_dosage=self.profiles[key].drug_dosage,
                                        afc_count=self.profiles[key].afc_count)
                self.profiles[key] = extended_prof

    def cut_profiles(self):
        # follicles removed
        follicles_removed = 0
        for key in self.profiles.keys():
            if len(self.profiles[key].follicles) > self.min_length:
                diff = len(self.profiles[key].follicles) - self.min_length
                self.profiles[key].follicles.sort()
                new_follicles = self.profiles[key].follicles[diff:]
                extended_prof = Profile(new_follicles, self.key, self.profiles[key].day,
                                        drug_dosage=self.profiles[key].drug_dosage,
                                        afc_count=self.profiles[key].afc_count)
                self.profiles[key] = extended_prof
                follicles_removed += diff
        return follicles_removed

    def pad_profiles(self):
        for key in self.profiles.keys():
            if len(self.profiles[key].follicles) < self.max_length:
                diff = self.max_length - len(self.profiles[key].follicles)
                self.profiles[key].follicles.sort()
                new_follicles = ([5] * diff) + self.profiles[key].follicles
                extended_prof = Profile(new_follicles, self.key, self.profiles[key].day,
                                        drug_dosage=self.profiles[key].drug_dosage,
                                        afc_count=self.profiles[key].afc_count)
                self.profiles[key] = extended_prof

    def __eq__(self, other):
        return self.key == other.key

    def to_json(self):
        """
        to_json transforms the Model instance into a JSON string
        """
        return jsonpickle.encode(self)




# https://stackoverflow.com/questions/42937612/why-must-a-flask-sessions-value-be-json-serializable
class PatientEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Cycle):
            return obj.to_json()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
