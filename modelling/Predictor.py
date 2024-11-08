from core.stats_gen import generate_stats_bin_based
from core.stats_gen_follicle import generate_stats_follicle_based
from data_prep.data_prep_tfp import split_cycles_by_group_conditions_dict


class Predictor:

    def __init__(self, name, groups_conditions, field_name_in_cycle):
        self.name = name
        self.groups = list(groups_conditions.keys())
        self.groups_conditions = groups_conditions
        self.dicts_follicles_stats = {group: None for group in self.groups}
        self.dicts_bins_stats = {group: None for group in self.groups}
        self.field_name_in_cycle = field_name_in_cycle


    def generate_stats(self, cycles, gaussian_filter, bins, split_by_groups=True, include_2_uniform=False):
        if split_by_groups:
            cycles_with_field = [cycle for cycle in cycles if hasattr(cycle, self.field_name_in_cycle)]
            # print('Cycles with field: ' + str(len(cycles_with_field)))
            cycles_with_field = [cycle for cycle in cycles_with_field if getattr(cycle, self.field_name_in_cycle) is not None]
            # print('Cycles with field: ' + str(len(cycles_with_field)))
            cycles_per_group = split_cycles_by_group_conditions_dict(cycles_with_field, self.groups_conditions)
            # print('Cycles per group: ' + str(len(cycles_per_group)))
        else:
            # If we have overall stats, we don't need to split by groups
            cycles_per_group = {group: cycles for group in self.groups}
        for group in self.groups:
            # print('Generating stats for group: ' + group)
            self.dicts_follicles_stats[group] = generate_stats_follicle_based(train_patients=cycles_per_group[group],
                                                                              gaussian_filter=gaussian_filter,
                                                                              include_2_uniform=include_2_uniform)
            self.dicts_bins_stats[group], _ = generate_stats_bin_based(return_object=True, bins=bins,
                                                                    train_patients=cycles_per_group[group],
                                                                    gaussian_filter=gaussian_filter)

    def get_cycle_group(self, cycle):
        for group in self.groups:
            if self.groups_conditions[group](cycle):
                return group
        print(self.name)
        print(cycle.__getattribute__(self.field_name_in_cycle))
        if self.name == 'protocol':
            print(cycle.suppressant_protocol)
        else:
            raise Exception('Cycle does not belong to any group')