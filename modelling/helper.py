import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import seaborn as sns
from core.models.models_helper import calc_95_ci
from sklearn.metrics import roc_curve

from helpers.results_generation import histosection
from copy import copy


def scores_per_delta(simulated):
    scores_per_delta_time = {}
    # Check scores per delta time
    for cycle in simulated:
        scan_list = list(cycle.profiles.keys())
        for idx, scan in enumerate(scan_list[1:]):
            if int(scan) > 18 or int(scan) <= 5:
                continue
            delta = int(scan) - int(scan_list[idx])
            if int(delta) not in scores_per_delta_time:
                scores_per_delta_time[delta] = []

            act_follicles = cycle.profiles[scan].follicles
            sim_follicles = cycle.simulated_profiles[scan].follicles
            scores_per_delta_time[delta].append(histosection(act_follicles, sim_follicles))

    scores_per_delta_time_mean = {}
    for delta in scores_per_delta_time:
        scores_per_delta_time_mean[delta] = np.mean(scores_per_delta_time[delta])

    scores_per_delta_time_mean = dict(sorted(scores_per_delta_time_mean.items()))

    return scores_per_delta_time, scores_per_delta_time_mean


def cycles_to_df_for_binpred(cycles, bin_size=5, bins=None, num_scans_to_use=None):
    '''
    Convert cycles to dataframe for bin prediction
    :param cycles: list of cycles
    :param bin_size: size of bin
    :param bins: list of bins edges to use
    '''
    df = pd.DataFrame()
    for cycle in tqdm(cycles):
        afc = cycle.afc_count
        amh = cycle.amh
        age = cycle.age
        weight = cycle.weight
        id = cycle.key
        clinic = cycle.clinic
        for idx, profile in enumerate(list(cycle.profiles.values())):
            if (num_scans_to_use is not None) and idx > num_scans_to_use:
                break
            follicles = profile.follicles
            follicles_features_dict_binned = {}
            if bins:
                bins_str = [str(bins[i]) + '-' + str(bins[i + 1]) for i in range(len(bins) - 1)]
                follicles_features_dict_binned = dict(zip(bins_str, [0] * len(bins)))
                for bin_start in range(len(bins) - 1):
                    follicles_features_dict_binned[bins_str[bin_start]] = len(
                        [follicle for follicle in follicles if bins[bin_start] <= follicle < bins[bin_start + 1]])
            else:
                for bin_start in range(5, 26, bin_size):
                    follicles_features_dict_binned[str(bin_start) + '-' + str(bin_start + bin_size)] = len(
                        [follicle for follicle in follicles if bin_start <= follicle < bin_start + bin_size])

            follicles_df = pd.DataFrame(follicles_features_dict_binned, index=[0])
            follicles_df['id'] = id
            follicles_df['day'] = profile.day
            follicles_df['afc'] = afc
            follicles_df['amh'] = amh
            follicles_df['age'] = age
            follicles_df['weight'] = weight
            follicles_df['scan_num'] = idx + 1
            follicles_df['clinic'] = clinic

            df = pd.concat([df, follicles_df], axis=0, join='outer', ignore_index=True)

    return df


def df_to_long_xy(df, bin_size=5, bins=None, drop_id=True, num_scans_to_use=None):
    '''
    Convert dataframe to long format for bin prediction
    :param df: 
    :param bin_size: 
    :param bins: 
    :param drop_id: 
    :param num_scans_to_use: 
    :return: 
    '''
    X = pd.DataFrame()
    ys = {}
    bin_names = []
    if bins:
        bin_names = [str(bins[i]) + '-' + str(bins[i + 1]) for i in range(len(bins) - 1)]
    else:
        for bin_start in range(5, 25, bin_size):
            bin_names.append(str(bin_start) + '-' + str(bin_start + bin_size))
    for bin_name in bin_names:
        ys[bin_name] = []
    for id in tqdm(df.id.unique()):
        df_id = df[df.id == id]
        df_id = df_id.sort_values(by=['day'])
        df_id = df_id.reset_index(drop=True)
        for i in range(len(df_id)):
            if i == len(df_id) - 1 and i != 0:
                break
            if i > 0:
                if (num_scans_to_use is not None) and i > num_scans_to_use:
                    break
                rename_dict = dict(zip(bin_names, ['prev_' + bin_name for bin_name in bin_names]))
                rename_dict['day'] = 'prev_day'
                prev_profile = pd.DataFrame(df_id.iloc[i - 1][bin_names + ['day']]).transpose()
                prev_profile = prev_profile.rename(columns=rename_dict)
                prev_profile['i'] = i
                prev_profile = prev_profile.set_index('i')

            else:
                prev_profile = pd.DataFrame(
                    dict(zip(['prev_' + bin_name for bin_name in bin_names], [0] * len(bin_names))), index=[0])
                prev_profile['prev_day'] = 0
            feature = pd.DataFrame(df_id.iloc[i]).transpose().join(prev_profile)
            if num_scans_to_use != 0:
                feature['delta_future_day'] = df_id.iloc[i + 1]['day'] - df_id.iloc[i]['day']
            else:
                feature['delta_future_day'] = df_id.iloc[i]['day'] + 1
            feature['delta_prev_day'] = df_id.iloc[i]['day'] - df_id.iloc[i - 1]['day'] if i > 0 else 0
            for bin_name in bin_names:
                feature['delta_' + bin_name] = feature[bin_name] - feature['prev_' + bin_name] if i > 0 else 0

            if drop_id:
                feature = feature.drop(columns=['id', 'prev_day'])
            else:
                feature = feature.drop(columns=['prev_day'])

            X = pd.concat([X, feature], ignore_index=True)

            # Convert all columns to numeric apart from id
            X[X.columns.difference(['id', 'clinic'])] = X[X.columns.difference(['id', 'clinic'])].apply(pd.to_numeric)

            # Label encode clinic
            X['clinic'] = X['clinic'].astype('category')
            X['clinic'] = X['clinic'].cat.codes

            if num_scans_to_use != 0:
                for bin_name in bin_names:
                    ys[bin_name].append(df_id.iloc[i + 1][bin_name])
            # print('X shape: {}'.format(X.shape))
            # print('ys shape: {}'.format(len(ys[bin_name])))

    return X, ys


def add_rows_for_pred(data_for_rf):
    '''
    Add rows in between scans with future days adjusted
    '''
    new_df = pd.DataFrame(columns=data_for_rf.columns)
    # For each patient add rows in between scans with future days adjusted
    for id in data_for_rf.id.unique():
        df_id = data_for_rf[data_for_rf.id == id]
        df_id = df_id.sort_values(by=['day'])
        df_id = df_id.reset_index(drop=True)

        # last_day = df_id.loc[1, 'day']
        len_df_id = len(df_id)
        for i in range(len_df_id):

            if i == len_df_id - 1:
                future_scan_day = 19
            else:
                future_scan_day = df_id.loc[i + 1, 'day']
            df_id['future_day'] = df_id['day'] + df_id['delta_future_day']
            for fut_day in range(df_id.loc[i, 'day'] + 1, future_scan_day):
                if fut_day in df_id.future_day.values:
                    continue
                # copy current row and change future day and append to df_id
                new_row = df_id.loc[i].copy()
                new_row['delta_future_day'] = fut_day - df_id.loc[i, 'day']
                df_id = pd.concat([df_id, new_row.to_frame().T], ignore_index=True)
                # print(df_id)
            df_id = df_id.drop(columns=['future_day'])
        new_df = pd.concat([new_df, df_id], ignore_index=True)
    return new_df


def plot_growth_rates_follicles(models, group, ax=None, mapping_dict=None, gaussian_filter=False):
    follicles = []
    diff_absolute_mean = []
    all_diffs = []
    all_diffs_tuples = []
    for key in models[group][0].keys():
        if isinstance(key, int):
            if key == 5 or key == 26:
                continue
            follicles.append(key)
            diff_absolute_mean.append(models[group][0][key]['diff_absolute_mean'])
            all_diffs.extend(models[group][0][key]['absolute_growth'])
            all_diffs_tuples.extend([(key, growth) for growth in models[group][0][key]['absolute_growth']])
    if ax:
        sns.barplot(x=follicles, y=diff_absolute_mean, ax=ax)
        ax.set_xlabel('Follicle size (mm)')
        ax.set_ylabel('Mean follicle growth rate (mm/day)')
        ax.set_title('Mean follicle growth rate for group ' + str(mapping_dict[group]))
        ax.set_ylim(0, 3.5)
        # add text with mean of all diffs
        ax.text(0.5, 0.9, 'Mean growth per day: {:.2f}'.format(np.mean(all_diffs)), horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

        # ax.set_xlim(0, 26)
        return
    # sns.barplot(x=follicles, y=diff_absolute_mean)
    print(diff_absolute_mean)
    follicle_diff_dict = dict(zip(follicles, diff_absolute_mean))
    # sort by follicle size
    follicle_diff_dict = dict(sorted(follicle_diff_dict.items()))
    if gaussian_filter:
        print(follicle_diff_dict.values())
        diff_absolute_mean = gaussian_filter1d(list(follicle_diff_dict.values()), sigma=1)
        print(diff_absolute_mean)
        # plt.plot([x for x in list(follicle_diff_dict.keys())], diff_absolute_mean, color='red')
    # create a dataframe from all diffs dict
    print(all_diffs_tuples)
    follicle_diff_df = pd.DataFrame(all_diffs_tuples, columns=['follicle_size', 'growth_rates'])
    print(follicle_diff_df)

    sns.lineplot(x=follicle_diff_df['follicle_size'], y=follicle_diff_df['growth_rates'], color='red')
    # set xlim
    plt.xlim(6, 26)
    # set xticks
    xticks = []
    for i in range(6, 26):
        xticks.append(str(i) + '\n{}'.format(len(models[group][0][i]['absolute_growth'])))
    plt.xticks(np.arange(6, 26, 1), xticks)
    # plt.xticks(np.arange(6, 26, 1))
    plt.xlabel('Follicle size')
    plt.ylabel('Mean follicle growth rate')
    plt.title('Mean follicle growth rate for group ' + str(mapping_dict[group]))
    # plt.show()


def follicle_dict_to_tuple_list(follicle_dict):
    all_diffs_tuples = []
    for follicle in follicle_dict.keys():
        if not isinstance(follicle, int):
            continue
        # print('Adding follicle {}'.format(follicle))
        # print(len(follicle_dict[follicle]['absolute_growth']))
        all_diffs_tuples.extend([(follicle, growth) for growth in follicle_dict[follicle]['absolute_growth']])
    return all_diffs_tuples


def days_dict_to_tuple_list(days_dict, include_2_days=False):
    all_diffs_tuples = []
    for dayx in days_dict.keys():
        if not isinstance(dayx, int):
            continue
        for dayy in days_dict[dayx].keys():
            if not isinstance(dayy, int):
                continue
            if dayx != dayy - 1:
                # print(dayx, dayy)
                if dayx == dayy - 2:
                    # print(dayx, dayy)
                    if include_2_days:
                        all_diffs_tuples.extend(
                            [(dayx, growth / 2) for growth in days_dict[dayx][dayy]['absolute_growth']])
                    continue
                continue
            all_diffs_tuples.extend([(dayx, growth) for growth in days_dict[dayx][dayy]['absolute_growth']])
    return all_diffs_tuples

def days_dict_to_tuple_list_mod(days_dict, include_2_days=False):
    all_diffs_tuples = []
    for dayx in days_dict.keys():
        if not isinstance(dayx, int):
            continue
        for dayy in days_dict[dayx].keys():
            if not isinstance(dayy, int):
                continue
            if dayx == dayy - 1:
                all_diffs_tuples.extend([(dayx, growth) for growth in days_dict[dayx][dayy]['absolute_growth']])
            if dayx == dayy - 2:
                all_diffs_tuples.extend(
                    [(dayx, growth / 2) for growth in days_dict[dayx][dayy]['absolute_growth']])
                # print(dayx, dayy)

    return all_diffs_tuples

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def get_percentiles(data, confidence=0.95):
    p_low = (1 - confidence)
    p_high = confidence
    return np.percentile(data, p_low * 100), np.percentile(data, p_high * 100)


def plot_days_diffs_newplot(days_dict, ax=None, group=None, title_group=None, mean=None, include_2_days=False,
                            follicle_dict=False, follicle_per_day=False):
    if follicle_dict:
        all_diffs_tuples = follicle_dict_to_tuple_list(days_dict)
    else:
        all_diffs_tuples = days_dict_to_tuple_list(days_dict, include_2_days=include_2_days)
    # all_diffs_tuples = follicle_dict_to_tuple_list(days_dict)
    # print(len(all_diffs_tuples))
    # all_diffs_tuples = [(tup[0], tup[1] * np.pi * 3 / 6) for tup in all_diffs_tuples]
    follicle_diff_df = pd.DataFrame(all_diffs_tuples, columns=['day', 'growth_rates'])
    if follicle_dict and not follicle_per_day:
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] >= 6]
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] <= 23]
    else:
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] >= 4]
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] <= 14]

    # add y ticks and limits
    ax.set_ylim(0, 3)
    ax.set_yticks(np.arange(0, 3, .5))

    # add horizontal lines and ticks on the right
    ax2 = ax.twinx()

    ax2.set_ylim(ax.get_ylim())

    ax2.set_yticks(np.arange(1, 2.1, .2))

    for i in range(11):
        ax2.axhline(y=1.0 + i * 0.1, color='black', linestyle='--', lw=1, alpha=0.1)

    # Add linear regression line throggh the data
    sns.regplot(x=follicle_diff_df['day'], y=follicle_diff_df['growth_rates'], color='blue', ax=ax, scatter=False,
                label='Linear regression', line_kws={'lw': 1})

    # add error bars
    sns.lineplot(x=follicle_diff_df['day'], y=follicle_diff_df['growth_rates'], color='red', ax=ax, err_style='bars',
                 errorbar=('sd', 1),
                 err_kws={'capsize': 5, 'capthick': 2, 'ecolor': 'black',
                          'label': '1 Standard deviation'},
                 legend='full', markers=True, marker='o', markersize=10,
                 linestyle='')

    # plot points on the means of days
    if follicle_dict and not follicle_per_day:
        for day in range(6, 25):
            mean_day = follicle_diff_df[follicle_diff_df['day'] == day]['growth_rates'].mean()
            ax.scatter(day, mean_day, color='red', s=100, zorder=10)
    else:
        for day in range(4, 15):
            mean_day = follicle_diff_df[follicle_diff_df['day'] == day]['growth_rates'].mean()
            ax.scatter(day, mean_day, color='red', s=100, zorder=10)

    # add text box to the upper right with lin reg slope and f test p value
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(follicle_diff_df['day'],
                                                                         follicle_diff_df['growth_rates'])
    ax.text(0.95, 0.90,
            'Slope: {:.2f}\nWald test p-value: {}'.format(slope, round(p_value, 2) if p_value >= 0.05 else '<0.05'),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=1', ), fontsize=12, fontweight='bold')

    # print(' All diffs: {}'.format(len(all_diffs_tuples)))

    # plot legend
    # plt.legend(loc="upper right", fontsize=20, prop={'size': 20})
    # set xticks
    xticks = []
    if follicle_dict and not follicle_per_day:
        ax.set_xlim(6, 23)
        total = 0
        for i in range(6, 24):
            # print(i)
            # print(len([x for x in all_diffs_tuples if x[0] == i]))
            # print('---')
            total += len([x for x in all_diffs_tuples if x[0] == i])
            xticks.append(str(i) + '\n{}'.format(len([x for x in all_diffs_tuples if x[0] == i])))
        # print('Total: {}'.format(total))
        # print('5s : {}'.format(len([x for x in all_diffs_tuples if x[0] == 5])))
        ax.set_xticks(np.arange(6, 24, 1), xticks)
    else:
        ax.set_xlim(4, 14)
        for i in range(4, 15):
            xticks.append(str(i) + '\n{}'.format(len([x for x in all_diffs_tuples if x[0] == i])))
        ax.set_xticks(np.arange(4, 15, 1), xticks)

    if mean is not None:
        ax.set_xlabel(('Follicle size (mm)' if follicle_dict else
                       'Day of ovarian stimulation'), fontweight='bold', fontsize=16)
    else:
        ax.set_xlabel('Follicle size (mm)' if (follicle_dict and not follicle_per_day) else 'Day of ovarian stimulation',
                      fontweight='bold', fontsize=16)
    ax.set_ylabel('Mean follicle growth rate (mm/day)', fontweight='bold', fontsize=16)
    # change label font size
    # ax2.tick_params(axis='both', which='major', labelsize=14)
    # ax2.tick_params(axis='both', which='major', labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels() + ax2.get_xticklabels()
    [label.set_fontweight('bold') for label in labels]

    # add padding to the x lim of 0.5 from the left and right
    ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)

    # ax.set_title('Mean follicle growth rate. {} {}'.format(title_group, group))
    ax.grid(False)
    ax2.grid(False)

    # Create dataframe with lin reg prediction for x values
    if follicle_dict and not follicle_per_day:
        x = np.arange(6, 24)
    else:
        x = np.arange(4, 15)
    y = slope * x + intercept
    x_axis_name = 'Follicle_size' if (follicle_dict and not follicle_per_day) else 'Day'
    df_lin_reg = pd.DataFrame({x_axis_name: follicle_diff_df['day'], 'growth_rates': follicle_diff_df['growth_rates']})
    df_lin_reg['group'] = [group] * df_lin_reg.shape[0]

    return df_lin_reg


def plot_days_diffs(days_dict, ax=None, group=None, title_group=None, mean=None, include_2_days=False,
                    follicle_dict=False, follicle_per_day=False):
    if follicle_dict:
        all_diffs_tuples = follicle_dict_to_tuple_list(days_dict)
    else:
        all_diffs_tuples = days_dict_to_tuple_list(days_dict, include_2_days=include_2_days)
    # all_diffs_tuples = follicle_dict_to_tuple_list(days_dict)
    # print(len(all_diffs_tuples))
    # all_diffs_tuples = [(tup[0], tup[1] * np.pi * 3 / 6) for tup in all_diffs_tuples]
    follicle_diff_df = pd.DataFrame(all_diffs_tuples, columns=['day', 'growth_rates'])
    if follicle_dict and not follicle_per_day:
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] >= 6]
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] <= 25]
    else:
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] >= 4]
        follicle_diff_df = follicle_diff_df[follicle_diff_df['day'] <= 15]

    # do log scale
    # follicle_diff_df['growth_rates'] = np.log(follicle_diff_df['growth_rates'])

    # plt.figure(figsize=(10, 5))
    if follicle_dict and not follicle_per_day:
        ax.set_ylim(0.5, 2.5)
        ax.set_yticks(np.arange(.5, 2.6, .1))
    else:
        ax.set_ylim(.5, 2)
        ax.set_yticks(np.arange(.5, 2.1, .1))

    ax2 = ax.twinx()

    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    if follicle_dict and not follicle_per_day:
        ax2.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                        1.7, 1.8, 1.9, 2.0])

    ax.axhline(y=1.0, color='black', linestyle='--', lw=1, alpha=0.1)
    ax.axhline(y=1.1, color='black', linestyle='--', lw=1, alpha=0.1)
    ax.axhline(y=1.2, color='black', linestyle='--', lw=1, alpha=0.1)
    ax.axhline(y=1.3, color='black', linestyle='--', lw=1, alpha=0.1)
    ax.axhline(y=1.4, color='black', linestyle='--', lw=1, alpha=0.1)
    ax.axhline(y=1.5, color='black', linestyle='--', lw=1, alpha=0.1)
    if follicle_dict and not follicle_per_day:
        ax.axhline(y=1.6, color='black', linestyle='--', lw=1, alpha=0.1)
        ax.axhline(y=1.7, color='black', linestyle='--', lw=1, alpha=0.1)
        ax.axhline(y=1.8, color='black', linestyle='--', lw=1, alpha=0.1)
        ax.axhline(y=1.9, color='black', linestyle='--', lw=1, alpha=0.1)
        ax.axhline(y=2.0, color='black', linestyle='--', lw=1, alpha=0.1)

    # Add linear regression line throggh the data
    sns.regplot(x=follicle_diff_df['day'], y=follicle_diff_df['growth_rates'], color='blue', ax=ax, scatter=False,
                label='Linear regression', line_kws={'lw': 1})

    # sns.lineplot(x=follicle_diff_df['day'], y=follicle_diff_df['growth_rates'], color='red', ax=ax, errorbar=('ci', 95),
    #              err_style='bars', legend='full', err_kws={'label': '95% CI for the mean'})
    # smooth the data
    # follicle_diff_df['growth_rates'] = gaussian_filter1d(follicle_diff_df['growth_rates'], sigma=2)

    sns.lineplot(x=follicle_diff_df['day'], y=follicle_diff_df['growth_rates'], color='red', ax=ax, err_style='bars',
                 errorbar=('sd', 1),
                 legend='full', err_kws={'label': '1 Standard deviation'}, markers=True, marker='o', markersize=5,
                 linestyle='')

    print(' All diffs: {}'.format(len(all_diffs_tuples)))

    # plot legend
    # plt.legend(loc="upper right", fontsize=20, prop={'size': 20})
    # set xticks
    xticks = []
    if follicle_dict and not follicle_per_day:
        ax.set_xlim(6, 25)
        total = 0
        for i in range(6, 26):
            print(i)
            print(len([x for x in all_diffs_tuples if x[0] == i]))
            print('---')
            total += len([x for x in all_diffs_tuples if x[0] == i])
            xticks.append(str(i) + '\n{}'.format(len([x for x in all_diffs_tuples if x[0] == i])))
        print('Total: {}'.format(total))
        print('5s : {}'.format(len([x for x in all_diffs_tuples if x[0] == 5])))
        ax.set_xticks(np.arange(6, 26, 1), xticks)
    else:
        ax.set_xlim(4, 15)
        for i in range(4, 16):
            xticks.append(str(i) + '\n{}'.format(len([x for x in all_diffs_tuples if x[0] == i])))
        ax.set_xticks(np.arange(4, 16, 1), xticks)

    if mean is not None:
        ax.set_xlabel(('Follicle size' if follicle_dict else
                       'Day of ovarian stimulation'), fontweight='bold', fontsize=16)
    else:
        ax.set_xlabel('Follicle size' if (follicle_dict and not follicle_per_day) else 'Day of ovarian stimulation',
                      fontweight='bold', fontsize=16)
    ax.set_ylabel('Mean follicle growth rate', fontweight='bold', fontsize=16)
    # change label font size
    # ax2.tick_params(axis='both', which='major', labelsize=14)
    # ax2.tick_params(axis='both', which='major', labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels() + ax2.get_xticklabels()
    [label.set_fontweight('bold') for label in labels]

    # ax.set_title('Mean follicle growth rate. {} {}'.format(title_group, group))
    ax.grid(False)
    ax2.grid(False)
    # plt.show()


def plot_growth_rates_per_day(dict_stats, gaussian_filter=False):
    days = []
    diff_absolute_mean = []
    all_diffs = []
    all_diffs_tuples = []
    for key in dict_stats.keys():
        if isinstance(key, int):
            if key == 5 or key == 26:
                continue
            days.append(key)
            diff_absolute_mean.append(models[group][0][key]['diff_absolute_mean'])
            all_diffs.extend(models[group][0][key]['absolute_growth'])
            all_diffs_tuples.extend([(key, growth) for growth in models[group][0][key]['absolute_growth']])


# def plot_growth_rates_per_day(models, group, ax=None, mapping_dict=None, gaussian_filter=False):

def plot_roc_curve(y_pred_proba, y_test, title, save_path=None):
    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    plt.figure(figsize=(6, 6))
    ci_low, ci_upp = calc_95_ci(y_pred_proba[:, 1], np.array(y_test))  # [1]
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f [95%% CI %0.2f-%0.2f])' % (roc_auc, ci_low, ci_upp))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()


def calculate_lag(follicles_act, follicles_pred):
    mean_lag = np.mean(follicles_act) / np.mean(follicles_pred)
    return mean_lag


def calculate_lag_profile(profile_act, profile_pred, profile_prev):
    '''
    Calculate lag between actual and predicted profile relative to the size of the follicles
    '''
    lag_profile = []
    for idx in range(len(profile_act.follicles)):
        print(profile_prev.follicles[idx], profile_act.follicles[idx], profile_pred.follicles[idx])
        # print(profile_pred.follicles[idx] - profile_act.follicles[idx])

        difference_act = profile_act.follicles[idx] - profile_prev.follicles[idx]
        if difference_act == 0:
            difference_act = 1
        difference_pred = profile_pred.follicles[idx] - profile_prev.follicles[idx]
        if difference_pred == 0:
            difference_pred = 1
        lag_profile.append(difference_act / difference_pred)
        print(lag_profile[-1])
        print('---')
    return lag_profile


def calculate_bins_lags(profile_act, profile_pred):
    '''
    Calculate lag between actual and predicted profile relative to the bins of the follicles
    '''
    lag_profile = []
    bottom_bin_lag = profile_pred.bins['bottom'].mean / profile_act.bins['bottom'].mean
    lower_bin_lag = profile_pred.bins['lower'].mean / profile_act.bins['lower'].mean
    upper_bin_lag = profile_pred.bins['upper'].mean / profile_act.bins['upper'].mean
    top_bin_lag = profile_pred.bins['top'].mean / profile_act.bins['top'].mean
    lag_profile.append([bottom_bin_lag] * profile_act.bins['bottom'].size)
    lag_profile.append([lower_bin_lag] * profile_act.bins['lower'].size)
    lag_profile.append([upper_bin_lag] * profile_act.bins['upper'].size)
    lag_profile.append([top_bin_lag] * profile_act.bins['top'].size)
    return lag_profile


def calc_scores_per_delta_time(simulated, scan_num=None, last_used_scan=None):
    scores_per_delta_time = {}
    # Check scores per delta time
    for cycle in simulated:
        scan_list = list(cycle.profiles.keys())
        if scan_num:
            if scan_num == -1:
                scan = scan_list[-1]
            else:
                scan = scan_list[scan_num - 1]
            if int(scan) > 18 or int(scan) <= 5:
                continue
            delta = int(scan) - int(scan_list[last_used_scan - 1])
            if int(delta) not in scores_per_delta_time:
                scores_per_delta_time[delta] = []
            act_follicles = cycle.profiles[scan].follicles
            sim_follicles = cycle.simulated_profiles[scan].follicles
            scores_per_delta_time[delta].append(histosection(act_follicles, sim_follicles))
            continue
        for idx, scan in enumerate(scan_list[1:]):
            if int(scan) > 18 or int(scan) <= 5:
                continue
            delta = int(scan) - int(scan_list[idx])
            if int(delta) not in scores_per_delta_time:
                scores_per_delta_time[delta] = []

            act_follicles = cycle.profiles[scan].follicles
            sim_follicles = cycle.simulated_profiles[scan].follicles
            scores_per_delta_time[delta].append(histosection(act_follicles, sim_follicles))
    return scores_per_delta_time


def pred_bin_to_follicles(dist_histo, rf_pred_num, bin_start, bin_end):
    # dist_histo: distribution of follicles in a bin
    # rf_pred_num: number of follicles predicted in the bin
    # returns: list of follicles in the bin
    # %%
    # normalise dist_histo
    # check if sum is 0
    if np.sum(dist_histo) == 0:
        return []

    # normalise
    dist_histo = dist_histo / np.sum(dist_histo)

    # multiply by rf_pred_num
    dist_histo = dist_histo * rf_pred_num

    corrected_follicles = []

    # print('#######################')
    # print(bin_start)
    # print(dist_histo)
    # print(rf_pred_num)

    for i in range(len(dist_histo)):
        if dist_histo[i] > 1:
            for j in range(int(dist_histo[i])):
                corrected_follicles.append(i + bin_start)
            dist_histo[i] = dist_histo[i] - int(dist_histo[i])
    if len(corrected_follicles) < rf_pred_num:
        # if there are still follicles to add, add the most likely ones
        diff = rf_pred_num - len(corrected_follicles)
        for i in range(diff):
            corrected_follicles.append(np.argmax(dist_histo) + bin_start)
            dist_histo[np.argmax(dist_histo)] = 0

    # print(corrected_follicles)
    # print('#######################')
    # return the values of the largest indices
    return sorted(corrected_follicles)


def average_histo_rf_preds_bins(histo_follicles, rf_pred_bins, n_intervals=20):
    # histo_follicles: list of follicles in each bin
    # rf_pred_bins: dict of predicted follicles in each bin
    # corr_param: parameter to control how much to correct towards target distribution
    # returns: list of follicles in each bin after correction
    # %%
    orig_x = np.linspace(5, 25, n_intervals)
    orig_dist = np.zeros(n_intervals)
    for val in histo_follicles:
        # find the index of the value closest to val but not necessarily less than val
        index = (np.abs(orig_x - val)).argmin()
        orig_dist[index] += 1

    # Smooth the distribution
    orig_dist = gaussian_filter1d(orig_dist, sigma=1)
    # print(orig_dist)
    histo_dist = orig_dist.copy()

    corrected_follicles = []
    for bin in rf_pred_bins:
        rf_pred_num = round(rf_pred_bins[bin])
        bin_dist_histo = histo_dist[(int(bin.split('-')[0]) - 5):(int(bin.split('-')[1]) - 5)]
        # print(pred_bin_to_follicles(bin_dist_histo, rf_pred_num, int(bin.split('-')[0]), int(bin.split('-')[1])))
        corrected_follicles.extend(
            pred_bin_to_follicles(bin_dist_histo, rf_pred_num, int(bin.split('-')[0]), int(bin.split('-')[1])))
    # average back to folls
    return sorted(corrected_follicles)


def average_to_profile(average, num_follicles):
    average_25s = average
    # for i in range(0, 20//5):
    #     average_25s.append(sum(average[i*5:i*5+5]))
    # Normalise average_25s
    # print(average_25s)
    average_25s = average_25s / np.sum(average_25s)
    # Multiply by num_follicles
    average_25s = average_25s * (num_follicles + 1)
    # print(average_25s)
    # round to nearest integer
    average_25s_rounded = np.round(average_25s)
    # get indeces of num_follicles largest values
    # print(average_25s_rounded)
    corrected_follicles = []
    for i in range(len(average_25s_rounded)):
        for j in range(int(average_25s_rounded[i])):
            corrected_follicles.append(i + 5)
    # print(corrected_follicles)
    # largest_n = np.argpartition(average_25s, -num_follicles)[-num_follicles:]
    while len(corrected_follicles) < num_follicles:
        # Append next largest value
        corrected_follicles.append(np.argmax(average_25s) + 5)
    corrected_follicles = np.sort(corrected_follicles)
    # return the values of the largest indices
    return corrected_follicles


def average_histo_rf_preds(histo_follicles, rf_pred_bins, corr_param=1, n_intervals=20):
    # histo_follicles: list of follicles in each bin
    # rf_pred_bins: dict of predicted follicles in each bin
    # corr_param: parameter to control how much to correct towards target distribution
    # returns: list of follicles in each bin after correction
    # %%
    orig_x = np.linspace(5, 25, n_intervals)
    orig_dist = np.zeros(n_intervals)
    for val in histo_follicles:
        # find the index of the value closest to val
        index = (np.abs(orig_x - val)).argmin()
        orig_dist[index] += 1

    orig_dist = gaussian_filter1d(orig_dist, sigma=1)
    # print(orig_dist)

    target_x = np.linspace(5, 25, n_intervals)
    target_dist = np.zeros(n_intervals)
    for bin in rf_pred_bins:
        bin_start = int(bin.split('-')[0])
        bin_end = int(bin.split('-')[1])
        bin_median = round((bin_start + bin_end) / 2)

        # find the index of the value closest to bin_median
        index = (np.abs(target_x - bin_median)).argmin()
        target_dist[index] = rf_pred_bins[bin]
    # print(target_dist)

    target_dist = gaussian_filter1d(target_dist, sigma=1)
    # print(target_dist)

    average = orig_dist * (1 - corr_param) + target_dist * corr_param

    # average back to folls
    average_folls = average_to_profile(average, len(histo_follicles))
    # print(average_folls)
    return average_folls


#####################
# Stats #
#####################
def per_follicle_diffs_cycles(simulated_cycles, only_last_scan=False):
    per_follicle_diffs = []

    for cycle in simulated_cycles:
        if only_last_scan:
            profile_key = list(cycle.profiles.keys())[-1]
            if int(profile_key) > 18:
                continue
            simulated_follicles = copy(cycle.simulated_profiles[profile_key].follicles)
            real_follicles = copy(cycle.profiles[profile_key].follicles)
            if len(simulated_follicles) < len(real_follicles):
                simulated_follicles.extend(
                    [round(np.median(simulated_follicles))] * (len(real_follicles) - len(simulated_follicles)))
                simulated_follicles.sort()
            elif len(simulated_follicles) > len(real_follicles):
                simulated_follicles = simulated_follicles[len(simulated_follicles) - len(real_follicles):]
            for idx, follicle in enumerate(real_follicles):
                per_follicle_diffs.append(follicle - simulated_follicles[idx])
        else:
            for profile_key in list(cycle.profiles.keys())[1:]:
                if int(profile_key) > 18:
                    continue
                simulated_follicles = copy(cycle.simulated_profiles[profile_key].follicles)
                real_follicles = copy(cycle.profiles[profile_key].follicles)
                if len(simulated_follicles) < len(real_follicles):
                    simulated_follicles.extend(
                        [round(np.median(simulated_follicles))] * (len(real_follicles) - len(simulated_follicles)))
                    simulated_follicles.sort()
                elif len(simulated_follicles) > len(real_follicles):
                    simulated_follicles = simulated_follicles[len(simulated_follicles) - len(real_follicles):]
                for idx, follicle in enumerate(real_follicles):
                    per_follicle_diffs.append(follicle - simulated_follicles[idx])

    return per_follicle_diffs


def follicle_more_than_n_diff(simulated_cycles, n):
    follicle_more_than_n_diff = []
    for cycle in simulated_cycles:
        # for profile_key in list(cycle.profiles.keys())[1:]:
        #     if int(profile_key) > 18:
        #         continue
        #     simulated_follicles = copy(cycle.simulated_profiles[profile_key].follicles)
        #     real_follicles = copy(cycle.profiles[profile_key].follicles)
        #     sim_follicles_more_than_n = [follicle for follicle in simulated_follicles if follicle > n]
        #     real_follicles_more_than_n = [follicle for follicle in real_follicles if follicle > n]
        #     if len(real_follicles_more_than_n) == 0 and len(sim_follicles_more_than_n) == 0:
        #         continue
        #     follicle_more_than_n_diff.append(len(sim_follicles_more_than_n) - len(real_follicles_more_than_n))
        profile_key = list(cycle.profiles.keys())[-1]
        if int(profile_key) > 18:
            continue
        simulated_follicles = copy(cycle.simulated_profiles[profile_key].follicles)
        real_follicles = copy(cycle.profiles[profile_key].follicles)
        sim_follicles_more_than_n = [follicle for follicle in simulated_follicles if follicle > n]
        real_follicles_more_than_n = [follicle for follicle in real_follicles if follicle > n]
        if len(real_follicles_more_than_n) == 0 and len(sim_follicles_more_than_n) == 0:
            continue
        follicle_more_than_n_diff.append(len(sim_follicles_more_than_n) - len(real_follicles_more_than_n))
    return follicle_more_than_n_diff


def follicles_more_than_n(simulated_cycles, n):
    diffs_more_than_n = []
    hit_more_than_n = 0
    missed_more_than_n = 0
    diffs_hit_more_than_n = []
    for cycle in simulated_cycles:
        profile_key = list(cycle.profiles.keys())[-1]
        if int(profile_key) > 18:
            continue
        simulated_follicles = copy(cycle.simulated_profiles[profile_key].follicles)
        real_follicles = copy(cycle.profiles[profile_key].follicles)
        sim_follicles_more_than_n = [follicle for follicle in simulated_follicles if follicle >= n]
        real_follicles_more_than_n = [follicle for follicle in real_follicles if follicle >= n]
        if len(real_follicles_more_than_n) == 0 and len(sim_follicles_more_than_n) == 0:
            continue
        if len(real_follicles_more_than_n) >= 3:
            if len(sim_follicles_more_than_n) >= 3:
                hit_more_than_n += 1
                diffs_hit_more_than_n.append(len(sim_follicles_more_than_n) - len(real_follicles_more_than_n))
            else:
                missed_more_than_n += 1
            diffs_more_than_n.append(len(sim_follicles_more_than_n) - len(real_follicles_more_than_n))
    return diffs_more_than_n, hit_more_than_n, missed_more_than_n, diffs_hit_more_than_n


def match_first_n_more_than_x(simulated_cycles, n, x):
    predicted_correctly = 0
    total = 0
    flag_not_in = False
    for cycle in simulated_cycles:
        first_sim_day_2more_17 = None
        for profile_key in list(cycle.simulated_profiles.keys()):
            if int(profile_key) > 18 or first_sim_day_2more_17 is not None:
                break
            num_more_than_17 = len(
                [follicle for follicle in cycle.simulated_profiles[profile_key].follicles if follicle >= x])
            if num_more_than_17 >= n:
                first_sim_day_2more_17 = int(profile_key)
                break
        # if first_sim_day_2more_17 is None:
        #     continue

        # check if real cycle has scan on the same day or the day after

        if str(first_sim_day_2more_17) not in cycle.profiles.keys():
            flag_not_in = True

        first_day_2more_17 = None
        for profile_key in list(cycle.profiles.keys()):
            if int(profile_key) > 18 or first_day_2more_17 is not None:
                break
            num_more_than_17 = len([follicle for follicle in cycle.profiles[profile_key].follicles if follicle >= x])
            if num_more_than_17 >= n:
                first_day_2more_17 = int(profile_key)
                break

        if first_day_2more_17 is None and first_sim_day_2more_17 is None:
            predicted_correctly += 1
            total += 1
            continue
        if not first_day_2more_17 or not first_sim_day_2more_17:
            continue
        # if first_day_2more_17 == first_sim_day_2more_17+1:
        #     # print('Predicted correctly')
        #     predicted_correctly += 1
        #     total += 1
        #     continue
        if first_day_2more_17 > first_sim_day_2more_17 and flag_not_in:
            continue
        if first_day_2more_17 == first_sim_day_2more_17:
            predicted_correctly += 1

        total += 1
    return predicted_correctly, total


def match_first_3_more_than_18(simulated_cycles):
    predicted_correctly = 0
    diffs = []
    diffs_cycles = []
    correct_cycles = []
    num_without_first = 0
    num_without_first_sim = 0
    miss_first = 0
    for cycle in simulated_cycles:
        first_day_3more_17 = None
        for profile_key in list(cycle.profiles.keys()):
            if int(profile_key) > 18 or first_day_3more_17 is not None:
                break
            num_more_than_17 = len([follicle for follicle in cycle.profiles[profile_key].follicles if follicle >= 18])
            if num_more_than_17 >= 2:
                first_day_3more_17 = int(profile_key)
                break

        first_sim_day_3more_17 = None
        for profile_key in list(cycle.simulated_profiles.keys()):
            if int(profile_key) > 18 or first_sim_day_3more_17 is not None:
                break
            num_more_than_17 = len(
                [follicle for follicle in cycle.simulated_profiles[profile_key].follicles if follicle >= 18])
            if num_more_than_17 >= 2:
                first_sim_day_3more_17 = int(profile_key)
                break
            # if first_sim_day_3more_17 is None or first_day_3more_17 is None:
            #     if first_sim_day_3more_17 is None:
            #         num_without_first_sim += 1
            #     if first_day_3more_17 is None:
            #         num_without_first += 1
            #     if num_without_first_sim == num_without_first:
            #         predicted_correctly += 1
            #     else:
            #         miss_first += 1

            continue
        if first_day_3more_17 == first_sim_day_3more_17 or first_day_3more_17 == first_sim_day_3more_17 + 1 or first_day_3more_17 == first_sim_day_3more_17 - 1:
            predicted_correctly += 1
            correct_cycles.append(len(cycle.profiles.keys()))
        else:
            diffs.append(first_day_3more_17 - first_sim_day_3more_17)
            diffs_cycles.append(len(cycle.profiles.keys()))
            # if len(diffs) < 30:
            #     print('#' * 20)
            #     print(cycle.profiles.keys())
            #     print(first_sim_day_3more_17)
            #     print(first_day_3more_17)

    return predicted_correctly, diffs, diffs_cycles, correct_cycles, num_without_first, num_without_first_sim, miss_first


#####################
# PLOTTING FUNCTIONS #
#####################

def plot_per_follicle_diffs(per_follicle_diffs, clinic=None, save_path=None):
    # %%
    # create a histogram of follicle differences
    hist, bins = np.histogram(per_follicle_diffs, bins=range(-5, 7))
    # normalise to percentages
    hist = hist / sum(hist)
    hist = hist * 100

    # plot using seaborn
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    font = {'size': 20, 'weight': 'bold'}
    # make bold font



    plt.rc('font', **font)
    plt.figure(figsize=(9, 7))
    sns.barplot(x=bins[:-1], y=hist, color='blue')
    # for bars between -2 and 2 make them green
    for idx, bar in enumerate(bins[:-1]):
        if bar >= -2 and bar <= 2:
            plt.gca().get_children()[idx].set_facecolor('green')
    # plot percentages on top of bars
    for idx, percentage in enumerate(hist):
        plt.text(idx - 0.45, percentage + 0.3, str(round(percentage, 1)) + '%', fontsize=16)
    # increase spacing
    # increase xticks font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    plt.subplots_adjust(bottom=0.2)
    # plt.title(
    #     'Follicle differences between real and simulated profiles {}'.format('for clinic ' + clinic if clinic else ''))
    plt.xlabel('Per follicle difference (mm)', fontweight='bold', fontsize=18)
    plt.ylabel('Percentage of follicles (%)', fontweight='bold', fontsize=18)
    if save_path:
        plt.savefig(save_path, dpi=600)


def plot_accuracy_per_time_delta_bar(scores_per_delta_time, save_path=None):
    # sort scores_per_delta_time by keys
    scores_per_delta_time = dict(sorted(scores_per_delta_time.items()))
    all_scores = []
    for delta in scores_per_delta_time:
        all_scores += scores_per_delta_time[delta]
    mean_all_scores = np.mean(all_scores)

    x = np.array(list(scores_per_delta_time.keys()))
    y = np.array([np.mean(scores_per_delta_time[delta]) for delta in x])
    # Create a dataframe
    df = pd.DataFrame({'x': x, 'y': y})

    num_cycles = np.array([len(scores_per_delta_time[delta]) for delta in x])
    # num_cycles = np.interp(num_cycles, (min(num_cycles), max(num_cycles)), (0.2, 1))
    # plot bars with size as accuracy and alpha as number of cycles using seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='x', y='y')
    # ax.bar_label(num_cycles, fontsize=10)
    for index, row in df.iterrows():
        ax.text(row.name, row.y, num_cycles[index], ha="center", fontsize=12)
    plt.xlabel('Delta time(days)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per time delta')
    plt.ylim(0, 1)
    # Increase font size
    plt.rc('font', size=14)
    # Increase spacing between ticks
    plt.tick_params(axis='both', which='major', pad=10)
    # plt.legend(loc="lower right")
    # Add overall accuracy
    plt.text(0.5, 0.9, 'Overall accuracy: {:.2f}%'.format(mean_all_scores * 100), horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()


def plot_accuracy_per_time_delta_lines(scores_per_delta_time, save_path=None):
    # sort scores_per_delta_time by keys
    scores_per_delta_time = dict(sorted(scores_per_delta_time.items()))

    x = np.array(list(scores_per_delta_time.keys()))
    y = np.array([np.mean(scores_per_delta_time[delta]) for delta in x])
    lwidths = [len(scores_per_delta_time[delta]) for delta in x]
    # scale lwidths to (1,5)
    lwidths = np.interp(lwidths, (min(lwidths), max(lwidths)), (1, 10))

    points = np.array([x, y]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, linewidths=lwidths, color='blue', alpha=1, label='Accuracy per time delta')

    fig, a = plt.subplots()
    a.add_collection(lc)
    a.legend()
    a.set_xlabel('Delta time(days)')
    a.set_ylabel('Accuracy')
    a.set_title('Accuracy per time delta')
    a.set_xlim(x.min(), x.max())
    a.set_ylim(0, 1)
    # plot_accuracy_per_time_delta(scores_per_delta_time)
    # plt.savefig('accuracy_per_time_delta.png')
    plt.show()
