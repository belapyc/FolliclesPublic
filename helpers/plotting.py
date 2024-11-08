import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import r2_score


import pandas as pd

from core.helper_core import get_next_day
from helpers.results_generation import histosection


def plot_patient_base(patient, swarm_flag=False):
    sns.set(rc={'figure.figsize': (12, 8)})
    profiles_ = []
    days = []

    for x, profile in enumerate(patient.profiles.values()):
        # profiles_.append(patient.simulated_profiles[x].follicles)
        profiles_.append([follicle for follicle in profile.follicles])
        days.append(str(profile.day))

    # Create a new df for plotting
    df_ = pd.DataFrame(days, columns=['day'])
    df_['follicles'] = profiles_
    df_ = df_.explode('follicles')

    if swarm_flag:
        sns_plot = sns.swarmplot(x='day', y='follicles', data=df_, hue='day')
    else:
        sns_plot = sns.stripplot(x='day', y='follicles', data=df_, jitter=True, hue='day')

    plt.legend([], [], frameon=False)


def plot_patient(patient, plot_until_last_act=False, swarm_flag=False, only_miss=False, ax=None,
                 y_limit=None, plot_dosage=False, marker_size=5):
    sns.set(rc={'figure.figsize': (14, 10)})
    # use ticks style
    sns.set(style="ticks")


    first_day = patient.profiles[get_next_day(0, patient.profiles)].day

    days = []
    profiles_ = []

    if plot_dosage:
        for dosage in patient.dosages:
            if dosage[1] < patient.profiles[get_next_day(0, patient.profiles)].day:
                # profiles_.append(dosage[0])
                profiles_.append([])
                days.append(str(int(dosage[1])))
    profiles_.append(patient.profiles[get_next_day(0, patient.profiles)].follicles)
    days.append(str(first_day))

    for x, sim_profile in enumerate(patient.simulated_profiles.values()):
        if str(sim_profile.day) in patient.profiles.keys():
            print('yes')
            profiles_.append([follicle for follicle in
                              patient.profiles[str(sim_profile.day)].follicles])
            days.append(str(patient.profiles[str(sim_profile.day)].day) + ' actual')
            # sim_profile.simulation_r2 = r2_score(patient.profiles[str(sim_profile.day)].follicles,
            #                                      sim_profile.follicles)
            # sim_profile.distance = close_follicles(patient.profiles[str(sim_profile.day)].follicles,
            #                                        sim_profile.follicles)
            # _, sim_profile.p_val = mannwhitneyu(patient.profiles[str(sim_profile.day)].follicles,
            #                                        sim_profile.follicles)
            sim_profile.histosection_score = histosection(patient.profiles[str(sim_profile.day)].follicles,
                                                    sim_profile.follicles)

            if only_miss:
                sim_follicles = [follicle for follicle in sim_profile.follicles]
                act_follicles = [follicle for follicle in patient.profiles[str(sim_profile.day)].follicles]
                for idx, follicle in enumerate(sim_follicles):
                    for idy, y in enumerate(act_follicles):
                        if follicle == y:
                            del sim_follicles[idx]
                            del act_follicles[idy]
            else:
                sim_follicles = [follicle for follicle in sim_profile.follicles]

        else:
            print('no')
            profiles_.append([])  # if follicle>10 if follicle <20])
            days.append(sim_profile.day)
            sim_follicles = [follicle for follicle in sim_profile.follicles]
        # profiles_.append(patient.simulated_profiles[x].follicles)
        profiles_.append(sim_follicles)
        days.append(str(sim_profile.day) + ' predicted')
        if plot_until_last_act and get_next_day(sim_profile.day, patient.profiles) == 'Not found':
            break

    # Create a new df for plotting
    df_ = pd.DataFrame(days, columns=['day'])
    df_['follicles'] = profiles_
    # print(df_)
    df_ = df_.explode('follicles')
    # print(df_)
    df_ = df_.reset_index()
    # print(df_)
    # add a column to flag if follicle is actual or simulated
    df_['predicted'] = df_['day'].str.contains('predicted')
    df_['predicted'] = df_['predicted'].replace({True: 'Predicted follicles', False: 'Actual follicles'})


    if swarm_flag:
        sns_plot = sns.swarmplot(x='day', y='follicles', data=df_, hue='predicted', ax=ax, size=marker_size)
    else:
        sns_plot = sns.stripplot(x='day', y='follicles', data=df_, jitter=True, hue='day', ax=ax, size=marker_size)

    if plot_dosage:
        dosages = patient.dosages
        df_dosage = pd.DataFrame(dosages, columns=['dosage', 'day'])
        # convert day to string
        df_dosage['day'] = df_dosage['day'].astype(int).astype(str)
        print(df_dosage)
        # normalise dosage for plotting relative to follicles
        df_dosage['dosage'] = df_dosage['dosage'] / 15

        sns_plot2 = sns.lineplot(x='day', y='dosage', data=df_dosage, ax=ax, color='black')
        # add dosage values to plot where dosage changes
        for idx, row in df_dosage.iterrows():
            # if idx == 0:
            #     continue
            if df_dosage.iloc[idx - 1]['dosage'] != row['dosage']:
                sns_plot2.text(row['day'], row['dosage'], row['dosage']*15, horizontalalignment='center')


    ax = sns_plot.axes
    if y_limit:
        ax.set_ylim((4, y_limit))
    for k, profile in enumerate(patient.simulated_profiles.values()):
        if profile.simulation_r2 is not None:
            x_loc = (k * 2) + 1.5
            ax.text(x_loc + 0.5, ax.get_ylim()[1], f'R2: {profile.simulation_r2:.2f}', horizontalalignment='center')
        if profile.distance is not None:
            x_loc = (k * 2) + 1.5
            ax.text(x_loc + 0.5, ax.get_ylim()[1] - 1, f'same: {profile.distance:.2f}%', horizontalalignment='center')

        if profile.histosection_score is not None:
            x_loc = (k * 2) + 1.5
            # ax.text(x_loc + 0.5, ax.get_ylim()[1] - 2, f'histo: {profile.histosection_score:.2f}', horizontalalignment='center')

        if 'p_val' in dir(profile):
            if profile.p_val is not None:
                x_loc = (k * 2) + 1.5
                ax.text(x_loc + 0.5, ax.get_ylim()[1] - 2, f'MannWhit: {profile.p_val:.2f}', horizontalalignment='center')


        ax.add_patch(
            plt.Rectangle(((k * 2) + 1.5, 2.5), 1, ax.get_ylim()[1], linewidth=1, edgecolor='r', facecolor='y',
                          alpha=0.1))
        # ax.text(0.1, 0.9, 'Trigger Day: {}\n1st max 1219 sim: {}\n first 3 18 sim: {} \n key: {}'.format(
        #     patient.trigger_day, patient.simulated_trigger_days['max_12_19'],
        #     patient.simulated_trigger_days['first_3_more'], patient.key), horizontalalignment='center',
        #         verticalalignment='center', transform=ax.transAxes)
        # ax.text(0.5, 1, 'Trigger Day: {}\n1st max 1219 sim: {}\n first 3 18 sim: {} \n key: {}'.format(
        #     patient.trigger_day, patient.simulated_trigger_days['max_12_19'],
        #     patient.simulated_trigger_days['first_3_more'], patient.key), horizontalalignment='center',
        #         verticalalignment='center', transform=ax.transAxes)
    # x label 'Day of cycle'
    ax.set_xlabel('Day of cycle')
    # y label 'Follicle size (mm)'
    ax.set_ylabel('Follicle size (mm)')
    # make bold font
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(14)
            item.set_fontweight('bold')
    # make bold font
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
            item.set_fontweight('bold')


    # add legend based on predicted or actual
    ax.legend()

    # increase font size of legend
    for label in ax.get_legend().get_texts():
        label.set_fontsize('10')
        label.set_fontweight('bold')

    # ax.legend([], [], frameon=False)
    # plt.legend([], [], frameon=False)
    return df_


def plot_compare_profiles(patient, day, save_fig=False, simulation_type='None'):
    sns.set(rc={'figure.figsize': (12, 8)})
    profiles_ = [patient.profiles[day].follicles, patient.simulated_profiles[day].follicles]
    days = [day, day + 'sim']

    # Create a new df for plotting
    df_ = pd.DataFrame(days, columns=['day'])
    df_['follicles'] = profiles_
    df_ = df_.explode('follicles')

    sns_plot = sns.swarmplot(x='day', y='follicles', data=df_, hue='day')

    ax = sns_plot.axes

    ax.set(ylim=(4, 30))

    # ax.text(0.5, 0.95, 'Simulation type: {}\nk:{}'.format(simulation_type, patient.key), horizontalalignment='center',
    #         verticalalignment='center', transform=ax.transAxes)

    ax.add_patch(
        plt.Rectangle((0.5, 1), 1, ax.get_ylim()[1], linewidth=1, edgecolor='r', facecolor='y',
                      alpha=0.1))

    plt.legend([], [], frameon=False)
    if save_fig:
        if int(day) <= 9:
            plt.savefig('./comp_images/59/{}_{}.png'.format(patient.key, day))
        if 9 < int(day) <= 13:
            plt.savefig('./comp_images/913/{}_{}.png'.format(patient.key, day))
        if 13 < int(day):
            plt.savefig('./comp_images/13up/{}_{}.png'.format(patient.key, day))


def plot_sim_v_actual(patient, act_first_days=None):
    sns.set(rc={'figure.figsize': (12, 8)})
    profiles_ = [patient.profiles[get_next_day(0, patient.profiles)].follicles]
    sim_profiles_ = [patient.profiles[get_next_day(0, patient.profiles)].follicles]
    act_days = [patient.profiles[get_next_day(0, patient.profiles)].day]
    sim_days = [patient.profiles[get_next_day(0, patient.profiles)].day]

    for x, sim_profile in enumerate(patient.simulated_profiles.values()):
        if sim_profile.day in sim_days:
            continue
        sim_profiles_.append([follicle for follicle in sim_profile.follicles])
        sim_days.append(int(sim_profile.day))

    if act_first_days is not None:
        act_profiles = list(patient.profiles.values())[:act_first_days]
    else:
        act_profiles = list(patient.profiles.values())

    for profile in act_profiles:
        if profile.day in act_days:
            continue
        profiles_.append([follicle for follicle in
                          patient.profiles[str(profile.day)].follicles])
        act_days.append(patient.profiles[str(profile.day)].day)

    print(act_days)
    print(sim_days)

    # Create a new df for plotting
    df_act = pd.DataFrame(act_days, columns=['day'])
    df_act['profiles'] = profiles_
    df_act = df_act.explode('profiles')

    df_sim = pd.DataFrame(sim_days, columns=['day'])
    df_sim['profiles'] = sim_profiles_
    df_sim = df_sim.explode('profiles')

    df_act = df_act.dropna()
    df_act['profiles'] = df_act['profiles'].astype('int64')
    df_act.reset_index(level=0).drop(['index'], axis=1)

    df_sim = df_sim.dropna()
    df_sim['profiles'] = df_sim['profiles'].astype('int64')
    df_sim.reset_index(level=0).drop(['index'], axis=1)

    sns.lineplot(x=df_act.reset_index(level=0).drop(['index'], axis=1)['day'],
                 y=df_act.reset_index(level=0).drop(['index'], axis=1)['profiles'],
                 label='actual')
    sns.lineplot(x=df_sim.reset_index(level=0).drop(['index'], axis=1)['day'],
                 y=df_sim.reset_index(level=0).drop(['index'], axis=1)['profiles'],
                 label='simulated')


def plot_sim_v_actual_updated(patient, act_first_days=None, start_sim_day=0):
    '''
    https://seaborn.pydata.org/generated/seaborn.lineplot.html
    Passing the entire dataset in long-form mode will aggregate over repeated values (each year) to show the mean and 95% confidence interval:
    Args:
        patient:
        act_first_days:
        start_sim_day:

    Returns:

    '''
    sns.set(rc={'figure.figsize': (12, 8)})

    first_day = get_next_day(0, patient.profiles)

    profiles_ = [patient.profiles[first_day].follicles]
    act_days = [patient.profiles[first_day].day]

    if act_first_days is not None:
        act_profiles = list(patient.profiles.values())[:act_first_days]
    else:
        act_profiles = list(patient.profiles.values())

    for profile in act_profiles:
        if profile.day in act_days:
            continue
        print('profile day: {}'.format(profile.day))
        print(patient.profiles.keys())
        profiles_.append([follicle for follicle in
                          patient.profiles[str(profile.day)].follicles])
        act_days.append(patient.profiles[str(profile.day)].day)

    orig_sim_days = [patient.profiles[first_day].day]
    orig_sim_profiles_ = [patient.profiles[first_day].follicles]

    for x, sim_profile in enumerate(patient.simulated_profiles.values()):
        if sim_profile.day in orig_sim_days:
            continue
        orig_sim_profiles_.append([follicle for follicle in sim_profile.follicles])
        orig_sim_days.append(int(sim_profile.day))

    sim_days = [act_days[start_sim_day]]
    sim_profiles_ = [act_profiles[start_sim_day].follicles]

    # diff = sim_days[0] - first_day
    # print('first day: {}, sim day: {}, diff: {}'.format(first_day, sim_days[0], diff))
    # print(list(patient.updated_simulated_profiles.values()))
    for x, sim_profile in enumerate(list(patient.updated_simulated_profiles.values())):
        if sim_profile.day in sim_days:
            continue
        sim_profiles_.append([follicle for follicle in sim_profile.follicles])
        sim_days.append(int(sim_profile.day))

    # print('act_days: {}'.format(act_days))
    # print('orig_sim_days: {}'.format(orig_sim_days))
    # print('sim_days: {}'.format(sim_days))

    # Create a new df for plotting
    df_act = pd.DataFrame(act_days, columns=['day'])
    df_act['profiles'] = profiles_
    df_act = df_act.explode('profiles')

    df_orig_sim = pd.DataFrame(orig_sim_days, columns=['day'])
    df_orig_sim['profiles'] = orig_sim_profiles_
    df_orig_sim = df_orig_sim.explode('profiles')

    df_sim = pd.DataFrame(sim_days, columns=['day'])
    df_sim['profiles'] = sim_profiles_
    df_sim = df_sim.explode('profiles')

    df_act = df_act.dropna()
    df_act['profiles'] = df_act['profiles'].astype('int64')
    df_act.reset_index(level=0).drop(['index'], axis=1)

    df_orig_sim = df_orig_sim.dropna()
    df_orig_sim['profiles'] = df_orig_sim['profiles'].astype('int64')
    df_orig_sim.reset_index(level=0).drop(['index'], axis=1)

    df_sim = df_sim.dropna()
    df_sim['profiles'] = df_sim['profiles'].astype('int64')
    df_sim.reset_index(level=0).drop(['index'], axis=1)

    sns.lineplot(x=df_act.reset_index(level=0).drop(['index'], axis=1)['day'],
                 y=df_act.reset_index(level=0).drop(['index'], axis=1)['profiles'],
                 label='actual')

    sns.lineplot(x=df_orig_sim.reset_index(level=0).drop(['index'], axis=1)['day'],
                 y=df_orig_sim.reset_index(level=0).drop(['index'], axis=1)['profiles'],
                 label='simulated_original')

    sns.lineplot(x=df_sim.reset_index(level=0).drop(['index'], axis=1)['day'],
                 y=df_sim.reset_index(level=0).drop(['index'], axis=1)['profiles'],
                 label='simulated')


def plot_follicles_dict_full(follicles_dict):
    '''
    Plot a matrix of histograms for each follicle in the follicles_dict
    Args:
        follicles_dict:

    Returns:

    '''
    fig, ax = plt.subplots(5, 5, figsize=(12, 12))
    fig.tight_layout(pad=3.0)
    for idx, follicle in enumerate(follicles_dict.keys()):
        # plt.figure()
        ax[int(idx / 5)][(idx % 5)].hist(follicles_dict[follicle]['absolute_growth'],
                                         bins=[-5, -2.5, -0.5, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        ax[int(idx / 5)][(idx % 5)].title.set_text(
            'Follicle {}\n num samples {}'.format(follicle, len(follicles_dict[follicle]['absolute_growth'])))


def plot_chance_of_growth_follicles(follicles_dict):
    x = []
    y = []
    hue = []
    for follicle in follicles_dict.keys():
        x.append(int(follicle))
        y.append(follicles_dict[follicle]['chance_of_growth'])

    max_y = 0

    for x_, y_ in zip(x, y):
        if y_ > max_y:
            max_y = y_
            max_x = x_

    sns.set(rc={'figure.figsize': (12, 8)})
    sns.barplot(x=x, y=y)


def plot_growth_rate_follicles(follicles_dict, title):
    x = []
    y = []
    hue = []
    counts = []
    keys = list(follicles_dict.keys())
    print(keys)
    keys.remove('total_cycles_used')
    keys.remove('total_follicles_used')
    keys.remove('total_negative_growth_count')
    keys.remove('total_negative_growth_follicle_second')
    keys_sorted = sorted(keys, key=lambda x: int(x))
    for follicle in keys_sorted:
        # print('Follicle: {}, growth rate: {}'.format(follicle, follicles_dict[follicle]['diff_absolute_median']))
        x.append(int(follicle))
        y.append(follicles_dict[follicle]['diff_absolute_median'])
        hue.append(len(follicles_dict[follicle]['absolute_growth']))
        counts.append(len(follicles_dict[follicle]['absolute_growth']))

    # absolute_growthcounts = zip(follicles_dict.keys(), counts)
    max_y = 0

    for x_, y_ in zip(x, y):
        if y_ > max_y:
            max_y = y_
            max_x = x_

    sns.set(rc={'figure.figsize': (12, 8)})
    ax = sns.barplot(x=x, y=y)

    # set y limit to 2.5
    ax.set_ylim(0, 3.5)

    ax.bar_label(ax.containers[0], labels=counts, padding=3)

    ax.set(xlabel='Follicle', ylabel='Median Growth Rate (mm/day)')
    ax.set_title('Growth rates of follicles for group: {} (total number of cycles used {})'.format(title,
                                                                                                   follicles_dict[
                                                                                                       'total_cycles_used']))


def plot_difference(follicle_dict_1, follicle_dict_2, title):
    """
    Plot the difference between two follicle dicts
    Plot the difference between two barcharts
    Args:
        follicle_dict_1:
        follicle_dict_2:
        title:

    Returns:

    """
    x = []
    y = []
    hue = []
    counts = []
    keys = list(follicle_dict_1.keys())
    keys.remove('total_cycles_used')
    keys.remove('total_follicles_used')
    keys.remove('total_negative_growth_count')
    keys.remove('total_negative_growth_follicle_second')
    keys_sorted = sorted(keys, key=lambda x: int(x))
    for follicle in keys_sorted:
        # print('Follicle: {}, growth rate: {}'.format(follicle, follicles_dict[follicle]['diff_absolute_median']))
        x.append(int(follicle))
        y.append(follicle_dict_1[follicle]['diff_absolute_median'] - follicle_dict_2[follicle]['diff_absolute_median'])
        hue.append(len(follicle_dict_1[follicle]['absolute_growth']))
        counts.append(len(follicle_dict_1[follicle]['absolute_growth']))

    # absolute_growthcounts = zip(follicles_dict.keys(), counts)
    max_y = 0

    for x_, y_ in zip(x, y):
        if y_ > max_y:
            max_y = y_
            max_x = x_

    sns.set(rc={'figure.figsize': (12, 8)})
    ax = sns.barplot(x=x, y=y)

    # set y limit to 2.5
    ax.set_ylim(-3.5, 3.5)

    ax.bar_label(ax.containers[0], labels=counts, padding=3)

    ax.set(xlabel='Follicle', ylabel='Median Growth Rate (mm/day)')
    ax.set_title('Growth rates of follicles for group: {} (total number of cycles used {})'.format(title,
                                                                                                   follicle_dict_1[
                                                                                                       'total_cycles_used']))


def plot_growth_chance_follicle_day(follicles_dict, follicle):
    '''
    Plot the chance of growth for a given follicle on different days
    Args:
        follicles_dict:
        follicle:

    Returns:

    '''
    x = []
    y = []
    hue = []
    counts = []
    keys = list(follicles_dict[follicle].keys())
    # keys.remove('total_cycles_used')
    keys_sorted = sorted(keys, key=lambda x: int(x))
    for day in keys_sorted:
        x.append(int(day))
        y.append(follicles_dict[follicle][day]['chance_of_growth'])
        hue.append(len(follicles_dict[follicle][day]))
        counts.append(len(follicles_dict[follicle][day]['next_follicles']))

    max_y = 0

    for x_, y_ in zip(x, y):
        if y_ > max_y:
            max_y = y_
            max_x = x_

    sns.set(rc={'figure.figsize': (12, 8)})
    plt.legend(title='Count', loc='upper left', labels=counts)
    ax = sns.barplot(x=x, y=y, capsize=.2)
    ax.bar_label(ax.containers[0], labels=counts, padding=3)

    ax.set(xlabel='Day', ylabel='Chance of growth(%)')
    ax.set_title('Growth chance of follicle {} per day'.format(follicle))


def plot_growth_rate_follicle_day(follicles_dict, follicle):
    x = []
    y = []
    hue = []
    counts = []
    keys = list(follicles_dict[follicle].keys())
    keys_sorted = sorted(keys, key=lambda x: int(x))
    for day in keys_sorted:
        x.append(int(day))
        y.append(np.mean(follicles_dict[follicle][day]['absolute_growth']))
        hue.append(len(follicles_dict[follicle][day]))
        counts.append(len(follicles_dict[follicle][day]['next_follicles']))

    max_y = 0

    for x_, y_ in zip(x, y):
        if y_ > max_y:
            max_y = y_
            max_x = x_

    sns.set(rc={'figure.figsize': (12, 8)})
    ax = sns.barplot(x=x, y=y, capsize=.2)
    ax.bar_label(ax.containers[0], labels=counts, padding=3)

    ax.set(xlabel='Day', ylabel='Median growth rate (mm/day)')
    ax.set_title('Growth rate of follicle {} per day'.format(follicle))
