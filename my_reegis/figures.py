import os
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime

from my_reegis import results
from my_reegis import reegis_plot as plot
from my_reegis import regional_results
from my_reegis import friedrichshagen_scenarios as fhg_sc

from oemof.tools import logger
from oemof import solph

import berlin_hp
import deflex

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.dates as mdates

from berlin_hp import electricity

from reegis import config as cfg
from reegis import energy_balance
from reegis import inhabitants
from reegis import geometries
from reegis import powerplants
from reegis import storages
from deflex import demand
from reegis import entsoe
from reegis import coastdat

from my_reegis import data_analysis
from berlin_hp import heat


NAMES = {
    'lignite': 'Braunkohle',
    'natural_gas': 'Gas',
    'oil': 'Öl',
    'hard_coal': 'Steinkohle',
    'netto_import': 'Stromimport',
    'other': 'sonstige',
    # 'nuclear': 'Atomkraft',
}


def create_subplot(default_size, **kwargs):
    size = kwargs.get('size', default_size)
    return plt.figure(figsize=size).add_subplot(1, 1, 1)


def fig_compare_feedin_solar():
    plt.rcParams.update({'font.size': 14})
    f, ax_ar = plt.subplots(2, 1, sharey=True, figsize=(15, 6))

    # Get entsoe time series from opsd
    cso = '#ff7e00'
    csr = '#500000'
    my_re = entsoe.get_entsoe_renewable_data().div(1000)
    ax = my_re['DE_solar_profile'].multiply(1000).plot(ax=ax_ar[0], color=cso)
    ax2 = my_re['DE_solar_profile'].multiply(1000).plot(ax=ax_ar[1], color=cso)

    # Get reegis time series
    my_path = os.path.join(cfg.get('paths', 'feedin'), 'federal_states')
    my_fn = os.path.join(my_path, 'federal_states_{0}'.format(2014))
    re_rg = pd.read_csv(my_fn, index_col=[0], header=[0, 1]).set_index(
        pd.date_range('1/1/2014', periods=8760, freq='H'))
    fs = geometries.get_federal_states_polygon()
    pp = powerplants.get_powerplants_by_region(fs, 2014, 'federal_states')
    total_capacity = pp.capacity_2014.swaplevel().loc['Solar'].sum()

    re_rg = re_rg.swaplevel(axis=1)['solar'].mul(
        pp.capacity_2014.swaplevel().loc['Solar'])

    # Plot reegis time series
    # June
    ax = re_rg.sum(axis=1).div(total_capacity).plot(
        ax=ax, rot=0, color=csr,
        xlim=(datetime.datetime(2014, 6, 1), datetime.datetime(2014, 6, 30)))

    # December
    ax2 = re_rg.sum(axis=1).div(total_capacity).plot(
        ax=ax2, rot=0, color=csr,
        xlim=(datetime.datetime(2014, 12, 1), datetime.datetime(2014, 12, 30)))

    # x-ticks for June
    dates = [datetime.datetime(2014, 6, 1),
             datetime.datetime(2014, 6, 5),
             datetime.datetime(2014, 6, 9),
             datetime.datetime(2014, 6, 13),
             datetime.datetime(2014, 6, 17),
             datetime.datetime(2014, 6, 21),
             datetime.datetime(2014, 6, 25),
             datetime.datetime(2014, 6, 29)]
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ''
    ax.set_xticklabels(labels, ha='center')

    # xticks for December
    dates = [datetime.datetime(2014, 12, 1),
             datetime.datetime(2014, 12, 5),
             datetime.datetime(2014, 12, 9),
             datetime.datetime(2014, 12, 13),
             datetime.datetime(2014, 12, 17),
             datetime.datetime(2014, 12, 21),
             datetime.datetime(2014, 12, 25),
             datetime.datetime(2014, 12, 29)]
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ''
    ax2.set_xticklabels(labels, ha='center')

    ax.legend(labels=['OPSD', 'reegis'])
    ax.set_xlabel('')
    ax.set_ylim((0, 1.1))
    ax2.set_xlabel('Juni/Dezember 2014')
    ax2.xaxis.labelpad = 20

    # Plot Text
    x0 = datetime.datetime(2014, 12, 1, 5, 0)
    x1 = datetime.datetime(2014, 12, 1, 8, 0)
    x2 = datetime.datetime(2014, 12, 2, 14, 0)

    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2015, 1, 1)

    # BMWI
    # https://www.bmwi.de/Redaktion/DE/Publikationen/Energie/
    #     erneuerbare-energien-in-zahlen-2017.pdf?__blob=publicationFile&v=27
    bmwi_sum = round(36.056)
    reegis_sum = round(re_rg.sum().sum() / 1000000)
    opsd_sum = round(
        my_re.DE_solar_generation_actual.loc[start:end].sum() / 1000)

    text = {
        'title': (x1, 1, ' Summe 2014'),
        "op1": (x1, 0.85, "OPSD"),
        "op2": (x2, 0.85, "{0} GWh".format(int(opsd_sum))),
        "reg1": (x1, 0.70, "reegis"),
        "reg2": (x2, 0.70, "{0} GWh".format(int(reegis_sum))),
        'bmwi1': (x1, 0.55, 'BMWi'),
        "bmwi2": (x2, 0.55, "{0} GWh".format(int(bmwi_sum))),
    }

    for t, c in text.items():
        if t == 'title':
            w = 'bold'
        else:
            w = 'normal'
        ax2.text(c[0], c[1], c[2], weight=w, size=12, ha="left", va="center")

    # Plot Box
    x3 = mdates.date2num(x0)
    b = patches.Rectangle((x3, 0.5), 2.9, 0.57, color='#cccccc')
    ax2.add_patch(b)
    ax2.add_patch(patches.Shadow(b, -0.05, -0.01))

    plt.subplots_adjust(right=0.99, left=0.05, bottom=0.16, top=0.97)
    return 'compare_feedin_solar', None


def fig_compare_feedin_wind():
    plt.rcParams.update({'font.size': 14})
    f, ax_ar = plt.subplots(2, 1, sharey=True, figsize=(15, 6))

    # colors
    cwo = '#665eff'
    cwr = '#0a085e'

    # Get entsoe time series from opsd
    my_re = entsoe.get_entsoe_renewable_data().div(1000)
    ax = my_re['DE_wind_profile'].multiply(1000).plot(ax=ax_ar[0], color=cwo)
    ax2 = my_re['DE_wind_profile'].multiply(1000).plot(ax=ax_ar[1], color=cwo)

    # Get reegis time series
    my_path = os.path.join(cfg.get('paths', 'feedin'), 'federal_states')
    my_fn = os.path.join(my_path, 'federal_states_{0}'.format(2014))
    re_rg = pd.read_csv(my_fn, index_col=[0], header=[0, 1]).set_index(
        pd.date_range('1/1/2014', periods=8760, freq='H'))
    fs = geometries.get_federal_states_polygon()
    pp = powerplants.get_powerplants_by_region(fs, 2014, 'federal_states')
    total_capacity = pp.capacity_2014.swaplevel().loc['Wind'].sum()
    re_rg = re_rg.swaplevel(axis=1)['wind'].mul(
        pp.capacity_2014.swaplevel().loc['Wind'])

    # Plot reegis time series (use multiply to adjust the overall sum)
    # June
    ax = re_rg.sum(axis=1).div(total_capacity).multiply(1).plot(
        ax=ax, rot=0, color=cwr,
        xlim=(datetime.datetime(2014, 6, 1), datetime.datetime(2014, 6, 30)))

    # December
    ax2 = re_rg.sum(axis=1).div(total_capacity).multiply(1).plot(
        ax=ax2, rot=0, color=cwr,
        xlim=(datetime.datetime(2014, 12, 1), datetime.datetime(2014, 12, 30)))

    # x-ticks for June
    dates = [datetime.datetime(2014, 6, 1),
             datetime.datetime(2014, 6, 5),
             datetime.datetime(2014, 6, 9),
             datetime.datetime(2014, 6, 13),
             datetime.datetime(2014, 6, 17),
             datetime.datetime(2014, 6, 21),
             datetime.datetime(2014, 6, 25),
             datetime.datetime(2014, 6, 29)]
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ''
    ax.set_xticklabels(labels, ha='center')

    # xticks for December
    dates = [datetime.datetime(2014, 12, 1),
             datetime.datetime(2014, 12, 5),
             datetime.datetime(2014, 12, 9),
             datetime.datetime(2014, 12, 13),
             datetime.datetime(2014, 12, 17),
             datetime.datetime(2014, 12, 21),
             datetime.datetime(2014, 12, 25),
             datetime.datetime(2014, 12, 29)]
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ''
    ax2.set_xticklabels(labels, ha='center')

    ax.legend(labels=['OPSD', 'reegis'])
    ax.set_xlabel('')
    ax.set_ylim((0, 1.1))
    ax2.set_xlabel('Juni/Dezember 2014')
    ax2.xaxis.labelpad = 20

    # Plot Text
    x0 = datetime.datetime(2014, 6, 1, 5, 0)
    x1 = datetime.datetime(2014, 6, 1, 8, 0)
    x2 = datetime.datetime(2014, 6, 2, 14, 0)

    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2015, 1, 1)

    # BMWI
    # https://www.bmwi.de/Redaktion/DE/Publikationen/Energie/
    #     erneuerbare-energien-in-zahlen-2017.pdf?__blob=publicationFile&v=27
    bmwi_sum = round((1471 + 57026) / 1000)
    reegis_sum = round(re_rg.sum().sum()/1000000)
    opsd_sum = round(my_re.DE_wind_generation_actual.loc[start:end].sum()/1000)

    print(opsd_sum/reegis_sum)

    text = {
        'title': (x1, 1, ' Summe 2014'),
        "op1": (x1, 0.85, "OPSD"),
        "op2": (x2, 0.85, "{0} GWh".format(int(opsd_sum))),
        "reg1": (x1, 0.70, "reegis"),
        "reg2": (x2, 0.70, "{0} GWh".format(int(reegis_sum))),
        'bmwi1': (x1, 0.55, 'BMWi'),
        "bmwi2": (x2, 0.55, "{0} GWh".format(int(bmwi_sum))),
      }

    for t, c in text.items():
        if t == 'title':
            w = 'bold'
        else:
            w = 'normal'
        ax.text(c[0], c[1], c[2], weight=w, size=12, ha="left", va="center")

    # Plot Box
    x3 = mdates.date2num(x0)
    b = patches.Rectangle((x3, 0.5), 2.9, 0.57, color='#cccccc')
    ax.add_patch(b)
    ax.add_patch(patches.Shadow(b, -0.05, -0.01))

    plt.subplots_adjust(right=0.99, left=0.05, bottom=0.16, top=0.97)
    return 'compare_feedin_wind', None


def fig_compare_full_load_hours():
    plt.rcParams.update({'font.size': 14})
    f, ax_ar = plt.subplots(2, 2, sharex=True, figsize=(15, 7))

    fn = os.path.join(cfg.get('paths', 'data_my_reegis'),
                      'full_load_hours_re_bdew_states.csv')
    flh = pd.read_csv(fn, index_col=[0], header=[0, 1])
    # flh['Solar (BDEW)'].plot(kind='bar', ax=ax_ar[1])
    for y in [2014, 2012]:
        my_path = os.path.join(cfg.get('paths', 'feedin'), 'federal_states')
        my_fn = os.path.join(my_path, 'federal_states_{0}'.format(y))
        re_rg = pd.read_csv(my_fn, index_col=[0], header=[0, 1]).swaplevel(
            axis=1)
        flh['Wind (reegis)', str(y)] = re_rg['wind'].sum()
        flh['Solar (reegis)', str(y)] = re_rg['solar'].sum()

    flh.drop(['DE', 'N1', 'O0'], inplace=True)

    ax_ar[0, 0] = flh[
        [('Wind (BDEW)', '2012'), ('Wind (reegis)', '2012')]].plot(
            kind='bar', ax=ax_ar[0, 0], color=['#4254ff', '#1b2053'],
            legend=False)
    ax_ar[0, 1] = flh[
        [('Wind (BDEW)', '2014'), ('Wind (reegis)', '2014')]].plot(
            kind='bar', ax=ax_ar[0, 1], color=['#4254ff', '#1b2053'],
            legend=False)
    ax_ar[1, 0] = flh[
        [('Solar (BDEW)', '2012'), ('Solar (reegis)', '2012')]].plot(
            kind='bar', ax=ax_ar[1, 0], color=['#ffba00', '#ff7000'],
            legend=False)
    ax_ar[1, 1] = flh[
        [('Solar (BDEW)', '2014'), ('Solar (reegis)', '2014')]].plot(
            kind='bar', ax=ax_ar[1, 1], color=['#ffba00', '#ff7000'],
            legend=False)
    ax_ar[0, 0].set_title('2012')
    ax_ar[0, 1].set_title('2014')
    ax_ar[0, 1].legend(loc='upper left', bbox_to_anchor=(1, 1),
                       labels=['BDEW', 'reegis'])
    ax_ar[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1),
                       labels=['BDEW', 'reegis'])
    ax_ar[0, 0].set_ylabel('Volllaststunden Windkraft')
    ax_ar[1, 0].set_ylabel('Volllaststunden Photovoltaik')

    plt.subplots_adjust(right=0.89, left=0.07, bottom=0.11, top=0.94,
                        wspace=0.16)
    return 'compare_full_load_hours', None


def fig_compare_re_capacity_years():
    plt.rcParams.update({'font.size': 14})
    f, ax_ar = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 5))

    my_re = entsoe.get_entsoe_renewable_data().div(1000)
    rn = {'DE_solar_capacity': 'Solar (OPSD)',
          'DE_wind_capacity': 'Wind (OPSD)'}
    my_re.rename(columns=rn, inplace=True)

    ax_ar[0] = my_re['Solar (OPSD)'].plot(
        ax=ax_ar[0], color='#ffba00', legend=True)
    ax_ar[1] = my_re['Wind (OPSD)'].plot(
        ax=ax_ar[1], color='#4254ff', legend=True)

    fs = geometries.get_federal_states_polygon()
    df = pd.DataFrame()
    for y in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        my_pp = powerplants.get_powerplants_by_region(fs, y, 'federal_states')
        for cat in ['Solar', 'Wind']:
            dt = datetime.datetime(y, 1, 1)
            cat_name = "{0} (reegis)".format(cat)
            col = 'capacity_{0}'.format(y)
            df.loc[dt, cat_name] = my_pp.groupby(level=1).sum().loc[cat, col]
    df = df.div(1000)
    ax_ar[0] = df['Solar (reegis)'].plot(
        drawstyle="steps-post", ax=ax_ar[0], color='#ff7000', legend=True)
    ax_ar[1] = df['Wind (reegis)'].plot(
        drawstyle="steps-post", ax=ax_ar[1], color=['#1b2053'], legend=True)

    fn = os.path.join(cfg.get('paths', 'data_my_reegis'),
                      'bmwi_installed_capacity_wind_pv.csv')
    bmwi = pd.read_csv(fn, index_col=[0]).transpose().div(1000)
    bmwi = bmwi.set_index(pd.to_datetime(bmwi.index.astype(str) + '-12-31'))

    ax_ar[0] = bmwi['Solar (BMWi)'].plot(
        marker='D', ax=ax_ar[0], linestyle='None', markersize=10,
        color='#ff5500', alpha=0.7, legend=True)
    ax_ar[1] = bmwi['Wind (BMWi)'].plot(
        marker='D', ax=ax_ar[1], linestyle='None', markersize=10,
        color='#111539', alpha=0.7, legend=True)

    ax_ar[0].set_xlim(left=datetime.datetime(2012, 1, 1))
    plt.ylim((25, 60))
    ax_ar[0].set_ylabel('Installierte Leistung [GW]')
    ax_ar[0].set_xlabel(' ')
    ax_ar[1].set_xlabel(' ')
    ax_ar[0].legend(loc='upper left')
    ax_ar[1].legend(loc='upper left')
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.11, top=0.94,
                        wspace=0.16)
    return 'compare_re_capacity_years', None


def fig_analyse_multi_files():
    path = os.path.join(cfg.get('paths', 'analysis'), 'pv_orientation_minus30')
    fn = os.path.join(path, 'multiyear_yield_sum.csv')
    df = pd.read_csv(fn, index_col=[0, 1])
    gdf = data_analysis.get_coastdat_onshore_polygons()
    gdf.geometry = gdf.buffer(0.005)

    for key in gdf.index:
        s = df[str(key)]
        pt = gdf.loc[key]
        gdf.loc[key, 'tilt'] = (
            s[s == s.max()].index.get_level_values('tilt')[0])
        gdf.loc[key, 'azimuth'] = (
            s[s == s.max()].index.get_level_values('azimuth')[0])
        gdf.loc[key, 'longitude'] = pt.geometry.centroid.x
        gdf.loc[key, 'latitude'] = pt.geometry.centroid.y
        gdf.loc[key, 'tilt_calc'] = round(pt.geometry.centroid.y - 15)
        gdf.loc[key, 'tilt_diff'] = abs(gdf.loc[key, 'tilt_calc'] -
                                        gdf.loc[key, 'tilt'])
        gdf.loc[key, 'tilt_diff_c'] = abs(gdf.loc[key, 'tilt'] - 36.5)
        gdf.loc[key, 'azimuth_diff'] = abs(gdf.loc[key, 'azimuth'] - 178.5)

    cmap_t = plt.get_cmap('viridis', 8)
    cmap_az = plt.get_cmap('viridis', 7)
    cm_gyr = LinearSegmentedColormap.from_list(
        'mycmap', [
            (0, 'green'),
            (0.5, 'yellow'),
            (1, 'red')], 6)

    f, ax_ar = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(13, 6))

    ax_ar[0][0].set_title("Azimuth (optimal)", loc='center', y=1)
    gdf.plot('azimuth', legend=True, cmap=cmap_az, vmin=173, vmax=187,
             ax=ax_ar[0][0])

    ax_ar[0][1].set_title("Neigung (optimal)", loc='center', y=1)
    gdf.plot('tilt', legend=True, vmin=32.5, vmax=40.5, cmap=cmap_t,
             ax=ax_ar[0][1])

    ax_ar[0][2].set_title("Neigung (nach Breitengrad)", loc='center', y=1)
    gdf.plot('tilt_calc', legend=True, vmin=32.5, vmax=40.5, cmap=cmap_t,
             ax=ax_ar[0][2])

    ax_ar[1][0].set_title(
        "Azimuth (Differenz - optimal zu 180°)", loc='center', y=1)
    gdf.plot('azimuth_diff', legend=True, vmin=-0.5, vmax=5.5, cmap=cm_gyr,
             ax=ax_ar[1][0])

    ax_ar[1][1].set_title(
        "Neigung (Differenz - optimal zu Breitengrad)", loc='center', y=1)
    gdf.plot('tilt_diff', legend=True, vmin=-0.5, vmax=5.5, cmap=cm_gyr,
             ax=ax_ar[1][1])

    ax_ar[1][2].set_title(
        "Neigung (Differenz - optimal zu 36,5°)", loc='center', y=1)
    gdf.plot('tilt_diff_c', legend=True, vmin=-0.5, vmax=5.5, cmap=cm_gyr,
             ax=ax_ar[1][2])

    plt.subplots_adjust(right=1, left=0.03, bottom=0.05, top=0.95)
    return 'analyse_optimal_orientation', None


def fig_polar_plot_pv_orientation():
    plt.rcParams.update({'font.size': 14})
    key = 1129089
    path = os.path.join(cfg.get('paths', 'analysis'),
                        'pv_yield_by_orientation_c')
    fn = os.path.join(path, '{0}_combined.csv'.format(key))

    df = pd.read_csv(fn, index_col=[0, 1])
    df.reset_index(inplace=True)
    df['rel'] = df['2'] / df['2'].max()

    azimuth_opt = float(df[df['2'] == df['2'].max()]['1'])
    tilt_opt = float(df[df['2'] == df['2'].max()]['0'])
    print(azimuth_opt, tilt_opt)
    print(tilt_opt-5)
    print(df[(df['1'] == azimuth_opt+5) & (df['0'] == tilt_opt+5)])
    print(df[(df['1'] == azimuth_opt-5) & (df['0'] == tilt_opt+5)])
    print(df[(df['1'] == azimuth_opt+5) & (df['0'] == round(tilt_opt-5, 1))])
    print(df[(df['1'] == azimuth_opt-5) & (df['0'] == round(tilt_opt-5, 1))])

    # Data
    tilt = df['0']
    azimuth = df['1'] / 180 * np.pi
    colors = df['2'] / df['2'].max()

    # Colormap
    cmap = plt.get_cmap('viridis', 20)

    # Plot
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111, projection='polar')
    sc = ax.scatter(azimuth, tilt, c=colors, cmap=cmap, alpha=1, vmin=0.8)
    ax.tick_params(pad=10)

    # Colorbar
    label = "Anteil vom maximalen Ertrag"
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.75])
    fig.colorbar(sc, cax=cax, label=label, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_theta_zero_location('S', offset=0)

    # Adjust radius
    # ax.set_rmax(90)
    ax.set_rlabel_position(110)
    t_upper = tilt_opt + 5
    t_lower = tilt_opt - 5
    az_upper = azimuth_opt + 5
    az_lower = azimuth_opt - 5
    bbox_props = dict(boxstyle="round", fc="white", alpha=0.5, lw=0)
    ax.annotate(">0.996",
                xy=((az_upper-5)/180 * np.pi, t_upper),
                xytext=((az_upper+3)/180 * np.pi, t_upper+3),
                # textcoords='figure fraction',
                arrowprops=dict(facecolor='black', arrowstyle="-"),
                horizontalalignment='left',
                verticalalignment='bottom', bbox=bbox_props)

    az = (np.array([az_lower, az_lower, az_upper, az_upper, az_lower]) /
          180 * np.pi)
    t = np.array([t_lower, t_upper, t_upper, t_lower, t_lower])
    ax.plot(az, t)

    ax.set_rmax(50)
    ax.set_rmin(20)
    ax.set_thetamin(90)
    ax.set_thetamax(270)
    # Adjust margins
    plt.subplots_adjust(right=0.94, left=0, bottom=-0.2, top=1.2)
    return 'polar_plot_pv_orientation.png', None


def fig_average_weather():
    plt.rcParams.update({'font.size': 15})
    f, ax_ar = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    my_cmap = LinearSegmentedColormap.from_list('mycmap', [
        (0, '#dddddd'),
        (1 / 7, '#c946e5'),
        (2 / 7, '#ffeb00'),
        (3 / 7, '#26a926'),
        (4 / 7, '#c15c00'),
        (5 / 7, '#06ffff'),
        (6 / 7, '#f24141'),
        (7 / 7, '#1a2663')])

    weather_path = cfg.get('paths', 'coastdat')

    # Download missing weather files
    pattern = 'coastDat2_de_{0}.h5'
    for year in range(1998, 2015):
        fn = os.path.join(weather_path, pattern.format(year))
        if not os.path.isfile(fn):
            coastdat.download_coastdat_data(filename=fn, year=year)

    pattern = 'average_data_{data_type}.csv'
    dtype = 'v_wind'
    fn = os.path.join(weather_path, pattern.format(data_type=dtype))
    if not os.path.isfile(fn):
        coastdat.store_average_weather(dtype, out_file_pattern=pattern)
    df = pd.read_csv(fn, index_col=[0])
    coastdat_poly = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('coastdat', 'coastdatgrid_polygon'))
    coastdat_poly = coastdat_poly.merge(df, left_index=True, right_index=True)
    ax = coastdat_poly.plot(
        column='v_wind_avg', cmap=my_cmap, vmin=1, vmax=8, ax=ax_ar[0])
    ax = geometries.get_germany_awz_polygon().simplify(0.05).boundary.plot(
        ax=ax, color='#555555')
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    norm = Normalize(vmin=1, vmax=8)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    n_cmap.set_array(np.array([]))
    cbar = plt.colorbar(n_cmap, ax=ax, extend='both', cax=cax)
    cbar.set_label('Windgeschwindigkeit [m/s]', rotation=270, labelpad=30)

    weather_path = cfg.get('paths', 'coastdat')
    dtype = 'temp_air'
    fn = os.path.join(weather_path, pattern.format(data_type=dtype))
    if not os.path.isfile(fn):
        coastdat.store_average_weather(dtype, out_file_pattern=pattern,
                                       years=[2014, 2013, 2012])
    df = pd.read_csv(fn, index_col=[0]) - 273.15
    print(df.mean())
    coastdat_poly = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('coastdat', 'coastdatgrid_polygon'))
    coastdat_poly = coastdat_poly.merge(df, left_index=True, right_index=True)
    ax = coastdat_poly.plot(
        column='temp_air_avg', cmap='rainbow', vmin=7, vmax=12, ax=ax_ar[1])
    ax = geometries.get_germany_awz_polygon().simplify(0.05).boundary.plot(
        ax=ax, color='#555555')
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    norm = Normalize(vmin=5, vmax=12)
    n_cmap = cm.ScalarMappable(norm=norm, cmap='rainbow')
    n_cmap.set_array(np.array([]))
    cbar = plt.colorbar(n_cmap, ax=ax, extend='both', cax=cax)
    cbar.set_label('Temperatur [°C]', rotation=270, labelpad=30)
    plt.subplots_adjust(left=0, top=0.97, bottom=0.03, right=0.95, wspace=0.1)
    return 'average_weather', None


def fig_compare_electricity_profile_berlin():
    f, ax_ar = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    year = 2014
    federal_states = geometries.get_federal_states_polygon()
    bln_elec = electricity.get_electricity_demand(year)
    fs_demand = demand.get_entsoe_profile_by_region(
        federal_states, year, 'federal_states')

    bln_vatten = bln_elec.usage
    bln_vatten.name = 'Berlin Profil'
    bln_entsoe = fs_demand['BE'].multiply(
        bln_elec.usage.sum()/fs_demand['BE'].sum())
    bln_entsoe.name = 'Entsoe-Profil (skaliert)'
    bln_reegis = fs_demand['BE'].div(1000)
    bln_reegis.name = 'Entsoe-Profil (reegis)'

    ax = ax_ar[0]
    start = datetime.datetime(year, 1, 13)
    end = datetime.datetime(year, 1, 20)
    ax = bln_vatten.loc[start:end].plot(ax=ax, x_compat=True)
    ax = bln_entsoe.loc[start:end].plot(ax=ax, x_compat=True)
    ax.set_title('Winterwoche (13. - 20. Januar)')
    ax.set_xticklabels(['13', '14', '15', '16', '17', '18', '19', '20'],
                       rotation=0)
    ax.set_xlabel('Januar 2014')
    ax.set_ylabel('[GW]')

    ax = ax_ar[1]
    start = datetime.datetime(year, 7, 14)
    end = datetime.datetime(year, 7, 21)
    ax = bln_vatten.loc[start:end].plot(ax=ax, x_compat=True)
    ax = bln_entsoe.loc[start:end].plot(ax=ax, x_compat=True)
    ax.set_title('Sommerwoche (14. - 20. Juli)')
    ax.set_xticklabels(['14', '15', '16', '17', '18', '19', '20', '21'],
                       rotation=0)
    ax.set_xlabel('Juli 2014')

    ax = ax_ar[2]
    ax = bln_vatten.resample('W').mean().plot(ax=ax, legend=True,
                                              x_compat=True)
    ax = bln_entsoe.resample('W').mean().plot(ax=ax, legend=True,
                                              x_compat=True)
    bln_reegis.resample('W').mean().plot(ax=ax, legend=True, x_compat=True)

    ax.set_title('Wochenmittel - 2014')
    ax.set_xticklabels(['Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug',
                        'Sep', 'Okt', 'Nov', 'Dez', 'Jan'])
    ax.set_xlabel('')
    plt.subplots_adjust(left=0.04, top=0.92, bottom=0.11, right=0.99)

    return 'compare_electricity_profile_berlin', None


def fig_inhabitants_per_area():
    plt.rcParams.update({'font.size': 15})
    ew = inhabitants.get_ew_geometry(2017, polygon=True)
    ew['ew_area'] = ew['EWZ'].div(ew['KFL']).fillna(0)
    ew['geometry'] = ew['geometry'].simplify(0.01)
    ax = ew.plot(column='ew_area', vmax=800, cmap='cividis')
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    norm = Normalize(vmin=0, vmax=800)
    n_cmap = cm.ScalarMappable(norm=norm, cmap='cividis')
    n_cmap.set_array(np.array([]))
    cbar = plt.colorbar(n_cmap, ax=ax, extend='max', cax=cax)
    cbar.set_label('Einwohner pro km²', rotation=270, labelpad=30)
    plt.subplots_adjust(left=0, top=1, bottom=0, right=0.85)
    return 'inhabitants_per_area', None


def fig_windzones():
    path = cfg.get('paths', 'geometry')
    filename = 'windzones_germany.geojson'
    df = geometries.load(path=path, filename=filename)
    df.set_index('zone', inplace=True)
    geo_path = cfg.get('paths', 'geometry')
    geo_file = cfg.get('coastdat', 'coastdatgrid_polygon')
    coastdat_geo = geometries.load(path=geo_path, filename=geo_file)
    coastdat_geo['poly'] = coastdat_geo.geometry
    coastdat_geo['geometry'] = coastdat_geo.centroid

    points = geometries.spatial_join_with_buffer(coastdat_geo, df, 'windzone')
    polygons = points.set_geometry('poly')

    cmap_bluish = LinearSegmentedColormap.from_list(
        'bluish', [
            (0, '#8fbbd2'),
            (1, '#00317a')], 4)

    ax = polygons.plot(column='windzone', edgecolor='#666666', linewidth=0.5,
                       cmap=cmap_bluish, vmin=0.5, vmax=4.5)
    ax.set_axis_off()
    df.boundary.simplify(0.01).plot(
        edgecolor='black', alpha=1, ax=ax, linewidth=1.5)
    text = {
        "1": (9, 50),
        "2": (12, 52),
        "3": (9.8, 54),
        "4": (6.5, 54.6)}

    for t, c in text.items():
        plt.text(c[0], c[1], t, size=15,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round", alpha=.5,
                           ec=(1, 1, 1), fc=(1, 1, 1)))
    plt.subplots_adjust(left=0, top=1, bottom=0, right=1)
    return 'windzones', None


def sankey_test():

    Sankey(flows=[1, -5727/22309, -14168/22309, -1682/22309, -727/22309],
           labels=[' ', ' ', ' ', ' ', ' '],
           orientations=[-1, 1, 0, -1, 1]).finish()
    plt.title("The default settings produce a diagram like this.")
    return 'sankey_test', None


def fig_powerplants(**kwargs):
    plt.rcParams.update({'font.size': 14})
    geo = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'federalstates_polygon'))

    my_name = 'my_federal_states'  # doctest: +SKIP
    my_year = 2015  # doctest: +SKIP
    pp_reegis = powerplants.get_powerplants_by_region(geo, my_year, my_name)

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'static')
    fn_bnetza = os.path.join(data_path, cfg.get('plot_data', 'bnetza'))
    pp_bnetza = pd.read_csv(fn_bnetza, index_col=[0], skiprows=2, header=[0])

    ax = create_subplot((10, 5), **kwargs)

    see = 'sonst. erneuerb.'

    my_dict = {
        'Bioenergy': see,
        'Geothermal': see,
        'Hard coal': 'Kohle',
        'Hydro': see,
        'Lignite': 'Kohle',
        'Natural gas': 'Erdgas',
        'Nuclear': 'Nuklear',
        'Oil': 'sonstige fossil',
        'Other fossil fuels': 'sonstige fossil',
        'Other fuels': 'sonstige fossil',
        'Solar': 'Solar',
        'Waste': 'sonstige fossil',
        'Wind': 'Wind',
        'unknown from conventional': 'sonstige fossil'}

    my_dict2 = {
        'Biomasse': see,
        'Braunkohle': 'Kohle',
        'Erdgas': 'Erdgas',
        'Kernenergie': 'Nuklear',
        'Laufwasser': see,
        'Solar': 'Solar',
        'Sonstige (ne)': 'sonstige fossil',
        'Steinkohle': 'Kohle',
        'Wind': 'Wind',
        'Sonstige (ee)': see,
        'Öl': 'sonstige fossil'}

    my_colors = ['#555555', '#6c3012', '#db0b0b', '#ffde32', '#335a8a',
                 '#163e16', '#501209']

    pp_reegis.capacity_2015.unstack().to_excel('/home/uwe/shp/wasser.xls')

    pp_reegis = pp_reegis.capacity_2015.unstack().groupby(
        my_dict, axis=1).sum()

    pp_reegis.loc['AWZ'] = (
            pp_reegis.loc['N0'] + pp_reegis.loc['N1'] + pp_reegis.loc['O0'])

    pp_reegis.drop(['N0', 'N1', 'O0', 'unknown', 'P0'], inplace=True)

    pp_bnetza = pp_bnetza.groupby(my_dict2, axis=1).sum()

    ax = pp_reegis.sort_index().sort_index(1).div(1000).plot(
        kind='bar', stacked=True, position=1.1, width=0.3, legend=False,
        color=my_colors, ax=ax)
    pp_bnetza.sort_index().sort_index(1).div(1000).plot(
        kind='bar', stacked=True, position=-0.1, width=0.3, ax=ax,
        color=my_colors, alpha=0.9)
    plt.xlabel('Bundesländer / AWZ')
    plt.ylabel('Installierte Leistung [GW]')
    plt.xlim(left=-0.5)
    plt.subplots_adjust(bottom=0.17, top=0.98, left=0.08, right=0.96)

    b_sum = pp_bnetza.sum()/1000
    b_total = int(round(b_sum.sum()))
    b_ee_sum = int(
        round(b_sum.loc[['Wind', 'Solar', see]].sum()))
    b_fs_sum = int(round(b_sum.loc[
        ['Erdgas', 'Kohle', 'Nuklear', 'sonstige fossil']].sum()))
    r_sum = pp_reegis.sum()/1000
    r_total = int(round(r_sum.sum()))
    r_ee_sum = int(
        round(r_sum.loc[['Wind', 'Solar', see]].sum()))
    r_fs_sum = int(round(r_sum.loc[
        ['Erdgas', 'Kohle', 'Nuklear', 'sonstige fossil']].sum()))

    text = {
        'reegis': (2.3, 42, 'reegis'),
        'BNetzA': (3.9, 42, 'BNetzA'),
        "b_sum1": (0, 39, "gesamt"),
        "b_sum2": (2.5, 39, "{0}       {1}".format(r_total, b_total)),
        "b_fs": (0, 36, "fossil"),
        "b_fs2": (2.5, 36, " {0}         {1}".format(r_fs_sum, b_fs_sum)),
        "b_ee": (0, 33, "erneuerbar"),
        "b_ee2": (2.5, 33, " {0}         {1}".format(r_ee_sum, b_ee_sum)),
      }

    for t, c in text.items():
        plt.text(c[0], c[1], c[2], size=14, ha="left", va="center")

    b = patches.Rectangle((-0.2, 31.8), 5.7, 12, color='#cccccc')
    ax.add_patch(b)
    ax.add_patch(patches.Shadow(b, -0.05, -0.2))
    return 'vergleich_kraftwerke_reegis_bnetza', None


def fig_anteil_import_stromverbrauch_berlin(**kwargs):

    ax = create_subplot((8.5, 5), **kwargs)
    fn_csv = os.path.join(os.path.dirname(__file__), 'data', 'static',
                          'electricity_import.csv')
    df = pd.read_csv(fn_csv)
    df['Jahr'] = df['year'].astype(str)

    print('10-Jahresmittel:', df['elec_import'].sum() / df['elec_usage'].sum())

    df['Erzeugung [TWh]'] = (df['elec_usage'] - df['elec_import']).div(3600)
    df['Import [TWh]'] = df['elec_import'].div(3600)

    df['Importanteil [%]'] = df['elec_import'] / df['elec_usage'] * 100
    ax1 = df[['Jahr', 'Importanteil [%]']].plot(
        x='Jahr', linestyle='-', marker='o', secondary_y=True,
        color='#555555', ax=ax)
    df[['Jahr', 'Import [TWh]', 'Erzeugung [TWh]']].plot(
        x='Jahr', kind='bar', ax=ax1, stacked=True,
        color=['#343e58', '#aebde3'])
    ax1.set_ylim(0, 100)

    h0, l0 = ax.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax.legend(h0 + h1, l0 + l1, bbox_to_anchor=(1.06, 0.8))
    plt.subplots_adjust(right=0.74, left=0.05)
    return 'anteil_import_stromverbrauch_berlin', None


def fig_netzkapazitaet_und_auslastung_de22():
    year = 2014
    cm_gyr = LinearSegmentedColormap.from_list(
        'gyr', [
            (0, '#aaaaaa'),
            (0.0001, 'green'),
            (0.5, '#d5b200'),
            (1, 'red')])

    static = LinearSegmentedColormap.from_list(
        'static', [
            (0, 'red'),
            (0.001, '#555555'),
            (1, '#000000')])

    sets = {
        'capacity': {
            'key': 'capacity',
            'vmax': 10,
            'label_min': 0,
            'label_max': None,
            'unit': '',
            'order': 0,
            'direction': False,
            'cmap_lines': static,
            'legend': False,
            'unit_to_label': False,
            'divide': 1000,
            'decimal': 1,
            'my_legend': False,
            'part_title': 'Kapazität in MW'},
        'absolut': {
            'key': 'es1_90+_usage',
            'vmax': 8760/2,
            'label_min': 10,
            'unit': '',
            'order': 1,
            'direction': False,
            'cmap_lines': cm_gyr,
            'legend': False,
            'unit_to_label': False,
            'divide': 1,
            'my_legend': True,
            'part_title': 'Stunden mit über 90% Auslastung \n'},
    }

    my_es1 = results.load_my_es('deflex', str(year), var='de22')
    my_es2 = results.load_my_es('berlin_hp', str(year), var='de22')
    transmission = results.compare_transmission(my_es1, my_es2)

    f, ax_ar = plt.subplots(1, 2, figsize=(15, 6))
    for k, v in sets.items():
        v['ax'] = ax_ar[v.pop('order')]
        my_legend = v.pop('my_legend')
        v['ax'].set_title(v.pop('part_title'))
        plot.plot_power_lines(transmission, **v)
        if my_legend is True:
            plot.geopandas_colorbar_same_height(f, v['ax'], 0, v['vmax'],
                                                v['cmap_lines'])
        plt.title(v['unit'])
    plt.subplots_adjust(right=0.96, left=0, wspace=0, bottom=0.03, top=0.96)

    return 'netzkapazität_und_auslastung_de22', None


def fig_tespy_heat_pumps_cop():
    """From TESPy examples."""
    plt.rcParams.update({'font.size': 16})
    f, ax_ar = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 5))

    t_range = [6, 9, 12, 15, 18, 21, 24]
    q_range = np.array([120e3, 140e3, 160e3, 180e3, 200e3, 220e3])

    n = 0
    for filename in ['COP_air.csv', 'COP_water.csv']:
        fn = os.path.join(cfg.get('paths', 'data_my_reegis'), filename)
        df = pd.read_csv(fn, index_col=0)

        colors = ['#00395b', '#74adc1', '#b54036', '#ec6707',
                  '#bfbfbf', '#999999', '#010101']
        plt.sca(ax_ar[n])
        i = 0
        for t in t_range:
            plt.plot(
                q_range / 200e3, df.loc[t], '-x', Color=colors[i],
                label='$T_{resvr}$ = ' + str(t) + ' °C', markersize=7,
                linewidth=2)
            i += 1

        ax_ar[n].set_xlabel('Relative Last')

        if n == 0:
            ax_ar[n].set_ylabel('COP')
        n += 1
    plt.ylim([0, 3.2])
    plt.xlim([0.5, 1.2])
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.subplots_adjust(right=0.82, left=0.06, wspace=0.11, bottom=0.13,
                        top=0.97)
    return 'tespy_heat_pumps', None


def fig_veraenderung_energiefluesse_durch_kopplung():
    year = 2014

    cm_gyr = LinearSegmentedColormap.from_list(
        'mycmap', [
            (0, '#aaaaaa'),
            (0.01, 'green'),
            (0.5, '#d5b200'),
            (1, 'red')])

    sets = {
        'fraction': {
            'key': 'diff_2-1_avg_usage',
            'vmax': 5,
            'label_min': 1,
            'label_max': None,
            'unit': '%-Punkte',
            'order': 1,
            'direction': False,
            'cmap_lines': cm_gyr,
            'legend': False,
            'unit_to_label': False},
        'absolut': {
            'key': 'diff_2-1',
            'vmax': 500,
            'label_min': 100,
            'unit': 'GWh',
            'order': 0,
            'direction': True,
            'cmap_lines': cm_gyr,
            'legend': False,
            'unit_to_label': False},
    }

    my_es1 = results.load_my_es('deflex', str(year), var='de22')
    my_es2 = results.load_my_es('berlin_hp', str(year), var='de22')
    transmission = results.compare_transmission(my_es1, my_es2).div(1)

    f, ax_ar = plt.subplots(1, 2, figsize=(15, 6))
    for k, v in sets.items():
        v['ax'] = ax_ar[v.pop('order')]
        plot.plot_power_lines(transmission, **v)
        plot.geopandas_colorbar_same_height(f, v['ax'], 0, v['vmax'],
                                            v['cmap_lines'])
        plt.title(v['unit'])
    plt.subplots_adjust(right=0.97, left=0, wspace=0, bottom=0.03, top=0.96)

    return 'veraenderung_energiefluesse_durch_kopplung', None


def fig_model_regions():
    f, ax_ar = plt.subplots(1, 4, figsize=(11, 2.5))

    # de = deflex.geometries.deflex_regions(rmap='de17').gdf
    # de['label'] = de.representative_point()
    maps = ['de02', 'de17', 'de21', 'de22']
    offshore = {'de02': [2],
                'de17': [17],
                'de21': [19, 20, 21],
                'de22': [19, 20, 21]}

    i = 0
    for rmap in maps:
        ax = ax_ar[i]
        plot.plot_regions(deflex_map=rmap, ax=ax, offshore=offshore[rmap],
                          legend=False, simple=0.05)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        ax.axis('off')
        ax.set_title(rmap)
        i += 1

    plt.subplots_adjust(right=1, left=0, wspace=0, bottom=0, top=0.88)
    return 'model_regions', None


def fig_absolute_power_flows():
    year = 2014

    cm_gyr = LinearSegmentedColormap.from_list(
        'mycmap', [
            (0, '#aaaaaa'),
            (0.01, 'green'),
            (0.5, '#d5b200'),
            (1, 'red')])

    sets = {
        'fraction': {
            'key': 'es1',
            'vmax': 500,
            'label_min': 100,
            'label_max': None,
            'unit': 'GWh',
            'order': 0,
            'direction': True,
            'cmap_lines': cm_gyr,
            'legend': False,
            'unit_to_label': False,
            'part_title': 'es1'},
        'absolut': {
            'key': 'es2',
            'vmax': 500,
            'label_min': 100,
            'unit': 'GWh',
            'order': 1,
            'direction': True,
            'cmap_lines': cm_gyr,
            'legend': False,
            'unit_to_label': False,
            'part_title': 'es2'},
    }

    my_es1 = results.load_my_es('deflex', str(year), var='de22')
    my_es2 = results.load_my_es('berlin_hp', str(year), var='de22')
    transmission = results.compare_transmission(my_es1, my_es2).div(1)

    f, ax_ar = plt.subplots(1, 2, figsize=(15, 6))
    for k, v in sets.items():
        v['ax'] = ax_ar[v.pop('order')]
        v['ax'].set_title(v.pop('part_title'))
        plot.plot_power_lines(transmission, **v)
        plot.geopandas_colorbar_same_height(f, v['ax'], 0, v['vmax'],
                                            v['cmap_lines'])
        # v['ax'].set_title(v.pop('part_title'))
        plt.title(v['unit'])
    plt.subplots_adjust(right=0.97, left=0, wspace=0, bottom=0.03, top=0.96)

    return 'absolute_energiefluesse_vor_nach_kopplung', None


def fig_deflex_de22_polygons(**kwargs):
    ax = create_subplot((9, 7), **kwargs)

    # change for a better/worse resolution (
    simple = 0.02

    fn = os.path.join(cfg.get('paths', 'geo_plot'),
                      'region_polygon_de22_reegis.csv')

    reg_id = ['DE{num:02d}'.format(num=x+1) for x in range(22)]
    idx = [x+1 for x in range(22)]
    data = pd.DataFrame({'reg_id': reg_id}, index=idx)
    data['class'] = 0
    data.loc[[19, 20, 21], 'class'] = 1
    data.loc[22, 'class'] = 0.5
    data.loc[22, 'reg_id'] = ''

    cmap = LinearSegmentedColormap.from_list(
        'mycmap', [(0.000000000, '#badd69'),
                   (0.5, '#dd5500'),
                   (1, '#a5bfdd')])

    ax = plot.plot_regions(edgecolor='#666666', data=data, legend=False,
                           label_col='reg_id', fn=fn, data_col='class',
                           cmap=cmap, ax=ax, simple=simple)
    plt.subplots_adjust(right=1, left=0, bottom=0, top=1)

    ax.set_axis_off()
    return 'deflex_de22_polygons', None


def fig_6_x_draft1(**kwargs):

    ax = create_subplot((5, 5), **kwargs)

    my_es1 = results.load_my_es('deflex', '2014', var='de21')
    my_es2 = results.load_my_es('deflex', '2014', var='de22')
    # my_es_2 = results.load_es(2014, 'de22', 'berlin_hp')
    transmission = results.compare_transmission(my_es1, my_es2)

    # PLOTS
    transmission = transmission.div(1000)
    transmission.plot(kind='bar', ax=ax)
    return 'name_6_x', None


def fig_compare_habitants_and_heat_electricity_share(**kwargs):
    plt.rcParams.update({'font.size': 14})
    ax = create_subplot((9, 4), **kwargs)
    eb = energy_balance.get_states_balance(2014).swaplevel()
    blnc = eb.loc['total', ('electricity', 'district heating')]
    ew = pd.DataFrame(inhabitants.get_ew_by_federal_states(2014))
    res = pd.merge(blnc, ew, right_index=True, left_index=True)
    fraction = pd.DataFrame(index=res.index)
    for col in res.columns:
        fraction[col] = res[col].div(res[col].sum())

    fraction.rename(columns={'electricity': 'Anteil Strombedarf',
                             'district heating': 'Anteil Fernwärme',
                             'EWZ': 'Anteil Einwohner'},
                    inplace=True)
    fraction = fraction[['Anteil Strombedarf', 'Anteil Einwohner',
                         'Anteil Fernwärme']]
    ax = fraction.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('')
    plt.subplots_adjust(right=0.99, left=0.07, bottom=0.09, top=0.98)
    return 'compare_habitants_and_heat_electricity_share', None


def fig_district_heating_areas(**kwargs):
    ax = create_subplot((7.8, 4), **kwargs)

    # get groups of district heating systems in Berlin
    district_heating_groups = pd.DataFrame(pd.Series(
        cfg.get_dict('district_heating_systems')), columns=['name'])

    # get district heating system areas in Berlin
    distr_heat_areas = heat.get_district_heating_areas()

    # Merge main groups on map
    distr_heat_areas = distr_heat_areas.merge(
        district_heating_groups, left_on='KLASSENNAM', right_index=True)

    # Create real geometries
    distr_heat_areas = geometries.create_geo_df(distr_heat_areas)

    # Plot berlin map
    berlin_fn = os.path.join(cfg.get('paths', 'geo_berlin'), 'berlin.csv')
    berlin = geometries.create_geo_df(pd.read_csv(berlin_fn))
    ax = berlin.plot(color='#ffffff', edgecolor='black', ax=ax)

    # Plot areas of district heating system groups
    ax = distr_heat_areas.loc[
        distr_heat_areas['name'] != 'decentralised_dh'].plot(
        column='name', ax=ax, cmap='tab10')

    # Remove frame around plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axis('off')

    text = {
        "Vattenfall 1": (13.3, 52.52),
        "Vattenfall 2": (13.5, 52.535),
        "Buch": (13.47, 52.63),
        "Märkisches Viertel": (13.31, 52.61),
        "Neukölln": (13.422, 52.47),
        "BTB": (13.483, 52.443),
        "Köpenick": (13.58, 52.43),
        "Friedrichshagen": (13.653, 52.44)}

    for t, c in text.items():
        plt.text(c[0], c[1], t, size=6,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round", alpha=.5,
                           ec=(1, 1, 1), fc=(1, 1, 1)))
    plt.draw()
    return 'distric_heating_areas', None


def fig_patch_offshore(**kwargs):
    ax = create_subplot((12, 4), **kwargs)
    federal_states = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'federalstates_polygon'))
    # federal_states.drop(['P0'], inplace=True)
    mydf = powerplants.patch_offshore_wind(pd.DataFrame(), [])
    mygdf = gpd.GeoDataFrame(mydf)
    fs = federal_states.set_index('iso').loc[
        ['NI', 'SH', 'HH', 'MV', 'BB', 'BE', 'HB', 'ST', 'NW']]
    offshore = federal_states.set_index('iso').loc[['N0', 'N1', 'O0']]
    fs['geometry'] = fs['geometry'].simplify(0.01)
    offshore['geometry'] = offshore['geometry'].simplify(0.01)

    ax = fs.plot(ax=ax, facecolor='#badd69', edgecolor='#777777')
    ax = offshore.plot(ax=ax, facecolor='#ffffff', edgecolor='#777777')
    mygdf.plot(markersize=mydf.capacity, alpha=0.5, ax=ax, legend=True)

    plt.ylim(bottom=52.5)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1)
    ax.legend()
    return 'patch_offshore', None


def fig_storage_capacity(**kwargs):
    plt.rcParams.update({'font.size': 12})
    ax = create_subplot((6, 4), **kwargs)

    federal_states = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'federalstates_polygon'))
    federal_states.set_index('iso', drop=True, inplace=True)
    federal_states['geometry'] = federal_states['geometry'].simplify(0.02)
    phes = storages.pumped_hydroelectric_storage(federal_states,
                                                 'federal_states')

    fs = federal_states.merge(
        phes, left_index=True, right_index=True, how='left').fillna(0)

    fs.drop(['N0', 'N1', 'O0', 'P0'], inplace=True)
    fs['energy'] = fs['energy'].div(1000)

    ax = fs.plot(column='energy', cmap='YlGn', ax=ax)
    fs.boundary.plot(ax=ax, color='#777777')

    coords = {
        'NI': (9.7, 52.59423440995961),
        'SH': (9.8, 53.9),
        'ST': (11.559203329244966, 51.99003282648907),
        'NW': (7.580292138948966, 51.4262307721131),
        'BW': (9.073099768325736, 48.5),
        'BY': (11.5, 48.91810114600406),
        'TH': (10.9, 50.8),
        'HE': (9.018890328297207, 50.52634809768823),
        'SN': (13.3, 50.928277090542124)}

    for idx, row in fs.iterrows():
        if row['energy'] > 0:
            if row['energy'] > 10:
                color = '#dddddd'
            else:
                color = '#000000'
            plt.annotate(
                s=round(row['energy'], 1), xy=coords[idx],
                horizontalalignment='center', color=color)
    ax.set_axis_off()
    scatter = ax.collections[0]
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speicherkapazität [GWh]', rotation=270, labelpad=15)
    plt.subplots_adjust(left=0, bottom=0.05, top=0.95)
    return 'storage_capacity_by_federal_states', None


def plot_upstream():
    year = 2014
    sc = fhg_sc.load_upstream_scenario_values()
    cols = ['deflex_{0}_de21_without_berlin', 'deflex_{0}_de22_without_berlin',
            'deflex_{0}_de22', 'deflex_2014_de21']
    cols = [c.format(year) for c in cols]

    ax = sc['deflex_2014_de22', 'meritorder'].plot()
    ax = sc['deflex_2014_de22_without_berlin', 'meritorder'].plot(ax=ax)
    ax = sc['deflex_2014_de21', 'meritorder'].plot(ax=ax)
    ax = sc['deflex_2014_de21_without_berlin', 'meritorder'].plot(ax=ax)
    ax.legend()
    sc[cols].mean().unstack()[['levelized']].plot(kind='bar')
    print(sc[cols].mean().unstack()['meritorder'])
    print(sc[cols].mean().unstack()['levelized'])
    return 'upstream', None


def fig_inhabitants(**kwargs):
    plt.rcParams.update({'font.size': 15})
    ax = create_subplot((8, 6), **kwargs)
    df = pd.DataFrame()
    for year in range(2011, 2018):
        df[year] = inhabitants.get_ew_by_federal_states(year)
    df.sort_values(2017, inplace=True)
    ax = df.transpose().div(1000).plot(kind='bar', stacked=True,
                                       cmap='tab20b_r', ax=ax)
    print(df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left',
              bbox_to_anchor=(1, 1.025))
    plt.subplots_adjust(left=0.14, bottom=0.15, top=0.9, right=0.8)
    plt.ylabel("Tsd. Einwohner")
    plt.xticks(rotation=0)
    return 'inhabitants_by_ferderal_states', None


def ego_demand_plot():
    ax = create_subplot((10.7, 9))

    de = deflex.geometries.deflex_regions(rmap='de02')
    de.drop('DE02', inplace=True)
    de['geometry'] = de['geometry'].simplify(0.01)
    ax = de.plot(ax=ax, alpha=0.5, color='white', edgecolor='#000000')

    ego_demand = geometries.load_csv(cfg.get('paths', 'static_sources'),
                                     cfg.get('open_ego', 'ego_input_file'))
    ego_demand = geometries.create_geo_df(ego_demand, wkt_column='st_astext')
    ax = ego_demand.plot(markersize=0.1, ax=ax, color='#272740')

    print("Number of points: {0}".format(len(ego_demand)))

    # Remove frame around plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axis('off')

    plt.subplots_adjust(right=1, left=0, bottom=0, top=1)

    return 'open_ego_map.png', None


def fig_module_comparison():
    plt.rcParams.update({'font.size': 15})
    plt.sca(create_subplot((10.7, 5)))
    df = pd.read_csv(os.path.join(cfg.get('paths', 'data_my_reegis'),
                                  'module_feedin.csv'),
                     index_col=0)['dc_norm']
    print(df)
    print(df.sort_values())
    # df = df[df > 943]
    df.sort_values().plot(linewidth=5, ylim=(0, df.max() + 20))
    print('avg:', df.mean())
    print('std div:', df.std())
    plt.plot((0, len(df)), (df.mean(), df.mean()), 'k-')
    plt.plot((0, len(df)), (df.mean() - df.std(), df.mean() - df.std()), 'k-.')
    plt.plot((0, len(df)), (df.mean() + df.std(), df.mean() + df.std()), 'k-.')
    plt.plot((253, 253), (0,  df.max() + 20), 'k-')
    plt.plot((479, 479), (0,  df.max() + 20), 'r-')
    plt.plot((394, 394), (0,  df.max() + 20), 'r-')
    plt.plot((253, 253), (0,  df.max() + 20), 'r-')
    plt.plot((62, 62), (0,  df.max() + 20), 'r-')
    plt.text(479, 800, 'SF 160S', ha='center',
             bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5,
                   'linewidth': 0},)
    plt.text(394, 800, 'LG290N1C', ha='center',
             bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5,
                   'linewidth': 0})
    plt.text(253, 800, 'STP280S', ha='center',
             bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5,
                   'linewidth': 0})
    plt.text(62, 800, 'BP2150S', ha='center',
             bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5,
                   'linewidth': 0})
    plt.xticks(np.arange(0, len(df), 40), range(0, len(df), 40))
    plt.ylim(500, 1400)
    plt.ylabel('Volllaststunden')
    plt.xlabel('ID des Moduls')
    plt.subplots_adjust(right=0.98, left=0.09, bottom=0.12, top=0.95)
    return 'module_comparison', None


def fig_show_de21_de22_without_berlin():
    plt.rcParams.update({'font.size': 13})
    figs = ('de21', 'Berlin', 'de22', 'de21_without_berlin')

    y_annotate = {
        'de21': 10,
        'de22': 1000,
        'de21_without_berlin': 1000,
        'Berlin': 1000}

    title_str = {
        'de21': 'DE01 in de21, Jahressumme: {0} GWh',
        'de22': 'DE01 in de22, Jahressumme: {0} GWh',
        'de21_without_berlin':
            'DE01 in de21 ohne Berlin, Jahressumme: {0} GWh',
        'Berlin': 'Berlin in berlin_hp, Jahressumme: {0} GWh'}

    ax = {}
    f, ax_ar = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(15, 10))

    i = 0
    for n in range(2):
        for m in range(2):
            ax[figs[i]] = ax_ar[n, m]
            i += 1

    ol = ['lignite', 'coal', 'natural_gas', 'bioenergy', 'oil', 'other',
          'shortage']

    data_sets = {}
    period = (576, 650)

    for var in ('de21', 'de22', 'de21_without_berlin'):
        data_sets[var] = {}
        es = results.load_my_es(
            'deflex', str(2014), var='{0}'.format(var))
        bus = [b[0] for b in es.results['Main'] if
               (b[0].label.region == 'DE01') &
               (b[0].label.tag == 'heat') &
               (isinstance(b[0], solph.Bus))][0]
        data = results.reshape_bus_view(es, bus)[bus.label.region]

        results.check_excess_shortage(es)
        # if data.sum().sum() > 500000:
        #     data *= 0.5
        annual = round(data['out', 'demand'].sum().sum(), -2)
        data = data.iloc[period[0]:period[1]]
        data_sets[var]['data'] = data
        data_sets[var]['title'] = title_str[var].format(int(annual/1000))

    var = 'Berlin'
    data_sets[var] = {}
    es = results.load_my_es(
        'berlin_hp', str(2014), var='single_up_None')
    data = results.get_multiregion_bus_balance(es, 'district').groupby(
            axis=1, level=[1, 2, 3, 4]).sum()
    data.rename(columns={'waste': 'other'}, level=3, inplace=True)
    annual = round(data['out', 'demand'].sum().sum(), -2)
    data = data.iloc[period[0]:period[1]]
    data_sets[var]['data'] = data
    data_sets[var]['title'] = title_str[var].format(int(annual/1000))
    results.check_excess_shortage(es)

    i = 0
    for k in figs:
        v = data_sets[k]
        if i == 1:
            legend = True
        else:
            legend = False

        av = float(v['data'].iloc[5]['out', 'demand'].sum())
        print(float(v['data'].iloc[6]['out', 'demand'].sum()))

        a = plot.plot_bus_view(data=v['data'], ax=ax[k], legend=legend,
                               xlabel='', ylabel='Leistung [MW]',
                               title=v['title'], in_ol=ol, out_ol=['demand'],
                               smooth=False)
        a.annotate(str(int(av)), xy=(5, av), xytext=(12, av + y_annotate[k]),
                   fontsize=14,
                   arrowprops=dict(facecolor='black',
                                   arrowstyle='->',
                                   connectionstyle='arc3,rad=0.2'))
        i += 1

    plt.subplots_adjust(right=0.81, left=0.06, bottom=0.08, top=0.95,
                        wspace=0.06)
    plt.arrow(600, 600, 200, 200)
    return 'compare_district_heating_de01_without_berlin', None


def berlin_resources_time_series():
    seq = regional_results.analyse_berlin_ressources()
    types = ['lignite', 'natural_gas', 'oil', 'hard_coal', 'netto_import']
    rows = len(types)
    f, ax_ar = plt.subplots(rows, 2, sharey='row', sharex=True, figsize=(9, 6))
    i = 0
    for c in types:
        seq[[(c, 'deflex_de22'), (c, 'berlin_deflex'),
             (c, 'berlin_up_deflex')]].multiply(1000).resample('D').mean(
            ).plot(ax=ax_ar[i][0], legend=False)
        ax = seq[[(c, 'deflex_de22'), (c, 'berlin_deflex'),
                  (c, 'berlin_up_deflex')]].multiply(1000).resample('M').mean(
            ).plot(ax=ax_ar[i][1], legend=False)
        ax.set_xlim([seq.index[0], seq.index[8759]])
        ax.text(seq.index[8759], ax.get_ylim()[1]/2, NAMES[c], size=12,
                verticalalignment='center', horizontalalignment='left',
                rotation=270)
        i += 1

    for i in range(rows):
        for j in range(2):
            ax = ax_ar[i, j]
            if i == 0 and j == 0:
                ax.set_title("Tagesmittel", loc='center', y=1)
            if i == 0 and j == 1:
                ax.set_title("Monatsmittel", loc='center', y=1)

    plt.subplots_adjust(right=0.96, left=0.05, bottom=0.13, top=0.95,
                        wspace=0.06, hspace=0.2)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        new_labels.append(lab.split(',')[1][:-1])
    plt.legend(handles, new_labels, bbox_to_anchor=(0, -0.9),
               loc='lower center', ncol=3)
    return 'ressource_use_berlin_time_series', None


def fig_berlin_resources(**kwargs):
    ax = create_subplot((7.8, 4), **kwargs)

    df = regional_results.analyse_berlin_ressources_total()
    df = df.drop(df.sum().loc[df.sum() < 0.1].index, axis=1)
    color_dict = plot.get_cdict_df(df)

    ax = df.plot(kind='bar', ax=ax,
                 color=[color_dict.get(x, '#bbbbbb') for x in df.columns])
    plt.subplots_adjust(right=0.79)

    # Adapt legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        new_labels.append(NAMES[label])
    plt.legend(handles, new_labels, bbox_to_anchor=(1.3, 1),
               loc='upper right', ncol=1)

    # Adapt xticks
    locs, labels = plt.xticks()
    new_labels = []
    for label in labels:
        if 'up' in label.get_text():
            new_labels.append(label.get_text().replace('up_', 'up_\n'))
        else:
            new_labels.append(label.get_text().replace('_', '_\n'))
    plt.xticks(locs, new_labels, rotation=0)

    plt.ylabel('Energiemenge 2014 [TWh]')
    plt.subplots_adjust(right=0.78, left=0.08, bottom=0.12, top=0.98)
    return 'resource_use_berlin_reduced', None


def fig_import_export_100PRZ_region():
    plt.rcParams.update({'font.size': 13})
    f, ax_ar = plt.subplots(1, 3, figsize=(15, 6))

    myp = '/home/uwe/express/reegis/scenarios_lux/friedrichshagen/results_cbc'

    my_filenames = [x for x in os.listdir(myp) if '.esys' in x and '_pv' in x]

    bil = pd.DataFrame()
    expdf = pd.Series()
    for mf in sorted(my_filenames):
        my_fn = os.path.join(myp, mf)
        my_es = results.load_es(my_fn)
        res = my_es.results['param']
        wind = int(round(
            [res[w]['scalars']['nominal_value'] for w in res
             if w[0].label.subtag == 'Wind' and w[1] is not None][0]))
        solar = int(round(
            [res[w]['scalars']['nominal_value'] for w in res
             if w[0].label.subtag == 'Solar' and w[1] is not None][0]))
        key = 'w {0:02}, pv {1:02}'.format(wind, solar)
        my_df = results.get_multiregion_bus_balance(my_es)
        imp = my_df['FHG', 'in', 'import', 'electricity', 'all'].div(1000)
        exp = my_df['FHG', 'out', 'export', 'electricity', 'all'].div(1000)
        demand = my_df['FHG', 'out', 'demand', 'electricity', 'all'].div(1000)
        expdf[key] = float(exp.sum())
        print('Autarkie:', (1 - float(exp.sum()) / demand.sum()) * 100, '%')
        if wind == 0:
            bil['export'] = exp.resample('M').sum()
            bil['import'] = imp.resample('M').sum()
            ax_ar[1] = bil.plot(
                ax=ax_ar[1], drawstyle="steps-mid", linewidth=2)
            ax_ar[1].set_xlabel('Wind: 0 MW, PV: 67 MWp')
            ax_ar[1].set_ylim(0, 7)
            ax_ar[1].legend(loc='upper left')
        if solar == 0:
            bil['export'] = exp.resample('M').sum()
            bil['import'] = imp.resample('M').sum()
            ax_ar[2] = bil.plot(
                ax=ax_ar[2], drawstyle="steps-mid", linewidth=2)
            ax_ar[2].set_xlabel('Wind: 39 MW, PV: 0 MWp')
            ax_ar[2].set_ylim(0, 7)
    ax_ar[0] = expdf.sort_index().plot(kind='bar', ax=ax_ar[0])
    ax_ar[0].set_ylabel('Energie [GWh]')
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.2, top=0.96)
    return 'import_export_100PRZ_region', None


def fig_import_export_emissions_100PRZ_region():
    f, ax_ar = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(15, 6))
    my_filename = 'market_clearing_price_{0}_{1}.csv'
    my_path = os.path.join(cfg.get('paths', 'scenario'), 'deflex')
    up_fn = os.path.join(my_path, my_filename)
    up_df = pd.read_csv(up_fn.format(2014, 'cbc'), index_col=[0],
                        header=[0, 1, 2])
    myp = '/home/uwe/express/reegis/scenarios_lux/friedrichshagen/results_cbc'

    my_filenames = [x for x in os.listdir(myp) if '.esys' in x and '_pv' in x]

    my_list = [x for x in up_df.columns.get_level_values(1).unique()
               if 'f10' in x or 'f15' in x or 'f20' in x]
    my_list = [x for x in my_list if 'de21' in x]
    my_list = [x for x in my_list if 'Li1_HP0_' in x]


    bil = pd.DataFrame()
    print(up_df.columns.get_level_values(2).unique())
    up_dict = {}
    for t1 in ['emission', 'emission_avg', 'emission_max']:
        up_dict[t1] = {}
        for up1 in my_list:
            up_dict[t1][up1] = pd.DataFrame()

    for mf in sorted(my_filenames):

        my_fn = os.path.join(myp, mf)
        my_es = results.load_es(my_fn)
        res = my_es.results['param']
        wind = int(round(
            [res[w]['scalars']['nominal_value'] for w in res
             if w[0].label.subtag == 'Wind' and w[1] is not None][0]))
        solar = int(round(
            [res[w]['scalars']['nominal_value'] for w in res
             if w[0].label.subtag == 'Solar' and w[1] is not None][0]))
        key = 'w {0:02}, pv {1:02}'.format(wind, solar)
        my_df = results.get_multiregion_bus_balance(my_es)
        imp = my_df['FHG', 'in', 'import', 'electricity', 'all']
        exp = my_df['FHG', 'out', 'export', 'electricity', 'all']
        for t2 in ['emission', 'emission_avg', 'emission_max']:
            for up in my_list:
                prc = up_df['deflex_cbc', up, t2]
                up_dict[t2][up].loc[key, 'import'] = (
                        (imp * prc).sum()/imp.sum()/prc.mean())
                up_dict[t2][up].loc[key, 'export'] = (
                        (exp * prc).sum()/exp.sum()/prc.mean())
    n2 = 0
    for k1, v1 in up_dict.items():
        n1 = 0
        for k2, v2 in v1.items():
            print(k1, k2, n1, n2)
            v2.sort_index().plot(kind='bar', ax=ax_ar[n2, n1], legend=False) 
            ax_ar[n2, n1].set_title("{0}, {1}".format(k1, k2))
            n1 += 1
        n2 += 1
    plt.legend()
    return 'import_export_emission_100PRZ_region', None


def fig_import_export_costs_100PRZ_region():
    f, ax_ar = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(15, 6))
    my_filename = 'market_clearing_price_{0}_{1}.csv'
    my_path = os.path.join(cfg.get('paths', 'scenario'), 'deflex')
    up_fn = os.path.join(my_path, my_filename)
    up_df = pd.read_csv(up_fn.format(2014, 'cbc'), index_col=[0],
                        header=[0, 1, 2])
    myp = '/home/uwe/express/reegis/scenarios_lux/friedrichshagen/results_cbc'

    my_filenames = [x for x in os.listdir(myp) if '.esys' in x and '_pv' in x]

    my_list = [x for x in up_df.columns.get_level_values(1).unique()
               if 'f10' in x or 'f15' in x or 'f20' in x]
    my_list = [x for x in my_list if 'de21' in x]
    new_list = [x for x in up_df.columns.get_level_values(1).unique()
               if 'no' not in x]
    my_list += new_list

    bil = pd.DataFrame()
    for mf in sorted(my_filenames):

        my_fn = os.path.join(myp, mf)
        my_es = results.load_es(my_fn)
        res = my_es.results['param']
        wind = int(round(
            [res[w]['scalars']['nominal_value'] for w in res
             if w[0].label.subtag == 'Wind' and w[1] is not None][0]))

        my_df = results.get_multiregion_bus_balance(my_es)
        imp = my_df['FHG', 'in', 'import', 'electricity', 'all']
        exp = my_df['FHG', 'out', 'export', 'electricity', 'all']

        pr = pd.DataFrame()
        my_import = pd.DataFrame()
        my_export = pd.DataFrame()
        for up in my_list:
            prc = up_df['deflex_cbc', up, 'mcp']
            pr.loc[up, 'mean'] = prc.mean()
            pr.loc[up, 'import_s'] = (imp * prc).sum()/imp.sum()/prc.mean()
            pr.loc[up, 'export_s'] = (exp * prc).sum()/exp.sum()/prc.mean()

            mean = (exp * prc).sum()/exp.sum()/prc.mean() - (imp * prc).sum()/imp.sum()/prc.mean()
            pr.loc[up, 'diff'] = mean * -1

            my_import[up] = imp.multiply(prc).resample('M').sum()
            my_export[up] = exp.multiply(prc).resample('M').sum()
        # if wind == 0:
        #     ax_ar[0] = pr.plot(kind='bar', secondary_y=['mean'], ax=ax_ar[0])
            # ax_ar[0].right_ax.set_ylim(0, 1120)
        if wind == 27:
            ax_ar = pr.plot(kind='bar', secondary_y=['mean'], ax=ax_ar)
            # ax_ar[1].right_ax.set_ylim(0, 1120)

    return 'import_export_costs_100PRZ_region', None


def plot_figure(number, save=False, path=None, show=False, **kwargs):

    filename, fig_show = get_number_name()[number](**kwargs)

    if fig_show is not None:
        show = fig_show

    if '.' not in filename:
        filename = filename + '.svg'

    if save is True:
        if path is None:
            path = ''
        fn = os.path.join(path, filename)
        logging.info("Save figure as {0}".format(fn))
        plt.savefig(fn)
    logging.info('Plot')
    if show is True or save is not True:
        plt.show()


def plot_all(save=False, path=None, show=False, **kwargs):
    for number in get_number_name().keys():
        plot_figure(number, save=save, path=path, show=show, **kwargs)


def get_number_name():
    return {
            '3.5': fig_tespy_heat_pumps_cop,
            '4.1': fig_patch_offshore,
            '4.4a': fig_inhabitants,
            '4.4b': fig_inhabitants_per_area,
            '4.5': fig_average_weather,
            # '4.6': 'strahlungsmittel',
            '6.3': fig_berlin_resources,
            '6.4': berlin_resources_time_series,

            '4.7': fig_module_comparison,
            '4.8': fig_analyse_multi_files,
            '4.9': fig_polar_plot_pv_orientation,
            '4.10': fig_windzones,

            '3.0': ego_demand_plot,
            '3.1': fig_model_regions,
            '6.0': fig_anteil_import_stromverbrauch_berlin,
            '6.1': fig_veraenderung_energiefluesse_durch_kopplung,
            '6.2': fig_deflex_de22_polygons,
            '6.3a': plot_upstream,
            '6.x': fig_6_x_draft1,
            '5.3': fig_district_heating_areas,
            '4.1x': fig_compare_habitants_and_heat_electricity_share,
            '6.4a': fig_show_de21_de22_without_berlin,
            '6.6': berlin_resources_time_series,
            '6.7': fig_netzkapazitaet_und_auslastung_de22,
            '6.8': fig_absolute_power_flows,
            '0.1': fig_windzones,
            '0.3': fig_powerplants,
            '0.4': fig_storage_capacity,

            '0.8': fig_compare_electricity_profile_berlin,
            '1.1': fig_polar_plot_pv_orientation,
            '1.2': fig_analyse_multi_files,
            '1.3': fig_compare_full_load_hours,
            '1.4': fig_compare_feedin_wind,
            '1.7': fig_import_export_100PRZ_region,
            '1.8': fig_import_export_costs_100PRZ_region,
            '0.2': fig_import_export_emissions_100PRZ_region,
        }


if __name__ == "__main__":
    logger.define_logging(screen_level=logging.INFO)
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__),
                    os.path.dirname(deflex.__file__)])
    cfg.tmp_set('results', 'dir', 'results_cbc')
    cfg.tmp_set('paths', 'scenario', "/home/uwe/data/reegis/scenarios_lux/")
    p = '/home/uwe/reegis/figures'
    # plot_all(show=True)
    plot_figure('6.4', save=True, show=True, path=p)
