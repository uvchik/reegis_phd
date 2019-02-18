import os
import logging
import pandas as pd
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

from berlin_hp import electricity

from reegis import config as cfg
from reegis import energy_balance
from reegis import inhabitants
from reegis import geometries
from reegis import powerplants
from reegis import storages
from reegis import demand
from my_reegis import data_analysis
from berlin_hp import heat


NAMES = {
    'lignite': 'Braunkohle',
    'natural_gas': 'Gas',
    'oil': 'Öl',
    'hard_coal': 'Steinkohle',
    'netto_import': 'Stromimport',
    'other': 'sonstige'
}


def create_subplot(default_size, **kwargs):
    size = kwargs.get('size', default_size)
    return plt.figure(figsize=size).add_subplot(1, 1, 1)


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
    return 'polar_plot_pv_orientation_c.png', None


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
    f = 'average_data_v_wind.csv'
    fn = os.path.join(weather_path, f)
    df = pd.read_csv(fn, index_col=[0])
    coastdat = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('coastdat', 'coastdatgrid_polygon'))
    coastdat = coastdat.merge(df, left_index=True, right_index=True)
    ax = coastdat.plot(
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
    f = 'average_data_temp_air.csv'
    fn = os.path.join(weather_path, f)
    df = pd.read_csv(fn, index_col=[0]) - 273.15
    coastdat = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('coastdat', 'coastdatgrid_polygon'))
    coastdat = coastdat.merge(df, left_index=True, right_index=True)
    ax = coastdat.plot(
        column='v_wind_avg', cmap='rainbow', vmin=7, vmax=12, ax=ax_ar[1])
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
    return 'inhabitants_per_area2', None


def windzones():
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
    geo = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'federalstates_polygon'))
    geo.set_index('iso', drop=True, inplace=True)

    my_name = 'my_federal_states'  # doctest: +SKIP
    my_year = 2015  # doctest: +SKIP
    pp_reegis = powerplants.get_powerplants_by_region(geo, my_year, my_name)

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'static')
    fn_bnetza = os.path.join(data_path, cfg.get('plot_data', 'bnetza'))
    pp_bnetza = pd.read_csv(fn_bnetza, index_col=[0], skiprows=2, header=[0])

    ax = create_subplot((8.5, 5), **kwargs)

    my_dict = {
        'Bioenergy': 'sonstige erneuerbar',
        'Geothermal': 'sonstige erneuerbar',
        'Hard coal': 'Kohle',
        'Hydro': 'sonstige erneuerbar',
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
        'Biomasse': 'sonstige erneuerbar',
        'Braunkohle': 'Kohle',
        'Erdgas': 'Erdgas',
        'Kernenergie': 'Nuklear',
        'Laufwasser': 'sonstige erneuerbar',
        'Solar': 'Solar',
        'Sonstige (ne)': 'sonstige fossil',
        'Steinkohle': 'Kohle',
        'Wind': 'Wind',
        'Sonstige (ee)': 'sonstige erneuerbar',
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
    plt.subplots_adjust(bottom=0.15, top=0.98, left=0.08, right=0.96)

    b_sum = pp_bnetza.sum()/1000
    b_total = int(round(b_sum.sum()))
    b_ee_sum = int(
        round(b_sum.loc[['Wind', 'Solar', 'sonstige erneuerbar']].sum()))
    b_fs_sum = int(round(b_sum.loc[
        ['Erdgas', 'Kohle', 'Nuklear', 'sonstige fossil']].sum()))
    r_sum = pp_reegis.sum()/1000
    r_total = int(round(r_sum.sum()))
    r_ee_sum = int(
        round(r_sum.loc[['Wind', 'Solar', 'sonstige erneuerbar']].sum()))
    r_fs_sum = int(round(r_sum.loc[
        ['Erdgas', 'Kohle', 'Nuklear', 'sonstige fossil']].sum()))

    text = {
        'reegis': (2.3, 42, 'reegis'),
        'BNetzA': (3.8, 42, 'BNetzA'),
        "b_sum1": (0, 39, "gesamt"),
        "b_sum2": (2.5, 39, "{0}       {1}".format(r_total, b_total)),
        "b_fs": (0, 36, "fossil"),
        "b_fs2": (2.5, 36, " {0}         {1}".format(r_fs_sum, b_fs_sum)),
        "b_ee": (0, 33, "erneuerbar"),
        "b_ee2": (2.5, 33, " {0}         {1}".format(r_ee_sum, b_ee_sum)),
      }

    for t, c in text.items():
        plt.text(c[0], c[1], c[2], size=12, ha="left", va="center")

    b = patches.Rectangle((-0.2, 31.8), 5.7, 11.8, color='#cccccc')
    ax.add_patch(b)
    ax.add_patch(patches.Shadow(b, -0.05, -0.2))
    return 'vergleich_kraftwerke_reegis_bnetza', None


def fig_6_0(**kwargs):

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


def fig_power_lines():
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


def fig_6_1():
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
                          legend=False)
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


def fig_regionen(**kwargs):
    ax = create_subplot((9, 7), **kwargs)
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
                           cmap=cmap, ax=ax)
    plt.subplots_adjust(right=1, left=0, bottom=0, top=1)

    ax.set_axis_off()
    return 'deflex_berlin_geometrien', None


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


def fig_4_1(**kwargs):

    ax = create_subplot((7, 4), **kwargs)
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
    return 'name_4_1', None


def figure_district_heating_areas(**kwargs):
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
    return 'ew_fw_elec_share', None


def fig_patch_offshore(**kwargs):
    ax = create_subplot((12, 4), **kwargs)
    federal_states = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'federalstates_polygon'))
    # federal_states.drop(['P0'], inplace=True)
    mydf = powerplants.patch_offshore_wind(pd.DataFrame(), get_patch=True)

    fs = federal_states.sort_values('SN_L').reset_index(drop=True)
    fs.loc[fs.index < 16, 'c'] = 0

    fs.drop([19], inplace=True)

    land = LinearSegmentedColormap.from_list(
        'mycmap', [(0, '#badd69'), (1, '#ffffff')], 2)
    fs['geometry'] = fs['geometry'].simplify(0.01)

    ax = fs.fillna(1).plot(ax=ax, column='c', edgecolor='#777777', cmap=land)
    mydf.plot(markersize=mydf.capacity, alpha=0.5, ax=ax, legend=True)
    plt.ylim(bottom=52.5)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1)
    ax.legend()
    return 'patch_offshore2', None


def fig_storage_capacity(**kwargs):
    ax = create_subplot((6, 4), **kwargs)

    federal_states = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'federalstates_polygon'))
    federal_states.set_index('iso', drop=True, inplace=True)
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
    # geo = reegis.geometries.load(
    #     cfg.get('paths', 'geometry'),
    #     cfg.get('geometry', 'federalstates_polygon'))
    # geo.set_index('iso', drop=True, inplace=True)
    # name = 'federal_states'
    # df = pd.DataFrame()
    # for year in range(2011, 2018):
    #     df[year] = get_ew_by_region(year, geo, name=name)
    # df.to_excel('/home/uwe/shp/einw.xls')
    df = pd.read_excel('/home/uwe/shp/einw.xls', index_col=[0])

    df.sort_values(2017, inplace=True)
    df.drop(['N0', 'N1', 'O0', 'P0'], inplace=True)
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

    return 'open_ego_map', None


def show_de21_de22_without_berlin():
    figs = ('de21', 'Berlin', 'de22', 'de21_without_berlin')

    y_annotate = {
        'de21': 50,
        'de22': 1000,
        'de21_without_berlin': 1000,
        'Berlin': 1000}

    title_str = {
        'de21': 'Region: DE01 in de21, Jahressumme: {0} GWh',
        'de22': 'Region: DE01 in de22, Jahressumme: {0} GWh',
        'de21_without_berlin':
            'Region: DE01 in de21 ohne Berlin, Jahressumme: {0} GWh',
        'Berlin': 'Region: Berlin in berlin_hp, Jahressumme: {0} GWh'}

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
        data_sets[var]['title'] = title_str[var].format(int(annual))

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
    data_sets[var]['title'] = title_str[var].format(int(annual))
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
                   arrowprops=dict(facecolor='black',
                                   arrowstyle='->',
                                   connectionstyle='arc3,rad=0.2'))
        i += 1

    plt.subplots_adjust(right=0.84, left=0.06, bottom=0.08, top=0.95,
                        wspace=0.06)
    plt.arrow(600, 600, 200, 200)
    return 'compare_district_heating_de01_without_berlin', None


def berlin_resources_time_series():
    seq = regional_results.analyse_berlin_ressources()
    f, ax_ar = plt.subplots(5, 2, sharey='row', sharex=True, figsize=(9, 6))
    i = 0
    for c in ['lignite', 'natural_gas', 'oil', 'hard_coal', 'netto_import']:
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

    for i in range(5):
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


def berlin_resources(**kwargs):
    ax = create_subplot((7.8, 4), **kwargs)

    df = regional_results.analyse_berlin_ressources_total()

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
    return 'resource_use_berlin', None


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
            '3.0': ego_demand_plot,
            '3.1': fig_model_regions,
            '6.0': fig_6_0,
            '6.1': fig_6_1,
            '6.2': fig_regionen,
            '6.3': plot_upstream,
            '6.x': fig_6_x_draft1,
            '5.3': figure_district_heating_areas,
            '4.1': fig_4_1,
            '6.4': show_de21_de22_without_berlin,
            '6.5': berlin_resources,
            '6.6': berlin_resources_time_series,
            '6.7': fig_power_lines,
            '6.8': fig_absolute_power_flows,
            '0.1': windzones,
            '0.3': fig_powerplants,
            '0.4': fig_storage_capacity,
            '0.5': fig_patch_offshore,
            '0.6': fig_inhabitants,
            '0.7': fig_inhabitants_per_area,
            '0.8': fig_compare_electricity_profile_berlin,
            '0.9': fig_average_weather,
            '1.1': fig_polar_plot_pv_orientation,
            '0.2': fig_analyse_multi_files,
        }


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__),
                    os.path.dirname(deflex.__file__)])
    cfg.tmp_set('results', 'dir', 'results_cbc')
    p = '/home/uwe/chiba/Promotion/Monographie/figures/'
    # plot_all(show=True)
    plot_figure('0.1', save=True, show=True, path=p)
