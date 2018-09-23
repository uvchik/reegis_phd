import os
import logging
import pandas as pd
import geopandas as gpd
import my_reegis
import results
from oemof.tools import logger
import berlin_hp
import reegis_tools.config as cfg
from matplotlib import pyplot as plt
import reegis_tools.gui as gui
from reegis_tools import energy_balance
from reegis_tools import inhabitants
from reegis_tools import geometries
from berlin_hp import heat
from matplotlib.colors import LinearSegmentedColormap
from my_reegis import friedrichshagen_scenarios as fhg_sc


def create_subplot(default_size, **kwargs):
    size = kwargs.get('size', default_size)
    return plt.figure(figsize=size).add_subplot(1, 1, 1)


def fig_6_0(**kwargs):
    ax = create_subplot((8.5, 5), **kwargs)
    fn_csv = os.path.join(os.path.dirname(__file__), 'data',
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
    return 'anteil_import_stromverbrauch_berlin'


def fig_6_1(**kwargs):

    size_6_1 = kwargs.get('size', (10, 10))
    year = 2014
    my_es = results.load_es('deflex', str(year), var='de21')
    my_es_2 = results.load_es('berlin_hp', str(year), var='de21')
    transmission = results.compare_transmission(my_es, my_es_2)

    print(transmission)
    print(transmission.columns)

    key = gui.get_choice(list(transmission.columns),
                         "Plot transmission lines", "Choose data column.")
    vmax = max([abs(transmission[key].max()), abs(transmission[key].min())])
    units = {'es1': 'GWh', 'es2': 'GWh', 'diff_2-1': '', 'fraction': '%',
             'es1_max': 'GW', 'es2_max': 'GW', 'diff_2-1_max': 'GW',
             'es1_abs': 'GWh', 'es2_abs': 'GWh', 'diff_2-1_abs': 'GWh'}
    if 'abs' in key:
        direct = False
    else:
        direct = True

    results.plot_power_lines(transmission, key, vmax=20, unit=units[key],
                             size=size_6_1, direction=direct, label_max=4.9)
    return 'name_6_1'


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

    ax = results.plot_regions(edgecolor='#666666', data=data, legend=False,
                              label_col='reg_id', fn=fn, column='class',
                              cmap=cmap, ax=ax)
    plt.subplots_adjust(right=1, left=0, bottom=0, top=1)

    # Remove frame around plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axis('off')
    return 'deflex_berlin_geometrien'


def fig_6_x_draft1(**kwargs):

    ax = create_subplot((5, 5), **kwargs)

    my_es = results.load_es(2014, 'de22', 'deflex')
    my_es_2 = results.load_es(2014, 'de22', 'berlin_hp')
    transmission = results.compare_transmission(my_es, my_es_2)

    # PLOTS
    transmission = transmission.div(1000)
    transmission.plot(kind='bar', ax=ax)
    return 'name_6_x'


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
    return 'name_4_1'


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
    return 'ew_fw_elec_share'


def plot_upstream(**kwargs):
    year = 2014
    sc = fhg_sc.load_upstream_scenario_values()
    cols = ['deflex_{0}_de21_without_berlin', 'deflex_{0}_de22_without_berlin',
            'deflex_{0}_de22', 'deflex_2014_de21']
    cols = [c.format(year) for c in cols]

    ax = sc['deflex_2014_de22', 'meritorder'].plot()
    ax = sc['deflex_2014_de22_without_berlin', 'meritorder'].plot(ax=ax)
    ax = sc['deflex_2014_de21', 'meritorder'].plot(ax=ax)
    # ax = sc['deflex_2014_de21_without_berlin', 'meritorder'].plot(ax=ax)
    ax.legend()
    sc[cols].mean().unstack()[['levelized']].plot(kind='bar')
    print(sc[cols].mean().unstack()['meritorder'])
    print(sc[cols].mean().unstack()['levelized'])


def plot_figure(number, save=False, path=None, show=False, **kwargs):

    number_name = {
        '6.0': fig_6_0,
        '6.1': fig_6_1,
        '6.2': fig_regionen,
        '6.3': plot_upstream,
        '6.x': fig_6_x_draft1,
        '5.3': figure_district_heating_areas,
        '4.1': fig_4_1,
    }

    filename = number_name[number](**kwargs)

    if save is True:
        if path is None:
            fn = filename + '.pdf'
        else:
            fn = os.path.join(path, filename + '.pdf')
        logging.info("Save figure as {0}".format(fn))
        plt.savefig(fn)

    if show is True or save is not True:
        plt.show()


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__),
                    os.path.dirname(my_reegis.__file__)])
    p = '/home/uwe/git_local/monographie/figures/'
    plot_figure('6.3', save=False, show=True, path=p)
