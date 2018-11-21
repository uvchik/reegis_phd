import os
import logging
import pandas as pd
import my_reegis
from my_reegis import results
from my_reegis import reegis_plot as plot
from my_reegis import regional_results
from oemof.tools import logger
from oemof import solph
import berlin_hp
import deflex
import reegis_tools.config as cfg
from matplotlib import pyplot as plt
from reegis_tools import energy_balance
from reegis_tools import inhabitants
from reegis_tools import geometries
from berlin_hp import heat
from matplotlib.colors import LinearSegmentedColormap
from my_reegis import friedrichshagen_scenarios as fhg_sc
from matplotlib.sankey import Sankey
# import reegis_tools.gui as gui


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


def sankey_test():

    Sankey(flows=[1, -5727/22309, -14168/22309, -1682/22309, -727/22309],
           labels=[' ', ' ', ' ', ' ', ' '],
           orientations=[-1, 1, 0, -1, 1]).finish()
    plt.title("The default settings produce a diagram like this.")
    return 'sankey_test'


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
    return 'anteil_import_stromverbrauch_berlin'


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

    return 'netzkapazität_und_auslastung_de22'


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

    return 'veraenderung_energiefluesse_durch_kopplung'


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
    return 'model_regions'


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
        plot.plot_power_lines(transmission, **v)
        plot.geopandas_colorbar_same_height(f, v['ax'], 0, v['vmax'],
                                            v['cmap_lines'])
        v['ax'].set_title(v.pop('part_title'))
        plt.title(v['unit'])
    plt.subplots_adjust(right=0.97, left=0, wspace=0, bottom=0.03, top=0.96)

    return 'absolute_energiefluesse_vor_nach_kopplung'


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

    my_es1 = results.load_my_es('deflex', '2014', var='de21')
    my_es2 = results.load_my_es('deflex', '2014', var='de22')
    # my_es_2 = results.load_es(2014, 'de22', 'berlin_hp')
    transmission = results.compare_transmission(my_es1, my_es2)

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
    return 'upstream'


def ego_demand_plot():
    ax = create_subplot((10.7, 9))

    de = deflex.geometries.deflex_regions(rmap='de02')
    de.gdf.drop('DE02', inplace=True)
    ax = de.gdf.plot(ax=ax, alpha=0.5, color='white', edgecolor='#000000')

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

    return 'open_ego_map'


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

        results.check_excess_shortage(es.results['Main'])
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
    results.check_excess_shortage(es.results['Main'])

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
    return 'compare_district_heating_de01_without_berlin'


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
    return 'ressource_use_berlin_time_series'


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
    return 'resource_use_berlin'


def plot_figure(number, save=False, path=None, show=False, **kwargs):

    number_name = {
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
                    os.path.dirname(my_reegis.__file__),
                    os.path.dirname(deflex.__file__)])
    p = '/home/uwe/git_local/monographie/figures/'
    plot_figure('3.0', save=True, show=True, path=p)
