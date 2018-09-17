import os
import logging
import pandas as pd
import geopandas as gpd
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


def create_subplot(default_size, **kwargs):
    size = kwargs.get('size', default_size)
    return plt.figure(figsize=size).add_subplot(1, 1, 1)


def fig_6_1(**kwargs):

    size_6_1 = kwargs.get('size', (10, 10))

    my_es = results.load_es(2013, 'de21', 'deflex')
    my_es_2 = results.load_es(2013, 'de21', 'berlin_hp')
    transmission = results.compare_transmission(my_es, my_es_2)

    key = gui.get_choice(list(transmission.columns),
                         "Plot transmission lines", "Choose data column.")
    vmax = max([abs(transmission[key].max()), abs(transmission[key].min())])
    units = {'es1': 'GWh', 'es2': 'GWh', 'diff_2-1': 'GWh', 'fraction': '%'}
    results.plot_power_lines(transmission, key, vmax=vmax/5, unit=units[key],
                             size=size_6_1)


def fig_6_x_draft1(**kwargs):

    ax = create_subplot((5, 5), **kwargs)

    my_es = results.load_es(2014, 'de21', 'deflex')
    my_es_2 = results.load_es(2014, 'de21', 'berlin_hp')
    transmission = results.compare_transmission(my_es, my_es_2)

    # PLOTS
    transmission = transmission.div(1000)
    transmission.plot(kind='bar', ax=ax)


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


def plot_figure(number, filename=None, show=False, **kwargs):

    number_name = {
        '6.1': fig_6_1,
        '6.x': fig_6_x_draft1,
        '5.3': figure_district_heating_areas,
        '4.1': fig_4_1,
    }

    number_name[number](**kwargs)

    if filename is not None:
        logging.info("Save figure as {0}".format(filename))
        plt.savefig(filename)

    if show is True or filename is None:
        plt.show()


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__)])
    path = '/home/uwe/git_local/monographie/figures/'
    # fn = os.path.join(path, 'ew_fw_elec_share' + '.pdf')
    fn = None
    plot_figure('5.3', filename=fn)
