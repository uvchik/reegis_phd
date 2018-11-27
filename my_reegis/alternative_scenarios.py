# Python libraries
import os
import logging
from datetime import datetime
# import time
# import traceback
# from shutil import copyfile

# oemof packages
from oemof.tools import logger
# from oemof import solph

# internal modules
import reegis_tools.config as cfg
# import reegis_tools.scenario_tools
import deflex
from deflex import inhabitants
import berlin_hp
from my_reegis import main
# import my_reegis
# from my_reegis import results as sys_results


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def reduce_power_plants(sc, nuclear=None, lignite=None, hard_coal=None):
    sc.table_collection['transformer'] = sc.table_collection[
        'transformer'].swaplevel(axis=1).sort_index(axis=1)
    # remove nuclear power (by law)

    if nuclear is not None:
        if nuclear == 0:
            sc.table_collection['transformer'].drop(
                'nuclear', axis=1, inplace=True)
        else:
            sc.table_collection['transformer']['nuclear'] = (
                sc.table_collection['transformer']['nuclear'].multiply(
                    nuclear))

    if lignite is not None:
        # remove have lignite (climate change)
        sc.table_collection['transformer']['lignite'] = (
            sc.table_collection['transformer']['lignite'].multiply(lignite))

    if hard_coal is not None:
        # remove have lignite (climate change)
        sc.table_collection['transformer']['hard coal'] = (
            sc.table_collection['transformer']['hard coal'].multiply(
                hard_coal))

    sc.table_collection['transformer'] = sc.table_collection[
        'transformer'].swaplevel(axis=1).sort_index(axis=1)


def more_heat_pumps(sc, heat_pump_fraction, cop):
    abs_decentr_heat = (
        sc.table_collection['time_series']['DE_demand'].sum(axis=1))
    heat_pump = abs_decentr_heat * heat_pump_fraction
    sc.table_collection['time_series']['DE_demand'] *= (1 - heat_pump_fraction)

    inhab = inhabitants.get_ew_by_deflex(2014)
    inhab_fraction = inhab.div(inhab.sum())

    for region in inhab_fraction.index:
        if inhab_fraction.loc[region] > 0:
            sc.table_collection['time_series'][
                (region, 'electrical_load')] += (
                    inhab_fraction.loc[region] * heat_pump.div(cop))


def increase_re_share(sc, factor):
    t = sc.table_collection['volatile_source']
    for region in t.columns.get_level_values(0).unique():
        for vs in t[region].columns:
            t[region, vs] += t[region, vs] * factor


def add_simple_gas_turbine(sc, nom_val, efficiency=0.39):
    sc.table_collection['commodity_source'][('DE', 'natural gas add')] = (
        sc.table_collection['commodity_source'][('DE', 'natural gas')])

    for region in nom_val.keys():
        sc.table_collection['transformer'].loc[
            'efficiency', (region, 'natural gas add')] = efficiency
        sc.table_collection['transformer'].loc[
            'capacity', (region, 'natural gas add')] = (
                int(5 * round((nom_val[region] / 100 + 2.5) / 5) * 100))
        sc.table_collection['transformer'].loc[
            'limit_elec_pp', (region, 'natural gas add')] = 'inf'


def create_scenario_XX_Nc00_Li05_HP02(subpath='new', factor=0.0):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = main.load_deflex_scenario(2014, create_scenario=False)

    increase_re_share(sc, factor)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    sub = 'XX_Nc00_Li05_HP02_f{0}'.format(str(factor).replace('.', ''))

    name = '{0}_{1}_{2}'.format('deflex', sub, cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), subpath)
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def create_scenario_XX_Nc00_Li05_HP00(subpath='new', factor=0.0):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = main.load_deflex_scenario(2014, create_scenario=False)

    increase_re_share(sc, factor)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    sub = 'XX_Nc00_Li05_HP00_f{0}'.format(str(factor).replace('.', ''))

    name = '{0}_{1}_{2}'.format('deflex', sub, cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), subpath)
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def create_scenario_XX_Nc00_Li05_HP02_GT(subpath='new', factor=0.0):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    nom_val = {
        'DE01': 1455.2,
        'DE02': 2012.2,
        'DE03': 1908.8,
        'DE04': 0.0,
        'DE05': 0.0,
        'DE06': 0.0,
        'DE07': 0.0,
        'DE08': 0.0,
        'DE09': 3527.6,
        'DE10': 1736.7,
        'DE11': 0.0,
        'DE12': 7942.3,
        'DE13': 947.5,
        'DE14': 0.0,
        'DE15': 1047.7,
        'DE16': 1981.7,
        'DE17': 3803.8,
        'DE18': 3481.9,
        'DE19': 0.0,
        'DE20': 0.0,
        'DE21': 0.0}

    sc = main.load_deflex_scenario(2014, create_scenario=False)

    increase_re_share(sc, factor)

    add_simple_gas_turbine(sc, nom_val)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    sub = 'XX_Nc00_Li05_HP02_GT_f{0}'.format(str(factor).replace('.', ''))

    name = '{0}_{1}_{2}'.format('deflex', sub, cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), subpath)
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def create_scenario_XX_Nc00_Li05_HP00_GT(subpath='new', factor=0.0):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    nom_val = {
        'DE01': 1455.2,
        'DE02': 2012.2,
        'DE03': 1908.8,
        'DE04': 0.0,
        'DE05': 0.0,
        'DE06': 0.0,
        'DE07': 0.0,
        'DE08': 0.0,
        'DE09': 3527.6,
        'DE10': 1736.7,
        'DE11': 0.0,
        'DE12': 7942.3,
        'DE13': 947.5,
        'DE14': 0.0,
        'DE15': 1047.7,
        'DE16': 1981.7,
        'DE17': 3803.8,
        'DE18': 3481.9,
        'DE19': 0.0,
        'DE20': 0.0,
        'DE21': 0.0}

    sc = main.load_deflex_scenario(2014, create_scenario=False)

    increase_re_share(sc, factor)

    add_simple_gas_turbine(sc, nom_val)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    sub = 'XX_Nc00_Li05_HP00_GT_f{0}'.format(str(factor).replace('.', ''))

    name = '{0}_{1}_{2}'.format('deflex', sub, cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), subpath)
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def simple_deflex_de21_2014(subpath='new', factor=0.0):
    sc = main.load_deflex_scenario(2014, create_scenario=False)

    increase_re_share(sc, factor)

    sub = 'de21_f{0}'.format(str(factor).replace('.', ''))

    name = '{0}_{1}_{2}'.format('deflex', sub, cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), subpath)
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def multi_scenario_deflex():
    # Once re-create deflex scenario
    main.load_deflex_scenario(2014, create_scenario=True)
    s = os.path.join('deflex', 're')
    for f in range(11):
        create_scenario_XX_Nc00_Li05_HP00_GT(subpath=s, factor=f/10)
        create_scenario_XX_Nc00_Li05_HP02_GT(subpath=s, factor=f/10)
        create_scenario_XX_Nc00_Li05_HP00(subpath=s, factor=f/10)
        create_scenario_XX_Nc00_Li05_HP02(subpath=s, factor=f/10)
        simple_deflex_de21_2014(subpath=s, factor=f/10)


def create_deflex_no_grid_limit(year, rmap, create_scenario=False):
    sc = main.load_deflex_scenario(
        year, sim_type=rmap, create_scenario=create_scenario)
    cond = sc.table_collection['transmission']['electrical', 'capacity'] > 0
    sc.table_collection['transmission'].loc[
        cond, ('electrical', 'capacity')] = float('inf')
    sc.table_collection['transmission']['electrical', 'efficiency'] = 1
    name = sc.name + '_no_grid_limit'
    path = sc.location.replace(sc.location.split(os.sep)[-1], '')
    sc.to_csv(os.path.join(path, name + '_csv'))
    sc.to_excel(os.path.join(path, name + '.xls'))


if __name__ == "__main__":
    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])
    logger.define_logging()
    create_deflex_no_grid_limit()
    # multi_scenario_deflex()
    stopwatch()
