# Python libraries
import os
import logging
from datetime import datetime
import time
import traceback
from shutil import copyfile

# oemof packages
from oemof.tools import logger
from oemof import solph

# internal modules
import reegis_tools.config as cfg
import reegis_tools.scenario_tools
import deflex
from deflex import inhabitants
import berlin_hp
import my_reegis
from my_reegis import results as sys_results


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def load_deflex(year, sim_type='de21', create_scenario=True):
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('deflex', year, cfg.get('init', 'map'))
    sc = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        deflex.basic_scenario.create_basic_scenario(year, rmap=sim_type)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')
    return sc


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


def create_scenario_XX_Nc00_Li05_HP02():
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = load_deflex(2014, sim_type='de21', create_scenario=False)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    name = '{0}_{1}_{2}'.format('deflex', 'XX_Nc00_Li05_HP02', cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), 'new')
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def create_scenario_XX_Nc00_Li05_HP00():
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = load_deflex(2014, sim_type='de21', create_scenario=False)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    name = '{0}_{1}_{2}'.format('deflex', 'XX_Nc00_Li05_HP00', cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), 'new')
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def create_scenario_XX_Nc00_Li05_HP02_GT():
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

    sc = load_deflex(2014, sim_type='de21', create_scenario=False)
    add_simple_gas_turbine(sc, nom_val)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    name = '{0}_{1}_{2}'.format('deflex', 'XX_Nc00_Li05_HP02_GT', cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), 'new')
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


def create_scenario_XX_Nc00_Li05_HP00_GT():
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

    sc = load_deflex(2014, sim_type='de21', create_scenario=False)
    add_simple_gas_turbine(sc, nom_val)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    name = '{0}_{1}_{2}'.format('deflex', 'XX_Nc00_Li05_HP02_GT', cfg.get(
        'init', 'map'))
    path = os.path.join(cfg.get('paths', 'scenario'), 'new')
    sc.to_excel(os.path.join(path, name + '.xls'))
    csv_path = os.path.join(path, '{0}_csv'.format(name))
    sc.to_csv(csv_path)


if __name__ == "__main__":
    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])
    logger.define_logging()
    stopwatch()
