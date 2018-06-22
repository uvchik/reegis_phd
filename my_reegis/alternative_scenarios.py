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


def create_expensive_scenario():
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = load_deflex(2014, sim_type='de21', create_scenario=False)

    sc.table_collection['transformer'] = sc.table_collection[
        'transformer'].swaplevel(axis=1).sort_index(axis=1)
    # remove nuclear power (by law)
    sc.table_collection['transformer'].drop(
        'nuclear', axis=1, inplace=True)

    # remove have lignite (climate change)
    sc.table_collection['transformer']['lignite'] = (
        sc.table_collection['transformer']['lignite'].div(2))

    sc.table_collection['transformer'] = sc.table_collection[
        'transformer'].swaplevel(axis=1).sort_index(axis=1)

    heat_pump_fraction = 0.2
    cop = 2

    abs_decentr_heat = (
        sc.table_collection['time_series']['DE_demand'].sum(axis=1))
    heat_pump = abs_decentr_heat * heat_pump_fraction
    sc.table_collection['time_series']['DE_demand'] *= (1 - heat_pump_fraction)

    # print(abs_decentr_heat.sum())
    # print(sc.table_collection['time_series']['DE_demand'].sum().sum())
    # print(heat_pump.sum())
    # print(sc.table_collection['time_series']['DE_demand'].sum().sum() +
    #       heat_pump.sum())
    inhab = inhabitants.get_ew_by_deflex(2014)
    inhab_fraction = inhab.div(inhab.sum())

    for region in inhab_fraction.index:
        if inhab_fraction.loc[region] > 0:
            sc.table_collection['time_series'][
                (region, 'electrical_load')] += (
                    inhab_fraction.loc[region] * heat_pump.div(cop))

    name = '{0}_{1}_{2}'.format('deflex', 'XX_Nc00_Li05_HP02', cfg.get(
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
    create_expensive_scenario()
