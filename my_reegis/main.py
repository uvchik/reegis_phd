# -*- coding: utf-8 -*-

"""Main script.

Copyright (c) 2016-2018 Uwe Krien <uwe.krien@rl-institut.de>

SPDX-License-Identifier: GPL-3.0-or-later
"""
__copyright__ = "Uwe Krien <uwe.krien@rl-institut.de>"
__license__ = "GPLv3"


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
import berlin_hp
import my_reegis
# from my_reegis import results as sys_results
from my_reegis import alternative_scenarios


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def compute(sc, dump_graph=False):
    scenario_path = os.path.dirname(sc.location)
    results_path = os.path.join(scenario_path, 'results')
    os.makedirs(results_path, exist_ok=True)

    # Save energySystem to '.graphml' file if dump_graph is True
    if dump_graph is True:
        sc.plot_nodes(filename=os.path.join(results_path, sc.name),
                      remove_nodes_with_substrings=['bus_cs'])

    logging.info("Create the concrete model for {0}: {1}".format(
        sc.name, stopwatch()))
    sc.create_model()

    logging.info("Solve the optimisation model ({0}): {1}".format(
        sc.name, stopwatch()))
    sc.solve()

    logging.info("Solved. Dump results: {0}".format(stopwatch()))
    out_file = os.path.join(results_path, sc.name + '.esys')
    logging.info("Dump file to {0}".format(out_file))
    sc.dump_es(out_file)

    logging.info("All done. {0} finished without errors: {0}".format(
        sc.name, stopwatch()))


def deflex_main(year, sim_type='de21', create_scenario=True):
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('deflex', year, cfg.get('init', 'map'))
    sc = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        deflex.basic_scenario.create_basic_scenario(year, rmap=sim_type)

    os.makedirs(os.path.join(scenario_path, 'results'), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, 'results', '{0}.xls'.format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')

    # Create nodes and add them to the EnergySystem
    sc.table2es()

    # Create concrete model, solve it and dump the results
    compute(sc)


def berlin_hp_main(year, sim_type='single', create_scenario=True):
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('berlin_hp', year, cfg.get('init', 'map'))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    os.makedirs(os.path.join(scenario_path, 'results'), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, 'results', '{0}.xls'.format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')

    # Create nodes and add them to the EnergySystem
    sc.table2es()

    # Create concrete model, solve it and dump the results
    compute(sc)


def embedded_main(year, sim_type='de21', create_scenario=True):
    # deflex
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('without_berlin', year, sim_type)
    sc_de = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc_de.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    # Create scenario files if they do exist or creation is forced
    if create_scenario or not os.path.isdir(sc_de.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        my_reegis.embedded_model.create_reduced_scenario(year, sim_type)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc_de.name, stopwatch()))
    sc_de.load_csv().check_table('time_series')
    nodes_de = sc_de.create_nodes()

    os.makedirs(os.path.join(scenario_path, 'results'), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc_de.name))
    dst = os.path.join(scenario_path, 'results', '{0}.xls'.format(sc_de.name))
    copyfile(src, dst)

    # berlin_hp
    cfg.tmp_set('init', 'map', 'single')
    name = '{0}_{1}_{2}'.format('berlin_hp', year, cfg.get('init', 'map'))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')

    nodes = sc.create_nodes(nodes_de)

    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(
        scenario_path, 'results', '{0}_embedded.xls'.format(sc.name))
    copyfile(src, dst)

    sc.add_nodes(nodes)
    sc.name = '{0}_{1}_{2}'.format('berlin_hp', year, sim_type)

    sc.add_nodes(my_reegis.embedded_model.connect_electricity_buses(
        'DE01', 'BE', sc.es))

    # Create concrete model, solve it and dump the results
    compute(sc)


def add_import_export_nodes(bus, import_costs, export_costs):
    nodes = reegis_tools.scenario_tools.NodeDict()
    nodes['import'] = solph.Source(
        label='import_elec_fhg',
        outputs={bus: solph.Flow(emission=0, variable_costs=import_costs)})
    nodes['export'] = solph.Sink(
        label='export_elec_fhg',
        inputs={bus: solph.Flow(emission=0, variable_costs=export_costs)})
    return nodes


def friedrichshagen_main(year, create_scenario=True):
    """Friedrichshagen"""
    name = '{0}_{1}_{2}'.format('friedrichshagen', year, 'single')
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.friedrichshagen.create_basic_scenario(year)

    os.makedirs(os.path.join(scenario_path, 'results'), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, 'results', '{0}.xls'.format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv()
    # print(sc.table_collection)
    sc.check_table('time_series')

    # Create nodes and add them to the EnergySystem
    sc.add_nodes(sc.create_nodes(region='FHG'))

    # print([x for x in sc.es.groups.keys() if 'elec' in str(x)])
    # costs = sys_results.analyse_system_costs(plot=False)

    # bus = sc.es.groups['bus_elec_FHG']
    # sc.add_nodes(add_import_export_nodes(
    #     bus, import_costs=costs/0.9, export_costs=costs*(-0.9)))

    # Create concrete model, solve it and dump the results
    compute(sc, dump_graph=True)


def optimise_scenario(path, name, create_fct=None, create_scenario=True,
                      year=None):

    if year is None:
        year = 2050

    sc = deflex.Scenario(name=name, year=year)
    sc.location = os.path.join(path, '{0}_csv'.format(name))

    if create_fct is not None:
        if create_scenario or not os.path.isdir(sc.location):
            logging.info("Create scenario for {0}: {1}".format(stopwatch(),
                                                               name))
            create_fct()

    os.makedirs(os.path.join(path, 'results'), exist_ok=True)
    src = os.path.join(path, '{0}.xls'.format(sc.name))
    dst = os.path.join(path, 'results', '{0}.xls'.format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')

    # Create nodes and add them to the EnergySystem
    sc.table2es()

    # Create concrete model, solve it and dump the results
    compute(sc)


def log_exception(e):
    logging.error(traceback.format_exc())
    time.sleep(0.5)
    logging.error(e)
    time.sleep(0.5)
    return False


def start_alternative_scenarios(checker, create_scenario=True):
    path = os.path.join(cfg.get('paths', 'scenario'), 'new')
    my_scenarios = {
        'deflex_XX_Nc00_Li05_HP02_de21':
            alternative_scenarios.create_scenario_XX_Nc00_Li05_HP02,
        'deflex_XX_Nc00_Li05_HP00_de21':
            alternative_scenarios.create_scenario_XX_Nc00_Li05_HP00,
        'deflex_XX_Nc00_Li05_HP02_GT_de21':
            alternative_scenarios.create_scenario_XX_Nc00_Li05_HP02_GT,
        'deflex_XX_Nc00_Li05_HP00_GT_de21':
            alternative_scenarios.create_scenario_XX_Nc00_Li05_HP00_GT,
    }

    for name, create_fct in my_scenarios.items():
        try:
            optimise_scenario(path, name, create_fct,
                              create_scenario=create_scenario, year=None)
        except Exception as e:
            checker = log_exception(e)

    return checker


def start_basic_scenarios(checker=True, create_scenario=True):
    for year in [2014, 2013, 2012]:
        # deflex and embedded
        for t in ['de21', 'de22']:
            try:
                deflex_main(year, sim_type=t, create_scenario=create_scenario)
                embedded_main(
                    year, sim_type=t, create_scenario=create_scenario)
            except Exception as e:
                checker = log_exception(e)

        # berlin_hp
        try:
            berlin_hp_main(year)
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_friedrichshagen(checker=True, create_scenario=True):
    try:
        friedrichshagen_main(2014, create_scenario=create_scenario)
    except Exception as e:
        checker = log_exception(e)

    return checker


def start_all(checker=True, create_scenario=True):

    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])

    # checker = start_friedrichshagen(checker, create_scenario=create_scenario)

    checker = start_basic_scenarios(checker, create_scenario=create_scenario)

    checker = start_alternative_scenarios(
        checker, create_scenario=create_scenario)

    return checker


def start_all_by_dir(checker=True):
    alternative_scenarios.multi_scenario_deflex()
    start_dir = os.path.join(cfg.get('paths', 'scenario'), 're')

    for root, directories, filenames in os.walk(start_dir):
        for d in directories:
            if d[-4:] == '_csv':
                name = d[:-4]
                try:
                    optimise_scenario(start_dir, name)
                except Exception as e:
                    checker = log_exception(e)
    return checker


def log_check(checker):
    if checker is True:
        logging.info("Everything is fine: {0}".format(stopwatch()))
    else:
        logging.info("Something went wrong see log: {0}".format(stopwatch()))


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])
    # deflex_main(2014, sim_type='de21', create_scenario=False)
    # exit(0)
    
    stopwatch()
    log_check(start_all_by_dir())
    # log_check(start_all(create_scenario=True))
    # log_check(
    #     start_alternative_scenarios(checker=True, create_scenario=True))
