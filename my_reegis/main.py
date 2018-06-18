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

# internal modules
import reegis_tools.config as cfg
import deflex
import berlin_hp


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def load_scenario(sc):
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv(sc.location)

    sc.check_table('time_series')

    return sc


def compute(sc, nodes=None, dump_graph=False):
    if nodes is not None:
        sc.add_nodes(nodes)
    else:
        sc.add_nodes2solph()

    scenario_path = os.path.dirname(sc.location)

    if dump_graph is True:
        # Save energySystem to '.graphml' file.
        sc.plot_nodes(filename=os.path.join(scenario_path, sc.name),
                      remove_nodes_with_substrings=['bus_cs'])

    logging.info("Create the concrete model for {0}: {1}".format(
        sc.name, stopwatch()))
    sc.create_model()

    logging.info("Solve the optimisation model ({0}: {1}".format(
        sc.name, stopwatch()))
    sc.solve()

    logging.info("Solved. Dump results: {0}".format(stopwatch()))
    out_file = os.path.join(scenario_path, 'results', sc.name + '.esys')
    logging.info("Dump file to {0}".format(out_file))
    sc.dump_es(out_file)

    # Copy used xls-file to results to avoid confusion if the file is change
    # afterwards by another program or test.
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, 'results', '{0}.xls'.format(sc.name))
    copyfile(src, dst)

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

    compute(load_scenario(sc))


def berlin_hp_main(year, sim_type='single', create_scenario=True):
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('berlin_hp', year, cfg.get('init', 'map'))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isfile(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    compute(load_scenario(sc))


def embedded_main(year, sim_type='de21', create_scenario=True):
    # deflex
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('without_berlin', year, sim_type)
    sc_de = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc_de.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isfile(sc_de.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.embedded_model.create_reduced_scenario(year, sim_type)

    sc_de = load_scenario(sc_de)
    nodes_de = sc_de.create_nodes()

    # berlin_hp
    cfg.tmp_set('init', 'map', 'single')
    name = '{0}_{1}_{2}'.format('berlin_hp', year, cfg.get('init', 'map'))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isfile(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    sc = load_scenario(sc)
    nodes = sc.create_nodes(nodes_de)

    sc.add_nodes(nodes)
    sc.name = '{0}_{1}_{2}'.format('berlin_hp', year, sim_type)

    p = berlin_hp.embedded_model.connect_electricity_buses('DE01', 'BE', sc.es)

    compute(load_scenario(sc), nodes=p)


def log_exception(e):
    logging.error(traceback.format_exc())
    time.sleep(0.5)
    logging.error(e)
    time.sleep(0.5)
    return False


def start_all(create_scenario=True):
    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])

    checker = True

    for year in [2014, 2013, 2012]:
        # deflex and embedded
        for t in ['de21', 'de22']:
            try:
                deflex_main(year, sim_type=t, create_scenario=create_scenario)
                embedded_main(year)
            except Exception as e:
                checker = log_exception(e)

        # berlin_hp
        try:
            berlin_hp_main(year)
        except Exception as e:
            checker = log_exception(e)

    if checker is True:
        logging.info("Everything is fine: {0}".format(stopwatch()))
    else:
        logging.info("Something went wrong see log: {0}".format(stopwatch()))


if __name__ == "__main__":
    logger.define_logging()
    stopwatch()
    start_all(create_scenario=True)
