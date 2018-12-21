# -*- coding: utf-8 -*-

"""Main script.

Copyright (c) 2016-2018 Uwe Krien <uwe.krien@rl-institut.de>

SPDX-License-Identifier: GPL-3.0-or-later
"""
__copyright__ = "Uwe Krien <uwe.krien@rl-institut.de>"
__license__ = "GPLv3"


# Python libraries
import os
import sys
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
from my_reegis import embedded_model
from reegis_tools.scenario_tools import Label
from my_reegis import upstream_analysis as upa


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def compute(sc, dump_graph=False, log_solver=True, duals=True):
    scenario_path = os.path.dirname(sc.location)

    results_path = os.path.join(
        scenario_path, 'results_{0}'.format(cfg.get('general', 'solver')))
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

    if log_solver is True:
        filename = os.path.join(results_path, sc.name + '.log')
    else:
        filename = None

    if duals is True:
        sc.model.receive_duals()

    sc.solve(logfile=filename)

    logging.info("Solved. Dump results: {0}".format(stopwatch()))
    out_file = os.path.join(results_path, sc.name + '.esys')
    logging.info("Dump file to {0}".format(out_file))
    sc.dump_es(out_file)

    logging.info("All done. {0} finished without errors: {0}".format(
        sc.name, stopwatch()))


def load_deflex_scenario(year, sim_type='de21', create_scenario=False):
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('deflex', year, cfg.get('init', 'map'))
    sc = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), 'deflex',
                                 str(year))
    if 'without_berlin' in sim_type:
        scenario_path = os.path.join(cfg.get('paths', 'scenario'), 'berlin_hp',
                                     str(year))

    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        deflex.basic_scenario.create_basic_scenario(year, rmap=sim_type)

    res_path_name = 'results_{0}'.format(cfg.get('general', 'solver'))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, res_path_name, '{0}.xls'.format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')
    return sc


def deflex_main(year, sim_type='de21', create_scenario=True, dump_graph=False,
                extra_regions=None):
    logging.info("Start deflex: {0}".format(stopwatch()))
    sc = load_deflex_scenario(year, sim_type, create_scenario)

    if extra_regions is not None:
        sc.extra_regions = extra_regions

    # Create nodes and add them to the EnergySystem
    sc.table2es()

    # Create concrete model, solve it and dump the results
    compute(sc, dump_graph=dump_graph)


def remove_shortage_excess_electricity(nodes):
    elec_nodes = [v for k, v in nodes.items() if v.label.tag == 'electricity']
    for v in elec_nodes:
        if v.label.cat == 'excess':
            flow = next(iter(nodes[v.label].inputs.values()))
            flow.nominal_value = 0
        elif v.label.cat == 'shortage':
            flow = next(iter(nodes[v.label].outputs.values()))
            flow.nominal_value = 0


def add_upstream_import_export(nodes, bus, upstream_prices):

    if isinstance(upstream_prices, str):
        if upstream_prices == 'no_costs':
            export_costs = -0.000001
            import_costs = 500
        elif upstream_prices == 'ee':
            remove_shortage_excess_electricity(nodes)
            export_costs = 5000
            import_costs = 5000
        else:
            export_costs = None
            import_costs = None
    else:
        export_costs = upstream_prices * -0.99
        import_costs = upstream_prices * 1.01

    exp_label = Label('export', 'electricity', 'all', bus.label.region)
    nodes[exp_label] = solph.Sink(
                label=exp_label,
                inputs={bus: solph.Flow(
                    variable_costs=export_costs)})

    imp_label = Label('import', 'electricity', 'all', bus.label.region)
    nodes[imp_label] = solph.Source(
                label=imp_label,
                outputs={bus: solph.Flow(
                    variable_costs=import_costs)})
    return nodes


def berlin_hp_with_upstream_sets(year, solver, method='mcp', checker=True,
                                 create_scenario=True):
    df = upa.get_upstream_set(solver, year, method)
    for name, series in df.iteritems():
        try:
            berlin_hp_main(year, upstream_prices=series,
                           create_scenario=create_scenario)
            create_scenario = False
        except Exception as e:
            checker = log_exception(e)
    return checker


def berlin_hp_single_scenarios(year, checker=True, create_scenario=True):
    for name in ['no_costs', 'ee', None]:
        try:
            berlin_hp_main(year, upstream_prices=name,
                           create_scenario=create_scenario)
            create_scenario = False
        except Exception as e:
            checker = log_exception(e)
    return checker


# def start_berlin_single_scenarios(checker=True, create_scenario=True):
#     for year in [2014, 2013, 2012]:
#         up_sc = fhg_sc.load_upstream_scenario_values(
#             ).columns.get_level_values(0).unique()
#         up_sc = [x for x in up_sc if str(year) in x]
#         up_sc.append(None)
#         for upstream in up_sc:
#             # Run scenario
#             try:
#                 berlin_hp_main(year, create_scenario=create_scenario,
#                                upstream=upstream)
#             except Exception as e:
#                 checker = log_exception(e)
#     return checker


def berlin_hp_main(year, sim_type='single', create_scenario=True,
                   dump_graph=False, upstream_prices=None):

    cfg.tmp_set('init', 'map', sim_type)

    name = '{0}_{1}_{2}'.format('berlin_hp', year, cfg.get('init', 'map'))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), 'berlin_hp',
                                 str(year))
    sc.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    res_path_name = 'results_{0}'.format(cfg.get('general', 'solver'))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, res_path_name, '{0}.xls'.format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table('time_series')

    nodes = sc.create_nodes(region='BE')

    if upstream_prices is not None:
        elec_bus = [v for k, v in nodes.items() if
                    v.label.tag == 'electricity' and isinstance(v, solph.Bus)]
        bus = elec_bus[0]
        nodes = add_upstream_import_export(nodes, bus, upstream_prices)
        upstream_name = upstream_prices.name
    else:
        upstream_name = None

    sc.es = sc.initialise_energy_system()
    sc.es.add(*nodes.values())

    sc.name = sc.name + '_up_' + str(upstream_name)

    # Create concrete model, solve it and dump the results
    compute(sc, dump_graph=dump_graph)


def embedded_main(year, sim_type='de21', create_scenario=True):
    # deflex
    cfg.tmp_set('init', 'map', sim_type)
    name = '{0}_{1}_{2}'.format('deflex', year, sim_type + '_without_berlin')
    sc_de = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), 'berlin_hp',
                                 str(year))
    sc_de.location = os.path.join(scenario_path, '{0}_csv'.format(name))

    # Create scenario files if they do exist or creation is forced
    if create_scenario or not os.path.isdir(sc_de.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        my_reegis.embedded_model.create_reduced_scenario(year, sim_type)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc_de.name, stopwatch()))
    sc_de.load_csv().check_table('time_series')
    nodes_de = sc_de.create_nodes()

    res_path_name = 'results_{0}'.format(cfg.get('general', 'solver'))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc_de.name))
    dst = os.path.join(scenario_path, res_path_name,
                       '{0}.xls'.format(sc_de.name))
    copyfile(src, dst)

    # berlin_hp
    cfg.tmp_set('init', 'map', 'single')
    name = '{0}_{1}_{2}'.format('berlin_hp', year, cfg.get('init', 'map'))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(cfg.get('paths', 'scenario'), 'berlin_hp',
                                 str(year))
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
        scenario_path, res_path_name, '{0}_{1}.xls'.format(sc.name, sim_type))
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

    res_path_name = 'results_{0}'.format(cfg.get('general', 'solver'))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, '{0}.xls'.format(sc.name))
    dst = os.path.join(scenario_path, res_path_name, '{0}.xls'.format(sc.name))
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
    res_path_name = 'results_{0}'.format(cfg.get('general', 'solver'))
    os.makedirs(os.path.join(path, res_path_name), exist_ok=True)
    src = os.path.join(path, '{0}.xls'.format(sc.name))
    dst = os.path.join(path, res_path_name, '{0}.xls'.format(sc.name))
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


def start_embedded_scenarios(year, checker=True, create_scenario=True):
    for t in ['de21', 'de22']:
        try:
            embedded_main(
                year, sim_type=t, create_scenario=create_scenario)
            deflex_main(year, sim_type=t + '_without_berlin',
                        create_scenario=False)
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_deflex_scenarios(year, checker=True, create_scenario=True):
    # deflex and embedded
    for t in ['de02', 'de17', 'de21', 'de22']:
        if t == 'de22':
            ex_reg = ['DE22']
        else:
            ex_reg = None

        try:
            deflex_main(year, sim_type=t, create_scenario=create_scenario,
                        extra_regions=ex_reg)
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_no_storage_scenarios(year, checker=True, create_scenario=False):
    for t in ['de21', 'de22', 'de17', 'de02']:
        if t == 'de22':
            ex_reg = ['DE22']
        else:
            ex_reg = None
        try:
            if create_scenario is True:
                alternative_scenarios.create_deflex_no_storage(
                    year, t, create_scenario=True)
            deflex_main(year, sim_type=t + '_no_storage',
                        create_scenario=False, extra_regions=ex_reg)
        except Exception as e:
            checker = log_exception(e)
        try:
            if create_scenario is True:
                alternative_scenarios.create_deflex_no_grid_limit_no_storage(
                    year, t, create_scenario=True)
            deflex_main(year, sim_type=t + '_no_grid_limit_no_storage',
                        create_scenario=False, extra_regions=ex_reg)
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_no_grid_limit_scenarios(year, checker=True, create_scenario=False):
    for t in ['de21', 'de22', 'de17', 'de02']:
        if t == 'de22':
            ex_reg = ['DE22']
        else:
            ex_reg = None
        try:
            if create_scenario is True:
                alternative_scenarios.create_deflex_no_grid_limit(
                    year, t, create_scenario=True)
            t = t + '_no_grid_limit'
            deflex_main(year, sim_type=t, create_scenario=False,
                        extra_regions=ex_reg)
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

    checker = start_deflex_scenarios(
        2014, checker, create_scenario=create_scenario)

    # checker = start_alternative_scenarios(
    #     checker, create_scenario=create_scenario)

    return checker


def start_all_by_dir(checker=True, start_dir=None):
    # alternative_scenarios.multi_scenario_deflex()
    if start_dir is None:
        start_dir = os.path.join(cfg.get('paths', 'scenario'), 'deflex', 're')

    scenarios = []
    for root, directories, filenames in os.walk(start_dir):
        for d in directories:
            scenarios.append(d)

    logging.info("All scenarios: {0}".format(sorted(scenarios)))
    logging.info("Number of found scenarios: {0}".format(len(scenarios)))

    remain = len(scenarios)
    for d in sorted(scenarios):
        if d[-4:] == '_csv':
            name = d[:-4]
            try:
                optimise_scenario(start_dir, name)
            except Exception as e:
                checker = log_exception(e)
        remain -= 1
        logging.info("Number of remaining scenarios: {0}".format(remain))
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

    sys.setrecursionlimit(50000)
    stopwatch()
    check = True
    for y in [2014, 2013, 2012]:
        for slv in ['gurobi', 'cbc']:
            cfg.tmp_set('general', 'solver', slv)
            logging.info("Start scenarios for {0} using the {1} solver".format(
                y, cfg.get('general', 'solver')))
            check = berlin_hp_with_upstream_sets(y, slv, checker=check)
            check = berlin_hp_single_scenarios(y, checker=check)
            # check = start_no_storage_scenarios(y, checker=check,
            #                                    create_scenario=True)
            # check = start_no_grid_limit_scenarios(y, checker=check,
            #                                       create_scenario=True)
            # check = start_deflex_scenarios(y, checker=check,
            #                                create_scenario=True)
            # check = start_embedded_scenarios(y, checker=check,
            #                                  create_scenario=True)
    log_check(check)
    # startdir = os.path.join(cfg.get('paths', 'scenario'), 'deflex', 're')
    # log_check(start_all_by_dir(start_dir=startdir))
    # log_check(start_berlin_single_scenarios())
    # log_check(start_berlin_single_scenarios())
    # log_check(start_all(create_scenario=True))
    # log_check(
    #     start_alternative_scenarios(checker=True, create_scenario=True))
