# -*- coding: utf-8 -*-

"""Main script.

SPDX-FileCopyrightText: 2016-2019 Uwe Krien <krien@uni-bremen.de>

SPDX-License-Identifier: MIT
"""
__copyright__ = "Uwe Krien <krien@uni-bremen.de>"
__license__ = "MIT"


# Python libraries
import os
import math
import sys
import logging
from datetime import datetime
import time
import traceback
from shutil import copyfile
import multiprocessing
import pandas as pd

# oemof packages
from oemof.tools import logger
from oemof import solph

# internal modules
from my_reegis import config as cfg
import reegis
from deflex.scenario_tools import Scenario, Label
import deflex
import berlin_hp
import my_reegis

# from my_reegis import results as sys_results
from my_reegis import alternative_scenarios
from my_reegis import embedded_model
from my_reegis import upstream_analysis as upa
from my_reegis import alternative_scenarios as alt
from my_reegis import results


CHECKER = None


def stopwatch():
    if not hasattr(stopwatch, "start"):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def compute(
    sc,
    dump_graph=False,
    log_solver=False,
    duals=False,
    solver="cbc",
    result_path=None,
):
    scenario_path = os.path.dirname(sc.location)

    if result_path is None:
        result_path = os.path.join(
            scenario_path, "results_{0}".format(cfg.get("general", "solver"))
        )
        os.makedirs(result_path, exist_ok=True)

    # Save energySystem to '.graphml' file if dump_graph is True
    if dump_graph is True:
        sc.plot_nodes(
            filename=os.path.join(result_path, sc.name),
            remove_nodes_with_substrings=["bus_cs"],
        )

    logging.info(
        "Create the concrete model for {0}: {1}".format(sc.name, stopwatch())
    )
    sc.create_model()

    logging.info(
        "Solve the optimisation model ({0}): {1}".format(sc.name, stopwatch())
    )

    if log_solver is True:
        filename = os.path.join(result_path, sc.name + ".log")
    else:
        filename = None

    if duals is True:
        sc.model.receive_duals()

    sc.solve(logfile=filename, solver=solver)

    logging.info("Solved. Dump results: {0}".format(stopwatch()))
    out_file = os.path.join(result_path, sc.name + ".esys")
    logging.info("Dump file to {0}".format(out_file))
    if sc.meta is None:
        sc.meta = {}
    sc.meta["end_time"] = datetime.now()
    sc.meta["filename"] = sc.name + ".esys"
    sc.dump_es(out_file)

    logging.info(
        "All done. {0} finished without errors: {0}".format(
            sc.name, stopwatch()
        )
    )
    return out_file


def load_deflex_scenario(year, sim_type="de21", create_scenario=False):
    cfg.tmp_set("init", "map", sim_type)
    name = "{0}_{1}_{2}".format("deflex", year, cfg.get("init", "map"))
    sc = Scenario(name=name, year=year)
    scenario_path = os.path.join(
        cfg.get("paths", "scenario"), "deflex", str(year)
    )
    if "without_berlin" in sim_type:
        scenario_path = os.path.join(
            cfg.get("paths", "scenario"), "berlin_hp", str(year)
        )

    sc.location = os.path.join(scenario_path, "{0}_csv".format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        deflex.basic_scenario.create_basic_scenario(year, rmap=sim_type)

    res_path_name = "results_{0}".format(cfg.get("general", "solver"))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, "{0}.xls".format(sc.name))
    dst = os.path.join(scenario_path, res_path_name, "{0}.xls".format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table("time_series")
    return sc


def deflex_main(
    year,
    sim_type="de21",
    create_scenario=True,
    dump_graph=False,
    extra_regions=None,
):
    logging.info("Start deflex: {0}".format(stopwatch()))
    sc = load_deflex_scenario(year, sim_type, create_scenario)

    if extra_regions is not None:
        sc.extra_regions = extra_regions

    # Create nodes and add them to the EnergySystem
    sc.table2es()

    # Create concrete model, solve it and dump the results
    compute(sc, dump_graph=dump_graph)


def deflex_alternative_scenarios(year):
    alt.create_XX_scenario_set(year)
    scenario_list = alt.fetch_XX_scenarios(year)
    n = len(scenario_list)
    logging.info("Number of scenarios: {0}".format(n))
    i = 0
    for scenario_path in scenario_list:
        i += 1
        name = scenario_path.split(os.sep)[-1][:-4]
        logging.info("Start scenario {0} from {1}: {2}".format(i, n, name))
        sc = deflex.Scenario(name=name, year=year)
        sc.location = scenario_path
        sc.load_csv().check_table("time_series")
        sc.table2es()
        compute(sc, dump_graph=False)


def remove_shortage_excess_electricity(nodes):
    elec_nodes = [v for k, v in nodes.items() if v.label.tag == "electricity"]
    for v in elec_nodes:
        if v.label.cat == "excess":
            flow = next(iter(nodes[v.label].inputs.values()))
            flow.nominal_value = 0
        elif v.label.cat == "shortage":
            flow = next(iter(nodes[v.label].outputs.values()))
            flow.nominal_value = 0
    return nodes


def add_upstream_import_export(nodes, bus, upstream_prices):

    if isinstance(upstream_prices, str):
        if upstream_prices == "no_costs":
            export_costs = -0.000001
            import_costs = 500
        elif upstream_prices == "ee":
            remove_shortage_excess_electricity(nodes)
            export_costs = 5000
            import_costs = 5000
        else:
            export_costs = None
            import_costs = None
    else:
        export_costs = upstream_prices * -0.99
        import_costs = upstream_prices * 1.01

    exp_label = Label("export", "electricity", "all", bus.label.region)
    nodes[exp_label] = solph.Sink(
        label=exp_label, inputs={bus: solph.Flow(variable_costs=export_costs)}
    )

    imp_label = Label("import", "electricity", "all", bus.label.region)
    nodes[imp_label] = solph.Source(
        label=imp_label, outputs={bus: solph.Flow(variable_costs=import_costs)}
    )
    return nodes


def berlin_hp_no_exit_multi(d):
    year = d["year"]
    meta = d["meta"]
    series = d["series"]
    create_scenario = d["create_scenario"]
    meta["start_time"] = datetime.now()
    meta["map"] = "berlin"
    meta["model_base"] = "berlin_hp"
    global CHECKER
    try:
        berlin_hp_main(
            year, meta, upstream_prices=series, create_scenario=create_scenario
        )
    except Exception as e:
        CHECKER = log_exception(e)


def create_upstream_sets(year, solver, method="mcp"):

    df = upa.get_upstream_set(solver, year, method, overwrite=False)

    sc_files = results.fetch_scenarios(
        os.path.join(cfg.get("paths", "scenario"), "deflex"),
        sc_filter={"solver": solver, "year": year},
    )
    scenarios = []
    for fn in sc_files:
        sc = Scenario(results_fn=fn)
        meta_up = sc.meta
        my_upstream = {
            "ee_factor": meta_up["ee_factor"],
            "gas_turbine": meta_up["gas_turbine"],
            "grid_limit": meta_up["grid_limit"],
            "heat_pump": meta_up["heat_pump"],
            "lignite": meta_up["lignite"],
            "map": meta_up["map"],
            "nuclear": meta_up["nuclear"],
            "storage": meta_up["storage"],
        }

        my_meta = {
            "ee_factor": 1.0,
            "excluded": None,
            "filename": None,
            "gas_turbine": 0,
            "grid_limit": True,
            "heat_pump": 0.0,
            "lignite": 1.0,
            "map": None,
            "model_base": None,
            "nuclear": 1.0,
            "solver": cfg.get("general", "solver"),
            "storage": True,
            "upstream": my_upstream,
            "year": year,
            "start_time": None,
        }
        base = "deflex_{0}_".format(year)
        name = str(fn.split(os.sep)[-1][:-5]).replace(base, "")
        series = df[name]
        scenarios.append(
            {
                "year": year,
                "meta": my_meta,
                "series": series,
                "create_scenario": False,
            }
        )
    return scenarios


def berlin_hp_with_upstream_sets(year, solver, method="mcp", checker=True):
    global CHECKER
    CHECKER = checker
    berlin_hp.basic_scenario.create_basic_scenario(year)
    scenarios = create_upstream_sets(year, solver, method)
    p = multiprocessing.Pool(int(multiprocessing.cpu_count() / 2))
    p.map(berlin_hp_no_exit_multi, scenarios)
    p.close()
    p.join()
    checker = CHECKER
    return checker


def berlin_hp_single_scenarios(
    year, meta=None, checker=True, create_scenario=True
):
    if meta is None:
        meta = {
            "ee_factor": 1.0,
            "excluded": None,
            "filename": None,
            "gas_turbine": 0,
            "grid_limit": True,
            "heat_pump": 0.0,
            "lignite": 1.0,
            "map": "berlin",
            "model_base": "berlin_hp",
            "nuclear": 1.0,
            "solver": cfg.get("general", "solver"),
            "storage": True,
            "upstream": None,
            "year": year,
            "start_time": datetime.now(),
        }

    for name in ["no_costs", "ee", None]:
        try:
            meta["upstream"] = name
            berlin_hp_main(
                year,
                meta,
                upstream_prices=name,
                create_scenario=create_scenario,
            )
            create_scenario = False
        except Exception as e:
            checker = log_exception(e)
    return checker


def berlin_hp_main(
    year,
    meta,
    sim_type="single",
    create_scenario=True,
    dump_graph=False,
    upstream_prices=None,
):

    cfg.tmp_set("init", "map", sim_type)

    name = "{0}_{1}_{2}".format("berlin_hp", year, cfg.get("init", "map"))
    sc = berlin_hp.Scenario(name=name, year=year, debug=False, meta=meta)
    scenario_path = os.path.join(
        cfg.get("paths", "scenario"), "berlin_hp", str(year)
    )
    sc.location = os.path.join(scenario_path, "{0}_csv".format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    res_path_name = "results_{0}".format(cfg.get("general", "solver"))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, "{0}.xls".format(sc.name))
    dst = os.path.join(scenario_path, res_path_name, "{0}.xls".format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table("time_series")

    nodes = sc.create_nodes(region="BE")

    if upstream_prices is not None:
        elec_bus = [
            v
            for k, v in nodes.items()
            if v.label.tag == "electricity" and isinstance(v, solph.Bus)
        ]
        bus = elec_bus[0]
        nodes = add_upstream_import_export(nodes, bus, upstream_prices)
        if not isinstance(upstream_prices, str):
            upstream_name = upstream_prices.name
        else:
            upstream_name = upstream_prices
    else:
        upstream_name = None

    sc.es = sc.initialise_energy_system()
    sc.es.add(*nodes.values())

    sc.name = sc.name + "_up_" + str(upstream_name)

    # Create concrete model, solve it and dump the results
    compute(sc, dump_graph=dump_graph)


def embedded_main(year, sim_type="de21", create_scenario=True):
    # deflex
    cfg.tmp_set("init", "map", sim_type)
    name = "{0}_{1}_{2}".format("deflex", year, sim_type + "_without_berlin")
    sc_de = deflex.Scenario(name=name, year=year)
    scenario_path = os.path.join(
        cfg.get("paths", "scenario"), "berlin_hp", str(year)
    )
    sc_de.location = os.path.join(scenario_path, "{0}_csv".format(name))

    # Create scenario files if they do exist or creation is forced
    if create_scenario or not os.path.isdir(sc_de.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        my_reegis.embedded_model.create_reduced_scenario(year, sim_type)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc_de.name, stopwatch()))
    sc_de.load_csv().check_table("time_series")
    nodes_de = sc_de.create_nodes()

    res_path_name = "results_{0}".format(cfg.get("general", "solver"))
    os.makedirs(os.path.join(scenario_path, res_path_name), exist_ok=True)
    src = os.path.join(scenario_path, "{0}.xls".format(sc_de.name))
    dst = os.path.join(
        scenario_path, res_path_name, "{0}.xls".format(sc_de.name)
    )
    copyfile(src, dst)

    # berlin_hp
    cfg.tmp_set("init", "map", "single")
    name = "{0}_{1}_{2}".format("berlin_hp", year, cfg.get("init", "map"))
    sc = berlin_hp.BerlinScenario(name=name, year=year, debug=False)
    scenario_path = os.path.join(
        cfg.get("paths", "scenario"), "berlin_hp", str(year)
    )
    sc.location = os.path.join(scenario_path, "{0}_csv".format(name))

    if create_scenario or not os.path.isdir(sc.location):
        logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
        berlin_hp.basic_scenario.create_basic_scenario(year)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table("time_series")

    nodes = sc.create_nodes(nodes_de)

    src = os.path.join(scenario_path, "{0}.xls".format(sc.name))
    dst = os.path.join(
        scenario_path, res_path_name, "{0}_{1}.xls".format(sc.name, sim_type)
    )
    copyfile(src, dst)

    sc.add_nodes(nodes)
    sc.name = "{0}_{1}_{2}".format("berlin_hp", year, sim_type)

    sc.add_nodes(
        my_reegis.embedded_model.connect_electricity_buses("DE01", "BE", sc.es)
    )

    # Create concrete model, solve it and dump the results
    compute(sc)


def add_import_export_nodes(bus, nodes, import_costs, export_costs):
    nodes["import"] = solph.Source(
        label=Label(
            cat="source", tag="import", subtag="electricity", region="all"
        ),
        outputs={bus: solph.Flow(emission=0, variable_costs=import_costs)},
    )
    nodes["export"] = solph.Sink(
        label=Label(
            cat="sink", tag="export", subtag="electricity", region="all"
        ),
        inputs={bus: solph.Flow(emission=0, variable_costs=export_costs)},
    )
    return nodes


def set_volatile_sources(nodes, add_capacity):
    """add_capacity = {'solar': 5,
                       'wind': 20}
    """

    ee_sources = {
        k: v
        for k, v in nodes.items()
        if v.label.tag == "ee" and v.label.cat == "source"
    }
    for label in ee_sources.keys():
        if label.subtag.lower() in add_capacity:
            if add_capacity[label.subtag.lower()] is not None:
                flow = next(iter(nodes[label].outputs.values()))
                flow.nominal_value = add_capacity[label.subtag.lower()]

    return nodes


def modellhagen_scenario_with_nodes(
    scenario, solar_capacity=None, wind_capacity=None
):
    logging.info("Read scenario {0}: {1}".format(scenario, stopwatch()))
    sc = berlin_hp.BerlinScenario(name="modellhagen", debug=False)
    sc.load_excel(scenario)
    sc.year = int([x for x in scenario.split("_") if x.isnumeric()][0])
    sc.name = "{0}_{1}_{2}".format("friedrichshagen", sc.year, "single")

    # Create nodes and add them to the EnergySystem
    nodes = sc.create_nodes(region="FHG")
    set_capacity = {"solar": solar_capacity, "wind": wind_capacity}
    nodes = set_volatile_sources(nodes, set_capacity)
    nodes = remove_shortage_excess_electricity(nodes)
    bus = [
        v
        for k, v in nodes.items()
        if k.tag == "electricity" and isinstance(v, solph.Bus)
    ][0]
    nodes = add_import_export_nodes(
        bus, nodes, import_costs=10, export_costs=0
    )
    sc.add_nodes(nodes)
    return sc


def modellhagen_main(
    scenario,
    result_path=None,
    solver="cbc",
    solar_capacity=None,
    wind_capacity=None,
):
    """Modellhagen"""

    sc, nodes = modellhagen_scenario_with_nodes(
        scenario=scenario,
        solar_capacity=solar_capacity,
        wind_capacity=wind_capacity,
    )

    if result_path is None:
        result_path = os.path.join(
            os.path.dirname(scenario), "results_{0}".format(solver)
        )
    os.makedirs(result_path, exist_ok=True)
    # print([x for x in sc.es.groups.keys() if 'elec' in str(x)])
    # costs = sys_results.analyse_system_costs(plot=False)

    # bus = sc.es.groups['bus_elec_FHG']
    # sc.add_nodes(add_import_export_nodes(
    #     bus, import_costs=costs/0.9, export_costs=costs*(-0.9)))

    # Create concrete model, solve it and dump the results
    compute(sc, dump_graph=False, solver=solver, result_path=result_path)


def modellhagen_re_variation(path, log_file=None):
    path = [p for p in path if "E100RE" in p][0]
    sc = berlin_hp.BerlinScenario(name="modellhagen", debug=False)
    sc.load_excel(path)
    flh = sc.table_collection["time_series"]["FHG"].sum()
    pv_flh = flh["solar"]
    wind_flh = flh["wind"]
    demand = sc.table_collection["time_series"]["electricity", "demand"].sum()
    scenarios_re = []
    for frac in range(11):
        solar = demand / pv_flh * frac / 10
        wind = demand / wind_flh * (1 - frac / 10)
        scenarios_re.append((path, solar, wind))
    model_multi_scenarios(scenarios_re, cpu_fraction=0.7)


def model_multi_scenarios(variations, cpu_fraction=0.2, log_file=None):
    """

    Parameters
    ----------
    variations : iterable
        Multiple scenarios to be modelled in parallel.
    cpu_fraction : float
        Fraction of available cpu cores to use for the parallel modelling.
        A resulting dezimal number of cores will be rounded up to an integer.
    log_file : str
        Filename to store the log file.

    Returns
    -------

    """
    start = datetime.now()
    maximal_number_of_cores = math.ceil(
        multiprocessing.cpu_count() * cpu_fraction
    )

    p = multiprocessing.Pool(maximal_number_of_cores)

    logs = p.map(batch_model_scenario, variations)
    p.close()
    p.join()
    failing = {n: r for n, r, t, f, s in logs if isinstance(r, BaseException)}

    log_df = pd.DataFrame()
    for log in logs:
        print(log)
        log_df.loc[log[0], "start"] = start
        if isinstance(log[1], BaseException):
            log_df.loc[log[0], "return_value"] = repr(log[1])
        else:
            log_df.loc[log[0], "return_value"] = log[1]
        log_df.loc[log[0], "trace"] = log[2]
        log_df.loc[log[0], "result_file"] = log[3]

    if log_file is None:
        log_file = os.path.join(
            os.path.expanduser("~"), ".deflex", "log_deflex.csv"
        )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_df.to_csv(log_file)

    if len(failing) < 1:
        logging.info("Finished all scenarios without errors")
    else:
        logging.info(failing)


def batch_model_scenario(var, ignore_errors=True):
    """
    Model a single scenario in batch mode. By default errors will be ignored
    and returned together with the traceback.

    Parameters
    ----------
    var : tuple
        Scenario variation.
    ignore_errors : bool
        Set True to stop the script if an error occurs for debugging. By
        default errors are ignored and returned.

    Returns
    -------
    tuple
    """
    start_time = datetime.now()
    sc = modellhagen_scenario_with_nodes(
        var[0], solar_capacity=var[1], wind_capacity=var[2]
    )
    sc.create_model()
    sc.name = sc.name + "_pv{0}_wind{1}".format(int(var[1]), int(var[2]))
    logging.info("Next scenario: %s", sc.name)
    if ignore_errors:
        try:
            result_file = compute(sc)
            return_value = datetime.now()
            trace = None
        except Exception as e:
            trace = traceback.format_exc()
            return_value = e
            result_file = None
    else:
        result_file = compute(sc)
        return_value = str(datetime.now())
        trace = None

    return sc.name, return_value, trace, result_file, start_time


def optimise_scenario(
    path, name, create_fct=None, create_scenario=True, year=None
):

    if year is None:
        year = 2050

    sc = deflex.Scenario(name=name, year=year)
    sc.location = os.path.join(path, "{0}_csv".format(name))

    if create_fct is not None:
        if create_scenario or not os.path.isdir(sc.location):
            logging.info(
                "Create scenario for {0}: {1}".format(stopwatch(), name)
            )
            create_fct()
    res_path_name = "results_{0}".format(cfg.get("general", "solver"))
    os.makedirs(os.path.join(path, res_path_name), exist_ok=True)
    src = os.path.join(path, "{0}.xls".format(sc.name))
    dst = os.path.join(path, res_path_name, "{0}.xls".format(sc.name))
    copyfile(src, dst)

    # Load scenario from csv-file
    logging.info("Read scenario {0}: {1}".format(sc.name, stopwatch()))
    sc.load_csv().check_table("time_series")

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
    path = os.path.join(cfg.get("paths", "scenario"), "new")
    my_scenarios = {
        "deflex_XX_Nc00_Li05_HP02_de21": alternative_scenarios.create_scenario_XX_Nc00_Li05_HP02,
        "deflex_XX_Nc00_Li05_HP00_de21": alternative_scenarios.create_scenario_XX_Nc00_Li05_HP00,
        "deflex_XX_Nc00_Li05_HP02_GT_de21": alternative_scenarios.create_scenario_xx_nc00_li05_hp02_gt,
        "deflex_XX_Nc00_Li05_HP00_GT_de21": alternative_scenarios.create_scenario_xx_nc00_li05_hp00_gt,
    }

    for name, create_fct in my_scenarios.items():
        try:
            optimise_scenario(
                path,
                name,
                create_fct,
                create_scenario=create_scenario,
                year=None,
            )
        except Exception as e:
            checker = log_exception(e)

    return checker


def start_embedded_scenarios(year, checker=True, create_scenario=True):
    for t in ["de21", "de22"]:
        try:
            embedded_main(year, sim_type=t, create_scenario=create_scenario)
            deflex_main(
                year, sim_type=t + "_without_berlin", create_scenario=False
            )
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_deflex_scenarios(year, checker=True, create_scenario=True):
    # deflex and embedded
    for t in ["de02", "de17", "de21", "de22"]:
        if t == "de22":
            ex_reg = ["DE22"]
        else:
            ex_reg = None

        try:
            deflex_main(
                year,
                sim_type=t,
                create_scenario=create_scenario,
                extra_regions=ex_reg,
            )
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_no_storage_scenarios(year, checker=True, create_scenario=False):
    for t in ["de21", "de22", "de17", "de02"]:
        if t == "de22":
            ex_reg = ["DE22"]
        else:
            ex_reg = None
        try:
            if create_scenario is True:
                alternative_scenarios.create_deflex_no_storage(
                    year, t, create_scenario=True
                )
            deflex_main(
                year,
                sim_type=t + "_no_storage",
                create_scenario=False,
                extra_regions=ex_reg,
            )
        except Exception as e:
            checker = log_exception(e)
        try:
            if create_scenario is True:
                alternative_scenarios.create_deflex_no_grid_limit_no_storage(
                    year, t, create_scenario=True
                )
            deflex_main(
                year,
                sim_type=t + "_no_grid_limit_no_storage",
                create_scenario=False,
                extra_regions=ex_reg,
            )
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_no_grid_limit_scenarios(year, checker=True, create_scenario=False):
    for t in ["de21", "de22", "de17", "de02"]:
        if t == "de22":
            ex_reg = ["DE22"]
        else:
            ex_reg = None
        try:
            if create_scenario is True:
                alternative_scenarios.create_deflex_no_grid_limit(
                    year, t, create_scenario=True
                )
            t = t + "_no_grid_limit"
            deflex_main(
                year, sim_type=t, create_scenario=False, extra_regions=ex_reg
            )
        except Exception as e:
            checker = log_exception(e)
    return checker


def start_deflex_without_scenarios(year, checker=True, raise_errors=False):
    start_dir = os.path.join(cfg.get("paths", "scenario"), "deflex", str(year))
    sc_filter = "without"
    checker = start_all_by_dir(
        checker, start_dir, sc_filter, raise_errors=raise_errors
    )
    return checker


def start_friedrichshagen(checker=True, create_scenario=True):
    try:
        friedrichshagen_main(2014, create_scenario=create_scenario)
    except Exception as e:
        checker = log_exception(e)

    return checker


def start_all(checker=True, create_scenario=True):

    cfg.init(
        paths=[
            os.path.dirname(deflex.__file__),
            os.path.dirname(berlin_hp.__file__),
        ]
    )

    # checker = start_friedrichshagen(checker, create_scenario=create_scenario)

    checker = start_deflex_scenarios(
        2014, checker, create_scenario=create_scenario
    )

    # checker = start_alternative_scenarios(
    #     checker, create_scenario=create_scenario)

    return checker


def start_all_by_dir(
    checker=True, start_dir=None, sc_filter=None, raise_errors=False
):
    # alternative_scenarios.multi_scenario_deflex()
    if start_dir is None:
        start_dir = os.path.join(cfg.get("paths", "scenario"), "deflex", "re")

    scenarios = []
    for root, directories, filenames in os.walk(start_dir):
        for d in directories:
            if sc_filter is None:
                scenarios.append(d)
            else:
                if sc_filter in d:
                    scenarios.append(d)

    logging.info("All scenarios: {0}".format(sorted(scenarios)))
    logging.info("Number of found scenarios: {0}".format(len(scenarios)))

    remain = len(scenarios)
    for d in sorted(scenarios):
        if d[-4:] == "_csv":
            name = d[:-4]
            if not raise_errors:
                try:
                    optimise_scenario(start_dir, name)
                except Exception as e:
                    checker = log_exception(e)
            else:
                optimise_scenario(start_dir, name)
        remain -= 1
        logging.info("Number of remaining scenarios: {0}".format(remain))
    return checker


def log_check(checker):
    if checker is True:
        logging.info("Everything is fine: {0}".format(stopwatch()))
    else:
        logging.info("Something went wrong see log: {0}".format(stopwatch()))


if __name__ == "__main__":
    pass
