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
from reegis_phd import config as cfg
import reegis
from deflex.scenario_tools import Scenario, Label
import deflex
import berlin_hp
import reegis_phd

# from reegis_phd import results as sys_results
from reegis_phd import alternative_scenarios
from reegis_phd import embedded_model
from reegis_phd import alternative_scenarios as alt
from reegis_phd import results


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


if __name__ == "__main__":
    pass
