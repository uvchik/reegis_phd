# -*- coding: utf-8 -*-

"""Main script.

SPDX-FileCopyrightText: 2016-2019 Uwe Krien <krien@uni-bremen.de>

SPDX-License-Identifier: MIT
"""
__copyright__ = "Uwe Krien <krien@uni-bremen.de>"
__license__ = "MIT"


import logging
import multiprocessing
import os
import traceback
from datetime import datetime

import berlin_hp
import deflex
import pandas as pd
from oemof import solph as solph
from reegis import config as cfg


def stopwatch():
    if not hasattr(stopwatch, "start"):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def get_scenario_name_from_excel(file):
    xls = pd.ExcelFile(file)
    return xls.parse("meta", index_col=[0]).loc["name", "value"]


def model_multi_scenarios(
    scenarios_de, scenarios_be, cpu_fraction=0.2, log_file=None, upstream=None,
):
    """

    Parameters
    ----------
    scenarios_de : iterable
        Multiple scenarios to be modelled in parallel.
    scenarios_be : iterable
        Multiple scenarios to be combined with scenarios_de.
    cpu_fraction : float
        Fraction of available cpu cores to use for the parallel modelling.
        A resulting decimal number of cores will be rounded up to an integer.
    log_file : str
        Filename to store the log file.
    upstream : str or None
        Path to the excel file with the prices.

    Returns
    -------

    """
    from pprint import pprint
    start = datetime.now()
    maximal_number_of_cores = int(
        round(multiprocessing.cpu_count() * cpu_fraction + 0.4999)
    )
    sc_combined = []
    if upstream is not None:
        upstream = pd.read_excel(upstream)

    for s_de in scenarios_de:
        for s_be in scenarios_be:
            if upstream is None:
                sc_combined.append((s_de, s_be))
            else:
                up_name = get_scenario_name_from_excel(s_de)
                mcp = upstream[up_name]
                up_sc = {
                    "name": up_name,
                    "import": mcp * 1.00001,
                    "export": mcp,
                }
                sc_combined.append((up_sc, s_be))

    p = multiprocessing.Pool(maximal_number_of_cores)
    pprint(sc_combined)
    if upstream is None:
        logs = p.map(combine_models_dcpl, sc_combined)
    else:
        logs = p.map(combine_models_up, sc_combined)

    p.close()
    p.join()
    failing = {n: r for n, r, t, f, s in logs if isinstance(r, BaseException)}

    logger = pd.DataFrame()
    for log in logs:
        logger.loc[log[0], "start"] = start
        if isinstance(log[1], BaseException):
            logger.loc[log[0], "return_value"] = repr(log[1])
        else:
            logger.loc[log[0], "return_value"] = log[1]
        logger.loc[log[0], "trace"] = log[2]
        logger.loc[log[0], "result_file"] = log[3]

    if log_file is None:
        log_file = os.path.join(
            os.path.expanduser("~"), ".deflex", "log_deflex.csv"
        )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.to_csv(log_file)

    if len(failing) < 1:
        logging.info("Finished all scenarios without errors")
    else:
        logging.info(failing)


def combine_models_dcpl(fn, ignore_errors=True):
    fn_de = fn[0]
    fn_be = fn[1]
    logging.info("Model DE: {0}".format(fn_de))
    logging.info("Model BE: {0}".format(fn_be))

    y_de = int([x for x in fn_de.split("_") if x.isnumeric()][0])
    y_be = int([x for x in fn_be.split("_") if x.isnumeric()][0])
    if y_be != y_de:
        raise ValueError(
            "You cannot combine scenarios from different years.\n"
            "Year DE:{0}, Year BE:{1}".format(y_be, y_de)
        )
    n_de = os.path.basename(fn_de).split(".")[0]
    n_be = os.path.basename(fn_be).split(".")[0]
    name = "{0}_DCPL_{1}".format(n_be, n_de)
    start_time = datetime.now()
    logging.info("Next scenario: {0} - {1}".format(name, stopwatch()))
    if ignore_errors:
        try:
            result_file = main(y_de, fn_de, fn_be)
            return_value = datetime.now()
            trace = None
        except Exception as e:
            trace = traceback.format_exc()
            return_value = e
            result_file = None
    else:
        result_file = main(y_de, fn_de, fn_be)
        return_value = str(datetime.now())
        trace = None

    return name, return_value, trace, result_file, start_time


def combine_models_up(fn, ignore_errors=True):
    up_sc = fn[0]
    fn_be = fn[1]
    logging.info("MCP DE: {0}".format(up_sc["name"]))
    logging.info("Model BE: {0}".format(fn_be))

    y = int([x for x in fn_be.split("_") if x.isnumeric()][0])
    n1 = up_sc["name"]
    n2 = os.path.basename(fn_be).split(".")[0]
    name = "{0}_dcpl_{1}".format(n1, n2)
    start_time = datetime.now()
    logging.info("Next scenario: {0} - {1}".format(name, stopwatch()))
    if ignore_errors:
        try:
            result_file = berlin_hp.main(y, fn_be, upstream_prices=up_sc)
            return_value = datetime.now()
            trace = None
        except Exception as e:
            trace = traceback.format_exc()
            return_value = e
            result_file = None
    else:
        result_file = berlin_hp.main(y, fn_be, upstream_prices=up_sc)
        return_value = str(datetime.now())
        trace = None

    return name, return_value, trace, result_file, start_time


def connect_electricity_buses(bus1, bus2, es):
    label = berlin_hp.Label
    nodes = deflex.scenario_tools.NodeDict()
    lines = [(bus1, bus2), (bus2, bus1)]
    for line in lines:
        line_label = label("line", "electricity", line[0], line[1])
        bus_label_in = label("bus", "electricity", "all", line[0])
        bus_label_out = label("bus", "electricity", "all", line[1])
        b_in = es.groups[str(bus_label_in)]
        b_out = es.groups[str(bus_label_out)]
        nodes[line_label] = solph.Transformer(
            label=line_label,
            inputs={b_in: solph.Flow(variable_costs=0.0000001)},
            outputs={b_out: solph.Flow()},
        )
    return nodes


def main(year, fn_de, fn_be):
    stopwatch()

    # Load data of the de21 model
    logging.info(
        "Read de21 scenario from csv collection: {0}".format(stopwatch())
    )

    n1 = os.path.basename(fn_de).split(".")[0]
    n2 = os.path.basename(fn_be).split(".")[0]
    name = "{0}_dcpl_{1}".format(n1, n2)

    sc_de = deflex.main.load_scenario(fn_de)

    # Create nodes for the de21 model
    nodes_de21 = sc_de.create_nodes()

    # Load data of the berlin_hp Model
    logging.info("Read scenario from excel-sheet: {0}".format(stopwatch()))
    sc_be = berlin_hp.BerlinScenario(name="berlin_basic", year=year)
    sc_be.load_excel(fn_be)
    sc_be.name = name

    # Create nodes for the berlin_hp model
    sc_be.add_nodes(nodes_de21)
    sc_be.add_nodes(sc_be.create_nodes())

    # Connect de21 and berlin_hp with a transmission line
    sc_be.add_nodes(connect_electricity_buses("DE01", "BE", sc_be.es))

    # Create model (all constraints)
    logging.info("Create the concrete model: {0}".format(stopwatch()))
    sc_be.create_model()

    # Pass the model to the solver and fetch the results afterwards..
    logging.info("Solve the optimisation model: {0}".format(stopwatch()))
    sc_be.solve(solver=cfg.get("general", "solver"))

    # Dump the energy system with the results to disc
    logging.info("Solved. Dump results: {0}".format(stopwatch()))
    res_path = os.path.join(
        os.path.dirname(fn_be),
        "results_{0}".format(cfg.get("general", "solver")),
    )
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    result_fn = os.path.join(res_path, "{0}.esys".format(name))
    sc_be.dump_es(result_fn)

    logging.info(
        "All done. Berlin {0} finished without errors: {0}".format(stopwatch())
    )
    return result_fn


if __name__ == "__main__":
    pass
