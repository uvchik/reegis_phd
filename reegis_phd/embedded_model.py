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


# def create_reduced_scenario(year, sim_type):
#     if sim_type == "de21":
#         create_reduced_de21_scenario(year)
#     elif sim_type == "de22":
#         create_reduced_de22_scenario(year)
#     else:
#         logging.error("Wrong sim_type {0}".format(sim_type))


# def remove_region_from_scenario(csv_path, name, year, region):
#     """
#
#     Parameters
#     ----------
#     csv_path
#     name
#     year
#     region : str
#         A region to remove e.g. 'DE22'.
#
#     """
#     de = scenario_tools.DeflexScenario(name=name, year=year)
#     de.load_csv(csv_path)
#     de.check_table("time_series")
#
#     logging.info("Remove region {0}....".format(region))
#     for sheet in de.table_collection.values():
#         if region in sheet.columns:
#             del sheet[region]
#
#     for i in de.table_collection["transmission"].index:
#         if region in i:
#             de.table_collection["transmission"].drop(i, inplace=True)
#
#     new_name = name + "_without_{0}".format(region)
#
#     sce = scenario_tools.Scenario(
#         table_collection=de.table_collection, name=new_name, year=year
#     )
#     path = os.path.join(cfg.get("paths", "scenario"), "deflex", str(year))
#     sce.to_excel(os.path.join(path, new_name + ".xls"))
#     csv_path = os.path.join(path, "{0}_csv".format(new_name))
#     sce.to_csv(csv_path)


# def create_reduced_de22_scenario(year):
#     name = "{0}_{1}_{2}".format("deflex", year, "de22")
#     de = scenario_tools.DeflexScenario(name=name, year=2014)
#     de_path = os.path.join(
#         cfg.get("paths", "scenario"),
#         "deflex",
#         str(year),
#         "{0}_csv".format(name),
#     )
#
#     if not os.path.isdir(de_path):
#         logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
#         basic_scenario.create_basic_scenario(year, rmap="de22")
#
#     de.load_csv(de_path)
#     de.check_table("time_series")
#
#     logging.info("Remove region DE22....")
#     for sheet in de.table_collection.values():
#         if "DE22" in sheet.columns:
#             del sheet["DE22"]
#
#     for i in de.table_collection["transmission"].index:
#         if "DE22" in i:
#             de.table_collection["transmission"].drop(i, inplace=True)
#
#     name = "{0}_{1}_{2}".format("deflex", year, "de22_without_berlin")
#
#     sce = scenario_tools.Scenario(
#         table_collection=de.table_collection, name=name, year=year
#     )
#     path = os.path.join(cfg.get("paths", "scenario"), "berlin_hp", str(year))
#     sce.to_excel(os.path.join(path, name + ".xls"))
#     csv_path = os.path.join(path, "{0}_csv".format(name))
#     sce.to_csv(csv_path)


# def create_reduced_de21_scenario(year):
#     stopwatch()
#
#     logging.info("Read scenarios from excel-sheet: {0}".format(stopwatch()))
#
#     # Berlin
#     name = "{0}_{1}_{2}".format("berlin_hp", year, "single")
#     berlin = berlin_hp.Scenario(name=name, year=year)
#     berlin_fn = os.path.join(
#         cfg.get("paths", "scenario"),
#         "berlin_hp",
#         str(year),
#         "{0}_csv".format(name),
#     )
#     if not os.path.isdir(berlin_fn):
#         logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
#         berlin_hp.basic_scenario.create_basic_scenario(year)
#
#     berlin.load_csv(berlin_fn.format(year=year))
#     berlin.check_table("time_series")
#
#     berlin.table_collection["time_series"].reset_index(drop=True, inplace=True)
#
#     # de21
#     name = "{0}_{1}_{2}".format("deflex", year, "de21")
#     de = scenario_tools.DeflexScenario(name=name, year=year)
#     de_path = os.path.join(
#         cfg.get("paths", "scenario"),
#         "deflex",
#         str(year),
#         "{0}_csv".format(name),
#     )
#
#     if not os.path.isdir(de_path):
#         logging.info("Create scenario for {0}: {1}".format(stopwatch(), name))
#         basic_scenario.create_basic_scenario(year, rmap="de21")
#
#     de.load_csv(de_path.format(year=year))
#     de.check_table("time_series")
#
#     # control table
#     ct = pd.DataFrame(
#         columns=["DE_orig", "DE01_orig", "BE", "DE01_new", "DE_new"]
#     )
#
#     region = "DE01"
#
#     de_be = {
#         "natural gas": "gas",
#         "hard coal": "coal",
#         "wind": "Wind",
#         "solar": "Solar",
#         "oil": "oil",
#         "geothermal": "Geothermal",
#         "hydro": "Hydro",
#     }
#
#     # TODO Check units (s. below)
#
#     # Demand of all district heating systems
#     berlin_district_heating = berlin.table_collection["time_series"][
#         "district_heating_demand"
#     ]
#     berlin_district_heating = berlin_district_heating.sum(axis=1)
#
#     ct.loc["district heating", "DE01_orig"] = round(
#         de.table_collection["time_series"][region, "district heating"].sum()
#     )
#     ct.loc["district heating", "BE"] = round(berlin_district_heating.sum())
#
#     de.table_collection["time_series"][
#         region, "district heating"
#     ] -= berlin_district_heating
#
#     ct.loc["district heating", "DE01_new"] = round(
#         de.table_collection["time_series"][region, "district heating"].sum()
#     )
#
#     # Electricity demand
#     berlin_elec_demand = berlin.table_collection["time_series"][
#         "electricity", "demand"
#     ]
#
#     ct.loc["electricity demand", "DE01_orig"] = round(
#         de.table_collection["time_series"][region, "electrical_load"].sum()
#     )
#     ct.loc["electricity demand", "BE"] = round(berlin_elec_demand.sum())
#
#     de.table_collection["time_series"][
#         region, "electrical_load"
#     ] -= berlin_elec_demand
#
#     ct.loc["electricity demand", "DE01_new"] = round(
#         de.table_collection["time_series"][region, "electrical_load"].sum()
#     )
#
#     # Decentralised heating
#     # TODO: electricity ????
#     # gas und natural_gas bei DE????
#     # Einheiten kontrollieren!
#
#     dch_berlin = list(
#         berlin.table_collection["time_series"]["decentralised_demand"].columns
#     )
#
#     for col in de.table_collection["time_series"]["DE_demand"].columns:
#         ct.loc["decentralised_" + col, "DE_orig"] = round(
#             de.table_collection["time_series"]["DE_demand", col].sum()
#         )
#         ct.loc["decentralised_" + col, "BE"] = round(
#             (
#                 berlin.table_collection["time_series"].get(
#                     ("decentralised_demand", de_be.get(col)), pd.Series([0, 0])
#                 )
#             ).sum()
#         )
#
#         de.table_collection["time_series"][
#             "DE_demand", col
#         ] -= berlin.table_collection["time_series"].get(
#             ("decentralised_demand", de_be.get(col)), 0
#         )
#
#         ct.loc["decentralised_" + col, "DE_new"] = round(
#             de.table_collection["time_series"]["DE_demand", col].sum()
#         )
#
#         if de_be.get(col) in dch_berlin:
#             dch_berlin.remove(de_be.get(col))
#
#     for col in dch_berlin:
#         ct.loc["decentralised_" + col, "BE"] = round(
#             (
#                 berlin.table_collection["time_series"][
#                     "decentralised_demand", col
#                 ]
#             ).sum()
#         )
#
#     # Volatile Sources
#     vs_berlin = list(berlin.table_collection["volatile_source"]["BE"].columns)
#     for col in de.table_collection["volatile_source"][region].columns:
#         ct.loc["re_" + col, "DE01_orig"] = round(
#             float(de.table_collection["volatile_source"][region, col])
#         )
#         # if de_be.get(col) in vs_berlin:
#         de.table_collection["volatile_source"][
#             region, col
#         ] -= berlin.table_collection["volatile_source"].get(
#             ("BE", de_be[col]), 0
#         )
#         ct.loc["re_" + col, "BE"] = round(
#             float(
#                 berlin.table_collection["volatile_source"].get(
#                     ("BE", de_be[col]), 0
#                 )
#             )
#         )
#
#         if de_be.get(col) in vs_berlin:
#             vs_berlin.remove(de_be.get(col))
#
#         ct.loc["re_" + col, "DE01_new"] = round(
#             float(de.table_collection["volatile_source"][region, col])
#         )
#
#     for col in vs_berlin:
#         ct.loc["re_" + col, "BE"] = round(
#             float(berlin.table_collection["volatile_source"]["BE", col])
#         )
#
#     # Elec. Storages
#     pass
#
#     # Power Plants
#     sub = pd.DataFrame(
#         columns=["DE_orig", "DE01_orig", "BE", "DE01_new", "DE_new"]
#     )
#
#     import reegis.powerplants
#
#     pwp = reegis.powerplants.get_pp_by_year(
#         year, overwrite_capacity=True, capacity_in=True
#     )
#
#     table_collect = basic_scenario.powerplants(
#         pwp, {}, year, region_column="federal_states"
#     )
#
#     heat_b = reegis.powerplants.get_chp_share_and_efficiency_states(year)
#
#     heat_b["BE"]["fuel_share"].rename(
#         columns={"re": "bioenergy"}, inplace=True
#     )
#
#     heat_demand = pd.DataFrame(
#         berlin_district_heating, columns=["district heating"]
#     )
#
#     heat_demand = pd.concat([heat_demand], axis=1, keys=["BE"]).sort_index(1)
#
#     table_collect = basic_scenario.chp_table(
#         heat_b, heat_demand, table_collect, regions=["BE"]
#     )
#
#     rows = [
#         r
#         for r in de.table_collection["transformer"].index
#         if "efficiency" not in r
#     ]
#
#     sub["BE"] = (table_collect["transformer"].loc[rows, "BE"]).sum(axis=1)
#
#     sub["DE01_orig"] = (
#         de.table_collection["transformer"].loc[rows, region].sum(axis=1)
#     )
#
#     asd = de.table_collection["transformer"].loc[rows, region]
#     bsd = table_collect["transformer"].loc[rows, "BE"]
#
#     for col in de.table_collection["transformer"][region].columns:
#         de.table_collection["transformer"].loc[
#             rows, (region, col)
#         ] -= table_collect["transformer"].loc[rows, ("BE", col)]
#         de.table_collection["transformer"].loc[rows, (region, col)] = (
#             de.table_collection["transformer"]
#             .loc[rows, (region, col)]
#             .fillna(float("inf"))
#         )
#
#     sub["DE01_new"] = (
#         de.table_collection["transformer"].loc[rows, region].sum(axis=1)
#     )
#     csd = de.table_collection["transformer"].loc[rows, region]
#     pd.concat([asd, bsd, csd]).to_excel(
#         os.path.join(
#             cfg.get("paths", "messages"), "summery_embedded_powerplants.xls"
#         )
#     )
#     ct = pd.concat([ct, sub])
#
#     ct.to_excel(
#         os.path.join(
#             cfg.get("paths", "messages"), "summery_embedded_model.xls"
#         )
#     )
#     name = "{0}_{1}_{2}".format("deflex", year, "de21_without_berlin")
#     sce = scenario_tools.Scenario(
#         table_collection=de.table_collection, name=name, year=year
#     )
#     path = os.path.join(cfg.get("paths", "scenario"), "berlin_hp", str(year))
#     sce.to_excel(os.path.join(path, name + ".xls"))
#     csv_path = os.path.join(path, "{0}_csv".format(name))
#     sce.to_csv(csv_path)


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
    n1 = os.path.basename(fn_de).split(".")[0]
    n2 = os.path.basename(fn_be).split(".")[0]
    name = "{0}_dcpl_{1}".format(n1, n2)
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
