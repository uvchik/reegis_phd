import os
import logging
from datetime import datetime

from deflex.scenario_tools import Scenario

from oemof.network import graph
from oemof.tools import logger
from oemof import solph as solph

import pandas as pd
import reegis.config as cfg

from deflex.scenario_tools import Label as Label


def stopwatch():
    if not hasattr(stopwatch, "start"):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def get_file_name_doctests():
    cfg.tmp_set("results", "dir", "results_cbc")

    fn1 = os.path.join(
        cfg.get("paths", "scenario"),
        "deflex",
        "2014",
        "results_cbc",
        "deflex_2014_de21.esys",
    )
    fn2 = os.path.join(
        cfg.get("paths", "scenario"),
        "deflex",
        "2013",
        "results_cbc",
        "deflex_2013_de21.esys",
    )
    return fn1, fn2


def compare_transmission(es1, es2, name1="es1", name2="es2", noreg="DE22"):
    """
    Compare the transmission between two scenarios in a DataFrame.

    Parameters
    ----------
    es1 : oemof.solph.EnergySystem
        Solph energy system with results.
    es2 : oemof.solph.EnergySystem
    name1 : str (optional)
        Short name to identify the first energy system in the DataFrame
    name2 : str (optional)
        Short name to identify the second energy system in the DataFrame
    noreg : str
        Name of a region that should be ignored. At the moment you can only
        choose one region.

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> fn1, fn2 = get_file_name_doctests()
    >>> my_es1 = load_es(fn1)
    >>> my_es2 = load_es(fn2)
    >>> my_df = compare_transmission(my_es1, my_es2, noreg='DE22')
    >>> isinstance(my_df, pd.DataFrame)
    True
    """
    results1 = es1.results["Main"]
    results2 = es2.results["Main"]
    parameter = es1.results["Param"]

    out_flow_lines = [
        x
        for x in results1.keys()
        if ("line" in x[0].label.cat)
        & (noreg not in x[0].label.subtag)
        & (noreg not in x[0].label.region)
    ]

    # Calculate results
    transmission = pd.DataFrame()

    for out_flow in out_flow_lines:

        # es1
        line_a_1 = out_flow[0]
        bus_reg1_1 = [x for x in line_a_1.inputs][0]
        bus_reg2_1 = out_flow[1]
        line_b_1 = es1.groups[
            "_".join(
                [
                    "line",
                    "electricity",
                    bus_reg2_1.label.region,
                    bus_reg1_1.label.region,
                ]
            )
        ]

        # es2
        bus_reg1_2 = es2.groups[str(bus_reg1_1.label)]
        bus_reg2_2 = es2.groups[str(bus_reg2_1.label)]
        line_a_2 = es2.groups[str(line_a_1.label)]
        line_b_2 = es2.groups[str(line_b_1.label)]

        from1to2_es1 = results1[(line_a_1, bus_reg2_1)]["sequences"]
        from2to1_es1 = results1[(line_b_1, bus_reg1_1)]["sequences"]
        from1to2_es2 = results2[(line_a_2, bus_reg2_2)]["sequences"]
        from2to1_es2 = results2[(line_b_2, bus_reg1_2)]["sequences"]

        try:
            line_capacity = parameter[(line_a_1, bus_reg2_1)][
                "scalars"
            ].nominal_value
        except AttributeError:
            line_capacity = float("inf")

        line_name = "-".join([line_a_1.label.subtag, line_a_1.label.region])

        transmission.loc[line_name, "capacity"] = line_capacity

        # Annual balance for each line and each energy system
        transmission.loc[line_name, name1] = (
            float(from1to2_es1.sum() - from2to1_es1.sum()) / 1000
        )
        transmission.loc[line_name, name2] = (
            float(from1to2_es2.sum() - from2to1_es2.sum()) / 1000
        )

        relative_usage1 = (
            (from1to2_es1 + from2to1_es1).div(line_capacity).flow.fillna(0)
        )
        relative_usage2 = (
            (from1to2_es2 + from2to1_es2).div(line_capacity).flow.fillna(0)
        )

        # How many days of min 90% usage
        transmission.loc[line_name, name1 + "_90+_usage"] = (
            relative_usage1.multiply(10)
            .astype(int)
            .value_counts()
            .sort_index()
            .iloc[9:]
            .sum()
        )
        transmission.loc[line_name, name2 + "_90+_usage"] = (
            relative_usage1.multiply(10)
            .astype(int)
            .value_counts()
            .sort_index()
            .iloc[9:]
            .sum()
        )
        transmission.loc[line_name, "diff_2-1_90+_usage"] = (
            transmission.loc[line_name, name2 + "_90+_usage"]
            - transmission.loc[line_name, name1 + "_90+_usage"]
        )

        # How many days of min MAX usage
        transmission.loc[line_name, name1 + "_max_usage"] = (
            relative_usage1.multiply(10)
            .astype(int)
            .value_counts()
            .sort_index()
            .iloc[10:]
            .sum()
        )
        transmission.loc[line_name, name2 + "_max_usage"] = (
            relative_usage1.multiply(10)
            .astype(int)
            .value_counts()
            .sort_index()
            .iloc[10:]
            .sum()
        )
        transmission.loc[line_name, "diff_2-1_max_usage"] = (
            transmission.loc[line_name, name2 + "_max_usage"]
            - transmission.loc[line_name, name1 + "_max_usage"]
        )

        # Average relative usage
        transmission.loc[line_name, name1 + "_avg_usage"] = (
            relative_usage1.mean() * 100
        )
        transmission.loc[line_name, name2 + "_avg_usage"] = (
            relative_usage2.mean() * 100
        )
        transmission.loc[line_name, "diff_2-1_avg_usage"] = (
            transmission.loc[line_name, name2 + "_avg_usage"]
            - transmission.loc[line_name, name1 + "_avg_usage"]
        )

        # Absolute annual flow for each line
        transmission.loc[line_name, name1 + "_abs"] = float(
            from1to2_es1.sum() + from2to1_es1.sum() / 1000
        )
        transmission.loc[line_name, name2 + "_abs"] = float(
            from1to2_es2.sum() + from2to1_es2.sum() / 1000
        )

        # Maximum peak value for each line and each energy system
        transmission.loc[line_name, name1 + "_max"] = (
            float(from1to2_es1.abs().max() - from2to1_es1.abs().max()) / 1000
        )
        transmission.loc[line_name, name2 + "_max"] = (
            float(from1to2_es2.abs().max() - from2to1_es2.abs().max()) / 1000
        )

    transmission["name1"] = transmission[name1]
    transmission["name2"] = transmission[name2]
    transmission["diff_2-1"] = transmission[name2] - transmission[name1]
    transmission["diff_2-1_max"] = (
        transmission[name2 + "_max"] - transmission[name1 + "_max"]
    )
    transmission["diff_2-1_abs"] = (
        transmission[name2 + "_abs"] - transmission[name1 + "_abs"]
    )
    transmission["fraction"] = (
        transmission["diff_2-1"] / abs(transmission[name1]) * 100
    )
    transmission["fraction"].fillna(0, inplace=True)

    return transmission


def reshape_bus_view(es, bus, data=None, aggregate_lines=True):
    """
    Create a MultiIndex DataFrame with all Flows around the given Bus object.

    Parameters
    ----------
    es : oemof.solph.EnergySystem
        Solph energy system with results.
    bus : solph.Bus or list
        Bus
    data : pandas.DataFrame
        MultiIndex DataFrame to add the results to.
    aggregate_lines : bool
        If True all incoming lines will be aggregated to import and outgoing
        lines to export.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> fn1, fn2 = get_file_name_doctests()
    >>> my_es = load_es(fn1)
    >>> my_bus = get_nodes_by_label(
    ...    my_es, ('bus', 'electricity', None, 'DE10'))
    >>> my_df = reshape_bus_view(my_es, my_bus).sum()
    """
    if data is None:
        m_cols = pd.MultiIndex(
            levels=[[], [], [], [], []], codes=[[], [], [], [], []]
        )
        data = pd.DataFrame(columns=m_cols)

    if not isinstance(bus, list):
        buses = [bus]
    else:
        buses = bus

    for bus in buses:
        # filter all nodes and sub-list import/exports
        node_flows = [
            x
            for x in es.results["Main"].keys()
            if (x[1] == bus or x[0] == bus) and x[1] is not None
        ]

        if (
            len(
                [
                    x
                    for x in node_flows
                    if (x[1].label.cat == "line") or (x[0].label.cat == "line")
                ]
            )
            == 0
        ):
            aggregate_lines = False

        # If True all power lines will be aggregated to import/export
        if aggregate_lines is True:
            export_flows = [x for x in node_flows if x[1].label.cat == "line"]
            import_flows = [x for x in node_flows if x[0].label.cat == "line"]

            # only export lines
            export_label = (
                bus.label.region,
                "out",
                "export",
                "electricity",
                "all",
            )

            data[export_label] = (
                es.results["Main"][export_flows[0]]["sequences"]["flow"] * 0
            )
            for export_flow in export_flows:
                data[export_label] += es.results["Main"][export_flow][
                    "sequences"
                ]["flow"]
                node_flows.remove(export_flow)

            # only import lines
            import_label = (
                bus.label.region,
                "in",
                "import",
                "electricity",
                "all",
            )
            data[import_label] = (
                es.results["Main"][import_flows[0]]["sequences"]["flow"] * 0
            )
            for import_flow in import_flows:
                data[import_label] += es.results["Main"][import_flow][
                    "sequences"
                ]["flow"]
                node_flows.remove(import_flow)

        # all flows without lines (import/export)
        for flow in node_flows:
            if flow[0] == bus:
                flow_label = (
                    bus.label.region,
                    "out",
                    flow[1].label.cat,
                    flow[1].label.tag,
                    flow[1].label.subtag,
                )
            elif flow[1] == bus:
                flow_label = (
                    bus.label.region,
                    "in",
                    flow[0].label.cat,
                    flow[0].label.tag,
                    flow[0].label.subtag,
                )
            else:
                flow_label = None

            if flow_label in data:
                data[flow_label] += es.results["Main"][flow]["sequences"][
                    "flow"
                ]
            else:
                data[flow_label] = es.results["Main"][flow]["sequences"][
                    "flow"
                ]

    return data.sort_index(axis=1)


def get_multiregion_bus_balance(es, bus_substring=None):
    """

    Parameters
    ----------
    es : solph.EnergySystem
        An EnergySystem with results.
    bus_substring : str or tuple
        A sub-string or sub-tuple to find a list of buses.

    Returns
    -------
    pd.DataFrame : Multi-regional results.

    Examples
    --------
    >>> fn1, fn2 = get_file_name_doctests()
    >>> my_es = load_es(fn1)
    >>> my_df = get_multiregion_bus_balance(
    ...    my_es, bus_substring='bus_electricity')
    >>> isinstance(my_df, pd.DataFrame)
    True

    """
    # regions = [x for x in es.results['tags']['region']
    #            if re.match(r"DE[0-9][0-9]", x)]

    if bus_substring is None:
        bus_substring = "bus_electricity_all"

    buses = set(
        [
            x[0]
            for x in es.results["Main"].keys()
            if str(bus_substring) in str(x[0].label)
        ]
    )

    data = None
    for bus in sorted(buses):
        data = reshape_bus_view(es, bus, data)

    return data


def check_excess_shortage(es, silent=False):
    """Check if shortage or excess is used in the given EnergySystem."""

    check = True
    result = es.results["Main"]
    flows = [x for x in result.keys() if x[1] is not None]
    ex_nodes = [
        x for x in flows if (x[1] is not None) & ("excess" in x[1].label)
    ]
    sh_nodes = [x for x in flows if "shortage" in x[0].label]
    ex = 0
    sh = 0
    for node in ex_nodes:
        f = solph.views.node(result, node[1])
        s = int(round(f["sequences"].sum()))
        if s > 0:
            if not silent:
                print(node[1], ":", s)
            ex += 1

    for node in sh_nodes:
        f = solph.views.node(result, node[0])
        s = int(round(f["sequences"].sum()))
        if s > 0:
            if not silent:
                print(node[0], ":", s)
            sh += 1

    if sh == 0:
        if not silent:
            print("No shortage usage found.")
    else:
        check = False

    if ex == 0:
        if not silent:
            print("No excess usage found.")
    else:
        check = False

    return check


def load_es(fn=None):
    """
    Load EnergySystem with the given filename (full path). If no file name
    is given a GUI window is popping up.
    """
    # if fn is None:
    #     fn = gui.select_filename(work_dir=cfg.get('paths', 'scenario'),
    #                              title='Select results file',
    #                              extension='esys')

    sc = Scenario()
    logging.debug("Restoring file from {0}".format(fn))
    file_datetime = datetime.fromtimestamp(os.path.getmtime(fn)).strftime(
        "%d. %B %Y - %H:%M:%S"
    )
    logging.info("Restored results created on: {0}".format(file_datetime))
    sc.restore_es(fn)
    return sc.es


if __name__ == "__main__":
    pass
