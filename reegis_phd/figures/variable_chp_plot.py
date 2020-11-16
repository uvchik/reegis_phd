# -*- coding: utf-8 -*-

"""
General description
-------------------
This example shows a complex 6 tiles plot using the i/o-plot function of the
oemof-visio package. This examples focuses on the plotting function and not on
the general oemof functions.


Data
----
variable_chp.csv


Installation requirements
-------------------------
This example requires the version v0.3.x of oemof. Install by:

    pip install 'oemof.solph>=0.4,<0.5'

The oemof-visio provides the base for the created i/o plot.

    pip install git+https://github.com/oemof/oemof_visio.git

5.1.2017 - uwe.krien@rl-institut.de
10.8.2019 - uwe.krien@uni-bremen.de
"""

__copyright__ = "Uwe Krien"
__license__ = "MIT"

import logging
import os

import oemof_visio as oev
import pandas as pd
from matplotlib import pyplot as plt
from oemof import solph
from oemof.network.network import Node


def shape_legend(node, reverse=True, **kwargs):
    handels = kwargs["handles"]
    labels = kwargs["labels"]
    axes = kwargs["ax"]
    parameter = {}

    new_labels = []
    for label in labels:
        label = label.replace("(", "")
        label = label.replace("), flow)", "")
        label = label.replace(node, "")
        label = label.replace("bedarf", "{0}bedarf".format(node))
        label = label.replace("erzeugung", "{0}erzeugung".format(node))
        label = label.replace("ae", "ä")
        label = label.replace(",", "")
        label = label.replace(" ", "")
        new_labels.append(label)
    labels = new_labels

    parameter["bbox_to_anchor"] = kwargs.get("bbox_to_anchor", (1, 0.5))
    parameter["loc"] = kwargs.get("loc", "center left")
    parameter["ncol"] = kwargs.get("ncol", 1)
    plotshare = kwargs.get("plotshare", 0.9)

    if reverse:
        handels.reverse()
        labels.reverse()

    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * plotshare, box.height])

    parameter["handles"] = handels
    parameter["labels"] = labels
    axes.legend(**parameter)
    return axes


def plot():
    logging.info("Initialize the energy system")
    date_time_index = pd.date_range("5/5/2012", periods=192, freq="H")
    energysystem = solph.EnergySystem(timeindex=date_time_index)
    Node.registry = energysystem

    full_filename = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "data",
        "static",
        "variable_chp.csv",
    )
    data = pd.read_csv(full_filename, sep=",").div(1000)

    logging.info("Create oemof.solph objects")

    bgas = solph.Bus(label="natural_gas")
    solph.Source(label="rgas", outputs={bgas: solph.Flow(variable_costs=50)})

    bel = solph.Bus(label="Strom")
    bel2 = solph.Bus(label="Strom_2")
    bth = solph.Bus(label="Waerme")
    bth2 = solph.Bus(label="Waerme_2")

    solph.Sink(label="excess_bth_2", inputs={bth2: solph.Flow()})
    solph.Sink(label="Waermeerzeugung", inputs={bth: solph.Flow()})
    solph.Sink(label="excess_bel_2", inputs={bel2: solph.Flow()})
    solph.Sink(label="Stromerzeugung", inputs={bel: solph.Flow()})

    solph.Sink(
        label="Strombedarf",
        inputs={bel: solph.Flow(fix=data["demand_el"], nominal_value=1)},
    )
    solph.Sink(
        label="Strombedarf_2",
        inputs={bel2: solph.Flow(fix=data["demand_el"], nominal_value=1)},
    )

    solph.Sink(
        label="Waermebedarf",
        inputs={bth: solph.Flow(fix=data["demand_th"], nominal_value=741000)},
    )
    solph.Sink(
        label="Waermebedarf_2",
        inputs={bth2: solph.Flow(fix=data["demand_th"], nominal_value=741000)},
    )

    # This is just a dummy transformer with a nominal input of zero
    # (for the plot)
    solph.Transformer(
        label="KWK_GDT",
        inputs={bgas: solph.Flow(nominal_value=0)},
        outputs={bel: solph.Flow(), bth: solph.Flow()},
        conversion_factors={bel: 0.3, bth: 0.5},
    )

    solph.Transformer(
        label="KWK_GDT_2",
        inputs={bgas: solph.Flow(nominal_value=10e10)},
        outputs={bel2: solph.Flow(), bth2: solph.Flow()},
        conversion_factors={bel2: 0.3, bth2: 0.5},
    )

    solph.components.ExtractionTurbineCHP(
        label="KWK_EKT",
        inputs={bgas: solph.Flow(nominal_value=10e10)},
        outputs={bel: solph.Flow(), bth: solph.Flow()},
        conversion_factors={bel: 0.3, bth: 0.5},
        conversion_factor_full_condensation={bel: 0.5},
    )

    logging.info("Optimise the energy system")
    om = solph.Model(energysystem)
    logging.info("Solve the optimization problem")
    om.solve(solver="cbc", solve_kwargs={"tee": False})

    results = solph.processing.results(om)

    ##########################################################################
    # Plotting
    ##########################################################################

    logging.info("Plot the results")
    smooth_plot = True

    cdict = {
        (("KWK_EKT", "Strom"), "flow"): "#42c77a",
        (("KWK_GDT_2", "Strom_2"), "flow"): "#20b4b6",
        (("KWK_GDT", "Strom"), "flow"): "#20b4b6",
        (("KWK_GDT", "Waerme"), "flow"): "#20b4b6",
        (("KWK_EKT", "Waerme"), "flow"): "#42c77a",
        (("Waerme", "Waermebedarf"), "flow"): "#5b5bae",
        (("Waerme_2", "Waermebedarf_2"), "flow"): "#5b5bae",
        (("Strom", "Strombedarf"), "flow"): "#5b5bae",
        (("Strom_2", "Strombedarf_2"), "flow"): "#5b5bae",
        (("Waerme", "Waermeerzeugung"), "flow"): "#f22222",
        (("Waerme_2", "excess_bth_2"), "flow"): "#f22222",
        (("Strom", "Stromerzeugung"), "flow"): "#f22222",
        (("Strom_2", "excess_bel_2"), "flow"): "#f22222",
        (("KWK_GDT_2", "Waerme_2"), "flow"): "#20b4b6",
    }

    fig = plt.figure(figsize=(18, 9))
    plt.rc("legend", **{"fontsize": 16})
    plt.rcParams.update({"font.size": 16})
    fig.subplots_adjust(
        left=0.06, bottom=0.07, right=0.83, top=0.95, wspace=0.03, hspace=0.2
    )

    # subplot of electricity bus (fixed chp) [1]
    electricity_2 = solph.views.node(results, "Strom_2")
    x_length = len(electricity_2["sequences"].index)
    myplot = oev.plot.io_plot(
        bus_label="Strom_2",
        df=electricity_2["sequences"],
        cdict=cdict,
        smooth=smooth_plot,
        line_kwa={"linewidth": 4},
        ax=fig.add_subplot(4, 2, 1),
        inorder=[(("KWK_GDT_2", "Strom_2"), "flow")],
        outorder=[
            (("Strom_2", "Strombedarf_2"), "flow"),
            (("Strom_2", "excess_bel_2"), "flow"),
        ],
    )
    myplot["ax"].set_ylabel("Leistung [GW]")
    myplot["ax"].set_xlabel("")
    myplot["ax"].get_xaxis().set_visible(False)
    myplot["ax"].set_xlim(0, x_length)
    myplot["ax"].set_title("Stromerzeugung Gegendruckturbine (GDT)")
    myplot["ax"].legend_.remove()

    # subplot of electricity bus (variable chp) [2]
    electricity = solph.views.node(results, "Strom")
    myplot = oev.plot.io_plot(
        bus_label="Strom",
        df=electricity["sequences"],
        cdict=cdict,
        smooth=smooth_plot,
        line_kwa={"linewidth": 4},
        ax=fig.add_subplot(4, 2, 2),
        inorder=[
            (("KWK_GDT", "Strom"), "flow"),
            (("KWK_EKT", "Strom"), "flow"),
        ],
        outorder=[
            (("Strom", "Strombedarf"), "flow"),
            (("Strom", "Stromerzeugung"), "flow"),
        ],
    )
    myplot["ax"].get_yaxis().set_visible(False)
    myplot["ax"].set_xlabel("")
    myplot["ax"].get_xaxis().set_visible(False)
    myplot["ax"].set_title("Stromerzeugung Entnahmekondensationsturbine (EKT)")
    myplot["ax"].set_xlim(0, x_length)
    shape_legend("Strom", plotshare=1, **myplot)

    # subplot of heat bus (fixed chp) [3]
    heat_2 = solph.views.node(results, "Waerme_2")
    myplot = oev.plot.io_plot(
        bus_label="Waerme_2",
        df=heat_2["sequences"],
        cdict=cdict,
        smooth=smooth_plot,
        line_kwa={"linewidth": 4},
        ax=fig.add_subplot(4, 2, 3),
        inorder=[(("KWK_GDT_2", "Waerme_2"), "flow")],
        outorder=[
            (("Waerme_2", "Waermebedarf_2"), "flow"),
            (("Waerme_2", "excess_bth_2"), "flow"),
        ],
    )
    myplot["ax"].set_ylabel("Leistung [GW]")
    myplot["ax"].set_ylim([0, 600])
    myplot["ax"].get_xaxis().set_visible(False)
    myplot["ax"].set_title("Wärmeerzeugung Gegendruckturbine (GDT)")
    myplot["ax"].set_xlim(0, x_length)
    myplot["ax"].legend_.remove()

    # subplot of heat bus (variable chp) [4]
    heat = solph.views.node(results, "Waerme")
    myplot = oev.plot.io_plot(
        bus_label="Waerme",
        df=heat["sequences"],
        cdict=cdict,
        smooth=smooth_plot,
        line_kwa={"linewidth": 4},
        ax=fig.add_subplot(4, 2, 4),
        inorder=[
            (("KWK_GDT", "Waerme"), "flow"),
            (("KWK_EKT", "Waerme"), "flow"),
        ],
        outorder=[
            (("Waerme", "Waermebedarf"), "flow"),
            (("Waerme", "Waermeerzeugung"), "flow"),
        ],
    )
    myplot["ax"].set_ylim([0, 600])
    myplot["ax"].get_yaxis().set_visible(False)
    myplot["ax"].get_xaxis().set_visible(False)
    myplot["ax"].set_title("Wärmeerzeugung Entnahmekondensationsturbine (EKT)")
    myplot["ax"].set_xlim(0, x_length)
    shape_legend("Waerme", plotshare=1, **myplot)

    if smooth_plot:
        style = None
    else:
        style = "steps-mid"

    # subplot of efficiency (fixed chp) [5]
    fix_chp_gas2 = solph.views.node(results, "KWK_GDT_2")
    ngas = fix_chp_gas2["sequences"][(("natural_gas", "KWK_GDT_2"), "flow")]
    df = pd.DataFrame(pd.concat([ngas], axis=1))
    my_ax = df.reset_index(drop=True).plot(
        drawstyle=style, ax=fig.add_subplot(4, 2, 5), linewidth=2
    )
    my_ax.set_ylabel("Leistung [GW]")
    my_ax.set_ylim([0, 1250])
    my_ax.set_xlim(0, x_length)
    my_ax.get_xaxis().set_visible(False)
    my_ax.set_title("Brennstoffzufuhr Gegendruckturbine (GDT)")
    my_ax.legend_.remove()

    # subplot of efficiency (variable chp) [6]
    var_chp_gas = solph.views.node(results, "KWK_EKT")
    ngas = var_chp_gas["sequences"][(("natural_gas", "KWK_EKT"), "flow")]
    ngas.name = "Brennstoffzufuhr"
    df = pd.DataFrame(pd.concat([ngas], axis=1))
    my_ax = df.reset_index(drop=True).plot(
        drawstyle=style, ax=fig.add_subplot(4, 2, 6), linewidth=2
    )
    my_ax.set_ylim([0, 1250])
    my_ax.set_xlim(0, x_length)
    my_ax.get_yaxis().set_visible(False)
    my_ax.get_xaxis().set_visible(False)

    my_ax.set_title("Brennstoffzufuhr  Entnahmekondensationsturbine (EKT))")
    my_box = my_ax.get_position()
    my_ax.set_position([my_box.x0, my_box.y0, my_box.width * 1, my_box.height])
    my_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

    # subplot of efficiency (fixed chp) [7]
    fix_chp_gas2 = solph.views.node(results, "KWK_GDT_2")
    ngas = fix_chp_gas2["sequences"][(("natural_gas", "KWK_GDT_2"), "flow")]
    elec = fix_chp_gas2["sequences"][(("KWK_GDT_2", "Strom_2"), "flow")]
    heat = fix_chp_gas2["sequences"][(("KWK_GDT_2", "Waerme_2"), "flow")]
    e_ef = elec.div(ngas)
    h_ef = heat.div(ngas)
    df = pd.DataFrame(pd.concat([h_ef, e_ef], axis=1))
    my_ax = df.reset_index(drop=True).plot(
        drawstyle=style, ax=fig.add_subplot(4, 2, 7), linewidth=2
    )
    my_ax.set_ylabel("Wirkungsgrad")
    my_ax.set_ylim([0, 0.55])
    my_ax.set_xlabel("Mai 2012")
    my_ax = oev.plot.set_datetime_ticks(
        my_ax,
        df.index,
        tick_distance=24,
        date_format="%d",
        offset=12,
        tight=True,
    )
    my_ax.set_title("Wirkungsgrad Gegendruckturbine (GDT)")
    my_ax.legend_.remove()

    # subplot of efficiency (variable chp) [8]
    var_chp_gas = solph.views.node(results, "KWK_EKT")
    ngas = var_chp_gas["sequences"][(("natural_gas", "KWK_EKT"), "flow")]
    elec = var_chp_gas["sequences"][(("KWK_EKT", "Strom"), "flow")]
    heat = var_chp_gas["sequences"][(("KWK_EKT", "Waerme"), "flow")]
    e_ef = elec.div(ngas)
    h_ef = heat.div(ngas)
    e_ef.name = "Strom           "
    h_ef.name = "Wärme"
    df = pd.DataFrame(pd.concat([h_ef, e_ef], axis=1))
    my_ax = df.reset_index(drop=True).plot(
        drawstyle=style, ax=fig.add_subplot(4, 2, 8), linewidth=2
    )
    my_ax.set_ylim([0, 0.55])
    my_ax = oev.plot.set_datetime_ticks(
        my_ax,
        df.index,
        tick_distance=24,
        date_format="%d",
        offset=12,
        tight=True,
    )
    my_ax.get_yaxis().set_visible(False)
    my_ax.set_xlabel("Mai 2012")

    my_ax.set_title("Wirkungsgrad  Entnahmekondensationsturbine (EKT))")
    my_box = my_ax.get_position()
    my_ax.set_position([my_box.x0, my_box.y0, my_box.width * 1, my_box.height])
    my_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
