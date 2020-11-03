import os

import pandas as pd
from berlin_hp import heat
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.sankey import Sankey
from oemof import solph
from reegis import config as cfg
from reegis import energy_balance
from reegis import geometries
from reegis import inhabitants

from .. import friedrichshagen_scenarios as fhg_sc
from .. import reegis_plot as plot
from .. import regional_results
from .. import reproduce
from .. import results
from .figures_base import NAMES
from .figures_base import create_subplot


def sankey_test():

    Sankey(
        flows=[1, -5727 / 22309, -14168 / 22309, -1682 / 22309, -727 / 22309],
        labels=[" ", " ", " ", " ", " "],
        orientations=[-1, 1, 0, -1, 1],
    ).finish()
    plt.title("The default settings produce a diagram like this.")
    return "sankey_test", None


def fig_anteil_import_stromverbrauch_berlin(**kwargs):

    ax = create_subplot((8.5, 5), **kwargs)
    fn_csv = os.path.join(
        os.path.dirname(__file__),
        "../data",
        "static",
        "electricity_import.csv",
    )
    df = pd.read_csv(fn_csv)
    df["Jahr"] = df["year"].astype(str)

    print("10-Jahresmittel:", df["elec_import"].sum() / df["elec_usage"].sum())

    df["Erzeugung [TWh]"] = (df["elec_usage"] - df["elec_import"]).div(3600)
    df["Import [TWh]"] = df["elec_import"].div(3600)

    df["Importanteil [%]"] = df["elec_import"] / df["elec_usage"] * 100
    ax1 = df[["Jahr", "Importanteil [%]"]].plot(
        x="Jahr",
        linestyle="-",
        marker="o",
        secondary_y=True,
        color="#555555",
        ax=ax,
    )
    df[["Jahr", "Import [TWh]", "Erzeugung [TWh]"]].plot(
        x="Jahr",
        kind="bar",
        ax=ax1,
        stacked=True,
        color=["#343e58", "#aebde3"],
    )
    ax1.set_ylim(0, 100)

    h0, l0 = ax.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax.legend(h0 + h1, l0 + l1, bbox_to_anchor=(1.06, 0.8))
    plt.subplots_adjust(right=0.74, left=0.05)
    return "anteil_import_stromverbrauch_berlin", None


def fig_netzkapazitaet_und_auslastung_de22():
    cm_gyr = LinearSegmentedColormap.from_list(
        "gyr",
        [(0, "#aaaaaa"), (0.0001, "green"), (0.5, "#d5b200"), (1, "red")],
    )

    static = LinearSegmentedColormap.from_list(
        "static", [(0, "red"), (0.001, "#555555"), (1, "#000000")]
    )

    sets = {
        "capacity": {
            "key": "capacity",
            "vmax": 10,
            "label_min": 0,
            "label_max": None,
            "unit": "",
            "order": 0,
            "direction": False,
            "cmap_lines": static,
            "legend": False,
            "unit_to_label": False,
            "divide": 1000,
            "decimal": 1,
            "my_legend": False,
            "part_title": "Kapazität in MW",
        },
        "absolut": {
            "key": "es1_90+_usage",
            "vmax": 8760 / 2,
            "label_min": 10,
            "unit": "",
            "order": 1,
            "direction": False,
            "cmap_lines": cm_gyr,
            "legend": False,
            "unit_to_label": False,
            "divide": 1,
            "my_legend": True,
            "part_title": "Stunden mit über 90% Auslastung \n",
        },
    }
    path = os.path.join(cfg.get("paths", "phd"), "base", "results_cbc")
    my_es1 = results.load_es(os.path.join(path, "c1_deflex_2014_de21.esys"))
    transmission = results.compare_transmission(my_es1, my_es1)

    # f, ax_ar = plt.subplots(1, 2, figsize=(15, 6))
    f, ax_ar = plt.subplots(2, 1, figsize=(7.8, 10.6))

    for k, v in sets.items():
        if len(sets) == 1:
            v["ax"] = ax_ar
            v.pop("order")
        else:
            v["ax"] = ax_ar[v.pop("order")]
        my_legend = v.pop("my_legend")
        v["ax"].set_title(v.pop("part_title"))
        plot.plot_power_lines(transmission, **v)
        if my_legend is True:
            plot.geopandas_colorbar_same_height(
                f, v["ax"], 0, v["vmax"], v["cmap_lines"]
            )
        plt.title(v["unit"])
    plt.subplots_adjust(right=0.96, left=0, hspace=0.15, bottom=0.01, top=0.97)

    return "netzkapazität_und_auslastung_de22", None


def fig_veraenderung_energiefluesse_durch_kopplung():
    year = 2014

    cm_gyr = LinearSegmentedColormap.from_list(
        "mycmap",
        [(0, "#aaaaaa"), (0.01, "green"), (0.5, "#d5b200"), (1, "red")],
    )

    sets = {
        "fraction": {
            "key": "diff_2-1_avg_usage",
            "vmax": 5,
            "label_min": 1,
            "label_max": None,
            "unit": "%-Punkte",
            "order": 1,
            "direction": False,
            "cmap_lines": cm_gyr,
            "legend": False,
            "unit_to_label": False,
        },
        "absolut": {
            "key": "diff_2-1",
            "vmax": 500,
            "label_min": 100,
            "unit": "GWh",
            "order": 0,
            "direction": True,
            "cmap_lines": cm_gyr,
            "legend": False,
            "unit_to_label": False,
        },
    }

    bpath = os.path.join(cfg.get("paths", "phd"), "base", "results_cbc")
    rpath = os.path.join(cfg.get("paths", "phd"), "region", "results_cbc")
    namees1 = "deflex_2014_de22.esys"
    namees2 = "c1_deflex_2014_de22_without_berlin_dcpl_berlin_hp_2014_single.esys"
    my_es1 = results.load_es(os.path.join(bpath, namees1))
    my_es2 = results.load_es(os.path.join(rpath, namees2))

    transmission = results.compare_transmission(my_es1, my_es2).div(1)

    f, ax_ar = plt.subplots(1, len(sets), figsize=(8 * len(sets), 6))

    for k, v in sets.items():
        if len(sets) == 1:
            v["ax"] = ax_ar
            v.pop("order")
        else:
            v["ax"] = ax_ar[v.pop("order")]
        plot.plot_power_lines(transmission, **v)
        plot.geopandas_colorbar_same_height(
            f, v["ax"], 0, v["vmax"], v["cmap_lines"]
        )
        plt.title(v["unit"])
    plt.subplots_adjust(right=0.94, left=0, wspace=0, bottom=0.03, top=0.96)

    return "veraenderung_energiefluesse_durch_kopplung", None


def fig_absolute_power_flows():
    year = 2014

    cm_gyr = LinearSegmentedColormap.from_list(
        "mycmap",
        [(0, "#aaaaaa"), (0.01, "green"), (0.5, "#d5b200"), (1, "red")],
    )

    sets = {
        "fraction": {
            "key": "es1",
            "vmax": 500,
            "label_min": 100,
            "label_max": None,
            "unit": "GWh",
            "order": 0,
            "direction": True,
            "cmap_lines": cm_gyr,
            "legend": False,
            "unit_to_label": False,
            "part_title": "es1",
        },
        "absolut": {
            "key": "es2",
            "vmax": 500,
            "label_min": 100,
            "unit": "GWh",
            "order": 1,
            "direction": True,
            "cmap_lines": cm_gyr,
            "legend": False,
            "unit_to_label": False,
            "part_title": "es2",
        },
    }

    my_es1 = results.load_my_es("deflex", str(year), var="de22")
    my_es2 = results.load_my_es("berlin_hp", str(year), var="de22")
    transmission = results.compare_transmission(my_es1, my_es2).div(1)

    f, ax_ar = plt.subplots(1, 2, figsize=(15, 6))
    for k, v in sets.items():
        v["ax"] = ax_ar[v.pop("order")]
        v["ax"].set_title(v.pop("part_title"))
        plot.plot_power_lines(transmission, **v)
        plot.geopandas_colorbar_same_height(
            f, v["ax"], 0, v["vmax"], v["cmap_lines"]
        )
        # v['ax'].set_title(v.pop('part_title'))
        plt.title(v["unit"])
    plt.subplots_adjust(right=0.97, left=0, wspace=0, bottom=0.03, top=0.96)

    return "absolute_energiefluesse_vor_nach_kopplung", None


def fig_deflex_de22_polygons(**kwargs):
    ax = create_subplot((9, 7), **kwargs)

    # change for a better/worse resolution (
    simple = 0.02

    fn = os.path.join(
        cfg.get("paths", "geo_plot"), "region_polygon_de22_reegis.csv"
    )

    reg_id = ["DE{num:02d}".format(num=x + 1) for x in range(22)]
    idx = [x + 1 for x in range(22)]
    data = pd.DataFrame({"reg_id": reg_id}, index=idx)
    data["class"] = 0
    data.loc[[19, 20, 21], "class"] = 1
    data.loc[22, "class"] = 0.5
    data.loc[22, "reg_id"] = ""

    cmap = LinearSegmentedColormap.from_list(
        "mycmap", [(0.000000000, "#badd69"), (0.5, "#dd5500"), (1, "#a5bfdd")]
    )

    ax = plot.plot_regions(
        edgecolor="#666666",
        data=data,
        legend=False,
        label_col="reg_id",
        fn=fn,
        data_col="class",
        cmap=cmap,
        ax=ax,
        simple=simple,
    )
    plt.subplots_adjust(right=1, left=0, bottom=0, top=1)

    ax.set_axis_off()
    return "deflex_de22_polygons", None


def fig_6_x_draft1(**kwargs):

    ax = create_subplot((5, 5), **kwargs)

    my_es1 = results.load_my_es("deflex", "2014", var="de21")
    my_es2 = results.load_my_es("deflex", "2014", var="de22")
    # my_es_2 = results.load_es(2014, 'de22', 'berlin_hp')
    transmission = results.compare_transmission(my_es1, my_es2)

    # PLOTS
    transmission = transmission.div(1000)
    transmission.plot(kind="bar", ax=ax)
    return "name_6_x", None


def fig_district_heating_areas(**kwargs):
    ax = create_subplot((7.8, 4), **kwargs)

    # get groups of district heating systems in Berlin
    district_heating_groups = pd.DataFrame(
        pd.Series(cfg.get_dict("district_heating_systems")), columns=["name"]
    )

    # get district heating system areas in Berlin
    distr_heat_areas = heat.get_district_heating_areas()

    # Merge main groups on map
    distr_heat_areas = distr_heat_areas.merge(
        district_heating_groups, left_on="KLASSENNAM", right_index=True
    )

    # Create real geometries
    distr_heat_areas = geometries.create_geo_df(distr_heat_areas)

    # Plot berlin map
    berlin_fn = os.path.join(cfg.get("paths", "geo_berlin"), "berlin.csv")
    berlin = geometries.create_geo_df(pd.read_csv(berlin_fn))
    ax = berlin.plot(color="#ffffff", edgecolor="black", ax=ax)

    # Plot areas of district heating system groups
    ax = distr_heat_areas.loc[
        distr_heat_areas["name"] != "decentralised_dh"
    ].plot(column="name", ax=ax, cmap="tab10")

    # Remove frame around plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axis("off")

    text = {
        "Vattenfall 1": (13.3, 52.52),
        "Vattenfall 2": (13.5, 52.535),
        "Buch": (13.47, 52.63),
        "Märkisches Viertel": (13.31, 52.61),
        "Neukölln": (13.422, 52.47),
        "BTB": (13.483, 52.443),
        "Köpenick": (13.58, 52.43),
        "Friedrichshagen": (13.653, 52.44),
    }

    for t, c in text.items():
        plt.text(
            c[0],
            c[1],
            t,
            size=6,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", alpha=0.5, ec=(1, 1, 1), fc=(1, 1, 1)),
        )
    plt.draw()
    return "distric_heating_areas", None


def plot_upstream():
    year = 2014
    sc = fhg_sc.load_upstream_scenario_values()
    cols = [
        "deflex_{0}_de21_without_berlin",
        "deflex_{0}_de22_without_berlin",
        "deflex_{0}_de22",
        "deflex_2014_de21",
    ]
    cols = [c.format(year) for c in cols]

    ax = sc["deflex_2014_de22", "meritorder"].plot()
    ax = sc["deflex_2014_de22_without_berlin", "meritorder"].plot(ax=ax)
    ax = sc["deflex_2014_de21", "meritorder"].plot(ax=ax)
    ax = sc["deflex_2014_de21_without_berlin", "meritorder"].plot(ax=ax)
    ax.legend()
    sc[cols].mean().unstack()[["levelized"]].plot(kind="bar")
    print(sc[cols].mean().unstack()["meritorder"])
    print(sc[cols].mean().unstack()["levelized"])
    return "upstream", None


def fig_show_de21_de22_without_berlin():
    plt.rcParams.update({"font.size": 13})
    year = 2014
    figs = ("de21", "Berlin", "de22", "de21_without_berlin")

    y_annotate = {
        "de21": 10,
        "de22": 1000,
        "de21_without_berlin": 1000,
        "Berlin": 1000,
    }

    title_str = {
        "de21": "DE01 in de21, Jahressumme: {0} GWh",
        "de22": "DE01 in de22, Jahressumme: {0} GWh",
        "de21_without_berlin": "DE01 in de21 ohne Berlin, Jahressumme: {0} GWh",
        "Berlin": "Berlin in berlin_hp, Jahressumme: {0} GWh",
    }

    ax = {}
    f, ax_ar = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(15, 10))

    i = 0
    for n in range(2):
        for m in range(2):
            ax[figs[i]] = ax_ar[n, m]
            i += 1

    ol = [
        "lignite",
        "coal",
        "natural_gas",
        "bioenergy",
        "oil",
        "other",
        "shortage",
    ]

    data_sets = {}
    period = (576, 650)

    base_path = os.path.join(cfg.get("paths", "phd"), "base", "results_cbc")
    reg_path = os.path.join(cfg.get("paths", "phd"), "region", "results_cbc")
    fn = {
        "de21": os.path.join(base_path, "c1_deflex_2014_de21.esys"),
        "de22": os.path.join(base_path, "c1_deflex_2014_de22.esys"),
        "de21_without_berlin": os.path.join(
            base_path, "c1_deflex_2014_de21_without_berlin.esys"
        ),
        "Berlin": os.path.join(reg_path, "berlin_hp_2014_single.esys"),
    }

    for var in ("de21", "de22", "de21_without_berlin"):
        data_sets[var] = {}
        if not os.path.isfile(fn[var]):
            pass
            # reproduce.reproduce_scenario("{0}_{1}".format(var, year))
        es = results.load_es(fn=fn[var])
        bus = [
            b[0]
            for b in es.results["Main"]
            if (b[0].label.region == "DE01")
            & (b[0].label.tag == "heat")
            & (isinstance(b[0], solph.Bus))
        ][0]
        data = results.reshape_bus_view(es, bus)[bus.label.region]

        results.check_excess_shortage(es)
        # if data.sum().sum() > 500000:
        #     data *= 0.5
        annual = round(data["out", "demand"].sum().sum(), -2)
        data = data.iloc[period[0] : period[1]]
        data_sets[var]["data"] = data
        data_sets[var]["title"] = title_str[var].format(int(annual / 1000))

    var = "Berlin"
    data_sets[var] = {}
    if not os.path.isfile(fn[var]):
        raise FileNotFoundError("File not found: {0}".format(fn[var]))
    es = results.load_es(fn=fn[var])
    data = (
        results.get_multiregion_bus_balance(es, "district")
        .groupby(axis=1, level=[1, 2, 3, 4])
        .sum()
    )
    data.rename(columns={"waste": "other"}, level=3, inplace=True)
    annual = round(data["out", "demand"].sum().sum(), -2)
    data = data.iloc[period[0] : period[1]]
    data_sets[var]["data"] = data
    data_sets[var]["title"] = title_str[var].format(int(annual / 1000))
    results.check_excess_shortage(es)

    i = 0
    for k in figs:
        v = data_sets[k]
        if i == 1:
            legend = True
        else:
            legend = False

        av = float(v["data"].iloc[5]["out", "demand"].sum())
        print(float(v["data"].iloc[6]["out", "demand"].sum()))

        a = plot.plot_bus_view(
            data=v["data"],
            ax=ax[k],
            legend=legend,
            xlabel="",
            ylabel="Leistung [MW]",
            title=v["title"],
            in_ol=ol,
            out_ol=["demand"],
            smooth=False,
        )
        a.annotate(
            str(int(av)),
            xy=(5, av),
            xytext=(12, av + y_annotate[k]),
            fontsize=14,
            arrowprops=dict(
                facecolor="black",
                arrowstyle="->",
                connectionstyle="arc3,rad=0.2",
            ),
        )
        i += 1

    plt.subplots_adjust(
        right=0.81, left=0.06, bottom=0.08, top=0.95, wspace=0.06
    )
    plt.arrow(600, 600, 200, 200)
    return "compare_district_heating_de01_without_berlin", None


def berlin_resources_time_series():
    """Plot time series of resource usage."""
    seq = regional_results.analyse_berlin_ressources()
    types = ["lignite", "natural_gas", "oil", "hard_coal", "netto_import"]
    rows = len(types)
    f, ax_ar = plt.subplots(rows, 2, sharey="row", sharex=True, figsize=(9, 6))
    i = 0
    axr = None
    for c in types:
        if c in ["lignite", "natural_gas", "oil"]:
            my_style = ["-", "-", "--"]
        else:
            my_style = None
        axl = (
            seq[
                [
                    (c, "deflex_de22"),
                    (c, "berlin_deflex"),
                    (c, "berlin_up_deflex_full"),
                ]
            ]
            .multiply(1000)
            .resample("D")
            .mean()
            .plot(ax=ax_ar[i][0], legend=False)
        )
        axr = (
            seq[
                [
                    (c, "deflex_de22"),
                    (c, "berlin_deflex"),
                    (c, "berlin_up_deflex_full"),
                ]
            ]
            .multiply(1000)
            .resample("M")
            .mean()
            .plot(ax=ax_ar[i][1], legend=False, style=my_style)
        )
        # axr.set_xlim([seq.index[0], seq.index[8759]])
        # axl.set_xlim([seq.index[0], seq.index[8650]])
        axr.text(
            seq.index[8100],
            axr.get_ylim()[1] / 2,
            NAMES[c],
            size=12,
            verticalalignment="center",
            horizontalalignment="left",
            rotation=270,
        )
        axl.text(
            seq.index[0] - seq.index.freq * 1200,
            axr.get_ylim()[1] / 2,
            "[GW]",
            size=12,
            verticalalignment="center",
            horizontalalignment="left",
            rotation=90,
        )
        i += 1

    for i in range(rows):
        for j in range(2):
            ax = ax_ar[i, j]
            if i == 0 and j == 0:
                ax.set_title("Tagesmittel", loc="center", y=1)
            if i == 0 and j == 1:
                ax.set_title("Monatsmittel", loc="center", y=1)

    plt.subplots_adjust(
        right=0.96, left=0.07, bottom=0.13, top=0.95, wspace=0.06, hspace=0.2
    )

    handles, labels = axr.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        new = lab.split(",")[1][:-1]
        new_labels.append(new.replace("_full", ""))

    plt.legend(
        handles,
        new_labels,
        bbox_to_anchor=(0, -1),
        loc="lower center",
        ncol=3,
        prop={"size": 14},
    )
    return "ressource_use_berlin_time_series", None


def fig_berlin_resources(**kwargs):
    ax = create_subplot((7.8, 4), **kwargs)

    df = regional_results.analyse_berlin_ressources_total()

    df = df.loc[
        [
            # "berlin_single",
            "berlin_deflex",
            "berlin_up_deflex",
            "berlin_up_deflex_full",
            "deflex_de22",
            "statistic",
        ]
    ]
    df = df.drop(df.sum().loc[df.sum() < 0.1].index, axis=1)
    color_dict = plot.get_cdict_df(df)

    ax = df.plot(
        kind="bar",
        ax=ax,
        color=[color_dict.get(x, "#bbbbbb") for x in df.columns],
    )
    plt.subplots_adjust(right=0.79)

    # Adapt legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        new_labels.append(NAMES[label])
    plt.legend(
        handles, new_labels, bbox_to_anchor=(1.3, 1), loc="upper right", ncol=1
    )

    # Adapt xticks
    locs, labels = plt.xticks()
    new_labels = []
    for label in labels:
        new_labels.append(label.get_text().replace("_full", ""))
    #     if 'up' in label.get_text():
    #         new_labels.append(label.get_text().replace('up_', 'up_\n'))
    #     else:
    #         new_labels.append(label.get_text().replace('_', '_\n'))
    plt.xticks(locs, new_labels, rotation=0)

    plt.ylabel("Energiemenge 2014 [TWh]")
    plt.subplots_adjust(right=0.78, left=0.08, bottom=0.12, top=0.98)
    return "resource_use_berlin_reduced", None


def fig_import_export_100prz_region():
    plt.rcParams.update({"font.size": 13})
    f, ax_ar = plt.subplots(1, 3, figsize=(15, 6))

    myp = os.path.join(cfg.get("paths", "phd"), "region", "results_cbc")

    my_filenames = [x for x in os.listdir(myp) if ".esys" in x and "_pv" in x]

    bil = pd.DataFrame()
    expdf = pd.DataFrame()
    for mf in sorted(my_filenames):
        my_fn = os.path.join(myp, mf)
        my_es = results.load_es(my_fn)
        res = my_es.results["param"]
        wind = int(
            round(
                [
                    res[w]["scalars"]["nominal_value"]
                    for w in res
                    if w[0].label.subtag == "Wind" and w[1] is not None
                ][0]
            )
        )
        solar = int(
            round(
                [
                    res[w]["scalars"]["nominal_value"]
                    for w in res
                    if w[0].label.subtag == "Solar" and w[1] is not None
                ][0]
            )
        )
        key = "w {0:02}, pv {1:02}".format(wind, solar)
        my_df = results.get_multiregion_bus_balance(my_es)
        imp = my_df["FHG", "in", "source", "import", "electricity"].div(1000)
        exp = my_df["FHG", "out", "sink", "export", "electricity"].div(1000)
        demand = my_df["FHG", "out", "demand", "electricity", "all"].div(1000)
        expdf.loc[key, "export"] = float(exp.sum())
        expdf.loc[key, "import"] = float(imp.sum())
        print("Autarkie:", (1 - float(exp.sum()) / demand.sum()) * 100, "%")
        if wind == 0:
            bil["export"] = exp.resample("M").sum()
            bil["import"] = imp.resample("M").sum()
            ax_ar[1] = bil.reset_index(drop=True).plot(
                ax=ax_ar[1], drawstyle="steps-mid", linewidth=2
            )
            ax_ar[1].set_xlabel("2014\n\nWind: 0 MW, PV: 67 MWp")
            ax_ar[1].legend(loc="upper left")
        if solar == 0:
            bil["export"] = exp.resample("M").sum()
            bil["import"] = imp.resample("M").sum()
            ax_ar[2] = bil.reset_index(drop=True).plot(
                ax=ax_ar[2], drawstyle="steps-mid", linewidth=2
            )
            ax_ar[2].set_xlabel("2014\n\nWind: 39 MW, PV: 0 MWp")
    ax_ar[0] = expdf.sort_index().plot(kind="bar", ax=ax_ar[0])
    ax_ar[0].set_ylabel("Energie [GWh]")
    for n in [1, 2]:
        ax_ar[n].set_ylim(0, 7)
        ax_ar[n].set_xlim(0, 11)
        ax_ar[n].set_xticks(list(range(12)))
        ax_ar[n].set_xticklabels(
            pd.date_range("2014", "2015", freq="MS").strftime("%b")[:-1]
        )
        # ,
        # rotation="horizontal",
        # horizontalalignment="center",
    # )
    plt.subplots_adjust(right=0.98, left=0.06, bottom=0.2, top=0.96)
    return "import_export_100PRZ_region", None


def fig_import_export_emissions_100prz_region():
    f, ax_ar = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(15, 6))
    # my_filename = "market_clearing_price_{0}_{1}.csv"
    my_path = os.path.join(cfg.get("paths", "phd"))
    up_raw = {}
    # up_df = pd.read_csv(
    #     up_fn.format(2014, "cbc"), index_col=[0], header=[0, 1, 2]
    # )
    myp = os.path.join(my_path, "region", "results_cbc")

    my_filenames = [x for x in os.listdir(myp) if ".esys" in x and "_pv" in x]

    # my_list = [
    #     x
    #     for x in up_df.columns.get_level_values(1).unique()
    #     if "f10" in x or "f15" in x or "f20" in x
    # ]
    # my_list = [x for x in my_list if "de21" in x]
    # my_list = [x for x in my_list if "Li1_HP0_" in x]

    for f in ["f1", "f15", "f2"]:
        fn_pattern = "deflex_XX_Nc00_HP00_{0}_de21.csv"
        fn = os.path.join(my_path, "values", fn_pattern.format(f))
        up_raw[f] = pd.read_csv(fn, index_col=[0], header=[0, 1, 2, 3, 4, 5])
        # deflex_XX_Nc00_HP00_f2_de21.csv
        # deflex_XX_Nc00_HP00_f15_de21.csv

    bil = pd.DataFrame()
    # print(up_df.columns.get_level_values(2).unique())
    up = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []]))
    up_plot = {}
    # for t1 in ["emission", "emission_avg", "emission_max"]:
    for f in ["f1", "f15", "f2"]:
        up[f, "emission_max"] = up_raw[f]["emission"].max(1)
        up[f, "mcp"] = up_raw[f]["cost", "specific"].max(axis=1)
        mcp_id = up_raw[f]["cost", "specific"].idxmax(axis=1)
        emissions = up_raw[f]["emission", "specific"]
        up[f, "emission_avg"] = (
            up_raw[f]["emission", "absolute"]
            .sum(axis=1)
            .div(up_raw[f]["values", "absolute"].sum(axis=1))
        )
        up[f, "mcpe"] = pd.Series(
            emissions.lookup(*zip(*pd.DataFrame(data=mcp_id).to_records()))
        )
    for t2 in ["mcpe", "emission_avg", "emission_max"]:
        up_plot[t2] = {}
        for f in ["f1", "f15", "f2"]:
            up_plot[t2][f] = pd.DataFrame()

    for mf in sorted(my_filenames):

        my_fn = os.path.join(myp, mf)
        my_es = results.load_es(my_fn)
        res = my_es.results["param"]
        wind = int(
            round(
                [
                    res[w]["scalars"]["nominal_value"]
                    for w in res
                    if w[0].label.subtag == "Wind" and w[1] is not None
                ][0]
            )
        )
        solar = int(
            round(
                [
                    res[w]["scalars"]["nominal_value"]
                    for w in res
                    if w[0].label.subtag == "Solar" and w[1] is not None
                ][0]
            )
        )
        key = "w {0:02}, pv {1:02}".format(wind, solar)
        print(key)
        my_df = results.get_multiregion_bus_balance(my_es)
        imp = my_df["FHG", "in", "source", "import", "electricity"]
        exp = my_df["FHG", "out", "sink", "export", "electricity"]
        up.set_index(imp.index, inplace=True)
        for t2 in ["mcpe", "emission_avg", "emission_max"]:
            for f in ["f1", "f15", "f2"]:
                prc = up[f, t2]
                up_plot[t2][f].loc[key, "import"] = (
                    (imp * prc).sum() / imp.sum() / prc.mean()
                )
                up_plot[t2][f].loc[key, "export"] = (
                    (exp * prc).sum() / exp.sum() / prc.mean()
                )
    n2 = 0
    for k1, v1 in up_plot.items():
        n1 = 0
        for k2, v2 in v1.items():
            print(k1, k2, n1, n2)
            v2.sort_index().plot(kind="bar", ax=ax_ar[n2, n1], legend=False)
            ax_ar[n2, n1].set_title("{0}, {1}".format(k1, k2))
            n1 += 1
        n2 += 1
    plt.legend()
    return "import_export_emission_100PRZ_region", None


def fig_import_export_costs_100prz_region():
    plt.rcParams.update({"font.size": 14})
    f, ax = plt.subplots(1, 1, figsize=(15, 6))
    my_filename = "market_clearing_price_phd_c1.xls"
    my_path = cfg.get("paths", "phd")
    up_fn = os.path.join(my_path, my_filename)
    up_df = pd.read_excel(up_fn, index_col=[0], header=[0])
    res_path = os.path.join(my_path, "region", "results_cbc")

    result = [x for x in os.listdir(res_path) if ".esys" in x and "_pv" in x]

    my_list = [
        x for x in up_df.columns if "f10" in x or "f15" in x or "f20" in x
    ]
    my_list = [x for x in my_list if "de21" in x]
    new_list = [x for x in up_df.columns if "no" not in x]
    my_list += new_list

    groups = (
        "deflex_XX_Nc00_Li05_HP02_GT_{0}_de21",
        "deflex_XX_Nc00_Li05_HP00_GT_{0}_de21",
        "deflex_XX_Nc00_HP02_{0}_de21",
        "deflex_XX_Nc00_HP00_{0}_de21",
        "deflex_2014_de02",
        "deflex_2014_de17",
        "deflex_2014_de21",
        "deflex_2014_de22",
    )
    print(result)
    mf = [x for x in result if "wind27" in x][0]
    # for mf in sorted(result):
    my_fn = os.path.join(res_path, mf)
    my_es = results.load_es(my_fn)
    res = my_es.results["param"]
    wind = int(
        round(
            [
                res[w]["scalars"]["nominal_value"]
                for w in res
                if w[0].label.subtag == "Wind" and w[1] is not None
            ][0]
        )
    )

    my_df = results.get_multiregion_bus_balance(my_es)
    imp = my_df["FHG", "in", "source", "import", "electricity"]
    exp = my_df["FHG", "out", "sink", "export", "electricity"]

    pr = pd.DataFrame()
    up_df.set_index(imp.index, inplace=True)
    my_import = pd.DataFrame()
    my_export = pd.DataFrame()
    my_list = {
        x: x.replace("f15", "g15").replace("f1", "g10").replace("f2", "g20")
        for x in my_list
        if "without" not in x
    }
    # print(my_list)
    for g in groups:
        for f in ["f1", "f15", "f2"]:
            up = g.format(f)
            print(up)
            name = up
            prc = up_df[up]
            pr.loc[name, "import"] = (imp * prc).sum() / imp.sum() / prc.mean()
            pr.loc[name, "export"] = (exp * prc).sum() / exp.sum() / prc.mean()

            mean = (exp * prc).sum() / exp.sum() / prc.mean() - (
                imp * prc
            ).sum() / imp.sum() / prc.mean()
            pr.loc[name, "diff"] = mean * -1
            pr.loc[name, "mean"] = prc.mean()

            my_import[up] = imp.multiply(prc).sum()
            my_export[up] = exp.multiply(prc).sum()
        # if wind == 0:
        #     ax_ar = pr.plot(kind="bar", secondary_y=["mean"], ax=ax_ar)
        #     ax_ar.right_ax.set_ylim(0, 1120)
        # if wind == 27:
    print(pr)
    ax = pr.plot(kind="bar", secondary_y=["mean"], ax=ax)
    # plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    secondary_max = 75
    ax.right_ax.set_ylim(0, secondary_max)
    # Shrink current axis by 20%
    # box = ax_ar.get_position()
    # ax_ar.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    # ax_ar.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 2.0)
    plt.xticks(
        list(range(16)),
        ["1.0", "1.5", "2.0"] * 4 + ["de02", "de17", "de21", "de22"],
        rotation="horizontal",
        horizontalalignment="center",
    )
    for n in range(1, 5):
        k = 3 * n - 0.5
        plt.plot((k, k), (0, secondary_max), "k-.")
    plt.text(
        8.5,
        66,
        "Atomausstieg",
        ha="center",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 1, "pad": 5, "linewidth": 0},
    )
    plt.text(
        2.5,
        62,
        "Atomausstieg\nBraunkohle: -50%\nGasturbine: +30 GW",
        ha="center",
        fontsize=15,
        bbox={"facecolor": "white", "alpha": 1, "pad": 5, "linewidth": 0},
    )
    for p, t in [
        (1, "WP +20%"),
        (7, "WP +20%"),
        (4, "Bestand"),
        (10, "Bestand"),
        (13, "Basismodelle"),
    ]:
        plt.text(
            p, 77, t, ha="center", fontsize=15,
        )
    plt.subplots_adjust(right=0.9, left=0.08, bottom=0.12, top=0.9)

    return "import_export_costs_100PRZ_region", None
