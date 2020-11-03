import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from reegis import geometries

from my_reegis import config as cfg
from my_reegis import reegis_plot as plot
from my_reegis.figures import figures_base


def fig_model_regions():
    """Plot one or more model regions in one plot."""
    maps = ["de02", "de17", "de21", "de22"]
    # maps = ["de22"]
    add_title = False

    top = 1
    ax = None
    ax_ar = []

    width = len(maps * 3)

    if len(maps) > 1:
        f, ax_ar = plt.subplots(1, len(maps), figsize=(width, 2.5))
    else:
        ax = plt.figure(figsize=(width, 2.5)).add_subplot(1, 1, 1)
    i = 0
    for rmap in maps:
        if len(maps) > 1:
            ax = ax_ar[i]

        de_map = geometries.load(
            cfg.get("paths", "geo_deflex"),
            cfg.get("geometry", "deflex_polygon").format(
                suffix=".geojson", map=rmap, type="polygons"
            ),
        )

        plot.plot_regions(
            map=de_map, ax=ax, legend=False, simple=0.005, offshore="auto"
        )
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        ax.axis("off")
        if add_title is True:
            ax.set_title(rmap)
            top = 0.88
        i += 1

    plt.subplots_adjust(right=1, left=0, wspace=0, bottom=0, top=top)
    return "model_regions", None


def fig_compare_de21_region_borders():
    ffe = geometries.load_shp(
        cfg.get("paths", "geo_plot"), "de21_nach_ffe_grafik.geojson"
    )
    tso = geometries.load_shp(
        cfg.get("paths", "geo_plot"), "de21_nach_tso_bild.geojson"
    ).set_index("region")
    renpass = geometries.load_shp(
        cfg.get("paths", "geo_plot"), "de21_renpass_frauke_wiese.geojson"
    )

    cmap = LinearSegmentedColormap.from_list(
        "mycmap", [(0.0, "#000000"), (0.99, "#ffffff"), (1, "#a5bfdd")]
    )
    data = pd.Series(
        {
            "DE01": 0.9,
            "DE02": 0.95,
            "DE03": 0.5,
            "DE04": 0.7,
            "DE05": 0.5,
            "DE06": 0.83,
            "DE07": 0.65,
            "DE08": 0.37,
            "DE09": 0.45,
            "DE10": 0.8,
            "DE11": 0.75,
            "DE12": 0.60,
            "DE13": 0.4,
            "DE14": 0.75,
            "DE15": 0.2,
            "DE16": 0.85,
            "DE17": 0.5,
            "DE18": 0.2,
            "DE19": 1,
            "DE20": 1,
            "DE21": 1,
        }
    )
    data = pd.DataFrame(data, columns=["value"])
    ax = plot.plot_regions(
        edgecolor="None",
        data=data,
        legend=False,
        label_col="sp_id_1",
        map=tso,
        data_col="value",
        cmap=cmap,
    )

    ax = ffe.boundary.plot(
        ax=ax, color="#720000", aspect="equal", linewidth=0.8
    )
    renpass.boundary.plot(
        ax=ax, color="#064304", aspect="equal", linewidth=0.8
    )

    plt.legend(["nach FfE", "renpass", "nach ÜNB"])
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return "netzregionen_vergleich", None


def fig_show_download_deutschland_modell():
    return figures_base.show_download_image(
        "deutschlandmodell_vereinfacht", ["graphml", "svg"]
    )


def fig_show_download_berlin_modell():
    return figures_base.show_download_image(
        "berlinmodell_vereinfacht", ["graphml", "svg"]
    )


def fig_district_heating_areas(**kwargs):
    from berlin_hp import heat

    ax = figures_base.create_subplot((7.8, 4), **kwargs)

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
    ax = berlin.plot(color="#ffffff", edgecolor="black", ax=ax, aspect="equal")

    # Plot areas of district heating system groups
    ax = distr_heat_areas.loc[
        distr_heat_areas["name"] != "decentralised_dh"
    ].plot(column="name", ax=ax, cmap="tab10", aspect="equal")

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
        "Köpenick": (13.58, 52.425),
        "Friedrichshagen": (13.653, 52.47),
    }

    for t, c in text.items():
        plt.text(
            c[0],
            c[1],
            t,
            size=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", alpha=0.5, ec=(1, 1, 1), fc=(1, 1, 1)),
        )
    plt.draw()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return "distric_heating_areas", None


def fig_deflex_de22_polygons(**kwargs):
    ax = figures_base.create_subplot((9, 7), **kwargs)

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
