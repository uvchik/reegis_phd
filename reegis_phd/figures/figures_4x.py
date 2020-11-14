import datetime
import logging
import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
from berlin_hp import electricity
from demandlib import bdew as bdew
from demandlib import particular_profiles as profiles
from matplotlib import cm
from matplotlib import dates as mdates
from matplotlib import image as mpimg
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from oemof.tools import logger
from reegis import bmwi
from reegis import coastdat
from reegis import config as cfg
from reegis import demand_elec
from reegis import energy_balance
from reegis import entsoe
from reegis import geometries
from reegis import inhabitants
from reegis import powerplants
from reegis import storages
from scenario_builder import feedin

from reegis_phd import data_analysis
from reegis_phd.figures.figures_base import create_subplot
from reegis_phd.figures.figures_base import show_download_image


def fig_patch_offshore(**kwargs):
    ax = create_subplot((12, 4), **kwargs)
    federal_states = geometries.load(
        cfg.get("paths", "geometry"),
        cfg.get("geometry", "federalstates_polygon"),
    )
    # federal_states.drop(['P0'], inplace=True)
    mydf = powerplants.patch_offshore_wind(pd.DataFrame(), [])
    mygdf = gpd.GeoDataFrame(mydf)
    fs = federal_states.set_index("iso").loc[
        ["NI", "SH", "HH", "MV", "BB", "BE", "HB", "ST", "NW"]
    ]
    offshore = federal_states.set_index("iso").loc[["N0", "N1", "O0"]]
    fs["geometry"] = fs["geometry"].simplify(0.01)
    offshore["geometry"] = offshore["geometry"].simplify(0.01)

    ax = fs.plot(
        ax=ax, facecolor="#badd69", edgecolor="#777777", aspect="equal"
    )
    ax = offshore.plot(
        ax=ax, facecolor="#ffffff", edgecolor="#777777", aspect="equal"
    )
    mygdf.plot(
        markersize=mydf.capacity, alpha=0.5, ax=ax, legend=True, aspect="equal"
    )

    plt.ylim(bottom=52.5)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1)
    ax.legend()
    return "patch_offshore", None


def fig_powerplants(**kwargs):
    plt.rcParams.update({"font.size": 14})
    geo = geometries.load(
        cfg.get("paths", "geometry"),
        cfg.get("geometry", "federalstates_polygon"),
    )

    my_name = "my_federal_states"  # doctest: +SKIP
    my_year = 2015  # doctest: +SKIP
    pp_reegis = powerplants.get_powerplants_by_region(geo, my_year, my_name)

    data_path = os.path.join(os.path.dirname(__file__), "../data", "static")
    fn_bnetza = os.path.join(data_path, cfg.get("plot_data", "bnetza"))
    pp_bnetza = pd.read_csv(fn_bnetza, index_col=[0], skiprows=2, header=[0])

    ax = create_subplot((10, 5), **kwargs)

    see = "sonst. erneuerb."

    my_dict = {
        "Bioenergy": see,
        "Geothermal": see,
        "Hard coal": "Kohle",
        "Hydro": see,
        "Lignite": "Kohle",
        "Natural gas": "Erdgas",
        "Nuclear": "Nuklear",
        "Oil": "sonstige fossil",
        "Other fossil fuels": "sonstige fossil",
        "Other fuels": "sonstige fossil",
        "Solar": "Solar",
        "Waste": "sonstige fossil",
        "Wind": "Wind",
        "unknown from conventional": "sonstige fossil",
    }

    my_dict2 = {
        "Biomasse": see,
        "Braunkohle": "Kohle",
        "Erdgas": "Erdgas",
        "Kernenergie": "Nuklear",
        "Laufwasser": see,
        "Solar": "Solar",
        "Sonstige (ne)": "sonstige fossil",
        "Steinkohle": "Kohle",
        "Wind": "Wind",
        "Sonstige (ee)": see,
        "Öl": "sonstige fossil",
    }

    my_colors = [
        "#555555",
        "#6c3012",
        "#db0b0b",
        "#ffde32",
        "#335a8a",
        "#163e16",
        "#501209",
    ]

    # pp_reegis.capacity_2015.unstack().to_excel('/home/uwe/shp/wasser.xls')

    pp_reegis = (
        pp_reegis.capacity_2015.unstack().groupby(my_dict, axis=1).sum()
    )

    pp_reegis = pp_reegis.merge(
        geo["iso"], left_index=True, right_index=True
    ).set_index("iso")

    pp_reegis.loc["AWZ"] = (
        pp_reegis.loc["N0"] + pp_reegis.loc["N1"] + pp_reegis.loc["O0"]
    )

    pp_reegis.drop(["N0", "N1", "O0", "P0"], inplace=True)

    pp_bnetza = pp_bnetza.groupby(my_dict2, axis=1).sum()

    ax = (
        pp_reegis.sort_index()
        .sort_index(1)
        .div(1000)
        .plot(
            kind="bar",
            stacked=True,
            position=1.1,
            width=0.3,
            legend=False,
            color=my_colors,
            ax=ax,
        )
    )
    pp_bnetza.sort_index().sort_index(1).div(1000).plot(
        kind="bar",
        stacked=True,
        position=-0.1,
        width=0.3,
        ax=ax,
        color=my_colors,
        alpha=0.9,
    )
    plt.xlabel("Bundesländer / AWZ")
    plt.ylabel("Installierte Leistung [GW]")
    plt.xlim(left=-0.5)
    plt.subplots_adjust(bottom=0.17, top=0.98, left=0.08, right=0.96)

    b_sum = pp_bnetza.sum() / 1000
    b_total = int(round(b_sum.sum()))
    b_ee_sum = int(round(b_sum.loc[["Wind", "Solar", see]].sum()))
    b_fs_sum = int(
        round(
            b_sum.loc[["Erdgas", "Kohle", "Nuklear", "sonstige fossil"]].sum()
        )
    )
    r_sum = pp_reegis.sum() / 1000
    r_total = int(round(r_sum.sum()))
    r_ee_sum = int(round(r_sum.loc[["Wind", "Solar", see]].sum()))
    r_fs_sum = int(
        round(
            r_sum.loc[["Erdgas", "Kohle", "Nuklear", "sonstige fossil"]].sum()
        )
    )

    text = {
        "reegis": (2.3, 42, "reegis"),
        "BNetzA": (3.9, 42, "BNetzA"),
        "b_sum1": (0, 39, "gesamt"),
        "b_sum2": (2.5, 39, "{0}       {1}".format(r_total, b_total)),
        "b_fs": (0, 36, "fossil"),
        "b_fs2": (2.5, 36, " {0}         {1}".format(r_fs_sum, b_fs_sum)),
        "b_ee": (0, 33, "erneuerbar"),
        "b_ee2": (2.5, 33, " {0}         {1}".format(r_ee_sum, b_ee_sum)),
    }

    for t, c in text.items():
        plt.text(c[0], c[1], c[2], size=14, ha="left", va="center")

    b = patches.Rectangle((-0.2, 31.8), 5.7, 12, color="#cccccc")
    ax.add_patch(b)
    ax.add_patch(patches.Shadow(b, -0.05, -0.2))
    return "vergleich_kraftwerke_reegis_bnetza", None


def fig_storage_capacity(**kwargs):
    plt.rcParams.update({"font.size": 12})
    ax = create_subplot((6, 4), **kwargs)

    federal_states = geometries.load(
        cfg.get("paths", "geometry"),
        cfg.get("geometry", "federalstates_polygon"),
    )

    federal_states.set_index("iso", drop=True, inplace=True)
    federal_states["geometry"] = federal_states["geometry"].simplify(0.02)
    phes = storages.pumped_hydroelectric_storage_by_region(
        federal_states, 2014, "federal_states"
    )

    fs = federal_states.merge(
        phes, left_index=True, right_index=True, how="left"
    ).fillna(0)

    fs.drop(["N0", "N1", "O0", "P0"], inplace=True)
    fs["energy"] = fs["energy"].div(1000)
    # colormap = "YlGn"
    colormap = "Greys"
    ax = fs.plot(column="energy", cmap=colormap, ax=ax, aspect="equal")
    ax = fs.boundary.plot(ax=ax, color="#777777", aspect="equal")
    coords = {
        "NI": (9.7, 52.59423440995961),
        "SH": (9.8, 53.9),
        "ST": (11.559203329244966, 51.99003282648907),
        "NW": (7.580292138948966, 51.4262307721131),
        "BW": (9.073099768325736, 48.5),
        "BY": (11.5, 48.91810114600406),
        "TH": (10.9, 50.8),
        "HE": (9.018890328297207, 50.52634809768823),
        "SN": (13.3, 50.928277090542124),
    }

    for idx, row in fs.iterrows():
        if row["energy"] > 0:
            if row["energy"] > 10:
                color = "#dddddd"
            else:
                color = "#000000"
            plt.annotate(
                s=round(row["energy"], 1),
                xy=coords[idx],
                horizontalalignment="center",
                color=color,
            )
    ax.set_axis_off()
    scatter = ax.collections[0]
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Speicherkapazität [GWh]", rotation=270, labelpad=15)
    plt.subplots_adjust(left=0, bottom=0.05, top=0.95)
    return "storage_capacity_by_federal_states", None


def fig_inhabitants():
    plt.rcParams.update({"font.size": 18})
    f, ax_ar = plt.subplots(1, 2, figsize=(16, 5.6))
    df = pd.DataFrame()
    for year in range(2011, 2018):
        df[year] = inhabitants.get_ew_by_federal_states(year)
    df.sort_values(2017, inplace=True)
    df.transpose().div(1000).plot(
        kind="bar", stacked=True, cmap="tab20b_r", ax=ax_ar[0]
    )

    print(df)
    handles, labels = ax_ar[0].get_legend_handles_labels()
    ax_ar[0].legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        bbox_to_anchor=(1, 1.025),
    )
    # plt.subplots_adjust(left=0.14, bottom=0.15, top=0.9, right=0.8)
    ax_ar[0].set_ylabel("Tsd. Einwohner")
    ax_ar[0].set_xticklabels(ax_ar[0].get_xticklabels(), rotation=0)
    plt.xticks(rotation=0)
    ew = inhabitants.get_ew_geometry(2017, polygon=True)
    ew["ew_area"] = ew["EWZ"].div(ew["KFL"]).fillna(0)
    ew["geometry"] = ew["geometry"].simplify(0.01)
    ew.plot(
        column="ew_area", vmax=800, cmap="cividis", ax=ax_ar[1], aspect="equal"
    )
    ax_ar[1].set_axis_off()
    divider = make_axes_locatable(ax_ar[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)

    norm = Normalize(vmin=0, vmax=800)
    n_cmap = cm.ScalarMappable(norm=norm, cmap="cividis")
    n_cmap.set_array(np.array([]))
    cbar = plt.colorbar(n_cmap, ax=ax_ar[1], extend="max", cax=cax)
    cbar.set_label("Einwohner pro km²", rotation=270, labelpad=30)
    plt.subplots_adjust(left=0.09, top=0.98, bottom=0.06, right=0.93)
    # plt.xticks(rotation=0)
    return "inhabitants_by_ferderal_states", None


def fig_average_weather():
    plt.rcParams.update({"font.size": 20})
    f, ax_ar = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    my_cmap = LinearSegmentedColormap.from_list(
        "mycmap",
        [
            (0, "#dddddd"),
            (1 / 7, "#c946e5"),
            (2 / 7, "#ffeb00"),
            (3 / 7, "#26a926"),
            (4 / 7, "#c15c00"),
            (5 / 7, "#06ffff"),
            (6 / 7, "#f24141"),
            (7 / 7, "#1a2663"),
        ],
    )

    weather_path = cfg.get("paths", "coastdat")

    # Download missing weather files
    pattern = "coastDat2_de_{0}.h5"
    for year in range(1998, 2015):
        fn = os.path.join(weather_path, pattern.format(year))
        if not os.path.isfile(fn):
            coastdat.download_coastdat_data(filename=fn, year=year)

    pattern = "average_data_{data_type}.csv"
    dtype = "v_wind"
    fn = os.path.join(weather_path, pattern.format(data_type=dtype))
    if not os.path.isfile(fn):
        coastdat.store_average_weather(dtype, out_file_pattern=pattern)
    df = pd.read_csv(fn, index_col=[0])
    coastdat_poly = geometries.load(
        cfg.get("paths", "geometry"),
        cfg.get("coastdat", "coastdatgrid_polygon"),
    )
    coastdat_poly = coastdat_poly.merge(df, left_index=True, right_index=True)
    ax = coastdat_poly.plot(
        column="v_wind_avg",
        cmap=my_cmap,
        vmin=1,
        vmax=8,
        ax=ax_ar[0],
        aspect="equal",
    )
    ax = (
        geometries.get_germany_with_awz_polygon()
        .simplify(0.05)
        .boundary.plot(ax=ax, color="#555555", aspect="equal")
    )
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    norm = Normalize(vmin=1, vmax=8)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=my_cmap)
    n_cmap.set_array(np.array([]))
    cbar = plt.colorbar(n_cmap, ax=ax, extend="both", cax=cax)
    cbar.set_label("Windgeschwindigkeit [m/s]", rotation=270, labelpad=30)

    weather_path = cfg.get("paths", "coastdat")
    dtype = "temp_air"
    fn = os.path.join(weather_path, pattern.format(data_type=dtype))
    if not os.path.isfile(fn):
        coastdat.store_average_weather(
            dtype, out_file_pattern=pattern, years=[2014, 2013, 2012]
        )
    df = pd.read_csv(fn, index_col=[0]) - 273.15
    print(df.mean())
    coastdat_poly = geometries.load(
        cfg.get("paths", "geometry"),
        cfg.get("coastdat", "coastdatgrid_polygon"),
    )
    coastdat_poly = coastdat_poly.merge(df, left_index=True, right_index=True)
    ax = coastdat_poly.plot(
        column="temp_air_avg",
        cmap="rainbow",
        vmin=7,
        vmax=11,
        ax=ax_ar[1],
        aspect="equal",
    )
    ax = (
        geometries.get_germany_with_awz_polygon()
        .simplify(0.05)
        .boundary.plot(ax=ax, color="#555555", aspect="equal")
    )
    ax.set_axis_off()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    norm = Normalize(vmin=5, vmax=11)
    n_cmap = cm.ScalarMappable(norm=norm, cmap="rainbow")
    n_cmap.set_array(np.array([]))
    cbar = plt.colorbar(n_cmap, ax=ax, extend="both", cax=cax)
    cbar.set_label("Temperatur [°C]", rotation=270, labelpad=30)
    plt.subplots_adjust(left=0, top=0.97, bottom=0.03, right=0.93, wspace=0.1)
    return "average_weather", None


def fig_strahlungsmittel():
    return show_download_image("strahlungsmittel_dwd_coastdat", ["svg"])


def fig_module_comparison():
    plt.rcParams.update({"font.size": 15})
    plt.sca(create_subplot((10.7, 5)))
    df = pd.read_csv(
        os.path.join(cfg.get("paths", "data_my_reegis"), "module_feedin.csv"),
        index_col=0,
    )["dc_norm"]
    print(df)
    print(df.sort_values())
    # df = df[df > 943]
    df.sort_values().plot(linewidth=5, ylim=(0, df.max() + 20))
    print("avg:", df.mean())
    print("std div:", df.std())
    plt.plot((0, len(df)), (df.mean(), df.mean()), "k-")
    plt.plot((0, len(df)), (df.mean() - df.std(), df.mean() - df.std()), "k-.")
    plt.plot((0, len(df)), (df.mean() + df.std(), df.mean() + df.std()), "k-.")
    plt.plot((253, 253), (0, df.max() + 20), "k-")
    plt.plot((479, 479), (0, df.max() + 20), "r-")
    plt.plot((394, 394), (0, df.max() + 20), "r-")
    plt.plot((253, 253), (0, df.max() + 20), "r-")
    plt.plot((62, 62), (0, df.max() + 20), "r-")
    plt.text(
        479,
        800,
        "SF 160S",
        ha="center",
        bbox={"facecolor": "white", "alpha": 1, "pad": 5, "linewidth": 0},
    )
    plt.text(
        394,
        800,
        "LG290N1C",
        ha="center",
        bbox={"facecolor": "white", "alpha": 1, "pad": 5, "linewidth": 0},
    )
    plt.text(
        253,
        800,
        "STP280S",
        ha="center",
        bbox={"facecolor": "white", "alpha": 1, "pad": 5, "linewidth": 0},
    )
    plt.text(
        62,
        800,
        "BP2150S",
        ha="center",
        bbox={"facecolor": "white", "alpha": 1, "pad": 5, "linewidth": 0},
    )
    plt.xticks(np.arange(0, len(df), 40), range(0, len(df), 40))
    plt.ylim(500, 1400)

    plt.xlim(0, len(df))
    plt.ylabel("Volllaststunden")
    plt.xlabel("ID des Moduls")
    plt.subplots_adjust(right=0.98, left=0.09, bottom=0.12, top=0.95)
    return "module_comparison", None


def fig_analyse_multi_files():
    plt.rcParams.update({"font.size": 10})
    path = os.path.join(cfg.get("paths", "data_my_reegis"))
    fn = os.path.join(path, "multiyear_yield_sum.csv")
    df = pd.read_csv(fn, index_col=[0, 1])
    gdf = data_analysis.get_coastdat_onshore_polygons()
    gdf.geometry = gdf.buffer(0.005)
    for key in gdf.index:
        s = df[str(key)]
        pt = gdf.loc[key]
        gdf.loc[key, "tilt"] = s[s == s.max()].index.get_level_values("tilt")[
            0
        ]
        gdf.loc[key, "azimuth"] = s[s == s.max()].index.get_level_values(
            "azimuth"
        )[0]
        gdf.loc[key, "longitude"] = pt.geometry.centroid.x
        gdf.loc[key, "latitude"] = pt.geometry.centroid.y
        gdf.loc[key, "tilt_calc"] = round(pt.geometry.centroid.y - 15)
        gdf.loc[key, "tilt_diff"] = abs(
            gdf.loc[key, "tilt_calc"] - gdf.loc[key, "tilt"]
        )
        gdf.loc[key, "tilt_diff_c"] = abs(gdf.loc[key, "tilt"] - 36.5)
        gdf.loc[key, "azimuth_diff"] = abs(gdf.loc[key, "azimuth"] - 178.5)

    cmap_t = plt.get_cmap("viridis", 8)
    cmap_az = plt.get_cmap("viridis", 7)
    cm_gyr = LinearSegmentedColormap.from_list(
        "mycmap", [(0, "green"), (0.5, "yellow"), (1, "red")], 6
    )

    f, ax_ar = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(7, 8))

    ax_ar[0][0].set_title("Azimuth (optimal)", loc="center", y=1)
    gdf.plot(
        "azimuth",
        legend=True,
        cmap=cmap_az,
        vmin=173,
        vmax=187,
        ax=ax_ar[0][0],
        aspect="equal",
    )

    ax_ar[1][0].set_title("Neigung (optimal)", loc="center", y=1)
    gdf.plot(
        "tilt",
        legend=True,
        vmin=32.5,
        vmax=40.5,
        cmap=cmap_t,
        ax=ax_ar[1][0],
        aspect="equal",
    )

    ax_ar[2][0].set_title("Neigung (nach Breitengrad)", loc="center", y=1)
    gdf.plot(
        "tilt_calc",
        legend=True,
        vmin=32.5,
        vmax=40.5,
        cmap=cmap_t,
        ax=ax_ar[2][0],
        aspect="equal",
    )

    ax_ar[0][1].set_title(
        "Azimuth (Differenz - optimal zu 180°)", loc="center", y=1,
    )
    gdf.plot(
        "azimuth_diff",
        legend=True,
        vmin=-0.5,
        vmax=5.5,
        cmap=cm_gyr,
        ax=ax_ar[0][1],
        aspect="equal",
    )

    ax_ar[1][1].set_title(
        "Neigung (Differenz - optimal zu Breitengrad)", loc="center", y=1
    )
    gdf.plot(
        "tilt_diff",
        legend=True,
        vmin=-0.5,
        vmax=5.5,
        cmap=cm_gyr,
        ax=ax_ar[1][1],
        aspect="equal",
    )

    ax_ar[2][1].set_title(
        "Neigung (Differenz - optimal zu 36,5°)", loc="center", y=1
    )
    gdf.plot(
        "tilt_diff_c",
        legend=True,
        vmin=-0.5,
        vmax=5.5,
        cmap=cm_gyr,
        ax=ax_ar[2][1],
        aspect="equal",
    )

    plt.subplots_adjust(right=1, left=0.05, bottom=0.05, top=0.95, wspace=0.11)
    return "analyse_optimal_orientation", None


def fig_polar_plot_pv_orientation():
    plt.rcParams.update({"font.size": 14})
    key = 1129089
    path = os.path.join(cfg.get("paths", "data_my_reegis"))
    fn = os.path.join(path, "{0}_combined_c.csv".format(key))

    df = pd.read_csv(fn, index_col=[0, 1])
    df.reset_index(inplace=True)
    df["rel"] = df["2"] / df["2"].max()

    azimuth_opt = float(df[df["2"] == df["2"].max()]["1"])
    tilt_opt = float(df[df["2"] == df["2"].max()]["0"])
    print(azimuth_opt, tilt_opt)
    print(tilt_opt - 5)
    print(df[(df["1"] == azimuth_opt + 5) & (df["0"] == tilt_opt + 5)])
    print(df[(df["1"] == azimuth_opt - 5) & (df["0"] == tilt_opt + 5)])
    print(
        df[(df["1"] == azimuth_opt + 5) & (df["0"] == round(tilt_opt - 5, 1))]
    )
    print(
        df[(df["1"] == azimuth_opt - 5) & (df["0"] == round(tilt_opt - 5, 1))]
    )

    # Data
    tilt = df["0"]
    azimuth = df["1"] / 180 * np.pi
    colors = df["2"] / df["2"].max()

    # Colormap
    cmap = plt.get_cmap("viridis", 20)

    # Plot
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111, projection="polar")
    sc = ax.scatter(azimuth, tilt, c=colors, cmap=cmap, alpha=1, vmin=0.8)
    ax.tick_params(pad=10)

    # Colorbar
    label = "Anteil vom maximalen Ertrag"
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.75])
    fig.colorbar(sc, cax=cax, label=label, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_theta_zero_location("S", offset=0)

    # Adjust radius
    # ax.set_rmax(90)
    ax.set_rlabel_position(110)
    t_upper = tilt_opt + 5
    t_lower = tilt_opt - 5
    az_upper = azimuth_opt + 5
    az_lower = azimuth_opt - 5
    bbox_props = dict(boxstyle="round", fc="white", alpha=0.5, lw=0)
    ax.annotate(
        ">0.996",
        xy=((az_upper - 5) / 180 * np.pi, t_upper),
        xytext=((az_upper + 3) / 180 * np.pi, t_upper + 3),
        # textcoords='figure fraction',
        arrowprops=dict(facecolor="black", arrowstyle="-"),
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox=bbox_props,
    )
    print(az_upper)
    print(t_upper)
    ax.text(
        238 / 180 * np.pi,
        60,
        "Ausrichtung (Süd=180°)",
        rotation=50,
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.text(
        65 / 180 * np.pi,
        35,
        "Neigungswinkel (horizontal=0°)",
        rotation=0,
        horizontalalignment="center",
        verticalalignment="center",
    )

    az = (
        np.array([az_lower, az_lower, az_upper, az_upper, az_lower])
        / 180
        * np.pi
    )
    t = np.array([t_lower, t_upper, t_upper, t_lower, t_lower])
    ax.plot(az, t)

    ax.set_rmax(50)
    ax.set_rmin(20)
    ax.set_thetamin(90)
    ax.set_thetamax(270)
    # Adjust margins
    plt.subplots_adjust(right=0.94, left=0, bottom=-0.15, top=1.2)
    return "polar_plot_pv_orientation.png", None


def fig_windzones():

    # ax.set_axis_off()
    plt.show()
    path = cfg.get("paths", "geometry")
    filename = "windzones_germany.geojson"
    df = geometries.load(path=path, filename=filename)
    df.set_index("zone", inplace=True)
    geo_path = cfg.get("paths", "geometry")
    geo_file = cfg.get("coastdat", "coastdatgrid_polygon")
    coastdat_geo = geometries.load(path=geo_path, filename=geo_file)
    coastdat_geo["poly"] = coastdat_geo.geometry
    coastdat_geo["geometry"] = coastdat_geo.centroid

    points = geometries.spatial_join_with_buffer(coastdat_geo, df, "windzone")
    polygons = points.set_geometry("poly")

    cmap_bluish = LinearSegmentedColormap.from_list(
        "bluish", [(0, "#8fbbd2"), (1, "#00317a")], 4
    )

    ax = polygons.plot(
        column="windzone",
        edgecolor="#666666",
        linewidth=0.5,
        cmap=cmap_bluish,
        vmin=0.5,
        vmax=4.5,
        aspect="equal",
    )
    ax.set_axis_off()
    df.boundary.simplify(0.01).plot(
        edgecolor="black", alpha=1, ax=ax, linewidth=1.5, aspect="equal",
    )
    text = {"1": (9, 50), "2": (12, 52), "3": (9.8, 54), "4": (6.5, 54.6)}

    for t, c in text.items():
        plt.text(
            c[0],
            c[1],
            t,
            size=15,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", alpha=0.5, ec=(1, 1, 1), fc=(1, 1, 1)),
        )
    plt.subplots_adjust(left=0, top=1, bottom=0, right=1)
    return "windzones", None


def fig_show_hydro_image():
    create_subplot((12, 4.4))
    file = "abflussregime.png"
    fn = os.path.join(cfg.get("paths", "figure_source"), file)
    fn_target = os.path.join(cfg.get("paths", "figures"), file)
    shutil.copy(fn, fn_target)
    img = mpimg.imread(fn)
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        "Image source: https://www.iksr.org/fileadmin/user_upload/DKDM/"
        "Dokumente/Fachberichte/DE/rp_De_0248.pdf; S.16"
    )
    plt.subplots_adjust(left=0, top=0.93, bottom=0, right=1)
    return "abflussregime", None


def fig_compare_re_capacity_years():
    # from reegis import bmwi
    plt.rcParams.update({"font.size": 18})
    f, ax_ar = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 5))

    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    my_bmwi = bmwi.bmwi_re_energy_capacity().loc[years].div(1000)
    my_bmwi.set_index(
        pd.to_datetime(my_bmwi.index.astype(str) + "-12-31")
        + pd.DateOffset(1),
        inplace=True,
    )
    bmwi_solar = my_bmwi["solar", "capacity"]
    bmwi_solar.name = "Solar (BMWI)"
    bmwi_wind = my_bmwi["wind", "capacity"]
    bmwi_wind.name = "Wind (BMWI)"

    ax_ar[0] = bmwi_solar.plot(
        marker="D",
        ax=ax_ar[0],
        linestyle="None",
        markersize=10,
        color="#ff5500",
        alpha=0.7,
        legend=True,
    )
    ax_ar[1] = bmwi_wind.plot(
        marker="D",
        ax=ax_ar[1],
        linestyle="None",
        markersize=10,
        color="#111539",
        alpha=0.7,
        legend=True,
    )

    my_re = entsoe.get_entsoe_renewable_data(version="2019-06-05").div(1000)
    my_re = my_re.resample("D").mean()
    print(my_re.index)
    rn = {
        "DE_solar_capacity": "Solar (OPSD)",
        "DE_wind_capacity": "Wind (OPSD)",
    }
    my_re.rename(columns=rn, inplace=True)

    ax_ar[0] = my_re["Solar (OPSD)"].plot(
        ax=ax_ar[0], color="#ffba00", legend=True
    )
    ax_ar[1] = my_re["Wind (OPSD)"].plot(
        ax=ax_ar[1], color="#4254ff", legend=True
    )

    fs = geometries.get_federal_states_polygon()
    df = pd.DataFrame()
    for y in years:
        my_pp = powerplants.get_powerplants_by_region(fs, y, "federal_states")
        for cat in ["Solar", "Wind"]:
            dt = datetime.datetime(y, 1, 1)
            cat_name = "{0} (reegis)".format(cat)
            col = "capacity_{0}".format(y)
            df.loc[dt, cat_name] = my_pp.groupby(level=1).sum().loc[cat, col]
    df = df.div(1000)
    ax_ar[0] = df["Solar (reegis)"].plot(
        drawstyle="steps-post", ax=ax_ar[0], color="#ff7000", legend=True
    )
    ax_ar[1] = df["Wind (reegis)"].plot(
        drawstyle="steps-post", ax=ax_ar[1], color=["#1b2053"], legend=True
    )

    ax_ar[0].set_xlim(
        left=datetime.datetime(2012, 1, 1), right=datetime.datetime(2018, 1, 1)
    )
    plt.ylim((25, 60))
    ax_ar[0].set_ylabel("Installierte Leistung [GW]")
    ax_ar[0].set_xlabel(" ")
    ax_ar[1].set_xlabel(" ")
    ax_ar[0].legend(loc="upper left")
    ax_ar[1].legend(loc="upper left")
    plt.subplots_adjust(
        right=0.98, left=0.06, bottom=0.11, top=0.94, wspace=0.16
    )

    return "compare_re_capacity_years", None


def fig_compare_full_load_hours():
    plt.rcParams.update({"font.size": 18})
    f, ax_ar = plt.subplots(2, 2, sharex=True, figsize=(15, 7))

    # # colors greyscale
    # wind1 = "#999999"
    # wind2 = "#333333"
    # solar1 = "#999999"
    # solar2 = "#333333"

    # colors
    wind1 = "#4254ff"
    wind2 = "#1b2053"
    solar1 = "#ffba00"
    solar2 = "#ff7000"

    fn = os.path.join(
        cfg.get("paths", "data_my_reegis"),
        "full_load_hours_re_bdew_states.csv",
    )
    flh = pd.read_csv(fn, index_col=[0], header=[0, 1])
    regions = geometries.get_federal_states_polygon()

    for y in [2014, 2012]:
        re_rg = feedin.scenario_feedin(regions, y, "fs").swaplevel(axis=1)

        flh["Wind (reegis)", str(y)] = re_rg["wind"].sum()
        flh["Solar (reegis)", str(y)] = re_rg["solar"].sum()

    ax_ar[0, 0] = flh[
        [("Wind (BDEW)", "2012"), ("Wind (reegis)", "2012")]
    ].plot(kind="bar", ax=ax_ar[0, 0], color=[wind1, wind2], legend=False)
    ax_ar[0, 1] = flh[
        [("Wind (BDEW)", "2014"), ("Wind (reegis)", "2014")]
    ].plot(kind="bar", ax=ax_ar[0, 1], color=[wind1, wind2], legend=False)
    ax_ar[1, 0] = flh[
        [("Solar (BDEW)", "2012"), ("Solar (reegis)", "2012")]
    ].plot(kind="bar", ax=ax_ar[1, 0], color=[solar1, solar2], legend=False)
    ax_ar[1, 1] = flh[
        [("Solar (BDEW)", "2014"), ("Solar (reegis)", "2014")]
    ].plot(kind="bar", ax=ax_ar[1, 1], color=[solar1, solar2], legend=False)
    ax_ar[0, 0].set_title("2012")
    ax_ar[0, 1].set_title("2014")
    ax_ar[0, 1].legend(
        loc="upper left", bbox_to_anchor=(1, 1), labels=["BDEW", "reegis"]
    )
    ax_ar[1, 1].legend(
        loc="upper left", bbox_to_anchor=(1, 1), labels=["BDEW", "reegis"]
    )
    ax_ar[0, 0].set_ylabel("Volllaststunden\nWindkraft")
    ax_ar[1, 0].set_ylabel("Volllaststunden\nPhotovoltaik")

    plt.subplots_adjust(
        right=0.871, left=0.098, bottom=0.11, top=0.94, wspace=0.16, hspace=0.1
    )
    return "compare_full_load_hours", None


def fig_compare_feedin_solar():
    plt.rcParams.update({"font.size": 18})
    f, ax_ar = plt.subplots(2, 1, sharey=True, figsize=(15, 6))

    # Get feedin time series from reegis
    regions = geometries.get_federal_states_polygon()
    re_rg = feedin.scenario_feedin(regions, 2014, "fs").set_index(
        pd.date_range(
            "31/12/2013 23:00:00", periods=8760, freq="H", tz="Europe/Berlin"
        )
    )

    # Get entsoe time series for pv profiles from opsd
    url = (
        "https://data.open-power-system-data.org/index.php?package"
        "=time_series&version={version}&action=customDownload&resource=3"
        "&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bfrom%5D=2005-01"
        "-01&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bto%5D=2019-05"
        "-01&filter%5BRegion%5D%5B%5D=DE&filter%5BVariable%5D%5B%5D"
        "=solar_capacity&filter%5BVariable%5D%5B%5D=solar_generation_actual"
        "&filter%5BVariable%5D%5B%5D=solar_profile&downloadCSV=Download+CSV"
    )
    my_re = entsoe.get_filtered_file(
        url=url, name="solar_de_2019-06-05", version="2019-06-05"
    )
    # Convert index to datetime
    my_re.set_index(
        pd.to_datetime(my_re["utc_timestamp"], utc=True).dt.tz_convert(
            "Europe/Berlin"
        ),
        inplace=True,
    )
    my_re.drop(["cet_cest_timestamp", "utc_timestamp"], axis=1, inplace=True)

    # Convert columns to numeric
    for c in my_re.columns:
        my_re[c] = pd.to_numeric(my_re[c]).div(1000)

    # Plot opsd data
    cso = "#ff7e00"
    csr = "#500000"

    my_re["DE_solar_profile"].multiply(1000)
    ax = my_re["DE_solar_profile"].multiply(1000).plot(ax=ax_ar[0], color=cso)
    ax2 = my_re["DE_solar_profile"].multiply(1000).plot(ax=ax_ar[1], color=cso)

    fs = geometries.get_federal_states_polygon()
    pp = powerplants.get_powerplants_by_region(fs, 2014, "federal_states")
    total_capacity = pp.capacity_2014.swaplevel().loc["Solar"].sum()

    re_rg = re_rg.swaplevel(axis=1)["solar"].mul(
        pp.capacity_2014.swaplevel().loc["Solar"]
    )

    # Plot reegis time series
    # June
    ax = (
        re_rg.sum(axis=1)
        .div(total_capacity)
        .plot(
            ax=ax,
            rot=0,
            color=csr,
            xlim=(
                datetime.datetime(2014, 6, 1),
                datetime.datetime(2014, 6, 30),
            ),
        )
    )

    # December
    ax2 = (
        re_rg.sum(axis=1)
        .div(total_capacity)
        .plot(
            ax=ax2,
            rot=0,
            color=csr,
            xlim=(
                datetime.datetime(2014, 12, 1),
                datetime.datetime(2014, 12, 30),
            ),
        )
    )

    # x-ticks for June
    dates = [
        datetime.datetime(2014, 6, 1),
        datetime.datetime(2014, 6, 5),
        datetime.datetime(2014, 6, 9),
        datetime.datetime(2014, 6, 13),
        datetime.datetime(2014, 6, 17),
        datetime.datetime(2014, 6, 21),
        datetime.datetime(2014, 6, 25),
        datetime.datetime(2014, 6, 29),
    ]
    ax.set_xticks([pandas_datetime for pandas_datetime in dates])
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ""
    ax.set_xticklabels(labels, ha="center", rotation=0)

    # xticks for December
    dates = [
        datetime.datetime(2014, 12, 1),
        datetime.datetime(2014, 12, 5),
        datetime.datetime(2014, 12, 9),
        datetime.datetime(2014, 12, 13),
        datetime.datetime(2014, 12, 17),
        datetime.datetime(2014, 12, 21),
        datetime.datetime(2014, 12, 25),
        datetime.datetime(2014, 12, 29),
    ]
    ax2.set_xticks([pandas_datetime for pandas_datetime in dates])
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ""
    ax2.set_xticklabels(labels, ha="center", rotation=0)

    ax.legend(labels=["OPSD", "reegis"])
    ax.set_xlabel("")
    ax.set_ylim((0, 1.1))
    ax2.set_xlabel("Juni/Dezember 2014")
    ax2.xaxis.labelpad = 20

    # Plot Text
    x0 = datetime.datetime(2014, 12, 1, 5, 0)
    x1 = datetime.datetime(2014, 12, 1, 8, 0)
    x2 = datetime.datetime(2014, 12, 3, 1, 0)

    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2015, 1, 1)

    # BMWI
    # https://www.bmwi.de/Redaktion/DE/Publikationen/Energie/
    #     erneuerbare-energien-in-zahlen-2017.pdf?__blob=publicationFile&v=27
    bmwi_sum = round(36.056)
    reegis_sum = round(re_rg.sum().sum() / 1000000)
    opsd_sum = round(
        my_re.DE_solar_generation_actual.loc[start:end].sum() / 1000
    )

    text = {
        "title": (x1, 1, " Summe 2014"),
        "op1": (x1, 0.85, "OPSD"),
        "op2": (x2, 0.85, "{0} GWh".format(int(opsd_sum))),
        "reg1": (x1, 0.70, "reegis"),
        "reg2": (x2, 0.70, "{0} GWh".format(int(reegis_sum))),
        "bmwi1": (x1, 0.55, "BMWi"),
        "bmwi2": (x2, 0.55, "{0} GWh".format(int(bmwi_sum))),
    }

    for t, c in text.items():
        if t == "title":
            w = "bold"
        else:
            w = "normal"
        ax2.text(c[0], c[1], c[2], weight=w, size=16, ha="left", va="center")

    # Plot Box
    x3 = mdates.date2num(x0)
    b = patches.Rectangle((x3, 0.5), 3.9, 0.57, color="#cccccc")
    ax2.add_patch(b)
    ax2.add_patch(patches.Shadow(b, -0.05, -0.01))

    plt.subplots_adjust(right=0.99, left=0.05, bottom=0.16, top=0.97)
    return "compare_feedin_solar", None


def fig_compare_feedin_wind_absolute():
    fig_compare_feedin_wind(scale_reegis=False)
    return "compare_feedin_wind_absolute", None


def fig_compare_feedin_wind_scaled():
    fig_compare_feedin_wind(scale_reegis=True)
    return "compare_feedin_wind_scaled", None


def fig_compare_feedin_wind(scale_reegis):
    plt.rcParams.update({"font.size": 18})
    f, ax_ar = plt.subplots(2, 1, sharey=True, figsize=(15, 6))

    # colors
    cwo = "#665eff"
    cwr = "#0a085e"

    # Get entsoe time series for wind profiles from opsd
    url = (
        "https://data.open-power-system-data.org/index.php?package"
        "=time_series&version={version}&action=customDownload&resource=3"
        "&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bfrom%5D=2005-01"
        "-01&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bto%5D=2019-05"
        "-01&filter%5BRegion%5D%5B%5D=DE&filter%5BVariable%5D%5B%5D"
        "=wind_capacity&filter%5BVariable%5D%5B%5D=wind_generation_actual"
        "&filter%5BVariable%5D%5B%5D=wind_profile&downloadCSV=Download+CSV"
    )
    re_en = entsoe.get_filtered_file(
        url=url, name="wind_de_2019-06-05", version="2019-06-05"
    )
    # Convert index to datetime
    re_en.set_index(
        pd.to_datetime(re_en["utc_timestamp"], utc=True).dt.tz_convert(
            "Europe/Berlin"
        ),
        inplace=True,
    )
    re_en.drop(["cet_cest_timestamp", "utc_timestamp"], axis=1, inplace=True)

    # Convert columns to numeric
    for c in re_en.columns:
        re_en[c] = pd.to_numeric(re_en[c]).div(1000)

    # Plot entsoe data
    ax = re_en["DE_wind_profile"].multiply(1000).plot(ax=ax_ar[0], color=cwo)
    ax2 = re_en["DE_wind_profile"].multiply(1000).plot(ax=ax_ar[1], color=cwo)

    # Get feedin time series from reegis
    regions = geometries.get_federal_states_polygon()
    re_rg = feedin.scenario_feedin(regions, 2014, "fs").set_index(
        pd.date_range(
            "31/12/2013 23:00:00", periods=8760, freq="H", tz="Europe/Berlin"
        )
    )

    fs = geometries.get_federal_states_polygon()
    pp = powerplants.get_powerplants_by_region(fs, 2014, "federal_states")
    total_capacity = pp.capacity_2014.swaplevel().loc["Wind"].sum()
    re_rg = re_rg.swaplevel(axis=1)["wind"].mul(
        pp.capacity_2014.swaplevel().loc["Wind"]
    )

    # Set interval
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2015, 1, 1)

    if scale_reegis is True:
        re_rg = re_rg.mul(
            re_en.DE_wind_generation_actual.loc[start:end].sum()
            / re_rg.sum().sum()
            * 1000
        )
    print(
        re_en.DE_wind_generation_actual.loc[start:end].sum()
        / re_rg.sum().sum()
        * 1000
    )
    # Plot reegis time series (use multiply to adjust the overall sum)
    # June
    ax = (
        re_rg.sum(axis=1)
        .div(total_capacity)
        .multiply(1)
        .plot(
            ax=ax,
            rot=0,
            color=cwr,
            xlim=(
                datetime.datetime(2014, 6, 1),
                datetime.datetime(2014, 6, 30),
            ),
        )
    )

    # December
    ax2 = (
        re_rg.sum(axis=1)
        .div(total_capacity)
        .multiply(1)
        .plot(
            ax=ax2,
            rot=0,
            color=cwr,
            xlim=(
                datetime.datetime(2014, 12, 1),
                datetime.datetime(2014, 12, 30),
            ),
        )
    )

    # x-ticks for June
    dates = [
        datetime.datetime(2014, 6, 1),
        datetime.datetime(2014, 6, 5),
        datetime.datetime(2014, 6, 9),
        datetime.datetime(2014, 6, 13),
        datetime.datetime(2014, 6, 17),
        datetime.datetime(2014, 6, 21),
        datetime.datetime(2014, 6, 25),
        datetime.datetime(2014, 6, 29),
    ]
    ax.set_xticks([pandas_datetime for pandas_datetime in dates])
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ""
    ax.set_xticklabels(labels, ha="center", rotation=0)

    # xticks for December
    dates = [
        datetime.datetime(2014, 12, 1),
        datetime.datetime(2014, 12, 5),
        datetime.datetime(2014, 12, 9),
        datetime.datetime(2014, 12, 13),
        datetime.datetime(2014, 12, 17),
        datetime.datetime(2014, 12, 21),
        datetime.datetime(2014, 12, 25),
        datetime.datetime(2014, 12, 29),
    ]
    ax2.set_xticks([pandas_datetime for pandas_datetime in dates])
    labels = [pandas_datetime.strftime("%d. %b.") for pandas_datetime in dates]
    labels[0] = ""
    ax2.set_xticklabels(labels, ha="center", rotation=0)

    ax.legend(labels=["OPSD", "reegis"])
    ax.set_xlabel("")
    ax.set_ylim((0, 1.1))
    ax2.set_xlabel("Juni/Dezember 2014")
    ax2.xaxis.labelpad = 20

    # Plot Text
    x0 = datetime.datetime(2014, 6, 1, 5, 0)
    x1 = datetime.datetime(2014, 6, 1, 8, 0)
    x2 = datetime.datetime(2014, 6, 3, 1, 0)

    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2015, 1, 1)

    # BMWI
    # https://www.bmwi.de/Redaktion/DE/Publikationen/Energie/
    #     erneuerbare-energien-in-zahlen-2017.pdf?__blob=publicationFile&v=27
    bmwi_sum = round((1471 + 57026) / 1000)
    reegis_sum = round(re_rg.sum().sum() / 1000000)
    opsd_sum = round(
        re_en.DE_wind_generation_actual.loc[start:end].sum() / 1000
    )

    print(opsd_sum / reegis_sum)

    text = {
        "title": (x1, 1, " Summe 2014"),
        "op1": (x1, 0.85, "OPSD"),
        "op2": (x2, 0.85, "{0} GWh".format(int(opsd_sum))),
        "reg1": (x1, 0.70, "reegis"),
        "reg2": (x2, 0.70, "{0} GWh".format(int(reegis_sum))),
        "bmwi1": (x1, 0.55, "BMWi"),
        "bmwi2": (x2, 0.55, "{0} GWh".format(int(bmwi_sum))),
    }

    for t, c in text.items():
        if t == "title":
            w = "bold"
        else:
            w = "normal"
        ax.text(c[0], c[1], c[2], weight=w, size=18, ha="left", va="center")

    # Plot Box
    x3 = mdates.date2num(x0)
    b = patches.Rectangle((x3, 0.5), 4.4, 0.57, color="#cccccc")
    ax.add_patch(b)
    ax.add_patch(patches.Shadow(b, -0.05, -0.01))

    plt.subplots_adjust(right=0.99, left=0.05, bottom=0.16, top=0.97)


def fig_ego_demand_plot():
    ax = create_subplot((10.7, 9))

    de = geometries.get_germany_polygon(with_awz=False)
    de["geometry"] = de["geometry"].simplify(0.01)
    ax = de.plot(
        ax=ax, alpha=0.5, color="white", edgecolor="#000000", aspect="equal"
    )

    ego_demand = geometries.load_csv(
        cfg.get("paths", "static_sources"),
        cfg.get("open_ego", "ego_input_file"),
    )
    ego_demand = geometries.create_geo_df(ego_demand, wkt_column="st_astext")
    ax = ego_demand.plot(
        markersize=0.1, ax=ax, color="#272740", aspect="equal"
    )

    print("Number of points: {0}".format(len(ego_demand)))

    # Remove frame around plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axis("off")

    plt.subplots_adjust(right=1, left=0, bottom=0, top=1)

    return "open_ego_map.png", None


def fig_compare_electricity_profile_berlin():
    plt.rcParams.update({"font.size": 16})
    f, ax_ar = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # # colors greyscale
    # v = "0.1"
    # e = "0.4"
    # e2 = "0.3"

    # no greyscale
    v = None
    e = None
    e2 = None

    year = 2014
    federal_states = geometries.get_federal_states_polygon()
    fn = "berlin_electricity_data_2014_berlin.csv"
    from_fn = os.path.join(cfg.get("paths", "data_my_reegis"), fn)
    to_fn = os.path.join(cfg.get("paths", "electricity"), fn)
    shutil.copy(from_fn, to_fn)
    bln_elec = electricity.get_electricity_demand(year)
    fs_demand = demand_elec.get_entsoe_profile_by_region(
        federal_states, year, "federal_states", "entsoe", version="2019-06-05"
    )

    bln_vatten = bln_elec.usage
    bln_vatten.name = "Berlin Profil"
    bln_entsoe = fs_demand["BE"].multiply(
        bln_elec.usage.sum() / fs_demand["BE"].sum()
    )
    bln_entsoe.name = "Entsoe-Profil (skaliert)"
    bln_reegis = fs_demand["BE"].div(1000)
    bln_reegis.name = "Entsoe-Profil (reegis)"

    ax = ax_ar[0]
    start = datetime.datetime(year, 1, 13)
    end = datetime.datetime(year, 1, 20)

    ax = bln_vatten.loc[start:end].plot(ax=ax, x_compat=True, linewidth=3, c=v)
    ax = bln_entsoe.loc[start:end].plot(ax=ax, x_compat=True, linewidth=3, c=e)
    ax.set_title("Winterwoche (13. - 20. Januar)")
    # ax.set_xticks()
    ax.set_xticks(
        [
            n
            for n in bln_vatten.index
            if n.month == 1 and 20 > n.day > 12 and n.hour == 12
        ]
    )
    ax.set_xticklabels(
        [
            n.strftime("%d")
            for n in bln_vatten.index
            if n.month == 1 and 20 > n.day > 12 and n.hour == 12
        ],
        rotation=0,
        horizontalalignment="center",
    )
    ax.set_xlabel("Januar 2014")
    ax.set_ylabel("[GW]")
    ax.set_xlim(start, end - datetime.timedelta(hours=1))
    ax = ax_ar[1]
    start = datetime.datetime(year, 7, 14)
    end = datetime.datetime(year, 7, 21)
    ax = bln_vatten.loc[start:end].plot(ax=ax, x_compat=True, linewidth=3, c=v)
    ax = bln_entsoe.loc[start:end].plot(ax=ax, x_compat=True, linewidth=3, c=e)
    ax.set_title("Sommerwoche (14. - 20. Juli)")
    ax.set_xticks(
        [
            n
            for n in bln_vatten.index
            if n.month == 7 and 21 > n.day > 13 and n.hour == 12
        ]
    )
    ax.set_xticklabels(
        [
            n.strftime("%d")
            for n in bln_vatten.index
            if n.month == 7 and 21 > n.day > 13 and n.hour == 12
        ],
        rotation=0,
        horizontalalignment="center",
    )
    ax.set_xlabel("Juli 2014")
    ax.set_xlim(start, end - datetime.timedelta(hours=2))
    ax = ax_ar[2]
    ax = (
        bln_vatten.resample("W")
        .mean()
        .plot(ax=ax, legend=True, x_compat=True, linewidth=3, c=v)
    )
    ax = (
        bln_entsoe.resample("W")
        .mean()
        .plot(ax=ax, legend=True, x_compat=True, linewidth=3, c=e)
    )
    ax = (
        bln_reegis.resample("W")
        .mean()
        .plot(ax=ax, legend=True, x_compat=True, linewidth=3, c=e2)
    )
    ax.set_title("Wochenmittel - 2014")
    ax.set_xticks(
        [
            n
            for n in bln_vatten.index
            if (n.month % 2) == 0 and n.day == 1 and n.hour == 1
        ]
    )
    ax.set_xticklabels(
        [
            n.strftime("%b")
            for n in bln_vatten.index
            if (n.month % 2) == 0 and n.day == 1 and n.hour == 1
        ],
        rotation=0,
        horizontalalignment="left",
    )
    dates = bln_vatten.resample("W").mean().index
    print(dates)
    ax.set_xlabel("2014")
    ax.set_xlim(dates[0], dates[-2])
    plt.subplots_adjust(left=0.04, top=0.92, bottom=0.11, right=0.99)

    return "compare_electricity_profile_berlin", None


def fig_entsoe_year_plots():
    logger.define_logging()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    my_entsoe = entsoe.split_timeseries_file(version="2019-06-05").load
    print(my_entsoe.index)
    for y in range(2006, 2015):
        start = datetime.datetime(y, 1, 1, 0, 0)
        end = datetime.datetime(y, 12, 31, 23, 0)
        start = start.astimezone(pytz.timezone("Europe/Berlin"))
        end = end.astimezone(pytz.timezone("Europe/Berlin"))
        de_load_profile = my_entsoe.loc[start:end].DE_load_
        ax = de_load_profile.resample("M").mean().plot(ax=ax)


def fig_entsoe_scaled_year_plots():
    # locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    logger.define_logging()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    netto2014 = 511500000
    federal_states = geometries.get_federal_states_polygon()
    for y in range(2006, 2015):
        logging.info("Calculating {0}".format(y))
        my_entsoe = demand_elec.get_entsoe_profile_by_region(
            federal_states,
            y,
            "openego_entsoe",
            annual_demand=netto2014,
            version="2019-06-05",
        ).sum(1)
        ax = my_entsoe.resample("M").mean().plot(ax=ax)


def fig_compare_entsoe_slp_germany():
    plt.rcParams.update({"font.size": 16})

    # # colors greyscale
    # colors = ["#999999", "#333333"]

    colors = None

    my_year = 2014
    federal_states = geometries.get_germany_polygon(with_awz=True)
    name = "germany"

    netto2014 = 511500000

    slp_de = demand_elec.get_open_ego_slp_profile_by_region(
        federal_states, my_year, name, annual_demand=netto2014
    ).sum(1)
    my_entsoe = demand_elec.get_entsoe_profile_by_region(
        federal_states,
        my_year,
        name,
        annual_demand=netto2014,
        version="2019-06-05",
    ).sum(1)

    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(
        wspace=0.05, left=0.07, right=0.98, bottom=0.11, top=0.95
    )
    slp_de_no_idx = slp_de.reset_index(drop=True)
    entsoe_no_idx = my_entsoe.reset_index(drop=True)

    df = pd.DataFrame(
        pd.concat(
            [slp_de_no_idx, entsoe_no_idx],
            axis=1,
            keys=["Standardlastprofil", "Entsoe-Profil", "geglättet"],
        )
    )

    my_ax1 = fig.add_subplot(1, 2, 1)
    df.loc[22 * 24 : 28 * 24].div(1000).plot(
        ax=my_ax1, linewidth=3, style=["-", "-"], color=colors
    )
    my_ax1.legend_.remove()
    plt.ylim([30.100, 90.000])
    plt.xlim([528, 28 * 24])
    plt.ylabel("Mittlerer Stromverbrauch [MW]")
    plt.xticks(
        [528 + 12, 552 + 12, 576 + 12, 600 + 12, 624 + 12, 648 + 12],
        ["Do", "Fr", "Sa", "So", "Mo", "Di"],
        rotation="horizontal",
        horizontalalignment="center",
    )
    plt.xlabel("23. - 28. Januar 2014")
    my_ax2 = fig.add_subplot(1, 2, 2)
    df.loc[204 * 24 : 210 * 24].div(1000).plot(
        ax=my_ax2, linewidth=3, style=["-", "-", "-."], color=colors
    )
    plt.ylim([30.100, 90.000])
    plt.xlim([204 * 24, 210 * 24])
    my_ax2.get_yaxis().set_visible(False)
    plt.xticks(
        [4908, 4932, 4956, 4980, 5004, 5028],
        ["Do", "Fr", "Sa", "So", "Mo", "Di"],
        rotation="horizontal",
        horizontalalignment="center",
    )
    plt.xlabel("24. - 29. Juli 2014")
    plt.legend(facecolor="white", framealpha=1, shadow=True)
    return "demand_SLP_vs_ENTSOE", None


def fig_compare_entsoe_slp_rolling_window():
    plt.rcParams.update({"font.size": 16})
    logger.define_logging()
    my_year = 2014
    federal_states = geometries.get_federal_states_polygon()
    name = "germany"
    netto2014 = 511500000

    slp_de = demand_elec.get_open_ego_slp_profile_by_region(
        federal_states,
        my_year,
        name,
        annual_demand=netto2014,
        dynamic_H0=False,
    ).sum(1)
    my_entsoe = demand_elec.get_entsoe_profile_by_region(
        federal_states,
        my_year,
        name,
        annual_demand=netto2014,
        version="2019-06-05",
    ).sum(1)

    print(slp_de.sum())
    print(my_entsoe.sum())

    fig, ax_ar = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
    fig.subplots_adjust(
        wspace=0.05, left=0.09, right=0.96, bottom=0.12, top=0.95
    )
    slp_de_no_idx = slp_de.reset_index(drop=True)
    entsoe_no_idx = my_entsoe.reset_index(drop=True)

    p4 = slp_de_no_idx.rolling(9, center=True).mean()
    p3 = slp_de_no_idx.rolling(7, center=True).mean()
    p2 = slp_de_no_idx.rolling(5, center=True).mean()
    p1 = slp_de_no_idx.rolling(3, center=True).mean()

    df = pd.DataFrame(
        pd.concat(
            [slp_de_no_idx, p1, p2, p3, p4, entsoe_no_idx],
            axis=1,
            keys=[
                "Standardlastprofil",
                "geglättet (1h)",
                "geglättet (2h)",
                "geglättet (3h)",
                "geglättet (4h)",
                "Entsoe-Profil",
            ],
        )
    )

    df.loc[525:556].plot(
        ax=ax_ar[0], linewidth=2, style=["-", "-", "-", "-", "-", "k-"]
    )
    ax_ar[0].legend_.remove()
    ax_ar[0].set_ylim([30100, 90000])
    ax_ar[0].set_xlim([525, 556])
    ax_ar[0].set_ylabel("Mittlerer Stromverbrauch [kW]")
    ax_ar[0].set_xlabel("23. Januar 2014")
    ax_ar[0].set_xticks([527, 533, 539, 545, 551])
    ax_ar[0].set_xticklabels(["00:00", "06:00", "12:00", "18:00", "00:00"])

    df.loc[4893:4924].plot(
        ax=ax_ar[1], linewidth=2, style=["-", "-", "-", "-", "-", "k-"]
    )
    ax_ar[1].set_xlim([4893, 4924])
    ax_ar[1].set_xlabel("24. Juli 2014")
    ax_ar[1].set_xticks([4895, 4901, 4907, 4913, 4919])
    ax_ar[1].set_xticklabels(["00:00", "06:00", "12:00", "18:00", "00:00"])
    plt.legend(
        facecolor="white",
        framealpha=1,
        shadow=True,
        loc="upper right",
        bbox_to_anchor=(1.1, 1),
        ncol=2,
    )
    return "demand_SLP_geglättet_vs_ENTSOE", None


def fig_compare_entsoe_slp_annual_profile():
    plt.rcParams.update({"font.size": 16})
    # locale.setlocale(locale.LC_ALL, "de_DE.utf8")

    # no greyscale
    # s = "0.1"
    # e = "0.4"

    # no greyscale
    s = None
    e = None

    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.11, top=0.98)
    logger.define_logging()
    my_year = 2014
    federal_states = geometries.get_federal_states_polygon()
    name = "federal_states"

    # Jährlicher Verlauf
    netto2014 = 511500000
    slp_de = demand_elec.get_open_ego_slp_profile_by_region(
        federal_states,
        my_year,
        name,
        annual_demand=netto2014,
        dynamic_H0=False,
    ).sum(1)

    my_entsoe = demand_elec.get_entsoe_profile_by_region(
        federal_states,
        my_year,
        name,
        annual_demand=netto2014,
        version="2019-06-05",
    ).sum(1)

    slp_de_month = slp_de.resample("M").mean()
    entsoe_month = my_entsoe.resample("M").mean()
    slp_ax = slp_de_month.reset_index(drop=True).plot(
        label="Standardlastprofil", linewidth=3, c=s,
    )
    entsoe_month.reset_index(drop=True).plot(
        ax=slp_ax, label="Entsoe-Profil", linewidth=3, c=e,
    )

    e_avg = entsoe_month.mean()
    e_max = entsoe_month.max()
    e_min = entsoe_month.min()
    d_e_max = int(round((e_max / e_avg - 1) * 100))
    d_e_min = int(round((1 - e_min / e_avg) * 100))
    s_avg = slp_de_month.mean()
    s_max = slp_de_month.max()
    s_min = slp_de_month.min()
    d_s_max = round((s_max / s_avg - 1) * 100, 1)
    # d_s_min = round((1 - s_min / s_avg) * 100, 1)
    plt.plot((0, 1000), (s_max, s_max), "k-.")
    plt.plot((0, 13), (s_min, s_min), "k-.")
    plt.plot((0, 12), (e_max, e_max), "k-.")
    plt.plot((0, 12), (e_min, e_min), "k-.")
    plt.text(5, e_max - 500, "+{0}%".format(d_e_max))
    plt.text(5, e_min + 250, "-{0}%".format(d_e_min))
    plt.text(5, s_max + 200, "+/-{0}%".format(d_s_max))
    plt.legend(facecolor="white", framealpha=1, shadow=True)
    plt.ylabel("Mittlerer Stromverbrauch [kW]")

    plt.xlim(0, 11)
    plt.xticks(
        list(range(12)),
        pd.date_range("2014", "2015", freq="MS").strftime("%b")[:-1],
        rotation="horizontal",
        horizontalalignment="center",
    )

    plt.xlabel("2014")
    return "demand_Saisonaler_Vergleich_SLP_ENTSOE", None


def fig_demand_share_of_sector_and_region():
    plt.rcParams.update({"font.size": 20})
    year = 2014
    fig = plt.figure(figsize=(16, 5))
    fig.subplots_adjust(
        wspace=0.538, left=0.062, right=0.888, bottom=0.14, top=0.926
    )
    logger.define_logging()
    federal_states = geometries.get_federal_states_polygon()
    name = "federal_states"

    # Jährlicher Verlauf
    netto2014 = 511500000
    demand_fs = demand_elec.get_open_ego_slp_profile_by_region(
        federal_states, year, name, annual_demand=netto2014, dynamic_H0=False
    )
    demand_fs.drop("P0", axis=1, inplace=True)
    share_relative = pd.DataFrame()

    for region in demand_fs.columns.get_level_values(0):
        sc_sum = demand_fs[region].sum().sum()
        for p_type in demand_fs[region].columns:
            share_relative.loc[region, p_type] = (
                demand_fs[region, p_type].sum() / sc_sum
            )

    share_relative.loc["  ", "h0"] = 0
    share_relative.loc["  ", "g0"] = 0
    share_relative.loc["  ", "l0"] = 0
    share_relative.loc["  ", "ind"] = 0
    share_relative.loc["DE "] = share_relative.sum().div(16)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1 = share_relative.plot(kind="bar", stacked=True, ax=ax1)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        bbox_to_anchor=(1, 0.87),
    )
    ax1.set_ylabel("Anteil")

    holidays = {
        datetime.date(year, 5, 24): "Whit Monday",
        datetime.date(year, 4, 5): "Easter Monday",
        datetime.date(year, 5, 13): "Ascension Thursday",
        datetime.date(year, 1, 1): "New year",
        datetime.date(year, 10, 3): "Day of German Unity",
        datetime.date(year, 12, 25): "Christmas Day",
        datetime.date(year, 5, 1): "Labour Day",
        datetime.date(year, 4, 2): "Good Friday",
        datetime.date(year, 12, 26): "Second Christmas Day",
    }

    ann_el_demand_per_sector = {
        "h0": 1000,
        "g0": 1000,
        "l0": 1000,
        "ind": 1000,
    }

    # read standard load profiles
    e_slp = bdew.ElecSlp(year, holidays=holidays)

    # multiply given annual demand with timeseries
    elec_demand = e_slp.get_profile(
        ann_el_demand_per_sector, dyn_function_h0=False
    )

    # Add the slp for the industrial group
    ilp = profiles.IndustrialLoadProfile(
        e_slp.date_time_index, holidays=holidays
    )

    # Beginning and end of workday, weekdays and weekend days, and scaling
    # factors by default.
    elec_demand["ind"] = ilp.simple_profile(ann_el_demand_per_sector["ind"])

    elec_demand["mix"] = (
        elec_demand["h0"] * 0.291229
        + elec_demand["g0"] * 0.185620
        + elec_demand["l0"] * 0.100173
        + elec_demand["ind"] * 0.422978
    )

    # Resample 15-minute values to monthly values.
    elec_demand = elec_demand.resample("M").mean()
    print(elec_demand.mean())
    mean_d = elec_demand.mean()
    elec_demand = (elec_demand - mean_d) / 0.114158 * 100
    # Plot demand
    ax2 = fig.add_subplot(1, 2, 2)
    elec_demand = elec_demand[["mix", "h0", "g0", "l0", "ind"]]
    print(elec_demand)
    ax2 = elec_demand.reset_index(drop=True).plot(
        style=["k-.", "-", "-", "-", "-"], ax=ax2, xticks=[], linewidth=3
    )
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        bbox_to_anchor=(1, 0.87),
    )

    ax2.set_xticks(
        [n for n in elec_demand.reset_index().index if (n % 2) == 0]
    )
    ax2.set_xticklabels(
        [n.strftime("%b") for n in elec_demand.index if (n.month % 2) != 0],
        rotation=0,
        horizontalalignment="left",
    )
    ax2.set_xlim(0, 11)
    ax2.set_xlabel("")
    ax2.set_ylabel("Abweichung vom Jahresmittel [%]")
    return "demand_composing_the_mixed_profile", None


def fig_compare_habitants_and_heat_electricity_share(**kwargs):
    plt.rcParams.update({"font.size": 16})
    ax = create_subplot((9, 4), **kwargs)
    eb = energy_balance.get_usage_balance(2014).swaplevel()
    blnc = eb.loc["total", ("electricity", "district heating")]
    print(blnc)
    ew = pd.DataFrame(inhabitants.get_ew_by_federal_states(2014))
    res = pd.merge(blnc, ew, right_index=True, left_index=True)
    fraction = pd.DataFrame(index=res.index)
    for col in res.columns:
        fraction[col] = res[col].div(res[col].sum())

    fraction.rename(
        columns={
            "electricity": "Anteil Strombedarf",
            "district heating": "Anteil Fernwärme",
            "EWZ": "Anteil Einwohner",
        },
        inplace=True,
    )
    fraction = fraction[
        ["Anteil Strombedarf", "Anteil Einwohner", "Anteil Fernwärme"]
    ]
    ax = fraction.plot(kind="bar", ax=ax, rot=0)
    ax.set_xlabel("")
    plt.subplots_adjust(right=0.99, left=0.07, bottom=0.09, top=0.98)
    return "compare_habitants_and_heat_electricity_share", None


def fig_compare_district_heating_habitants_bw():
    """
    Source:

    https://www.landtag-bw.de/files/live/sites/LTBW/files/dokumente/WP15/Drucksachen/6000/15_6086_D.pdf

    """
    source_url = (
        "https://www.landtag-bw.de/files/live/sites/LTBW/files/dokumente/WP15"
        "/Drucksachen/6000/15_6086_D.pdf"
    )
    share_dh = {"de11": 48, "de12": 34, "unloc": 18}

    # load geometries
    federal_states = geometries.get_federal_states_polygon()
    de21 = geometries.load_shp(
        cfg.get("paths", "geo_plot"), "region_polygons_de21.geojson"
    )
    de21.set_index("region", inplace=True)

    # fetch share of habitants
    share = inhabitants.get_share_of_federal_states_by_region(
        2014, de21, "de21"
    )
    sde11 = int(round(share.loc["DE11", "BW"] * 100))
    sde12 = int(round(share.loc["DE12", "BW"] * 100))

    # plot geometries
    ax = (
        de21.loc[["DE11"], "geometry"]
        .simplify(0.01)
        .boundary.plot(color="#aaaaaa", aspect="equal", linewidth=3)
    )
    federal_states.loc[["BW"], "geometry"].simplify(0.01).boundary.plot(
        color="#555555", linewidth=4, aspect="equal", ax=ax
    )

    # add text to plot
    text_a = "unbestimmt\nFW: ~{0}%"
    text_b = "EW: ~{0}%\nFW: ~{1}%"
    text = [
        (7.5, 49.7, text_a.format(share_dh["unloc"]), "#000000", None,),
        (8.58, 49.2, text_b.format(sde11, share_dh["de11"]), "black", None,),
        (8.25, 48.2, text_b.format(sde12, share_dh["de12"]), "black", None,),
        (9.4, 49.46, "DE11", "red", "bold"),
        (9.4, 48.43, "DE12", "red", "bold"),
    ]

    for c in text:
        plt.text(
            c[0],
            c[1],
            c[2],
            color=c[3],
            size=20,
            ha="left",
            va="center",
            fontweight=c[4],
        )

    # adjust plot
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    print("Data source: {0}".format(source_url))
    return "BadenWuertemberg.svg", None
