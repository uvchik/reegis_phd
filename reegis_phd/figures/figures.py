import datetime
import logging
import os

import berlin_hp
import deflex
import geopandas as gpd
import numpy as np
import pandas as pd
#from berlin_hp import electricity
#from berlin_hp import heat
#from deflex import demand
from matplotlib import cm
from matplotlib import dates as mdates
from matplotlib import image as mpimg
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
from oemof import solph
from oemof.tools import logger
from reegis import bmwi
from reegis import coastdat
from reegis import config as cfg
from reegis import energy_balance
from reegis import entsoe
from reegis import geometries
from reegis import inhabitants
from reegis import powerplants
from reegis import storages
from scenario_builder import feedin

#from my_reegis import data_analysis
#from my_reegis import friedrichshagen_scenarios as fhg_sc
#from my_reegis import reegis_plot as plot
#from my_reegis import regional_results
#from my_reegis import results
from my_reegis.figures import figures_3x as fig3x
from my_reegis.figures import figures_4x as fig4x
from my_reegis.figures import figures_5x as fig5x
from my_reegis.figures import figures_6x as fig6x
import locale

locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')


def plot_figure(number, save=False, path=None, show=False, **kwargs):
    filename, fig_show = get_number_name()[number](**kwargs)

    if fig_show is not None:
        show = fig_show

    if "." not in filename:
        filename = filename + ".svg"

    if save is True:
        if path is None:
            path = ""
        fn = os.path.join(path, filename)
        logging.info("Save figure as {0}".format(fn))
        plt.savefig(fn)
    logging.info("Plot")
    if show is True or save is not True:
        plt.show()


def plot_all(
    save=False, path=None, show=False, lower=0.0, upper=99.9, **kwargs
):
    for number in get_number_name().keys():
        if lower < float(number) < upper:
            plot_figure(number, save=save, path=path, show=show, **kwargs)


def get_number_name():
    return {
        # ***** chapter 2 ****************************************************
        "2.1": fig3x.fig_solph_modular_example,
        # ***** chapter 3 ****************************************************
        "3.1": fig3x.fig_solph_energy_system_example,
        "3.2": fig3x.fig_transformer_with_flow,
        "3.3": fig3x.fig_extraction_turbine_characteristics,
        #"3.4": fig3x.fig_extraction_turbine_and_fixed_chp,
        "3.5": fig3x.fig_tespy_heat_pumps_cop,
        # ***** chapter 4 ****************************************************
        "4.1": fig4x.fig_patch_offshore,
        "4.2": fig4x.fig_powerplants,
        "4.3": fig4x.fig_storage_capacity,
        "4.4": fig4x.fig_inhabitants,
        "4.5": fig4x.fig_average_weather,
        "4.6": fig4x.fig_strahlungsmittel,
        "4.7": fig4x.fig_module_comparison,
        "4.8": fig4x.fig_analyse_multi_files,
        "4.9": fig4x.fig_polar_plot_pv_orientation,
        "4.10": fig4x.fig_windzones,
        "4.11": fig4x.fig_show_hydro_image,
        "4.12": fig4x.fig_compare_re_capacity_years,
        "4.13": fig4x.fig_compare_full_load_hours,
        "4.14": fig4x.fig_compare_feedin_solar,
        "4.15": fig4x.fig_compare_feedin_wind_absolute,
        "4.16": fig4x.fig_compare_feedin_wind_scaled,
        "4.17": fig4x.fig_ego_demand_plot,
        "4.18": fig4x.fig_compare_electricity_profile_berlin,
        "4.19": fig4x.fig_compare_entsoe_slp_germany,
        "4.20": fig4x.fig_compare_entsoe_slp_rolling_window,
        "4.21": fig4x.fig_compare_entsoe_slp_annual_profile,
        "4.22": fig4x.fig_demand_share_of_sector_and_region,
        "4.23": fig4x.fig_compare_habitants_and_heat_electricity_share,
        "4.24": fig4x.fig_compare_district_heating_habitants_bw,
        # ***** chapter 5 ****************************************************
        "5.1": fig5x.fig_model_regions,
        "5.2": fig5x.fig_compare_de21_region_borders,
        "5.3": fig5x.fig_show_download_deutschland_modell,
        "5.4": fig5x.fig_show_download_berlin_modell,
        "5.5": fig5x.fig_district_heating_areas,
        "5.6": fig5x.fig_deflex_de22_polygons,
        # ***** chapter 6 ****************************************************
        "6.1": fig6x.fig_anteil_import_stromverbrauch_berlin,
        "6.2": fig6x.fig_show_de21_de22_without_berlin,
        "6.3": fig6x.fig_berlin_resources,
        "6.0": fig6x.fig_absolute_power_flows,
        "6.02": fig6x.plot_upstream,
        "6.4": fig6x.berlin_resources_time_series,
        "6.5": fig6x.fig_netzkapazitaet_und_auslastung_de22,
        "6.6": fig6x.fig_veraenderung_energiefluesse_durch_kopplung,
        "6.7": fig6x.fig_import_export_100prz_region,
        "6.8": fig6x.fig_import_export_costs_100prz_region,
        "6.9": fig6x.fig_import_export_emissions_100prz_region,
        "6.10": fig6x.fig_6_x_draft1,
        "6.11": fig6x.fig_show_de21_de22_without_berlin,
    }


if __name__ == "__main__":
    logger.define_logging(screen_level=logging.INFO)
    cfg.init(
        paths=[
            os.path.dirname(berlin_hp.__file__),
            os.path.dirname(deflex.__file__),
        ]
    )
    p = cfg.get("paths", "figures")
    os.makedirs(p, exist_ok=True)
    # plot_all(save=True, upper=5.9, path=p)
    plot_figure("6.3", save=True, show=True, path=p)
