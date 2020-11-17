import locale
import logging
import os
from zipfile import ZipFile

import berlin_hp
import deflex
import requests
from matplotlib import pyplot as plt
from oemof.tools import logger
from reegis import config as cfg

from reegis_phd import __file__
from reegis_phd.figures import figures_3x as fig3x
from reegis_phd.figures import figures_4x as fig4x
from reegis_phd.figures import figures_5x as fig5x
from reegis_phd.figures import figures_6x as fig6x

locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")


def download_scenario_results(path):
    url = "https://osf.io/rptve/download"

    ppath = os.path.join(path, "phd")

    if os.path.isdir(ppath) and len(os.listdir(ppath)) > 0:
        return ppath

    os.makedirs(ppath, exist_ok=True)
    fn = os.path.join(path, "phd_scenario_results.zip")

    if not os.path.isfile(fn):
        logging.info("Downloading '{0}'".format(os.path.basename(fn)))
        req = requests.get(url)
        with open(fn, "wb") as fout:
            fout.write(req.content)
            logging.info("{1} downloaded from {0}.".format(url, fn))

    with ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(path)
    logging.info("Scenarios results extracted to {0}".format(ppath))
    return ppath


def plot_figure(number, save=False, path=None, show=False, **kwargs):
    logging.info("***** PLOT FIGURE {0} ************".format(number))

    if path is None:
        path = cfg.get("paths", "local_root")

    fpath = os.path.join(path, "figures")
    cfg.tmp_set("paths", "figures", fpath)
    os.makedirs(fpath, exist_ok=True)

    if number not in get_number_name():
        msg = (
            "Figure {0} not found. Please choose from the following list: {1}"
        )
        raise ValueError(msg.format(number, list(get_number_name().keys())))
    elif float(number) > 6.1:
        ppath = download_scenario_results(path)
        cfg.tmp_set("paths", "phd", ppath)

    filename, fig_show = get_number_name()[number](**kwargs)

    if fig_show is not None:
        show = fig_show

    if "." not in filename:
        filename = filename + ".svg"

    if save is True:
        fn = os.path.join(fpath, filename)
        logging.info("Save figure as {0}".format(fn))
        plt.savefig(fn)
    logging.info("Plot")
    if show is True or save is not True:
        plt.show()


def plot_all(
    save=True, path=None, show=False, lower=0.0, upper=99.9, **kwargs
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
        "3.4": fig3x.fig_extraction_turbine_and_fixed_chp,
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
        "6.4": fig6x.berlin_resources_time_series,
        "6.5": fig6x.fig_netzkapazitaet_und_auslastung_de22,
        "6.6": fig6x.fig_veraenderung_energiefluesse_durch_kopplung,
        "6.7": fig6x.fig_import_export_100prz_region,
        "6.8": fig6x.fig_import_export_costs_100prz_region,
        "6.9": fig6x.fig_import_export_emissions_100prz_region,
    }


def main():
    import sys

    logger.define_logging(screen_level=logging.INFO)
    cfg.init(
        paths=[
            os.path.dirname(berlin_hp.__file__),
            os.path.dirname(deflex.__file__),
            os.path.dirname(__file__),
        ]
    )
    msg = "Unknown parameter: >>{0}<<. Only floats or 'all' are allowed."
    arg1 = sys.argv[1]
    if len(sys.argv) > 2:
        path = sys.argv[2]
    else:
        path = cfg.get("paths", "figures")

    try:
        arg1 = float(arg1)
    except ValueError:
        arg1 = arg1

    os.makedirs(path, exist_ok=True)
    if isinstance(arg1, str):
        if arg1 == "all":
            plot_all(path=path)
        else:
            raise ValueError(msg.format(arg1))
    elif isinstance(arg1, float):
        plot_figure(str(arg1), save=True, show=True, path=path)


if __name__ == "__main__":
    pass
