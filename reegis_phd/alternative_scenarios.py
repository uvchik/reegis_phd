# Python libraries
import copy
import os
from datetime import datetime

import deflex
from deflex import geometries
from deflex.scenario_tools import Scenario
from reegis_phd import main
from reegis import config as cfg
from reegis import inhabitants


def stopwatch():
    if not hasattr(stopwatch, "start"):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def reduce_power_plants(sc, nuclear=None, lignite=None, hard_coal=None):
    sc.table_collection["transformer"] = (
        sc.table_collection["transformer"].swaplevel().sort_index()
    )
    # remove nuclear power (by law)
    if nuclear is not None:
        if nuclear == 0:
            sc.table_collection["transformer"].drop(
                "nuclear", axis=0, inplace=True
            )
        else:
            sc.table_collection["transformer"].loc["nuclear", "capacity"] = (
                sc.table_collection["transformer"]
                .loc["nuclear", "capacity"]
                .multiply(nuclear)
            )

    if lignite is not None:
        # remove have lignite (climate change)
        sc.table_collection["transformer"].loc["lignite", "capacity"] = (
            sc.table_collection["transformer"]
            .loc["lignite", "capacity"]
            .multiply(lignite)
        )

    if hard_coal is not None:
        # remove have lignite (climate change)
        sc.table_collection["transformer"].loc["hard coal"] = (
            sc.table_collection["transformer"]
            .loc["hard coal"]
            .multiply(hard_coal)
        )

    sc.table_collection["transformer"] = (
        sc.table_collection["transformer"].swaplevel().sort_index()
    )


def more_heat_pumps(sc, heat_pump_fraction, cop):
    year = 2014
    abs_decentr_heat = sc.table_collection["demand_series"]["DE_demand"].sum(
        axis=1
    )
    heat_pump = abs_decentr_heat * heat_pump_fraction
    sc.table_collection["demand_series"]["DE_demand"] *= 1 - heat_pump_fraction

    deflex_regions = geometries.deflex_regions(rmap=sc.map)
    name = "{0}_region".format(sc.map)
    inhab = inhabitants.get_inhabitants_by_region(
        year, deflex_regions, name=name
    )

    inhab_fraction = inhab.div(inhab.sum())

    for region in inhab_fraction.index:
        if inhab_fraction.loc[region] > 0:
            sc.table_collection["demand_series"][
                (region, "electrical_load")
            ] += inhab_fraction.loc[region] * heat_pump.div(cop)


def increase_re_share(sc, factor):
    t = sc.table_collection["volatile_source"]
    for region in t.index.get_level_values(0).unique():
        for vs in t.loc[region].index:
            t.loc[(region, vs), "capacity"] *= factor


def add_simple_gas_turbine(sc, nom_val, efficiency=0.39):
    sc.table_collection["commodity_source"].loc[
        ("DE", "natural gas add"),
        sc.table_collection["commodity_source"].columns,
    ] = sc.table_collection["commodity_source"].loc[("DE", "natural gas")]

    for region in nom_val.keys():
        sc.table_collection["transformer"].loc[
            (region, "natural gas add"), "efficiency"
        ] = efficiency
        sc.table_collection["transformer"].loc[
            (region, "natural gas add"), "capacity"
        ] = int(5 * round((nom_val[region] / 100 + 2.5) / 5) * 100)
        sc.table_collection["transformer"].loc[
            (region, "natural gas add"), "limit_elec_pp"
        ] = "inf"
        sc.table_collection["transformer"].loc[
            (region, "natural gas add"), "fuel"
        ] = "natural gas add"


def create_scenario_xx_nc00_hp02(base_sc, subpath="new", factor=0.0):
    """remove nuclear
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)
    sub = "XX_Nc00_HP02_f{0}".format(str(factor).replace(".", ""))
    base = base_sc.table_collection["meta"].loc["name", "value"]
    sc.name = "{0}_{1}".format(base, sub)
    sc.table_collection["meta"].loc["name", "value"] = sc.name

    increase_re_share(sc, factor)

    reduce_power_plants(sc, nuclear=0)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_scenario_xx_nc00_hp00(base_sc, subpath="new", factor=0.0):
    """remove nuclear
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)
    sub = "XX_Nc00_HP00_f{0}".format(str(factor).replace(".", ""))
    base = base_sc.table_collection["meta"].loc["name", "value"]
    sc.name = "{0}_{1}".format(base, sub)
    sc.table_collection["meta"].loc["name", "value"] = sc.name

    increase_re_share(sc, factor)

    reduce_power_plants(sc, nuclear=0)

    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_scenario_xx_nc00_li05_hp02_gt(base_sc, subpath="new", factor=0.0):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)
    sub = "XX_Nc00_Li05_HP02_GT_f{0}".format(str(factor).replace(".", ""))
    base = base_sc.table_collection["meta"].loc["name", "value"]
    sc.name = "{0}_{1}".format(base, sub)
    sc.table_collection["meta"].loc["name", "value"] = sc.name

    nom_val = {
        "DE01": 1455.2,
        "DE02": 2012.2,
        "DE03": 1908.8,
        "DE04": 0.0,
        "DE05": 0.0,
        "DE06": 0.0,
        "DE07": 0.0,
        "DE08": 0.0,
        "DE09": 3527.6,
        "DE10": 1736.7,
        "DE11": 0.0,
        "DE12": 7942.3,
        "DE13": 947.5,
        "DE14": 0.0,
        "DE15": 1047.7,
        "DE16": 1981.7,
        "DE17": 3803.8,
        "DE18": 3481.9,
        "DE19": 0.0,
        "DE20": 0.0,
        "DE21": 0.0,
    }

    # sc = main.load_deflex_scenario(2014, create_scenario=False)

    increase_re_share(sc, factor)

    add_simple_gas_turbine(sc, nom_val)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_scenario_xx_nc00_li05_hp00_gt(
    base_sc, subpath="alternative", factor=0.0
):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)
    sub = "XX_Nc00_Li05_HP00_GT_f{0}".format(str(factor).replace(".", ""))
    base = base_sc.table_collection["meta"].loc["name", "value"]
    sc.name = "{0}_{1}".format(base, sub)
    sc.table_collection["meta"].loc["name", "value"] = sc.name

    nom_val = {
        "DE01": 1455.2,
        "DE02": 2012.2,
        "DE03": 1908.8,
        "DE04": 0.0,
        "DE05": 0.0,
        "DE06": 0.0,
        "DE07": 0.0,
        "DE08": 0.0,
        "DE09": 3527.6,
        "DE10": 1736.7,
        "DE11": 0.0,
        "DE12": 7942.3,
        "DE13": 947.5,
        "DE14": 0.0,
        "DE15": 1047.7,
        "DE16": 1981.7,
        "DE17": 3803.8,
        "DE18": 3481.9,
        "DE19": 0.0,
        "DE20": 0.0,
        "DE21": 0.0,
    }

    increase_re_share(sc, factor)

    add_simple_gas_turbine(sc, nom_val)

    reduce_power_plants(sc, nuclear=0, lignite=0.5)

    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def simple_re_variant(base_sc, subpath, factor=0.0):
    sc = copy.deepcopy(base_sc)
    sub = "de21_f{0}".format(str(factor).replace(".", ""))
    base = base_sc.table_collection["meta"].loc["name", "value"]
    sc.name = "{0}_{1}".format(base, sub)
    sc.table_collection["meta"].loc["name", "value"] = sc.name

    increase_re_share(sc, factor)

    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_deflex_no_grid_limit(base_sc, subpath):
    sc = copy.deepcopy(base_sc)

    cond = sc.table_collection["transmission"]["electrical", "capacity"] > 0
    sc.table_collection["transmission"].loc[
        cond, ("electrical", "capacity")
    ] = float("inf")
    sc.table_collection["transmission"]["electrical", "efficiency"] = 1
    sc.name = sc.name + "_no_grid_limit"
    if "meta" in sc.table_collection:
        sc.table_collection["meta"].loc["name", "value"] = sc.name
    path = os.path.join(os.path.dirname(sc.location), subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # sc.to_csv(os.path.join(path, sc.name + "_csv"))


def create_deflex_no_storage(base_sc, subpath):
    sc = copy.deepcopy(base_sc)

    del sc.table_collection["storages"]
    sc.name = sc.name + "_no_storage"
    if "meta" in sc.table_collection:
        sc.table_collection["meta"].loc["name", "value"] = sc.name
    path = os.path.join(os.path.dirname(sc.location), subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # sc.to_csv(os.path.join(path, sc.name + "_csv"))


def create_deflex_no_grid_limit_no_storage(base_sc, subpath):
    sc = copy.deepcopy(base_sc)

    del sc.table_collection["storages"]
    cond = sc.table_collection["transmission"]["electrical", "capacity"] > 0
    sc.table_collection["transmission"].loc[
        cond, ("electrical", "capacity")
    ] = float("inf")
    sc.table_collection["transmission"]["electrical", "efficiency"] = 1
    sc.name = sc.name + "_no_grid_limit_no_storage"
    if "meta" in sc.table_collection:
        sc.table_collection["meta"].loc["name", "value"] = sc.name
    path = os.path.join(os.path.dirname(sc.location), subpath)
    sc.to_excel(os.path.join(path, sc.name + ".xls"))
    # sc.to_csv(os.path.join(path, sc.name + "_csv"))


if __name__ == "__main__":
    pass
