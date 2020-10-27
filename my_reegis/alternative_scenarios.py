# Python libraries
import copy
import os
from datetime import datetime

import deflex
from deflex import geometries
from deflex.scenario_tools import Scenario
from my_reegis import main
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


def add_one_gas_turbine(sc, nominal_value, efficiency=0.39):
    sc.table_collection["commodity_source"][
        ("DE", "natural gas add")
    ] = sc.table_collection["commodity_source"][("DE", "natural gas")]

    region = (
        sc.table_collection["transformer"]
        .columns.get_level_values(0)
        .unique()[0]
    )

    sc.table_collection["transformer"].loc[
        "efficiency", (region, "natural gas add")
    ] = efficiency
    sc.table_collection["transformer"].loc[
        "capacity", (region, "natural gas add")
    ] = nominal_value
    sc.table_collection["transformer"].loc[
        "limit_elec_pp", (region, "natural gas add")
    ] = "inf"


def find_scenarios(path, year, sub="", notsub="ß"):
    scenarios = []
    for root, directories, filenames in os.walk(path):
        for d in directories:
            if (
                d[-4:] == "_csv"
                and sub in d
                and notsub not in d
                and str(year) in d
            ):
                scenarios.append(os.path.join(root, d))
    return scenarios


def fetch_xx_scenarios(year):
    path = os.path.join(cfg.get("paths", "scenario"), "deflex")
    substring = "_XX_"
    return find_scenarios(path, year, substring)


def create_xx_scenario_set(year):
    fetch_xx_scenarios(year)
    path = os.path.join(cfg.get("paths", "scenario"), "deflex")
    substring = "no_grid_limit_no_storage"
    notsub = "_XX_"
    base_scenarios = find_scenarios(path, year, substring, notsub)
    nuclear = 0
    for fn in base_scenarios:
        sc_name = fn.split(os.sep)[-1][:-4]
        sc_map = str(sc_name).split("_")[2]
        sc = Scenario(name=sc_name)
        sc.map = sc_map
        sc.year = year
        for lignite_set in [(1, 0), (0.5, 30000)]:
            lignite = lignite_set[0]
            gas_capacity = lignite_set[1]
            for hp_frac in [0, 0.2]:
                for ee in range(0, 11):
                    ee_f = 1 + ee / 10
                    sc.load_csv(fn)
                    create_xx_scenarios(
                        sc, ee_f, gas_capacity, nuclear, lignite, hp_frac
                    )


def create_xx_scenarios(sc, ee_f, gas_capacity, nuclear, lignite, hp_frac):
    """Only for scenarios with one electricity market."""

    if ee_f > 1:
        increase_re_share(sc, ee_f)

    if gas_capacity > 0:
        add_one_gas_turbine(sc, gas_capacity)

    if nuclear != 1 or lignite != 1:
        reduce_power_plants(sc, nuclear, lignite)

    if hp_frac > 0:
        more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    sub = "XX_Nc{0}_Li{1}_HP{2}_GT{3}_f{4}".format(
        str(nuclear).replace(".", ""),
        str(lignite).replace(".", ""),
        str(hp_frac).replace(".", ""),
        int(gas_capacity / 1000),
        str(ee_f).replace(".", ""),
    )

    name = "{0}_{1}_{2}".format(sc.name, "alt", sub)
    path = os.path.join(cfg.get("paths", "scenario"), "deflex", str(sc.year))
    sc.to_excel(os.path.join(path, name + ".xls"))
    csv_path = os.path.join(path, "{0}_csv".format(name))
    sc.to_csv(csv_path)


def create_scenario_xx_nc00_hp02(base_sc, subpath="new", factor=0.0):
    """remove nuclear
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)

    increase_re_share(sc, factor)

    reduce_power_plants(sc, nuclear=0)

    more_heat_pumps(sc, heat_pump_fraction=0.2, cop=2)

    sub = "XX_Nc00_HP02_f{0}".format(str(factor).replace(".", ""))

    name = "{0}_{1}_{2}".format("deflex", sub, cfg.get("init", "map"))
    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.table_collection["meta"].loc["name"] = name
    sc.to_excel(os.path.join(path, name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_scenario_xx_nc00_hp00(base_sc, subpath="new", factor=0.0):
    """remove nuclear
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)

    increase_re_share(sc, factor)

    reduce_power_plants(sc, nuclear=0)

    sub = "XX_Nc00_HP00_f{0}".format(str(factor).replace(".", ""))

    name = "{0}_{1}_{2}".format("deflex", sub, sc.map)
    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.table_collection["meta"].loc["name"] = name
    sc.to_excel(os.path.join(path, name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_scenario_xx_nc00_li05_hp02_gt(base_sc, subpath="new", factor=0.0):
    """remove nuclear
    # reduce lignite by 50%
    # use massive heat_pumps
    """
    sc = copy.deepcopy(base_sc)

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

    sub = "XX_Nc00_Li05_HP02_GT_f{0}".format(str(factor).replace(".", ""))

    name = "{0}_{1}_{2}".format("deflex", sub, sc.map)
    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.table_collection["meta"].loc["name"] = name
    sc.to_excel(os.path.join(path, name + ".xls"))
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

    sub = "XX_Nc00_Li05_HP00_GT_f{0}".format(str(factor).replace(".", ""))

    name = "{0}_{1}_{2}".format("deflex", sub, sc.map)
    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.table_collection["meta"].loc["name"] = name
    sc.to_excel(os.path.join(path, name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def simple_re_variant(base_sc, subpath, factor=0.0):
    sc = copy.deepcopy(base_sc)

    increase_re_share(sc, factor)

    sub = "de21_f{0}".format(str(factor).replace(".", ""))

    name = "{0}_{1}_{2}".format("deflex", sub, sc.map)
    path = os.path.join(os.path.dirname(sc.location), os.pardir, subpath)
    sc.table_collection["meta"].loc["name"] = name
    sc.to_excel(os.path.join(path, name + ".xls"))
    # csv_path = os.path.join(path, "{0}_csv".format(name))
    # sc.to_csv(csv_path)


def create_deflex_no_grid_limit(base_sc, subpath):
    sc = copy.deepcopy(base_sc)

    cond = sc.table_collection["transmission"]["electrical", "capacity"] > 0
    sc.table_collection["transmission"].loc[
        cond, ("electrical", "capacity")
    ] = float("inf")
    sc.table_collection["transmission"]["electrical", "efficiency"] = 1
    name = sc.name + "_no_grid_limit"
    path = os.path.join(os.path.dirname(sc.location), subpath)
    sc.to_excel(os.path.join(path, name + ".xls"))
    # sc.to_csv(os.path.join(path, name + "_csv"))


def create_deflex_no_storage(base_sc, subpath):
    sc = copy.deepcopy(base_sc)

    del sc.table_collection["storages"]
    name = sc.name + "_no_storage"
    path = os.path.join(os.path.dirname(sc.location), subpath)
    sc.to_excel(os.path.join(path, name + ".xls"))
    # sc.to_csv(os.path.join(path, name + "_csv"))


def create_deflex_no_grid_limit_no_storage(base_sc, subpath):
    sc = copy.deepcopy(base_sc)

    del sc.table_collection["storages"]
    cond = sc.table_collection["transmission"]["electrical", "capacity"] > 0
    sc.table_collection["transmission"].loc[
        cond, ("electrical", "capacity")
    ] = float("inf")
    sc.table_collection["transmission"]["electrical", "efficiency"] = 1
    name = sc.name + "_no_grid_limit_no_storage"
    path = os.path.join(os.path.dirname(sc.location), subpath)
    sc.to_excel(os.path.join(path, name + ".xls"))
    # sc.to_csv(os.path.join(path, name + "_csv"))


if __name__ == "__main__":
    pass
