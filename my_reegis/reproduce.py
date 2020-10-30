import logging
import os
from pprint import pprint

import berlin_hp
import deflex
import pandas as pd
from oemof.tools import logger
from reegis import config as cfg

from my_reegis import __file__ as my_regis_file
from my_reegis import alternative_scenarios as alt
from my_reegis import embedded_model as emb
from my_reegis import main

SCENARIOS = {
    "berlin_single_2014": "berlin_hp_2014_single.xls",
    "de21_2014": "deflex_2014_de21_csv",
    "de22_2014": "deflex_2014_de21_csv",
    "de21_without_berlin_2014": "deflex_2014_de21_without_berlin_csv",
}

SPLITTER = {
    "berlin": ["berlin_hp", "berlin_single"],
    "base_var": ["base_var", "upstream"],
    "extend": ["extend", "alt"],
    "deflex": ["deflex"],
    "modellhagen": ["modellhagen", "friedrichshagen"],
}

cfg.init(
    paths=[
        os.path.dirname(berlin_hp.__file__),
        os.path.dirname(deflex.__file__),
        os.path.dirname(my_regis_file),
    ]
)


def split_scenarios(sc):
    splitted = {}
    for g, kws in SPLITTER.items():
        splitted[g] = []
        for keyword in kws:
            for s in sc:
                if keyword in s.split(os.sep)[-1]:
                    splitted[g].append(s)
    return splitted


def create_variant_extend_scenarios(fn, file_type):
    sc = deflex.main.load_scenario(fn, file_type)
    sc.map = sc.table_collection["meta"].loc["map"].value
    name = sc.table_collection["meta"].loc["name"].value
    sub_p = "extend_{0}".format(name)
    for f in [1, 1.5, 2]:
        alt.create_scenario_xx_nc00_li05_hp02_gt(sc, subpath=sub_p, factor=f)
        alt.create_scenario_xx_nc00_li05_hp00_gt(sc, subpath=sub_p, factor=f)
        alt.create_scenario_xx_nc00_hp02(sc, subpath=sub_p, factor=f)
        alt.create_scenario_xx_nc00_hp00(sc, subpath=sub_p, factor=f)
        alt.simple_re_variant(sc, subpath=sub_p, factor=f)


def create_variant_base_scenarios(scenarios, sub_path=None):
    if sub_path is None:
        sub_path = os.path.join(os.pardir, "base")

    for file in scenarios:
        sc = deflex.main.load_scenario(file, "excel")
        alt.create_deflex_no_grid_limit(sc, subpath=sub_path)
        alt.create_deflex_no_storage(sc, subpath=sub_path)
        alt.create_deflex_no_grid_limit_no_storage(sc, subpath=sub_path)


def fetch_create_scenarios(path):
    # download first scenarios to base_path + "base"
    base_path = os.path.join(path, "base")
    os.makedirs(base_path, exist_ok=True)
    scfn = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    scfn = split_scenarios(scfn)
    create_variant_base_scenarios(scfn["deflex"])
    base_scenario = [x for x in scfn["deflex"] if "deflex_2014_de21" in x][0]
    create_variant_extend_scenarios(base_scenario, "excel")


def get_costs_from_upstream_scenarios(path, filter_chp=True):
    # {"name": "test", "export": 5, "import": 4}

    res_list = deflex.results.search_results(path)
    res_list = [
        x
        for x in res_list
        if "deflex" in x and "UP" not in x and "dcpl" not in x
    ]
    n = len(res_list)
    res_dict = {k: v for v, k in zip(sorted(res_list), range(1, n + 1))}
    pprint(res_dict)
    my_results = deflex.results.restore_results(res_list)
    mcp = pd.DataFrame()
    results_d = {}
    for r in my_results:
        name = r["meta"]["scenario"]["name"]
        logging.info(name)
        results_d[name] = r
        cost_spec = deflex.analyses.get_flow_results(r)["cost", "specific"]
        if filter_chp:
            cost_spec = cost_spec.loc[
                slice(None), (slice(None), ["ee", "pp", "electricity"])
            ]
        mcp[name] = cost_spec.max(axis=1)
    result_file = os.path.join(
        path, "market_clearing_price_{0}.xls".format(os.path.basename(path))
    )
    mcp.to_excel(result_file)
    return result_file


def reproduce_folder(path):
    # fetch_create_scenarios(path)
    #
    # # Model deflex scenarios
    # sc = deflex.fetch_scenarios_from_dir(path=path, xls=True, recursive=True)
    # sc = split_scenarios(sc)
    # logd = os.path.join(path, "log_deflex.csv")
    # deflex.model_multi_scenarios(sc["deflex"], cpu_fraction=0.7, log_file=logd)
    mcp_file = get_costs_from_upstream_scenarios(path, filter_chp=True)
    #
    # # Model berlin scenarios
    # berlin_hp.model_scenarios(sc["berlin"])
    # main.modellhagen_re_variation(sc["modellhagen"][0])

    # Model directly combined scenarios
    base_path = os.path.join(path, "base")
    reg_path = os.path.join(path, "region")
    de = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    reg = deflex.fetch_scenarios_from_dir(path=reg_path, xls=True, csv=False)
    de.extend(reg)
    sc = split_scenarios(de)
    log_dcpl = os.path.join(path, "log_combined.csv")
    logging.info("Coupling {0} with {1}".format(sc["berlin"], sc["deflex"]))
    emb.model_multi_scenarios(
        sc["deflex"],
        sc["berlin"],
        cpu_fraction=0.6,
        log_file=log_dcpl,
    )

    # Model upstream combination of scenarios
    extend = os.path.join(path, "extend_deflex_2014_de21")
    b_sc = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    e_sc = deflex.fetch_scenarios_from_dir(path=extend, xls=True, csv=False)
    b_sc.extend(e_sc)
    sc = split_scenarios(b_sc)
    log_up = os.path.join(path, "log_b_upstream.csv")
    logging.info("{0} with upstream {1}".format(sc["berlin"], sc["deflex"]))
    emb.model_multi_scenarios(
        sc["deflex"],
        sc["berlin"],
        cpu_fraction=0.8,
        log_file=log_up,
        upstream=mcp_file,
    )


if __name__ == "__main__":
    logger.define_logging()
    cfg.tmp_set("paths", "phd", "/home/uwe/reegis/phd_c1")
    reproduce_folder(cfg.get("paths", "phd"))
