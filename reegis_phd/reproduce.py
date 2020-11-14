import logging
import os
from pprint import pprint
from zipfile import ZipFile

import berlin_hp
import deflex
import pandas as pd
import requests
from oemof.tools import logger
from reegis import config as cfg

from reegis_phd import __file__ as my_regis_file
from reegis_phd import alternative_scenarios as alt
from reegis_phd import embedded_model as emb
from reegis_phd import main as phd_main

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
    path = os.path.join(os.path.dirname(sc.location), os.pardir, sub_p)
    for f in [1, 1.5, 2]:
        alt.create_scenario_xx_nc00_li05_hp02_gt(sc, path=path, factor=f)
        alt.create_scenario_xx_nc00_li05_hp00_gt(sc, path=path, factor=f)
        alt.create_scenario_xx_nc00_hp02(sc, path=path, factor=f)
        alt.create_scenario_xx_nc00_hp00(sc, path=path, factor=f)
        alt.simple_re_variant(sc, path=path, factor=f)
    return path


def create_variant_base_scenarios(scenarios, sub_path=None):
    if sub_path is None:
        sub_path = os.path.join(os.pardir, "base")

    for file in scenarios:
        sc = deflex.main.load_scenario(file, "excel")
        alt.create_deflex_no_grid_limit(sc, subpath=sub_path)
        alt.create_deflex_no_storage(sc, subpath=sub_path)
        alt.create_deflex_no_grid_limit_no_storage(sc, subpath=sub_path)


def download_base_scenarios(path):
    urlb = "https://files.de-1.osf.io/v1/resources/86dtf/providers/osfstorage"
    urls = {
        "base": "{0}/5fa3c4e0a5bb9d01170a2e3d/?zip=".format(urlb),
        "region": "{0}/5fa3f4fe9c4dcb0166e49208/?zip=".format(urlb),
    }

    for d, url in urls.items():
        spath = os.path.join(path, d)
        os.makedirs(spath, exist_ok=True)
        fn = os.path.join(spath, "phd_{0}_scenarios.zip".format(d))

        if not os.path.isfile(fn):
            logging.info("Downloading '{0}'".format(os.path.basename(fn)))
            req = requests.get(url)
            with open(fn, "wb") as fout:
                fout.write(req.content)
                logging.info("{1} downloaded from {0}.".format(url, fn))

        with ZipFile(fn, "r") as zip_ref:
            zip_ref.extractall(spath)
        logging.info("All {0} scenarios extracted to {1}".format(d, spath))


def fetch_create_scenarios(path):
    base_path = os.path.join(path, "base")
    os.makedirs(base_path, exist_ok=True)
    download_base_scenarios(path)

    scfn = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    base_scenario = [x for x in scfn if "phd_deflex_2014_de21" in x][0]
    return create_variant_extend_scenarios(base_scenario, "excel")


def get_costs_from_upstream_scenarios(
    path, infile=None, outfile=None, all_values=False, filter_chp=True
):
    value_path = os.path.join(path, "values")
    os.makedirs(value_path, exist_ok=True)
    res_list = deflex.results.search_results(path)
    res_list = [
        x
        for x in res_list
        if "deflex" in x and "UP" not in x and "dcpl" not in x
    ]
    if infile is not None:
        res_list = [x for x in res_list if infile in x]
    n = len(res_list)
    res_dict = {k: v for v, k in zip(sorted(res_list), range(1, n + 1))}
    pprint(res_dict)
    if len(res_list) > 0:
        my_results = deflex.results.restore_results(res_list)
    else:
        my_results = []
    mcp = pd.DataFrame()
    results_d = {}
    for r in my_results:
        name = r["meta"]["scenario"]["name"]
        logging.info(name)
        results_d[name] = r
        all_res_values = deflex.analyses.get_flow_results(r)
        if all_values is True:
            all_res_values.to_csv(os.path.join(value_path, name + ".csv"))
        cost_spec = all_res_values["cost", "specific"]
        if filter_chp:
            cost_spec = cost_spec.loc[
                slice(None), (slice(None), ["ee", "pp", "electricity"])
            ]
        mcp[name] = cost_spec.max(axis=1)

    if outfile is None:
        result_file_mcp = os.path.join(
            path,
            "market_clearing_price_{0}.xls".format(os.path.basename(path)),
        )
    else:
        result_file_mcp = outfile

    mcp.to_excel(result_file_mcp)
    return result_file_mcp


def reproduce_scenario_results(path):
    extend_path = fetch_create_scenarios(path)

    base_path = os.path.join(path, "base")

    # Model deflex scenarios
    sc = deflex.fetch_scenarios_from_dir(path=path, xls=True, recursive=True)
    sc = split_scenarios(sc)
    logd = os.path.join(path, "log_deflex.csv")
    deflex.model_multi_scenarios(sc["deflex"], cpu_fraction=0.7, log_file=logd)

    # Model berlin scenarios
    region_path = os.path.dirname(sc["berlin"][0])
    berlin_hp.model_scenarios(sc["berlin"])

    # Model "Modellhagen" scenarios
    phd_main.modellhagen_re_variation(sc["modellhagen"])

    # Model directly combined scenarios
    de = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    reg = deflex.fetch_scenarios_from_dir(
        path=region_path, xls=True, csv=False
    )
    de.extend(reg)
    sc = split_scenarios(de)
    log_dcpl = os.path.join(path, "log_combined.csv")
    logging.info("Coupling {0} with {1}".format(sc["berlin"], sc["deflex"]))
    emb.model_multi_scenarios(
        sc["deflex"], sc["berlin"], cpu_fraction=0.6, log_file=log_dcpl,
    )

    # Model upstream combination of scenarios
    mcp_file = get_costs_from_upstream_scenarios(path)
    b_sc = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    e_sc = deflex.fetch_scenarios_from_dir(
        path=extend_path, xls=True, csv=False
    )
    r_sc = deflex.fetch_scenarios_from_dir(
        path=region_path, xls=True, csv=False
    )

    sc = split_scenarios(b_sc + e_sc + r_sc)
    log_up = os.path.join(path, "log_b_upstream.csv")
    logging.info("{0} with upstream {1}".format(sc["berlin"], sc["deflex"]))
    emb.model_multi_scenarios(
        sc["deflex"],
        sc["berlin"],
        cpu_fraction=0.8,
        log_file=log_up,
        upstream=mcp_file,
    )


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
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = cfg.get("paths", "figures")
    os.makedirs(path, exist_ok=True)
    reproduce_scenario_results(path)


if __name__ == "__main__":
    pass
