import berlin_hp
import deflex
from my_reegis import __file__ as my_regis_file
from my_reegis import alternative_scenarios as alt
from my_reegis import embedded_model as emb
from my_reegis import main
from reegis import config as cfg
from oemof.tools import logger
import os


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
        sub_path = os.path.join(os.pardir, "variant_base")

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


def reproduce_folder(path):
    fetch_create_scenarios(path)

    # Model deflex scenarios
    sc = deflex.fetch_scenarios_from_dir(path=path, xls=True, recursive=True)
    sc = split_scenarios(sc)
    logd = os.path.join(path, "log_deflex.csv")
    deflex.model_multi_scenarios(sc["deflex"], cpu_fraction=0.8, log_file=logd)

    # Model berlin scenarios
    berlin_hp.model_scenarios(sc["berlin"])
    main.modellhagen_re_variation(sc["modellhagen"][0])

    # Model directly combined scenarios
    base_path = os.path.join(path, "base")
    sc = deflex.fetch_scenarios_from_dir(path=base_path, xls=True, csv=False)
    sc = split_scenarios(sc)
    logc = os.path.join(path, "log_combined.csv")
    emb.model_multi_scenarios(
        sc["deflex"], sc["berlin"], cpu_fraction=0.8, log_file=logc
    )

    # Model upstream combination of scenarios


if __name__ == "__main__":
    logger.define_logging()
    reproduce_folder(cfg.get("paths", "phd"))
