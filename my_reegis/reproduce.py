from berlin_hp.main import main as bmain
from deflex.main import model_scenario
from my_reegis import config as cfg
from oemof.tools import logger
import os


SCENARIOS = {
    "berlin_single_2014": "berlin_hp_2014_single.xls",
    "de21_2014": "deflex_2014_de21_csv",
    "de22_2014": "deflex_2014_de21_csv",
    "de21_without_berlin_2014": "deflex_2014_de21_without_berlin_csv",
}


def init():
    # download files from a server
    # unpack and copy to p = cfg.get("paths", "phd")
    pass


def reproduce_scenario(name):
    file = SCENARIOS[name]
    p = cfg.get("paths", "phd")

    if "berlin_hp" in file:
        y = int([x for x in file.split("_") if x.isnumeric()][0])
        bmain(year=y, path=p, file=file)
    elif "deflex" in file:
        fn_csv = os.path.join(p, file)
        model_scenario(csv_path=fn_csv)


if __name__ == "__main__":
    logger.define_logging()
    reproduce_scenario("de21_2014")
