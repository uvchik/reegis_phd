from berlin_hp.main import main as bmain
from deflex import main
from my_reegis import config as cfg
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
    "combined": ["combined"],
    "upstream": ["up_", "upstream"],
    "deflex": ["deflex"],
}


def init():
    # download files from a server
    # unpack and copy to p = cfg.get("paths", "phd")
    pass


def split_scenarios(sc):
    splitted = {}
    for g, kws in SPLITTER.items():
        splitted[g] = []
        for keyword in kws:
            for s in sc:
                if keyword in s.split(os.sep)[-1]:
                    splitted[g].append(s)
    all_s = []
    for s in splitted.values():
        all_s.extend(s)
    duplicates = set([x for x in all_s if all_s.count(x) > 1])
    if len(duplicates) > 0:
        raise ValueError(
            "A scenario can only be in one group. The following scenarios "
            "are in more than one group: {0}\nCheck the SPLITTER keywords."
            "".format(list(duplicates))
        )
    print(all_s)
    return splitted


def reproduce_folder(path):
    sc = main.fetch_scenarios_from_dir(path=path, xls=True)
    sc = split_scenarios(sc)
    log_fn = os.path.join(path, "log.csv")
    main.model_multi_scenarios(sc["deflex"], cpu_fraction=0.8, log_file=log_fn)


# def reproduce_scenario(name):
#     file = SCENARIOS[name]
#     p = cfg.get("paths", "phd")
#
#     if "berlin_hp" in file:
#         y = int([x for x in file.split("_") if x.isnumeric()][0])
#         bmain(year=y, path=p, file=file)
#     elif "deflex" in file:
#         fn_csv = os.path.join(p, file)
#         model_scenario(csv_path=fn_csv)


if __name__ == "__main__":
    logger.define_logging()
    reproduce_folder(cfg.get("paths", "phd"))
