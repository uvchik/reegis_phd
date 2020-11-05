import os
import pandas as pd
from reegis_phd import results
from reegis import config as cfg


def analyse_berlin_ressources():
    s = {}
    scenarios = {
        "berlin_single": {
            "path": ["region"],
            "file": "berlin_hp_2014_single",
            "var": None,
            "region": "BE",
        },
        "berlin_deflex": {
            "path": ["region"],
            "file": (
                "berlin_hp_2014_single_DCPL_c1_deflex_2014_de22_without_berlin"
            ),
            "var": "de22",
            "region": "BE",
        },
        "berlin_up_deflex": {
            "path": ["region"],
            "file": "berlin_hp_2014_single_UP_deflex_2014_de22_without_DE22",
            "var": "single_up_deflex_2014_de22_without_berlin",
            "region": "BE",
        },
        "berlin_up_deflex_full": {
            "path": ["region"],
            "file": "berlin_hp_2014_single_UP_deflex_2014_de22",
            "var": "single_up_deflex_2014_de22",
            "region": "BE",
        },
        "deflex_de22": {
            "path": ["base"],
            "file": "c1_deflex_2014_de22",
            "var": "de22",
            "region": "DE22",
        },
    }

    for k, v in scenarios.items():
        if cfg.has_option("results", "dir"):
            res_dir = cfg.get("results", "dir")
        else:
            res_dir = "results_cbc"
        path = os.path.join(cfg.get("paths", "phd"), *v["path"], res_dir)
        fn = os.path.join(path, v["file"] + ".esys")
        es = results.load_es(fn)
        results.check_excess_shortage(es)
        resource_balance = results.get_multiregion_bus_balance(
            es, "bus_commodity"
        )

        s[k] = resource_balance[
            v["region"], "in", "source", "commodity"
        ].copy()

        if v["var"] is not None:
            elec_balance = results.get_multiregion_bus_balance(es)
            s[k]["ee"] = elec_balance[v["region"], "in", "source", "ee"].sum(
                axis=1
            )

            import_berlin = elec_balance[
                v["region"], "in", "import", "electricity", "all"
            ]
            export_berlin = elec_balance[
                v["region"], "out", "export", "electricity", "all"
            ]
            s[k]["netto_import"] = import_berlin - export_berlin

        else:
            s[k]["netto_import"] = 0

    seq = pd.concat(s, axis=1).div(1000000)

    for scenario in seq.columns.get_level_values(0).unique():
        seq[scenario, "other"] = seq[scenario].get("other", 0)
        for c in ["waste", "bioenergy", "ee"]:
            seq[scenario, "other"] += seq[scenario].get(c, 0)

    for c in ["waste", "bioenergy", "ee"]:
        seq.drop(c, level=1, inplace=True, axis=1)

    return seq.swaplevel(axis=1)


def analyse_berlin_ressources_total():
    # Energiebilanz Berlin 2014
    statistic = pd.Series(
        {
            "bioenergy": 7152000,
            "hard_coal": 43245000,
            "lignite": 12274000,
            "natural_gas": 80635000,
            "oil": 29800000,
            "other": 477000,
            "ee": 337000,
            "netto_import": 19786000,
        }
    ).div(3.6)

    statistic["other"] = 0

    for c in ["bioenergy", "ee"]:
        statistic["other"] += statistic[c]
        del statistic[c]

    seq = analyse_berlin_ressources()
    df = seq.sum().unstack().T
    df.loc["statistic"] = statistic.div(1000000)
    return df.fillna(0)


if __name__ == "__main__":
    pass
