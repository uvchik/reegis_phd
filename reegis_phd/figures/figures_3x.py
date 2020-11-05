import os
import pandas as pd
import numpy as np
from reegis_phd import config as cfg
from reegis_phd.figures import figures_base
from matplotlib import pyplot as plt


def fig_solph_modular_example():
    return figures_base.show_download_image(
        "solph_modular", ["graphml", "svg"]
    )


def fig_solph_energy_system_example():
    return figures_base.show_download_image(
        "solph_example_deutsch", ["graphml", "svg"]
    )


def fig_transformer_with_flow():
    return figures_base.show_download_image(
        "transformer_mit_flow", ["graphml", "svg"]
    )


def fig_extraction_turbine_characteristics():
    return figures_base.show_download_image(
        "entnahmekondensationsturbine_kennfeld", ["svg"]
    )


def fig_extraction_turbine_and_fixed_chp():
    pass


def fig_tespy_heat_pumps_cop():
    """From TESPy examples."""
    plt.rcParams.update({"font.size": 16})
    f, ax_ar = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 5))

    t_range = [6, 9, 12, 15, 18, 21, 24]
    q_range = np.array([120e3, 140e3, 160e3, 180e3, 200e3, 220e3])

    n = 0
    for filename in ["COP_air.csv", "COP_water.csv"]:
        fn = os.path.join(cfg.get("paths", "data_my_reegis"), filename)
        df = pd.read_csv(fn, index_col=0)

        colors = [
            "#00395b",
            "#74adc1",
            "#b54036",
            "#ec6707",
            "#bfbfbf",
            "#999999",
            "#010101",
        ]
        plt.sca(ax_ar[n])
        i = 0
        for t in t_range:
            plt.plot(
                q_range / 200e3,
                df.loc[t],
                "-x",
                Color=colors[i],
                label="$T_{resvr}$ = " + str(t) + " °C",
                markersize=7,
                linewidth=2,
            )
            i += 1

        ax_ar[n].set_xlabel("Relative Last")

        if n == 0:
            ax_ar[n].set_ylabel("COP")
        n += 1
    plt.ylim([0, 3.2])
    plt.xlim([0.5, 1.2])
    plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.subplots_adjust(
        right=0.82, left=0.06, wspace=0.11, bottom=0.13, top=0.97
    )
    return "tespy_heat_pumps", None
