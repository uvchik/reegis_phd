from matplotlib import pyplot as plt
from my_reegis import reegis_plot as plot


def fig_model_regions():
    """Plot one or more model regions in one plot."""
    maps = ["de02", "de17", "de21", "de22"]
    # maps = ["de22"]
    add_title = False

    top = 1
    ax = None
    ax_ar = []

    width = len(maps * 3)

    if len(maps) > 1:
        f, ax_ar = plt.subplots(1, len(maps), figsize=(width, 2.5))
    else:
        ax = plt.figure(figsize=(width, 2.5)).add_subplot(1, 1, 1)
    i = 0
    for rmap in maps:
        if len(maps) > 1:
            ax = ax_ar[i]

        plot.plot_regions(
            deflex_map=rmap, ax=ax, legend=False, simple=0.005, offshore="auto"
        )
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        ax.axis("off")
        if add_title is True:
            ax.set_title(rmap)
            top = 0.88
        i += 1

    plt.subplots_adjust(right=1, left=0, wspace=0, bottom=0, top=top)
    return "model_regions", None
