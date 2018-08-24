import os
import results
from oemof.tools import logger
import berlin_hp
import reegis_tools.config as cfg
from matplotlib import pyplot as plt
import reegis_tools.gui as gui


def fig_6_1(**kwargs):

    if kwargs.get('size') is None:
        size_6_1 = (10, 10)
    else:
        size_6_1 = kwargs.get('size')

    my_es = results.load_es(2013, 'de21', 'deflex')
    my_es_2 = results.load_es(2013, 'de21', 'berlin_hp')
    transmission = results.compare_transmission(my_es, my_es_2)

    key = gui.get_choice(list(transmission.columns),
                         "Plot transmission lines", "Choose data column.")
    vmax = max([abs(transmission[key].max()), abs(transmission[key].min())])
    units = {'es1': 'GWh', 'es2': 'GWh', 'diff_2-1': 'GWh', 'fraction': '%'}
    results.plot_power_lines(transmission, key, vmax=vmax/5, unit=units[key],
                             size=size_6_1)


def fig_6_x_draft1(**kwargs):
    if kwargs.get('size') is None:
        size = (5, 5)
    else:
        size = kwargs.get('size')

    ax = plt.figure(figsize=size).add_subplot(1, 1, 1)

    my_es = results.load_es(2014, 'de21', 'deflex')
    my_es_2 = results.load_es(2014, 'de21', 'berlin_hp')
    transmission = results.compare_transmission(my_es, my_es_2)

    # PLOTS
    transmission = transmission.div(1000)
    transmission.plot(kind='bar', ax=ax)


def plot_figure(number, filename=None, show=False, **kwargs):

    number_name = {
        '6.1': fig_6_1,
        '6.x': fig_6_x_draft1,
    }

    number_name[number](**kwargs)

    if filename is not None:
        plt.savefig(filename)

    if show is True or filename is None:
        plt.show()


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__)])
    plot_figure('6.1')
