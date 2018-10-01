import logging
import numpy as np
import results
import reegis_tools.config as cfg
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import Normalize
import math
import reegis_tools.geometries
import oemof_visio as oev
from oemof import outputlib

ORDER_KEYS = ['hydro', 'geothermal', 'solar', 'pv', 'wind', 'chp', 'hp', 'pp',
              'import', 'shortage', 'power_line', 'demand',
              'heat_elec_decentralised', 'storage', 'export', 'excess']


def geopandas_colorbar_same_height(f, ax, vmin, vmax, cmap):
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    norm = Normalize(vmin=vmin, vmax=vmax)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    n_cmap.set_array(np.array([]))

    f.colorbar(n_cmap, cax=cax)


def shape_legend(node, rm_list, reverse=False, **kwargs):
    """Deprecated ?"""
    handels = kwargs['handles']
    labels = kwargs['labels']
    axes = kwargs['ax']
    parameter = {}

    new_labels = []
    for label in labels:
        label = label.replace('(', '')
        label = label.replace('), flow)', '')
        label = label.replace(str(node), '')
        label = label.replace(',', '')
        label = label.replace(' ', '')
        for item in rm_list:
            label = label.replace(item, '')
        new_labels.append(label)
    labels = new_labels

    parameter['bbox_to_anchor'] = kwargs.get('bbox_to_anchor', (1, 0.5))
    parameter['loc'] = kwargs.get('loc', 'center left')
    parameter['ncol'] = kwargs.get('ncol', 1)
    plotshare = kwargs.get('plotshare', 0.9)

    if reverse:
        handels = handels.reverse()
        labels = labels.reverse()

    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * plotshare, box.height])

    parameter['handles'] = handels
    parameter['labels'] = labels
    axes.legend(**parameter)
    return axes


def shape_tuple_legend(reverse=False, **kwargs):
    rm_list = ['source', 'trsf', 'electricity']
    handels = kwargs['handles']
    labels = kwargs['labels']
    axes = kwargs['ax']
    parameter = {}

    new_labels = []
    for label in labels:
        label = label.replace('(', '')
        label = label.replace(')', '')
        label = [x for x in label.split(', ') if x not in rm_list]
        label = ', '.join(label)
        new_labels.append(label)
    labels = new_labels

    parameter['bbox_to_anchor'] = kwargs.get('bbox_to_anchor', (1, 0.5))
    parameter['loc'] = kwargs.get('loc', 'center left')
    parameter['ncol'] = kwargs.get('ncol', 1)
    plotshare = kwargs.get('plotshare', 0.9)

    if reverse:
        handels = handels.reverse()
        labels = labels.reverse()

    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * plotshare, box.height])

    parameter['handles'] = handels
    parameter['labels'] = labels
    axes.legend(**parameter)
    return axes


def get_orderlist_from_multiindex(index, orderkeys=None):
    """Create an order list by searching the label for key words"""
    order = []

    if orderkeys is None:
        orderkeys = ORDER_KEYS

    index = list(index)

    for element in orderkeys:
        tmp = [x for x in index if element in str(x).lower()]
        for t in tmp:
            index.remove(t)
        order.extend(tmp)
    return order


def get_orderlist(my_node, flow=None):
    """Create an order list by searching the label for key words

    my_node : pd.DataFrame
        Sequence DataFrame of a node view.

    flow : str
        Direction of the flow 'in' or 'out'.

    """
    cols = list(my_node.columns)
    if flow == 'in':
        f = 0
    elif flow == 'out':
        f = 1
    else:
        logging.error("A flow has to be 'in' or 'out.")
        f = None
    order = []

    for element in ORDER_KEYS:
        tmp = [x for x in cols if element in x[0][f].lower()]
        for t in tmp:
            cols.remove(t)
        order.extend(tmp)
    return order


def get_cdict(my_node):
    my_colors = cfg.get_dict_list('plot_colors', string=True)
    color_dict = {}
    for col in my_node.columns:
        n = 0
        color_keys = list(my_colors.keys())

        try:
            while color_keys[n] not in str(col).lower():
                n += 1
            if len(my_colors[color_keys[n]]) > 1:
                color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
            else:
                color = '#{0}'.format(my_colors[color_keys[n]][0])
            color_dict[col] = color
        except IndexError:
            n = 0
            try:
                while color_keys[n] not in str(col).lower():
                    n += 1
                if len(my_colors[color_keys[n]]) > 1:
                    color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
                else:
                    color = '#{0}'.format(my_colors[color_keys[n]][0])
                color_dict[col] = color
            except IndexError:
                color_dict[col] = '#ff00f0'

    return color_dict


def get_cdict_df(df):
    my_colors = cfg.get_dict_list('plot_colors', string=True)
    color_dict = {}
    for col in df.columns:
        n = 0
        color_keys = list(my_colors.keys())
        try:
            while color_keys[n] not in str(col).lower():
                n += 1
            if len(my_colors[color_keys[n]]) > 1:
                color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
            else:
                color = '#{0}'.format(my_colors[color_keys[n]][0])
            color_dict[col] = color
        except IndexError:
            color_dict[col] = '#ff00f0'
    return color_dict


def plot_power_lines(data, key, cmap_lines=None, cmap_bg=None, direction=True,
                     vmax=None, label_min=None, label_max=None, unit='GWh',
                     size=None, ax=None, legend=True, unit_to_label=False,
                     divide=1, decimal=0, **kwargs):

    if size is None and ax is None:
        ax = plt.figure(figsize=(5, 5)).add_subplot(1, 1, 1)
    elif size is not None and ax is None:
        ax = plt.figure(figsize=size).add_subplot(1, 1, 1)

    if unit_to_label is True:
        label_unit = unit
    else:
        label_unit = ''

    lines = reegis_tools.geometries.load(
        cfg.get('paths', 'geometry'), cfg.get('geometry', 'de21_power_lines'))
    polygons = reegis_tools.geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('geometry', 'de21_polygons_simple'))

    lines = lines.merge(data.div(divide), left_index=True, right_index=True)

    lines['centroid'] = lines.centroid

    if cmap_bg is None:
        cmap_bg = LinearSegmentedColormap.from_list(
            'mycmap', [(0, '#aed8b4'), (1, '#bddce5')])

    if cmap_lines is None:
        cmap_lines = LinearSegmentedColormap.from_list(
            'mycmap', [
                (0, '#aaaaaa'),
                (0.0001, 'green'),
                (0.5, 'yellow'),
                (1, 'red')])

    for i, p in polygons.iterrows():
        if 'see' in p['name'].lower():
            polygons.loc[i, 'color'] = 1
        else:
            polygons.loc[i, 'color'] = 0

    lines['reverse'] = lines[key] < 0

    # if direction is False:
    lines.loc[lines['reverse'], key] = (
        lines.loc[lines['reverse'], key] * -1)

    if vmax is None:
        vmax = lines[key].max()

    if label_min is None:
        label_min = vmax * 0.5

    if label_max is None:
        label_max = float('inf')

    ax = polygons.plot(edgecolor='#9aa1a9', cmap=cmap_bg, column='color',
                       ax=ax)
    ax = lines.plot(cmap=cmap_lines, legend=legend, ax=ax, column=key, vmin=0,
                    vmax=vmax)
    for i, v in lines.iterrows():
        x1 = v['geometry'].coords[0][0]
        y1 = v['geometry'].coords[0][1]
        x2 = v['geometry'].coords[1][0]
        y2 = v['geometry'].coords[1][1]

        value_relative = v[key] / vmax
        mc = cmap_lines(value_relative)

        orient = math.atan(abs(x1-x2)/abs(y1-y2))

        if (y1 > y2) & (x1 > x2) or (y1 < y2) & (x1 < x2):
            orient *= -1

        if v['reverse']:
            orient += math.pi

        if v[key] == 0 or not direction:
            polygon = patches.RegularPolygon(
                (v['centroid'].x, v['centroid'].y),
                4,
                0.15,
                orientation=orient,
                color=(0, 0, 0, 0),
                zorder=10)
        else:
            polygon = patches.RegularPolygon(
                (v['centroid'].x, v['centroid'].y),
                3,
                0.15,
                orientation=orient,
                color=mc,
                zorder=10)
        ax.add_patch(polygon)

        if decimal == 0:
            value = int(round(v[key]))
        else:
            value = round(v[key], decimal)

        if label_min <= value <= label_max:
            if v['reverse'] is True and direction is False:
                value *= -1
            ax.text(
                v['centroid'].x, v['centroid'].y,
                '{0} {1}'.format(value, label_unit),
                color='#000000',
                fontsize=9.5,
                zorder=15,
                path_effects=[
                    path_effects.withStroke(linewidth=3, foreground="w")])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.axis('off')

    polygons.apply(lambda x: ax.annotate(
        x.name, xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

    return ax


def plot_regions(deflex_map=None, fn=None, data=None, textbox=True,
                 column=None, cmap=None, label_col=None, color=None,
                 edgecolor='#9aa1a9', legend=True, ax=None):
    if ax is None:
        ax = plt.figure().add_subplot(1, 1, 1)

    polygons = reegis_tools.geometries.Geometry()

    if deflex_map is not None:
        polygons.load(cfg.get('paths', 'geometry'),
                      cfg.get('geometry', deflex_map))
    elif fn is not None:
        polygons.load(fullname=fn)
    else:
        polygons.load(cfg.get('paths', 'geometry'),
                      cfg.get('geometry', 'de21_polygons_simple'))

    if label_col is None:
        label_col = column
    elif label_col == 'index':
        polygons.gdf['my_index'] = polygons.gdf.index
        label_col = 'my_index'

    if data is not None:
        polygons.gdf = polygons.gdf.merge(data, left_index=True,
                                          right_index=True)

    if cmap is None and color is None:
        cmap = LinearSegmentedColormap.from_list(
                'mycmap', [
                    # (0, '#aaaaaa'),
                    (0.000000000, 'green'),
                    (0.5, 'yellow'),
                    (1, 'red')])

    ax = polygons.gdf.plot(edgecolor=edgecolor, cmap=cmap, vmin=0, ax=ax,
                           legend=legend, column=column, color=color)

    if textbox is True:
        bb = dict(boxstyle="round", alpha=.5, ec=(1, 1, 1), fc=(1, 1, 1))
    elif isinstance(textbox, dict):
        bb = textbox
    else:
        bb = None

    if label_col is not None:
        polygons.gdf.apply(
            lambda x: ax.text(
                x.geometry.representative_point().x,
                x.geometry.representative_point().y,
                x[label_col], size=9,
                ha="center", va="center", bbox=bb),
            axis=1)

    return ax


def plot_bus(es, node_label, rm_list=None):
    """benutzt ???"""

    node = outputlib.views.node(es.results['Main'], node_label)

    fig = plt.figure(figsize=(10, 5))

    my_node = node['sequences']

    if rm_list is None:
        rm_list = []

    plot_slice = oev.plot.slice_df(my_node,)
    # date_from=datetime(2014, 5, 31),
    # date_to=datetime(2014, 6, 8))

    my_plot = oev.plot.io_plot(node_label, plot_slice,
                               cdict=get_cdict(my_node),
                               inorder=get_orderlist(my_node, 'in'),
                               outorder=get_orderlist(my_node, 'out'),
                               ax=fig.add_subplot(1, 1, 1),
                               smooth=True)
    ax = shape_legend(node_label, rm_list, **my_plot)
    ax = oev.plot.set_datetime_ticks(ax, plot_slice.index, tick_distance=48,
                                     date_format='%d-%m-%H', offset=12,
                                     tight=True)

    ax.set_ylabel('Power in MW')
    ax.set_xlabel('Year')
    ax.set_title("Electricity bus")
    plt.show()


def plot_bus_view(es=None, bus=None, data=None, ax=None, legend=True,
                  xlabel=None, ylabel=None, title=None, out_ol=None,
                  in_ol=None, period=None, smooth=True):
    """
    It is possible to pass an solph.EnergySystem with results, or a DataFrame
     with Multiindex columns and 'in' and 'out' in the first column level.
    If bus is specified only this bus of the EnergySystem will be plotted.

    Last check: 09/18
    """

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)

    if isinstance(bus, str):
        bus = es.groups[bus]

    if es is not None and bus is None:
        data = results.get_multiregion_bus_balance(es).groupby(
            axis=1, level=[1, 2, 3, 4]).sum()
        default_title = 'Germany'
    elif es is not None and bus is not None:
        data = results.reshape_bus_view(es, bus)[bus.label.region]
        default_title = repr(bus.label)
    else:
        default_title = None

    if period is not None:
        data = data.iloc[period[0]:period[1]]

    my_colors = get_cdict(data['in'])
    my_colors.update(get_cdict(data['out']))

    io_plot = oev.plot.io_plot(
        df_in=data['in'], df_out=data['out'], cdict=my_colors,
        inorder=get_orderlist_from_multiindex(data['in'].columns, in_ol),
        outorder=get_orderlist_from_multiindex(data['out'].columns, out_ol),
        ax=ax, smooth=smooth)

    if legend is True:
        io_plot['ax'] = shape_tuple_legend(**io_plot)
    else:
        io_plot['ax'].legend().set_visible(False)
        plt.draw()

    ax = oev.plot.set_datetime_ticks(io_plot['ax'], data.index,
                                     tick_distance=24, date_format='%d-%m-%Y',
                                     offset=12, tight=True)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is None:
        ax.set_title(default_title)
    else:
        ax.set_title(title)
    return ax
