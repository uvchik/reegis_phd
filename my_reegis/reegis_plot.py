import logging
import numpy as np
import pandas as pd
from my_reegis import results
import reegis.config as cfg
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import Normalize
import math
import reegis.geometries
from deflex import geometries as d_geometries
import oemof_visio as oev
from oemof import outputlib
# import reegis.gui as gui


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


def add_geopandas_label_coordinates(gdf, column='coords',
                                    representative_point=False):
    """Add a column with coordinates of centroid or representative point."""
    if representative_point:
        gdf[column] = gdf.geometry.apply(
            lambda x: x.representative_point().coords[:])
    else:
        gdf[column] = gdf.geometry.apply(
            lambda x: x.centroid.coords[:])

    gdf[column] = [coords[0] for coords in gdf['coords']]
    return gdf


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
                     divide=1, decimal=0):
    """
    Parameters
    ----------
    data
    key
    cmap_lines
    cmap_bg
    direction
    vmax
    label_min
    label_max
    unit
    size
    ax
    legend
    unit_to_label
    divide
    decimal

    Returns
    -------

    """
    if size is None and ax is None:
        ax = plt.figure(figsize=(5, 5)).add_subplot(1, 1, 1)
    elif size is not None and ax is None:
        ax = plt.figure(figsize=size).add_subplot(1, 1, 1)

    if unit_to_label is True:
        label_unit = unit
    else:
        label_unit = ''

    lines = reegis.geometries.load(
        cfg.get('paths', 'geometry'), cfg.get('geometry', 'de21_power_lines'))
    polygons = d_geometries.deflex_regions(rmap="de21", rtype='polygons')

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

    offshore = d_geometries.divide_off_and_onshore(polygons).offshore
    polygons["color"] = 0
    polygons.loc[offshore, "color"] = 1

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
                 data_col=None, cmap=None, label_col='data_col', color=None,
                 edgecolor='#9aa1a9', legend=True, ax=None, offshore=None,
                 simple=None):
    """
    Plot regions with special colors.

    1. Plot offshore and onshore regions with different colors.
    2. Plot all polygons with one color.
    3. Color polygons with a data column and a cmap

    Parameters
    ----------
    ax : matplotlib.axes
    cmap : matplotlib.cmap
        A color map.
    color : str
        A python color to draw all polygons in the given color. Overwrites
        the data colors.
    data : pd.DataFrame
        A data table where the index has the name index as the map.
    data_col : str
        Column of the data table containing the data for the color map.
    deflex_map : gpd.geoDataFrame
        A map with polygons.
    edgecolor : str
        The color of the border between the polygons.
    fn : str
        A path to map with polygons (e.g. shp, csv).
    label_col : None or str
        Name of the column with the labels. By default the data_col is used. If
        set to None or no data_col is specified no label is plotted. If 'index'
        is used the index will be plotted as label.
    legend : bool
        Draw a legend.
    offshore : list or string
        All elements of the geoDataFrame index that should be colored as
        offshore regions.
    textbox : bool
        Draw a box arround the label.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot

    Examples
    --------
    >>> d = {'DE01': 0.7, 'DE02': 0.5, 'DE03': 2, 'DE04': 2, 'DE05': 1,
    ...      'DE06': 1.5, 'DE07': 1.5, 'DE08': 2, 'DE09': 2.5, 'DE10': 3,
    ...      'DE11': 2.5, 'DE12': 3, 'DE13': 0.1, 'DE14': 0.5, 'DE15': 1,
    ...      'DE16': 2, 'DE17': 2.5, 'DE18': 3, 'DE19': 0, 'DE20': 0, 'DE21': 0
    ...     }
    >>> s = pd.Series(d)
    >>> df = pd.DataFrame(s, columns=['value'])
    >>> p1 = plot_regions(offshore=['DE21', 'DE20', 'DE19'], legend=False,
    ...                   label_col='index', textbox=False)
    >>> p2 = plot_regions(offshore=['DE21', 'DE20', 'DE19'], legend=False)
    >>> p3 = plot_regions(data=df, data_col='value', label_col=None)
    >>> plt.show()

    """
    if ax is None:
        ax = plt.figure().add_subplot(1, 1, 1)

    if deflex_map is not None:
        polygons = reegis.geometries.load(
            cfg.get('paths', 'geo_deflex'),
            cfg.get('geometry', 'deflex_polygon').format(
                suffix=".geojson", map=deflex_map, type='polygons'))
    elif fn is not None:
        polygons = reegis.geometries.load(fullname=fn)
    else:
        polygons = reegis.geometries.load(
            cfg.get('paths', 'geometry'),
            cfg.get('geometry', 'de21_polygons_simple'))

    if label_col == 'data_col':
        label_col = data_col
    elif label_col == 'index':
        polygons['my_index'] = polygons.index
        label_col = 'my_index'

    if data is not None:
        polygons = polygons.merge(data, left_index=True, right_index=True)

    if offshore == "auto":
        offshore = d_geometries.divide_off_and_onshore(polygons).offshore

    if offshore is not None:
        polygons['onshore'] = 1
        for o in offshore:
            polygons.loc[o, 'onshore'] = 0
        polygons.loc[polygons.numeric_id == 22, "onshore"] = 0.5
        cmap = LinearSegmentedColormap.from_list(
                'mycmap', [
                    (0, '#a5bfdd'),
                    (0.5, "red"),
                    (1, '#badd69')])
        data_col = 'onshore'

    if cmap is None and color is None and offshore is None:
        cmap = LinearSegmentedColormap.from_list(
                'mycmap', [
                    (0, '#aaaaaa'),
                    (0.00000001, 'green'),
                    (0.5, 'yellow'),
                    (1, 'red')])

    if simple is not None:
        polygons['geometry'] = polygons['geometry'].simplify(simple)

    ax = polygons.plot(
        edgecolor=edgecolor, cmap=cmap, vmin=0, ax=ax, legend=legend,
        column=data_col, color=color)

    if textbox is True:
        bb = dict(boxstyle="round", alpha=.5, ec=(1, 1, 1), fc=(1, 1, 1))
    elif isinstance(textbox, dict):
        bb = textbox
    else:
        bb = None

    if label_col is not None:
        polygons.apply(
            lambda x: ax.text(
                x.geometry.representative_point().x,
                x.geometry.representative_point().y,
                x[label_col], size=9,
                ha="center", va="center", bbox=bb),
            axis=1)

    return ax


def plot_bus(es, node_label, rm_list=None):
    """benutzt ??? Siehe plot_bus_view()"""

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

    Examples
    --------
    >>> es = results.get_test_es()
    >>> bus = results.get_nodes_by_label(
    ...     es, label_args=['bus', 'electricity', None, 'DE01'])
    >>> p = plot_bus_view(es, bus=bus, legend=True, xlabel='',
    ...         ylabel='Leistung [MW]', title='title', smooth=True)
    >>> plt.show()
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
        data = data.loc[period[0]:period[1]]

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


def plot_multiregion_io(es):
    """
    Ansatzweise redundant mit plot_bus_view().


    Parameters
    ----------
    es : solph.EnergySystem
        An EnergySystem with results.

    Returns
    -------

    Examples
    --------
    >>> my_es = results.get_test_es()
    >>> df = plot_multiregion_io(my_es)
    >>> plt.show()
    """
    multi_reg_res = results.get_multiregion_bus_balance(es).groupby(
        level=[1, 2], axis=1).sum()

    multi_reg_res[('out', 'losses')] = (multi_reg_res[('out', 'export')] -
                                        multi_reg_res[('in', 'import')])
    del multi_reg_res[('out', 'export')]
    del multi_reg_res[('in', 'import')]

    my_cdict = get_cdict_df(multi_reg_res['in'])
    my_cdict.update(get_cdict_df(multi_reg_res['out']))

    oev.plot.io_plot(df_in=multi_reg_res['in'],
                     df_out=multi_reg_res['out'],
                     smooth=True, cdict=my_cdict)
    return multi_reg_res


def show_region_values_gui(es):
    data = results.get_multiregion_bus_balance(es).groupby(
        level=[1, 2, 3, 4], axis=1).sum()
    data = data.agg(['sum', 'min', 'max', 'mean'])
    data = data.reorder_levels([1, 2, 0], axis=1)

    data.sort_index(1, inplace=True)

    key1 = gui.get_choice(data.columns.get_level_values(0).unique(),
                          "Plot transmission lines", "Choose data column.")
    key2 = gui.get_choice(data[key1].columns.get_level_values(0).unique(),
                          "Plot transmission lines", "Choose data column.")
    key3 = gui.get_choice(data.index.unique(),
                          "Plot transmission lines", "Choose data column.")

    plot_data = pd.DataFrame(data.loc[key3, (key1, key2)], columns=[key3])

    plot_data = plot_data.div(1000).round().astype(int)

    plot_regions(plot_data, key3)


def analyse_plot(cost_values, merit_values, multiregion, demand_elec,
                 demand_distr):
    orderkeys = ['nuclear', 'lignite', 'hard_coal', 'natural_gas', 'other',
                 'bioenergy', 'oil', 'natural_gas_add', 'phe_storage',
                 'shortage']

    in_list = get_orderlist_from_multiindex(multiregion['in'].columns,
                                            orderkeys)
    my_cdict = get_cdict_df(multiregion['in'])
    my_cdict.update(get_cdict_df(multiregion['out']))
    # print(my_cdict)

    oplot = oev.plot.io_plot(df_in=multiregion['in'],
                             df_out=multiregion['out'],
                             smooth=True, inorder=in_list, cdict=my_cdict)

    # ax = demand.plot()
    ax = demand_elec.reset_index(drop=True).plot(
        ax=oplot['ax'], color='g', legend=True)

    # ax = demand.plot()
    ax = demand_distr.reset_index(drop=True).plot(
        ax=ax, color='m', legend=True)

    # df.to_excel('/home/uwe/lcoe.xls')

    # my_cols = pd.MultiIndex(levels=[[], [], []], labels=[[], [], []],
    #                         names=[u'type', u'src', u'region'])
    # ana_df = pd.DataFrame(columns=my_cols)

    # for c in cost_values.columns:
    #     ana_df[str(c)] = cost_values[c]
    # ana_df = ana_df.groupby(level=[0, 1], axis=1).sum()

    # ana_df[('chp', 'chp')] = ana_df['chp'].sum(axis=1)

    # ana_x = ana_df.reset_index(drop=True).plot()
    # ana_x .set_xlim(3000, 3200)

    cost_values.reset_index(drop=True, inplace=True)
    cost_values.name = 'cost_value (right)'
    ax2 = cost_values.sum(axis=1).plot(
        ax=ax, secondary_y=True, legend=True, color='#7cfff0')

    merit_values.reset_index(drop=True, inplace=True)
    merit_values.name = 'merit_value (right)'
    merit_values.max(axis=1).plot(
        ax=ax2, secondary_y=True, legend=True, color='#4ef3bc')

    # ax2.set_ylim(0, 20)
    # ax2.set_xlim(3000, 3200)
    plt.show()
