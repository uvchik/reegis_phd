import berlin_hp
import os
import logging
import deflex
import reegis_tools.geometries
import pprint as pp
from oemof import graph
from oemof.tools import logger
from oemof import solph as solph
from datetime import datetime
from oemof import outputlib
from my_reegis import analyzer
from matplotlib import pyplot as plt
import oemof_visio as oev
import pandas as pd
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import math
from matplotlib.colors import LinearSegmentedColormap
import reegis_tools.config as cfg
import reegis_tools.gui as gui
from my_reegis import plot
from my_reegis import friedrichshagen_scenarios as fsc
import my_reegis.plot as my_plot
from collections import namedtuple


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


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def plot_power_lines(data, key, cmap_lines=None, cmap_bg=None,
                     vmax=None, label_max=None, unit='GWh', size=None):
    if size is None:
        ax = plt.figure(figsize=(5, 5)).add_subplot(1, 1, 1)
    else:
        ax = plt.figure(figsize=size).add_subplot(1, 1, 1)

    lines = reegis_tools.geometries.Geometry()
    lines.load(cfg.get('paths', 'geometry'),
               cfg.get('geometry', 'de21_power_lines'))
    polygons = reegis_tools.geometries.Geometry()
    polygons.load(cfg.get('paths', 'geometry'),
                  cfg.get('geometry', 'de21_polygons_simple'))

    lines.gdf = lines.gdf.merge(data, left_index=True, right_index=True)

    lines.gdf['centroid'] = lines.gdf.centroid

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

    for i, p in polygons.gdf.iterrows():
        if 'see' in p['name'].lower():
            polygons.gdf.loc[i, 'color'] = 1
        else:
            polygons.gdf.loc[i, 'color'] = 0

    lines.gdf['reverse'] = lines.gdf[key] < 0
    lines.gdf.loc[lines.gdf['reverse'], key] = (
        lines.gdf.loc[lines.gdf['reverse'], key] * -1)

    if vmax is None:
        vmax = lines.gdf[key].max()

    if label_max is None:
        label_max = vmax * 0.5

    ax = polygons.gdf.plot(edgecolor='#9aa1a9', cmap=cmap_bg,
                           column='color', ax=ax)
    ax = lines.gdf.plot(cmap=cmap_lines, legend=True, ax=ax, column=key,
                        vmin=0, vmax=vmax)
    for i, v in lines.gdf.iterrows():
        x1 = v['geometry'].coords[0][0]
        y1 = v['geometry'].coords[0][1]
        x2 = v['geometry'].coords[1][0]
        y2 = v['geometry'].coords[1][1]

        value = v[key] / vmax
        mc = cmap_lines(value)

        orient = math.atan(abs(x1-x2)/abs(y1-y2))

        if (y1 > y2) & (x1 > x2):
            orient *= -1

        if v['reverse']:
            orient += math.pi

        if round(v[key]) == 0:
            pass
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

        if v[key] > label_max:
            ax.text(
                v['centroid'].x, v['centroid'].y,
                '{0} {1}'.format(round(v[key]), unit),
                color='#000000',
                fontsize=9.5,
                zorder=15,
                path_effects=[
                    path_effects.withStroke(linewidth=3, foreground="w")])

    polygons.gdf.apply(lambda x: ax.annotate(
        s=x.name, xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

    plt.title(key)

    # plt.show()


def compare_transmission(es1, es2, name1='es1', name2='es2'):

    results1 = es1.results['Main']
    results2 = es2.results['Main']

    out_flow_lines = [x for x in results1.keys() if 'line' in x[0].label.cat]

    # Calculate results
    transmission = pd.DataFrame()
    for out_flow in out_flow_lines:
        # es1
        line_a_1 = out_flow[0]
        bus_reg1_1 = [x for x in line_a_1.inputs][0]
        bus_reg2_1 = out_flow[1]
        line_b_1 = es1.groups['_'.join(['line', 'electricity',
                                        bus_reg2_1.label.region,
                                        bus_reg1_1.label.region])]

        # es2
        bus_reg1_2 = es2.groups[str(bus_reg1_1.label)]
        bus_reg2_2 = es2.groups[str(bus_reg2_1.label)]
        line_a_2 = es2.groups[str(line_a_1.label)]
        line_b_2 = es2.groups[str(line_b_1.label)]

        from1to2_1 = results1[(line_a_1, bus_reg2_1)]['sequences'].sum()
        from2to1_1 = results1[(line_b_1, bus_reg1_1)]['sequences'].sum()
        from1to2_2 = results2[(line_a_2, bus_reg2_2)]['sequences'].sum()
        from2to1_2 = results2[(line_b_2, bus_reg1_2)]['sequences'].sum()

        line_name = '-'.join([line_a_1.label.subtag, line_a_1.label.region])

        transmission.loc[line_name, name1] = float(from1to2_1 - from2to1_1)
        transmission.loc[line_name, name2] = float(from1to2_2 - from2to1_2)

    # transmission['diff_1-2'] = transmission[name1] - transmission[name2]

    transmission = transmission.div(1000)

    transmission['diff_2-1'] = transmission[name2] - transmission[name1]
    transmission['fraction'] = (transmission['diff_2-1'] /
                                abs(transmission[name1]) * 100)
    transmission['fraction'].fillna(0, inplace=True)

    return transmission


def plot_regions(data, column):
    polygons = reegis_tools.geometries.Geometry()
    polygons.load(cfg.get('paths', 'geometry'),
                  cfg.get('geometry', 'de21_polygons_simple'))
    polygons.gdf = polygons.gdf.merge(data, left_index=True, right_index=True)

    print(polygons.gdf)

    cmap = LinearSegmentedColormap.from_list(
            'mycmap', [
                # (0, '#aaaaaa'),
                (0.000000000, 'green'),
                (0.5, 'yellow'),
                (1, 'red')])

    ax = polygons.gdf.plot(edgecolor='#9aa1a9', cmap=cmap, vmin=0,
                           column=column, legend=True)

    polygons.gdf.apply(lambda x: ax.annotate(
        s=x[column], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

    plt.show()


def reshape_bus_view(es, bus, data=None, aggregate_lines=True):

    if data is None:
        m_cols = pd.MultiIndex(levels=[[], [], [], [], []],
                               labels=[[], [], [], [], []])
        data = pd.DataFrame(columns=m_cols)

    # filter all nodes and sub-list import/exports
    node_flows = [x for x in es.results['Main'].keys()
                  if x[1] == bus or x[0] == bus]

    if len([x for x in node_flows if (x[1].label.cat == 'line') or
                                     (x[0].label.cat == 'line')]) == 0:
        aggregate_lines = False

    # If True all powerlines will be aggregated to import/export
    if aggregate_lines is True:
        export_flows = [x for x in node_flows if x[1].label.cat == 'line']
        import_flows = [x for x in node_flows if x[0].label.cat == 'line']

        # only export lines
        export_label = (bus.label.region, 'out', 'export', 'electricity',
                        'all')

        data[export_label] = (
                es.results['Main'][export_flows[0]]['sequences']['flow'] * 0)
        for export_flow in export_flows:
            data[export_label] += (
                es.results['Main'][export_flow]['sequences']['flow'])
            node_flows.remove(export_flow)

        # only import lines
        import_label = (bus.label.region, 'in', 'import', 'electricity', 'all')
        data[import_label] = (
                es.results['Main'][import_flows[0]]['sequences']['flow'] * 0)
        for import_flow in import_flows:
            data[import_label] += (
                es.results['Main'][import_flow]['sequences']['flow'])
            node_flows.remove(import_flow)

    # all flows without lines (import/export)
    for flow in node_flows:
        if flow[0] == bus:
            flow_label = (bus.label.region, 'out', flow[1].label.cat,
                          flow[1].label.tag, flow[1].label.subtag)
        elif flow[1] == bus:
            flow_label = (bus.label.region, 'in', flow[0].label.cat,
                          flow[0].label.tag, flow[0].label.subtag)
        else:
            flow_label = None

        data[flow_label] = es.results['Main'][flow]['sequences']['flow']

    return data.sort_index(axis=1)


def get_multiregion_bus_balance(es, bus_substring=None):
    """

    Parameters
    ----------
    es : solph.EnergySystem
        An EnergySystem with results.
    bus_substring : str or tuple
        A sub-string or sub-tuple to find a list of buses.

    Returns
    -------
    pd.DataFrame : Multiregional results.

    """
    # regions = [x for x in es.results['tags']['region']
    #            if re.match(r"DE[0-9][0-9]", x)]

    if bus_substring is None:
        bus_substring = 'bus_electricity_all'

    buses = set([x[0] for x in es.results['Main'].keys()
                 if str(bus_substring) in str(x[0].label)])

    data = None
    for bus in sorted(buses):
        data = reshape_bus_view(es, bus, data)

    return data


def get_nominal_values(es):
    de_results = es.results['Param']

    regions = [x[0].label.region for x in es.results['Param'].keys()
               if ('shortage' in x[0].label.cat) &
               ('electricity' in x[0].label.tag)]

    midx = pd.MultiIndex(levels=[[], [], [], []], labels=[[], [], [], []])
    dt = pd.DataFrame(index=midx, columns=['nominal_value'])
    for region in sorted(set(regions)):
        node = 'bus_electricity_all_{0}'.format(region)
        node = es.groups[node]
        in_c = [x for x in de_results.keys() if x[1] == node]
        out_c = [x for x in de_results.keys()
                 if x[0] == node and x[1] is not None]
        for k in in_c:
            # print(repr(k.label))
            # idx = region, 'in', k[0].replace('_{0}'.format(region), '')
            dt.loc[k[0].label] = de_results[k]['scalars'].get('nominal_value')

        for k in out_c:
            dt.loc[k[1].label] = de_results[k]['scalars'].get('nominal_value')

    return dt


def plot_multiregion_io(es):
    multi_reg_res = get_multiregion_bus_balance(es).groupby(
        level=[1, 2, 3, 4], axis=1).sum()

    multi_reg_res[('out', 'losses')] = (multi_reg_res[('out', 'export')] -
                                        multi_reg_res[('in', 'import')])
    del multi_reg_res[('out', 'export')]
    del multi_reg_res[('in', 'import')]

    in_list = ['trsf_pp_nuclear', 'trsf_pp_lignite', 'source_geothermal',
               'source_hydro', 'source_solar', 'source_wind',
               'trsf_pp_hard_coal', 'trsf_pp_natural_gas', 'trsf_pp_other',
               'trsf_pp_bioenergy', 'trsf_pp_oil', 'trsf_pp_natural_gas_add',
               'trsf_chp_other', 'trsf_chp_lignite', 'trsf_chp_natural_gas',
               'trsf_chp_oil', 'trsf_chp_bioenergy', 'trsf_chp_hard_coal',
               'phe_storage', 'import', 'shortage_bus_elec', ]

    my_cdict = plot.get_cdict_df(multi_reg_res['in'])
    my_cdict.update(plot.get_cdict_df(multi_reg_res['out']))
    # print(my_cdict)

    oev.plot.io_plot(df_in=multi_reg_res['in'],
                     df_out=multi_reg_res['out'],
                     smooth=True, inorder=in_list, cdict=my_cdict)

    return multi_reg_res


def show_region_values_gui(es):
    data = get_multiregion_bus_balance(es).groupby(
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


def load_param_results():
    path = os.path.join(
        cfg.get('paths', 'scenario'), 'berlin_basic', str(2014))
    file = gui.select_filename(work_dir=path,
                               title='Berlin results!',
                               extension='esys')
    sc = berlin_hp.Scenario()
    sc.restore_es(os.path.join(path, file))
    return outputlib.processing.convert_keys_to_strings(sc.es.results['param'])
    # return sc.es.results['param']


def create_tags(nodes):
    """
    sc.es.results['tags'] = create_tags(sc.es.nodes)
    """
    tags = None
    fields = None
    for node in nodes:
        if tags is None:
            fields = node.label.__getattribute__('_fields')
            tags = {f: [] for f in fields}
        for field, value in node.label.__getattribute__('_asdict')().items():
            tags[field].append(value)
    for field in fields:
        tags[field] = set(tags[field])
    return tags


def write_graph(es):
    graph.create_nx_graph(es, filename='/home/uwe/mygraph',
                          remove_nodes_with_substrings=['commodity'])


def load_es(*args, var=None):
    path = os.path.join(cfg.get('paths', 'scenario'), *args, 'results')

    if var is None:
        var = []
    else:
        var = [var]

    name = '_'.join(list(args) + var)

    fn = os.path.join(path, name + '.esys')

    sc = deflex.Scenario()
    logging.debug("Restoring file from {0}".format(fn))
    file_datetime = datetime.fromtimestamp(os.path.getmtime(fn)).strftime(
        '%d. %B %Y - %H:%M:%S')
    logging.info("Restored results created on: {0}".format(file_datetime))
    sc.restore_es(fn)
    sc.es.name = name
    return sc.es


def check_excess_shortage(results):
    ex_nodes = [x for x in results.keys() if 'excess' in x[1]]
    sh_nodes = [x for x in results.keys() if 'shortage' in x[0]]
    for node in ex_nodes:
        f = outputlib.views.node(results, node[1])
        s = int(round(f['sequences'].sum()))
        if s > 0:
            print(node[1], ':', s)

    for node in sh_nodes:
        f = outputlib.views.node(results, node[0])
        s = int(round(f['sequences'].sum()))
        if s > 0:
            print(node[0], ':', s)


def find_input_flow(out_flow, nodes):
    return [x for x in list(nodes.keys()) if x[1] == out_flow[0][0]]


def get_full_load_hours(results):
    bus_label = 'bus_elec_BE'
    params = load_param_results()
    my_node = outputlib.views.node(results, bus_label)['sequences']
    sums = my_node.sum()
    max_vals = my_node.max()

    flh = pd.DataFrame()

    input_flows = [c for c in my_node.columns if bus_label == c[0][1]]
    for col in input_flows:
        inflow = [x for x in list(params.keys()) if x[1] == col[0][0]]
        node_label = col[0][0]
        if 'nominal_value' in params[(node_label, bus_label)]['scalars']:
            flh.loc[node_label, 'nominal_value'] = (
                params[(col[0][0], bus_label)]['scalars']['nominal_value'])
        else:
            if len(inflow) > 0:
                inflow = inflow[0]
                if 'nominal_value' in params[inflow]['scalars']:
                    try:
                        cf = (
                            params[(inflow[1], 'None')]['scalars']
                            ['conversion_factor_full_condensation_{0}'.format(
                                bus_label)])
                    except KeyError:
                        cf = params[(inflow[1], 'None')]['scalars'][
                            'conversion_factors_{0}'.format(bus_label)]

                    flh.loc[node_label, 'nominal_value'] = (
                            params[inflow]['scalars']['nominal_value'] * cf)
                else:
                    flh.loc[node_label, 'nominal_value'] = 0
            else:
                flh.loc[node_label, 'nominal_value'] = 0
        if len(inflow) > 0:
            if isinstance(inflow, list):
                inflow = inflow[0]

            flh.loc[node_label, 'average_efficiency'] = (
                sums[col] / results[inflow]['sequences']['flow'].sum())

        flh.loc[node_label, 'energy'] = sums[col]
        flh.loc[node_label, 'max'] = max_vals[col]
        if flh.loc[node_label, 'nominal_value'] > 0:
            flh.loc[node_label, 'full_load_hours'] = (
                    sums[col] / flh.loc[node_label, 'nominal_value'])
            flh.loc[node_label, 'average_power'] = sums[col] / 8760
    flh['check'] = flh['max'] > flh['nominal_value']
    flh['full_load_hours'].plot(kind='bar')

    print(flh)
    plt.show()


def plot_bus_view(es, bus=None, ax=None):
    """

    Parameters
    ----------
    es
    bus : str or Node or None
    ax

    Returns
    -------

    """

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
    if isinstance(bus, str):
        bus = es.groups[bus]

    if bus is None:
        data = get_multiregion_bus_balance(es).groupby(
            axis=1, level=[1, 2, 3, 4]).sum()
        title = 'Germany'
    else:
        data = reshape_bus_view(es, bus)[bus.label.region]
        title = repr(bus.label)

    my_colors = plot.get_cdict(data['in'])
    my_colors.update(plot.get_cdict(data['out']))

    my_plot = oev.plot.io_plot(
        df_in=data['in'], df_out=data['out'], cdict=my_colors,
        inorder=plot.get_orderlist_from_multiindex(data['in'].columns),
        outorder=plot.get_orderlist_from_multiindex(data['out'].columns),
        ax=ax, smooth=True)
    ax = shape_tuple_legend(**my_plot)
    ax = oev.plot.set_datetime_ticks(ax, data.index, tick_distance=48,
                                     date_format='%d-%m-%H', offset=12,
                                     tight=True)
    ax.set_ylabel('Power in MW')
    ax.set_xlabel('Date')
    ax.set_title(title)
    plt.show()


def plot_bus(es, node_label, rm_list=None):

    node = outputlib.views.node(es.results['Main'], node_label)

    fig = plt.figure(figsize=(10, 5))

    my_node = node['sequences']

    if rm_list is None:
        rm_list = []

    plot_slice = oev.plot.slice_df(my_node,)
                                   # date_from=datetime(2014, 5, 31),
                                   # date_to=datetime(2014, 6, 8))

    my_plot = oev.plot.io_plot(node_label, plot_slice,
                               cdict=plot.get_cdict(my_node),
                               inorder=plot.get_orderlist(my_node, 'in'),
                               outorder=plot.get_orderlist(my_node, 'out'),
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


def ee_analyser(results, ee_type):
    analysis = analyzer.Analysis(
        results['Main'], results['Param'],
        iterator=analyzer.FlowNodeIterator)
    ee = analyzer.FlowFilterSubstring(ee_type, position=0)
    analysis.add_analyzer(ee)
    analysis.analyze()

    df = pd.concat(ee.result, axis=1)
    df.columns = df.columns.droplevel(-1)
    return df.sum(axis=1)


def emissions(es):
    r = es.results['Main']
    p = es.results['Param']

    emission_df = pd.DataFrame()

    for i in r.keys():
        if (i[0].label.cat == 'source') & (i[0].label.tag == 'commodity'):
            emission_df.loc[i[0].label.subtag, 'specific_emission'] = (
                p[i]['scalars']['emission'])
            emission_df.loc[i[0].label.subtag, 'summed_flow'] = (
                r[i]['sequences']['flow'].sum())

    emission_df['total_emission'] = (emission_df['specific_emission'] *
                                     emission_df['summed_flow'])

    emission_df.sort_index(inplace=True)

    return emission_df


def fullloadhours(es, grouplevel=None):
    if grouplevel is None:
        grouplevel = [0, 1, 2, 3, 4]

    r = es.results['Main']
    p = es.results['Param']

    idx = pd.MultiIndex(levels=[[], [], [], [], []],
                        labels=[[], [], [], [], []])
    cols = ['nominal_value', 'summed_flow']
    sort_results = pd.DataFrame(index=idx, columns=cols)
    logging.info('Start')
    for i in r.keys():
        if isinstance(i[1], solph.Transformer):
            out_node = [o for o in i[1].outputs][0]
            cf_name = 'conversion_factors_' + str(out_node.label)
            try:
                nom_value = (
                    p[(i[1], out_node)]['scalars']['nominal_value'] /
                    p[(i[1], None)]['scalars'][cf_name])
                label = tuple(i[1].label) + ('nvo', )
            except KeyError:
                try:
                    nom_value = p[i]['scalars']['nominal_value']
                    label = tuple(i[1].label) + ('nvi', )
                except KeyError:
                    nom_value = r[i]['sequences']['flow'].max()
                    label = tuple(i[1].label) + ('max', )

            summed_flow = r[i]['sequences']['flow'].sum()

        elif isinstance(i[0], solph.Source):
            try:
                nom_value = p[i]['scalars']['nominal_value']
                label = tuple(i[0].label) + ('nvo', )
            except KeyError:
                nom_value = r[i]['sequences']['flow'].max()
                label = tuple(i[0].label) + ('max', )

            summed_flow = r[i]['sequences']['flow'].sum()

        else:
            label = None
            nom_value = None
            summed_flow = None

        if nom_value is not None:
            sort_results.loc[label, 'nominal_value'] = nom_value
            if not sort_results.index.is_lexsorted():
                sort_results.sort_index(inplace=True)
            sort_results.loc[label, 'summed_flow'] = summed_flow

    logging.info('End')
    new = sort_results.groupby(level=grouplevel).sum()
    new['flh'] = new['summed_flow'] / new['nominal_value']
    return new


def results_iterator(es, demand):

    back = namedtuple('analysis', ['cost', 'emission', 'merit_order', 'misc',
                                   'emission_specific'])

    r = es.results['Main']
    p = es.results['Param']

    df_sort_res = pd.DataFrame()
    dict_analyse = {'cost': {}, 'emission': {}, 'merit_order': {},
                    'emission_specific': {}}
    up_dict = {}

    for i in r.keys():
        df_sort_res = sorted_results(i, df_sort_res, r)
        dict_analyse = analyze(i, dict_analyse, up_dict,
                               demand, r, p)

    df_sort_res.sort_index(axis=1, inplace=True)
    df_cost_analyse = pd.DataFrame.from_dict(dict_analyse['cost'])
    df_cost_analyse.reset_index(drop=True, inplace=True)
    df_emission_analyse = pd.DataFrame.from_dict(dict_analyse['emission'])
    df_emission_analyse.reset_index(drop=True, inplace=True)
    df_merit_analyse = pd.DataFrame.from_dict(dict_analyse['merit_order'])
    df_merit_analyse.reset_index(drop=True, inplace=True)
    df_emission_total_analyse = pd.DataFrame.from_dict(
        dict_analyse['emission_specific'])
    df_emission_total_analyse.reset_index(drop=True, inplace=True)
    return back(df_cost_analyse, df_emission_analyse, df_merit_analyse,
                df_sort_res, df_emission_total_analyse)


def analyze(args, result, up_dict, demand, r, p):
    flow = None
    conversion_factor = None
    label = args

    if (isinstance(args[1], solph.Transformer) and
            args[0].label.tag != 'electricity'):
        # eta = {}
        label = args[1].label
        if len(args[1].outputs) == 2:
            pass
            # for o in args[1].outputs:
            #     key = 'conversion_factors_{0}'.format(o)
            #     eta_key = o.label.split('_')[-2]
            #     eta_val = self.psc((args[1], None))[key]
            #     eta[eta_key] = eta_val
            # eta['heat_ref'] = 0.9
            # eta['elec_ref'] = 0.55
            #
            # pee = (1 / (eta['heat'] / eta['heat_ref'] + eta['elec'] /
            #             eta['elec_ref'])) * (eta['elec'] / eta['elec_ref'])
            #
            # cost_flow = self.rsq(args) * pee

        elif len(args[1].outputs) == 1:
            t_out = [o for o in args[1].outputs][0].label.tag
            t_in = [i for i in args[1].inputs][0].label.tag
            if t_out == 'electricity' and t_in != 'electricity':
                flow = r[args]['sequences']
                for k, v in p[(args[1], None)]['scalars'].items():
                    if 'conversion_factors' in k and 'electricity' in k:
                        conversion_factor = v
            else:
                flow = None
                conversion_factor = None

        else:
            print(args[1].label, len(args[1].outputs))

        if args[0] not in up_dict and args[1].label.cat != 'line':
            up_dict[args[0]] = {}
            var_costs = 0
            emission = 0
            flow_seq = 0
            for i in args[0].inputs:
                if i.label.cat != 'shortage':
                    var_costs += (
                            p[(i, args[0])]['scalars']['variable_costs'] *
                            r[(i, args[0])]['sequences'].sum())
                    emission += (
                            p[(i, args[0])]['scalars']['emission'] *
                            r[(i, args[0])]['sequences'].sum())
                    flow_seq += r[(i, args[0])]['sequences'].sum()

            if float(flow_seq) == 0:
                flow_seq = 1
            up_dict[args[0]]['cost'] = var_costs / flow_seq
            up_dict[args[0]]['emission'] = emission / flow_seq

    elif 'shortage' == args[0].label.cat and args[1] is not None:
        up_dict[args[0]] = {}
        label = args[0].label
        # self.costs[args[0]] = self.psc(args)['variable_costs']
        up_dict[args[0]]['cost'] = 50
        up_dict[args[0]]['emission'] = 500
        flow = r[args]['sequences']
        conversion_factor = 0.1

    if flow is not None:
        for value in ['cost', 'emission']:
            result[value][label] = flow * up_dict[args[0]][value]
            result[value][label] = result[value][label]['flow'].div(demand)
        result['merit_order'][label] = flow.div(flow).multiply(
            up_dict[args[0]]['cost']).div(conversion_factor).fillna(0)['flow']
        result['emission_specific'][label] = flow.div(flow).multiply(
            up_dict[args[0]]['emission']).div(
            conversion_factor).fillna(0)['flow']
    return result


def sorted_results(i, sort_res, r):
    # find source flows
    if isinstance(i[0], solph.Source):
        column_name = i[0].label.cat + '_' + i[0].label.tag
        try:
            sort_res[column_name] += r[i]['sequences']['flow']
        except KeyError:
            sort_res[column_name] = r[i]['sequences']['flow']

    # find storage discharge
    if isinstance(i[0], solph.components.GenericStorage):
        if i[1] is not None:
            column_name = i[0].label.subtag + '_discharge'
            try:
                sort_res[column_name] += r[i]['sequences']['flow']
            except KeyError:
                sort_res[column_name] = r[i]['sequences']['flow']

    # find elec_demand
    if isinstance(i[1], solph.Sink):
        column_name = i[1].label.cat + '_' + i[1].label.tag
        try:
            sort_res[column_name] += r[i]['sequences']['flow']
        except KeyError:
            sort_res[column_name] = r[i]['sequences']['flow']

    return sort_res


def analyse_plot(cost_values, merit_values, multiregion, demand_elec,
                 demand_distr):
    orderkeys = ['nuclear', 'lignite', 'hard_coal', 'natural_gas', 'other',
                 'bioenergy', 'oil', 'natural_gas_add', 'phe_storage',
                 'shortage']

    in_list = plot.get_orderlist_from_multiindex(multiregion['in'].columns,
                                            orderkeys)
    my_cdict = plot.get_cdict_df(multiregion['in'])
    my_cdict.update(plot.get_cdict_df(multiregion['out']))
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


def reshape_multiregion_df(es):
    """Remove"""
    res = get_multiregion_bus_balance(es).groupby(
        axis=1, level=[1, 2, 3, 4]).sum()

    res[('out', 'demand', 'electricity', 'all')] += (
            res[('out', 'export', 'electricity', 'all')] -
            res[('in', 'import', 'electricity', 'all')] +
            res[('out', 'storage', 'electricity', 'phes')] -
            res[('in', 'storage', 'electricity', 'phes')])
    del res[('out', 'export', 'electricity', 'all')]
    del res[('in', 'import', 'electricity', 'all')]
    del res[('out', 'storage', 'electricity', 'phes')]
    del res[('in', 'storage', 'electricity', 'phes')]

    for ee_src, values in res[('in', 'source', 'ee')].iteritems():
        res[('out', 'demand', 'electricity', 'all')] -= values
        del res[('in', 'source', 'ee', ee_src)]

    for fuel, values in res[('in', 'trsf', 'chp')].iteritems():
        res[('out', 'demand', 'electricity', 'all')] -= values
        del res[('in', 'trsf', 'chp', fuel)]

    for c in res.columns:
        if res[c].sum() < 0.0001:
            del res[c]
    return res


def get_one_node_type(es, node=None):
    mr_df = get_multiregion_bus_balance(es).groupby(
            axis=1, level=[1, 2, 3, 4]).sum()

    nom_values = get_nominal_values(es)

    if node is None:
        items = sorted(mr_df.columns.get_level_values(2).unique())
        node = gui.get_choice(items, 'Node selection')

    smry = pd.DataFrame()
    smry['sum'] = mr_df.loc[pd.IndexSlice[:], pd.IndexSlice[:, :, node]].sum()
    smry['max'] = mr_df.loc[pd.IndexSlice[:], pd.IndexSlice[:, :, node]].max()
    smry['mlh'] = smry['sum'].div(smry['max'])
    smry['mlf'] = smry['sum'].div(smry['max']).div(len(mr_df)).multiply(100)

    smry['nominal_value'] = (
        nom_values.loc[(pd.IndexSlice[:, :, node]), 'nominal_value'])
    smry['flh'] = smry['sum'].div(smry['nominal_value'])
    smry['flf'] = (
        smry['sum'].div(smry['nominal_value']).div(len(mr_df)).multiply(100))

    return smry


def analyse_system_costs(es, plot=False):
    back = namedtuple('res', ['levelized', 'meritorder', 'emission',
                              'emission_last'])
    multi_res = reshape_multiregion_df(es)

    demand = multi_res['out', 'demand', 'electricity', 'all']

    iter_res = results_iterator(es, demand)
    c_values = iter_res.cost
    e_values = iter_res.emission
    mo_values = iter_res.merit_order
    e_spec = iter_res.emission_specific

    ax = mo_values.max(axis=1).plot()

    emission_last = pd.Series(index=mo_values.index)

    for index, column in mo_values.idxmax(axis=1).iteritems():
        emission_last.at[index] = e_spec.loc[index, column]

    ax = emission_last.plot(ax=ax)
    e_spec.max(axis=1).plot(ax=ax)

    if plot is True:
        sorted_flows = iter_res.misc
        analyse_plot(c_values, mo_values, multi_res, sorted_flows[
            'demand_electricity'], sorted_flows['demand_heat'])

    return back(c_values.sum(axis=1), mo_values.max(axis=1),
                e_values.sum(axis=1), emission_last)


def something():
    sres = {'avg_costs': pd.DataFrame(),
            'absolute_costs': pd.DataFrame(),
            'absolute_flows': pd.DataFrame(),
            'emissions': pd.DataFrame(),
            'absolute_emissions': pd.DataFrame(),
            'upstream_emissions': pd.DataFrame(),
            'meta': pd.DataFrame()}
    upstream_data = fsc.load_upstream_scenario_values()
    upstream_over = fsc.load_upstream_overall_values()

    number = 1
    start_dir = os.path.join(cfg.get('paths', 'scenario'), 'friedrichshagen')
    for root, directories, filenames in os.walk(start_dir):
        for fn in filenames:
            if fn[-5:] == '.esys':
                upstream = '_'.join(fn.split('_')[3:-4])
                cost_type = fn.split('_')[2]
                region = '_'.join(fn.split('_')[-4:-2])
                fuel = '_'.join(fn.split('_')[-2:]).replace('.esys', '')
                if ('berlin_hp' not in upstream and
                        'levelized' not in cost_type):
                    logging.info("{0} - {1}".format(number, fn))
                    # try:
                    tmp_es = solph.EnergySystem()
                    tmp_es.restore(root, fn)

                    ud = upstream_data.get(upstream, pd.DataFrame())
                    if 'single' not in fuel:
                        sres = something_b(tmp_es, sres, number, ud)
                    sres['meta'].loc[number, 'upstream'] = upstream
                    sres['meta'].loc[number, 'region'] = region
                    sres['meta'].loc[number, 'fuel'] = fuel
                    sres['meta'].loc[number, 'cost_type'] = cost_type
                    for i in upstream_over.get(upstream, pd.Series()).index:
                        sres['upstream_emissions'].loc[number, i] = (
                            upstream_over[upstream, i])
                    number += 1

    return sres


def something_b(es, sres, number, upstream_data):
    ebus = es.groups['bus_electricity_all_FHG']
    export_node = es.groups['export_electricity_all_FHG']
    import_node = es.groups['import_electricity_all_FHG']

    reg_emissions = emissions(es).loc[emissions(es)['summed_flow'] > 0]

    for col in ['summed_flow', 'total_emission']:
        for fuel in reg_emissions.index:
            sres['emissions'].loc[number, col + '_' + fuel] = (
                reg_emissions.loc[fuel, col])
        sres['emissions'].loc[number, col + '_all'] = reg_emissions[col].sum()

    export_costs = pd.Series(es.flows()[(ebus, export_node)].variable_costs
                             ).div(0.99)
    import_costs = pd.Series(es.flows()[(import_node, ebus)].variable_costs
                             ).div(1.01)

    avg_cost_all = pd.Series(es.flows()[(import_node, ebus)].variable_costs
                             ).mean()

    export_flow = es.results['Main'][(ebus, export_node)]['sequences']
    import_flow = es.results['Main'][(import_node, ebus)]['sequences']

    export_flow.reset_index(drop=True, inplace=True)
    import_flow.reset_index(drop=True, inplace=True)

    upstream_emissions = upstream_data.get('emission', 0)
    total_export_emissions = export_flow.multiply(upstream_emissions, axis=0)
    total_import_emissions = import_flow.multiply(upstream_emissions, axis=0)
    sres['absolute_emissions'].loc[number, 'import emissions'] = float(
        total_import_emissions.sum())
    sres['absolute_emissions'].loc[number, 'export emissions'] = float(
        total_export_emissions.sum())

    sres['absolute_flows'].loc[number, 'import flow'] = float(
        import_flow.sum())
    sres['absolute_flows'].loc[number, 'export flow'] = float(
        export_flow.sum())

    total_export_costs = export_flow.multiply(export_costs, axis=0)
    total_import_costs = import_flow.multiply(import_costs, axis=0)
    sres['absolute_costs'].loc[number, 'import costs'] = float(
        total_import_costs.sum())
    sres['absolute_costs'].loc[number, 'export costs'] = -1 * float(
        total_export_costs.sum())

    sres['avg_costs'].loc[number, 'import avg costs'] = float(
        total_import_costs.sum() / import_flow.sum())
    sres['avg_costs'].loc[number, 'export avg costs'] = float(
            total_export_costs.sum() / export_flow.sum() * -1)
    sres['avg_costs'].loc[number, 'avg costs'] = float(avg_cost_all)
    return sres


def get_optional_export_emissions(es, label, bus_balnc):
    heat_blnc = get_multiregion_bus_balance(es, 'bus_heat')
    heat_demand_distr = heat_blnc[('vattenfall_friedrichshagen', 'out',
                                   'demand', 'heat', 'district')]
    heat_supply_hp = heat_blnc[('vattenfall_friedrichshagen', 'in',
                                'hp', 'heat', 'natural_gas')]
    heat_demand_distr -= heat_supply_hp
    export = bus_balnc[('FHG', 'out', 'export', 'electricity', 'all')]
    import_elec = bus_balnc[('FHG', 'in', 'import', 'electricity', 'all')]
    mustrun = pd.DataFrame()
    parameter = {}
    for node in es.results['Main'].keys():
        if node[1].label.tag == 'ext':
            if float(es.results['Main'][node]['sequences'].sum()) > 0:
                parameter[label] = {}
                params = es.results['Param'][(node[1], None)]['scalars']
                parameter[label]['cf_elec_chp'] = None
                parameter[label]['cf_heat_chp'] = None
                parameter[label]['cf_elec_condense'] = None
                for k, v in params.items():
                    if 'electricity' in k and 'full' not in k:
                        parameter[label]['cf_elec_chp'] = v
                    elif 'heat' in k:
                        parameter[label]['cf_heat_chp'] = v
                    elif 'electricity' in k and 'full' in k:
                        parameter[label]['cf_elec_condense'] = v
                mustrun['chp'] = (heat_demand_distr /
                                  parameter[label]['cf_heat_chp'] *
                                  parameter[label]['cf_elec_chp'])
        elif node[0].label.tag == 'ee':
            mustrun[node[0].label.subtag] = (
                es.results['Main'][node]['sequences']['flow'])

    df = pd.DataFrame()
    df['mustrun'] = mustrun.sum(axis=1)
    df['demand'] = bus_balnc[('FHG', 'out', 'demand', 'electricity', 'all')]
    df['diff'] = df['mustrun'] - df['demand']
    df['diff'][df['diff'] < 0] = 0
    print('diff', df['diff'].sum())
    print(heat_supply_hp.sum())
    df['export'] = export - df['diff']
    df['fuel_use_export'] = (
        df['export'] / parameter[label]['cf_elec_condense'])
    df['fuel_save_import'] = (
        import_elec / parameter[label]['cf_elec_condense'])
    return df


def multi_analyse_fhg_emissions():
    emission_analysis = pd.DataFrame()
    start_dir = os.path.join(cfg.get('paths', 'scenario'),
                             'friedrichshagen', 'Sonntag')
    n = 34
    for root, directories, filenames in os.walk(start_dir):
        for fn in filenames:
            if fn[-5:] == '.esys':
                name = fn.replace('friedrichshagen_2014_meritorder_', '')
                name = name.replace('.esys', '')
                elements = name.split('_')
                upstream = '_'.join(elements[:-7])
                # local = '_'.join(elements[-7:])
                if 'berlin_hp' not in upstream and 'de22' not in upstream:
                    r = analyse_fhg_emissions(upstream,
                                              os.path.join(start_dir, fn))
                    for field in r._fields:
                        emission_analysis.loc[name, field] = getattr(r, field)
                n -= 1
                logging.info("Rest: {0}".format(n))
    emission_analysis.to_excel('/home/uwe/emissions_analysis.xls')


def analyse_fhg_emissions(upstream_name, fn_local):
    results = namedtuple('e', ('total', 'optional_export', 'optional_import',
                               'displaced', 'supplement'))
    local_es = load_es(fn=fn_local)
    local_emissions = emissions(local_es)  # a)
    label = 'chp_ext'
    bus_balnc = get_multiregion_bus_balance(local_es)
    export = bus_balnc[('FHG', 'out', 'export', 'electricity', 'all')]
    import_elec = bus_balnc[('FHG', 'in', 'import', 'electricity', 'all')]
    fuel_use = get_optional_export_emissions(local_es, label, bus_balnc)

    local_chp_fuel = '_'.join(local_es.name.split('_')[-5:-3])
    emission_optional_export = (
        fuel_use['fuel_use_export'] *
        local_emissions.loc[local_chp_fuel]['specific_emission'])
    emission_optional_import = (
        fuel_use['fuel_save_import'] *
        local_emissions.loc[local_chp_fuel]['specific_emission'])
    total_local_emissions = local_emissions.total_emission.sum()

    if upstream_name != 'no_costs':
        up_val = fsc.load_upstream_scenario_values()[upstream_name]
        displaced_emissions = up_val['emission_last'].multiply(
            export.reset_index(drop=True)).sum()
        supplement_emissions = up_val['emission_last'].multiply(
            import_elec.reset_index(drop=True)).sum()
    else:
        displaced_emissions = float('nan')
        supplement_emissions = float('nan')

    return results(total_local_emissions, emission_optional_export.sum(),
                   emission_optional_import.sum(), displaced_emissions,
                   supplement_emissions)


def analyse_ee_basic_core(res, fn, path, upstream_values):

    solar = float(fn.split('_')[-3].replace('solar', ''))
    wind = float(fn.split('_')[-4].replace('wind', ''))
    name = 'pv{:02}_wind{:02}'.format(int(solar), int(wind))
    local_es = load_es(fn=os.path.join(path, fn))
    bus_balnc = get_multiregion_bus_balance(local_es)

    flow = {
        'import': bus_balnc[('FHG', 'in', 'import', 'electricity',
                             'all')].reset_index(drop=True),
        'export': bus_balnc[('FHG', 'out', 'export', 'electricity',
                             'all')].reset_index(drop=True)}

    for t in ['import', 'export']:
        res['energy'].loc[name, t] = flow[t].sum()
        res['costs'].loc[name, t] = flow[t].multiply(
            upstream_values['meritorder']).sum()
        res['emissions'].loc[name, t] = flow[t].multiply(
            upstream_values['emission_last']).sum()
    return res


def analyse_ee_basic():
    start_dir = os.path.join(cfg.get('paths', 'scenario'),
                             'friedrichshagen', 'ee_results')

    results = {
        'energy': pd.DataFrame(),
        'costs': pd.DataFrame(),
        'emissions': pd.DataFrame()}

    upstream_values = fsc.load_upstream_scenario_values()
    up_scen = gui.get_choice(
        upstream_values.columns.get_level_values(0).unique(),
        "Upstream scenarios", "Select an upstream scenario")
    upstream_values = upstream_values[up_scen]

    for root, directories, filenames in os.walk(start_dir):
        for fn in filenames:
            if fn[-5:] == '.esys':
                results = analyse_ee_basic_core(results, fn, start_dir,
                                                upstream_values)

    for key, value in results.items():
        value.sort_index(0, inplace=True)
        value.sort_index(1, inplace=True)
        value.plot(kind='bar', title=key)
    plt.show()


# def get_resource_usage(es):
#     print(get_multiregion_bus_balance(es, 'bus_commodity').sum())


def analyse_berlin_basic():
    r = {}
    year = 2014
    scenarios = {
        'deflex_de22': {'es': ['deflex', str(year)],
                        'var': 'de22',
                        'region': 'DE22'},
        'berlin_de22': {'es': ['berlin_hp', str(year)],
                        'var': 'de22',
                        'region': 'BE'},
        'berlin_de21': {'es': ['berlin_hp', str(year)],
                        'var': 'de21',
                        'region': 'BE'},
        'berlin_up_de21': {'es': ['berlin_hp', str(year)],
                          'var': 'single_up_deflex_2014_de21',
                          'region': 'BE'},
        'berlin_single': {'es': ['berlin_hp', str(year)],
                          'var': 'single_up_None',
                          'region': 'BE'},
        # 'deflex_de22b': {'es': ['deflex', str(year), 'de22'],
        #                 'region': 'DE22'},
        # 'deflex_de22c': {'es': ['deflex', str(year), 'de22'],
        #                 'region': 'DE22'},
    }

    for k, v in scenarios.items():
        es = load_es(*v['es'], var=v['var'])

        resource_blnc = get_multiregion_bus_balance(es, 'bus_commodity').sum()
        r[k] = resource_blnc.loc[v['region'], 'in', 'source', 'commodity']

        if 'None' not in v['var']:
            elec_balance = get_multiregion_bus_balance(es)
            import_berlin = elec_balance.sum().loc[v['region'], 'in', 'import',
                                                   'electricity', 'all']
            export_berlin = elec_balance.sum().loc[v['region'], 'out', 'export',
                                                   'electricity', 'all']
            netto_import = import_berlin - export_berlin
            r[k]['netto_import'] = netto_import  # /0.357411
        else:
            r[k]['netto_import'] = 0

        r[k]['other'] = r[k].get('other', 0) + r[k].get('waste', 0)

    # Energiebilanz Berlin 2014
    r['statistic'] = pd.Series({
        'bioenergy': 7152000,
        'hard_coal': 43245000,
        'lignite': 12274000,
        'natural_gas': 80635000,
        'oil': 29800000,
        'other': 477000,
        'netto_import': 19786000}).div(3.6)

    df = pd.concat(r).drop('waste', axis=0, level=1).unstack().div(1000000)

    print(df.sum(axis=1))
    print(df.sum(axis=0))

    color_dict = my_plot.get_cdict_df(df)

    # use get to specify dark gray as the default color.
    df.plot(
        kind='bar', color=[color_dict.get(x, '#bbbbbb') for x in df.columns])

    print(df)
    df['import_ressources'] = df['netto_import'] / 0.40
    df.drop('netto_import', axis=1, inplace=True)
    print(df.sum(axis=1))
    # r[2].plot(kind='bar', axis=a)
    plt.show()


def analyse_fhg_basic():
    # upstream_es = load_es(2014, 'de21', 'deflex')
    path = os.path.join(
        cfg.get('paths', 'scenario'), 'friedrichshagen', 'ee_results')
    filename = gui.select_filename(work_dir=path,
                                   title='EE results!',
                                   extension='esys')
    print(filename)
    local_es = load_es(fn=os.path.join(path, filename))

    flh = fullloadhours(local_es)
    print(flh.loc[flh.nominal_value != 0])
    bus_balnc = get_multiregion_bus_balance(local_es)
    imp = bus_balnc[('FHG', 'in', 'import', 'electricity', 'all')]
    exp = bus_balnc[('FHG', 'out', 'export', 'electricity', 'all')]
    imp.name = 'import'
    exp.name = 'export'
    # print(bus_balnc.min())
    bus_balnc[bus_balnc < 0] = 0
    print(bus_balnc.min())
    print(bus_balnc.sum())
    # exit(0)
    heat_blnc = get_multiregion_bus_balance(local_es, 'bus_heat')

    heat_demand = heat_blnc[('vattenfall_friedrichshagen',
                            'out', 'demand', 'heat', 'district')].reset_index(
        drop=True)
    cost_scenario = 'deflex_2014_de21'
    value = 'meritorder'
    upstream = fsc.load_upstream_scenario_values()
    costs = upstream[cost_scenario][value]
    df = pd.concat([imp.reset_index(drop=True),
                    exp.reset_index(drop=True),
                    costs], axis=1)
    df['imp_cost'] = df['import'].div(df['import']).multiply(
        df[value])
    df['exp_cost'] = df['export'].div(df['export']).multiply(
        df[value])
    df['exp_cost'].plot()
    print(heat_demand.max())
    print(heat_demand.min())
    print(heat_demand.value_counts(sort=True, bins=10))
    print(df['exp_cost'].value_counts().sort_index())
    print(df.max())
    print(df.min())
    ax = df.plot()
    # ax = costs.plot()
    ax = heat_demand.plot(ax=ax)
    plot_bus_view(local_es, ax=ax)


def analyse_upstream_scenarios():
    start_dir = os.path.join(cfg.get('paths', 'scenario'),
                             'friedrichshagen', 'Sonntag')
    up_scenarios = []
    for root, directories, filenames in os.walk(start_dir):
        for fn in filenames:
            if fn[-5:] == '.esys' and 'berlin_hp' not in fn:
                up_scenarios.append(
                    fn.replace('hard_coal', '{0}'
                               ).replace('natural_gas', '{0}'))

    # fn_pattern = gui.get_choice(sorted(set(up_scenarios)),
    #                     "Select scenario", "Upstream scenarios")

    # for fuel in ['hard_coal', 'natural_gas']:
    fuel = 'natural_gas'
    for fn_pattern in sorted(set(up_scenarios)):
        fn = (fn_pattern.format(fuel))
        local_es = load_es(fn=os.path.join(
            cfg.get('paths', 'scenario'), 'friedrichshagen', 'Sonntag', fn))
        print(get_multiregion_bus_balance(local_es).sum())
        heat_blnc = get_multiregion_bus_balance(local_es, 'bus_heat')
        print(heat_blnc.sum())
        print(get_multiregion_bus_balance(
            local_es, 'bus_commodity').sum())


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__)])
    analyse_berlin_basic()
    exit(0)
    # analyse_ee_basic()
    # exit(0)
    analyse_fhg_basic()
    # analyse_upstream_scenarios()
    # exit(0)
    # analyse_fhg_basic()
    # multi_analyse_fhg_emissions()
    df = pd.read_excel('/home/uwe/emissions_analysis.xls')
    # for c in df.index:
    #     print(c)
    #     if 'hard_coal' in c:
    #         df.drop(c, inplace=True)
    df['optional_tot'] = (df['total'] - df['optional_export'] +
                          df['optional_import'])
    df['upstream_tot'] = df['total'] - df['displaced'] + df['supplement']
    df.sort_index(axis=1).sort_index(axis=0).plot(kind='bar')
    plt.show()
    # analyse_fhg_emissions()
    exit(0)
    # my_es = load_es(2014, 'de21', 'deflex')
    # analyse_system_costs(my_es, plot=True)
    # exit(0)
    my_sres = something()
    sort_col = ['upstream', 'fuel', 'region']  #'export avg costs'
    sort_idx = my_sres['meta'].sort_values(sort_col).index
    for df in ['avg_costs', 'absolute_costs', 'absolute_flows',
               'absolute_emissions']:
        my_sres[df].loc[sort_idx].plot(kind='bar')
    # my_sres['avg_costs'].sort_values(sort_col).plot(kind='bar')
    # my_sres['avg_costs'].loc[sort_idx].plot(kind='bar')

    # my_sres['absolute_flows'][cols].sort_values('export')

    pd.concat([my_sres['meta'],
               my_sres['avg_costs'],
               my_sres['absolute_costs'],
               my_sres['absolute_flows'],
               my_sres['absolute_emissions'],
               my_sres['upstream_emissions'],
               my_sres['emissions']],
              axis=1).to_excel('/home/uwe/test.xls')
    plt.show()
    exit(0)
    # analyse_system_costs(plot=True)
    # my_es = load_es(2014, 'deflex_2014_de21', 'friedrichshagen')

    exit(0)
    # ebus_seq = outputlib.views.node(my_es.results['Main'],
    #                                 'bus_electricity_all_FHG')['sequences']
    # print(ebus_seq.sum())
    #
    # ebus = my_es.groups['bus_electricity_all_FHG']
    # export_node = my_es.groups['export_electricity_all_FHG']
    # import_node = my_es.groups['import_electricity_all_FHG']
    # export_costs = my_es.flows()[(ebus, export_node)].variable_costs / 0.99
    # import_costs = my_es.flows()[(import_node, ebus)].variable_costs / 1.01
    # export_flow = my_es.results['Main'][(ebus, export_node)]['sequences']
    # import_flow = my_es.results['Main'][(import_node, ebus)]['sequences']
    #
    # export_flow.reset_index(drop=True, inplace=True)
    # import_flow.reset_index(drop=True, inplace=True)
    #
    # total_export_costs = export_flow.multiply(export_costs, axis=0)
    # total_import_costs = import_flow.multiply(import_costs, axis=0)
    #
    # print(total_import_costs.sum() / import_flow.sum())
    # print(total_export_costs.sum() / export_flow.sum() * -1)
    #
    # ax = total_export_costs.plot()
    # total_import_costs.plot(ax=ax)
    # plt.show()
    # exit(0)
    plot_bus(my_es, 'bus_electricity_all_DE01')
    # exit(0)
    # analyse_bus(2014, 'de21', 'deflex', 'DE01')


    exit(0)

    # scenarios = pd.read_csv('scenarios.csv')
    # scenarios['description'] = scenarios.apply(lambda x: '_'.join(x), axis=1)
    #
    # # scenarios.to_csv('scenarios.csv')
    # costs_df = pd.DataFrame()
    # for scen in scenarios.itertuples():
    #     logging.info("Process scenario: {0}".format(scen.description))
    #     if scen.rmap != 'single':
    #         my_es = load_es(scen.variant, scen.rmap, scen.cat, with_tags=True)
    #         costs_df[scen.description] = analyse_system_costs(my_es)

    # costs_df.to_excel('/home/uwe/costs.xls')
    costs_df = pd.read_excel('/home/uwe/costs.xls')
    print(costs_df.sum())
    print(costs_df.mean())
    costs_df.plot(legend=True)
    plt.show()
    # print(get_nominal_values(my_es))
    # exit(0)
    # plot_bus_view(my_es)
    exit(0)
    my_es_2 = load_es(2014, 'de21', 'berlin_hp')
    # reshape_bus_view(my_es, my_es.groups['bus_electricity_all_DE01'])
    get_multiregion_bus_balance(my_es, 'bus_electricity_all')

    compare_transmission(my_es, my_es_2)
    exit(0)

    print(emissions(my_es).div(1000000))
    exit(0)
    fullloadhours(my_es, [0, 1, 2, 3, 4]).to_excel('/home/uwe/test.xls')
    exit(0)
    analyse_system_costs(my_es, plot=True)
    plt.show()
    print('Done')
    df1 = get_one_node_type(my_es, node='shortage_bus_elec')
    df2 = get_one_node_type(my_es)
    print(df1.round(1))
    print(df2.round(1))
    exit(0)

    my_res = plot_multiregion_io(my_es)
    print(my_res.sum())
    print(my_res.max())

    print(my_res.sum().div(my_res.max()))

    plt.show()

    # analyse_bus(2014, 'single', 'friedrichshagen', 'FHG')
    exit(0)
    # all_res = load_all_results(2014, 'de21', 'deflex')
    # wind = ee_analyser(all_res, 'wind')
    # solar = ee_analyser(all_res, 'solar')
    # new_analyser(all_res, wind, solar)
    # print(bl['bus_elec_DE07'])
    exit(0)
    # system = load_es(2014, 'de21', 'deflex')
    # results = load_es(2014, 'de21', 'deflex').results['Main']
    # param = load_es(2014, 'de21', 'deflex').results['Param']
    # cs_bus = [x for x in results.keys() if ('_cs_' in x[0].label) & (
    #     isinstance(x[1], solph.Transformer))]

    # print(flows.sum())
    # exit(0)
    # system = load_es(2014, 'de21', 'deflex')
    # FINNISCHE METHODE
    eta_th_kwk = 0.5
    eta_el_kwk = 0.3
    eta_th_ref = 0.9
    eta_el_ref = 0.5
    pee = (1/(eta_th_kwk/eta_th_ref + eta_el_kwk/eta_el_ref)) * (
            eta_el_kwk/eta_el_ref)
    pet = (1/(eta_th_kwk/eta_th_ref + eta_el_kwk/eta_el_ref)) * (
            eta_th_kwk/eta_th_ref)

    # print(param)
    # Refferenzwirkungsgrade Typenabhängig (Kohle, Gas...)

    print(pee * 200, pet * 200)

    # https://www.ffe.de/download/wissen/
    # 334_Allokationsmethoden_CO2/ET_Allokationsmethoden_CO2.pdf
