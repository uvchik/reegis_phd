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
# import reegis_tools.config as cfg
import reegis_tools.gui as gui
from my_reegis.plot import *
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
                     vmax=None, label_max=None, unit='GWh'):
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
                           column='color')
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

    plt.show()


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

    # PLOTS
    transmission = transmission.div(1000)
    transmission.plot(kind='bar')

    # transmission['diff_1-2'] = transmission[name1] - transmission[name2]
    transmission['diff_2-1'] = transmission[name2] - transmission[name1]
    transmission['fraction'] = (transmission['diff_2-1'] /
                                abs(transmission[name1]) * 100)
    transmission['fraction'].fillna(0, inplace=True)

    key = gui.get_choice(list(transmission.columns),
                         "Plot transmission lines", "Choose data column.")

    vmax = max([abs(transmission[key].max()), abs(transmission[key].min())])

    units = {'es1': 'GWh', 'es2': 'GWh', 'diff_2-1': 'GWh', 'fraction': '%'}

    plot_power_lines(transmission, key, vmax=vmax/10, unit=units[key])

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

    my_cdict = get_cdict_df(multi_reg_res['in'])
    my_cdict.update(get_cdict_df(multi_reg_res['out']))
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


def load_es(variant, rmap, cat, write_graph=False, with_tags=True):
    if isinstance(variant, int):
        var_path = str(variant)
    else:
        var_path = 'new'

    path = os.path.join(cfg.get('paths', 'scenario'), var_path, 'results')
    file = '{cat}_{var}_{rmap}.esys'.format(cat=cat, var=variant, rmap=rmap)
    sc = deflex.Scenario()
    fn = os.path.join(path, file)
    logging.info("Restoring file from {0}".format(fn))
    file_datetime = datetime.fromtimestamp(os.path.getmtime(fn)).strftime(
        '%d. %B %Y - %H:%M:%S')
    logging.info("Restored results created on: {0}".format(file_datetime))
    sc.restore_es(fn)

    if write_graph:
        graph.create_nx_graph(sc.es, filename='/home/uwe/mygraph',
                              remove_nodes_with_substrings=['commodity'])
    if with_tags:
        sc.es.results['tags'] = create_tags(sc.es.nodes)
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


def plot_bus_view(es, bus=None):
    """

    Parameters
    ----------
    es
    bus : str or Node or None

    Returns
    -------

    """
    if isinstance(bus, str):
        bus = es.groups[bus]

    if bus is None:
        data = get_multiregion_bus_balance(es).groupby(
            axis=1, level=[1, 2, 3, 4]).sum()
        title = 'Germany'
    else:
        data = reshape_bus_view(es, bus)[bus.label.region]
        title = repr(bus.label)

    fig = plt.figure(figsize=(10, 5))

    my_colors = get_cdict(data['in'])
    my_colors.update(get_cdict(data['out']))

    my_plot = oev.plot.io_plot(
        df_in=data['in'], df_out=data['out'], cdict=my_colors,
        inorder=get_orderlist_from_multiindex(data['in'].columns),
        outorder=get_orderlist_from_multiindex(data['out'].columns),
        ax=fig.add_subplot(1, 1, 1), smooth=True)
    ax = shape_tuple_legend(**my_plot)
    ax = oev.plot.set_datetime_ticks(ax, data.index, tick_distance=48,
                                     date_format='%d-%m-%H', offset=12,
                                     tight=True)
    ax.set_ylabel('Power in MW')
    ax.set_xlabel('Date')
    ax.set_title(title)
    plt.show()
    exit(0)


def plot_bus(node, node_label, rm_list=None):

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
    # plt.show()


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

    back = namedtuple('analysis', ['lcoe', 'misc'])

    r = es.results['Main']
    p = es.results['Param']

    df_sort_res = pd.DataFrame()
    dict_analyse = {}
    costs = {}
    for i in r.keys():
        df_sort_res = sorted_results(i, df_sort_res, r)
        dict_analyse = analyze(i, dict_analyse, costs, demand, r, p)

    df_sort_res.sort_index(axis=1, inplace=True)
    df_analyse = pd.DataFrame.from_dict(dict_analyse)
    df_analyse.reset_index(drop=True, inplace=True)
    return back(df_analyse, df_sort_res)


def analyze(args, result, costs, demand, r, p):
    cost_flow = None
    label = args

    if isinstance(args[1], solph.Transformer):
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
                cost_flow = r[args]['sequences']
            else:
                cost_flow = None

        else:
            print(args[1].label, len(args[1].outputs))

        if args[0] not in costs:
            var_costs = 0
            flow_seq = 1
            for i in args[0].inputs:
                var_costs += (p[(i, args[0])]['scalars']['variable_costs'] *
                              r[(i, args[0])]['sequences'].sum())
                flow_seq += r[(i, args[0])]['sequences'].sum()
            costs[args[0]] = var_costs / flow_seq

    elif 'shortage' == args[0].label.cat and args[1] is not None:
        label = args[0].label
        # self.costs[args[0]] = self.psc(args)['variable_costs']
        costs[args[0]] = 50
        cost_flow = r[args]['sequences']

    if cost_flow is not None:
        result[label] = cost_flow * costs[args[0]]
        result[label] = result[label]['flow'].div(demand)

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


def analyse_plot(cost_values, multiregion, demand_elec, demand_distr):
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

    my_cols = pd.MultiIndex(levels=[[], [], []], labels=[[], [], []],
                            names=[u'type', u'src', u'region'])
    ana_df = pd.DataFrame(columns=my_cols)

    for c in cost_values.columns:
        ana_df[str(c)] = cost_values[c]
    ana_df = ana_df.groupby(level=[0, 1], axis=1).sum()

    # ana_df[('chp', 'chp')] = ana_df['chp'].sum(axis=1)

    ana_x = ana_df.reset_index(drop=True).plot()
    ana_x .set_xlim(3000, 3200)

    cost_values.reset_index(drop=True, inplace=True)
    cost_values.name = 'cost_value (right)'
    ax2 = cost_values.sum(axis=1).plot(
        ax=ax, secondary_y=True, legend=True, color='#7cfff0')
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
    multi_res = reshape_multiregion_df(es)

    demand = multi_res['out', 'demand', 'electricity', 'all']

    iter_res = results_iterator(es, demand)
    c_values = iter_res.lcoe
    # c_values = new_analyser(all_res, multi_res[
    #     ('out', 'demand_electricity_all')])

    if plot is True:
        sorted_flows = iter_res.misc
        analyse_plot(c_values, multi_res, sorted_flows[
            'demand_electricity'], sorted_flows['demand_heat'])

    return c_values.sum(axis=1)


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__)])
    # analyse_system_costs(plot=True)
    # my_es = load_es(2014, 'single', 'friedrichshagen')
    # exit(0)
    # analyse_bus(2014, 'de21', 'deflex', 'DE01')

    my_es = load_es(2014, 'de21', 'deflex')
    analyse_system_costs(my_es, plot=True)
    # print(get_nominal_values(my_es))
    # exit(0)
    plot_bus_view(my_es)
    exit(0)
    # my_es_2 = load_es(2014, 'de21', 'berlin_hp')
    # reshape_bus_view(my_es, my_es.groups['bus_electricity_all_DE01'])
    get_multiregion_bus_balance(my_es, 'bus_electricity_all')

    # compare_transmission(my_es, my_es_2)
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

    # for src in trsf_temp.keys():
    #     if src not in trsf:
    #         trsf[src] = {}
    #     for tpl in trsf_temp[src]:
    #         k = tpl[1].label.split('_')[1]
    #         if k not in trsf[src]:
    #             trsf[src][k] = []
    #         trsf[src][k].append(tpl[1].label)
    # pprint.pprint(trsf)

    # pp.pprint(cs_bus)
    exit(0)
    stopwatch()
    # show_region_values_gui(2014)
    # sum_up_electricity_bus(2014, 'single', 'berlin_hp', 'BE')
    # analyse_bus(2014, 'de21', 'deflex', 'DE01')

    compare_transmission(2014)
    exit(0)
    # get_full_load_hours(2014)
    # check_excess_shortage(2014)
    # plot_bus('bus_distr_heat_vattenfall_mv')
    # plot_bus('bus_elec_BE')
    compare_transmission(2014)
    exit(0)
