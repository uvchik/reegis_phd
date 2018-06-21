import berlin_hp
import os
import logging
import reegis_tools.gui as gui
import reegis_tools.geometries
import pprint as pp
from oemof.tools import logger
from oemof import solph as solph
from datetime import datetime
from oemof import outputlib
from oemof.outputlib import analyzer
from matplotlib import pyplot as plt
import oemof_visio as oev
import pandas as pd
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import math
from matplotlib.colors import LinearSegmentedColormap
import reegis_tools.config as cfg


def shape_legend(node, rm_list, reverse=False, **kwargs):
    handels = kwargs['handles']
    labels = kwargs['labels']
    axes = kwargs['ax']
    parameter = {}

    new_labels = []
    for label in labels:
        label = label.replace('(', '')
        label = label.replace('), flow)', '')
        label = label.replace(node, '')
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
                     vmax=None, label_max=None):
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
                '{0} GWh'.format(round(v[key])),
                color='#000000',
                fontsize=9.5,
                zorder=15,
                path_effects=[
                    path_effects.withStroke(linewidth=3, foreground="w")])

    polygons.gdf.apply(lambda x: ax.annotate(
        s=x.name, xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

    plt.show()


def compare_transmission(year):
    # DE21
    sc_de21_results = load_results(year, 'de21', 'deflex')

    # DE21 + BE
    sc_debe_results = load_results(year, 'de22', 'berlin_hp')

    lines = [x for x in sc_de21_results.keys() if 'power_line' in x[0]]

    # Calculate results
    transmission = pd.DataFrame()
    for line in lines:
        line_name = line[0].replace('power_line_', '').replace('_', '-')
        regions = line_name.split('-')
        name1 = 'power_line_{0}_{1}'.format(regions[0], regions[1])
        name2 = 'power_line_{0}_{1}'.format(regions[1], regions[0])
        # sc_de21.results
        de21a = outputlib.views.node(sc_de21_results, name1)['sequences'].sum()
        de21b = outputlib.views.node(sc_de21_results, name2)['sequences'].sum()
        debea = outputlib.views.node(sc_debe_results, name1)['sequences'].sum()
        debeb = outputlib.views.node(sc_debe_results, name2)['sequences'].sum()
        transmission.loc[line_name, 'DE'] = de21a.iloc[0] - de21b.iloc[0]
        transmission.loc[line_name, 'BE'] = debea.iloc[0] - debeb.iloc[0]

    # PLOTS
    transmission.plot(kind='bar')

    transmission['diff'] = transmission['BE'] - transmission['DE']

    key = gui.get_choice(list(transmission.columns),
                         "Plot transmission lines", "Choose data column.")
    transmission[key] = transmission[key] / 1000
    vmax = max([abs(transmission[key].max()), abs(transmission[key].min())])
    plot_power_lines(transmission, key, vmax=vmax/2)

    return transmission


def powerlines2export_import(bus):
    print(bus['sequences'].sum())
    export_cols = [x for x in bus['sequences'].columns
                   if 'power_line' in x[0][1]]
    import_cols = [x for x in bus['sequences'].columns
                   if 'power_line' in x[0][0]]
    bus_name = [x[0][0] for x in bus['sequences'].columns
                if 'power_line' in x[0][1]][0]
    export_name = ((bus_name, 'export'), 'flow')
    import_name = (('import', bus_name), 'flow')
    bus['sequences'][export_name] = bus['sequences'][export_cols].sum(axis=1)
    bus['sequences'][import_name] = bus['sequences'][import_cols].sum(axis=1)
    bus['sequences'].drop(export_cols, axis=1, inplace=True)
    bus['sequences'].drop(import_cols, axis=1, inplace=True)
    return bus


def analyse_bus(year, rmap, cat, region):
    results = load_results(year, rmap, cat)
    print(results.keys())
    bus_elec = outputlib.views.node(results, 'bus_elec_{0}'.format(region))

    # bus_elec = powerlines2export_import(bus_elec)
    print(bus_elec['sequences'].sum())
    plot_bus(bus_elec, 'bus_elec_{0}'.format(region),
             rm_list=['_{0}'.format(region)])
    h_bus = 'bus_distr_heat_vattenfall_friedrichshagen'
    bus_heat = outputlib.views.node(results, h_bus)
    print(bus_heat['sequences'].sum())
    plot_bus(bus_heat, h_bus,
             rm_list=['_{0}'.format(region)])
    plt.show()


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


def reshape_results(results, data, region, node=None):
    if node is None:
        node = 'bus_elec_{0}'.format(region)
    tmp = outputlib.views.node(results, node)['sequences']

    # aggregate powerline columns to import and export
    exp = [x for x in tmp.columns if 'power_line' in x[0][1]]
    imp = [x for x in tmp.columns if 'power_line' in x[0][0]]
    tmp[((node, 'export'), 'flow')] = tmp[exp].sum(axis=1)
    tmp[(('import', node), 'flow')] = tmp[imp].sum(axis=1)
    tmp.drop(exp + imp, axis=1, inplace=True)

    in_c = [x for x in tmp.columns if x[0][1] == node]
    out_c = [x for x in tmp.columns if x[0][0] == node]

    data[region] = pd.concat({'in': tmp[in_c], 'out': tmp[out_c]}, axis=1)
    dc = {}
    for c in data[region]['in'].columns:
        dc[c] = c[0][0].replace('_{0}'.format(region), '')
    for c in data[region]['out'].columns:
        dc[c] = c[0][1].replace('_{0}'.format(region), '')
    data[region] = data[region].rename(columns=dc, level=1)
    return data


def get_multiregion_results(year):
    de_results = load_results(year, 'de21', 'deflex')

    regions = [x[0].replace('shortage_bus_elec_', '')
               for x in de_results.keys()
               if 'shortage_bus_elec' in x[0]]

    data = {}
    for region in sorted(regions):
        data = reshape_results(de_results, data, region)
    return pd.concat(data, axis=1)


def show_region_values_gui(year):
    data = get_multiregion_results(year)
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


def load_results(year, rmap, cat):
    path = os.path.join(cfg.get('paths', 'scenario'), str(year), 'results')
    file = '{cat}_{year}_{rmap}.esys'.format(cat=cat, year=year, rmap=rmap)
    sc = berlin_hp.Scenario()
    fn = os.path.join(path, file)
    logging.info("Restoring file from {0}".format(fn))
    print(datetime.fromtimestamp(os.path.getmtime(fn)).strftime(
        '%d. %B %Y - %H:%M:%S'))
    sc.restore_es(fn)
    return outputlib.processing.convert_keys_to_strings(sc.results)


def load_es(year, rmap, cat):
    path = os.path.join(cfg.get('paths', 'scenario'), str(year), 'results')
    file = '{cat}_{year}_{rmap}.esys'.format(cat=cat, year=year, rmap=rmap)
    sc = berlin_hp.Scenario()
    fn = os.path.join(path, file)
    logging.info("Restoring file from {0}".format(fn))
    print(datetime.fromtimestamp(os.path.getmtime(fn)).strftime(
        '%d. %B %Y - %H:%M:%S'))
    sc.restore_es(fn)
    return sc.es


def load_all_results(year, rmap, cat):
    return load_es(year, rmap, cat).results


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


def plot_bus(node, node_label, rm_list=None):

    fig = plt.figure(figsize=(10, 5))

    my_node = node['sequences']

    if rm_list is None:
        rm_list = []

    plot_slice = oev.plot.slice_df(my_node,)
                                   # date_from=datetime(2014, 5, 31),
                                   # date_to=datetime(2014, 6, 8))

    # pprint.pprint(get_cdict(my_node))

    pp.pprint(my_node.columns)
    # exit(0)
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


def get_orderlist(my_node, flow):
    my_order = ['source_solar', 'source_wind', 'chp', 'hp', 'pp', 'import',
                'shortage', 'power_line', 'demand', 'export', 'excess']
    cols = list(my_node.columns)
    if flow == 'in':
        f = 0
    elif flow == 'out':
        f = 1
    else:
        logging.error("A flow has to be 'in' or 'out.")
    order = []

    for element in my_order:
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
            while color_keys[n] not in col[0][0].lower():
                n += 1
            if len(my_colors[color_keys[n]]) > 1:
                color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
            else:
                color = '#{0}'.format(my_colors[color_keys[n]][0])
            color_dict[col] = color
        except IndexError:
            n = 0
            try:
                while color_keys[n] not in col[0][1].lower():
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
            while color_keys[n] not in col.lower():
                n += 1
            if len(my_colors[color_keys[n]]) > 1:
                color = '#{0}'.format(my_colors[color_keys[n]].pop(0))
            else:
                color = '#{0}'.format(my_colors[color_keys[n]][0])
            color_dict[col] = color
        except IndexError:
            color_dict[col] = '#ff00f0'
    return color_dict


def my_analyser(results):
    import pprint as pp
    analysis = analyzer.Analysis(
        results['Main'],
        results['Param'],
        iterator=analyzer.FlowNodeIterator)
    seq = analyzer.SequenceFlowSumAnalyzer()
    ft = analyzer.FlowTypeAnalyzer()
    demand_nodes = [x[1] for x in results['Main'].keys() if x[1] is not None and 'demand_elec' in x[1].label]
    print(demand_nodes)
    lcoe = analyzer.LCOEAnalyzer(demand_nodes)
    nb_analyzer = analyzer.NodeBalanceAnalyzer()
    analysis.add_analyzer(analyzer.SequenceFlowSumAnalyzer())
    analysis.add_analyzer(analyzer.VariableCostAnalyzer())
    analysis.add_analyzer(analyzer.InvestAnalyzer())
    analysis.add_analyzer(lcoe)
    analysis.add_analyzer(ft)
    analysis.add_analyzer(nb_analyzer)
    analysis.analyze()
    balance = lcoe.result
    print(lcoe.total)
    # pp.pprint(seq.result)
    pp.pprint(balance)
    blnc = {}
    for b in balance.keys():
        blnc[b[0].label] = balance[b]

    return blnc


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


def results_iterator(results):
    r = results['Main']
    p = results['Param']

    sorted_results = pd.DataFrame()
    for i in r.keys():
        # find source flows
        if isinstance(i[0], solph.Source):
            column_name = i[0].label.split('_')[1]
            try:
                sorted_results[column_name] += r[i]['sequences']['flow']
            except KeyError:
                sorted_results[column_name] = r[i]['sequences']['flow']

        # find storage discharge
        if isinstance(i[0], solph.components.GenericStorage):
            if i[1] is not None:
                column_name = i[0].label.split('_')[1] + '_discharge'
                try:
                    sorted_results[column_name] += r[i]['sequences']['flow']
                except KeyError:
                    sorted_results[column_name] = r[i]['sequences']['flow']

        # find elec_demand
        if isinstance(i[1], solph.Sink):
            column_name = '_'.join(i[1].label.split('_')[:2])
            try:
                sorted_results[column_name] += r[i]['sequences']['flow']
            except KeyError:
                sorted_results[column_name] = r[i]['sequences']['flow']

    return sorted_results.sort_index(axis=1)


def new_analyser(results, demand):
    analysis = analyzer.Analysis(
        results['Main'], results['Param'],
        iterator=analyzer.FlowNodeIterator)
    lcoe = analyzer.LCOEAnalyzerCHP(demand)
    analysis.add_analyzer(lcoe)
    analysis.analyze()
    df = pd.DataFrame.from_dict(lcoe.result)
    df.reset_index(drop=True, inplace=True)
    df.name = 'value (right)'
    return df


def analyse_plot(cost_values, multiregion, demand_elec, demand_distr):
    in_list = ['trsf_pp_nuclear', 'trsf_pp_lignite', 'trsf_pp_hard_coal',
               'trsf_pp_natural_gas', 'trsf_pp_other', 'trsf_pp_bioenergy',
               'trsf_chp_other', 'trsf_chp_lignite', 'trsf_chp_natural_gas',
               'trsf_chp_oil', 'trsf_chp_bioenergy', 'trsf_chp_hard_coal',
               'phe_storage']
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
        cn = c.split('_')
        ana_df[(cn[1], cn[3], cn[2])] = cost_values[c]
    ana_df = ana_df.groupby(level=[0, 1], axis=1).sum()
    # ana_df[('chp', 'chp')] = ana_df['chp'].sum(axis=1)

    ana_x = ana_df.reset_index(drop=True).plot()
    ana_x .set_xlim(3000, 3200)

    cost_values.reset_index(drop=True, inplace=True)
    cost_values.name = 'cost_value (right)'
    ax2 = cost_values.sum(axis=1).plot(
        ax=ax, secondary_y=True, legend=True, color='#7cfff0')
    ax2.set_ylim(0, 20)
    # ax2.set_xlim(3000, 3200)
    plt.show()


def reshape_multiregion_df():
    res = get_multiregion_results(2014).groupby(level=[1, 2], axis=1).sum()

    res[('out', 'losses')] = res[('out', 'export')] - res[('in', 'import')]
    del res[('out', 'export')]
    del res[('in', 'import')]

    volatile_sources = ['source_geothermal', 'source_hydro', 'source_solar',
                        'source_wind', 'phe_storage']

    for src in volatile_sources:
        res[('out', 'demand_elec')] -= res[('in', src)]
        del res[('in', src)]

    chp_trsf = [x for x in res['in'].columns if 'chp' in x]

    for chp in chp_trsf:
        res[('out', 'demand_elec')] -= res[('in', chp)]
        del res[('in', chp)]

    additional_demand = ['phe_storage', 'losses']
    for add in additional_demand:
        res[('out', 'demand_elec')] += res[('out', add)]
        del res[('out', add)]

    for c in res.columns:
        if res[c].sum() < 0.0001:
            del res[c]
    print(res.groupby(level=0, axis=1).sum().sum())
    print(res.sum())
    # exit(0)
    return res


def analyse_system_costs(plot=False):
    all_res = load_all_results(2014, 'de21', 'deflex')
    multi_res = reshape_multiregion_df()
    c_values = new_analyser(all_res, multi_res[('out', 'demand_elec')])

    if plot is True:
        sorted_flows = results_iterator(all_res)
        analyse_plot(c_values, multi_res, sorted_flows['demand_elec'],
                     sorted_flows['demand_distr'])

    return c_values.sum(axis=1)


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__)])
    # analyse_system_costs(plot=True)

    analyse_bus(2014, 'single', 'friedrichshagen', 'FHG')
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

    # trsf_temp = {}
    # trsf = {}
    # src_keys = []
    # type_keys = []
    # for x in cs_bus:
    #     k = '_'.join(x[0].label.split('_')[2:])
    #     src_keys.append(k)
    # for x in cs_bus:
    #     k = (x[1].label.split('_')[1])
    #     type_keys.append(k)
    #
    # print(set(src_keys))
    # print(set(type_keys))

    my_cols = pd.MultiIndex(levels=[[], [], []], labels=[[], [], []],
                            names=[u'type', u'src', u'region'])
    dt_idx = results[cs_bus[0]]['sequences'].index
    flows = pd.DataFrame(columns=my_cols, index=dt_idx)
    para = pd.DataFrame(columns=my_cols)
    for key in param.keys():
        try:
            print(param[key]['scalars'])
        except:
            pass

    n = 0
    for x in cs_bus:
        src = '_'.join(x[0].label.split('_')[2:])
        tp = x[1].label.split('_')[1]
        reg = x[1].label.split('_')[2]
        if tp == 'dectrl':
            reg = str(n)
            n += 1
        flows[(tp, src, reg)] = results[x]['sequences']
        print(param[(system.groups[x[1].label], None)])
        print(system.groups[x[1].label].conversion_factors)
        para[(tp, src, reg)] = param[(system.groups[x[1].label], None)]
        exit(0)

    flows.sort_index(1, inplace=True)
    # print(flows.sum().groupby(level=0).sum())
    # print(flows['dectrl', 'natural_gas'])
    # exit(0)
    print(flows.sum().sum())
    flows.drop('hp', level=0, axis=1, inplace=True)
    flows.drop('dectrl', level=0, axis=1, inplace=True)
    print(flows.sum())
    print(para)
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

    param = outputlib.processing.parameter_as_dict(system)
    # print(param)
    # Refferenzwirkungsgrade Typenabhängig (Kohle, Gas...)

    print(pee * 200, pet * 200)


    # https://www.ffe.de/download/wissen/334_Allokationsmethoden_CO2/ET_Allokationsmethoden_CO2.pdf



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
