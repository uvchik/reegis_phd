import os
import logging
import pandas as pd
from oemof import solph
from my_reegis import results
from my_reegis import reegis_plot as plot
from my_reegis import friedrichshagen_scenarios as fsc
from matplotlib import pyplot as plt
from collections import namedtuple
import reegis.config as cfg
import reegis.gui as gui


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
        eta = {}
        label = args[1].label
        if len(args[1].outputs) == 2:
            for o in args[1].outputs:
                key = 'conversion_factors_{0}'.format(o)
                eta_key = o.label.tag
                eta_val = p[(args[1], None)]['scalars'][key]
                eta[eta_key] = eta_val
            eta['heat_ref'] = 0.9
            eta['elec_ref'] = 0.55
            pee = (1 / (eta['heat'] / eta['heat_ref'] + eta['electricity'] /
                        eta['elec_ref'])) * (
                    eta['electricity'] / eta['elec_ref'])

            flow = r[args]['sequences'] * pee
            conversion_factor = eta['electricity'] / pee

        elif len(args[1].outputs) == 1:
            t_out = [o for o in args[1].outputs][0].label.tag
            t_in = [i for i in args[1].inputs][0].label.tag
            if t_out == 'electricity' and t_in != 'electricity':
                flow = r[args]['sequences']
                for k, v in p[(args[1], None)]['scalars'].items():
                    if 'conversion_factors' in k and 'electricity' in k:
                        conversion_factor = v
            else:
                flow = pd.Series()
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

    if conversion_factor is not None:
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


def analyse_system_costs(es, plot_cost=False):
    back = namedtuple('res', ['levelized', 'meritorder', 'emission',
                              'emission_last', 'mo_full'])
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

    if plot_cost is True:
        sorted_flows = iter_res.misc
        print(c_values['trsf', 'pp'].sort_index(axis=1))
        plot.analyse_plot(c_values, mo_values, multi_res, sorted_flows[
            'demand_electricity'], sorted_flows['demand_heat'])

    return back(c_values.sum(axis=1), mo_values.max(axis=1),
                e_values.sum(axis=1), emission_last, mo_values)


def reshape_multiregion_df(es):
    """Remove"""
    res = results.get_multiregion_bus_balance(es).groupby(
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


def get_optional_export_emissions(es, label, bus_balnc):
    heat_blnc = results.get_multiregion_bus_balance(es, 'bus_heat')
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
    res = namedtuple('e', ('total', 'optional_export', 'optional_import',
                           'displaced', 'supplement'))
    local_es = results.load_es(fn_local)
    local_emissions = results.emissions(local_es)  # a)
    label = 'chp_ext'
    bus_balnc = results.get_multiregion_bus_balance(local_es)
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

    return res(total_local_emissions, emission_optional_export.sum(),
               emission_optional_import.sum(), displaced_emissions,
               supplement_emissions)


def get_fhg_ee_import_export(ie_df, start_dir):
    esys_files = []
    for root, directories, filenames in os.walk(start_dir):
        for fn in filenames:
            if fn[-5:] == '.esys':
                esys_files.append((root, fn))

    for root, fn in esys_files:
        solar = float(fn.split('_')[-3].replace('solar', ''))
        wind = float(fn.split('_')[-4].replace('wind', ''))
        name = 'pv{:02}_wind{:02}'.format(int(solar), int(wind))
        local_es = results.load_my_es(fn=os.path.join(root, fn))
        bus_balnc = results.get_multiregion_bus_balance(local_es)

        ie_df[name, 'import'] = bus_balnc[
            ('FHG', 'in', 'import', 'electricity', 'all')].reset_index(
                drop=True)
        ie_df[name, 'export'] = bus_balnc[
            ('FHG', 'out', 'export', 'electricity', 'all')].reset_index(
                drop=True)
    return ie_df


def analyse_ee_basic_core(res, upstream_values, flow):
    res['costs'] = flow.mul(upstream_values['meritorder'], axis=0).sum()
    res['emissions'] = flow.mul(upstream_values['emission_last'], axis=0).sum()
    res['energy'] = flow.sum()
    return res


def analyse_ee_basic(up_values=None):
    start_dir = os.path.join(cfg.get('paths', 'scenario'),
                             'friedrichshagen', 'ee_results')

    result = {
        'energy': pd.DataFrame(),
        'costs': pd.DataFrame(),
        'emissions': pd.DataFrame()}

    if up_values is None:
        upstream_values = fsc.load_upstream_scenario_values()
        up_scen = gui.get_choice(
            upstream_values.columns.get_level_values(0).unique(),
            "Upstream scenarios", "Select an upstream scenario")
        up_values = upstream_values[up_scen]

    fhg_ex_import = pd.DataFrame(
        columns=pd.MultiIndex(levels=[[], []], labels=[[], []]))

    ee_results_table = os.path.join(cfg.get('paths', 'friedrichshagen'),
                                    'fhg_ee_import_export.csv')
    if not os.path.isfile(ee_results_table):
        fhg_ex_import = get_fhg_ee_import_export(fhg_ex_import, start_dir)
        fhg_ex_import.to_csv(ee_results_table)
    else:
        fhg_ex_import = pd.read_csv(ee_results_table, index_col=[0],
                                    header=[0, 1])
    # print(fhg_ex_import)
    result = analyse_ee_basic_core(result, up_values, fhg_ex_import)

    return result


def plot_analyse_ee_basic():
    result = analyse_ee_basic()
    for key, value in result.items():
        value.sort_index(0, inplace=True)
        value.sort_index(1, inplace=True)
        value.plot(kind='bar', title=key)
    plt.show()


def multi_plot_analyse_ee_basic():
    fn = '/home/uwe/express/reegis/data/friedrichshagen/ups.csv'
    upstream_values = fsc.load_upstream_scenario_values(fn)
    f, ax_ar = plt.subplots(3, 3, sharey='row', sharex=True, figsize=(9, 6))
    col = 0
    my_list = list(upstream_values.columns.get_level_values(0).unique())
    my_list.remove('deflex_XX_Nc00_Li05_HP02_de21')
    for up_scen in my_list:
        row = 0
        print(up_scen)
        up_values = upstream_values[up_scen]
        result = analyse_ee_basic(up_values)
        for key, value in result.items():
            value = value.unstack()
            ax = ax_ar[row][col]
            value.sort_index(0, inplace=True)
            value.sort_index(1, inplace=True)
            value.plot(kind='bar', title=key, ax=ax)
            row += 1
        col += 1
    plt.show()
    exit(0)


# def get_resource_usage(es):
#     print(get_multiregion_bus_balance(es, 'bus_commodity').sum())

def analyse_DE01_basic():
    r = {}
    year = 2014
    scenarios = {
        'deflex_de22': {'es': ['deflex', str(year)],
                        'var': 'de22',
                        'region': 'DE22'},
        # 'berlin_de22': {'es': ['berlin_hp', str(year)],
        #                 'var': 'de22',
        #                 'region': 'BE'},
        # 'berlin_de21': {'es': ['berlin_hp', str(year)],
        #                 'var': 'de21',
        #                 'region': 'BE'},
        # 'berlin_up_de21': {'es': ['berlin_hp', str(year)],
        #                   'var': 'single_up_deflex_2014_de21',
        #                   'region': 'BE'},
        # 'berlin_single': {'es': ['berlin_hp', str(year)],
        #                   'var': 'single_up_None',
        #                   'region': 'BE'},
        # 'deflex_de22b': {'es': ['deflex', str(year), 'de22'],
        #                 'region': 'DE22'},
        # 'deflex_de22c': {'es': ['deflex', str(year), 'de22'],
        #                 'region': 'DE22'},
    }
    for k, v in scenarios.items():
        es = results.load_my_es(*v['es'], var=v['var'])

        result = es.results['Main']
        flows = [f for f in result.keys()]
        for f in flows:
            print(str(f[0].label))
        exit(0)
        lines_from = [x for x in flows
                      if (x[0].label.region == 'DE01') &
                      (x[0].label.cat == 'line')]
        for l in lines_from:
            key = l[0].label.subtag
            print(l[0].label)
            r['from_' + key] = result[l]['sequences']['flow'].multiply(-1)

        lines_to = [x for x in flows
                    if (x[1].label.subtag == 'DE01') &
                    (x[1].label.cat == 'line')]
        for l in lines_to:
            key = l[1].label.region
            print(l[1].label)
            r['to_' + key] = result[l]['sequences']['flow']
        # print(outputlib.views.node(results, ))
        # exit(0)
        df = pd.concat(r).swaplevel().unstack()
        df['DE13'] = df['from_DE13'] + df['to_DE13']
        print(df.sum())
        df[['DE13']].plot()
        plt.show()
        exit(0)


def analyse_fhg_basic():
    # upstream_es = load_es(2014, 'de21', 'deflex')
    path = os.path.join(
        cfg.get('paths', 'scenario'), 'friedrichshagen', 'ee_results')
    filename = gui.select_filename(work_dir=path,
                                   title='EE results!',
                                   extension='esys')
    print(filename)
    local_es = results.load_my_es(fn=os.path.join(path, filename))

    flh = results.fullloadhours(local_es)
    print(flh.loc[flh.nominal_value != 0])
    bus_balnc = results.get_multiregion_bus_balance(local_es)
    imp = bus_balnc[('FHG', 'in', 'import', 'electricity', 'all')]
    exp = bus_balnc[('FHG', 'out', 'export', 'electricity', 'all')]
    imp.name = 'import'
    exp.name = 'export'
    # print(bus_balnc.min())
    bus_balnc[bus_balnc < 0] = 0
    print(bus_balnc.min())
    print(bus_balnc.sum())
    # exit(0)
    heat_blnc = results.get_multiregion_bus_balance(local_es, 'bus_heat')

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
    plot.plot_bus_view(local_es, ax=ax)


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
        local_es = results.load_my_es(fn=os.path.join(
            cfg.get('paths', 'scenario'), 'friedrichshagen', 'Sonntag', fn))
        print(results.get_multiregion_bus_balance(local_es).sum())
        heat_blnc = results.get_multiregion_bus_balance(local_es, 'bus_heat')
        print(heat_blnc.sum())
        print(results.get_multiregion_bus_balance(
            local_es, 'bus_commodity').sum())
