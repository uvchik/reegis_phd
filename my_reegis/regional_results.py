import os
import logging
import pandas as pd
import berlin_hp
from oemof import solph
from oemof.tools import logger
from my_reegis import results
from my_reegis import friedrichshagen_scenarios as fsc
from reegis import config as cfg
from matplotlib import pyplot as plt


def collect_berlin_ressource_data(year=2014):

    scenarios = {
        'deflex_de22': {'es': ['deflex', str(year)],
                        'var': 'de22',
                        'region': 'DE22'},
        'berlin_deflex': {'es': ['berlin_hp', str(year)],
                          'var': 'de22',
                          'region': 'BE'},
        'berlin_up_deflex': {
            'es': ['berlin_hp', str(year)],
            'var': 'single_up_deflex_2014_de22_without_berlin',
            'region': 'BE'},
        'berlin_up_deflex_full': {
            'es': ['berlin_hp', str(year)],
            'var': 'single_up_deflex_2014_de22',
            'region': 'BE'},
        'berlin_single': {'es': ['berlin_hp', str(year)],
                          'var': 'single_up_None',
                          'region': 'BE'}
        }
    df = pd.DataFrame(columns=pd.MultiIndex(levels=[[], [], []],
                                            codes=[[], [], []]),
                      index=['bioenergy', 'hard_coal', 'lignite',
                             'natural_gas', 'oil', 'waste', 'other', 're'])

    ms = pd.Series(index=pd.MultiIndex(levels=[[], []],
                                       codes=[[], []]))

    for k, v in scenarios.items():
        logging.info("Process: {0}".format(k))
        es = results.load_my_es(*v['es'], var=v['var'])

        resource_balance = results.get_multiregion_bus_balance(
            es, 'bus_commodity')

        buses = [b for b in es.results['Main'] if
                 isinstance(b[0], solph.Bus) &
                 (b[0].label.tag == 'commodity') &
                 (b[0].label.region == v['region'])]

        for outflow in buses:
            lab = outflow[1].label
            f = float(es.results['Main'][outflow]['sequences'].sum())
            ms[k, str(lab)] = f

        heat = results.get_multiregion_bus_balance(es, 'district')

        elec = results.get_multiregion_bus_balance(es).sum()
        print(elec.groupby(level=[0, 1, 2, 3]).sum())

        df[k, 'commodity', 'in'] = (
            resource_balance[v['region'], 'in', 'source', 'commodity'].sum())

        for t in ['chp', 'pp', 'hp', 'trsf']:
            if 'de22' not in k:
                df[k, 'commodity', t] = (
                    resource_balance[v['region'], 'out', t].sum().groupby(
                        level=1).sum())

                if t in ['hp', 'chp']:
                    df[k, 'to_heat', t] = (
                        heat.groupby(axis=1, level=[1, 2, 3, 4]).sum().sum()
                            .loc['in', t].groupby(level=1).sum())
                if t in ['pp', 'chp']:
                    df[k, 'to_elec', t] = (
                        elec.loc[v['region'], 'in', t].groupby(level=1).sum())

            else:
                if t == 'trsf':
                    t = 'heat'
                df[k, 'commodity', t] = (
                    resource_balance[v['region'], 'out', 'trsf', t].sum())
                if t in heat[v['region'], 'in', 'trsf']:
                    df[k, 'to_heat', t] = (
                        heat[v['region'], 'in', 'trsf', t].sum())
                if t in elec.loc[v['region'], 'in', 'trsf']:
                    df[k, 'to_elec', t] = (
                        elec.loc[v['region'], 'in', 'trsf', t])

    df.div(1000).sort_index(axis=1).to_excel('/home/uwe/temp_BE.xlsx')
    ms.div(1000).sort_index().to_excel('/home/uwe/temp_BE2.xlsx')


def analyse_berlin_ressources():
    s = {}
    year = 2014
    scenarios = {
        'berlin_single': {
            'path': ['berlin_hp', str(year)],
            'file': 'berlin_hp_2014_single_up_None',
            'var': None,
            'region': 'BE'},
        'berlin_deflex': {
            'path': ['berlin_hp', str(year)],
            'file': 'berlin_hp_2014_de22',
            'var': 'de22',
            'region': 'BE'},
        'berlin_up_deflex': {
            'path': ['berlin_hp', str(year)],
            'file': 'berlin_hp_2014_single_up_de22_csv_without_DE22',
            'var': 'single_up_deflex_2014_de22_without_berlin',
            'region': 'BE'},
        'berlin_up_deflex_full': {
            'path': ['berlin_hp', str(year)],
            'file': 'berlin_hp_2014_single_up_de22',
            'var': 'single_up_deflex_2014_de22',
            'region': 'BE'},
        'deflex_de22': {
            'path': ['deflex', str(year)],
            'file': 'deflex_2014_de22',
            'var': 'de22',
            'region': 'DE22'},
        'deflex_de22_neu': {
            'path': ['deflex', str(year)],
            'file': 'deflex_neu_2014_de22',
            'var': 'de22',
            'region': 'DE22'},
        # 'deflex_de22_new': {
        #     'path': ['deflex', str(year)],
        #     'file': 'deflex_new_2014_de22',
        #     'var': 'de22',
        #     'region': 'DE22'},
        }

    for k, v in scenarios.items():
        if cfg.has_option('results', 'dir'):
            res_dir = cfg.get('results', 'dir')
        else:
            res_dir = 'results'
        path = os.path.join(cfg.get('paths', 'scenario'), *v['path'], res_dir)
        fn = os.path.join(path, v['file'] + '.esys')
        es = results.load_es(fn)
        results.check_excess_shortage(es)
        resource_balance = results.get_multiregion_bus_balance(
            es, 'bus_commodity')

        s[k] = resource_balance[
            v['region'], 'in', 'source', 'commodity'].copy()

        if v['var'] is not None:
            elec_balance = results.get_multiregion_bus_balance(es)
            s[k]['ee'] = elec_balance[v['region'], 'in', 'source', 'ee'].sum(
                axis=1)

            import_berlin = elec_balance[
                v['region'], 'in', 'import', 'electricity', 'all']
            export_berlin = elec_balance[
                v['region'], 'out', 'export', 'electricity', 'all']
            s[k]['netto_import'] = import_berlin - export_berlin

        else:
            s[k]['netto_import'] = 0

    seq = pd.concat(s, axis=1).div(1000000)

    for scenario in seq.columns.get_level_values(0).unique():
        seq[scenario, 'other'] = seq[scenario].get('other', 0)
        for c in ['waste', 'bioenergy', 'ee']:
            seq[scenario, 'other'] += seq[scenario].get(c, 0)

    for c in ['waste', 'bioenergy', 'ee']:
        seq.drop(c, level=1, inplace=True, axis=1)

    return seq.swaplevel(axis=1)


def analyse_berlin_ressources_total():
    # Energiebilanz Berlin 2014
    statistic = pd.Series({
        'bioenergy': 7152000,
        'hard_coal': 43245000,
        'lignite': 12274000,
        'natural_gas': 80635000,
        'oil': 29800000,
        'other': 477000,
        'ee': 337000,
        'netto_import': 19786000}).div(3.6)

    statistic['other'] = 0

    for c in ['bioenergy', 'ee']:
        statistic['other'] += statistic[c]
        del statistic[c]

    seq = analyse_berlin_ressources()
    df = seq.sum().unstack().T
    df.loc['statistic'] = statistic.div(1000000)
    return df.fillna(0)


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

    reg_emissions = results.emissions(es).loc[
        results.emissions(es)['summed_flow'] > 0]

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


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__)])
    # analyse_ee_basic()
    # plot_analyse_ee_basic()
    # multi_plot_analyse_ee_basic()
    my_es = results.load_my_es('deflex', '2014', var='de21')
    exit(0)
    my_sres = something()
    sort_col = ['upstream', 'fuel', 'region']  # 'export avg costs'
    sort_idx = my_sres['meta'].sort_values(sort_col).index
    for my_df in ['avg_costs', 'absolute_costs', 'absolute_flows',
                  'absolute_emissions']:
        my_sres[my_df].loc[sort_idx].plot(kind='bar')
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
