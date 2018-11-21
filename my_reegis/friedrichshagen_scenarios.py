import os
import logging
from datetime import datetime
from shutil import copyfile

import pandas as pd

import reegis_tools.config as cfg
from reegis_tools.scenario_tools import Label

import oemof.tools.logger as logger
from oemof import solph

import berlin_hp
import deflex

from my_reegis import results
from my_reegis import main
import multiprocessing


def stopwatch():
    if not hasattr(stopwatch, 'start'):
        stopwatch.start = datetime.now()
    return str(datetime.now() - stopwatch.start)[:-7]


def fetch_esys_files():
    esys_files = []
    for root, directories, filenames in os.walk(cfg.get('paths', 'scenario')):
        for fn in filenames:
            if (fn[-5:] == '.esys' and
                    'single' not in fn and
                    'friedrichshagen' not in fn):
                esys_files.append((root, fn))
    return esys_files


def fetch_upstream_overall_values(full_file_name):
    esys_files = fetch_esys_files()
    idx = pd.MultiIndex(levels=[[], []], labels=[[], []])
    upstream_values = pd.Series(index=idx)
    number = len(esys_files)
    ref_es = solph.EnergySystem()
    ref_es.restore('/home/uwe/express/reegis/scenarios/new/results',
                   'deflex_XX_Nc00_Li05_HP00_GT_de21.esys')
    reference = results.emissions(ref_es)
    del reference['specific_emission']
    for col in reference.columns:
        reference.rename(columns={col: 'ref_' + col}, inplace=True)
    for root, fn in esys_files:
        logging.info("Remaining scenarios: {0}".format(number))
        tmp_es = solph.EnergySystem()
        tmp_es.restore(root, fn)
        emsn = results.emissions(tmp_es)
        del emsn['specific_emission']
        tmp_series = pd.concat([emsn, reference], axis=1).sum()
        for i in tmp_series.index:
            upstream_values[fn[:-5], i] = float(tmp_series[i])
        number -= 1
    upstream_values.name = 'overall'
    upstream_values.to_csv(full_file_name)


def fetch_upstream_scenario_values(full_file_name):
    esys_files = fetch_esys_files()
    idx = pd.MultiIndex(levels=[[], []], labels=[[], []])
    upstream_values = pd.DataFrame(columns=idx)
    number = len(esys_files)
    for root, fn in esys_files:
        logging.info("Remaining scenarios: {0}".format(number))
        logging.info("Analyse scenario: {0}".format(fn))
        tmp_es = solph.EnergySystem()
        tmp_es.restore(root, fn)
        for val in ['levelized', 'meritorder', 'emission', 'emission_last']:
            upstream_values[fn[:-5], val] = (
                getattr(results.analyse_system_costs(tmp_es), val))
        number -= 1
    upstream_values.to_csv(full_file_name)


def load_upstream_scenario_values(fn=None, overwrite=False):
    if fn is None:
        path = cfg.get('paths', 'friedrichshagen')
        filename = 'upstream_scenario_values.csv'
        full_file_name = os.path.join(path, filename)
    else:
        full_file_name = fn

    if not os.path.isfile(full_file_name) or overwrite:
        fetch_upstream_scenario_values(full_file_name)
    return pd.read_csv(
        full_file_name, index_col=[0], header=[0, 1]).sort_index()


def load_upstream_overall_values():
    path = cfg.get('paths', 'friedrichshagen')
    filename = 'upstream_overall_values.csv'
    full_file_name = os.path.join(path, filename)
    if not os.path.isfile(full_file_name):
        fetch_upstream_overall_values(full_file_name)
    return pd.read_csv(
        full_file_name, index_col=[0, 1], squeeze=True, header=None).sort_index()


def remove_shortage_excess_electricity(nodes):
    elec_nodes = [v for k, v in nodes.items() if v.label.tag == 'electricity']
    for v in elec_nodes:
        if v.label.cat == 'excess':
            flow = next(iter(nodes[v.label].inputs.values()))
            flow.nominal_value = 0
        elif v.label.cat == 'shortage':
            flow = next(iter(nodes[v.label].outputs.values()))
            flow.nominal_value = 0


def ee_invest_nodes(nodes):
    ee_sources = {k: v for k, v in nodes.items() if v.label.tag == 'ee'}
    av = {}
    for label in ee_sources.keys():
        flow = next(iter(nodes[label].outputs.values()))
        av[label.subtag] = flow.actual_value

    elec_bus_label = Label('bus', 'electricity', 'all', 'FHG')
    for t in av.keys():
        ee_label = Label('invest_source', 'ee', t, 'FHG')
        print(ee_label.subtag, av[ee_label.subtag].sum())
        nodes[ee_label] = solph.Source(
            label=ee_label,
            outputs={nodes[elec_bus_label]: solph.Flow(
                actual_value=av[ee_label.subtag],
                fixed=True,
                investment=solph.Investment(ep_costs=10))})
    return add_storage(nodes)


def add_storage(nodes):
    elec_bus_label = Label('bus', 'electricity', 'all', 'FHG')
    storage_label = Label('storage', 'electricity', 'no_losses', 'FHG')
    nodes[storage_label] = solph.components.GenericStorage(
        label=storage_label,
        inputs={nodes[elec_bus_label]: solph.Flow(variable_costs=10)},
        outputs={nodes[elec_bus_label]: solph.Flow()},
        capacity_loss=0.00, initial_capacity=0,
        invest_relation_input_capacity=1,
        invest_relation_output_capacity=1,
        inflow_conversion_factor=1, outflow_conversion_factor=1,
        investment=solph.Investment(ep_costs=10),
    )
    return nodes
    # pprint.pprint(shortage)


def add_import_export(nodes, cost_scenario='no_costs', value=None,
                      region='FHG'):

    if cost_scenario == 'no_costs':
        export_costs = -0.000001
        import_costs = 500

    elif cost_scenario == 'ee':
        remove_shortage_excess_electricity(nodes)
        export_costs = 5000
        import_costs = 5000

    else:
        upstream = load_upstream_scenario_values()
        export_costs = upstream[cost_scenario][value] * -0.99
        import_costs = upstream[cost_scenario][value] * 1.01

    elec_bus_label = Label('bus', 'electricity', 'all', region)

    exp_label = Label('export', 'electricity', 'all', region)
    nodes[exp_label] = solph.Sink(
                label=exp_label,
                inputs={nodes[elec_bus_label]: solph.Flow(
                    variable_costs=export_costs)})

    imp_label = Label('import', 'electricity', 'all', region)
    nodes[imp_label] = solph.Source(
                label=imp_label,
                outputs={nodes[elec_bus_label]: solph.Flow(
                    variable_costs=import_costs)})
    return nodes


def add_bio_powerplant(nodes):
    biogas_label = Label('pp', 'fix', 'bioenergy', 'FHG')
    elec_bus_label = Label('bus', 'electricity', 'all', 'FHG')

    nodes[biogas_label] = solph.Source(
                label=biogas_label,
                outputs={nodes[elec_bus_label]: solph.Flow(
                    fixed=True, actual_value=1, nominal_value=2)})
    return nodes


def add_dectrl_wp(nodes, frac_wp=0.2):
    dec_heat_demand = {k: v for k, v in nodes.items() if k.cat == 'demand' and
                       k.tag == 'heat' and k.region == 'decentralised_BE'}

    dectrl_demand = pd.DataFrame()
    for label in dec_heat_demand:
        flow = next(iter(nodes[label].inputs.values()))
        dectrl_demand[label.subtag] = flow.actual_value

    frac = dectrl_demand.sum().div(dectrl_demand.sum().sum()).multiply(
        1 - frac_wp)
    frac['elec'] = frac_wp
    total_demand = dectrl_demand.sum(axis=1)

    for label in dec_heat_demand:
        flow = next(iter(nodes[label].inputs.values()))
        flow.actual_value = frac[label.subtag] * total_demand

    for label in dec_heat_demand:
        flow = next(iter(nodes[label].inputs.values()))
        dectrl_demand[label.subtag] = flow.actual_value
    return nodes


def add_volatile_sources(nodes, add_capacity):
    """add_capacity = {'solar': 5,
                       'wind': 20}
    """

    ee_sources = {k: v for k, v in nodes.items() if v.label.tag == 'ee' and
                  v.label.cat == 'source'}
    if add_capacity.get('set', False) is True:
        for label in ee_sources.keys():
            if label.subtag.lower() in add_capacity:
                flow = next(iter(nodes[label].outputs.values()))
                flow.nominal_value = add_capacity[label.subtag.lower()]
    else:
        for label in ee_sources.keys():
            if label.subtag.lower() in add_capacity:
                flow = next(iter(nodes[label].outputs.values()))
                flow.nominal_value += add_capacity[label.subtag.lower()]

    return nodes


def deactivate_fix_pp(nodes):
    fix_trsf = [k for k, v in nodes.items() if v.label.cat == 'chp'
                and v.label.tag == 'fix'
                and v.label.subtag == 'natural_gas']
    flow = next(iter(nodes[fix_trsf[0]].inputs.values()))
    flow.nominal_value = 0
    return nodes


def choose_pp(nodes, fuel):
    chp_trsf = {k: v for k, v in nodes.items() if v.label.cat == 'chp'
                and v.label.tag == 'ext' and not v.label.subtag == fuel}
    for label in chp_trsf.keys():
        flow = next(iter(nodes[label].inputs.values()))
        flow.nominal_value = 0

    return nodes


def adapted(year, name, cost_scenario, cost_value, add_wp, add_bio, pp, fix_pp,
            volatile_src=None, ee_invest='ee_invest0', overwrite=False):
    if name is None:
        vs_str = 'vs'
        if volatile_src is not None:
            for k, v in volatile_src.items():
                vs_str += '_{0}{1}'.format(k, v)

        name = '_'.join([cost_value, cost_scenario, add_wp, add_bio, pp,
                         fix_pp, vs_str, ee_invest])

    stopwatch()
    base_name = '{0}_{1}_{2}'.format('friedrichshagen', year, 'single')
    scenario_name = '{0}_{1}_{2}'.format('friedrichshagen', year, name)
    sc = berlin_hp.Scenario(name=scenario_name, year=year, debug=False)

    path = os.path.join(cfg.get('paths', 'scenario'), 'friedrichshagen')

    logging.info("Read scenario from excel-sheet: {0}".format(stopwatch()))
    excel_fn = os.path.join(path, base_name + '.xls')

    if not os.path.isfile(excel_fn) or overwrite:
        berlin_hp.friedrichshagen.create_basic_scenario(year, excel=excel_fn)

    os.makedirs(os.path.join(path, 'results'), exist_ok=True)
    src = os.path.join(path, '{0}.xls'.format(base_name))
    dst = os.path.join(path, 'results', '{0}.xls'.format(base_name))
    copyfile(src, dst)

    sc.load_excel(excel_fn)
    sc.check_table('time_series')

    logging.info("Add nodes to the EnergySystem: {0}".format(stopwatch()))

    nodes = sc.create_nodes(region='FHG')
    if ee_invest == 'ee_invest1':
        nodes = ee_invest_nodes(nodes)

    if add_wp == 'wp02':
        nodes = add_dectrl_wp(nodes)
    nodes = add_import_export(nodes, cost_scenario, value=cost_value)

    if add_bio == 'bio1':
        nodes = add_bio_powerplant(nodes)

    if fix_pp == 'fix0':
        nodes = deactivate_fix_pp(nodes)

    nodes = choose_pp(nodes, pp)

    if volatile_src is not None:
        nodes = add_volatile_sources(nodes, volatile_src)

    sc.es = sc.initialise_energy_system()
    sc.es.add(*nodes.values())

    # Save energySystem to '.graphml' file.
    sc.plot_nodes(filename=os.path.join(path, 'friedrichshagen'),
                  remove_nodes_with_substrings=['bus_cs'])

    main.compute(sc)


def compute_adapted_scenarios():
    up_scenarios = list(
        load_upstream_scenario_values().columns.get_level_values(0).unique())
    up_scenarios.append('no_costs')
    nr = 0
    scenarios = {}
    for cs in up_scenarios:
        for cv in ['meritorder']:
            for wp in ['wp02', 'wp00']:
                for bio in ['bio1', 'bio0']:
                    for pp in ['hard_coal', 'natural_gas']:
                        sc_text = '_'.join([cv, cs, wp, bio, pp])
                        scenarios[nr] = {
                            'name': sc_text,
                            'cost_scenario': cs,
                            'cost_value': cv,
                            'add_wp': wp,
                            'add_bio': bio,
                            'pp': pp,
                            'year': 2014}
                        nr += 1

    sl = sorted(scenarios.keys(), reverse=True)
    for n in sl:
        scenario = scenarios[n]
        logging.info("Start scenario {0}: {1}".format(
            scenario['name'], stopwatch()))
        adapted(**scenario)
        logging.info("{0} scenarios left".format(n))


def basic_chp_extension_scenario(scen):
    chp_fuel = scen[0]
    cost_scenario = scen[1]
    scenario = {
        'name': None,
        'cost_scenario': cost_scenario,
        'cost_value': 'meritorder',
        'fix_pp': 'fix0',
        'add_wp': 'wp00',
        'add_bio': 'bio0',
        'pp': chp_fuel,
        'year': 2014,
        'volatile_src': {'wind': 0}}

    adapted(**scenario)


def basic_ee_wrapper(scenario):
    adapted(**scenario)


def basic_ee_scenario():
    pv_flh = 895.689228779882
    wind_flh = 1562.51612773669
    demand = 60586.6486548771

    scenario_list = []
    for frac in range(11):
        pv = demand / pv_flh * frac / 10
        wind = demand / wind_flh * (1 - frac / 10)

        scenario = {
            'name': None,
            'cost_scenario': 'ee',
            'cost_value': 'meritorder',
            'fix_pp': 'fix0',
            'add_wp': 'wp00',
            'add_bio': 'bio0',
            'pp': 'chp_fuel',
            'year': 2014,
            'ee_invest': 'ee_invest0',
            'volatile_src': {'wind': wind, 'solar': pv, 'set': True}}
        scenario_list.append(scenario)

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.map(basic_ee_wrapper, scenario_list)
    p.close()
    p.join()


def my_scenarios():
    up_scenarios = list(
        load_upstream_scenario_values().columns.get_level_values(0).unique())
    # up_scenarios = list()
    up_scenarios.append('no_costs')
    fuels = ['hard_coal', 'natural_gas']
    my_list = []
    for upc in up_scenarios:
        for f in fuels:
            my_list.append((f, upc))

    # length = len(my_list)
    # for scenario in my_list:
    #     basic_chp_extension_scenario(
    #                             args=(scenario[0], scenario[1])).start()
    #     logging.info("Rest: {0}".format(length))
    #     length -= 1

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p.map(basic_chp_extension_scenario, my_list)
    p.close()
    p.join()


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])
    stopwatch()
    load_upstream_scenario_values(overwrite=True)
    print(fetch_esys_files())
    exit(0)
    # load_upstream_scenario_values().columns.get_level_values(0).unique()
    basic_ee_scenario()
    # my_scenarios()
    # basic_chp_extension_scenario('hard_coal')
    #
    # for fuel in ['hard_coal', 'natural_gas']:
    #     Process(target=basic_chp_extension_scenario,
    #             args=(fuel, cost_scenario)).start()

    # compute_adapted_scenarios()
    logging.info("Done: {0}".format(stopwatch()))
