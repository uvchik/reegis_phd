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
        tmp_es = solph.EnergySystem()
        tmp_es.restore(root, fn)
        for val in ['levelized', 'meritorder', 'emission']:
            upstream_values[fn[:-5], val] = (
                getattr(results.analyse_system_costs(tmp_es), val))
        number -= 1
    upstream_values.to_csv(full_file_name)


def load_upstream_scenario_values():
    path = cfg.get('paths', 'friedrichshagen')
    filename = 'upstream_scenario_values.csv'
    full_file_name = os.path.join(path, filename)
    if not os.path.isfile(full_file_name):
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


def add_import_export(nodes, cost_scenario='no_costs', value=None):
    if cost_scenario != 'no_costs':
        upstream = load_upstream_scenario_values()
        export_costs = upstream[cost_scenario][value] * -0.99
        import_costs = upstream[cost_scenario][value] * 1.01
    else:
        export_costs = 0
        import_costs = 100

    elec_bus_label = Label('bus', 'electricity', 'all', 'FHG')

    exp_label = Label('export', 'electricity', 'all', 'FHG')
    nodes[exp_label] = solph.Sink(
                label=exp_label,
                inputs={nodes[elec_bus_label]: solph.Flow(
                    variable_costs=export_costs)})

    imp_label = Label('import', 'electricity', 'all', 'FHG')
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


def add_volatile_sources(nodes):
    ee_sources = {k: v for k, v in nodes.items() if v.label.tag == 'ee'}

    add_capacity = {'solar': 5,
                    'wind': 20}

    for label in ee_sources.keys():
        flow = next(iter(nodes[label].outputs.values()))
        flow.nominal_value += add_capacity[label.subtag.lower()]
    return nodes


def choose_pp(nodes, fuel):
    chp_trsf = {k: v for k, v in nodes.items() if v.label.cat == 'chp'
                and v.label.tag == 'ext' and not v.label.subtag == fuel}
    for label in chp_trsf.keys():
        flow = next(iter(nodes[label].inputs.values()))
        flow.nominal_value = 0

    return nodes


def adapted(year, name, cost_scenario, cost_value, add_wp, add_bio, pp,
            overwrite=False):
    stopwatch()
    base_name = '{0}_{1}_{2}'.format('friedrichshagen', year, 'single')
    name = '{0}_{1}_{2}'.format('friedrichshagen', year, name)
    sc = berlin_hp.Scenario(name=name, year=year, debug=False)

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
    if add_wp:
        nodes = add_dectrl_wp(nodes)
    nodes = add_import_export(nodes, cost_scenario, value=cost_value)
    if add_bio:
        nodes = add_bio_powerplant(nodes)

    nodes = choose_pp(nodes, pp)

    nodes = add_volatile_sources(nodes)

    sc.es = sc.initialise_energy_system()
    sc.es.add(*nodes.values())

    # Save energySystem to '.graphml' file.
    sc.plot_nodes(filename=os.path.join(path, 'friedrichshagen'),
                  remove_nodes_with_substrings=['bus_cs'])

    main.compute(sc)


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(deflex.__file__),
                    os.path.dirname(berlin_hp.__file__)])
    stopwatch()
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
    logging.info("Done: {0}".format(stopwatch()))
