import os
import logging
from my_reegis import results
import pandas as pd
import reegis.config as cfg


OFFSHORE = {'de02': ['DE02'],
            'de17': ['DE17'],
            'de21': ['DE19', 'DE20', 'DE21'],
            'de22': ['DE19', 'DE20', 'DE21']
            }


def fetch_duals_max_min(es, var=None):
    """Fetch the duals and add a max/min column excluding the offshore buses.

    Returns
    -------
    pd.DataFrame
    """
    df = results.get_electricity_duals(es)

    if var is None:
        var = es.var.split('_')[0]
    else:
        var = var.split('_')[0]

    # remove offshore
    for cl in OFFSHORE[var]:
        del df[cl]
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    if round((df['max'] - df['min']).sum(), 1) == 0:
        df['uniform'] = df['max']
    else:
        df['uniform'] = float('nan')
    return df


def get_market_clearing_price(es, with_chp=False):
    parameter = results.fetch_cost_emission(es, with_chp=with_chp)
    my_results = es.results['main']

    # Filter all flows
    flows = [x for x in my_results if x[1] is not None]
    flows_to_elec = [x for x in flows if x[1].label.cat == 'bus' and
                     x[1].label.tag == 'electricity']

    # Filter tags and categories that should not(!) be considered
    tag_list = ['ee']
    cat_list = ['line', 'storage', 'shortage']
    if with_chp is False:
        tag_list.append('chp')
    flows_trsf = [x for x in flows_to_elec if x[0].label.cat not in cat_list
                  and x[0].label.tag not in tag_list]

    # Filter unused flows
    flows_not_null = [x for x in flows_trsf if sum(
        my_results[x]['sequences']['flow']) > 0]

    # Create merit order for each time step
    merit_order = pd.DataFrame()
    for flow in flows_not_null:
        seq = my_results[flow]['sequences']['flow']
        merit_order[flow[0]] = seq * 0
        lb = flow[0].label
        var_costs = parameter.loc[(lb.subtag, lb.region, lb.tag), 'var_costs']
        merit_order[flow[0]].loc[seq > 0] = var_costs
    merit_order['max'] = merit_order.max(axis=1)

    return merit_order['max']


def fetch_scenarios(path, solver, year=None):
    esys_files = []
    for root, directories, filenames in os.walk(path):
        for fn in filenames:
            if fn[-5:] == '.esys':
                if year is not None and str(year) in fn and solver in root:
                    esys_files.append((root, fn))
                elif year is None and solver in path:
                    esys_files.append((root, fn))
    return esys_files


def fetch_mcp_for_all_scenarios(path, solver, year=None):
    mcp = {}
    df = pd.DataFrame(columns=pd.MultiIndex(
        levels=[[], [], []], codes=[[], [], []]))
    scenarios = fetch_scenarios(path, solver, year=year)
    logging.info("{0} sceanrios found".format(len(scenarios)))
    for root, fn in scenarios:
        es = results.load_es(os.path.join(root, fn))
        name = fn.split('_')
        base = '_'.join([name[0], solver])
        year = int(name[1])
        var = '_'.join(name[2:]).split('.')[0]
        if year not in mcp:
            mcp[year] = df.copy()
        mcp[year][base, var, 'mcp'] = get_market_clearing_price(es)
        mcp[year][base, var, 'duals'] = fetch_duals_max_min(es, var)['uniform']
        mcp[year].sort_index(axis=1, inplace=True)
    return mcp


def get_mcp_for_all_scenarios(path, solver, year, overwrite=False):
    if path is None:
        path = os.path.join(cfg.get('paths', 'scenario'), 'deflex')
    fn = dump_mcp_for_all_scenarios(path, solver, year=year,
                                    overwrite=overwrite)
    return pd.read_csv(fn.format(year, solver), index_col=[0],
                       header=[0, 1, 2])


def dump_mcp_for_all_scenarios(path, solver, year=None, overwrite=False):
    filename = 'market_clearing_price_{0}_{1}.csv'
    if year is not None:
        if not os.path.isfile(os.path.join(path, filename.format(
                year, solver))) or overwrite:
            mcp = fetch_mcp_for_all_scenarios(path, solver, year)
        else:
            mcp = {}
    else:
        mcp = fetch_mcp_for_all_scenarios(path, solver)
    for year, df in mcp.items():
        df.to_csv(os.path.join(path, filename.format(year, solver)))
    return os.path.join(path, filename)


def get_upstream_set(solver, year, method, overwrite=True):
    df = get_mcp_for_all_scenarios(None, solver, year, overwrite=overwrite)
    base = 'deflex_{0}'.format(solver)
    return df.swaplevel(axis=1).sort_index(1)[base, method]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    get_upstream_set('cbc', 2014, 'mcp', overwrite=False).plot()
    plt.show()
