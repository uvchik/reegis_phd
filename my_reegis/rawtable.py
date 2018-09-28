import os
import results
import pandas as pd
import my_reegis
from oemof.tools import logger
import berlin_hp
import deflex
import reegis_tools.config as cfg
from reegis_tools import inhabitants
from reegis_tools import bmwi

from berlin_hp import friedrichshagen


def table_model_regions(path, year=2014):
    table = pd.DataFrame(
        columns=pd.MultiIndex(levels=[[], []], labels=[[], []]))

    # table = pd.read_excel(
    #     os.path.join(path, 'kennzahlen_modellregionen' + '.xlsx'),
    #     index_col=[0], header=[0, 1])

    # inhabitants
    ew = inhabitants.get_ew_by_federal_states(year)
    ew_bln = friedrichshagen.calculate_inhabitants_districts(year)['EW'].sum()
    fhg_ew = friedrichshagen.calculate_inhabitants_friedrichshagen(year)
    ew_de01 = deflex.inhabitants.get_ew_by_deflex(2014, rmap='de21')['DE01']

    # electricity_demand
    fhg_elec = friedrichshagen.calculate_elec_demand_friedrichshagen(
        year).sum()
    bln_share = deflex.demand.openego_demand_share()['DE22']
    de01_share = deflex.demand.openego_demand_share()[['DE22', 'DE01']].sum()
    bln_usage = berlin_hp.electricity.get_electricity_demand(year).sum()[
        'usage']
    de_demand = bmwi.get_annual_electricity_demand_bmwi(2014) * 1000

    # heat demand
    bln_heat = berlin_hp.heat.create_heat_profiles(2014).sum().sum() / 1000
    fhg_heat = berlin_hp.heat.create_heat_profiles(
        2014, region=90517).sum().sum() / 1000
    heat_states = deflex.demand.get_heat_profiles_by_state(2014).groupby(
        level=0, axis=1).sum().sum().div(3.6)
    de01_heat = deflex.demand.get_heat_profiles_deflex(
        2014, separate_regions=['DE01'])['DE01'].sum().sum() / 1000

    sec = 'Bevölkerung'
    table.loc[sec, ('Berlin (deflex)', 'absolut')] = int(ew['BE'])
    table.loc[sec, ('Berlin', 'absolut')] = ew_bln
    table.loc[sec, ('Modellregion', 'absolut')] = int(fhg_ew)
    table.loc[sec, ('Deutschland', 'absolut')] = int(ew.sum())
    table.loc[sec, ('DE01 (de21)', 'absolut')] = int(ew_de01)

    sec = 'Strombedarf [GWh]'
    table.loc[sec, ('Berlin (deflex)', 'absolut')] = int(bln_share * de_demand)
    table.loc[sec, ('Berlin', 'absolut')] = bln_usage
    table.loc[sec, ('Modellregion', 'absolut')] = int(fhg_elec.sum())
    table.loc[sec, ('Deutschland', 'absolut')] = int(de_demand)
    table.loc[sec, ('DE01 (de21)', 'absolut')] = int(de01_share * de_demand)

    sec = 'Wärmebedarf [GWh]'
    table.loc[sec, ('Berlin (deflex)', 'absolut')] = int(heat_states['BE'])
    table.loc[sec, ('Berlin', 'absolut')] = int(bln_heat)
    table.loc[sec, ('Modellregion', 'absolut')] = int(fhg_heat)
    table.loc[sec, ('Deutschland', 'absolut')] = int(heat_states.sum())
    table.loc[sec, ('DE01 (de21)', 'absolut')] = int(de01_heat)

    for c in table.columns.get_level_values(0).unique():
        table[c, '%'] = round(table[c, 'absolut'].div(
                table['Deutschland', 'absolut']).multiply(100), 2)

    table = table[['Modellregion', 'Berlin', 'Berlin (deflex)', 'DE01 (de21)',
                   'Deutschland']]
    print(table)
    table.to_csv(os.path.join(path, 'kennzahlen_modellregionen' + '.csv'))


def berlin_ressources(path):
    df = results.analyse_berlin_ressources_total()
    df = pd.concat([df], keys=['[TWh]'])
    for c in df.index.get_level_values(1):
        df.loc[('%', c), slice(None)] = round(
            df.loc[('[TWh]', c), slice(None)].div(
                df.loc[('[TWh]', 'statistic'), slice(None)]) * 100, 2)
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(path, 'berlin_resource_usage' + '.csv'))


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(berlin_hp.__file__),
                    os.path.dirname(my_reegis.__file__),
                    os.path.dirname(deflex.__file__)])
    p = '/home/uwe/git_local/monographie/tables/'
    berlin_ressources(p)
    table_model_regions(p)
