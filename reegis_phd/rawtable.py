import os
# from my_reegis import results
import pandas as pd
# import my_reegis
from oemof.tools import logger
# import berlin_hp
# import deflex
import reegis as cfg
from reegis import inhabitants, openego as oego, bmwi

# from berlin_hp import friedrichshagen


def table_demand_federal_states(outpath):
    ego_year = 2014

    path = cfg.get('paths', 'data_my_reegis')
    filename = 'lak_electricity_demand_federal_states.csv'
    lak = pd.read_csv(os.path.join(path, filename), index_col=[0, 1], sep=';')
    lak = lak.swaplevel()
    ego = oego.get_ego_demand_by_federal_states(bmwi=False).groupby(
        'federal_states').sum()['consumption']
    ego_sum = ego.sum()

    lak = lak.rename(index=cfg.get_dict('STATES'))
    lak['Strom'] = pd.to_numeric(lak['Strom'], errors='coerce')

    new_table = pd.DataFrame(
        columns=pd.MultiIndex(levels=[[], []], codes=[[], []]),
        index=ego.index)

    new_table[('openEgo', ego_year)] = ego

    print(new_table)

    for y in [2010, 2011, 2012, 2013, 2014]:
        key = ('lak', y)
        new_table[key] = lak.loc[y]['Strom'] / 3.6
        f = new_table[key].sum() / ego_sum
        new_table[('frac', y)] = (((new_table[('openEgo', ego_year)] * f) -
                                  new_table[key]) / new_table[key])

    fk = new_table[('lak', ego_year)].sum() / ego_sum
    new_table[('openEgo', ego_year)] = new_table[('openEgo', ego_year)] * fk
    new_table.sort_index(axis=1, inplace=True)
    new_table.to_excel(
        os.path.join(outpath, 'federal_states_demand_ego_lak.xls'))
    print(new_table)


def table_model_regions(path, year=2014):
    table = pd.DataFrame(
        columns=pd.MultiIndex(levels=[[], []], codes=[[], []]))

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
    # cfg.init(paths=[os.path.dirname(berlin_hp.__file__),
    #                 os.path.dirname(my_reegis.__file__),
    #                 os.path.dirname(deflex.__file__)])
    p = '/home/uwe/'#chiba/Promotion/Monographie/tables'
    table_demand_federal_states(p)
    exit(0)
    berlin_ressources(p)
    table_model_regions(p)
