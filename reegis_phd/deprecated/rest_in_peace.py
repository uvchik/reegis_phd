

def get_one_node_type(es, node=None):
    mr_df = get_multiregion_bus_balance(es).groupby(
        axis=1, level=[1, 2, 3, 4]).sum()
    print(mr_df)

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








# # analyse_system_costs(plot=True)
# # my_es = load_es(2014, 'deflex_2014_de21', 'friedrichshagen')
#
# exit(0)
# # ebus_seq = outputlib.views.node(my_es.results['Main'],
# #                                 'bus_electricity_all_FHG')['sequences']
# # print(ebus_seq.sum())
# #
# # ebus = my_es.groups['bus_electricity_all_FHG']
# # export_node = my_es.groups['export_electricity_all_FHG']
# # import_node = my_es.groups['import_electricity_all_FHG']
# # export_costs = my_es.flows()[(ebus, export_node)].variable_costs / 0.99
# # import_costs = my_es.flows()[(import_node, ebus)].variable_costs / 1.01
# # export_flow = my_es.results['Main'][(ebus, export_node)]['sequences']
# # import_flow = my_es.results['Main'][(import_node, ebus)]['sequences']
# #
# # export_flow.reset_index(drop=True, inplace=True)
# # import_flow.reset_index(drop=True, inplace=True)
# #
# # total_export_costs = export_flow.multiply(export_costs, axis=0)
# # total_import_costs = import_flow.multiply(import_costs, axis=0)
# #
# # print(total_import_costs.sum() / import_flow.sum())
# # print(total_export_costs.sum() / export_flow.sum() * -1)
# #
# # ax = total_export_costs.plot()
# # total_import_costs.plot(ax=ax)
# # plt.show()
# # exit(0)
# plot_bus(my_es, 'bus_electricity_all_DE01')
# # exit(0)
# # analyse_bus(2014, 'de21', 'deflex', 'DE01')
#
#
# exit(0)
#
# # scenarios = pd.read_csv('scenarios.csv')
# # scenarios['description'] = scenarios.apply(lambda x: '_'.join(x),
#  axis=1)
# #
# # # scenarios.to_csv('scenarios.csv')
# # costs_df = pd.DataFrame()
# # for scen in scenarios.itertuples():
# #     logging.info("Process scenario: {0}".format(scen.description))
# #     if scen.rmap != 'single':
# #         my_es = load_es(scen.variant, scen.rmap, scen.cat,
#  with_tags=True)
# #         costs_df[scen.description] = analyse_system_costs(my_es)
#
# # costs_df.to_excel('/home/uwe/costs.xls')
# costs_df = pd.read_excel('/home/uwe/costs.xls')
# print(costs_df.sum())
# print(costs_df.mean())
# costs_df.plot(legend=True)
# plt.show()
# # print(get_nominal_values(my_es))
# # exit(0)
# # plot_bus_view(my_es)
# exit(0)
# my_es_2 = load_es(2014, 'de21', 'berlin_hp')
# # reshape_bus_view(my_es, my_es.groups['bus_electricity_all_DE01'])
# get_multiregion_bus_balance(my_es, 'bus_electricity_all')
#
# compare_transmission(my_es, my_es_2)
# exit(0)
#
# print(emissions(my_es).div(1000000))
# exit(0)
# fullloadhours(my_es, [0, 1, 2, 3, 4]).to_excel('/home/uwe/test.xls')
# exit(0)
# analyse_system_costs(my_es, plot=True)
# plt.show()
# print('Done')
# df1 = get_one_node_type(my_es, node='shortage_bus_elec')
# df2 = get_one_node_type(my_es)
# print(df1.round(1))
# print(df2.round(1))
# exit(0)
#
# my_res = plot_multiregion_io(my_es)
# print(my_res.sum())
# print(my_res.max())
#
# print(my_res.sum().div(my_res.max()))
#
# plt.show()
#
# # analyse_bus(2014, 'single', 'friedrichshagen', 'FHG')
# exit(0)
# # all_res = load_all_results(2014, 'de21', 'deflex')
# # wind = ee_analyser(all_res, 'wind')
# # solar = ee_analyser(all_res, 'solar')
# # new_analyser(all_res, wind, solar)
# # print(bl['bus_elec_DE07'])
# exit(0)
# # system = load_es(2014, 'de21', 'deflex')
# # results = load_es(2014, 'de21', 'deflex').results['Main']
# # param = load_es(2014, 'de21', 'deflex').results['Param']
# # cs_bus = [x for x in results.keys() if ('_cs_' in x[0].label) & (
# #     isinstance(x[1], solph.Transformer))]
#
# # print(flows.sum())
# # exit(0)
# # system = load_es(2014, 'de21', 'deflex')
# # FINNISCHE METHODE
# eta_th_kwk = 0.5
# eta_el_kwk = 0.3
# eta_th_ref = 0.9
# eta_el_ref = 0.5
# pee = (1/(eta_th_kwk/eta_th_ref + eta_el_kwk/eta_el_ref)) * (
#         eta_el_kwk/eta_el_ref)
# pet = (1/(eta_th_kwk/eta_th_ref + eta_el_kwk/eta_el_ref)) * (
#         eta_th_kwk/eta_th_ref)
#
# # print(param)
# # Refferenzwirkungsgrade Typenabhängig (Kohle, Gas...)
#
# print(pee * 200, pet * 200)
#
# # https://www.ffe.de/download/wissen/
# # 334_Allokationsmethoden_CO2/ET_Allokationsmethoden_CO2.pdf
# analyse_berlin_ressources()
# collect_berlin_ressource_data()
# exit(0)
# es = load_es('deflex', str(2014), var='de22_without_berlin')
# plot_bus_view(es)
# plt.show()
# plot_bus_view(es)
# check_excess_shortage(es.results['Main'])
# exit(0)
# analyse_ee_basic()
# exit(0)
# analyse_berlin_basic()
# exit(0)
# analyse_upstream_scenarios()
# exit(0)
# analyse_fhg_basic()
# multi_analyse_fhg_emissions()
# df = pd.read_excel('/home/uwe/emissions_analysis.xls')
# # for c in df.index:
# #     print(c)
# #     if 'hard_coal' in c:
# #         df.drop(c, inplace=True)
# df['optional_tot'] = (df['total'] - df['optional_export'] +
#                       df['optional_import'])
# df['upstream_tot'] = df['total'] - df['displaced'] + df['supplement']
# df.sort_index(axis=1).sort_index(axis=0).plot(kind='bar')
# plt.show()
# # analyse_fhg_emissions()
# exit(0)
# # my_es = load_es(2014, 'de21', 'deflex')
# # analyse_system_costs(my_es, plot=True)
# # exit(0)
