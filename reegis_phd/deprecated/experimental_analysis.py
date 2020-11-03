import os
from matplotlib import pyplot as plt
from my_reegis import results
from oemof.tools import logger
import pandas as pd
import datetime
import reegis.config as cfg
from my_reegis import upstream_analysis as upa


# warnings.filterwarnings("error")
logger.define_logging()
start = datetime.datetime.now()

cfg.tmp_set('results', 'dir', 'results_cbc')
my_es1 = results.load_my_es('deflex', '2014', var='de21_no_grid_limit')
my_es2 = results.load_my_es('deflex', '2014',
                            var='de21_no_grid_limit_no_storage')
pldf1 = pd.DataFrame()
pldf2 = pd.DataFrame()
pldf1['duals_no_storage'] = upa.fetch_duals_max_min(my_es2, var=None)[
    'uniform']
pldf1['no_storages'] = upa.get_emissions_and_costs(my_es2)['mcp']
pldf2['duals_storage'] = upa.fetch_duals_max_min(my_es1, var=None)['uniform']
pldf2['storages'] = upa.get_emissions_and_costs(my_es1)['mcp']
duals2 = upa.fetch_duals_max_min(my_es2, var=None)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
pldf1.plot(ax=ax1)
pldf2.plot(ax=ax2)
(pldf1['duals_no_storage'] - pldf1['no_storages']).round().plot(ax=ax3)
(pldf2['duals_storage'] - pldf2['storages']).round().plot(ax=ax4)
plt.show()
exit(0)
my_es = results.load_my_es('deflex', '2014', var='de21_no_grid_limit')
# Get the full merit_order table with storages
results.fetch_cost_emission(my_es).to_excel('/home/uwe/merit_order_neu.xls')
parameter_full = results.fetch_cost_emission_with_storages(my_es)
parameter_full.sort_values(
    ('no_storage', 'var_costs')).to_excel('/home/uwe/merit_order2.xls')

# plot merit order for all pp/chp without storages
parameter_nos = parameter_full['no_storage']
parameter_nos.sort_values('var_costs')['var_costs'].plot(kind='bar')
parameter_nos.sort_values('var_costs')['var_costs'].to_excel(
    '/home/uwe/merit_order.xls')

# Fetch the mutli-region results
mdf = results.get_multiregion_bus_balance(my_es)

# plot in/out of all storages for all timesteps
f, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
ax[0] = mdf.groupby(level=[1, 2, 3, 4], axis=1).sum()['in', 'storage'].plot(
    ax=ax[0])
mdf.groupby(level=[1, 2, 3, 4], axis=1).sum()['out', 'storage'].plot(ax=ax[0])

# plot the chp-output together with the hp-output
chp_df = pd.DataFrame()
chp_df['e_chp'] = mdf.groupby(level=[1, 3], axis=1).sum()['in', 'chp']
heat = results.get_multiregion_bus_balance(my_es, 'heat_district')
chp_df['h_chp'] = heat.groupby(level=[1, 3], axis=1).sum()['in', 'chp']
chp_df['h_hp'] = heat.groupby(level=[1, 3], axis=1).sum()['in', 'hp']
chp_df.plot(ax=ax[2])

# Load data from "most expensive power plant calculation"
my_path = '/home/uwe/express/reegis/data/friedrichshagen/'
my_filename = 'upstream_scenario_values.csv'
oldway = pd.read_csv(os.path.join(my_path, my_filename),
                     header=[0, 1], index_col=[0])

# Plot dual variables and most expensive power plant together in one plot
plot_df = pd.DataFrame()
plot_df['no_grid_limit'] = upa.fetch_duals_max_min(my_es)['DE01']
oldway.set_index(plot_df.index, inplace=True)
plot_df['merit_order_no_limit'] = (
    oldway['deflex_2014_de21_no_grid_limit', 'meritorder'])
# amo.set_index(plot_df.index, inplace=True)
# plot_df['alt_merit_order_no_limit'] = amo['max']
print(int((plot_df['alt_merit_order_no_limit'] -
           plot_df['merit_order_no_limit']).sum()))
plot_df.plot(ax=ax[1])
plot_df.to_excel('/home/uwe/plot_df.xls')

res = pd.DataFrame()
n = 0
m = 1
parameter_full.to_excel('/home/uwe/df.xls')
df_costs = parameter_full.swaplevel(axis=1)['var_costs']
# for idx, cv in plot_df['no_grid_limit'].iteritems():
#     n += 1
#     if idx.month == m:
#         print(idx.strftime("%B"))
#         print((datetime.datetime.now() - start))
#         m += 1
#     mask = (df_costs.round(5) == round(cv, 5))
#     i, j = np.where(mask)
#     fv = list(zip(df_costs.index[i], df_costs.columns[j]))
#     if len(fv) > 1:
#         if fv[0][0][0] != 'nuclear':
#             print(fv)
#     if len(fv) > 0:
#         res.loc[idx, 'emission'] = parameter_full.loc[fv[0][0], (
#             fv[0][1], 'emission')]
#         res.loc[idx, 'name'] = str(fv[0])
#     else:
#         res.loc[idx, 'emission'] = float('nan')
#         res.loc[idx, 'name'] = 'no match'
#     res.loc[idx, 'var_costs'] = cv
# res.to_excel('/home/uwe/res.xls')
# print('number of nans:', len(res[res.emission.isnull()]))
print((datetime.datetime.now() - start))

plt.show()
