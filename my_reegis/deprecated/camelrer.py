from reegis import Scenario
import datetime
import os
from oemof import solph
import dill as pickle
import pprint as pp
from my_reegis import results


fn = '/home/uwe/deflex_2014_de22_no_grid_limit.esys'

# start1 = datetime.datetime.now()
# for n in range(100):
#     sc1 = Scenario(results_fn=fn)
#     meta = sc1.meta
# z1 = datetime.datetime.now() - start1
#
# start2 = datetime.datetime.now()
# for n in range(100):
#     sc2 = Scenario(results_fn=fn)
#     sc2.restore_es()
#     meta = sc2.meta
# z2 = datetime.datetime.now() - start2
#
# print(z1, z2)
# exit(0)


def find_scenarios(path):
    scenarios = []
    for root, directories, filenames in os.walk(path):
        for f in filenames:
            if f[-5:] == '.esys':
                scenarios.append(os.path.join(root, f))
    return scenarios


p = '/home/uwe/express/reegis/scenarios_lux'
# sl = results.fetch_scenarios(p)
# print(len(sl))
# exit(0)
# filt = {'map': 'berlin',
#         'solver': 'cbc',
#         'upstream': {'grid_limit': False}}
# sl = results.fetch_scenarios(p, filt)
# for s in sl:
#     print(s)
# exit(0)
# filt = {'map': 'de17',
#         'lignite': 0.5}
# sl = results.fetch_scenarios(p, filt)
# for s in sl:
#     print(s)
# exit(0)
import pprint as pp
filt = {'map': 'berlin',
        'storage': True,
        'solver': 'gurobi',
        'year': 2014}
sl = results.fetch_scenarios(p, filt)
for s in sl:
    sc = Scenario(results_fn=s)
    pp.pprint(sc.meta)

exit(0)
p = '/home/uwe/express/reegis/scenarios_lux'
# year = 2014
# solver = 'cbc'
sl = find_scenarios(p)
for fn in sl:
    meta = {}
    sc = Scenario()
    sc.es = solph.EnergySystem()
    f = open(fn, 'rb')
    sc.es.__dict__ = pickle.load(f)
    f.close()
    if 'cbc' in fn:
        meta['solver'] = 'cbc'
    elif 'gurobi' in fn:
        meta['solver'] = 'gurobi'
    else:
        raise ValueError
    filename = str(fn.split(os.sep)[-1])
    meta['filename'] = filename
    filename = filename.replace('.esys', '')
    fs = filename.split('_')
    order = 0
    if fs[0] == 'berlin':
        meta['model_base'] = 'berlin_hp'
        order = 2
    else:
        meta['model_base'] = fs[0]
        order = 1
    meta['year'] = int(fs[order])
    order += 1
    meta['map'] = fs[order]
    if meta['map'] == 'single':
        meta['map'] = 'berlin'
    # deflex_2014_de02_no_grid_limit_no_storage_alt_XX_Nc0_Li1_HP0_GT0_f12
    if 'no_grid_limit' in filename:
        meta['grid_limit'] = False
    else:
        meta['grid_limit'] = True

    if 'without_berlin' in filename:
        meta['excluded'] = 'Berlin'
    else:
        meta['excluded'] = None

    if 'no_storage' in filename:
        meta['storage'] = False
    else:
        meta['storage'] = True

    if '_up_' in filename:
        meta['upstream'] = {}
        fup = str(filename.split('_up_')[-1])
        fups = fup.split('_')
        meta['upstream']['map'] = fups[0]
        if 'no_grid_limit' in fup:
            meta['upstream']['grid_limit'] = False
        else:
            meta['upstream']['grid_limit'] = True

        if 'no_storage' in fup:
            meta['upstream']['storage'] = False
        else:
            meta['upstream']['storage'] = True
        meta['upstream']['nuclear'] = 1.0
        meta['upstream']['lignite'] = 1.0
        meta['upstream']['heat_pump'] = 0.0
        meta['upstream']['gas_turbine'] = 0
        meta['upstream']['ee_factor'] = 1.0

    else:
        meta['upstream'] = None

    if '_XX_' in filename:
        fx = str(filename.split('_XX_')[-1]).split('_')
        meta['nuclear'] = float(fx[0][2])
        if len(fx[1][2:]) == 2:
            meta['lignite'] = float(fx[1][2:]) / 10
        elif len(fx[1][2:]) == 1:
            meta['lignite'] = float(fx[1][2:])
        meta['heat_pump'] = float(fx[2][2:]) / 10
        meta['gas_turbine'] = int(float(fx[3][2:]) * 1000)
        meta['ee_factor'] = float(fx[4][1:]) / 10

    else:
        meta['nuclear'] = 1.0
        meta['lignite'] = 1.0
        meta['heat_pump'] = 0.0
        meta['gas_turbine'] = 0
        meta['ee_factor'] = 1.0
    pp.pprint(meta)
    # exit(0)
    sc.meta = meta
    sc.dump_es(fn)

print(len(sl))

# table = Table(25, 'peter')
# print(Camel([my_types]).dump(table))
# fn = '/home/uwe/deflex_2014_de22_no_grid_limit.esys'
# es = solph.EnergySystem()
# start = datetime.datetime.now()
# f = open(fn, "rb")
# z1 = datetime.datetime.now() - start
# me = pickle.load(f)
# z2 = datetime.datetime.now() - start
# es.__dict__ = pickle.load(f)
# z3 = datetime.datetime.now() - start
# print(es)
# print(z1, z2, z3)
# # meta = {'name': 'de22', 'category': 'deflex', 'year': 2014}
# # f = open(fn, "wb")
# # pickle.dump(meta, f)
# # pickle.dump(es.__dict__, f)
# # objects = [x for x in es.results['Main'] if x[0].label.region == 'DE21']
# # print(objects)
# print(me)
