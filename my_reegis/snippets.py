# Python libraries
import logging
import os
import deflex

# External libraries
from shapely.geometry import Point
import pandas as pd

# oemof packages
from oemof.tools import logger

# reegis modules
import reegis_tools.geometries as geometries


def time_analysis():
    df = pd.read_csv('/home/uwe/log_time_test.csv', parse_dates=True)
    print(df.index)
    idx = pd.MultiIndex(levels=[[], [], []], labels=[[], [], []])
    time_results = pd.DataFrame(index=idx)
    df['t_value'] = df.index
    df['delta'] = (df['t_value']-df['t_value'].shift()).fillna(0)
    del df['ms']
    del df['t_value']
    del df['log']
    df = df[df.module != 'main']
    del df['module']
    for row in df.iterrows():
        elapsed_time = row[1].delta
        msg = row[1].msg.replace('.esys.', '').split(os.sep)
        name = msg[-1]
        name_split = name.split('_')
        try:
            year = int(name_split[1])
            base = name_split[0]
            var = '_'.join(name_split[2:])
        except ValueError:
            year = int(name_split[2])
            base = '_'.join(name_split[:2])
            var = '_'.join(name_split[3:])
        solver = msg[-2].split('_')[-1]
        col = 'time_{0}'.format(solver)
        time_results.loc[(year, base, var), col] = (
            elapsed_time)
        time_results.sort_index(inplace=True)
    time_results['diff'] = (
            time_results['time_cbc'].abs() - time_results['time_gurobi'].abs())
    print(time_results.groupby(level=[0, 1]).sum())
    print(time_results.groupby(level=[0, 1]).sum().sum())
    time_results['Prz'] = time_results['diff'].div(time_results.time_cbc)
    time_results.to_excel('/home/uwe/time_test.xls')
    time_results.to_csv('/home/uwe/time_test.csv')


def convert_shp2csv(infile, outfile, index_col='region'):
    logging.info("Converting {0} to {1}.".format(infile, outfile))
    df = geometries.load(fullname=infile, index_col=index_col)
    print(df)
    df = df.set_index(index_col, drop=True).sort_index()
    print(df)
    # keep_cols = {'AUFSCHRIFT', 'BEZIRK', 'geometry'}

    # rm_cols = set(df.columns) - keep_cols

    # for c in rm_cols:
    #     del df[c]
    # print(df)
    # exit(0)
    df.to_csv(outfile)


def convert_csv2shp(infile, outfile):
    logging.info("Converting {0} to {1}.".format(infile, outfile))
    geo = geometries.Geometry()
    gdf = geo.load(fullname=infile).gdf
    gdf.to_file(outfile)


def create_grid_for_deflex_regions():
    de = deflex.geometries.deflex_regions(rmap='de17').gdf
    de['label'] = de.representative_point()

    de.loc['DE03', 'label'] = Point(9.27, 52.97)
    de.loc['DE12', 'label'] = Point(12.74, 52.62)

    lines = pd.DataFrame(columns=['geometry'])
    for reg1 in de.index:
        for reg2 in de.index:
            if reg1 != reg2:
                if de.loc[reg1].geometry.touches(de.loc[reg2].geometry):
                    line_id = '{0}-{1}'.format(reg1, reg2)
                    geo = 'LINESTRING({0} {1}, {2} {3})'.format(
                        de.loc[reg1].label.x, de.loc[reg1].label.y,
                        de.loc[reg2].label.x, de.loc[reg2].label.y)
                    if de.loc[reg1].label.y < de.loc[reg2].label.y:
                        lines.loc[line_id, 'geometry'] = geo
    lines['name'] = lines.index
    lin = geometries.create_geo_df(lines)
    lin.to_file('/home/uwe/tmp_lines.shp')
    lin.to_csv('/home/uwe/tmp_lines.csv')
    lin.plot()


if __name__ == "__main__":
    logger.define_logging()
    time_analysis()
    exit(0)
    p = '/home/uwe/chiba/Promotion/reegis_geometries/regions_states'
    inf = os.path.join(p, 'de17_state_no_tso.shp')
    outf = os.path.join(p, 'csv', 'de17_state_no_tso.csv')
    convert_shp2csv(inf, outf, index_col='region')
