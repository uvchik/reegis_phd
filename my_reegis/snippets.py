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
    p = '/home/uwe/chiba/Promotion/reegis_geometries/regions_states'
    inf = os.path.join(p, 'de17_state_no_tso.shp')
    outf = os.path.join(p, 'csv', 'de17_state_no_tso.csv')
    convert_shp2csv(inf, outf, index_col='region')
