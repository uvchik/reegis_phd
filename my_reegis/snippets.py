# Python libraries
import logging

# External libraries

# oemof packages
from oemof.tools import logger

# reegis modules
import reegis_tools.geometries as geometries


def convert_shp2csv(infile, outfile, index_col='region'):
    logging.info("Converting {0} to {1}.".format(infile, outfile))
    geo = geometries.Geometry()
    df = geo.load(fullname=infile, index_col=index_col).get_df()
    print(df)
    df = df.set_index(index_col).sort_index().sort_index(1)

    keep_cols = {'AUFSCHRIFT', 'BEZIRK', 'geometry'}

    rm_cols = set(df.columns) - keep_cols

    for c in rm_cols:
        del df[c]
    # print(df)
    # exit(0)
    df.to_csv(outfile)


def convert_csv2shp(infile, outfile):
    logging.info("Converting {0} to {1}.".format(infile, outfile))
    geo = geometries.Geometry()
    gdf = geo.load(fullname=infile).gdf
    gdf.to_file(outfile)


if __name__ == "__main__":
    logger.define_logging()
    # inf = '/home/uwe/chiba/Promotion/Statstik/Fernwaerme/Fernwaerme_2007/district_heat_blocks_mit_Vattenfall_1_2.shp'
    # outf = '/home/uwe/git_local/reegis/berlin_hp/berlin_hp/data/static/map_district_heating_areas_berlin.csv'
    # inf = '/home/uwe/chiba/Promotion/deu21_geometries/de22_vg250_new.shp'
    # outf = '/home/uwe/chiba/Promotion/deu21_geometries/deu22_vg250_new.csv'
    # inf = '/home/uwe/git_local/reegis/deflex/deflex/data/geometries/powerlines_lines_de22.csv'
    # outf ='/home/uwe/git_local/reegis/deflex/deflex/data/geometries/powerlines_lines_de22.shp'
    path = '/home/uwe/chiba/Promotion/reegis_geometries/friedrichshagen/'
    # inf = path + 'de22_tso_aligned.shp'
    # outf = path + 'csv/de22_vg250_vwg_tso_aligned.csv'
    # print(gpd.read_file(inf))
    # exit(0)
    inf = path + 'berlin_bezirke.shp'
    outf = path + 'berlin_bezirke.csv'
    convert_shp2csv(inf, outf, index_col='BEZIRK12')
    # convert_csv2shp(inf, outf)