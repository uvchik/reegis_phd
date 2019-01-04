import os
import pandas as pd
import geopandas as gpd
from reegis_tools import coastdat
from reegis_tools import geometries
import reegis_tools.config as cfg
import multiprocessing
import pvlib
import deflex
import datetime
import logging
from oemof.tools import logger
from matplotlib import pyplot as plt
import numpy as np


START = datetime.datetime.now()
YEARS = [2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004,
         2003, 2002, 2001, 1999, 1998, 2000]


def pv_orientation(key, geom, weather, system, orientation, path):
    year = weather.index[0].year
    os.makedirs(path, exist_ok=True)
    latitude = geom.y
    longitude = geom.x

    start = datetime.datetime(year-1, 12, 31, 23, 30)
    end = datetime.datetime(year, 12, 31, 22, 30)
    naive_times = pd.DatetimeIndex(start=start, end=end, freq='1h')

    location = pvlib.location.Location(latitude=latitude, longitude=longitude)
    times = naive_times.tz_localize('Etc/GMT-1')
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(location.altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

    # weather
    dhi = weather['dhi'].values
    ghi = dhi + weather['dirhi'].values
    dni = pvlib.irradiance.dni(ghi, dhi, solpos['zenith'],
                               clearsky_dni=None).fillna(0)
    wind_speed = weather['v_wind'].values
    temp_air = weather['temp_air'].values - 273.15

    s = pd.Series(index=pd.MultiIndex(levels=[[], []], labels=[[], []]))
    fix_tilt = 0.0
    for tilt, azimuth in orientation:
        if tilt == fix_tilt:
            dt = datetime.datetime.now() - START
            logging.info("{0} - {1}".format(tilt, dt))
            fix_tilt += 0.1
        system['surface_azimuth'] = azimuth
        system['surface_tilt'] = tilt
        aoi = pvlib.irradiance.aoi(
            system['surface_tilt'], system['surface_azimuth'],
            solpos['apparent_zenith'], solpos['azimuth'])
        total_irrad = pvlib.irradiance.get_total_irradiance(
            system['surface_tilt'], system['surface_azimuth'],
            solpos['apparent_zenith'], solpos['azimuth'], dni, ghi, dhi,
            dni_extra=dni_extra, model='haydavies')
        temps = pvlib.pvsystem.sapm_celltemp(
            total_irrad['poa_global'], wind_speed, temp_air)
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            total_irrad['poa_direct'], total_irrad['poa_diffuse'], am_abs, aoi,
            system['module'])
        dc = pvlib.pvsystem.sapm(effective_irradiance, temps['temp_cell'],
                                 system['module'])
        ac = pvlib.pvsystem.snlinverter(dc['v_mp'], dc['p_mp'],
                                        system['inverter'])
        s[tilt, azimuth] = ac.sum() / system['installed_capacity']
    s.to_csv(os.path.join(path, str(key) + '.csv'))
    logging.info("{0}: {1} - {2}".format(
        year, key, datetime.datetime.now() - START))


def _pv_orientation(d):
    pv_orientation(**d)


def get_coastdat_onshore_polygons():
    cstd = geometries.load(
        cfg.get('paths', 'geometry'),
        cfg.get('coastdat', 'coastdatgrid_polygon'),
        index_col='gid')

    de02 = geometries.load(
        cfg.get('paths', 'geo_deflex'),
        cfg.get('geometry', 'deflex_polygon').format(
            type='polygon', map='de02', suffix='reegis'),
        index_col='region')

    cstd_pt = gpd.GeoDataFrame(cstd.centroid, columns=['geometry'])

    cstd_pt = geometries.spatial_join_with_buffer(
        cstd_pt, de02, 'coastdat', limit=0)
    reduced = cstd.loc[cstd_pt.coastdat == 1]
    return reduced.sort_index()


def pv_yield_by_orientation():
    global START
    START = datetime.datetime.now()

    reduced = get_coastdat_onshore_polygons()

    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['LG_LG290N1C_G3__2013_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    system = {'module': module, 'inverter': inverter}

    system['installed_capacity'] = (system['module']['Impo'] *
                                    system['module']['Vmpo'])

    orientation = sorted(
        set((x/10, y/10) for x in range(0, 901) for y in range(0, 3601)))
    # orientation = sorted(
    #     set((x, y) for x in range(0, 91) for y in range(0, 361)))

    year = 2014
    key = 1129089

    weather_file_name = os.path.join(
        cfg.get('paths', 'coastdat'),
        cfg.get('coastdat', 'file_pattern').format(year=year))
    if not os.path.isfile(weather_file_name):
        coastdat.get_coastdat_data(year, weather_file_name)
    weather = pd.read_hdf(weather_file_name, mode='r', key='/A' + str(key))

    path = os.path.join(
            cfg.get('paths', 'analysis'), 'pv_yield_by_orientation', str(year))

    point = reduced.centroid[key]
    print(point)

    pv_orientation(key, point, weather, system, orientation, path)


def optimal_pv_orientation():
    global START
    START = datetime.datetime.now()

    reduced = get_coastdat_onshore_polygons()

    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['LG_LG290N1C_G3__2013_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    system = {'module': module, 'inverter': inverter}

    system['installed_capacity'] = (system['module']['Impo'] *
                                    system['module']['Vmpo'])

    orientation = sorted(
        set((x, y) for x in range(30, 46) for y in range(168, 193)))

    for year in YEARS:
        # Open coastdat-weather data hdf5 file for the given year or try to
        # download it if the file is not found.
        weather_file_name = os.path.join(
            cfg.get('paths', 'coastdat'),
            cfg.get('coastdat', 'file_pattern').format(year=year))
        if not os.path.isfile(weather_file_name):
            coastdat.get_coastdat_data(year, weather_file_name)

        path = os.path.join(
            cfg.get('paths', 'analysis'), 'pv_orientation_minus30', str(year))

        weather = pd.HDFStore(weather_file_name, mode='r')

        coastdat_fields = []
        for key, point in reduced.centroid.iteritems():
            coastdat_fields.append({
                'key': key,
                'geom': point,
                'weather': weather['/A' + str(key)],
                'system': system,
                'orientation': orientation,
                'path': path,
            })

        p = multiprocessing.Pool(4)
        p.map(_pv_orientation, coastdat_fields)
        p.close()
        p.join()
        weather.close()


def collect_orientation_files(year=None):
    base_path = os.path.join(cfg.get('paths', 'analysis'),
                             'pv_orientation_minus30')
    multi = pd.MultiIndex(levels=[[], []], labels=[[], []])
    df = pd.DataFrame(columns=multi, index=multi)

    if year is None:
        years = YEARS
    else:
        years = [year]

    for year in years:
        print(year)
        full_path = os.path.join(base_path, str(year))
        if os.path.isdir(full_path):
            for file in os.listdir(full_path):
                s = pd.read_csv(os.path.join(full_path, file),
                                index_col=[0, 1], squeeze=True,
                                header=None).sort_index()
                key = file[:-4]
                df[key, year] = s

    keys = df.columns.get_level_values(0).unique()

    new_df = pd.DataFrame(index=df.index)
    for key in keys:
        new_df[key] = df[key].sum(axis=1)
    new_df.index.set_names(['tilt', 'azimuth'], inplace=True)
    if year is None:
        outfile = 'multiyear_yield_sum.csv'
    else:
        outfile = 'yield_{0}.csv'.format(year)
    new_df.to_csv(os.path.join(base_path, outfile))


def analyse_multi_files():
    path = os.path.join(cfg.get('paths', 'analysis'), 'pv_orientation_minus30')
    fn = os.path.join(path, '2014_sum.csv')
    df = pd.read_csv(fn, index_col=[0, 1])
    gdf = get_coastdat_onshore_polygons()
    gdf.geometry = gdf.buffer(0.005)

    for key in gdf.index:
        s = df[str(key)]
        p = gdf.loc[key]
        gdf.loc[key, 'tilt'] = (
            s[s == s.max()].index.get_level_values('tilt')[0])
        gdf.loc[key, 'azimuth'] = (
            s[s == s.max()].index.get_level_values('azimuth')[0])
        gdf.loc[key, 'longitude'] = p.geometry.centroid.x
        gdf.loc[key, 'latitude'] = p.geometry.centroid.y
        gdf.loc[key, 'tilt_calc'] = round(p.geometry.centroid.y - 15)
    cmap = plt.get_cmap('viridis', 12)
    print(gdf.azimuth.max(), gdf.azimuth.min())
    print(gdf.tilt.max(), gdf.tilt.min())
    gdf.plot('azimuth', legend=True, cmap=cmap, vmin=172.5, vmax=184.5)
    gdf.plot('tilt', legend=True)
    gdf.plot('tilt_calc', legend=True)
    plt.show()
    # scatter(gdf['longitude'], gdf['latitude'], gdf['tilt'], gdf['azimuth'])

    # results.to_csv(os.path.join(path, 'optimal_orientation_multi_year.csv'))


def scatter(x, y, c, s):

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # x = np.arange(0.0, 50.0, 2.0)
    # y = x ** 1.3 + np.random.rand(*x.shape) * 30.0
    # s = np.random.rand(*x.shape) * 800 + 500

    print(x, len(x))
    print(y, len(y))
    print(s, len(s))

    plt.scatter(x, y, c=c, s=s)
    plt.xlabel("Leprechauns")
    plt.ylabel("Gold")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(deflex.__file__)])
    # optimal_pv_orientation()
    pv_yield_by_orientation()
    # collect_orientation_files()
    # plt.show()
    # scatter()
    # analyse_multi_files()
