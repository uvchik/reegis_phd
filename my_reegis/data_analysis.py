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


START = datetime.datetime.now()


def pv_orientation(key, geom, weather, system):
    year = weather.index[0].year
    path = os.path.join(
        cfg.get('paths', 'analysis'), 'pv_orientation', str(year))
    os.makedirs(path, exist_ok=True)
    latitude = geom.y
    longitude = geom.x
    naive_times = pd.DatetimeIndex(
        start=str(year), end=str(year+1), freq='1h')[:-1]
    location = pvlib.location.Location(latitude=latitude, longitude=longitude)
    times = naive_times.tz_localize('Etc/GMT-1')
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(location.altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

    # weather
    dhi = weather['dhi']
    ghi = dhi + weather['dirhi']
    dni = pvlib.irradiance.dni(ghi, dhi, solpos['zenith'],
                               clearsky_dni=None).fillna(0)
    wind_speed = weather['v_wind']
    temp_air = weather['temp_air'] - 273.15

    orientation = sorted(
        set((x, y) for x in range(30, 41) for y in range(170, 200)))
    s = pd.Series(index=pd.MultiIndex(levels=[[], []], labels=[[], []]))
    for tilt, azimuth in orientation:
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
        s[tilt, azimuth] = ac.sum()
    s.to_csv(os.path.join(path, str(key) + '.csv'))
    logging.info("{0}: {1} - {2}".format(
        year, key, datetime.datetime.now() - START))


def _pv_orientation(d):
    pv_orientation(**d)


def optimal_pv_orientation():
    global START
    START = datetime.datetime.now()

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
    reduced = cstd_pt.loc[cstd_pt.coastdat == 1]
    reduced.sort_index(inplace=True)

    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['LG_LG290N1C_G3__2013_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    system = {'module': module, 'inverter': inverter}

    system['installed_capacity'] = (system['module']['Impo'] *
                                    system['module']['Vmpo'])

    years = [2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004,
             2003, 2002, 2001, 1999, 1998, 2000]

    for year in years:
        # Open coastdat-weather data hdf5 file for the given year or try to
        # download it if the file is not found.
        weather_file_name = os.path.join(
            cfg.get('paths', 'coastdat'),
            cfg.get('coastdat', 'file_pattern').format(year=year))
        if not os.path.isfile(weather_file_name):
            coastdat.get_coastdat_data(year, weather_file_name)

        weather = pd.HDFStore(weather_file_name, mode='r')

        coastdat_fields = []
        for key, point in reduced.centroid.iteritems():
            coastdat_fields.append({
                'key': key,
                'geom': point,
                'weather': weather['/A' + str(key)],
                'system': system,
            })

        p = multiprocessing.Pool(5)
        p.map(_pv_orientation, coastdat_fields)
        p.close()
        p.join()


if __name__ == "__main__":
    logger.define_logging()
    cfg.init(paths=[os.path.dirname(deflex.__file__)])
    optimal_pv_orientation()
