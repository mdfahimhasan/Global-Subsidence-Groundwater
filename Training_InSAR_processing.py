# Author: Md Fahim Hasan
# Email: Fahim.Hasan@colostate.edu

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from glob import glob
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point
from System_operations import makedirs
from Raster_operations import read_raster_arr_object, write_raster, shapefile_to_raster


def classify_insar_raster(input_raster, output_raster_name, unit_scale,
                          cnra_data=False, start_date=None, end_date=None, resampled_raster_name='Resampled.tif',
                          res=0.02, output_dir='../InSAR_Data/Resampled_subsidence_data/resampled_insar_data'):
    """
    Classify InSAR subsidence raster to project classes (<1cm/yr, 1-5cm/yr and >5cm/yr).

    Parameters :
    input_raster : Input Raster filepath.
    output_raster_name : Output raster name.
    unit_scale : Unit scale value (i.e. unit_scale=100 for m to cm conversion) for conversion.
    cnra_data : If the data is from 'California National Resources Agency', set True to convert values to cm/year.
    start_date : If cnra data, start day of the data in string format. Set as "2015/06/13"
                 ("Year/month/day"). Default Set to None.
    end_date : If cnra data, end day of the data in string format. Set as "2019/09/19" ("Year/month/day")
               Default Set to None.
    resampled_raster_name : Resampled raster name. Default is 'Resampled.tif'.
    res : Pixel resolution in degree. Default is 0.02 degree.
    output_dir : Output Directory path. Default set to '../InSAR_Data/Resampled_subsidence_data/resampled_insar_data'

    Returns : Classified (and resampled if modify raster=True) subsidence raster.

    **** For California cnra data processing cnra_data=True, Unit_scale=1

    """
    arr, file = read_raster_arr_object(input_raster)

    if cnra_data:
        start_day = datetime.strptime(start_date, "%Y/%m/%d")
        end_day = datetime.strptime(end_date, "%Y/%m/%d")
        months_between = round(int(str(end_day - start_day).split(" ")[0]) / 30)
        arr = arr * 30.48 * 12 / months_between  # 1 ft = 30.48 cm

    arr = arr * unit_scale

    # New_classes
    sub_less_1cm = 1
    sub_1cm_to_5cm = 5
    sub_greater_5cm = 10
    other_values = np.nan

    arr = np.where(arr >= 0, other_values, arr)
    arr = np.where(arr >= -1, sub_less_1cm, arr)
    arr = np.where((arr < -1) & (arr >= -5), sub_1cm_to_5cm, arr)
    arr = np.where(arr < -5, sub_greater_5cm, arr)

    makedirs([output_dir])
    output_raster = os.path.join(output_dir, output_raster_name)

    outfilepath = write_raster(raster_arr=arr, raster_file=file, transform=file.transform, outfile_path=output_raster)

    resampled_raster = os.path.join(output_dir, resampled_raster_name)

    gdal.Warp(destNameOrDestDS=resampled_raster, srcDSOrSrcDSTab=outfilepath, dstSRS='EPSG:4326', xRes=res, yRes=res,
              outputType=gdal.GDT_Float32)

    return resampled_raster


def process_primary_insar_data(processing_areas=('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin',
                                                 'China_Hebei', 'China_Hefei', 'Colorado'),
                               output_dir='../InSAR_Data/Merged_subsidence_data/resampled_insar_data'):
    """
    Resamples and reclassifies insar data for 'California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin', 'China_Hebei',
                                              'China_Hefei', 'Colorado'.

    Parameters:
    processing_areas : A tuple of insar data areas. Default set as
                       ('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin', 'China_Hebei')
    output_dir : Output directory filepath to store processed data. Default set to
                 '../InSAR_Data/Resampled_subsidence_data/resampled_insar_data'

    Returns: Resampled and reclassified insar datasets.
    """

    existing_files = glob(output_dir + '/' + '*.tif')
    for file in existing_files:
        os.remove(file)

    data_dict = {'California': '../InSAR_Data/California/California_vert_disp_20150613_20190919.tif',
                 'Arizona': '../InSAR_Data/Arizona/2010_2019/MS_2010_2019.tif',
                 'Pakistan_Quetta': '../InSAR_Data/Pakistan_Quetta/Quetta_2017_2021.tif',
                 'Iran_Qazvin': '../InSAR_Data/Iran/Iran_Qazvin.tif',
                 'China_Hebei': '../InSAR_Data/China_Hebei/China_Hebei.tif',
                 'China_Hefei': '../InSAR_Data/China_Hefei/China_Hefei.tif',
                 'Colorado': '../InSAR_Data/Colorado/Colorado.tif'}

    for area in processing_areas:
        if area == 'California':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='California_reclass.tif',
                                  cnra_data=True, unit_scale=1, start_date='2015/06/13', end_date='2019/09/19',
                                  resampled_raster_name='California_reclass_resampled.tif', output_dir=output_dir)

        elif area == 'Arizona':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='Arizona_reclass.tif',
                                  cnra_data=False, unit_scale=1, resampled_raster_name='Arizona_reclass_resampled.tif',
                                  output_dir=output_dir)

        elif area == 'Pakistan_Quetta':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='Pakistan_Quetta_reclass.tif',
                                  cnra_data=False, unit_scale=100,
                                  resampled_raster_name='Pakistan_Quetta_reclass_resampled.tif',
                                  output_dir=output_dir)

        elif area == 'Iran_Qazvin':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='Iran_Qazvin_reclass.tif',
                                  cnra_data=False, unit_scale=0.1,
                                  resampled_raster_name='Iran_Qazvin_reclass_resampled.tif',
                                  output_dir=output_dir)

        elif area == 'China_Hebei':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='China_Hebei_reclass.tif',
                                  unit_scale=1, resampled_raster_name='China_Hebei_reclass_resampled.tif',
                                  output_dir=output_dir)

        elif area == 'China_Hefei':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='China_Hefei_reclass.tif',
                                  unit_scale=0.1, resampled_raster_name='China_Hefei_reclass_resampled.tif',
                                  output_dir=output_dir)
        elif area == 'Colorado':
            classify_insar_raster(input_raster=data_dict[area], output_raster_name='Colorado_reclass.tif',
                                  unit_scale=1, resampled_raster_name='Colorado_reclass_resampled.tif',
                                  output_dir=output_dir)


def rasterize_coastal_subsidence(mean_output_points, output_dir,
                                 input_csv='../InSAR_Data/Coastal_Subsidence/Fig3_data.csv'):
    """
    Rasterize coastal subsidence data from Shirzaei_et_al 2020.

    Parameters:
    mean_output_points : Filepath to save filtered points (with mean subsidence values) from input_csv as point shapefile.
    output_dir : Output directory filepath to save converted Geotiff file.
    input_csv : Input csv filepath. Set to path '../InSAR_Data/Coastal_Subsidence/Fig3_data.csv'.

    Return : A Geotiff file of 0.02 degree containing coastal subsidence data.
    """
    coastal_df = pd.read_csv(input_csv)
    new_col_dict = {'Longitude_deg': 'Long_deg', 'Latitude_deg': 'Lat_deg', 'first_epoch': '1st_epoch',
                    'last_epoch': 'last_epoch', 'VLM_mm_yr': 'VLM_mm_yr','VLM_std_mm_yr': 'VLMstdmmyr'}
    coastal_df.rename(columns=new_col_dict, inplace=True)
    coastal_df = coastal_df[(coastal_df['1st_epoch'] >= 2006) & (coastal_df['VLM_mm_yr'] < 0)]
    coastal_df['VLM_cm_yr'] = coastal_df['VLM_mm_yr'] / 10
    coords = [Point(xy) for xy in zip(coastal_df['Long_deg'], coastal_df['Lat_deg'])]
    coastal_shp = gpd.GeoDataFrame(coastal_df, geometry=coords, crs='EPSG:4326')
    intermediate_shp = '../InSAR_Data/Coastal_Subsidence/filtered_point.shp'
    coastal_shp.to_file(intermediate_shp)

    # blank_raster will only be used to pixel locations (indices)
    blank_raster = shapefile_to_raster(intermediate_shp, output_dir, 'blank_raster.tif',
                                       use_attr=False, attribute='VLM_cm_yr',
                                       add=False, burnvalue=0, alltouched=True)
    blank_arr, blank_file = read_raster_arr_object(blank_raster)
    index_list = list()
    for lon, lat in zip(coastal_shp['Long_deg'], coastal_shp['Lat_deg']):
        row, col = blank_file.index(lon, lat)
        index_list.append(str(row + col))
    coastal_shp['pix_index'] = index_list
    mean_subsidence = coastal_shp.groupby('pix_index', as_index=True)['VLM_cm_yr'].transform('mean')
    coastal_shp['mean_cm_yr'] = mean_subsidence
    coastal_shp.to_file(mean_output_points)

    coastal_subsidence_raster = shapefile_to_raster(mean_output_points, output_dir, 'coastal_subsidence.tif',
                                                    use_attr=True, attribute='mean_cm_yr',
                                                    add=None, burnvalue=0, alltouched=True)
    return coastal_subsidence_raster


def subsidence_point_to_geotiff(inputshp, output_raster, res=0.02):
    """
    Convert point shapefile (*) to geotiff.
    * point geometry must have subsidence (z) value. Typically such point shapefile is converted
    from kml file (using QGIS) processed from InSAR.

    Parameters :
    inputshp : Input point shapefile path.
    res : Default set to 0.02 degree.
    output_raster : Output raster filepath.

    Returns : Raster in Geotiff format.
    """
    point_shp = gpd.read_file(inputshp)

    def getXYZ(pt):
        return pt.x, pt.y, pt.z

    lon, lat, z = [list(t) for t in zip(*map(getXYZ, point_shp['geometry']))]

    minx, maxx = min(lon), max(lon)
    miny, maxy = min(lat), max(lat)
    point_shp['value'] = z
    bounds = [minx, miny, maxx, maxy]

    point_shp.to_file(inputshp)

    cont_raster = gdal.Rasterize(output_raster, inputshp, format='GTiff', outputBounds=bounds,outputSRS='EPSG:4326',
                                 outputType=gdal.GDT_Float32, xRes=res, yRes=res, noData=-9999, attribute='value',
                                 allTouched=True)
    del cont_raster

