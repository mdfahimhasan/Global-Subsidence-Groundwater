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
from Raster_operations import read_raster_arr_object, write_raster, shapefile_to_raster, resample_reproject, \
    rename_copy_raster, mosaic_rasters, mosaic_two_rasters

No_Data_Value = -9999
referenceraster = r'../Data/Reference_rasters_shapes/Global_continents_ref_raster.tif'


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
                    'last_epoch': 'last_epoch', 'VLM_mm_yr': 'VLM_mm_yr', 'VLM_std_mm_yr': 'VLMstdmmyr'}
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

    cont_raster = gdal.Rasterize(output_raster, inputshp, format='GTiff', outputBounds=bounds, outputSRS='EPSG:4326',
                                 outputType=gdal.GDT_Float32, xRes=res, yRes=res, noData=-9999, attribute='value',
                                 allTouched=True)
    del cont_raster


def process_EGMS_insar(insar_data_dir='../InSAR_Data/Europe_EGMS/Interim_processing',
                       output_dir='../InSAR_Data/Europe_EGMS', nodata=No_Data_Value,
                       copy_dir='../InSAR_Data/Final_subsidence_data/interim_working_dir',
                       already_prepared=False, ref_raster=referenceraster):
    """
    Processes InSAR devided subsidence data from EGMS portal for Europe.

    Parameters:
    insar_data_dir: Directory path of EGMS InSAR data.
                    ** I am copying all merged/clipped subsidence data files to this folder first.
    output_dir: Directory path of output folder.
    nodata: Default set to -9999.
    copy_dir: Directory path to copy the final (mosaiced + resampled + classified) subsidence Data for Europe.
    already_prepared: Set to True if want to use already processed data. Default set to False.
    ref_raster: Reference raster (global) filepath.

    Returns: Filepath of final processed subsidence raster for Europe.
    """
    if not already_prepared:
        _, ref_file = read_raster_arr_object(ref_raster)

        makedirs(insar_data_dir)
        EGMS_insar_data = glob(os.path.join(insar_data_dir, '*[Clipped-Merged].tif'))

        # Removing positive (uplift) values
        for data in EGMS_insar_data:
            arr, file = read_raster_arr_object(data)
            neg_arr = np.where(arr < 0, arr, nodata)

            basename = os.path.basename(data)
            sep_idx = basename.rfind('_')
            name = basename[: sep_idx] + '_neg_values.tif'
            output_raster = os.path.join(insar_data_dir, name)
            write_raster(raster_arr=neg_arr, raster_file=file, transform=file.transform, outfile_path=output_raster,
                         no_data_value=nodata)

        # mosaicing and resampling individual insar raster
        mosaiced_arr, _ = mosaic_rasters(input_dir=insar_data_dir, output_dir=output_dir,
                                         raster_name='Europe_Subsidence_mosaiced_mm.tif',
                                         search_by='*neg_values*.tif', ref_raster=ref_raster, resolution=0.02)
        # converting mm/year to cm/year
        mosaiced_arr_cm = mosaiced_arr * 0.1
        mosaiced_arr_cm = np.where(mosaiced_arr < -200, nodata, mosaiced_arr_cm)  # -200 is arbitrary big to set nodata

        # Saving EGMS InSAR mosaic in cm value
        write_raster(raster_arr=mosaiced_arr_cm, raster_file=ref_file, transform=ref_file.transform,
                     outfile_path=os.path.join(output_dir, 'Europe_Subsidence_cm.tif'))

        # Classifying to model classes
        sub_less_1cm = 1
        sub_1cm_to_5cm = 5
        sub_greater_5cm = 10

        classified_arr = np.where(mosaiced_arr_cm >= -1, sub_less_1cm, mosaiced_arr_cm)
        classified_arr = np.where((classified_arr < -1) & (classified_arr >= -5), sub_1cm_to_5cm, classified_arr)
        classified_arr = np.where((classified_arr > nodata) & (classified_arr < -5), sub_greater_5cm, classified_arr)
        classified_arr = np.where(np.isnan(mosaiced_arr_cm), np.nan, classified_arr)

        classified_EGMS_insar = os.path.join(output_dir, 'Europe_Subsidence_reclass_resampled.tif',)
        write_raster(raster_arr=classified_arr, raster_file=ref_file, transform=ref_file.transform,
                     outfile_path=classified_EGMS_insar)

        final_EGMS_raster = rename_copy_raster(input_raster=classified_EGMS_insar,
                                               output_dir=copy_dir, rename=True,
                                               new_name='final_EGMS_insar_data.tif')
    else:
        final_EGMS_raster = os.path.join('../Final_subsidence_data/interim_working_dir', 'final_EGMS_insar_data.tif')

    return final_EGMS_raster


def join_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygons, exclude_areas=None,
                                           search_criteria='*Subsidence*.shp'):
    """
    Joining georeferenced subsidence polygons.

    Area processed -'Australia_Perth', 'Bangladesh_GBDelta', 'China_Beijing', 'China_Shanghai', 'China_Tianjin',
    'China_Wuhan', 'China_Xian', 'China_YellowRiverDelta', 'Egypt_NileDelta', 'England_London', 'India_Delhi',
    'Indonesia_Bandung', 'Indonesia_Semarang', 'Iran_MarandPlain', 'Iran_Tehran', 'Iraq_TigrisEuphratesBasin',
    'Italy_PoDelta', 'Italy_VeniceLagoon', 'Mexico_MexicoCity', 'Nigeria_Lagos', 'Spain_Murcia', 'Taiwan_Yunlin',
    'Turkey_Bursa', 'Turkey_Karapinar', 'US_Huston', 'Vietnam_Hanoi', 'Vietnam_HoChiMinh'

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    exclude_areas : Tuple of area names to be excluded from processing. Default set to None.
                    For excluding single area follow tuple pattern ('Bangladesh_GBDelta',)
    search_criteria : Search criteria for input polygons.

    Returns : Joined subsidence polygon.
    """
    # global df
    subsidence_polygons = glob(os.path.join(input_polygons_dir, search_criteria))

    sep = joined_subsidence_polygons.rfind(os.sep)
    makedirs([joined_subsidence_polygons[:sep]])  # creating directory for the prepare_subsidence_raster function

    for each in range(0, len(subsidence_polygons)):
        if each == 0:
            gdf = gpd.read_file(subsidence_polygons[each])
            df = pd.DataFrame(gdf)

        gdf_new = gpd.read_file(subsidence_polygons[each])
        df_new = pd.DataFrame(gdf_new)
        add_to_df = pd.concat([df, df_new], ignore_index=True)
        df = add_to_df
        df['Class_name'] = pd.to_numeric(df['Class_name'], downcast='float')

    if exclude_areas is not None:
        exclude_areas = list(exclude_areas)
        areas = list(df['Area_name'].unique())
        keep_areas = [area for area in areas if area not in exclude_areas]
        df = df.loc[df['Area_name'].isin(keep_areas)]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.to_file(joined_subsidence_polygons)
    return joined_subsidence_polygons


def prepare_subsidence_raster(input_polygons_dir='../InSAR_Data/Georeferenced_subsidence_data',
                              joined_subsidence_polygon='../InSAR_Data/Final_subsidence_data'
                                                        '/interim_working_dir/georef_subsidence_polygons.shp',
                              insar_data_dir='../InSAR_Data/Final_subsidence_data/resampled_insar_data',
                              EGMS_insar_dir='../InSAR_Data/Europe_EGMS/Interim_processing',
                              interim_dir='../InSAR_Data/Final_subsidence_data/interim_working_dir',
                              output_dir='../InSAR_Data/Final_subsidence_data/final_subsidence_raster',
                              skip_polygon_merge=False, subsidence_column='Class_name', resample_algorithm='near',
                              final_subsidence_raster='Subsidence_training.tif', exclude_georeferenced_areas=None,
                              process_insar_areas=('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin',
                                                   'China_Hebei', 'China_Hefei', 'Colorado'),
                              polygon_search_criteria='*Subsidence*.shp',
                              insar_search_criteria='*reclass_resampled*.tif', already_prepared=False,
                              refraster=referenceraster, merge_coastal_subsidence_data=True):
    """
    Prepare subsidence raster for training data by joining georeferenced polygons and insar data.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    insar_data_dir : InSAR data directory.
    EGMS_insar_dir : EGMS InSAR data directory.
    interim_dir : Intermediate working directory for storing interdim data.
    output_dir : Output raster directory.
    skip_polygon_merge : Set to True if polygon merge is not required.
    subsidence_column : Subsidence value column in joined subsidence polygon. Default set to 'Class_name'.
    resample_algorithm : Algorithm for resampling polygon subsidence data. Default set to 'near'.
    final_subsidence_raster : Final subsidence raster including georeferenced and insar data.
    exclude_georeferenced_areas : Tuple of area names to be excluded from processing. Default set to None to include all
                                  gereferenced areas. For excluding single area follow
                                  tuple pattern ('Bangladesh_GBDelta',)
    polygon_search_criteria : Input subsidence polygon search criteria.
    process_insar_areas : Tuple of insar data regions to be included in the model.
                          Default set to ('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin', 'China_Hebei',
                                          'China_Hefei', 'Colorado')
    insar_search_criteria : InSAR data search criteria.
    already_prepared : Set to True if subsidence raster is already prepared.
    refraster : Global Reference raster.
    merge_coastal_subsidence_data : Default set to True to merge coastal GNSS-based subsidence data to the training
    subsidence data.

    Returns : Final subsidence raster to be used as training data.
    """

    if not already_prepared:
        makedirs([interim_dir, output_dir])
        # processing georeferenced subsidence data
        if not skip_polygon_merge:
            print('Processing Subsidence Polygons...')
            subsidene_polygons = join_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygon,
                                                                        exclude_georeferenced_areas,
                                                                        polygon_search_criteria)
        else:
            subsidene_polygons = joined_subsidence_polygon

        interim_georeferenced_subsidence = \
            shapefile_to_raster(subsidene_polygons, interim_dir, resolution=0.005,
                                raster_name='interim_subsidence_raster_0005.tif', use_attr=True,
                                attribute=subsidence_column, ref_raster=refraster, alltouched=False)
        resample_reproject(interim_georeferenced_subsidence, interim_dir,
                           raster_name='final_georef_subsidence_raster.tif', resample=True, reproject=False,
                           both=False, resample_algorithm=resample_algorithm)

        print('Processed Subsidence Polygons')

        print('Processing Processed InSAR Data...')

        # processing processed subsidence data
        process_primary_insar_data(processing_areas=process_insar_areas, output_dir=insar_data_dir)
        mosaic_rasters(insar_data_dir, interim_dir, resolution=0.02, raster_name='final_processed_insar_data.tif',
                       ref_raster=refraster, search_by=insar_search_criteria)

        print('Processing EGMS InSAR Data...')
        # processing EGMS subsidence data
        process_EGMS_insar(insar_data_dir=EGMS_insar_dir, output_dir='../InSAR_Data/Europe_EGMS', nodata=No_Data_Value,
                           already_prepared=already_prepared, ref_raster=referenceraster)

        # Merging georeferenced, processed, and EGMS insar data tp create the final subsidence raster
        final_subsidence_arr, subsidence_data = mosaic_rasters(input_dir=interim_dir, output_dir=output_dir,
                                                               raster_name=final_subsidence_raster,
                                                               ref_raster=refraster, search_by='*final*.tif',
                                                               resolution=0.02)
        print('Processed Georeference, processed, and EGMS InSAR Data')

        if merge_coastal_subsidence_data:
            print('Processing Coastal SUbsidence (GNSS) Data...')
            coastal_raster = rasterize_coastal_subsidence(mean_output_points='../InSAR_Data/Coastal_Subsidence'
                                                                             '/filtered_mean_point.shp',
                                                          output_dir='../InSAR_Data/Coastal_Subsidence',
                                                          input_csv='../InSAR_Data/Coastal_Subsidence/Fig3_data.csv')
            coastal_arr = read_raster_arr_object(coastal_raster, get_file=False)
            ref_arr, ref_file = read_raster_arr_object(refraster)

            # New_classes
            sub_less_1cm = 1
            sub_1cm_to_5cm = 5
            sub_greater_5cm = 10
            other_values = np.nan

            coastal_arr = np.where(coastal_arr >= 0, other_values, coastal_arr)
            coastal_arr = np.where(coastal_arr >= -1, sub_less_1cm, coastal_arr)
            coastal_arr = np.where((coastal_arr < -1) & (coastal_arr >= -5), sub_1cm_to_5cm, coastal_arr)
            coastal_arr = np.where(coastal_arr < -5, sub_greater_5cm, coastal_arr)
            coastal_arr = coastal_arr.flatten()

            final_subsidence_arr = final_subsidence_arr.flatten()
            final_subsidence_arr = np.where(final_subsidence_arr > 0, final_subsidence_arr, coastal_arr)

            # filtering pixels of coastal subsidence that has been added to final subsidence raster
            coastal_arr_used = np.where(np.logical_and(coastal_arr > 0, coastal_arr == final_subsidence_arr),
                                        coastal_arr, np.nan)
            coastal_arr_used = coastal_arr_used.reshape(ref_file.shape[0], ref_file.shape[1])
            coastal_subsidence_raster = '../InSAR_Data/Final_subsidence_data/resampled_insar_data' \
                                        '/Coastal_subsidence.tif'
            write_raster(coastal_arr_used, ref_file, ref_file.transform, coastal_subsidence_raster, ref_file=refraster)

            final_subsidence_arr = final_subsidence_arr.reshape(ref_file.shape[0], ref_file.shape[1])
            write_raster(final_subsidence_arr, ref_file, ref_file.transform, subsidence_data, ref_file=refraster)
            print('Processed Coastal SUbsidence (GNSS) Data')

        print('Created Final Subsidence Raster')
        return subsidence_data

    else:
        subsidence_data = os.path.join(output_dir, final_subsidence_raster)
        return subsidence_data