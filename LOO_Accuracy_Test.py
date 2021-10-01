# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from System_operations import makedirs
from Raster_operations import shapefile_to_raster, mosaic_rasters, mosaic_two_rasters, read_raster_arr_object, \
    write_raster

referenceraster2 = '../Data/Reference_rasters_shapes/Global_continents_ref_raster_002.tif'


def combine_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygons,
                                              search_criteria='*Subsidence*.shp', skip_polygon_processing=True):
    """
    Combining georeferenced subsidence polygons.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    search_criteria : Search criteria for input polygons.
    skip_polygon_processing : Set False if want to process georeferenced subsidence polygons.

    Returns : Joined subsidence polygon.
    """
    global gdf

    if not skip_polygon_processing:
        subsidence_polygons = glob(os.path.join(input_polygons_dir, search_criteria))

        sep = joined_subsidence_polygons.rfind(os.sep)
        makedirs([joined_subsidence_polygons[:sep]])  # creating directory for the  prepare_subsidence_raster function

        for each in range(1, len(subsidence_polygons) + 1):
            if each == 1:
                gdf = gpd.read_file(subsidence_polygons[each - 1])

            gdf_new = gpd.read_file(subsidence_polygons[each - 1])
            add_to_gdf = gdf.append(gdf_new, ignore_index=True)
            gdf = add_to_gdf
            gdf['Class_name'] = gdf['Class_name'].astype(float)

        unique_area_name = gdf['Area_name'].unique().tolist()
        unique_area_name_code = [i + 1 for i in range(len(unique_area_name))]
        polygon_area_name_dict = {}

        for name, code in zip(unique_area_name, unique_area_name_code):
            polygon_area_name_dict[name] = code

        Area_code = []
        for index, row in gdf.iterrows():
            Area_code.append(polygon_area_name_dict[row['Area_name']])
        gdf['Area_code'] = pd.Series(Area_code)

        gdf.to_file(joined_subsidence_polygons)

        pickle.dump(polygon_area_name_dict, open('../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                'polygon_area_name_dict.pkl', mode='wb+'))

        return joined_subsidence_polygons, polygon_area_name_dict

    else:
        joined_subsidence_polygons = '../InSAR_Data/Resampled_subsidence_data/' \
                                     'LOO_test_dir/georef_subsidence_polygons.shp'
        polygon_area_name_dict = pickle.load(open('../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                  'polygon_area_name_dict.pkl', mode='rb'))

        return joined_subsidence_polygons, polygon_area_name_dict


def substitute_area_code_on_raster(input_raster, value_to_substitute, output_raster):
    raster_arr, raster_file = read_raster_arr_object(input_raster)

    raster_arr = np.where(np.isnan(raster_arr), raster_arr, value_to_substitute)

    area_coded_raster = write_raster(raster_arr, raster_file, raster_file.transform, output_raster)

    return area_coded_raster


def combine_georef_insar_subsidence_raster(input_polygons_dir='../InSAR_Data/Georeferenced_subsidence_data',
                                           joined_subsidence_polygon='../InSAR_Data/Resampled_subsidence_data/'
                                                                     'LOO_test_dir/georef_subsidence_polygons.shp',
                                           insar_data_dir='../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                          'interim_working_dir',
                                           interim_dir='../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                       'interim_working_dir',
                                           output_dir='../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                      'final_subsidence_raster',
                                           skip_polygon_processing=False,
                                           area_code_column='Area_code',
                                           final_subsidence_raster='Subsidence_area_coded.tif',
                                           polygon_search_criteria='*Subsidence*.shp', already_prepared=False,
                                           refraster=referenceraster2):
    """
    Prepare area coded subsidence raster for training data by joining georeferenced polygons and insar data.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    insar_data_dir : InSAR data directory.
    interim_dir : Intermediate working directory for storing interdim data.
    output_dir : Output raster directory.
    skip_polygon_processing : Set to True if polygon merge is not required.
    final_subsidence_raster : Final subsidence raster including georeferenced and insar data.
    polygon_search_criteria : Input subsidence polygon search criteria.
    insar_search_criteria : InSAR data search criteria.
    already_prepared : Set to True if subsidence raster is already prepared.
    refraster : Global Reference raster.

    Returns : Final subsidence raster to be used as training data.
    """

    global subsidence_areaname_dict
    if not already_prepared:
        makedirs([interim_dir, output_dir])

        print('Processing area coded subsidence polygons...')
        subsidence_polygons, subsidence_areaname_dict = \
            combine_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygon,
                                                      polygon_search_criteria, skip_polygon_processing)


        print('Processed area coded subsidence polygons')
        subsidence_raster_area_coded = shapefile_to_raster(subsidence_polygons, interim_dir,
                                                           raster_name='interim_georef_subsidence_raster_areacode.tif',
                                                           burn_attr=True, attribute=area_code_column,
                                                           ref_raster=refraster, alltouched=False)

        print('Processing area coded InSAR data...')
        georef_subsidence_gdf = gpd.read_file(joined_subsidence_polygon)
        num_of_georef_subsidence = len(georef_subsidence_gdf)

        california_area_code = num_of_georef_subsidence + 1
        arizona_area_code = california_area_code + 1
        quetta_area_code = arizona_area_code + 1
        subsidence_areaname_dict['California'] = california_area_code
        subsidence_areaname_dict['Arizona'] = arizona_area_code
        subsidence_areaname_dict['Quetta'] = quetta_area_code

        california_subsidence = '../InSAR_Data/Resampled_subsidence_data/California_reclass_resampled.tif'
        arizona_subsidence = '../InSAR_Data/Resampled_subsidence_data/Arizona_reclass_resampled.tif'
        quetta_subsidence = '../InSAR_Data/Resampled_subsidence_data/Quetta_reclass_resampled.tif'

        substitute_area_code_on_raster(california_subsidence, california_area_code,
                                       '../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                       'interim_working_dir/California_area_raster.tif')
        substitute_area_code_on_raster(arizona_subsidence, arizona_area_code,
                                       '../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                       'interim_working_dir/Arizona_area_raster.tif')
        substitute_area_code_on_raster(quetta_subsidence, quetta_area_code,
                                       '../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                       'interim_working_dir/Quetta_area_raster.tif')

        insar_arr, merged_insar = mosaic_rasters(insar_data_dir, output_dir=insar_data_dir,
                                                 raster_name='joined_insar_Area_data.tif',
                                                 ref_raster=refraster, search_by='*area*.tif', resolution=0.02)

        final_subsidence_arr, subsidence_data = mosaic_two_rasters(merged_insar, subsidence_raster_area_coded,
                                                                   output_dir, final_subsidence_raster, resolution=0.02)
        print('Created final area coded subsidence raster')
        pickle.dump(subsidence_areaname_dict, open(os.path.join(output_dir, 'subsidence_areaname_dict.pkl'),
                                                   mode='wb+'))

        return subsidence_data, subsidence_areaname_dict

    else:
        subsidence_data = os.path.join(output_dir, final_subsidence_raster)
        subsidence_areaname_dict = pickle.load(open(os.path.join(output_dir, 'subsidence_areaname_dict.pkl'),
                                                    mode='rb'))
        return subsidence_data, subsidence_areaname_dict


def create_dataframe_for_loo_accuracy(input_raster_dir, output_csv, subsidence_areacode_dict,
                                      search_by='*.tif', skip_dataframe_creation=False):
    """
    create dataframe from predictor rasters along with area code.

    Parameters:
    input_raster_dir : Input rasters directory.
    output_csv : Output csv file with filepath.
    search_by : Input raster search criteria. Defaults to '*.tif'.
    skip_predictor_subsidence_compilation : Set to True if want to skip processing.

    Returns: predictor_df dataframe created from predictor rasters.
    """
    print('Creating area coded predictors csv...')
    if not skip_dataframe_creation:
        predictors = glob(os.path.join(input_raster_dir, search_by))

        predictor_dict = {}
        for predictor in predictors:
            variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind('.')]
            raster_arr, file = read_raster_arr_object(predictor, get_file=True)
            raster_arr = raster_arr.flatten()
            predictor_dict[variable_name] = raster_arr

        subsidence_area_arr, subsidence_area_file = \
            read_raster_arr_object('../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                   'final_subsidence_raster/Subsidence_area_coded.tif')
        predictor_dict['Area_code'] = subsidence_area_arr.flatten()

        predictor_df = pd.DataFrame(predictor_dict)
        predictor_df = predictor_df.dropna(axis=0)

        area_code = predictor_df['Area_code'].tolist()

        area_name_list = list(subsidence_areacode_dict.keys())
        area_code_list = list(subsidence_areacode_dict.values())

        area_name = []
        for code in area_code:
            position = area_code_list.index(code)
            name = area_name_list[position]
            area_name.append(name)

        predictor_df['Area_name'] = area_name
        predictor_df.to_csv(output_csv, index=False)

        print('Area coded predictors csv created')
        return predictor_df
    else:
        predictor_df = pd.read_csv(output_csv)
        return predictor_df


subsidence_raster, areaname_dict = combine_georef_insar_subsidence_raster(already_prepared=True,
                                                                          skip_polygon_processing=True)

predictor_raster_dir = '../Model Run/Predictors_2013_2019'
train_test_csv = '../Model Run/Predictors_csv/train_test_area_coded_2013_2019.csv'
create_dataframe_for_loo_accuracy(predictor_raster_dir, train_test_csv, areaname_dict)
