# Author: Md Fahim Hasan
# Email: Fahim.Hasan@colostate.edu

import os
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping
from System_operations import makedirs
from Raster_operations import read_raster_arr_object, write_raster, mask_by_ref_raster, clip_resample_raster_cutline, \
    paste_val_on_ref_raster


def prediction_landuse_stat(model_prediction, land_use='../Model Run/Predictors_2013_2019/MODIS_Land_Use.tif',
                            training_raster='../Model Run/Predictors_2013_2019/Subsidence.tif'):
    """
    Calculates percentage of subsidence prediction on different land use types.

    Land Use Classes in MODIS Data:
    1 - Forest
    2 - Vegetation
    3 - Cropland
    4 - Urban and built-Up
    5 - Snow and Ice
    6 - Barren land
    7 - Water body

    Parameters:
    model_prediction : filepath of subsidence prediction raster.
    land_use : filepath of MODIS land use raster.
    training_raster : filepath of training subsidence raster.

    Returns : An excel file with '% prediction on different land use' stat.
    """
    subsidence_prediction = read_raster_arr_object(model_prediction, get_file=False)
    land_use = read_raster_arr_object(land_use, get_file=False)
    training = read_raster_arr_object(training_raster, get_file=False)

    training_samples_5_10 = np.count_nonzero(np.where((training == 5) | (training == 10), 1, 0))
    subsidence_prediction_5_10 = np.count_nonzero(np.where((subsidence_prediction == 5) | (subsidence_prediction == 10),
                                                           1, 0))

    training_of_cropland = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                     & (land_use == 3), 1, 0))
    prediction_of_cropland = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                       & (land_use == 3), 1, 0))
    perc_cropland_in_training = round(training_of_cropland * 100 / training_samples_5_10, 2)
    perc_cropland_in_subsidence = round(prediction_of_cropland * 100 / subsidence_prediction_5_10, 2)

    training_of_urban = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                  & (land_use == 4), 1, 0))
    prediction_of_urban = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                    & (land_use == 4), 1, 0))
    perc_urban_in_training = round(training_of_urban * 100 / training_samples_5_10, 2)
    perc_urban_in_subsidence = round(prediction_of_urban * 100 / subsidence_prediction_5_10, 2)

    training_of_vegetation = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                       & (land_use == 2), 1, 0))
    prediction_of_vegetation = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                         & (land_use == 2), 1, 0))
    perc_vegetation_in_training = round(training_of_vegetation * 100 / training_samples_5_10, 2)
    perc_vegetation_in_subsidence = round(prediction_of_vegetation * 100 / subsidence_prediction_5_10, 2)

    training_of_others = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                   & ((land_use == 1) | (land_use == 5) | (land_use == 6) |
                                                      (land_use == 7)), 1, 0))
    prediction_of_others = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                     & ((land_use == 1) | (land_use == 5) | (land_use == 6) |
                                                        (land_use == 7)), 1, 0))
    perc_others_in_training = round(training_of_others * 100 / training_samples_5_10, 4)
    perc_others_in_subsidence = round(prediction_of_others * 100 / subsidence_prediction_5_10, 2)

    stat_dict = {'% of Training from Cropland': [perc_cropland_in_training],
                 '% Predicted on Cropland': [perc_cropland_in_subsidence],
                 '% of Training from Urban': [perc_urban_in_training],
                 '% Predicted on Urban': [perc_urban_in_subsidence],
                 '% of Training from Vegetation': [perc_vegetation_in_training],
                 '% Predicted on Vegetation': [perc_vegetation_in_subsidence],
                 '% of Training from Others': [perc_others_in_training],
                 '% Predicted on Others': [perc_others_in_subsidence],
                 ' ': ['cells before nan removal'],
                 'training cells Cropland': [training_of_cropland],
                 'training cells Urban': [training_of_urban],
                 'training cells Vegetation': [training_of_vegetation],
                 'training cells Others': [training_of_others],
                 'Total >1cm cells': [training_of_cropland + training_of_urban + training_of_vegetation +
                                      training_of_others]
                 }

    stat_df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['percent'])
    print(stat_df)

    outdir = '../Model Run/Stats'
    makedirs([outdir])
    out_excel = outdir + '/' + 'Subsidence_on_LandUse.xlsx'
    stat_df.to_excel(out_excel, index=True)


# prediction_landuse_stat(model_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019.tif',
#                         land_use='../Model Run/Predictors_2013_2019/MODIS_Land_Use.tif')


def stat_irrigation_datasets(gfsad_lu='../Data/Raw_Data/Land_Use_Data/Raw/'
                                      'Global Food Security- GFSAD1KCM/GFSAD1KCM.tif',
                             meier_irrigated='../Data/Raw_Data/Land_Use_Data/Raw/global_irrigated_areas/'
                                             'global_irrigated_areas.tif', outdir='../Model Run/Stats'):
    """
    Comparison between two irrigation datasets (GFSAD irrigation and Meier irrigation).

    Parameters :
    gfsad_lu : GFSAD irrigation data filepath.
    meier_irrigated : Meier irrigation data filepath.
    outdir : Output directory to save created excel file.

    Returns : An excel file with stats calculated.
    """
    gfsad_raster = read_raster_arr_object(gfsad_lu, get_file=False)
    meier_raster = read_raster_arr_object(meier_irrigated, get_file=False)

    # in gfsad_major only major irrigation (areas irrigated by large reservoirs created by large and medium dams,
    # barrages, and even large ground water pumping
    gfsad_major = np.count_nonzero(np.where((gfsad_raster == 1), 1, 0))
    gfsad_all = np.count_nonzero(np.where((gfsad_raster == 1) | (gfsad_raster == 2), 1, 0))

    # in meier_high_suitability only high suitability classes, low agricultural suitability not considered
    meier_high_suitability = np.count_nonzero(np.where((meier_raster == 1) | (meier_raster == 3), 1, 0))
    meier_all = np.count_nonzero(meier_raster)

    perc_higher_meier_from_gfsad = round((meier_high_suitability - gfsad_major) * 100 / gfsad_major, 2)
    perc_higher_gfsad_from_meier = round((gfsad_all - meier_all) * 100 / meier_all, 2)
    dict = {'GFSAD major irrigation': gfsad_major,
            'Meier high suitability': meier_high_suitability,
            'GFSAD major+minor irrigation': gfsad_all,
            'Meier all irrigation': meier_all,
            ' ': ['percent'],
            'Meirer high suitablity higher than GFSAD major (%)': perc_higher_meier_from_gfsad,
            'GFSAD all irrigated higher than Meier all (%)': perc_higher_gfsad_from_meier}
    df = pd.DataFrame.from_dict(dict, orient='index', columns=['cells'])
    print(df)

    makedirs([outdir])
    out_excel = outdir + '/' + 'Irrigation_datasets_stat.xlsx'
    df.to_excel(out_excel, index=True)


# stat_irrigation_datasets()


def overlap_all_irrigation_gw_irrigation(irrigated_area_meier='../Data/Raw_Data/Land_Use_Data/Raw/'
                                                              'global_irrigated_areas/global_irrigated_areas.tif',
                                         irrigated_area_gfsad='../Data/Raw_Data/Land_Use_Data/Raw/'
                                                              'Global Food Security- GFSAD1KCM/GFSAD1KCM.tif',
                                         gw_irrigation='../Data/Raw_Data/Land_Use_Data/Raw/gmlulca_10classes_global/'
                                                       'gmlulca_10classes_global.tif', outdir='../Model Run/Stats'):
    """
    Counting overlap between Irrigation data (Meier), Irrigation data (GFSAD) and GW irrigation data (GIAM).

    Parameters :
    irrigated_area_meier : Meier irrigated data filepath.
    irrigated_area_gfsad : GFSAD irrigated data filepath.
    gw_irrigation : GIAM GW irrigation data filepath.
    outdir : Output directory to save created excel file.

    Returns : An excel file with stats calculated.
    """
    irrigation_data = mask_by_ref_raster(irrigated_area_meier,
                                         '../Data/Raw_Data/Land_Use_Data/Raw/global_irrigated_areas',
                                         'global_irrigated_area_ref_clipped.tif')
    meier_arr = read_raster_arr_object(irrigation_data, get_file=False)

    irrigation_gfsad = mask_by_ref_raster(irrigated_area_gfsad, '../Data/Raw_Data/Land_Use_Data/Raw/'
                                                                'Global Food Security- GFSAD1KCM',
                                          'GFSAD1KCM_ref_clipped.tif')
    gfsad_arr = read_raster_arr_object(irrigation_gfsad, get_file=False)

    gw_irrigation_data = mask_by_ref_raster(gw_irrigation,
                                            '../Data/Raw_Data/Land_Use_Data/Raw/gmlulca_10classes_global',
                                            'gmlulca_10classes_global_ref_clipped.tif')
    gw_irrigation_raster = read_raster_arr_object(gw_irrigation_data, get_file=False)

    meier_arr = np.where(meier_arr != 0, True, False)
    meier_arr_count = np.count_nonzero(meier_arr)

    gfsad_arr_all = np.where((gfsad_arr == 1) | (gfsad_arr == 2), True, False)
    gfsad_arr_major = np.where(gfsad_arr == 1, True, False)
    gfsad_arr_count = np.count_nonzero(gfsad_arr_all)
    gfsad_major_arr_count = np.count_nonzero(gfsad_arr_major)

    gw_irrigation = np.where(gw_irrigation_raster == 2, True, False)  # major irrigation (gw) considered
    gw_irrigation_count = np.count_nonzero(gw_irrigation)

    overlap_irrigation_meier_gfsad_major_irrigation = np.count_nonzero(np.logical_and(meier_arr, gfsad_arr_major))
    overlap_irrigation_meier_gfsad_irrigation = np.count_nonzero(np.logical_and(meier_arr, gfsad_arr_all))
    overlap_irrigation_meier_gw_irrigation = np.count_nonzero(np.logical_and(meier_arr, gw_irrigation))
    overlap_irrigation_gfsad_gw_irrigation = np.count_nonzero(np.logical_and(gfsad_arr_major, gw_irrigation))

    dict = {'Number of cells in irrigated Meier data': meier_arr_count,
            'Number of cells in irrigated GFSAD (major) data': gfsad_major_arr_count,
            'Number of cells in irrigated GFSAD data': gfsad_arr_count,
            'Number of cells in GW irrigated data': gw_irrigation_count,
            'Overlapped cells between Meier-GFSAD (major) data': overlap_irrigation_meier_gfsad_major_irrigation,
            'Overlapped cells between Meier-GFSAD data': overlap_irrigation_meier_gfsad_irrigation,
            'Overlapped cells between Meier-GW data': overlap_irrigation_meier_gw_irrigation,
            'Overlapped cells between GFSAD (major)-GW data': overlap_irrigation_gfsad_gw_irrigation}

    df = pd.DataFrame.from_dict(dict, orient='index', columns=['Cells'])
    print(df)

    makedirs([outdir])
    out_excel = outdir + '/' + 'Irrigation_GW_irrigation_overlap.xlsx'
    df.to_excel(out_excel, index=True)


# overlap_all_irrigation_gw_irrigation()


def area_subsidence_by_country(subsidence_prediction, outdir='../Model Run/Stats'):
    """
    Estimated area of subsidence >1cm/yr by country.

    Parameters:
    subsidence_prediction : Subsidence prediction raster path.
    outdir : Directory path to save output excel file.

    Returns : An excel file with calculated stats.
    """
    outdir_country_arr = outdir + '/country_predictions'
    makedirs([outdir, outdir_country_arr])

    country_shapes = glob('../Data/Reference_rasters_shapes/Country_shapes/Individual_country' + '/' + '*.shp')

    # Area Calculation (1 deg = ~ 111km)
    deg_002 = 111 * 0.02  # unit km
    area_per_002_pixel = deg_002 ** 2

    area_sqkm = []
    country_name = []
    area_subsidence = []
    for shape in country_shapes:
        country = gpd.read_file(shape)
        name = shape[shape.rfind(os.sep) + 1:shape.rfind('.')]
        area_sqkm.append(country['Area_sqkm'].values[0])
        country_name.append(name)
        save_clipped_raster_as = name + '.tif'

        country_arr, country_file = clip_resample_raster_cutline(subsidence_prediction, outdir_country_arr,
                                                                 shape, naming_from_both=False,
                                                                 naming_from_raster=False,
                                                                 assigned_name=save_clipped_raster_as)

        prediction_1_to_5 = np.count_nonzero(np.where(country_arr == 5, 1, 0))
        prediction_greater_5 = np.count_nonzero(np.where(country_arr == 10, 1, 0))
        prediction_greater_1 = np.count_nonzero(np.where(country_arr > 1, 1, 0))

        area_prediction_1_to_5 = round(prediction_1_to_5 * area_per_002_pixel, 0)
        area_prediction_greater_5 = round(prediction_greater_5 * area_per_002_pixel, 0)
        area_prediction_greater_1 = round(prediction_greater_1 * area_per_002_pixel, 0)
        area_subsidence.append([area_prediction_greater_1, area_prediction_1_to_5, area_prediction_greater_5])

    stat_dict = {'country_name': country_name,
                 'area_sqkm': area_sqkm,
                 'area subsidence >1cm/yr': [i[0] for i in area_subsidence],
                 'area subsidence 1-5cm/yr': [i[1] for i in area_subsidence],
                 'area subsidence >5cm/yr': [i[2] for i in area_subsidence]}
    stat_df = pd.DataFrame(stat_dict)
    stat_df['perc_subsidence_of_cntry_area'] = round(stat_df['area subsidence >1cm/yr'] * 100 / stat_df['area_sqkm'], 4)
    stat_df = stat_df.sort_values(by='area subsidence >1cm/yr', ascending=False)
    stat_df.to_excel(os.path.join(outdir, 'subsidence_area_by_country.xlsx'), index=False)


# area_subsidence_by_country(
#     subsidence_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019.tif')


def subsidence_on_aridity(subsidence_prediction, outdir='../Model Run/Stats'):
    """
    Estimated area of subsidence of >1cm/yr in different aridity regions.

    Aridity Index Value	Climate Class
    <0.03	                 Hyper Arid
    0.03-0.2	               Arid
    0.2-0.5	                 Semi-Arid
    0.5-0.65	           Dry sub-humid
    >0.65	                   Humid

    Parameters:
    subsidence_prediction : Subsidence prediction raster path.
    outdir : Directory path to save output excel file.

    Returns : An excel file with calculated stats.
    """
    makedirs([outdir])
    aridity = read_raster_arr_object('../Model Run/Predictors_2013_2019/Aridity_Index.tif', get_file=False)
    subsidence_arr = read_raster_arr_object(subsidence_prediction, get_file=False)

    subsidence_pixels = np.count_nonzero(np.where(subsidence_arr > 1, 1, 0))

    hyper_arid = np.count_nonzero(np.where(((subsidence_arr > 1) & (aridity < 0.03)), 1, 0))
    arid = np.count_nonzero(np.where(((subsidence_arr > 1) & ((0.03 <= aridity) & (aridity < 0.2))), 1, 0))
    semi_arid = np.count_nonzero(np.where(((subsidence_arr > 1) & ((0.2 <= aridity) & (aridity < 0.5))), 1, 0))
    dry_subhumid = np.count_nonzero(np.where(((subsidence_arr > 1) & ((0.5 <= aridity) & (aridity < 0.65))), 1, 0))
    humid = np.count_nonzero(np.where(((subsidence_arr > 1) & (aridity > 0.65)), 1, 0))

    perc_hyper_arid = hyper_arid * 100 / subsidence_pixels
    perc_arid = arid * 100 / subsidence_pixels
    perc_semi_arid = semi_arid * 100 / subsidence_pixels
    perc_dry_subhumid = dry_subhumid * 100 / subsidence_pixels
    perc_humid = humid * 100 / subsidence_pixels

    aridity = ['Hyper Arid', 'Arid', 'Semi-Arid', 'Dry sub-humid', 'Humid']
    perc_subsidence = [perc_hyper_arid, perc_arid, perc_semi_arid, perc_dry_subhumid, perc_humid]

    df = pd.DataFrame(list(zip(aridity, perc_subsidence)), columns=['Aridity Class', '% Subsidence on Aridity'])
    df.to_excel(os.path.join(outdir, 'subsidence_perc_by_aridity.xlsx'), index=False)


# subsidence_on_aridity(subsidence_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019.tif')


def classify_gw_depletion_data(input_raster='../Data/result_comparison_Wada/georeferenced/gw_depletion_cmyr.tif',
                               referenceraster='../Data/Reference_rasters_shapes/Global_continents_ref_raster.tif'):
    """
    Classify groundwater depletion data from Wada et al. 2010.

    Parameters:
    input_raster : Filepath of input raster.
    referenceraster : Filepath of global reference raster.

    Returns : Classified and resampled depletion data.
    """
    resampled_raster = mask_by_ref_raster(input_raster, outdir='../Data/result_comparison_Wada/georeferenced',
                                          raster_name='gw_depletion_cmyr_resampled.tif',
                                          ref_raster=referenceraster, resolution=0.02, nodata=-9999,
                                          paste_on_ref_raster=False)

    depletion_arr, depletion_file = read_raster_arr_object(resampled_raster)

    # New_classes
    dep_less_1cm = 1
    dep_1cm_to_5cm = 5
    dep_greater_5cm = 10
    other_values = np.nan

    depletion_arr = np.where(depletion_arr <= 0, other_values, depletion_arr)
    depletion_arr = np.where(depletion_arr <= 1, dep_less_1cm, depletion_arr)
    depletion_arr = np.where((depletion_arr > 1) & (depletion_arr <= 5), dep_1cm_to_5cm, depletion_arr)
    depletion_arr = np.where(depletion_arr > 5, dep_greater_5cm, depletion_arr)

    write_raster(depletion_arr, depletion_file, depletion_file.transform,
                 '../Data/result_comparison_Wada/georeferenced/gw_depletion_cmyr_classified.tif')


# classify_gw_depletion_data()


def comparison_subsidence_depletion(
        subsidence_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019.tif',
        depletion_data='../Data/result_comparison_Wada/georeferenced/gw_depletion_cmyr_classified.tif',
        outdir='../Model Run/Stats/prediction_comparison'):
    """
    Compare model prediction with groundwater depletion data from Wada et al. 2010.

    Classes:
    1 - subsidence prediction and depletion both are >1cm/yr
    2 - subsidence prediction is >1cm/yr, no depletion estimated.
    3 - depletion is >1cm/yr, no subsidence predicted.

    Parameters:
    subsidence_prediction: Filepath of model subsidence prediction.
    depletion_data: Filepath of classified depletion data.
    outdir : Filepath of output directory.

    Returns: A comparison raster with 3 classes (1, 2, 3).
    """
    subsidence_arr, file = read_raster_arr_object(subsidence_prediction)
    depletion_arr = read_raster_arr_object(depletion_data, get_file=False)

    shape = subsidence_arr.shape
    subsidence_arr = subsidence_arr.flatten()
    depletion_arr = depletion_arr.flatten()

    subsidence_arr = np.where(subsidence_arr == 1, np.nan, subsidence_arr)
    depletion_arr = np.where(depletion_arr == 1, np.nan, depletion_arr)

    arr = np.where(~np.isnan(subsidence_arr), 2, np.nan)
    arr = np.where(~np.isnan(depletion_arr), 3, arr)
    arr = np.where(((subsidence_arr > 1) & (depletion_arr > 1)), 1, arr)

    arr = arr.reshape(shape)
    makedirs([outdir])
    interim_raster = \
        write_raster(arr, file, file.transform,
                     '../Model Run/Stats/prediction_comparison/subsidence_depletion_comparison_interim.tif')
    paste_val_on_ref_raster(interim_raster, outdir, 'subsidence_depletion_comparison_final.tif', value=0)


# comparison_subsidence_depletion()


def country_landuse_subsiding_stats(countries='../shapefiles/Country_continent_full_shapes/World_countries.shp',
                                    landuse='../Model Run/Predictors_2013_2019/MODIS_Land_Use.tif',
                                    model_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019.tif',
                                    outdir='../Model Run/Stats'):
    """
    calculate % of country's crop and urban areas subsiding. Used MODIS Land Use data where cropland=3 and urban=4.

    Parameters:
    countries: filepath of global country shapefile.
    landuse: filepath of land use raster data. Default set to MODIS Land Use data.
    model_prediction: filepath of model predicted subsidence. Default set to model 127.
    outdir: filepath of output directory.

    Returns: An excel file with country level subsidence stats on cropland and urban areas.
    """
    countries_df = gpd.read_file(countries)
    countries_df['geom_geojson'] = countries_df['geometry'].apply(mapping)

    landuse_arr, landuse_file = read_raster_arr_object(landuse)
    subsidence_arr, subsidence_file = read_raster_arr_object(model_prediction)

    def compute_num_cells_in_landuse(geom_geojson):
        masked_arr, masked_transform = mask(dataset=landuse_file, shapes=[geom_geojson], filled=True, crop=True,
                                            invert=False)
        masked_arr = masked_arr.squeeze()

        num_crop_pixels = np.count_nonzero(np.where(masked_arr == 3, 1, 0))
        num_urban_pixels = np.count_nonzero(np.where(masked_arr == 4, 1, 0))

        return num_crop_pixels, num_urban_pixels

    def compute_num_cells_of_landuse_in_subsidence(geom_geojson):
        lu_arr, landuse_transform = mask(dataset=landuse_file, shapes=[geom_geojson], filled=True, crop=True)
        lu_arr = lu_arr.squeeze()

        subside_arr, subside_transform = mask(dataset=subsidence_file, shapes=[geom_geojson], filled=True, crop=True)
        subside_arr = subside_arr.squeeze()

        num_crop_pixels_subsiding = np.count_nonzero(np.where(((lu_arr == 3) & ((subside_arr == 5) |
                                                                                (subside_arr == 10))), 1, 0))
        num_urban_pixels_subsiding = np.count_nonzero(np.where(((lu_arr == 4) & ((subside_arr == 5) |
                                                                                 (subside_arr == 10))), 1, 0))
        return num_crop_pixels_subsiding, num_urban_pixels_subsiding

    countries_df['num_crop_pixels'], countries_df['num_urban_pixels'] = \
        zip(*countries_df['geom_geojson'].apply(compute_num_cells_in_landuse))

    countries_df['num_crop_pixels_subsiding'], countries_df['num_urban_pixels_subsiding'] = \
        zip(*countries_df['geom_geojson'].apply(compute_num_cells_of_landuse_in_subsidence))

    countries_df['% subsiding crops'] = countries_df['num_crop_pixels_subsiding'] * 100 / countries_df[
        'num_crop_pixels']
    countries_df['% subsiding urban'] = \
        countries_df['num_urban_pixels_subsiding'] * 100 / countries_df['num_urban_pixels']

    countries_df = countries_df.sort_values(by='% subsiding crops', axis=0, ascending=False)
    countries_df.to_excel(os.path.join(outdir, 'country_subsidence_on_landuse.xlsx'))


# country_landuse_subsiding_stats()


def country_subsidence_on_aridity_stats(countries='../shapefiles/Country_continent_full_shapes/World_countries.shp',
                                        aridity='../Model Run/Predictors_2013_2019/Aridity_Index.tif',
                                        model_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019'
                                                         '.tif',
                                        outdir='../Model Run/Stats'):
    """
    Estimated % area of subsidence in different aridity regions of a country.

    Aridity Index Value	Climate Class
    <0.03	                 Hyper Arid
    0.03-0.2	               Arid
    0.2-0.5	                 Semi-Arid
    0.5-0.65	           Dry sub-humid
    >0.65	                   Humid

    Parameters:
    countries: filepath of global country shapefile.
    aridity: filepath of aridity raster data.
    model_prediction: filepath of model predicted subsidence. Default set to model 127.
    outdir: filepath of output directory.

    Returns: An excel file with country level aridity stats.
    """
    countries_df = gpd.read_file(countries)
    countries_df['geom_geojson'] = countries_df['geometry'].apply(mapping)

    aridity_arr, aridity_file = read_raster_arr_object(aridity)
    subsidence_arr, subsidence_file = read_raster_arr_object(model_prediction)

    def compute_num_cells_in_aridity(geom_geojson):
        arid_arr, arid_transform = mask(dataset=aridity_file, shapes=[geom_geojson], filled=True, crop=True,
                                        invert=False)
        arid_arr = arid_arr.squeeze()

        subside_arr, subside_transform = mask(dataset=subsidence_file, shapes=[geom_geojson], filled=True, crop=True)
        subside_arr = subside_arr.squeeze()

        hyper_arid = np.count_nonzero(np.where(((subside_arr > 1) & (arid_arr < 0.03)), 1, 0))
        arid = np.count_nonzero(np.where(((subside_arr > 1) & ((0.03 <= arid_arr) & (arid_arr < 0.2))), 1, 0))
        semi_arid = np.count_nonzero(np.where(((subside_arr > 1) & ((0.2 <= arid_arr) & (arid_arr < 0.5))), 1, 0))
        dry_subhumid = np.count_nonzero(np.where(((subside_arr > 1) & ((0.5 <= arid_arr) & (arid_arr < 0.65))), 1, 0))
        humid = np.count_nonzero(np.where(((subside_arr > 1) & (arid_arr > 0.65)), 1, 0))

        return hyper_arid, arid, semi_arid, dry_subhumid, humid

    countries_df['hyperarid_pixels'], countries_df['arid_pixels'], \
    countries_df['semiarid_pixels'], countries_df['drysubhumid_pixels'], countries_df['humid_pixels'] = \
        zip(*countries_df['geom_geojson'].apply(compute_num_cells_in_aridity))

    area_country_df = pd.read_excel('../Model Run/Stats/country_area_record_google.xlsx',
                                    sheet_name='countryarea_corrected')
    area_country_df = area_country_df[['country_name', 'area_sqkm_google']]

    new_df = countries_df.merge(area_country_df, how='left', left_on='CNTRY_NAME', right_on='country_name')
    new_df = new_df.drop(columns='country_name')

    # Area Calculation (1 deg = ~ 111km)
    deg_002 = 111 * 0.02  # unit km
    area_per_002_pixel = deg_002 ** 2

    new_df['perc_hyperarid_area'] = new_df['hyperarid_pixels'] * area_per_002_pixel * 100 / new_df['area_sqkm_google']
    new_df['perc_arid_area'] = new_df['arid_pixels'] * area_per_002_pixel * 100 / new_df['area_sqkm_google']
    new_df['perc_semiarid_area'] = new_df['semiarid_pixels'] * area_per_002_pixel * 100 / new_df['area_sqkm_google']
    new_df['perc_drysubhumid_area'] = new_df['drysubhumid_pixels'] * area_per_002_pixel * 100 / new_df[
        'area_sqkm_google']
    new_df['perc_humid_area'] = new_df['humid_pixels'] * area_per_002_pixel * 100 / new_df['area_sqkm_google']

    new_df = new_df.sort_values(by='perc_semiarid_area', axis=0, ascending=False)
    new_df.to_excel(os.path.join(outdir, 'country_subsidence_on_aridity.xlsx'))


# country_subsidence_on_aridity_stats()


def compute_volume_gw_loss(countries='../shapefiles/Country_continent_full_shapes/World_countries.shp',
                           model_prediction='../Model Run/Prediction_rasters/RF127_prediction_2013_2019.tif',
                           outdir='../Model Run/Stats'):
    """
    Calculates average volume of permanent groundwater storage loss in confined aquifer country-wise.

    Parameters:
    countries: filepath of global country shapefile.
    model_prediction: filepath of model predicted subsidence. Default set to model 127.
    outdir: filepath of output directory.

    Returns: An excel file with country level average gw storage loss stats.
    """
    countries_df = gpd.read_file(countries)
    countries_df['geom_geojson'] = countries_df['geometry'].apply(mapping)

    subsidence_arr, subsidence_file = read_raster_arr_object(model_prediction)

    def compute_num_subsidence_pixel(geom_geojson):
        masked_arr, masked_transform = mask(dataset=subsidence_file, shapes=[geom_geojson], filled=True, crop=True,
                                            invert=False)
        masked_arr = masked_arr.squeeze()

        num_1_5_pixels = np.count_nonzero(np.where(masked_arr == 5, 1, 0))  # pixels with 1-5 cm/year subsidence
        num_10_pixels = np.count_nonzero(np.where(masked_arr == 10, 1, 0))  # pixels >5 cm/year subsidence

        return num_1_5_pixels, num_10_pixels

    countries_df['num 1-5cm/yr pixels'], countries_df['num >5cm/yr pixels'] = \
        zip(*countries_df['geom_geojson'].apply(compute_num_subsidence_pixel))

    # Area Calculation (1 deg = ~ 111km)
    deg_002 = 111 * 0.02  # unit km (1 side length of a pixel)
    area_per_002_pixel = deg_002 ** 2

    # Assumptions on average subsidence in moderate and high subsidence pixels
    avg_subsidence_1_5cm_yr = 3 / 100000  # unit in km/yr
    avg_subsidence_greater_5cm_yr = 10 / 100000  # unit in km/yr

    countries_df['vol avg gwloss in 1-5cm/yr (km3/yr)'] = countries_df['num 1-5cm/yr pixels'] * area_per_002_pixel * \
                                                          avg_subsidence_1_5cm_yr
    countries_df['vol avg gwloss in >5cm/yr (km3/yr)'] = countries_df['num >5cm/yr pixels'] * area_per_002_pixel * \
                                                         avg_subsidence_greater_5cm_yr
    countries_df['volume avg total gw loss (km3/yr)'] = countries_df['vol avg gwloss in 1-5cm/yr (km3/yr)'] + \
                                                        countries_df['vol avg gwloss in >5cm/yr (km3/yr)']

    countries_df.to_excel(os.path.join(outdir, 'country_gw_volume_loss.xlsx'))


# compute_volume_gw_loss()


def subsidence_on_TWS(subsidence_train_data='../Model Run/Predictors_2013_2019/Subsidence.tif',
                      grace_data='../Model Run/Predictors_2013_2019/Grace.tif',
                      output_file='../Model Run/Stats/Subsidence_on_TWS.xlsx'):
    """
    Calculates the number of subsidence training pixels on positive and negative TWS (Grace) values.

    Parameters:
    subsidence_train_data: Filepath of subsidence training data.
    grace_data: Filepath of grace TWS change data.
    output_file: Filepath of output excel.

    Returns: An excel consisting counts of subsidence pixels of positive and negative TWS values.
    """
    subsidence_arr = read_raster_arr_object(subsidence_train_data, get_file=False)
    grace_tws = read_raster_arr_object(grace_data, get_file=False)

    analysis_dict = {}

    analysis_dict['total_subsidence_pixel'] = np.count_nonzero(np.where(~np.isnan(subsidence_arr), 1, 0))

    # Subsidence training data on negative TWS
    analysis_dict['subsidence_negative_TWS'] = np.count_nonzero(~np.isnan(subsidence_arr) & (grace_tws < 0))
    analysis_dict['less_1_negative_TWS'] = np.count_nonzero((subsidence_arr == 1) & (grace_tws < 0))
    analysis_dict['one_five_negative_TWS'] = np.count_nonzero((subsidence_arr == 5) & (grace_tws < 0))
    analysis_dict['greater_five_negative_TWS'] = np.count_nonzero((subsidence_arr == 10) & (grace_tws < 0))

    # Subsidence training data on negative TWS
    analysis_dict['subsidence_positive_TWS'] = np.count_nonzero(~np.isnan(subsidence_arr) & (grace_tws >= 0))
    analysis_dict['less_1_positive_TWS'] = np.count_nonzero((subsidence_arr == 1) & (grace_tws >= 0))
    analysis_dict['one_five_positive_TWS'] = np.count_nonzero((subsidence_arr == 5) & (grace_tws >= 0))
    analysis_dict['greater_five_positive_TWS'] = np.count_nonzero((subsidence_arr == 10) & (grace_tws >= 0))

    analysis_df = pd.DataFrame(analysis_dict, index=[0])
    analysis_df.to_excel(output_file)


# subsidence_on_TWS()
