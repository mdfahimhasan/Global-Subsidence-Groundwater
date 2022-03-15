import os
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from System_operations import makedirs
from Raster_operations import read_raster_arr_object, mask_by_ref_raster, clip_resample_raster_cutline


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

    Returns : An excel file with '% prediction on differnet land use' stat.
    """
    subsidence_prediction = read_raster_arr_object(model_prediction, get_file=False)
    land_use = read_raster_arr_object(land_use, get_file=False)
    training = read_raster_arr_object(training_raster, get_file=False)

    cropland = np.count_nonzero(np.where(land_use == 3, True, False))
    urban = np.count_nonzero(np.where(land_use == 4, True, False))
    vegetation = np.count_nonzero(np.where(land_use == 2, True, False))
    others = np.count_nonzero(np.where(((land_use == 1) | (land_use == 5) | (land_use == 6) | (land_use == 7)) &
                                       (land_use != np.nan), True, False))

    training_of_cropland = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                     & (land_use == 3), 1, 0))
    prediction_of_cropland = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                       & (land_use == 3), 1, 0))
    perc_training_of_cropland = round(training_of_cropland * 100 / cropland, 2)
    perc_subsidence_of_cropland = round(prediction_of_cropland * 100 / cropland, 2)

    training_of_urban = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                  & (land_use == 4), 1, 0))
    prediction_of_urban = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                    & (land_use == 4), 1, 0))
    perc_training_of_urban = round(training_of_urban * 100 / urban, 2)
    perc_subsidence_of_urban = round(prediction_of_urban * 100 / urban, 2)

    training_of_vegetation = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                       & (land_use == 2), 1, 0))
    prediction_of_vegetation = np.count_nonzero(np.where((subsidence_prediction == 5) | (subsidence_prediction == 10)
                                                         & (land_use == 2), 1, 0))
    perc_training_of_vegetation = round(training_of_vegetation * 100 / vegetation, 2)
    perc_subsidence_of_vegetation = round(prediction_of_vegetation * 100 / vegetation, 2)

    training_of_others = np.count_nonzero(np.where(((training == 5) | (training == 10))
                                                   & ((land_use == 1) | (land_use == 5) | (land_use == 6) |
                                                      (land_use == 7)), 1, 0))
    prediction_of_others = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                     & ((land_use == 1) | (land_use == 5) | (land_use == 6) |
                                                        (land_use == 7)), 1, 0))
    perc_training_of_others = round(training_of_others * 100 / others, 4)
    perc_subsidence_of_others = round(prediction_of_others * 100 / others, 2)

    # Area Calculation (1 deg = ~ 111km)

    deg_002 = 111 * 0.02  # unit km
    area_per_002_pixel = deg_002 ** 2

    area_cropland = round(area_per_002_pixel * cropland, 0)
    area_training_cropland = round(area_per_002_pixel * training_of_cropland, 0)
    area_prediction_cropland = round(area_per_002_pixel * prediction_of_cropland, 0)
    area_urban = round(area_per_002_pixel * urban, 0)
    area_training_urban = round(area_per_002_pixel * training_of_urban, 0)
    area_prediction_urban = round(area_per_002_pixel * prediction_of_urban, 0)
    area_vegetation = round(area_per_002_pixel * vegetation, 0)
    area_training_vegetation = round(area_per_002_pixel * training_of_vegetation, 0)
    area_prediction_vegetation = round(area_per_002_pixel * prediction_of_vegetation, 0)
    area_others = round(area_per_002_pixel * others, 0)
    area_training_others = round(area_per_002_pixel * training_of_others, 0)
    area_prediction_others = round(area_per_002_pixel * prediction_of_others, 0)

    stat_dict = {'% of Training from Cropland': [perc_training_of_cropland],
                 '% Predicted on Cropland': [perc_subsidence_of_cropland],
                 '% of Training from Urban': [perc_training_of_urban],
                 '% Predicted on Urban': [perc_subsidence_of_urban],
                 '% of Training from Vegetation': [perc_training_of_vegetation],
                 '% Predicted on Vegetation': [perc_subsidence_of_vegetation],
                 '% of Training from Others': [perc_training_of_others],
                 '% Predicted on Others': [perc_subsidence_of_others],
                 ' ': ['cells before nan removal'],
                 'training cells Cropland': [training_of_cropland],
                 'training cells Urban': [training_of_urban],
                 'training cells Vegetation': [training_of_vegetation],
                 'training cells Others': [training_of_others],
                 'Total >1cm cells': [training_of_cropland + training_of_urban + training_of_vegetation +
                                      training_of_others],
                 '  ': ['km2'],  # unit of area
                 'Area Trained on Cropland': [area_training_cropland],
                 'Area Predicted on Cropland': [area_prediction_cropland],
                 'Area Cropland': [area_cropland],
                 'Area Trained on Urban': [area_training_urban],
                 'Area Predicted on Urban': [area_prediction_urban],
                 'Area urban': [area_urban],
                 'Area Trained on Vegetation': [area_training_vegetation],
                 'Area Predicted on vegetation': [area_prediction_vegetation],
                 'Area vegetation': [area_vegetation],
                 'Area Trained on Others': [area_training_others],
                 'Area Predicted on Others': [area_prediction_others],
                 'Area others': [area_others]}

    stat_df = pd.DataFrame.from_dict(stat_dict, orient='index', columns=['percent'])
    print(stat_df)

    outdir = '../Model Run/Stats'
    makedirs([outdir])
    out_excel = outdir + '/' + 'Subsidence_on_LandUse.xlsx'
    stat_df.to_excel(out_excel, index=True)


# prediction_landuse_stat(model_prediction='../Model Run/Prediction_rasters/RF125_prediction_2013_2019.tif',
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
                                                                 shape, naming_from_both=False, naming_from_raster=False,
                                                                 assigned_name=save_clipped_raster_as)

        prediction_1_to_5 = np.count_nonzero(np.where(country_arr == 5, 1, 0))
        prediction_greater_5 = np.count_nonzero(np.where(country_arr == 10, 1, 0))
        prediction_greater_1 = np.count_nonzero(np.where(country_arr > 1, 1, 0))

        area_prediction_1_to_5 = round(prediction_1_to_5 * area_per_002_pixel, 2)
        area_prediction_greater_5 = round(prediction_greater_5 * area_per_002_pixel, 2)
        area_prediction_greater_1 = round(prediction_greater_1 * area_per_002_pixel, 2)
        area_subsidence.append((int(round(area_prediction_greater_1, 0)), int(round(area_prediction_1_to_5, 0)),
                                int(round(area_prediction_greater_5, 0))))

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
#     subsidence_prediction='../Model Run/Prediction_rasters/RF125_prediction_2013_2019.tif')
