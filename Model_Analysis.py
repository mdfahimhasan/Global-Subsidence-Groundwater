import pandas as pd
import numpy as np
from Raster_operations import read_raster_arr_object
from System_operations import makedirs


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

    Returns : A csv with '% prediction on differnet land use' stat.
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


# prediction_landuse_stat(model_prediction='../Model Run/Prediction_rasters/RF86_prediction_2013_2019.tif',
#                         land_use='../Model Run/Predictors_2013_2019/MODIS_Land_Use.tif')

def stat_irrigation_datasets(gfsad_lu='../Data/Raw_Data/Land_Use_Data/Raw/'
                                      'Global Food Security- GFSAD1KCM/GFSAD1KCM.tif',
                             meier_irrigated='../Data/Raw_Data/Land_Use_Data/Raw/global_irrigated_areas/'
                                             'global_irrigated_areas.tif', outdir='../Model Run/Stats'):
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


stat_irrigation_datasets()