import os
import pandas as pd
import numpy as np
from Raster_operations import read_raster_arr_object
from System_operations import makedirs


def prediction_landuse_stat(model_prediction, land_use='../Model Run/Predictors_2013_2019/MODIS_Land_Use.tif',
                            subsidence_raster='../Model Run/Predictors_2013_2019/Subsidence.tif'):
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
    subsidence_raster : filepath of training subsidence raster.

    Returns : A csv with '% prediction on differnet land use' stat.
    """
    subsidence_prediction = read_raster_arr_object(model_prediction, get_file=False)
    land_use = read_raster_arr_object(land_use, get_file=False)
    subsidence = read_raster_arr_object(subsidence_raster, get_file=False)

    cropland = np.count_nonzero(np.where(land_use == 3, True, False))
    urban = np.count_nonzero(np.where(land_use == 4, True, False))
    vegetation = np.count_nonzero(np.where(land_use == 2, True, False))
    others = np.count_nonzero(np.where((land_use != 2) | (land_use != 3) | (land_use != 4) & (land_use != np.nan),
                                       True, False))

    training_data_on_cropland = np.count_nonzero(np.where((subsidence == 5) | (subsidence == 10)
                                                          & (land_use == 3), True, False))
    prediction_on_cropland = np.count_nonzero(np.where((subsidence_prediction == 5) | (subsidence_prediction == 10)
                                                       & (land_use == 3), True, False))
    perc_training_subsidence_on_cropland = round(training_data_on_cropland * 100 / cropland, 2)
    perc_subsidence_on_cropland = round(prediction_on_cropland * 100 / cropland, 2)

    training_data_on_urban = np.count_nonzero(np.where((subsidence == 5) | (subsidence == 10)
                                                       & (land_use == 4), True, False))
    prediction_on_urban = np.count_nonzero(np.where((subsidence_prediction == 5) | (subsidence_prediction == 10)
                                                    & (land_use == 4), True, False))
    perc_training_subsidence_on_urban = round(training_data_on_urban * 100 / urban, 2)
    perc_subsidence_on_urban = round(prediction_on_urban * 100 / urban, 2)

    training_data_on_vegetation = np.count_nonzero(np.where((subsidence == 5) | (subsidence == 10)
                                                            & (land_use == 2), True, False))
    prediction_on_vegetation = np.count_nonzero(np.where((subsidence_prediction == 5) | (subsidence_prediction == 10)
                                                         & (land_use == 2), True, False))
    perc_training_subsidence_on_vegetation = round(training_data_on_vegetation * 100 / vegetation, 2)
    perc_subsidence_on_vegetation = round(prediction_on_vegetation * 100 / vegetation, 2)

    training_data_on_others = np.count_nonzero(np.where(((subsidence == 5) | (subsidence == 10))
                                                        & ((land_use != 2) | (land_use != 3) | (land_use != 4) &
                                                        (land_use != np.nan)), True, False))
    prediction_on_others = np.count_nonzero(np.where(((subsidence_prediction == 5) | (subsidence_prediction == 10))
                                                     & ((land_use != 2) | (land_use != 3) | (land_use != 4) &
                                                     (land_use != np.nan)), True, False))
    perc_training_subsidence_on_others = round(training_data_on_others * 100 / others, 2)
    perc_subsidence_on_others = round(prediction_on_others * 100 / others, 2)

    stat_dict = {'% Training data on Cropland': [perc_training_subsidence_on_cropland],
                 '% Subsidence on Cropland': [perc_subsidence_on_cropland],
                 '% Training data on Urban': [perc_training_subsidence_on_urban],
                 '% Subsidence on Urban': [perc_subsidence_on_urban],
                 '% Training data on Vegetation': [perc_training_subsidence_on_vegetation],
                 '% Subsidence on Vegetation': [perc_subsidence_on_vegetation],
                 '% Training data on Others': [perc_training_subsidence_on_others],
                 '% Subsidence on Others': [perc_subsidence_on_others]}
    stat_df = pd.DataFrame(stat_dict, index=None)
    print(stat_df)

    outdir = '../Model Run/Stats'
    makedirs([outdir])
    outcsv = outdir + '/' + 'Subsidence_on_LandUse.xlsx'
    stat_df.to_excel(outcsv, index=False)


# prediction_landuse_stat(model_prediction='../Model Run/Prediction_rasters/RF86_prediction_2013_2019.tif',
#                         land_use='../Model Run/Predictors_2013_2019/MODIS_Land_Use.tif')
