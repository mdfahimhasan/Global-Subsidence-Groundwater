import os
import pandas as pd
import numpy as np
from Raster_operations import read_raster_arr_object, write_raster
from System_operations import makedirs


def create_training_stats(train_test_csv, exclude_columns=['ALOS_Landform', 'Grace', 'Surfacewater_proximity'],
                          pred_attr='Subsidence',
                          outdir='../Model Run/Model'):
    """
    create a csv with stats of predictor datasets for training regions.

    Parameters :
    train_test_csv : filepath of train_test data csv (used for training and testing the model).
    exclude_columns : List of columns to exclude.
    pred_attr : Prediction column in the train_test data csv. Default set to 'Subsidence'.
    outdir : filepath to save the created csv.

    Returns : A csv file with predictor stats.
    """

    train_test_df = pd.read_csv(train_test_csv)
    train_test_df = train_test_df.drop(columns=exclude_columns + [pred_attr])
    predictor_dict = {'Alexi_ET': 'Alexi ET (mm)', 'Aridity_Index': 'Aridity Index',
                      'Clay_content_PCA': 'Clay_content_PCA', 'EVI': 'EVI',
                      'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                      'Global_Sed_Thickness_Exx': 'Sediment Thickness Exxon (km)',
                      'GW_Irrigation_Density_fao': 'GW Irrigation Density fao',
                      'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                      'Irrigated_Area_Density': 'Irrigated Area Density', 'MODIS_ET': 'MODIS ET (mm)',
                      'MODIS_PET': 'MODIS PET (mm)', 'NDWI': 'NDWI', 'Population_Density': 'Population Density',
                      'SRTM_Slope': 'Slope (%)', 'Subsidence': 'Subsidence (cm/yr)',
                      'TRCLM_PET': 'PET (mm)', 'TRCLM_precp': 'Precipitation (mm)',
                      'TRCLM_soil': 'Soil moisture (mm)', 'TRCLM_Tmax': 'Tmax (deg C)', 'TRCLM_Tmin': 'Tmin (deg C)'}

    Name = []
    Min = []
    Max = []
    Median = []
    first_quartile = []
    third_quartile = []
    No_count_1st_quantile = []
    No_count_2nd_quantile = []
    No_count_3rd_quantile = []

    for name, col_data in train_test_df.iteritems():
        name = predictor_dict[name]
        min = round(np.min(col_data.values), 2)
        max = round(np.max(col_data.values), 2)
        median = round(np.median(col_data.values), 2)
        quartile_1st = round(np.quantile(col_data.values, 0.25), 2)
        quartile_3rd = round(np.quantile(col_data.values, 0.75), 2)
        no_count_1st_quantile = col_data[col_data <= quartile_1st].count()
        no_count_2nd_quantile = col_data[(col_data > quartile_1st) & (col_data < quartile_3rd)].count()
        no_count_3rd_quantile = col_data[(col_data > quartile_3rd)].count()

        Name.append(name)
        Min.append(min)
        Max.append(max)
        Median.append(median)
        first_quartile.append(quartile_1st)
        third_quartile.append(quartile_3rd)
        No_count_1st_quantile.append(no_count_1st_quantile)
        No_count_2nd_quantile.append(no_count_2nd_quantile)
        No_count_3rd_quantile.append(no_count_3rd_quantile)

    stat_df = pd.DataFrame(list(zip(Name, Min, Max, first_quartile, Median, third_quartile, No_count_1st_quantile,
                                    No_count_2nd_quantile, No_count_3rd_quantile)),
                           columns=['Dataname', 'min value', 'max value', '1st quartile', 'median value',
                                    '3rd quartile', 'No_count_1st_quantile', 'No_count_2nd_quantile',
                                    'No_count_3rd_quantile'])
    stat_df.to_csv(os.path.join(outdir, 'predictor_stats.csv'), index=False)


csv = '../Model Run/Predictors_csv/train_test_2013_2019.csv'


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
