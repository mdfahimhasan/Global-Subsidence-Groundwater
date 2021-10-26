import os
import pandas as pd
import numpy as np
from Raster_operations import read_raster_arr_object, write_raster


def create_training_stats(train_test_csv, exclude_columns=['ALOS_Landform', 'Grace', 'Surfacewater_proximity'],
                          pred_attr='Subsidence',
                          outdir ='../Model Run/Model'):
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
                      'MODIS_PET': 'MODIS PET (mm)', 'NDWI' : 'NDWI', 'Population_Density': 'Population Density',
                      'SRTM_Slope': 'Slope (%)', 'Subsidence': 'Subsidence (cm/yr)',
                      'TRCLM_PET': 'PET (mm)', 'TRCLM_precp': 'Precipitation (mm)',
                      'TRCLM_soil': 'Soil moisture (mm)', 'TRCLM_Tmax' : 'Tmax (deg C)', 'TRCLM_Tmin' : 'Tmin (deg C)'}

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

# create_training_stats(csv)


def stat_clay_data(clay_0cm, clay_10cm, clay_30cm, clay_60cm, clay_100cm, clay_200cm,
                   subsidence_training_data='../InSAR_Data/Resampled_subsidence_data/final_subsidence_raster/'
                                            'Subsidence_training.tif'):
    clay_list = [clay_0cm, clay_10cm, clay_30cm, clay_60cm, clay_100cm, clay_200cm]
    Name = ['Clay_0cm', 'Clay_10cm', 'Clay_30cm', 'Clay_60cm', 'Clay_100cm', 'Clay_200cm']

    subsidence_arr = read_raster_arr_object(subsidence_training_data, get_file=False)
    subsidence_arr = np.isnan(subsidence_arr)

    Min = []
    Max = []
    Median = []
    Quartile = []
    for layer in clay_list:
        clay_arr = read_raster_arr_object(layer, get_file=False)
        clay_arr[subsidence_arr] = np.nan
        min = np.nanmin(clay_arr)
        max = np.nanmax(clay_arr)
        median = np.nanmedian(clay_arr)
        first_quantile = np.nanquantile(clay_arr, 0.25)
        Min.append(min)
        Max.append(max)
        Median.append(median)
        Quartile.append(first_quantile)

    clay_stat_df = pd.DataFrame(list(zip(Name, Min, Median, Max, Quartile)),
                                columns=['Dataname', 'min value', 'median value', 'max value', 'Quartile_value'])
    print(clay_stat_df)

# stat_clay_data(Clay_0cm, Clay_10cm, Clay_30cm, Clay_60cm, Clay_100cm, Clay_200cm)



# # Ensembling Model Outputs

RF1_arr, RF1_file = read_raster_arr_object('../Model Run/Prediction_rasters/RF53_prediction_2013_2019.tif')
RF2_arr, RF2_file = read_raster_arr_object('../Model Run/Prediction_rasters/RF54_prediction_2013_2019.tif')

# RF1_proba = read_raster_arr_object('../Model Run/Prediction_rasters/RF53_proba_2013_2019.tif', get_file=False)
# RF2_proba= read_raster_arr_object('../Model Run/Prediction_rasters/RF54_proba_2013_2019.tif', get_file=False)
#
# shape = RF1_arr.shape
#
# RF1_arr = RF1_arr.flatten()
# RF2_arr = RF2_arr.flatten()
# RF1_proba = RF1_proba.flatten()
# RF2_proba = RF2_proba.flatten()
#
# nan_pos_dict = {'nan_pos': np.isnan(RF1_arr)}
#
# high_proba_arr = np.where((RF1_proba >= RF2_proba), RF1_arr, RF2_arr)
# high_proba_arr[nan_pos_dict['nan_pos']] = RF1_file.nodata
#
# high_proba_arr = high_proba_arr.reshape(shape)
#
# write_raster(high_proba_arr, RF1_file, RF1_file.transform, '../scratch_files/Ensemble_model_result.tif')


# # maximum prediction Raster

# max_arr = np.where(RF1_arr > RF2_arr, RF1_arr, RF2_arr)
# write_raster(max_arr, RF1_file, RF1_file.transform, '../scratch_files/maxsubsidence_model_result.tif')


# # Filtering Result with Clay % data (for maximum prediction raster)

# prediction = '../scratch_files/maxsubsidence_model_result.tif'
#
# prediction_arr, file = read_raster_arr_object(prediction)
#
#
# Clay_0cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_0cm/merged_rasters/' \
#            'clay_content_0cm_2013_2019.tif'
# Clay_10cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_10cm/merged_rasters/' \
#             'clay_content_10cm_2013_2019.tif'
# Clay_30cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_30cm/merged_rasters/' \
#             'clay_content_30cm_2013_2019.tif'
# Clay_60cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_60cm/merged_rasters/' \
#             'clay_content_60cm_2013_2019.tif'
# Clay_100cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_100cm/merged_rasters/' \
#              'clay_content_100cm_2013_2019.tif'
# Clay_200cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_200cm/merged_rasters/' \
#              'clay_content_200cm_2013_2019.tif'
#
# Clay_0cm_arr = read_raster_arr_object(Clay_0cm, get_file=False)
# Clay_10cm_arr = read_raster_arr_object(Clay_10cm, get_file=False)
# Clay_30cm_arr = read_raster_arr_object(Clay_30cm, get_file=False)
# Clay_60cm_arr = read_raster_arr_object(Clay_60cm, get_file=False)
# Clay_100cm_arr = read_raster_arr_object(Clay_100cm, get_file=False)
# Clay_200cm_arr = read_raster_arr_object(Clay_200cm, get_file=False)
#
# filtered_rf_51 = np.where((Clay_0cm_arr < 10) & (Clay_10cm_arr < 10) & (Clay_30cm_arr < 10) & (Clay_60cm_arr < 10) &
#                           (Clay_100cm_arr < 10) & (Clay_200cm_arr < 10) & (~np.isnan(prediction_arr)), 1,
#                           prediction_arr)
#
# write_raster(filtered_rf_51, file, file.transform, '../scratch_files/maxsubsidence_model_result_filtered.tif')