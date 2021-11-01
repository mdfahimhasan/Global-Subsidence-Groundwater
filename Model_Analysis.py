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

