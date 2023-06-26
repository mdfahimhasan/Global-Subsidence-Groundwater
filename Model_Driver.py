# Author: Md Fahim Hasan
# Email: Fahim.Hasan@colostate.edu

import timeit
import warnings
from ML_operations import *
from Data_operations import *
from Training_InSAR_processing import prepare_subsidence_raster


warnings.simplefilter(action='ignore', category=FutureWarning)  # to ignore future warning coming from pandas
start = timeit.default_timer()

gee_data_list = ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_soil', 'TRCLM_RET', 'MODIS_ET', 'MODIS_EVI',
                 'MODIS_NDWI', 'MODIS_PET', 'GPW_pop', 'SRTM_DEM', 'Aridity_Index', 'Grace',
                 'clay_content_0cm', 'clay_content_10cm', 'clay_content_30cm', 'clay_content_60cm',
                 'clay_content_100cm', 'clay_content_200cm', 'MODIS_Land_Use', 'TRCLM_ET', 'Grace']

yearlist = [2013, 2019]
start_month = 1
end_month = 12
resampled_dir = '../Data/Resampled_Data/GEE_data_2013_2019'
gfsad_lu = '../Data/Raw_Data/Land_Use_Data/Raw/Global Food Security- GFSAD1KCM/GFSAD1KCM.tif'
giam_lu = '../Data/Raw_Data/Land_Use_Data/Raw/gmlulca_10classes_global/gmlulca_10classes_global.tif'
irrigated_meier = '../Data/Raw_Data/Land_Use_Data/Raw/global_irrigated_areas/global_irrigated_areas.tif'
outdir_lu = '../Data/Resampled_Data/Land_Use'
intermediate_dir = '../Data/Intermediate_working_dir'
sediment_thickness = '../Data/Raw_Data/Global_Sediment_Thickness/average_soil_and_sedimentary-deposit_thickness.tif'
outdir_sed_thickness = '../Data/Resampled_Data/Sediment_Thickness'
outdir_pop = '../Data/Resampled_Data/Pop_Density'
river_shape = '../Data/Raw_Data/Surface_Water/mrb_shp/mrb_rivers.shp'
outdir_sw = '../Data/Resampled_Data/Surface_Water'
confining_layer = '../Data/Raw_Data/Global_confining_layer/global_confining_layer.tif'
outdir_confining_layers = '../Data/Resampled_Data/Global_confining_layers'

# # skip_download = False if new data needs to be downloaded from google earth engine
# # skip_processing = False if processing of already downloaded data (secondary processing) is required
gee_raster_dict, gfsad_raster, irrigated_meier_raster, giam_gw_raster, \
    sediment_thickness_raster, clay_thickness_raster, normalized_clay_indicator, popdensity_raster, \
    river_distance, confining_layers = \
    download_process_predictor_datasets(yearlist, start_month, end_month, resampled_dir,
                                        gfsad_lu, giam_lu, irrigated_meier, intermediate_dir,
                                        outdir_lu, sediment_thickness, outdir_sed_thickness,
                                        outdir_pop, river_shape, outdir_sw, confining_layer, outdir_confining_layers,
                                        perform_pca=False,  # #
                                        skip_download=True, skip_processing=False,  # #
                                        geedatalist=gee_data_list, downloadcsv=csv, gee_scale=2000)

input_polygons_dir = '../InSAR_Data/Georeferenced_subsidence_data'
joined_subsidence_polygon = '../InSAR_Data/Final_subsidence_data/interim_working_dir/georef_subsidence_polygons.shp'
insar_data_dir = '../InSAR_Data/Final_subsidence_data/resampled_insar_data'
EGMS_insar_dir = '../InSAR_Data/Europe_EGMS/Interim_processing'
interim_dir = '../InSAR_Data/Final_subsidence_data/interim_working_dir'
training_insar_dir = '../InSAR_Data/Final_subsidence_data/final_subsidence_raster'

# # skip_polygon_merge = False if new georeferenced subsidence polygons needs to be added
# # already prepared = False if new georeferenced subsidence polygons needs to be added/new InSAR processed subsidence
# # data has o be integrated
exclude_areas = None  # if all areas are to be included, set None.
include_insar_areas = ('California',  'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin', 'China_Hebei', 'China_Hefei',
                       'Colorado')

subsidence_raster =\
    prepare_subsidence_raster(input_polygons_dir=input_polygons_dir, joined_subsidence_polygon=joined_subsidence_polygon,
                              insar_data_dir=insar_data_dir, EGMS_insar_dir=EGMS_insar_dir, interim_dir=interim_dir,
                              output_dir=training_insar_dir, subsidence_column='Class_name',
                              resample_algorithm='near', final_subsidence_raster='Subsidence_training.tif',
                              exclude_georeferenced_areas=None,
                              process_insar_areas=('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin',
                                                   'China_Hebei', 'China_Hefei', 'Colorado'),
                              polygon_search_criteria='*Subsidence*.shp',
                              insar_search_criteria='*reclass_resampled*.tif',
                              skip_polygon_merge=True,  # #
                              already_prepared=True,  # #
                              merge_coastal_subsidence_data=True)

predictor_dir = '../Model Run/Predictors_2013_2019'

# # skip_compiling_predictor_subsidence_data = False if any change in predictors or subsidence data are made
predictor_dir = compile_predictors_subsidence_data(gee_raster_dict, gfsad_raster, giam_gw_raster,
                                                   irrigated_meier_raster, sediment_thickness_raster,
                                                   clay_thickness_raster, normalized_clay_indicator, popdensity_raster,
                                                   river_distance, confining_layers, subsidence_raster, predictor_dir,
                                                   skip_compiling_predictor_subsidence_data=False)  # #

csv_dir = '../Model Run/Predictors_csv'
makedirs([csv_dir])
train_test_csv = '../Model Run/Predictors_csv/train_test_2013_2019.csv'

# # skip_dataframe_creation = False if any change occur in predictors or subsidence data
predictor_df = create_dataframe(predictor_dir, train_test_csv, search_by='*.tif',
                                skip_dataframe_creation=False)  # #

modeldir = '../Model Run/Model'
model = 'rf'

# change for fitted_model run
exclude_columns = ['Alexi ET', 'MODIS ET (kg/m2)', 'Irrigated Area Density (gfsad)',
                   'GW Irrigation Density giam', 'MODIS PET (kg/m2)', 'Clay content PCA',
                   'MODIS Land Use', 'Grace', 'Sediment Thickness (m)', 'Clay % 200cm',
                   'Tmin (°C)', 'RET (mm)', 'Clay Thickness (m)']
# 'EVI', 'NDWI', 'Soil moisture (mm)', '% Slope', 'Precipitation (average monthly) (mm)',
# 'Tmax (°C)', 'ET (average monthly) (mm)']

variables_in_pdp = ('Soil moisture (mm)', 'River Distance (km)', 'Normalized Clay Indicator',
                    'Normalized Irrigated Area Density', 'Normalized Population Density', 'Tmax (°C)',
                    'Precipitation (average monthly) (mm)', 'ET (average monthly) (mm)', 'Confining Layers')
pdp_combinations = (('Normalized Irrigated Area Density', 'Normalized Clay Indicator'),
                    ('Precipitation (average monthly) (mm)', 'Soil moisture (mm)'))
prediction_raster_keyword = 'RF137'

# # predictor_importance = False if predictor importance plot is not required
# # plot_pdp = False if partial dependence plots are not required
# # plot_confusion_matrix = False if confusion matrix plot (as image) is not required
ML_model, predictor_name_dict = \
    build_ml_classifier(train_test_csv, modeldir, exclude_columns, model, load_model=False,
                        pred_attr='Subsidence', test_size=0.3, random_state=0, output_dir=csv_dir,
                        n_estimators=300, min_samples_leaf=1e-05, min_samples_split=7, max_depth=14, max_features=7,
                        max_samples=None, max_leaf_nodes=None, class_weight='balanced',
                        estimate_accuracy=True,
                        predictor_imp_keyword=prediction_raster_keyword,
                        predictor_importance=True,  # #
                        variables_pdp=variables_in_pdp, plot_pdp=True,  # #
                        pdp_combinations=pdp_combinations,
                        plot_confusion_matrix=True,  # #
                        tune_hyperparameter=False,  # #
                        k_fold=10, n_iter=80,
                        repeatedstratified=False,  # #  if False performs StratifiedKFold
                        random_searchCV=True)  # #

predictors_dir = '../Model Run/Predictors_2013_2019'

# # predictor_csv_exists =  False if new predictor has been added so new data has to be added or predictor
# # combination changes
# # predictor_probability_greater_1cm = False if probability plot is not required
create_prediction_raster(predictors_dir, ML_model, predictor_name_dict, yearlist=[2013, 2019], search_by='*.tif',
                         continent_search_by='*continent.shp',
                         continent_shapes_dir='../Data/Reference_rasters_shapes/continent_extents',
                         prediction_raster_dir='../Model Run/Prediction_rasters', exclude_columns=exclude_columns,
                         pred_attr='Subsidence', prediction_raster_keyword=prediction_raster_keyword,
                         predictor_csv_exists=False,  # #
                         predict_probability_greater_1cm=True)  # #


# # Applying land use filter on model subsidence prediction + subsidence probability prediction
model_subsidence_prediction = '../Model Run/Prediction_rasters/RF137_prediction_2013_2019.tif'
irrigation_data = '../Model Run/Predictors_2013_2019/Irrigated_Area_Density_meier.tif'
population_data = '../Model Run/Predictors_2013_2019/Population_Density.tif'

final_subsidence_prediction = '../Model Run/Prediction_rasters/Final_subsidence_prediction.tif'

apply_landuse_filter_on_prediction(model_subsidence_prediction=model_subsidence_prediction,
                                   irrigation_data=irrigation_data, population_data=population_data,
                                   final_subsidence_prediction=final_subsidence_prediction)


model_runtime = True
if model_runtime:
    stop = timeit.default_timer()
    print('Model Run Time :', round((stop - start) / 60, 2), 'min')
