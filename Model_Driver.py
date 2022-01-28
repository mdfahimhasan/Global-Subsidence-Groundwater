# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

from Data_operations import *
from ML_operations import *

gee_data_list = ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_soil', 'TRCLM_RET', 'MODIS_ET', 'MODIS_EVI',
                 'MODIS_NDWI', 'MODIS_PET', 'GPW_pop', 'SRTM_DEM', 'ALOS_Landform', 'Aridity_Index', 'Grace',
                 'clay_content_0cm', 'clay_content_10cm', 'clay_content_30cm', 'clay_content_60cm',
                 'clay_content_100cm', 'clay_content_200cm', 'MODIS_Land_Use', 'TRCLM_ET']

yearlist = [2013, 2019]
start_month = 1
end_month = 12
resampled_dir = '../Data/Resampled_Data/GEE_data_2013_2019'
gfsad_lu = '../Data/Raw_Data/Land_Use_Data/Raw/Global Food Security- GFSAD1KCM/GFSAD1KCM.tif'
giam_lu = '../Data/Raw_Data/Land_Use_Data/Raw/gmlulca_10classes_global/gmlulca_10classes_global.tif'
fao_lu = '../Data/Raw_Data/Land_Use_Data/Raw/FAO_LC/RasterFile/aeigw_pct_aei.tif'
outdir_lu = '../Data/Resampled_Data/Land_Use'
intermediate_dir = '../Data/Intermediate_working_dir'
sediment_thickness = '../Data/Raw_Data/Global_Sediment_Thickness/average_soil_and_sedimentary-deposit_thickness.tif'
outdir_sed_thickness = '../Data/Resampled_Data/Sediment_Thickness'
sediment_thickness_exx = '../Data/Raw_Data/EXXON_Sediment_Thickness/Global_Sediment_thickness_EXX.tif'
outdir_pop = '../Data/Resampled_Data/Pop_Density'

# skip_download = False if new data needs to be downloaded from google earth engine
# skip_processing = False if processing of already downloaded data (secondary processing) is required
gee_raster_dict, gfsad_raster, giam_gw_raster, fao_gw_raster, sediment_thickness_raster, \
sediment_thickness_raster_exxon, \
popdensity_raster = download_process_predictor_datasets(yearlist, start_month, end_month, resampled_dir,
                                                        gfsad_lu, giam_lu, fao_lu, intermediate_dir, outdir_lu,
                                                        sediment_thickness, outdir_sed_thickness,
                                                        sediment_thickness_exx,
                                                        outdir_pop, perform_pca=False,
                                                        skip_download=True, skip_processing=True,  # #
                                                        geedatalist=gee_data_list, downloadcsv=csv, gee_scale=2000)

input_polygons_dir = '../InSAR_Data/Georeferenced_subsidence_data'
joined_subsidence_polygon = '../InSAR_Data/Resampled_subsidence_data/interim_working_dir/georef_subsidence_polygons.shp'
insar_data_dir = '../InSAR_Data/Resampled_subsidence_data/resampled_insar_data'
interim_dir = '../InSAR_Data/Resampled_subsidence_data/interim_working_dir'
training_insar_dir = '../InSAR_Data/Resampled_subsidence_data/final_subsidence_raster'

# skip_polygon_merge = False if new georeferenced subsidence polygons needs to be added
# already prepared = False if new georeferenced subsidence polygons needs to be added or new InSAR processed subsidence
# data has o be integrated
subsidence_raster = prepare_subsidence_raster(input_polygons_dir, joined_subsidence_polygon,
                                              insar_data_dir, interim_dir, training_insar_dir,
                                              subsidence_column='Class_name',
                                              final_subsidence_raster='Subsidence_training.tif',
                                              polygon_search_criteria='*Subsidence*.shp',
                                              insar_search_criteria='*reclass_resampled*.tif',
                                              skip_polygon_merge=True,  # #
                                              already_prepared=True,  # #
                                              merge_coastal_subsidence_data=True)

predictor_dir = '../Model Run/Predictors_2013_2019'

# skip_compiling_predictor_subsidence_data = False if any change in predictors or subsidence data are made
predictor_dir = compile_predictors_subsidence_data(gee_raster_dict, gfsad_raster, giam_gw_raster, fao_gw_raster,
                                                   sediment_thickness_raster, popdensity_raster, subsidence_raster,
                                                   predictor_dir,
                                                   skip_compiling_predictor_subsidence_data=True)  # #

csv_dir = '../Model Run/Predictors_csv'
makedirs([csv_dir])
train_test_csv = '../Model Run/Predictors_csv/train_test_2013_2019.csv'

# skip_dataframe_creation = False if any change occur in predictors or subsidence data
predictor_df = create_dataframe(predictor_dir, train_test_csv, search_by='*.tif',
                                skip_dataframe_creation=True)  # #

modeldir = '../Model Run/Model'
model = 'RF'

# change for model run
exclude_columns = ['Alexi_ET', 'Grace', 'MODIS_ET', 'GW_Irrigation_Density_fao',
                   'ALOS_Landform', 'MODIS_PET', 'Global_Sed_Thickness_Exx',
                   # 'Global_Sediment_Thickness','SRTM_Slope', 'Clay_content_PCA'
                   ]
prediction_raster_keyword = 'RF82'

# predictor_importance = False if predictor importance plot is not required
# plot_pdp = False if partial dependence plots are not required
# plot_confusion_matrix = False if confusion matrix plot (as image) is not required
ML_model = build_ml_classifier(train_test_csv, modeldir, exclude_columns, model, load_model=False,
                               pred_attr='Subsidence', test_size=0.3, random_state=0, output_dir=csv_dir,
                               n_estimators=200, min_samples_leaf=1, min_samples_split=2, max_depth=20,
                               max_features='auto', class_weight='balanced',
                               predictor_imp_keyword=prediction_raster_keyword,
                               predictor_importance=True,  # #
                               plot_pdp=True,  # #
                               plot_confusion_matrix=True,  # #
                               tune_hyperparameter=False,  # #
                               k_fold=5, n_iter=70,
                               random_searchCV=True)  # #

predictors_dir = '../Model Run/Predictors_2013_2019'

# predictor_csv_exists =  False if new predictor has been added so new data has to be added in each continent's
# dataframe
# predictor_probability_greater_1cm = False if probability plot is not required
create_prediction_raster \
    (predictors_dir, ML_model, yearlist=[2013, 2019], search_by='*.tif', continent_search_by='*continent.shp',
     continent_shapes_dir='../Data/Reference_rasters_shapes/continent_extents',
     prediction_raster_dir='../Model Run/Prediction_rasters', exclude_columns=exclude_columns, pred_attr='Subsidence',
     prediction_raster_keyword=prediction_raster_keyword,
     predictor_csv_exists=False,  # #
     predict_probability_greater_1cm=False)  # #
