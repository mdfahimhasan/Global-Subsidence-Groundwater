from Data_operations import *
from ML_operations import *

gee_data_list = ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_soil',
                 'MODIS_ET', 'MODIS_EVI', 'SRTM_DEM', 'SRTM_Slope',
                 'ALOS_Landform', 'Aridity_Index', 'Clay_content', 'Grace', 'MODIS_NDWI']

yearlist = [2013, 2019]
start_month = 1
end_month = 12
resampled_dir = r'../Data/Resampled_Data/GEE_data_2013_2019'
gfsad_lu = r'../Data/Raw_Data/Land_Use_Data/Raw/Global Food Security- GFSAD1KCM/GFSAD1KCM.tif'
faolc = r'../Data/Raw_Data/Land_Use_Data/Raw/FAO_LC/RasterFile/aeigw_pct_aei.tif'
outdir_lu = r'../Data/Resampled_Data/Land_Use'
lithology = r'../Data/Raw_Data/Global_Lithology/glim_wgs84_0point5deg.tif'
permeability = r'../Data/Raw_Data/Global_Hydrogeology/GLHYMPS_permeability.tif'
intermediate_dir = r'../Data/Intermediate_working_dir'
outdir_lith_perm = r'../Data/Resampled_Data/Lithology_Permeability'
sediment_thickness = r'../Data/Raw_Data/Global_Sediment_Thickness/average_soil_and_sedimentary-deposit_thickness.tif'
outdir_sed_thickness = r'../Data/Resampled_Data/Sediment_Thickness'
outdir_pop = r'../Data/Resampled_Data/Pop_Density'

gee_raster_dict, gfsad_raster, faolc_raster, lithology_raster, permeability_raster, sediment_thickness_raster,\
 popdensity_raster = prepare_predictor_datasets(yearlist, start_month, end_month, resampled_dir,
                                                gfsad_lu, faolc, intermediate_dir, outdir_lu,
                                                lithology, permeability, outdir_lith_perm,
                                                sediment_thickness, outdir_sed_thickness,
                                                outdir_pop,
                                                skip_download=True, skip_processing=True,
                                                geedatalist=gee_data_list, downloadcsv=csv, gee_scale=2000)

input_polygons_dir = r'../InSAR_Data/Georeferenced_subsidence_data'
joined_subsidence_polygon = '../InSAR_Data/Resampled_subsidence_data/interim_working_dir/georef_subsidence_polygons.shp'
insar_data_dir = r'../InSAR_Data/Resampled_subsidence_data'
interim_dir = r'../InSAR_Data/Resampled_subsidence_data/interim_working_dir'
training_insar_dir = r'../InSAR_Data/Resampled_subsidence_data/final_subsidence_raster'

subsidence_raster = prepare_subsidence_raster(input_polygons_dir, joined_subsidence_polygon,
                                              insar_data_dir, interim_dir, training_insar_dir,
                                              skip_polygon_merge=True, subsidence_column='Class_name',
                                              final_subsidence_raster='Subsidence_training.tif',
                                              polygon_search_criteria='*Subsidence*.shp',
                                              insar_search_criteria='*reclass_resampled*.tif', already_prepared=True)

predictor_dir = r'../Model Run/Predictors_2013_2019'
predictor_dir = compile_predictors_subsidence_data(gee_raster_dict, gfsad_raster, faolc_raster, lithology_raster,
                                                   permeability_raster, popdensity_raster, sediment_thickness_raster,
                                                   subsidence_raster, predictor_dir, skip_processing=True)
csv_dir = r'../Model Run/Predictors_csv'
makedirs([csv_dir])
train_test_csv = r'../Model Run/Predictors_csv/train_test_2013_2019.csv'
predictor_df = create_dataframe(predictor_dir, train_test_csv, search_by='*.tif',
                                skip_processing=True)

modeldir = r'../Model Run/Model'
model = 'RF'

# change for model run
exclude_columns = ()
prediction_raster_keyword = 'RF1'
cmatrix_name = 'RF1_cmatrix.csv'

ML_model = build_ml_classifier(train_test_csv, modeldir, exclude_columns, model, load_model=False,
                               pred_attr='Subsidence', test_size=0.3, random_state=0, shuffle=True, output_dir=csv_dir,
                               n_estimators=500, bootstrap=True, oob_score=True, n_jobs=-2, max_features='auto',
                               accuracy=True, save=True, cm_name=cmatrix_name, predictor_importance=True,
                               predictor_imp_keyword=prediction_raster_keyword)

predictors_dir = r'../Model Run/Predictors_2013_2019'
create_prediction_raster(predictors_dir, ML_model, yearlist=[2013, 2019], search_by='*.tif',
                         continent_shapes_dir=r'../Data/Reference_rasters_shapes/continent_extents',
                         prediction_raster_dir=r'../Model Run/Prediction_rasters',
                         exclude_columns=exclude_columns, pred_attr='Subsidence',
                         prediction_raster_keyword=prediction_raster_keyword)
