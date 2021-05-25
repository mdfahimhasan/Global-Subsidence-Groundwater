import os
from glob import glob

import Raster_operations as rops
import Vector_operations as vops
import ML_codes as mcd

NO_DATA_VALUE = -9999
referenceraster="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster_with_antrc.tif"
referenceraster2="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster.tif"


#converting GS_G5 shapes to raster
# =============================================================================
# GS_G5_shp="E:\\NGA_Project_Data\\Model Run\\Subsidence_shapefiles\\GS_G5.shp"
# outras="E:\\NGA_Project_Data\\Model Run\\Working_dir\\Subsidence_G5.tif"
# rops.shapefile_to_raster(input_shape=GS_G5_shp, output_raster=outras,burn_attr=False,burnvalue=1,allTouched=False)
# #creating nan-filled subsidence raster
# subsidence_raster="E:\\NGA_Project_Data\\Model Run\\Working_dir\\Subsidence_G5.tif"
# outras="E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster\\Subsidence_G5.tif"
# rops.create_nanfilled_raster(input_raster=subsidence_raster, output_raster=outras)
# =============================================================================

#Cutting datasets with initial shape G5 Buffer(for training and testing model)
# =============================================================================
# GS_shp="E:\\NGA_Project_Data\\Model Run\\Subsidence_shapefiles\\GS_G5_buffer.shp"
# rasters=glob(os.path.join("E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster",'*.tif'))
# outdir="E:\\NGA_Project_Data\\Model Run\\Train_test_raster"
# 
# for raster in rasters:
#     rops.clip_resample_raster_cutline(input_raster_dir=raster, output_raster_dir=outdir,input_shape_dir=GS_shp,
#                                      naming_from_both=False)
# =============================================================================

###G5_L5 Raster Processing
#GS_L5_buffer based on small and large area
# =============================================================================
# GS_L5="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_L5.shp"
# GS_L5_large="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_Large.shp"
# GS_L5_small="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_Small.shp"
# GS_L5_large_buffer="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_Large_Buffer.shp"
# GS_L5_small_buffer="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_Small_Buffer.shp"
# Final_buffer_joined="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_L5_Buffer.shp"
# vops.select_by_attribute(shape=GS_L5, column="Area_Large", value="Yes", outshape=GS_L5_large)
# vops.overlay(shape1=GS_L5, shape2=GS_L5_large, outshape=GS_L5_small,how='difference')
# vops.buffer(shape=GS_L5_large,outshape=GS_L5_large_buffer,buffer=30000)
# vops.buffer_variable(shape=GS_L5_small, outshape=GS_L5_small_buffer,buffer_coef=0.003)
# vops.append_shapefile(shape1=GS_L5_large_buffer, shape2=GS_L5_small_buffer, outshape=Final_buffer_joined)
# 
# #GS_G5_L5_Joined Training_Testing Shapefile Prep for raster creation
# G5="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_G5.shp"
# L5="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_L5.shp"
# G5_L5_joined="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_G5_L5_joined.shp"
# 
# vops.append_shapes(shape1=G5,shape2=L5,output_shape=G5_L5_joined)
# 
# #converting GS_G5 and L5 shapes to raster
# GS_shp="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_G5_L5_joined.shp"
# outras="E:\\NGA_Project_Data\\Model Run\\Working_dir\\Subsidence_G5_L5.tif"
# rops.shapefile_to_raster(input_shape=GS_shp, output_raster=outras,burn_attr=True,
#                          attribute="Subsidence",resolution=0.05)
# 
# 
# #creating nan-filled subsidence raster
# subsidence_raster="E:\\NGA_Project_Data\\Model Run\\Working_dir\\Subsidence_G5_L5.tif"
# outdir="E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster_G5_L5"
# rops.create_nanfilled_raster(input_raster=subsidence_raster, outdir=outdir,
#                              raster_name="Subsidence_G5_L5.tif")
# =============================================================================


#Compiling all dataset in a single folder for Model Run

# =============================================================================
# Predictor_dict={}
# Predictor_dict['LU_FAOLC']="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_FAOLC_GW_nanfill.tif"
# Predictor_dict['LU_FAOLC_cropmask']="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_FAOLC_cropmask.tif"
# Predictor_dict['LU_GFSAD1KCM']="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_GFSAD1KCM.tif"
# Predictor_dict['MODIS_ET_13_19']="E:\\NGA_Project_Data\\ET_products\MODIS_ET\\ET_2013_2019\\World_ET_2013_2019\\ET_2013_2019.tif"
# Predictor_dict['Alexi_ET_13_19']="E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\mean_rasters_Step2\\Alexi_ET_2013_2019.tif"
# Predictor_dict['NDWI_13_19']="E:\\NGA_Project_Data\\NDWI_dataset\\2013_2019\\World_NDWI_13_19_Step02\\NDWI_2013_2019.tif"
# Predictor_dict['Rainfall_TRCLM_13_19']="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\2013_2019\\World_TRCLM_Step02\\TRCLM_pr_2013_2019.tif"
# Predictor_dict['Sediment_Thickness']="E:\\NGA_Project_Data\\Sediment_thickness\\Resampled\\Global_sedeiment_thickness_resampled\\Global_sediment_thickness.tif"
# Predictor_dict['population']="E:\\NGA_Project_Data\\population_density\\Global_mean_pop_raster\\Global_mean_pop_2000_2020.tif"
# Predictor_dict['Grace_gradient']="E:\\NGA_Project_Data\\GRACE\\Global_resampled_Step2\\GRACE_2013_2017.tif"
# Predictor_dict['Soil_moisture']="E:\\NGA_Project_Data\\Soil_Moisture\\Soil_Moisture_SMAP_2015_2019\\Global_resampled_Step2\\SM_SMAP_2015_2019.tif"
# Predictor_dict['Slope']="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM_Slope\\World_Slope_Step02\\SRTM_DEM_Slope.tif"
# Predictor_dict['DEM']="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM\\World_DEM_Step02\\SRTM_DEM_World.tif"
# Predictor_dict['Landform']="E:\\NGA_Project_Data\\DEM_Landform\\ALOS_Landform\\World_ALOS_LF_without_Ocean_Step03\\LF_ALOS_World.tif"
# Predictor_dict['Temp_mean_13_19']="E:\\NGA_Project_Data\\Temperature_data\\World_Temp__mean_Step03\\T_mean_2013_2019.tif"
# Predictor_dict['z_soiltype']="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\\Resampled_Data\\z_soiltype.tif"
# Predictor_dict['EVI_13_19']="E:\\NGA_Project_Data\\Enhanced_Veg_Index\\World_EVI_Step02\\EVI_2013_2019\\EVI_2013_2019.tif"
# 
# outdir="E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster_G5_L5\\"
# 
# for each in Predictor_dict.keys():
#     rops.rename_copy_raster(input_raster=Predictor_dict[each],output_dir=outdir,new_name=each,
#                             change_dtype=False)
# =============================================================================


#Cutting datasets with initial shape (for training and testing model)
# =============================================================================
# GS_L5_shp="E:\\NGA_Project_Data\\shapefiles\\global_subsidence\\GS_L5_Buffer.shp"
# rasters=glob(os.path.join("E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster_G5_L5",'*.tif'))
# outdir="E:\\NGA_Project_Data\\Model Run\\Train_test_raster_G5_L5"
# 
# for raster in rasters:
#     rops.clip_resample_raster_cutline(input_raster_dir=raster, output_raster_dir=outdir,input_shape_dir=GS_L5_shp,
#                                      naming_from_both=False)
# =============================================================================
