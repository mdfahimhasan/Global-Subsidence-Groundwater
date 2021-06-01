#Done : LandUse (GFSAD1KCM, FAOLC-cropmask)-create function for adding crop mask on FAOLC, then process to agricultural density
#Done : Population - convert to population density raster
#ET (Modis, Alexi)
#Rainfall 
#Temperature
#Soil Moisture 
#NDWI 
#EVI
#Slope and Landform 
#Grace


import os
from Raster_operations import *
from sysops import *

os.chdir(r'..\Codes_Global_GW')

##Resampling Land Use Rasters
#GFSAD1KCM
#Resampling GFSAD1KCM LU data with reference raster
# inras=r'..\Raw_Data\Land_Use_Data\Raw\Global Food Security- GFSAD1KCM\GFSAD1KCM.tif'
# outdir='..\Raw_Data\Land_Use_Data\Intermediate_Global'
# pasted_outdir='..\Resampled Data\Land_Use'
# mask_by_ref_raster(input_raster=inras, outdir=outdir, raster_name='Global_GFSAD1KCM_raw.tif')

##Gaussian Filtering GFSAD1KCM Landuse
# inras='..\Raw_Data\Land_Use_Data\Intermediate_Global\Global_GFSAD1KCM_raw.tif'
# outdir='..\Resampled Data\Land_Use'
# filter_specific_values(input_raster=inras, outdir=outdir, raster_name='GFSAD_irrig_only.tif',filter_value=[1,2],
#                        new_value=True,value_new=1,paste_on_ref_raster=True,ref_raster=referenceraster2)
#
# rastername='..\Resampled Data\Land_Use\GFSAD_irrig_only.tif'
# outdir='..\Resampled Data\Land_Use'
# apply_gaussian_filter(input_raster=rastername, outdir=outdir, raster_name='Irrigated_Area_Density.tif', ignore_nan=True,normalize=True)

# Creating Irrigated cropland Mask from GFSAD1KCM LC
# gfsad_LC=r'..\Resampled Data\Land_Use\GFSAD_irrig_only.tif'
# cropland_global=r'..\Reference_rasters'
# filter_specific_values(input_raster=gfsad_LC, outdir=cropland_global,raster_name="Global_irrigated_cropmask.tif",
#                        filter_value=[1],paste_on_ref_raster=True)

##FAO LC
#Masking FAOLC raster with reference raster
# inras=r'..\Raw_Data\Land_Use_Data\Raw\FAO_LC\RasterFile\aeigw_pct_aei.tif'
# outdir='..\Raw_Data\Land_Use_Data\Intermediate_Global'
# mask_by_ref_raster(input_raster=inras, outdir=outdir, raster_name='Global_FAOLC.tif')
#
# #Creating nanfilled FAOLC raster
# inras=r'..\Raw_Data\Land_Use_Data\Intermediate_Global\Global_FAOLC.tif'
# outdir=r'..\Raw_Data\Land_Use_Data\Intermediate_Global'
# create_nanfilled_raster(input_raster=inras, outdir=outdir,raster_name='Global_FAOLC_nanfilled.tif',ref_raster=referenceraster2)
#
# #FAO_LC LU cropland masked
# inras=r'..\Raw_Data\Land_Use_Data\Intermediate_Global\Global_FAOLC_nanfilled.tif'
# crop_ras=r'..\Reference_rasters\Global_irrigated_cropmask.tif'
# outdir=r'..\Resampled Data\Land_Use'
# array_multiply(input_raster1=inras, input_raster2=crop_ras, outdir=outdir,raster_name='Global_FAOLC_cropmasked.tif')

##Gaussian Filtering FAO_LC GW irrigation % Landuse
# raster=r'..\Resampled Data\Land_Use\Global_FAOLC_cropmasked.tif'
# outdir=r'..\Resampled Data\Land_Use'
# apply_gaussian_filter(input_raster=raster, outdir=outdir, raster_name='GW_Irrigation_Density.tif',ignore_nan=True,normalize=True)

##Resampling Population Density Raster (Gaussian Filtering)
# inras=r'..\Raw_Data\population_density\GPWv411 UN-Adjusted Population Density\2010_2020\World_pop_data_step02\Pop_density_GPW_2010_2020.tif'
# outdir=r'..\Resampled Data\Population_Density'
# apply_gaussian_filter(input_raster=inras, outdir=outdir, raster_name='Population_Density.tif', ignore_nan=True,normalize=True)

