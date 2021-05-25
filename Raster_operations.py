import os
import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from glob import glob
import numpy as np
import gdal
import fiona 
from fiona import transform
import astropy.convolution as apc
from scipy.ndimage import gaussian_filter

NO_DATA_VALUE = -9999
referenceraster="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster_with_antrc.tif"
referenceraster2="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster.tif"

# =============================================================================
# #Reading raster array and raster reader object
# =============================================================================
def read_raster_arr_object(input_raster,band=1,raster_object=False,get_file=True,change_dtype=True):
    """
    read raster as raster object and array. If raster_object=True get only the raster array 
    
    Parameters
    ----------
    input_raster : Input raster file path
    band : Selected band to read (Default 1)
    raster_object : Set true if raster_file is a rasterio object
    get_file : Get both rasterio object and raster array file if set to True
    change_dtype : Change raster data type to float if true
    ----------    
    Returns : Raster numpy array and rasterio object file (rasterio_obj=False and get_file=True)
    """
    if not raster_object:
        raster_file=rio.open(input_raster)
    else:
        get_file=False
    
    raster_arr=raster_file.read(band)
    
    if change_dtype:
        raster_arr=raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr,raster_file.nodata)]=np.nan
    
    if get_file:
        return raster_arr,raster_file        
    return raster_arr


# =============================================================================
# #Writing Raster to File
# =============================================================================
def write_raster(raster_arr, raster_file, transform, outfile_path, no_data_value=NO_DATA_VALUE,
                 ref_file=None):
    """
    Write raster file in GeoTIFF format
    
    Parameters
    ----------
    raster_arr: Raster array data to be written
    raster_file: Original rasterio raster file containing geo-coordinates
    transform: Affine transformation matrix
    outfile_path: Outfile file path with filename
    no_data_value: No data value for raster (default float32 type is considered)
    ref_file: Write output raster considering parameters from reference raster file
    ----------    
    Returns : None
    """
    if ref_file:
        raster_file = rio.open(ref_file)
        transform = raster_file.transform
    with rio.open(
            outfile_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=raster_arr.dtype,
            crs=raster_file.crs,
            transform=transform,
            count=raster_file.count,
            nodata=no_data_value
    ) as dst:
        dst.write(raster_arr, raster_file.count)

    
# =============================================================================
# #filter raster
# =============================================================================
def filter_lower_larger_value(input_raster,output_dir,band=1,lower=True,larger=False,filter_value=0,new_value=np.nan,no_data_value=NO_DATA_VALUE):
    """
    filter out and replace value in raster

    Parameters
    ----------
    input_raster : input raster directory with raster name
    output_dir : output raster directory
    band : band  to read. Default is 1
    lower : If lower than filter value need to be filtered out. Default set to True.
    larger: If lower than filter value need to be filtered out. Default set to False. If True set lower to False.
    filter_value : value to filter out. Default is 0
    new_value : value to replace in filtered out value. Default is np.nan
    no_data_value : No data value. Default is -9999
    """
        
    raster_arr,raster_data=read_raster_arr_object(input_raster)
    if lower:
        raster_arr[raster_arr<filter_value]=new_value
    if larger:
        raster_arr[raster_arr>filter_value]=new_value
    raster_arr[np.isnan(raster_arr)]=no_data_value
    
    out_name=os.path.join(output_dir,input_raster[input_raster.rfind(os.sep)+1:])
    
    write_raster(raster_arr=raster_arr, raster_file=raster_data, 
                 transform=raster_data.transform, outfile_path=out_name)

def filter_specific_values(input_raster,outdir,raster_name,fillvalue=np.nan,filter_value=[10,11],
                          new_value=False,value_new=1,no_data_value=NO_DATA_VALUE,paste_on_ref_raster=False):
    """
    Filter and replace values in raster.

    Parameters
    ----------
    input_raster : input raster directory with raster name.
    outdir : Output raster directory.
    raster_name: Output raster name.
    fillvalue : Value that new raster will be filled with initially. Default set to np.nan.
    filter_value : List of values to filter. Default is [10,11].
    new_value : Set to True if filtered value needs a new value. Default set to False.
    value_new : Value that the filtered value will get if new_value is set to True. Default set to 1.
    no_data_value : No data value. Default is -9999.
    paste_on_ref_raster : Set to True if filtered values should bepasted on reference raster.
    
    Returns : None
    """
    arr,data=read_raster_arr_object(input_raster)
    ref_arr,ref_file=read_raster_arr_object(referenceraster2)
    
    if paste_on_ref_raster:
        new_arr=ref_arr
    else:
        new_arr=np.full_like(arr, fill_value=fillvalue)
    
    if new_value:
        for value in filter_value:
            new_arr[arr==value]=value_new
    else:
        for value in filter_value:
            new_arr[arr==value]=value
    new_arr[np.isnan(new_arr)]=no_data_value
    
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    write_raster(raster_arr=new_arr, raster_file=data, transform=data.transform, outfile_path=output_raster)
    

# =============================================================================
# #Resample or Reproject Raster
# =============================================================================

def resample_reproject(input_raster,outdir,raster_name,reference_raster=referenceraster2,resample=True,reproject=False,
                       change_crs_to="EPSG:4326",both=False):
    """
    Resample a raster according to a reference raster/ Reproject a raster/ Both resample and reprojection.

    Parameters:
    input_raster : Input raster Directory with filename.
    outdir : Output raster directory.
    raster_name: Output raster name.
    reference_raster : Reference raster path with file name.
    resample : Set True to resample only.
    reproject : Set True to reproject only.
    both : Set True to both resample and reproject.
    
    Returns : Resampled/Reprojected raster.
    """
    ref_arr,ref_file=read_raster_arr_object(reference_raster)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    if resample:
        gdal.Warp(destNameOrDestDS=output_raster,srcDSOrSrcDSTab=input_raster,width=ref_arr.shape[1],
              height=ref_arr.shape[0],outputType=gdal.GDT_Float32)
    if reproject:
        gdal.Warp(destNameOrDestDS=output_raster,srcDSOrSrcDSTab=input_raster,dstSRS=change_crs_to,outputType=gdal.GDT_Float32)   

    if both:
        gdal.Warp(destNameOrDestDS=output_raster,srcDSOrSrcDSTab=input_raster,width=ref_arr.shape[1],
              height=ref_arr.shape[0],dstSRS=change_crs_to,outputType=gdal.GDT_Float32)

# =============================================================================
# #Reproject Coordinates
# =============================================================================
def reproject_coords(src_crs, dst_crs, coords):
    """
    Reproject coordinates. Copied from https://bit.ly/3mBtowB
    Author: user2856 (StackExchange user)
    
    Parameters:
    src_crs: Source CRS
    dst_crs: Destination CRS
    coords: Coordinates as tuple of lists
    
    Returns : Transformed coordinates as tuple of lists
    """
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xs, ys = transform.transform(src_crs, dst_crs, xs, ys)
    return [[x, y] for x, y in zip(xs, ys)]

# =============================================================================
# #Rename/Copy Raster
# =============================================================================
def rename_copy_raster(input_raster,output_dir,new_name,change_dtype=False):
    """
    renaming/copy a raster file and changing datatype if needed.
    Parameters:
    input_raster: input raster directory with raster name.
    output_dir : output raster (renamed) directory.
    new_name : new name of raster.
    change_dtype : False if datatype should remain unchanged. Set True to change data type to 'Float32'.
    Returns : Renamed and copied raster.
    """
    
    arr,file=read_raster_arr_object(input_raster,change_dtype=change_dtype)
    output_raster=os.path.join(output_dir,new_name+'.tif')
    write_raster(raster_arr=arr, raster_file=file, transform=file.transform, outfile_path=output_raster)       

# =============================================================================
# #Changing No Data Value
# =============================================================================

def change_nodata_value(input_raster_dir, new_nodata=NO_DATA_VALUE):
    """
    change no data value for single banded raster

    Parameters
    ----------
    input_raster_dir : input raster directory
    new_nodata : new no data value. Can be -9999 or np.nan. The default is -9999
    """
    
    #seeing the existing No Data value
    dataset=gdal.Open(input_raster_dir,1)  #1 for update mode
    no_data_value=dataset.GetRasterBand(1).GetNoDataValue()
    print(no_data_value)
    
    #changing no data value in band
    band1=dataset.GetRasterBand(1).ReadAsArray()
    band1[band1==no_data_value]=new_nodata
    dataset.GetRasterBand(1).SetNoDataValue(new_nodata)
    
    #writing changed band to dataset  
    dataset.GetRasterBand(1).WriteArray(band1)
    
    #closing raster properly
    dataset=None 
    

# =============================================================================
# #Changing a specific array value to no data
# =============================================================================

def change_band_value_to_nodata(input_raster,outfile_path,band_val_to_change=0,nodata=NO_DATA_VALUE):
    """
    changing a band value of a raster to no data value 

    Parameters
    ----------
    input_raster_dir : Input raster directory with file name
    band_val_to_change : Existing band value that need to be changed. Default is 0
    new_value : New band value that will be set as No Data Value of the raster. The default is -9999.
    """
    
    #opening raster dataset and read as array
    raster_arr,raster_file=read_raster_arr_object(input_raster)
    
    #changing band value to No Data Value
    raster_arr=np.where(raster_arr==band_val_to_change,nodata,raster_arr)  
    
    #writing raster file
    write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=outfile_path)
       
    
# =============================================================================
# #Clipping Raster and saving
# =============================================================================

def mask_raster(input_raster_dir,mask_shp_dir,output_dir,invert=False,crop=True,add_title=""):
    """
    Masks a raster with given shapefile.

    Parameters
    ----------
    input_raster_dir : input raster file path
    mask_shp_dir : shapefile (mask) directory with file name
    output_dir : output raster directory
    invert : If False (default) pixels outside shapes will be masked. 
             If True, pixels inside shape will be masked.
    crop : Whether to crop the raster to the extent of the shapes. Change to False if invert=True is used.
    add_title : additional output raster title word if needed. By default blank.
    -------
    Returns : None
    """
 
    #opening input raster
    raset_arr,raster_file=read_raster_arr_object(input_raster_dir)
    
    #masking
    with fiona.open(mask_shp_dir,'r') as shapefile:
        shape=[feature['geometry'] for feature in shapefile]      #shapefile opened with fiona to make sure it can be read as a list/json format by mask
    
    masked_arr,masked_transform=mask(dataset=raster_file,shapes=shape,filled=True,crop=crop,invert=invert)
    
    #naming output file
    title_keyword=add_title+mask_shp_dir[mask_shp_dir.rfind(os.sep)+1:mask_shp_dir.rfind(".")]+"_" 
    output_raster=os.path.join(output_dir,title_keyword+input_raster_dir[input_raster_dir.rfind(os.sep)+1:])
    
    #saving output raster
    with rio.open(
        output_raster,
        'w',
        driver='GTiff',
        height=masked_arr.shape[1],
        width=masked_arr.shape[2],
        dtype=masked_arr.dtype,
        crs=raster_file.crs,
        transform=masked_transform,
        count=raster_file.count,
        nodata=NO_DATA_VALUE
        ) as dst:
        dst.write(masked_arr)

# =============================================================================
# #Mask and Resample Global Raster Data by Reference Raster
# =============================================================================
def mask_by_ref_raster(input_raster,outdir,raster_name,ref_raster=referenceraster2,resolution=0.05,
                      nodata=NO_DATA_VALUE):
    """
    Mask a Global Raster Data by Reference Raster. 

    Parameters:
    input_raster : Input raster name with filepath.
    outdir : Output raster directory.
    raster_name : Output raster name.
    ref_raster : Global reference raster filepath. Defaults to referenceraster2.
    resolution : Resolution of output raster. Defaults to 0.05 degree in GCS_WGS_1984.
    nodata : No data value. Defaults to NO_DATA_VALUE of -9999.

    Returns:None.
    """
    ref_arr,ref_file=read_raster_arr_object(ref_raster)
    minx,miny,maxx,maxy=ref_file.bounds
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    gdal.Warp(destNameOrDestDS=output_raster, srcDSOrSrcDSTab=input_raster,format='GTiff',
              outputBounds=(minx,miny,maxx,maxy),xRes=resolution,yRes=resolution,dstSRS=ref_file.crs,
              dstNodata=nodata,targetAlignedPixels=True,outputType=gdal.GDT_Float32)

# =============================================================================
# #Clipping Raster by Shapefile Cutline, Processing NoData, Pixel Size, CRS
# =============================================================================

def clip_resample_raster_cutline(input_raster_dir, output_raster_dir, input_shape_dir, coordinate="EPSG:4326",
                                 xpixel=0.05,ypixel=0.05,NoData=NO_DATA_VALUE,naming_from_both=True):
    """
    clip raster by shapefile (cutline) to exact extent, resample pixel size and coordinate system

    Parameters
    ----------
    input_raster_dir : Input raster directory with filename
    output_raster_dir : Output raster directory
    input_shape_dir : Input shapefile (cutline) directory 
    coordinate : Output coordinate system. The default is "EPSG:4326".
    xpixel : X pixel size. The default is 0.05.
    ypixel :  Y pixel size. The default is 0.05.
    NoData : No Data value. By default None.
    naming_from_both : If clipped raster need to contain both name from raster and shapefile set True. Otherwise set False.
    """
    
    input_raster=gdal.Open(input_raster_dir)
    
    if not os.path.exists(output_raster_dir):
        os.makedirs(output_raster_dir)
        
    #naming output raster
    if naming_from_both:
        raster_part=input_raster_dir[input_raster_dir.rfind(os.sep)+1:]
        shape_part=input_shape_dir[input_shape_dir.rfind(os.sep)+1:input_shape_dir.rfind("_")]
        output_path=os.path.join(output_raster_dir,shape_part+"_"+raster_part)
    else:
        raster_part=input_raster_dir[input_raster_dir.rfind(os.sep)+1:]
        output_path=os.path.join(output_raster_dir,raster_part)
    #Warping
    dataset= gdal.Warp(destNameOrDestDS=output_path, srcDSOrSrcDSTab=input_raster,dstSRS=coordinate, targetAlignedPixels=True,
              xRes=xpixel,yRes=ypixel, cutlineDSName=input_shape_dir,cropToCutline=True,dstNodata=NoData)
    
    del dataset
    
# =============================================================================
# #Clipping MODIS Raster by Shapefile Cutline, Processing NoData, Pixel Size, CRS (Data Only Downloaded by Python Gee Download Code)
# =============================================================================

def clip_resample_MODIS_cutline(input_raster_dir, output_raster_dir, input_shape_dir, coordinate="EPSG:4326",
                                xpixel=0.05,ypixel=0.05,NoData=NO_DATA_VALUE):
    """
    clip MODIS raster by shapefile (cutline) to exact extent, resample pixel size and coordinate system

    Parameters
    ----------
    input_raster_dir : Input raster directory with filename. Use glob for multiple rasters
    output_raster_dir : Output raster directory
    input_shape_dir : Input shapefile (cutline) directory 
    coordinate : Output coordinate system. The default is "EPSG:4326".
    xpixel : X pixel size. The default is 0.05.
    ypixel :  Y pixel size. The default is 0.05.
    NoData : No Data value. By default None.
    """
    
    input_raster=gdal.Open(input_raster_dir)
    
    #naming output raster
    Out_raster_name=input_raster_dir[input_raster_dir.rfind(os.sep)+1:]  #Raster name will remain the same, only directory will change
    output_path=os.path.join(output_raster_dir,Out_raster_name)
        
    #Warping
    
    dataset= gdal.Warp(destNameOrDestDS=output_path, srcDSOrSrcDSTab=input_raster,dstSRS=coordinate, targetAlignedPixels=True,
              xRes=xpixel,yRes=ypixel, cutlineDSName=input_shape_dir,cropToCutline=True,dstNodata=NoData)
    
    del dataset
    
# =============================================================================
# #Mosaic Multiple Rasters    
# =============================================================================
def mosaic_rasters(input_dir,output_dir,raster_name,ref_raster=referenceraster2,search_by="*.tif",
                   resolution=0.05,no_data=NO_DATA_VALUE, create_outdir=False):
    """
    Mosaics multiple rasters into a single raster (rasters have to be in a single directory) 

    Parameters
    ----------
    input_dir : Input rasters directory 
    output_dir : Outpur raster directory. Defaults to None. Include raster name as .tif if output_dir=None and 
                    create_outdir=False
    raster_name : Outpur raster name (output raster name with filepath if create_outdir=True) 
    ref_raster : Reference raster with filepath
    no_data : No data value. Default -9999
    resolution: Resolution of the output raster
    -------
    Returns: None
    """
    input_rasters=glob(os.path.join(input_dir,search_by))
    
    raster_list=[]
    for raster in input_rasters:
        arr,file=read_raster_arr_object(raster)
        raster_list.append(file)
    
    ref_arr,ref_file=read_raster_arr_object(ref_raster)
    merged_arr,out_transform=merge(raster_list,bounds=ref_file.bounds,res=(resolution,resolution),nodata=no_data)
    
    merged_arr=np.where(ref_arr==0,merged_arr,ref_arr)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_raster=os.path.join(output_dir,raster_name)
    with rio.open(
            out_raster,
            'w',
            driver='GTiff',
            height=merged_arr.shape[1],
            width=merged_arr.shape[2],
            dtype=merged_arr.dtype,
            crs=ref_file.crs,
            transform=out_transform,
            count=file.count,
            nodata=NO_DATA_VALUE
            ) as dst:
            dst.write(merged_arr)
        
# =============================================================================
# #Mean rasters from a folder 
# =============================================================================

def mean_rasters(input_dir,outdir,raster_name,reference_raster=None,searchby="*.tif",no_data_value=NO_DATA_VALUE):
    """
    mean multiple rasters from a directory. 

    Parameters
    ----------
    input_dir :Input raster directory
    outdir : Output raster directory
    raster_name : Output raster name
    reference_raster : Reference raster for setting affine
    searchby : Searching criteria for input raster. The default is "*.tif".
    no_data_value: No Data Value default set as -9999.
    ----------
    Returns: None
    """
    input_rasters=glob(os.path.join(input_dir,searchby))
    
    val=0
    
    for each in input_rasters:
        each_arr,ras_file=read_raster_arr_object(each)
        if val==0:
            arr_new=each_arr
        else:
            arr_new=arr_new+each_arr
        val=val+1
    arr_mean=arr_new/val
    arr_mean[np.isnan(arr_mean)] = NO_DATA_VALUE
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    write_raster(raster_arr=arr_mean,raster_file=ras_file,transform=ras_file.transform,
                 outfile_path=output_raster, no_data_value=no_data_value,ref_file=reference_raster) 

# =============================================================================
# # Mean 2 Rasters
# =============================================================================
def mean_2_rasters(input1,input2,outdir,raster_name,nodata=NO_DATA_VALUE):
    """
    mean 2 rasters . 

    Parameters
    ----------
    input1 :Input raster 1 with filepath
    input2 :Input raster 2 with filepath
    outdir : Output raster directory
    raster_name : Output raster name
    nodata : No data value. Defaults to -9999
    ----------
    Returns: None
    """
    arr1,rasfile1=read_raster_arr_object(input1)
    arr2,rasfile2=read_raster_arr_object(input2)
    
    mean_arr=np.mean(np.array([arr1,arr2]),axis=0)
    
    mean_arr[np.isnan(mean_arr)] = NO_DATA_VALUE
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    write_raster(raster_arr=mean_arr,raster_file=rasfile1,transform=rasfile1.transform,
                 outfile_path=output_raster, no_data_value=nodata)
      
# =============================================================================
# #Multiply Raster Arrays
# =============================================================================
def array_multiply(input_raster1,input_raster2,outdir,raster_name):
    """
    Multiplies 2 rasters. the rasters should be of same shape (row, column size).
    
    Parameters:
    input_raster1 : Raster 1 file with file name.
    input_raster2 : Raster 1 file with file name.
    outdir : Output Raster Directory.
    raster_name : Output raster name.

    Returns:None.
    """
    arr1,data1=read_raster_arr_object(input_raster1)
    arr2,data2=read_raster_arr_object(input_raster2)
    new_arr=np.multiply(arr1,arr2)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    write_raster(raster_arr=new_arr, raster_file=data1, transform=data1.transform, outfile_path=output_raster)

# =============================================================================
# #Convert Shapefile to Raster
# =============================================================================
def shapefile_to_raster(input_shape,output_raster,burn_attr=False,attribute="",ref_raster=referenceraster2,
                        resolution=0.05,burnvalue=1,allTouched=False,nodatavalue=NO_DATA_VALUE):
    """
    Converts polygon shapefile to raster by attribute value or burn value.

    Parameters:
    input_shape : Shapefile name.
    output_raster : Output raster name.
    burn_attr : Set to True if raster needs to be created using a specific attribute value. Defaults to False.
    attribute : Attibute name to use creating raster file. Defaults to "".
    ref_raster : Reference raster to get minx,miny,maxx,maxy. Defaults to referenceraster2.
    resolution : Resolution of the raster. Defaults to 0.05.
    burnvalue : Value for burning into raster. Only needed when burn_attr is False. Defaults to 1.
    allTouched : If True all pixels touched by lines or polygons will be updated.
    nodatavalue : NO_DATA_VALUE.

    Returns:None.
    """
    
    ref_arr,ref_file=read_raster_arr_object(ref_raster)
    total_bounds=ref_file.bounds
    if burn_attr:
        raster_options=gdal.RasterizeOptions(format='Gtiff',outputBounds= list(total_bounds),
                                         outputType=gdal.GDT_Float32,xRes=resolution,yRes=resolution,
                                         noData=nodatavalue,attribute=attribute,allTouched=allTouched)
    else:    
        raster_options=gdal.RasterizeOptions(format='Gtiff',outputBounds= list(total_bounds),
                                         outputType=gdal.GDT_Float32,xRes=resolution,yRes=resolution,
                                         noData=nodatavalue,burnValues=burnvalue,allTouched=allTouched)

    gdal.Rasterize(destNameOrDestDS=output_raster, srcDS=input_shape,options=raster_options)

# =============================================================================
# #Creating Slope Raster from DEM Data
# =============================================================================
def create_slope_raster(input_raster,outdir,raster_name):
    """
    Create Slope raster in Percent from DEM raster.

    Parameter:
    input_raster : Input raster with filepath.
    outdir : Output raster directory.
    raster_name : Output raster name.

    Returns: None.
    """
    dem_options=gdal.DEMProcessingOptions(format="GTiff",computeEdges=True,alg='Horn',slopeFormat='percent',scale=100000)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    gdal.DEMProcessing(destName=output_raster, srcDS=input_raster, processing='slope', options=dem_options)

# =============================================================================
# #creating nanfilled raster from original raster
# =============================================================================
def create_nanfilled_raster(input_raster,outdir,raster_name,value=0,ref_raster=referenceraster2):
    """
    Create a nan-filled raster with a reference raster. If there is nan value on raster that 
    will be filled by zero from reference raster.

    parameters:
    input_raster : Input raster.
    outdir : Output raster directory.
    raster_name : output raster name.
    value : value used in filtering. Defaults to 0.
    ref_raster : Reference raster on which initial raster value is pasted. Defaults to referenceraster2.

    Returns:None.
    """
    ras_arr,ras_file=read_raster_arr_object(input_raster)
    ref_arr,ref_file=read_raster_arr_object(ref_raster)
    
    ras_arr=ras_arr.flatten()
    ref_arr=ref_arr.flatten()
    
    new_arr=np.where(ras_arr>value,ras_arr,ref_arr)
    new_arr=new_arr.reshape(ref_file.shape[0],ref_file.shape[1])
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    write_raster(raster_arr=new_arr, raster_file=ras_file, transform=ras_file.transform, outfile_path=output_raster)
    
def paste_val_on_ref_raster(input_raster,outdir,raster_name,ref_raster=referenceraster2):
    """
    Paste value from a raster on the reference raster. If there are nan values on raster that 
    will be filled by nan from reference raster.

    parameters:
    input_raster : Input raster.
    outdir : Output raster directory.
    raster_name : output raster name.
    ref_raster : Reference raster on which initial raster value is pasted. Defaults to referenceraster2.

    Returns:None.
    """
    ras_arr,ras_file=read_raster_arr_object(input_raster)
    ref_arr,ref_file=read_raster_arr_object(ref_raster)
    
    ras_arr=ras_arr.flatten()
    ref_arr=ref_arr.flatten()
    
    new_arr=np.where(ref_arr==0,ras_arr,ref_arr)
    new_arr=new_arr.reshape(ref_file.shape[0],ref_file.shape[1])
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    write_raster(raster_arr=new_arr, raster_file=ras_file, transform=ras_file.transform, outfile_path=output_raster)    

# =============================================================================
# #Gaussian Filter
# =============================================================================
def apply_gaussian_filter(input_raster,outdir,raster_name,sigma=3,ignore_nan=True,normalize=False,
                         nodata=NO_DATA_VALUE,ref_raster=referenceraster2):
    """
    Applies Gaussian filter to raster.

    Parameters:
    input_raster : Input Raster.
    outdir : Output Raster Directory.
    raster_name : Output raster name.
    sigma : Standard Deviation for gaussian kernel. Defaults to 3.
    ignore_nan :  Set true to ignore nan values during convolution.
    normalize : Set true to normalize the filtered raster at the end.
    nodata : NO_DATA_VALUE.
    ref_raster : Reference Raster. Defaults to referenceraster2.

    Returns: Gaussian filtered raster.
    """
    raster_arr,raster_file=read_raster_arr_object(input_raster)
    if ignore_nan:
        Gauss_kernel=apc.Gaussian2DKernel(x_stddev=sigma,x_size=3*sigma,y_size=3*sigma)
        raster_arr_flt=apc.convolve(raster_arr,kernel=Gauss_kernel,preserve_nan=True)
    else:
        raster_arr[np.isnan(raster_arr)]=0
        raster_arr_flt=gaussian_filter(input=raster_arr, sigma=sigma,order=0) #order 0 corresponds to convolution with a Gaussian kernel
    
    if normalize:
        raster_arr_flt = np.abs(raster_arr_flt)
        raster_arr_flt -= np.min(raster_arr_flt)
        raster_arr_flt /= np.ptp(raster_arr_flt)
     
    ref_arr=read_raster_arr_object(ref_raster,get_file=False)
    raster_arr_flt[np.isnan(ref_arr)]=nodata
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    write_raster(raster_arr=raster_arr_flt, raster_file=raster_file, transform=raster_file.transform, 
                 outfile_path=os.path.join(outdir,raster_name))

    
##Resampling Alexi ET Produce

#creating mean rasters       
# =============================================================================
# data_dir = 'E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\year_wise\\2013_2019'
# outdir = "E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\mean_rasters_Step2"
# ref_raster = "E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_Continents_ref_raster.tif"
# mean_rasters(data_dir, outdir,raster_name="Alexi_ET_2013_2019.tif", ref_raster, searchby='*ET*.tif')
# 
# data_dir2 = 'E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\year_wise\\2018_2019'
# outdir2 = "E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\mean_rasters_Step2"      
# mean_rasters(data_dir2, outdir2,raster_name="Alexi_ET_2018_2019.tif", ref_raster, searchby='*ET*.tif')
# 
# #Clipping Alexi ET Product with Continent Cutlines
# input_rasters=glob(os.path.join("E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\mean_rasters_Step2","*2013*.tif"))
# output_dir="E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\Alexi_continent_yearly_mean_Step3\\Alexi_2013_2019"
# input_shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\continent_extents","*continent*.shp"))
# 
# for shape in input_shapes:
#     for raster in input_rasters:
#         clip_resample_raster_cutline(input_raster_dir=raster, output_raster_dir=output_dir, input_shape_dir=shape)
# 
# input_rasters2=glob(os.path.join("E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\mean_rasters_Step2","*2018*.tif"))
# output_dir2="E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\Alexi_continent_yearly_mean_Step3\\Alexi_2018_2019"
# for shape in input_shapes:
#     for raster in input_rasters2:
#         clip_resample_raster_cutline(input_raster_dir=raster, output_raster_dir=output_dir2, input_shape_dir=shape)
# =============================================================================

###Resampling Land Use Rasters

##GFSAD1KCM
# =============================================================================
# #Resampling GFSAD1KCM LU data with reference raster
# inras="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\Global Food Security- GFSAD1KCM\\GFSAD1KCM.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\Global_files"
# mask_by_ref_raster(input_raster=inras, outdir=outdir, raster_name="Global_GFSAD1KCM_raw.tif")
# 
# #Remove Ocean's 0 value from GFSAD1KCM
# inras="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\Global_files\\Global_GFSAD1KCM_raw.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# paste_val_on_ref_raster(input_raster=inras, outdir=outdir, raster_name="Global_GFSAD1KCM.tif")
# =============================================================================

##Gaussian Filtering GFSAD1KCM Landuse
# =============================================================================
# raster="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_GFSAD1KCM.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# filter_specific_values(input_raster=raster, outdir=outdir, raster_name='GFSAD_irrig_only.tif',filter_value=[1,2],
#                        new_value=True,value_new=1,paste_on_ref_raster=True)
# 
# rastername="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\GFSAD_irrig_only.tif"
# apply_gaussian_filter(input_raster=rastername, outdir=outdir, raster_name='Irrigated_Area_Density.tif')
# =============================================================================

##FAO LC
# =============================================================================
# #Creating Irrigated cropland data layer from GFSAD1KCM LC
# gfsad_LC="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_GFSAD1KCM.tif"
# cropland_global="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# filter_specific_values(input_raster=gfsad_LC, outdir=cropland_global,raster_name="Global_cropland_irrigated.tif",
#                        filter_value=[1,2],new_value=True,value_new=1,fillvalue=np.nan)   #Only filtered Irrigated cropland values
# #Creating Irrigated+rainfed cropland data layer from GFSAD1KCM LC
# gfsad_LC="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_GFSAD1KCM.tif"
# cropland_global="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# filter_specific_values(input_raster=gfsad_LC, outdir=cropland_global,raster_name="Global_cropland_all.tif",
#                        filter_value=[1,2,3,4,5],new_value=True,value_new=1,fillvalue=np.nan)   
# 
# #Modifying FAOLC raster with reference raster
# inras="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\FAO_LC\\RasterFile\\aeigw_pct_aei.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\Global_files"
# mask_by_ref_raster(input_raster=inras, outdir=outdir, raster_name="Global_FAOLC_GW01.tif")
# 
# #Creating nanfilled FAOLC raster
# inras="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\Global_files\\Global_FAOLC_GW01.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# create_nanfilled_raster(input_raster=inras, outdir=outdir,raster_name="FAOLC_GW_nanfilled02.tif",ref_raster=referenceraster2)
# =============================================================================
     
##FAO_LC LU cropland masked
##Multiply cropland (Irrigated) and FAO LC raster
# =============================================================================
# inras1="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\FAOLC_GW_nanfilled02.tif"
# inras2="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_cropland_irrigated.tif" 
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# array_multiply(input_raster1=inras1, input_raster2=inras2, outdir=outdir,raster_name="FAOLC_cropmasked_irrigated.tif")
# 
# inras3="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\FAOLC_cropmasked_irrigated.tif" 
# create_nanfilled_raster(input_raster=inras3, outdir=outdir, raster_name='FAOLC_cropmasked_irrigated_nanfilled03.tif')
# 
# ##Gaussian Filtering FAO_LC GW irrigation % Landuse
# raster="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\FAOLC_cropmasked_irrigated_nanfilled03.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# apply_gaussian_filter(input_raster=raster, outdir=outdir, raster_name='GW_Irrigation_Density_01.tif')
# =============================================================================

##Multiply cropland (Irrigated+Rainfed) and FAO LC raster
# =============================================================================
# inras1="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\FAOLC_GW_nanfilled02.tif"
# inras2="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\Global_cropland_all.tif" 
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# array_multiply(input_raster1=inras1, input_raster2=inras2, outdir=outdir,raster_name="FAOLC_cropmasked_all.tif")
# 
# inras3="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\FAOLC_cropmasked_all.tif" 
# create_nanfilled_raster(input_raster=inras3, outdir=outdir, raster_name='FAOLC_cropmasked_all_nanfilled03.tif')
# 
# ##Gaussian Filtering FAO_LC GW irrigation % Landuse
# raster="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files\\FAOLC_cropmasked_all_nanfilled03.tif"
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Resampled\\Global_files"
# apply_gaussian_filter(input_raster=raster, outdir=outdir, raster_name='GW_Irrigation_Density_02.tif')
# =============================================================================
  
##Sediment Thickness
#Renaming
# =============================================================================
# indir="E:\\NGA_Project_Data\\Sediment_thickness\\sediment_thickness_NASA_raw\\data\\average_soil_and_sedimentary-deposit_thickness.tif"
# outdir="E:\\NGA_Project_Data\\Sediment_thickness\\Resampled\\Average_sediment_thickness_unsampled"
# rename_copy_raster(input_raster=indir, output_dir=outdir, new_name="sediment_thickness",change_dtype=True)
# =============================================================================

#Clipping with continent shapefiles
# =============================================================================
# cutline_shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\continent_extents","*continent*.shp"))
# input_raster="E:\\NGA_Project_Data\\Sediment_thickness\\Resampled\\Average_sediment_thickness_unsampled\\sediment_thickness.tif"
# output="E:\\NGA_Project_Data\\Sediment_thickness\\Resampled\\continentwide_sediment_thickness"
# 
# for shape in cutline_shapes:
#      clip_resample_raster_cutline(input_raster_dir=input_raster, output_raster_dir=output, 
#                                       input_shape_dir=shape)
# =============================================================================

#Mosaicing resampled continent rasters into a global raster
# =============================================================================
# indir="E:\\NGA_Project_Data\\Sediment_thickness\\Resampled\\continentwide_sediment_thickness"
# outras="E:\\NGA_Project_Data\\Sediment_thickness\\Resampled\\Global_sedeiment_thickness_resampled\\Global_sediment_thickness.tif"
# ref_ras="E:\\NGA_Project_Data\\shapefiles\\continent_extents\\Global_continents_ref_raster.tif"
# 
# mosaic_rasters(input_dir=indir, output_raster=outras, ref_raster=ref_ras)
# =============================================================================


# =============================================================================
# #Processing Landsat_NDWI Data
# #mosaicing grided rasters
# inputdir="E:\\NGA_Project_Data\\NDWI_dataset\\Raw_NDWI_Step1\\NDWI_Grid_2013_2019"
# outras="E:\\NGA_Project_Data\\NDWI_dataset\\NDWI_World_Step2\\NDWI_world_2013_2019.tif"
# reference="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster.tif"
# mosaic_rasters(input_dir=inputdir, output_raster=outras, ref_raster=reference)
# 
# inputdir2="E:\\NGA_Project_Data\\NDWI_dataset\\Raw_NDWI_Step1\\NDWI_Grid_2018_2019"
# outras2="E:\\NGA_Project_Data\\NDWI_dataset\\NDWI_World_Step2\\NDWI_world_2018_2019.tif"
# mosaic_rasters(input_dir=inputdir2, output_raster=outras2, ref_raster=reference)
# 
# #Clipping with continent shapefiles
# cutline_shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\continent_extents","*continent*.shp"))
# input_raster="E:\\NGA_Project_Data\\NDWI_dataset\\NDWI_World_Step2\\NDWI_world_2013_2019.tif"
# output="E:\\NGA_Project_Data\\NDWI_dataset\\NDWI_Continentwide_Step3\\NDWI_continent_2013_2019"
# 
# for shape in cutline_shapes:
#      clip_resample_raster_cutline(input_raster_dir=input_raster, output_raster_dir=output, 
#                                       input_shape_dir=shape)
# 
# input_raster2="E:\\NGA_Project_Data\\NDWI_dataset\\NDWI_World_Step2\\NDWI_world_2018_2019.tif"
# output2="E:\\NGA_Project_Data\\NDWI_dataset\\NDWI_Continentwide_Step3\\NDWI_continent_2018_2019"
# 
# for shape in cutline_shapes:
#      clip_resample_raster_cutline(input_raster_dir=input_raster2, output_raster_dir=output2, 
#                                       input_shape_dir=shape)
# =============================================================================


##Processing Terraclimate Rainfall Data     
# =============================================================================
# #mosaicing grided rasters
# inputdir="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\Raw_TERRACLIMATE_Step1\\TERRACLIMATE_2013_2019"
# outras="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\World_TERRACLIMATE_Step2\\TRCLM_pr_2013_2019.tif"
# reference="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster.tif"
# mosaic_rasters(input_dir=inputdir, output_raster=outras, ref_raster=reference)
# 
# inputdir2="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\Raw_TERRACLIMATE_Step1\\TERRACLIMATE_2018_2019"
# outras2="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\World_TERRACLIMATE_Step2\\TRCLM_pr_2018_2019.tif"
# mosaic_rasters(input_dir=inputdir2, output_raster=outras2, ref_raster=reference)
# 
# #Clipping with continent shapefiles
# cutline_shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\continent_extents","*continent*.shp"))
# input_raster="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\World_TERRACLIMATE_Step2\\TRCLM_pr_2013_2019.tif"
# output="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\Terraclimate_pr_continentwide_Step3\\TRCLM_pr_continentwide_2013_2019"
# 
# for shape in cutline_shapes:
#      clip_resample_raster_cutline(input_raster_dir=input_raster, output_raster_dir=output, 
#                                       input_shape_dir=shape)
# 
# input_raster2="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\World_TERRACLIMATE_Step2\\TRCLM_pr_2018_2019.tif"
# output2="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\Terraclimate_pr_continentwide_Step3\\TRCLM_pr_continentwide_2018_2019"
# 
# for shape in cutline_shapes:
#      clip_resample_raster_cutline(input_raster_dir=input_raster2, output_raster_dir=output2, 
#                                       input_shape_dir=shape)
# =============================================================================


##Removing Ocean Values from Landform ALOS Raster
# =============================================================================
# inras="E:\\NGA_Project_Data\\DEM_Landform\\ALOS_Landform\\World_ALOS_LF_Step02\\LF_ALOS_World.tif"
# outdir="E:\\NGA_Project_Data\\DEM_Landform\\ALOS_Landform\\World_ALOS_LF_without_Ocean_Step03"
# paste_val_on_ref_raster(input_raster=inras, outdir=outdir, raster_name="LF_ALOS_World.tif")
# =============================================================================

#Create Slope Raster from SRTM DEM
# =============================================================================
# inras="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM\\World_DEM_Step02\\SRTm_DEM_World.tif"    
# outdir="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM\\World_Slope_Step03"
# create_slope_raster(input_raster=inras, outdir=outdir, raster_name='SRTM_Slope_world.tif')
# =============================================================================

#Creating Tmean raster from Tmax and Tmnn
# =============================================================================
# inras1="E:\\NGA_Project_Data\\Temperature_data\\World_Temp_Step02\\TMAX_2013_2019\\Tmax_2013_2019.tif"
# inras2="E:\\NGA_Project_Data\\Temperature_data\\World_Temp_Step02\\TMIN_2013_2019\\Tmin_2013_2019.tif" 
# outdir="E:\\NGA_Project_Data\\Temperature_data\\World_Temp__mean_Step03"
# mean_2_rasters(input1=inras1, input2=inras2, outdir=outdir, raster_name="T_mean_2013_2019.tif")
# ==========================================================================

##Resampling Zombler SOil Data with Reference Raster
# =============================================================================
# inras="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\\data\\z_soiltype.tif"
# outdir="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\data"
# mask_by_ref_raster(input_raster=inras, outdir=outdir, raster_name="z_soiltype_resampled.tif")
# 
# #Removing Water 108 value from Ocean
# inras2="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\data\z_soiltype_resampled.tif"
# outdir2="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\\Resampled_Data"
# paste_val_on_ref_raster(input_raster=inras2, outdir=outdir2, raster_name="z_soiltype.tif")
# =============================================================================

##Facebook Population Nanfilled raster
# =============================================================================
# #2013_2019
# raster="E:\\NGA_Project_Data\\population_density\\Facebook_dataset\\2013_2019\\World_pop_data_step02\\Pop_density_FB_2013_2019.tif"
# outdir="E:\\NGA_Project_Data\\population_density\\Facebook_dataset\\2013_2019\\World_pop_data__nanfilled_step03"
# create_nanfilled_raster(input_raster=raster, outdir=outdir, raster_name="Pop_density_FB_2013_2019.tif")
# =============================================================================
