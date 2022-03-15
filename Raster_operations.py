# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from glob import glob
import numpy as np
from osgeo import gdal
import json
from fiona import transform
from shapely.geometry import box, mapping
import geopandas as gpd
import astropy.convolution as apc
from scipy.ndimage import gaussian_filter
from System_operations import *
import subprocess


No_Data_Value = -9999

referenceraster = r'../Data/Reference_rasters_shapes/Global_continents_ref_raster.tif'


def read_raster_arr_object(input_raster, band=1, raster_object=False, get_file=True, change_dtype=True):
    """
    read raster as raster object and array. If raster_object=True get only the raster array 
    
    Parameters:
    input_raster : Input raster file path
    band : Selected band to read (Default 1)
    raster_object : Set true if raster_file is a rasterio object
    get_file : Get both rasterio object and raster array file if set to True
    change_dtype : Change raster data type to float if true

    Returns : Raster numpy array and rasterio object file (rasterio_obj=False and get_file=True)
    """
    if not raster_object:
        raster_file = rio.open(input_raster)
    else:
        get_file = False

    raster_arr = raster_file.read(band)

    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan

    if get_file:
        return raster_arr, raster_file
    return raster_arr


def write_raster(raster_arr, raster_file, transform, outfile_path, no_data_value=No_Data_Value,
                 ref_file=None):
    """
    Write raster file in GeoTIFF format
    
    Parameters:
    raster_arr: Raster array data to be written
    raster_file: Original rasterio raster file containing geo-coordinates
    transform: Affine transformation matrix
    outfile_path: Outfile file path with filename
    no_data_value: No data value for raster (default float32 type is considered)
    ref_file: Write output raster considering parameters from reference raster file

    Returns : filepath of of output raster
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

    return outfile_path


def filter_lower_larger_value(input_raster, output_dir, band=1, lower=True, larger=False, filter_value=0,
                              new_value=np.nan, no_data_value=No_Data_Value):
    """
    filter out and replace value in raster

    Parameters:
    input_raster : input raster directory with raster name
    output_dir : output raster directory
    band : band  to read. Default is 1
    lower : If lower than filter value need to be filtered out. Default set to True.
    larger: If lower than filter value need to be filtered out. Default set to False. If True set lower to False.
    filter_value : value to filter out. Default is 0
    new_value : value to replace in filtered out value. Default is np.nan
    no_data_value : No data value. Default is -9999
    """

    raster_arr, raster_data = read_raster_arr_object(input_raster)
    if lower:
        raster_arr[raster_arr < filter_value] = new_value
    if larger:
        raster_arr[raster_arr > filter_value] = new_value
    raster_arr[np.isnan(raster_arr)] = no_data_value

    out_name = os.path.join(output_dir, input_raster[input_raster.rfind(os.sep) + 1:])

    write_raster(raster_arr=raster_arr, raster_file=raster_data,
                 transform=raster_data.transform, outfile_path=out_name)


def filter_specific_values(input_raster, outdir, raster_name, fillvalue=np.nan, filter_value=[10, 11],
                           new_value=False, value_new=1, no_data_value=No_Data_Value, paste_on_ref_raster=False,
                           ref_raster=referenceraster):
    """
    Filter and replace values in raster.

    Parameters:
    input_raster : input raster directory with raster name.
    outdir : Output raster directory.
    raster_name: Output raster name.
    fillvalue : Value that new raster will be filled with initially. Default set to np.nan.
    filter_value : List of values to filter. Default is [10,11].
    new_value : Set to True if filtered value needs a new value. Default set to False.
    value_new : Value that the filtered value will get if new_value is set to True. Default set to 1.
    no_data_value : No data value. Default is -9999.
    paste_on_ref_raster : Set to True if filtered values should be pasted on reference raster.
    ref_raster : Reference raster to paste values on. Defaults to referenceraster.

    Returns : Raster with filtered values.
    """
    arr, data = read_raster_arr_object(input_raster)
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    if paste_on_ref_raster:
        new_arr = ref_arr
    else:
        new_arr = np.full_like(arr, fill_value=fillvalue)

    if new_value:
        for value in filter_value:
            new_arr[arr == value] = value_new
    else:
        for value in filter_value:
            new_arr[arr == value] = value
    new_arr[np.isnan(new_arr)] = no_data_value

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)
    write_raster(raster_arr=new_arr, raster_file=data, transform=data.transform, outfile_path=output_raster)
    return output_raster


def resample_reproject(input_raster, output_dir, raster_name, reference_raster=referenceraster, resample=True,
                       reproject=False, change_crs_to="EPSG:4326", both=False, resample_algorithm='near',
                       nodata=No_Data_Value):
    """
    Resample/Reproject/Both resample and reproject a raster according to a reference raster.

    ** might not have no effect on changing nodata value in some cases. Have to do array operations in those cases.

    Parameters:
    input_raster : Input raster Directory with filename.
    output_dir : Output raster directory.
    output_raster_name: Output raster name.
    reference_raster : Reference raster path with file name.
    resample : Set True to resample only. Set reproject and both to False when resample=True.
    reproject : Set True to reproject only. Set resample and both to False when reproject=True.
    both : Set True to both resample and reproject. Set resample and reproject to False when both=True.
    resample_algorithm : Algorithm for resampling. Defaults set to 'near' (worst resampling/but fast). Also takes
                        'mode', 'max', 'min', 'sum', 'bilinear', 'cubic' etc. See gdal documentation for detail.
    nodata : No Data value in the processed raster.
    
    Returns : Resampled/Reprojected raster.
    """
    ref_arr, ref_file = read_raster_arr_object(reference_raster)
    makedirs([output_dir])
    output_raster = os.path.join(output_dir, raster_name)

    if resample:
        resampled_raster = gdal.Warp(destNameOrDestDS=output_raster, srcDSOrSrcDSTab=input_raster, format='GTiff',
                                     width=ref_arr.shape[1], height=ref_arr.shape[0], outputType=gdal.GDT_Float32,
                                     resampleAlg=resample_algorithm, dstNodata=nodata)
        del resampled_raster
    if reproject:
        reprojected_raster = gdal.Warp(destNameOrDestDS=output_raster, srcDSOrSrcDSTab=input_raster,
                                       dstSRS=change_crs_to, format='GTiff', outputType=gdal.GDT_Float32,
                                       dstNodata=nodata)
        del reprojected_raster

    if both:
        processed_raster = gdal.Warp(destNameOrDestDS=output_raster, srcDSOrSrcDSTab=input_raster,
                                     width=ref_arr.shape[1], height=ref_arr.shape[0], format='GTiff',
                                     dstSRS=change_crs_to, outputType=gdal.GDT_Float32, resampleAlg=resample_algorithm,
                                     dstNodata=nodata)
        del processed_raster

    return output_raster


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


def rename_copy_raster(input_raster, output_dir, rename=False, new_name=None, change_dtype=False):
    """
    renaming/copy a raster file and changing datatype if needed.
    Parameters:
    input_raster: input raster directory with raster name.
    output_dir : output raster (renamed) directory.
    rename : Set to True if want to rename the raster.
    new_name : new name of the raster if rename is True. Default set to None. For example 'ET.tif'.
    change_dtype : False if datatype should remain unchanged. Set True to change data type to 'Float32'.

    Returns : Renamed and copied raster with filepath.
    """

    arr, file = read_raster_arr_object(input_raster, change_dtype=change_dtype)
    makedirs([output_dir])
    if rename:
        output_raster = os.path.join(output_dir, new_name)
        write_raster(raster_arr=arr, raster_file=file, transform=file.transform, outfile_path=output_raster)
    else:
        try:
            output_raster_name = input_raster[input_raster.rfind('/') + 1:]
            output_raster = output_dir + '/' + output_raster_name
            write_raster(raster_arr=arr, raster_file=file, transform=file.transform, outfile_path=output_raster)
        except:
            output_raster_name = input_raster[input_raster.rfind(os.sep) + 1:]
            output_raster = os.path.join(output_dir, output_raster_name)
            write_raster(raster_arr=arr, raster_file=file, transform=file.transform, outfile_path=output_raster)

    return output_raster


def change_nodata_value(input_raster, new_nodata=No_Data_Value):
    """
    change no data value for single banded raster

    Parameters :
    input_raster : input raster filepath
    new_nodata : new no data value. Can be -9999 or np.nan. The default is -9999
    """

    # seeing the existing No Data value
    dataset = gdal.Open(input_raster, 1)  # 1 for update mode
    no_data_value = dataset.GetRasterBand(1).GetNoDataValue()

    # changing no data value in band
    band1 = dataset.GetRasterBand(1).ReadAsArray()
    band1[band1 == no_data_value] = new_nodata
    dataset.GetRasterBand(1).SetNoDataValue(new_nodata)

    # writing changed band to dataset
    dataset.GetRasterBand(1).WriteArray(band1)

    # closing raster properly
    dataset = None


def change_band_value_to_nodata(input_raster, outfile_path, band_val_to_change=0, nodata=No_Data_Value):
    """
    changing a band value of a raster to no data value 

    Parameters
    ----------
    input_raster_dir : Input raster directory with file name
    band_val_to_change : Existing band value that need to be changed. Default is 0
    new_value : New band value that will be set as No Data Value of the raster. The default is -9999.
    """

    # opening raster dataset and read as array
    raster_arr, raster_file = read_raster_arr_object(input_raster)

    # changing band value to No Data Value
    raster_arr = np.where(raster_arr == band_val_to_change, nodata, raster_arr)

    # writing raster file
    write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=outfile_path)


def crop_raster_by_extent(input_raster, ref_file, output_dir, raster_name, invert=False, crop=True):
    """
    Crop a raster with a given shapefile/raster. Only use to crop to extent. Cannot perform cropping to exact shapefile.

    Parameters:
    input_raster: Input raster file path.
    ref_file : Reference raster or shape file to crop input_raster. 
    output_dir : Output raster directory.
    raster_name : Cropped raster name.
    invert : If False (default) pixels outside shapes will be masked. 
             If True, pixels inside shape will be masked.
    crop : Whether to crop the raster to the extent of the shapes. Change to False if invert=True is used.

    Returns : Cropped raster.
    """

    # opening input raster
    raster_arr, input_file = read_raster_arr_object(input_raster)

    if '.shp' in ref_file:
        ref_extent = gpd.read_File(ref_file)
    else:
        ref_raster = rio.open(ref_file)
        minx, miny, maxx, maxy = ref_raster.bounds
        ref_extent = gpd.GeoDataFrame({'geometry': box(minx, miny, maxx, maxy)}, index=[0],
                                      crs=ref_raster.crs.to_string())

    ref_extent = ref_extent.to_crs(crs=input_file.crs.data)
    coords = [json.loads(ref_extent.to_json())['features'][0]['geometry']]

    # masking
    cropped_arr, cropped_transform = mask(dataset=input_file, shapes=coords, filled=True, crop=crop, invert=invert)
    cropped_arr = cropped_arr.squeeze()  # Remove axes of length 1 from the array

    # naming output file
    makedirs([output_dir])
    output_raster = os.path.join(output_dir, raster_name)

    # saving output raster
    write_raster(raster_arr=cropped_arr, raster_file=input_file, transform=cropped_transform,
                 outfile_path=output_raster)


# Unstable for Austrlia
def extract_raster_array_by_shapefile(input_raster, ref_shape, output_dir=None, raster_name=None, invert=False,
                                      crop=True, save_cropped_arr=False):
    """
    Extract a raster array within the input shapefile.

    Parameters:
    input_raster: Input raster file path.
    ref_shape : Reference shape file to crop input_raster.
    output_dir : Defaults to None. Set a output raster directory path if saved_cropped_arr is True.
    raster_name : Defaults to None. Set a output raster name if saved_cropped_arr is True.
    invert : If False (default) pixels outside shapes will be masked.
             If True, pixels inside shape will be masked.
    crop : Whether to crop the raster to the extent of the shapes. Set to False if invert=True is used.
    save_cropped_arr : Set to true if want to save cropped/masked raster array. If True, must provide output_raster_name and
                       output_dir.

    Returns : Cropped raster.
    """
    # opening input raster
    input_arr, input_file = read_raster_arr_object(input_raster)
    shapefile = gpd.read_file(ref_shape)
    geoms = shapefile['geometry'].values  # list of shapely geometries
    geoms = [mapping(geoms[0])]
    # masking
    cropped_arr, cropped_transform = mask(dataset=input_file, shapes=geoms, filled=True, crop=crop, invert=invert)
    cropped_arr = cropped_arr.squeeze()  # Remove axes of length 1 from the array

    if save_cropped_arr:
        # naming output file
        makedirs([output_dir])
        output_raster = os.path.join(output_dir, raster_name)

        # saving output raster
        write_raster(raster_arr=cropped_arr, raster_file=input_file, transform=cropped_transform,
                     outfile_path=output_raster)
    return cropped_arr, cropped_transform


def mask_by_ref_raster(input_raster, outdir, raster_name, ref_raster=referenceraster, resolution=0.02,
                       nodata=No_Data_Value, paste_on_ref_raster=False, pasted_outdir=None, pasted_raster_name=None):
    """
    Mask a Global Raster Data by Reference Raster. 

    Parameters:
    input_raster : Input raster name with filepath.
    output_dir : Output raster directory.
    output_raster_name : Output raster name.
    ref_raster : Global reference raster filepath. Defaults to referenceraster.
    resolution : Resolution of output raster. Defaults to 0.02 degree in GCS_WGS_1984.
    nodata : No data value. Defaults to No_Data_Value of -9999.
    
    #second part of the code, use if necessary.
    paste_on_ref_raster : Set True if the masked raster's value need to be pasted on reference raster.
    pasted_outdir : Set a directory for the final pasted raster.
    pasted_raster_name : Set a raster name for the pasted raster.

    Returns:None.
    """
    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    minx, miny, maxx, maxy = ref_file.bounds

    makedirs([outdir, pasted_outdir])

    output_raster = os.path.join(outdir, raster_name)
    gdal.Warp(destNameOrDestDS=output_raster, srcDSOrSrcDSTab=input_raster, format='GTiff',
              outputBounds=(minx, miny, maxx, maxy), xRes=resolution, yRes=resolution, dstSRS=ref_file.crs,
              dstNodata=nodata, targetAlignedPixels=True, outputType=gdal.GDT_Float32)

    if paste_on_ref_raster:
        pasted_raster = paste_val_on_ref_raster(input_raster=output_raster, outdir=pasted_outdir,
                                                raster_name=pasted_raster_name,
                                                ref_raster=referenceraster)
        output_raster = pasted_raster
    return output_raster


def clip_resample_raster_cutline(input_raster, output_raster_dir, input_shape, coordinate="EPSG:4326",
                                 xpixel=0.02, ypixel=0.02, NoData=No_Data_Value, naming_from_both=False,
                                 naming_from_raster=False, assigned_name=None):
    """
    clip raster by shapefile (cutline) to exact extent, resample pixel size and coordinate system

    Parameters:
    input_raster : Input raster.
    output_raster_dir : Output raster directory.
    input_shape : Input shapefile (cutline).
    coordinate : Output coordinate system. The default is "EPSG:4326".
    xpixel : X pixel size. The default is 0.02.
    ypixel :  Y pixel size. The default is 0.02.
    NoData : No Data value. By default None.
    naming_from_both : If clipped raster need to contain both name from raster and shapefile set True.
                       Otherwise set False (Default False).
    naming_from_raster : If clipped raster need to contain name from raster set True (naming_from_both must be False) .
                         Otherwise set False (Default False).
    assigned_name : If naming_from_both, naming_from_raster=False, assign a name to save processed raster ('*.tif').
                    Default set to None.

    Returns : Clipped raster array and raster file.
    """

    raster_file = gdal.Open(input_raster)

    makedirs([output_raster_dir])

    if naming_from_both:
        raster_part = input_raster[input_raster.rfind('/') + 1:]
        shape_part = input_shape[input_shape.rfind(os.sep) + 1:input_shape.rfind("_")]
        output_path = os.path.join(output_raster_dir, shape_part + "_" + raster_part)

    elif naming_from_raster:
        raster_part = input_raster[input_raster.rfind(os.sep) + 1:]
        output_path = os.path.join(output_raster_dir, raster_part)

    else:
        output_path = os.path.join(output_raster_dir, assigned_name)

    dataset = gdal.Warp(destNameOrDestDS=output_path, srcDSOrSrcDSTab=raster_file, dstSRS=coordinate,
                        targetAlignedPixels=True, xRes=xpixel, yRes=ypixel, cutlineDSName=input_shape,
                        cropToCutline=True, dstNodata=NoData)
    del dataset

    clipped_arr, clipped_file = read_raster_arr_object(output_path)

    return clipped_arr, clipped_file


def mosaic_rasters(input_dir, output_dir, raster_name, ref_raster=referenceraster, search_by="*.tif",
                   resolution=0.02, no_data=No_Data_Value):
    """
    Mosaics multiple rasters into a single raster (rasters have to be in the same directory).

    Parameters:
    input_dir : Input rasters directory.
    output_dir : Output raster directory.
    output_raster_name : Output raster name.
    ref_raster : Reference raster with filepath.
    search_by : Input raster search criteria.
    no_data : No data value. Default -9999.
    resolution: Resolution of the output raster.

    Returns: Mosaiced Raster.
    """
    input_rasters = glob(os.path.join(input_dir, search_by))
    raster_list = []
    for raster in input_rasters:
        arr, file = read_raster_arr_object(raster)
        raster_list.append(file)

    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    merged_arr, out_transform = merge(raster_list, bounds=ref_file.bounds, res=(resolution, resolution), nodata=no_data)

    merged_arr = np.where(ref_arr == 0, merged_arr, ref_arr)
    merged_arr = merged_arr.squeeze()

    makedirs([output_dir])
    out_raster = os.path.join(output_dir, raster_name)
    write_raster(raster_arr=merged_arr, raster_file=ref_file, transform=ref_file.transform, outfile_path=out_raster,
                 no_data_value=no_data, ref_file=ref_raster)

    return merged_arr, out_raster


def mosaic_two_rasters(input_raster1, input_raster2, output_dir, raster_name, ref_raster=referenceraster,
                       resolution=0.02, no_data=No_Data_Value):
    """
    Mosaics two rasters into a single raster (rasters have to be in the same directory).

    Parameters:
    input_raster1 : Input raster 1.
    input_raster2 : Input raster 2.
    output_dir : Output raster directory.
    output_raster_name : Output raster name.
    ref_raster : Reference raster with filepath.
    no_data : No data value. Default -9999.
    resolution: Resolution of the output raster.

    Returns: Mosaiced Raster.
    """
    input_rasters = [input_raster1, input_raster2]

    raster_list = []
    for raster in input_rasters:
        arr, file = read_raster_arr_object(raster)
        raster_list.append(file)

    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    merged_arr, out_transform = merge(raster_list, bounds=ref_file.bounds, res=(resolution, resolution), nodata=no_data)

    merged_arr = np.where(ref_arr == 0, merged_arr, ref_arr)
    merged_arr = merged_arr.squeeze()

    makedirs([output_dir])
    out_raster = os.path.join(output_dir, raster_name)
    write_raster(raster_arr=merged_arr, raster_file=ref_file, transform=ref_file.transform, outfile_path=out_raster,
                 no_data_value=no_data, ref_file=ref_raster)

    return merged_arr, out_raster


def mean_rasters(input_dir, outdir, raster_name, reference_raster=None, searchby="*.tif", no_data_value=No_Data_Value):
    """
    mean multiple rasters from a directory. 

    Parameters:
    input_dir :Input raster directory
    outdir : Output raster directory
    raster_name : Output raster name
    reference_raster : Reference raster for setting affine
    searchby : Searching criteria for input raster. The default is "*.tif".
    no_data_value: No Data Value default set as -9999.

    Returns: Mean output raster.
    """
    input_rasters = glob(os.path.join(input_dir, searchby))

    val = 0

    for each in input_rasters:
        each_arr, ras_file = read_raster_arr_object(each)
        if val == 0:
            arr_new = each_arr
        else:
            arr_new = arr_new + each_arr
        val = val + 1
    arr_mean = arr_new / val
    arr_mean[np.isnan(arr_mean)] = No_Data_Value

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)

    write_raster(raster_arr=arr_mean, raster_file=ras_file, transform=ras_file.transform,
                 outfile_path=output_raster, no_data_value=no_data_value, ref_file=reference_raster)


def mean_2_rasters(input1, input2, outdir, raster_name, nodata=No_Data_Value):
    """
    mean 2 rasters . 

    Parameters:
    input1 :Input raster 1 with filepath
    input2 :Input raster 2 with filepath
    outdir : Output raster directory
    raster_name : Output raster name
    nodata : No data value. Defaults to -9999

    Returns: Mean output raster.
    """
    arr1, rasfile1 = read_raster_arr_object(input1)
    arr2, rasfile2 = read_raster_arr_object(input2)

    mean_arr = np.mean(np.array([arr1, arr2]), axis=0)

    mean_arr[np.isnan(mean_arr)] = No_Data_Value

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)

    write_raster(raster_arr=mean_arr, raster_file=rasfile1, transform=rasfile1.transform,
                 outfile_path=output_raster, no_data_value=nodata)

    return output_raster


def array_multiply(input_raster1, input_raster2, outdir, raster_name, scale=None):
    """
    Multiplies 2 rasters. The rasters should be of same shape (row, column size).
    
    Parameters:
    input_raster1 : Raster 1 file with file name.
    input_raster2 : Raster 1 file with file name.
    output_dir : Output Raster Directory.
    output_raster_name : Output raster name.
    scale : Set appropriate scale value if multiplied array needs to changed with a factor. Default set to None.

    Returns: Multiplied output raster.
    """
    arr1, data1 = read_raster_arr_object(input_raster1)
    arr2, data2 = read_raster_arr_object(input_raster2)
    new_arr = np.multiply(arr1, arr2)

    if scale is not None:
        new_arr = new_arr * scale

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)
    write_raster(raster_arr=new_arr, raster_file=data1, transform=data1.transform, outfile_path=output_raster)

    return output_raster


def shapefile_to_raster(input_shape, output_dir, raster_name, use_attr=True, attribute="", add=None,
                        ref_raster=referenceraster, resolution=0.02, burnvalue=1, alltouched=False,
                        nodatavalue=No_Data_Value):
    """
    Converts polygon shapefile to raster by attribute value or burn value.

    Parameters:
    input_shape : Input shapefile filepath.
    output_raster : Output raster directory.
    output_raster_name : Output raster name.
    use_attr : Set to True if raster needs to be created using a specific attribute value. Defaults to False.
    attribute : Attribute name to use creating raster file. Defaults to "".
    add : Set to True if all values inside the raster grid should be summed. Default set to None to perform rasterizing
          with an attribute without summing.
    ref_raster : Reference raster to get minx,miny,maxx,maxy. Defaults to referenceraster.
    resolution : Resolution of the raster. Defaults to 0.05.
    burnvalue : Value for burning into raster. Only needed when use_attr is False. Defaults to 1.
    alltouched : If True all pixels touched by lines or polygons will be updated.
    nodatavalue : No_Data_Value.

    Returns: Created raster filepath.
    """

    ref_arr, ref_file = read_raster_arr_object(ref_raster)
    total_bounds = ref_file.bounds

    makedirs([output_dir])
    output_raster = os.path.join(output_dir, raster_name)

    if use_attr:
        if add is not None:
            minx, miny, maxx, maxy = total_bounds
            layer_name = input_shape[input_shape.rfind('/') + 1: input_shape.rfind('.')]
            args = ['-l', layer_name, '-a', attribute, '-tr', str(resolution), str(resolution), '-te', str(minx),
                    str(miny), str(maxx), str(maxy), '-init', str(0.0), '-add', '-ot', 'Float32', '-of', 'GTiff',
                    '-a_nodata', str(nodatavalue), input_shape, output_raster]
            sys_call = make_gdal_sys_call(gdal_command='gdal_rasterize', args=args)
            subprocess.call(sys_call)

        else:
            raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                                   outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                                   noData=nodatavalue, attribute=attribute, allTouched=alltouched)
            gdal.Rasterize(destNameOrDestDS=output_raster, srcDS=input_shape, options=raster_options, resolution=0.02)

    else:
        raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                               outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                               noData=nodatavalue, burnValues=burnvalue,
                                               allTouched=alltouched)
        gdal.Rasterize(destNameOrDestDS=output_raster, srcDS=input_shape, options=raster_options, resolution=0.02)

    return output_raster


def create_slope_raster(input_raster, outdir, raster_name):
    """
    Create Slope raster in Percent from DEM raster.

    Parameter:
    input_raster : Input raster with filepath.
    output_dir : Output raster directory.
    output_raster_name : Output raster name.

    Returns: Slope raster.
    """
    dem_options = gdal.DEMProcessingOptions(format="GTiff", computeEdges=True, alg='Horn', slopeFormat='percent',
                                            scale=100000)

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)

    slope_raster=gdal.DEMProcessing(destName=output_raster, srcDS=input_raster, processing='slope', options=dem_options)

    del slope_raster

    return output_raster


def create_nanfilled_raster(input_raster, outdir, raster_name, ref_raster=referenceraster):
    """
    Create a nan-filled raster with a reference raster. If there is nan value on raster that 
    will be filled by zero from reference raster.

    parameters:
    input_raster : Input raster.
    output_dir : Output raster directory.
    output_raster_name : output raster name.
    ref_raster : Reference raster on which initial raster value is pasted. Defaults to referenceraster.

    Returns:None.
    """
    ras_arr, ras_file = read_raster_arr_object(input_raster)
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    ras_arr = ras_arr.flatten()
    ref_arr = ref_arr.flatten()

    new_arr = np.where(np.isnan(ras_arr), ref_arr, ras_arr)
    new_arr = new_arr.reshape(ref_file.shape[0], ref_file.shape[1])

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)
    write_raster(raster_arr=new_arr, raster_file=ras_file, transform=ras_file.transform, outfile_path=output_raster)

    return output_raster


def paste_val_on_ref_raster(input_raster, outdir, raster_name, value=0, ref_raster=referenceraster):
    """
    Paste value from a raster on the reference raster. If there are nan values on raster that 
    will be filled by nan from reference raster.

    parameters:
    input_raster : Input raster.
    output_dir : Output raster directory.
    output_raster_name : output raster name.
    value : Value in reference raster used in comparison.
    ref_raster : Reference raster on which initial raster value is pasted. Defaults to referenceraster.

    Returns:None.
    """
    ras_arr, ras_file = read_raster_arr_object(input_raster)
    ref_arr, ref_file = read_raster_arr_object(ref_raster)

    ras_arr = ras_arr.flatten()
    ref_arr = ref_arr.flatten()

    new_arr = np.where(ref_arr == value, ras_arr, ref_arr)
    new_arr = new_arr.reshape(ref_file.shape[0], ref_file.shape[1])

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)
    write_raster(raster_arr=new_arr, raster_file=ras_file, transform=ras_file.transform, outfile_path=output_raster)

    return output_raster


def apply_gaussian_filter(input_raster, outdir, raster_name, sigma=3, ignore_nan=True, normalize=True,
                          nodata=No_Data_Value, ref_raster=referenceraster):
    """
    Applies Gaussian filter to raster.

    Parameters:
    input_raster : Input Raster.
    output_dir : Output Raster Directory.
    output_raster_name : Output raster name.
    sigma : Standard Deviation for gaussian kernel. Defaults to 3.
    ignore_nan :  Set true to ignore nan values during convolution.
    normalize : Set true to normalize the filtered raster at the end.
    nodata : No_Data_Value.
    ref_raster : Reference Raster. Defaults to referenceraster.

    Returns: Gaussian filtered raster.
    """
    raster_arr, raster_file = read_raster_arr_object(input_raster)
    if ignore_nan:
        Gauss_kernel = apc.Gaussian2DKernel(x_stddev=sigma, x_size=3 * sigma, y_size=3 * sigma)
        raster_arr_flt = apc.convolve(raster_arr, kernel=Gauss_kernel, preserve_nan=True)

    else:
        raster_arr[np.isnan(raster_arr)] = 0
        raster_arr_flt = gaussian_filter(input=raster_arr, sigma=sigma,
                                         order=0)  # order 0 corresponds to convolution with a Gaussian kernel

    if normalize:
        if ignore_nan:
            raster_arr_flt[np.isnan(raster_arr_flt)] = 0
        raster_arr_flt = np.abs(raster_arr_flt)
        raster_arr_flt -= np.min(raster_arr_flt)
        raster_arr_flt /= np.ptp(raster_arr_flt)

    ref_arr = read_raster_arr_object(ref_raster, get_file=False)
    raster_arr_flt[np.isnan(ref_arr)] = nodata

    makedirs([outdir])
    output_raster = os.path.join(outdir, raster_name)
    write_raster(raster_arr=raster_arr_flt, raster_file=raster_file, transform=raster_file.transform,
                 outfile_path=output_raster)

    return output_raster


def compute_proximity(input_raster, output_dir, raster_name, target_values=(1,), nodatavalue=No_Data_Value):

    makedirs([output_dir])
    output_raster = os.path.join(output_dir, raster_name)

    inras_file = gdal.Open(input_raster, gdal.GA_ReadOnly)
    inras_band = inras_file.GetRasterBand(1)
    driver = gdal.GetDriverByName('GTiff')

    x_size = inras_file.RasterXSize
    y_size = inras_file.RasterYSize
    dest_ds = driver.Create(output_raster, x_size, y_size, 1, gdal.GDT_Float32)
    dest_ds.SetProjection(inras_file.GetProjection())
    dest_ds.SetGeoTransform(inras_file.GetGeoTransform())
    dest_band = dest_ds.GetRasterBand(1)
    dest_band.SetNoDataValue(nodatavalue)

    target_values_list = list(target_values)
    values = 'VALUES=' + ','.join(str(val) for val in target_values_list)

    gdal.ComputeProximity(inras_band, dest_band, [values, "DISTUNITS=GEO"])

    inras_file, inras_band, dest_ds, dest_band = None, None, None, None

    return output_raster
