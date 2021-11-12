# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

# # Codes in this script is only useful for resampling and further processing InSAR processed raster data.

import os
import numpy as np
import gdal
from datetime import datetime
from System_operations import makedirs
from Raster_operations import read_raster_arr_object, write_raster


def classify_insar_raster(input_raster, output_raster_name, unit_scale, unit_change=False,
                          cnra_data=False, start_date=None, end_date=None,
                          resample_raster=True, resampled_raster_name='Resampled.tif',
                          res=0.02, output_dir='../InSAR_Data/Resampled_subsidence_data/resampled_insar_data', ):
    """
    Classify InSAR subsidence raster to project classes (<1cm/yr, 1-5cm/yr and >5cm/yr).

    Parameters :
    input_raster : Input Raster filepath.
    output_raster_name : Output raster name.
    unit_scale : Unit scale value (i.e. unit_scale=100 for m to cm conversion) for conversion.
    cnra_data : If the data is from 'California National Resources Agency', set True to convert values into cm/year.
    start_date : If cnra data, start day of the data in string format. Format must be like "2015/12/31"
                 ("Year/month/day"). Default Set to None.
    end_date : If cnra data, end day of the data in string format. Format must be like "2015/12/31" ("Year/month/day")
               Default Set to None.

    resample_raster : Set True if classified raster needs resampling. Defaults to True.
    resampled_raster_name : Resampled raster name. Default is 'Resampled.tif'.
    res : Pixel resoultion in degree. Default is 0.02 degree.
    output_dir : Output Directory path. Default set to '../InSAR_Data/Resampled_subsidence_data/resampled_insar_data'

    Returns : Classified (and resampled if modify raster=True) subsidence raster.
    """
    arr, file = read_raster_arr_object(input_raster)

    if cnra_data:
        start_day = datetime.strptime(start_date, "%Y/%m/%d")
        end_day = datetime.strptime(end_date, "%Y/%m/%d")
        months_between = round(int(str(end_day - start_day).split(" ")[0]) / 30)
        arr = arr * 30.48 * 12 / months_between

    if unit_change:
        arr = arr * unit_scale
    # New_classes
    sub_less_1cm = 1
    sub_1cm_to_5cm = 5
    sub_greater_5cm = 10
    other_values = np.nan

    arr = np.where(arr >= 0, other_values, arr)
    arr = np.where(arr >= -1, sub_less_1cm, arr)
    arr = np.where((arr < -1) & (arr >= -5), sub_1cm_to_5cm, arr)
    arr = np.where(arr < -5, sub_greater_5cm, arr)

    makedirs([output_dir])
    output_raster = os.path.join(output_dir, output_raster_name)

    outfilepath = write_raster(raster_arr=arr, raster_file=file, transform=file.transform, outfile_path=output_raster)

    if resample_raster:
        resampled_raster = os.path.join(output_dir, resampled_raster_name)

        gdal.Warp(destNameOrDestDS=resampled_raster, srcDSOrSrcDSTab=outfilepath, dstSRS='EPSG:4326', xRes=res,
                  yRes=res,
                  outputType=gdal.GDT_Float32)

    return resampled_raster


# # Iran Data Resampling
#
# input_iran = '../InSAR_Data/Iran/Iran.tif'
#
# classify_insar_raster(input_raster=input_iran, output_raster_name= 'Iran_reclass.tif',
#                       unit_change=True, unit_scale=.1, resample_raster=True,
#                       resampled_raster_name='Iran_reclass_resampled.tif', res=0.02)

