import os
import numpy as np
import rasterio
from glob import glob
import pandas as pd
from System_operations import makedirs

NO_DATA_VALUE = -9999


def Alexi_dat_to_tif_avg(input_dir, output_fname, searchby="*.dat", row=3000, column=7200, data_type="Float32",
                         separator="",
                         cellsize=0.05, first_x=-180, first_y=90):
    """
    convert .dat (binary file) to geotiff.

    Parameters
    ----------
    input_dir : Input raster Directory
    output_fname : Output raster directory with file name
    searchby= Input raster selection criteria. Default is *.dat
    row : Number of rows in the data. Have to be known. Default is set for Alexi ET Product
    column : Number of columns in the data. Have to be known. Default is set for Alexi ET Product
    data_type : Data format. Have to be known. Default "Float32" for Alexi ET Product.
    separator : Separator. Default is "".
    cellsize : Pixel size. Default is 0.05 degree for GCS WGS 1984.
    first_x : X coordinate of first cell at top left corner.
    first_y : Y coordinate of first cell at top left corner.
    """
    arr_year = np.zeros((row, column), dtype=data_type)

    days_dat = glob(os.path.join(input_dir, searchby))

    for each in days_dat:
        arr = np.fromfile(each, dtype=data_type, count=-1, sep=separator, offset=0)
        arr_day = np.flipud(arr.reshape((row, column)))  # check this for other dataset. flipud may/may not be needed
        arr_day[(arr_day < 0) | (arr_day > 100)] = 0  # adjust the filter values according to the raster values
        arr_year = arr_year + arr_day

    arr_year = arr_year / 365
    arr_year[arr_year == 0] = np.nan

    with rasterio.open(output_fname, 'w',
                       driver='GTiff',
                       height=arr_year.shape[0],
                       width=arr_year.shape[1],
                       dtype=arr_year.dtype,
                       crs="EPSG:4326",
                       transform=(cellsize, 0.0, first_x, 0.0, -cellsize, first_y),
                       nodata=-9999,
                       count=1) as dest:
        dest.write(arr_year, 1)


# Alexi_dat_to_tif_avg(input_dir="E:\\Alexi\\2013",output_fname="E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\year_wise\\Alexi_ET_2013.tif")


def txt_to_tif(input_file, outdir=None, raster_name=None, skiprows=0, separator=None, nrows=360, ncols=720,
               datatype="Float32", cellsize=0.5, first_x=-180, first_y=90, nodata=NO_DATA_VALUE):
    """
    Converts an ascii file (with initial rows as text as to GeoTIFF).

    Params:
    input_file : Input .ascii/.dat file.
    output_dir : Output raster directory.
    output_raster_name : Output raster name.
    skiprows : Number of starting rows to Skip. Defaults to 0.
    separator : Separator. Defaults to None.
    nrows : Number of rows to read. Defaults to 360.
    ncols : Number of rows to read. Defaults to 720.
    datatype : Datatype. Defaults to "Float32".
    cellsize : Pixel size. Default is 0.5 degree for GCS WGS 1984.
    first_x : X coordinate of first cell at top left corner.
    first_y : Y coordinate of first cell at top left corner.
    nodata: No data value in the final raster. Defaults to No_Data_Value of -9999.

    Returns:None.
    """

    data = np.loadtxt(fname=input_file, skiprows=skiprows, dtype=datatype, delimiter=separator)
    arr = data.reshape((nrows, ncols))

    if outdir == None:
        split = input_file.split(os.sep)
        raster_name = split[-1]
        raster_name = raster_name[:raster_name.find('.')] + '.tif'
        sep = input_file.rfind(os.sep)
        outdir = input_file[:sep + 1]
        output_raster = outdir + raster_name
    else:
        makedirs([outdir])
        output_raster = os.path.join(outdir, raster_name)

    with rasterio.open(output_raster, 'w',
                       driver='GTiff',
                       height=arr.shape[0],
                       width=arr.shape[1],
                       dtype=arr.dtype,
                       crs="EPSG:4326",
                       transform=(cellsize, 0.0, first_x, 0.0, -cellsize, first_y),
                       nodata=nodata,
                       count=1) as dest:
        dest.write(arr, 1)


# # Converting Global Lithology Data
# ascii_data=r'..\Data\Raw_Data\Global_Lithology\glim_wgs84_0point5deg.txt.asc'
# output_dir=r'..\Data\Raw_Data\Global_Lithology'
#
# txt_to_tif(input_file = ascii_data, output_dir = None, output_raster_name = None,
#              skiprows=6, separator=None, nrows = 360, ncols = 720, datatype = "Float32",
#                cellsize=0.5, first_x=-180, first_y=90, nodata=No_Data_Value)


def sedthick_csv_to_tif(sed_thickness_csv='../Data/Raw_Data/Global_Sediment_Thickness/'
                                           'EXXON_Sediment_Thickness/sedthk.csv',
                        output_ras='../Data/Raw_Data/Global_Sediment_Thickness/'
                                           'EXXON_Sediment_Thickness/Global_Sediment_thickness_EXX.tif'):
    """
    Convert Global EXXON Sediment thickness csv to tif.

    Parameters:
    sed_thickness_csv: Filepath of csv.
    output_ras: Filepath of output raster.

    Returns: A raster file of Global Sediment thickness in GeoTiff format.
    """
    sed_df = pd.read_csv(sed_thickness_csv, header=None)
    sed_df = sed_df.rename(columns={0: 'lon', 1: 'lat', 2: 'z'})
    lon_list = list(sed_df['lon'])
    lat_list = list(sed_df['lat'])
    z_list = list(sed_df['z'])

    cell_size = lon_list[1] - lon_list[0]
    num_cell_x_dir = ((max(lon_list) - min(lon_list)) / cell_size) + 1
    num_cell_y_dir = ((max(lat_list) - min(lat_list)) / cell_size) + 1

    modified_z_list = []
    first_index = 0
    last_index = int(num_cell_x_dir)

    for i in range(1, int(num_cell_y_dir) + 1):
        modified_z_list.append(list(z_list[first_index: last_index]))
        first_index += 360
        last_index += 360

    z_array = np.array(modified_z_list)

    with rasterio.open(output_ras, 'w',
                       driver='GTiff',
                       height=z_array.shape[0],
                       width=z_array.shape[1],
                       dtype=z_array.dtype,
                       crs="EPSG:4326",
                       transform=(cell_size, 0.0, -180, 0.0, -cell_size, 90),
                       nodata=NO_DATA_VALUE,
                       count=1) as dest:
        dest.write(z_array, 1)


# sedthick_csv_to_tif()
