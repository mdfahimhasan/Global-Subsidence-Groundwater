import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from glob import glob

NO_DATA_VALUE=-9999

def Alexi_dat_to_tif_avg(input_dir,output_fname,searchby="*.dat",row=3000,column=7200,data_type="Float32",separator="",
                   cellsize=0.05,first_x=-180,first_y=90):
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
    arr_year=np.zeros((row,column),dtype=data_type)
    
    days_dat=glob(os.path.join(input_dir,searchby))
    
    for each in days_dat:
        arr= np.fromfile(each, dtype=data_type,count=-1, sep=separator, offset=0)
        arr_day=np.flipud(arr.reshape((row,column)))  #check this for other dataset. flipud may/may not be needed
        arr_day[(arr_day<0)|(arr_day>100)]=0  #adjust the filter values according to the raster values
        arr_year=arr_year+arr_day
    
    arr_year=arr_year/365
    arr_year[arr_year==0]=np.nan
    
    with rasterio.open(output_fname,'w',
                driver='GTiff',
                height=arr_year.shape[0],
                width=arr_year.shape[1],
                dtype=arr_year.dtype,
                crs="EPSG:4326",
                transform=(cellsize, 0.0, first_x,0.0, -cellsize, first_y),
                nodata=-9999,
                count=1) as dest:
        dest.write(arr_year,1)


#Alexi_dat_to_tif_avg(input_dir="E:\\Alexi\\2013",output_fname="E:\\NGA_Project_Data\\ET_products\\Alexi_ET\\year_wise\\Alexi_ET_2013.tif")


def dat_to_tif(input_dat,outdir,raster_name,skiprows=0,separator=None,nrows=360,ncols=720,datatype="Float32",
               cellsize=0.5,first_x=-180,first_y=90,nodata=NO_DATA_VALUE):
    """
    Converts a text/dat file (with initial rows as text as to GeoTIFF.

    Params:
    input_dat : Iput .dat file.
    outdir : Output raster directory.
    raster_name : Output raster name.
    skiprows : Number of starting rows to Skip. Defaults to 0.
    separator : Separator. Defaults to None.
    nrows : Number of rows to read. Defaults to 360.
    ncols : Number of rows to read. Defaults to 720.
    datatype : Datatype. Defaults to "Float32".
    cellsize : Pixel size. Default is 0.5 degree for GCS WGS 1984.
    first_x : X coordinate of first cell at top left corner.
    first_y : Y coordinate of first cell at top left corner.
    nodata: No data value in the final raster. Defaults to NO_DATA_VALUE of -9999.

    Returns:None.
    """

    data=np.loadtxt(fname=input_dat,skiprows=skiprows,dtype=datatype,delimiter=separator)
    arr=data.reshape((nrows,ncols))
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_raster=os.path.join(outdir,raster_name)
    
    with rasterio.open(output_raster,'w',
                    driver='GTiff',
                    height=arr.shape[0],
                    width=arr.shape[1],
                    dtype=arr.dtype,
                    crs="EPSG:4326",
                    transform=(cellsize, 0.0, first_x, 0.0, -cellsize, first_y),
                    nodata=nodata,
                    count=1) as dest:
            dest.write(arr,1)

##Coverting SOil Data by Zombler to Geotif
# =============================================================================
# path="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\\data\\z_soiltype.dat"
# outfp="E:\\NGA_Project_Data\\Soil_Data\\ZOBLERSOILDERIVED_540\\data"
# 
# dat_to_tif(input_dat=path, outdir=outfp, raster_name="z_soiltype.tif",skiprows=6,nrows=360,ncols=720)
# =============================================================================
