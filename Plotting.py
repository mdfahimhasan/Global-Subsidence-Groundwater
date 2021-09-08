# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import os
from glob import glob
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show,show_hist
import numpy as np


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


    
def plot_multi_raster_hist(input_raster_dir,nrow=1,ncol=2,search_by="*.tif",figsize=(15,5),sharey=True,sharex=True,
                           bins=20,density=True,histtype='bar',alpha=0.5, ec='skyblue',fc='skyblue',
                           change_xlim=False,xlim_start=0.01,xlimmax=True,xlim_end="",
                           change_ylim=False,ylim_start=0,ylim_end=""):
    """
    Plot histogram from raster values.

    Parameters
    ----------    
    input_raster_dir : Input raster directory
    nrow : Number of rows in subplot. Defaults to 1.
    ncol : Number of columns in subplot. Defaults to 1.
    search_by : Search raster by keyword. Defaults to "*.tif".
    figsize : Histogram figure size. Defaults to (15,5).
    sharey : If Y-axis share is needed. Default is True.
    sharex : If X-axis share is needed. Default is True.
    
    bins : number of bins. Defaults to 20.
    density: If True, probability density function is plotted at Y-axis, normalized such that the integral over the range is 1
    histtype: Histogram type. Defaults to 'bar'.
    alpha : Transparency level. Defaults to 0.5.
    ec,fc : Facecolor of histogram. Defaults to 'skyblue'.
    
    change_xlim : Set to True if xlim change is required.Default set to False.
    xlim_start : Xlim start value. Default set to 0.01.
    xlimmax : If True xlim maximum will be np.nanmax(array). If False need to set xlim_end value.
    xlim_end : If xlimmax set to False give a valid integer value to set x axis end limit.
    change_ylim : Set to True if ylim change is required.Default set to False.
    ylim_start : Ylim start value. Default set to 0.
    ylim_end= Ylim end value. Default set to "" blank.If change_ylim is True set a valid integer value.
    ---------- 
    Returns: Histogram Plots.
    """
    
    plot_rasters=glob(os.path.join(input_raster_dir,search_by))
    
    i = 0
    fig, ax = plt.subplots(nrows=nrow,ncols=ncol,figsize=figsize, sharey=True,sharex=True)
    
    for raster in plot_rasters: 
        arr=read_raster_arr_object(raster,get_file=False)
        show_hist(arr,bins,alpha,histtype=histtype,density=density,edgecolor=ec,facecolor=fc,ax=ax[i])
        
        title_name=raster[raster.rfind(os.sep)+1:raster.rfind(".")]
        ax[i].set_title(title_name)
        
        if change_xlim:
            if xlimmax:
                ax[i].set_xlim(xlim_start,np.nanmax(arr))
            else:
                ax[i].set_xlim(xlim_start,xlim_end)
                
        if change_ylim:    
            ax[i].set_ylim(ylim_start,ylim_end)
        i=i+1
        