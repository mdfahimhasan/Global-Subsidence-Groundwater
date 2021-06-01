import os
import ee
import zipfile
import rasterio
import requests
import pandas as pd
import geopandas as gpd
from glob import glob
from sysops import makedirs
from Raster_operations import mosaic_rasters
import math

NO_DATA_VALUE=-9999
referenceraster2="E:\\NGA_Project_Data\\shapefiles\\Country_continent_full_shapes\\Global_continents_ref_raster.tif"

# =============================================================================
# # For MODIS/ImageCollection Data
# =============================================================================

def download_imagecollection_gee(yearlist,start_month,end_month,output_dir,inputshp,
                                 gee_scale=5000,dataname="ET_",name="MODIS",
                                  bandname="ET",imagecollection="MODIS/006/MOD16A2"):
    """
    Download Imagecollection data (i.e. MODIS) from Google Earth Engine by yearly basis
    Parameters
    ----------
    yearlist : range: Year for which data will be downloaded, i.e., range(2010,2020)
    start_month : Integer: Start month of data
    end_month : Integer: End month of data
    output_dir : File directory path: Location to downloaded data 
    inputshp : File directory path: Location of input shapefile (data download extent)
    gee_scale : Download Scale
    bandname : Band to download from Google earth engine
    imagecollection : Imagecollection name
    dataname : Name extension added to downloaded file
    name : Name added to file when downloading
    """
    ##Initialize
    ee.Initialize()
    data_download=ee.ImageCollection(imagecollection)
    
    ##Define Extent
    minx,miny,maxx,maxy=gpd.read_file(inputshp).total_bounds

    gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
    
    ##Date range
    for year in yearlist:
        start_date=ee.Date.fromYMD(year,start_month,1)
        
        if end_month==12:
            end_date=ee.Date.fromYMD(year+1,1,1)
        
        else:
            end_date=ee.Date.fromYMD(year,end_month+1,1)
        
        ##for timeseries data only
        if start_month<=end_month:
            start_date=ee.Date.fromYMD(year-1,start_month,1)
            
        ##Get URL
        data_total=data_download.select(bandname).filterDate(start_date,end_date).max().multiply(1).toFloat()  #Change the function max() and value within Multiply based on purpose
    
        data_url=data_total.getDownloadURL({'name':name,
                                            'crs':"EPSG:4326",
                                            'scale':gee_scale,
                                            'region':gee_extent})
    
        #dowloading the data
        gee_vars=[dataname]
        gee_url=[data_url]
        for var,url in zip(gee_vars,gee_url):
            key_word=var+inputshp[inputshp.rfind(os.sep)+1:inputshp.rfind("_")]
            local_file_name=os.path.join(output_dir,key_word+'_'+str(year)+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)


def download_imagecollection_mean(yearlist,start_month,end_month,output_dir,shapecsv=None,inputshp_dir=None, 
                                  search_criteria="*worldGrid*.shp", gee_scale=2000,dataname="ET_",name="MODIS",
                                  bandname="ET",imagecollection="MODIS/006/MOD16A2",factor=1,select_bandname=True):
    """
    Download Imagecollection data (i.e. MODIS/Landsat) from Google Earth Engine by range years' mean
    Parameters
    ----------
    yearlist : List of years for which data will be downloaded, i.e., [2010,2020]
    start_month : Start month of data
    end_month : End month of data
    output_dir : File directory path to downloaded data
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.
    inputshp_dir : File directory path of input shapefiles (data download extent). Defaults to None.
    search_criteria : Search criteria for input shapefiles. Defaults to "*worldGrid*.shp"
    gee_scale : Download Scale
    bandname : Band to download from Google earth engine
    imagecollection : Imagecollection name
    dataname : Name extension added to downloaded file
    name : Name added to file when downloading
    factor : Factor (if needed) to multiply with the band 
    select_bandname : Set to False if imagecollection has no designated bandname (for example: https://samapriya.github.io/awesome-gee-community-datasets/projects/hrsl/#earth-engine-snippet)
    """
    ##Initialize
    ee.Initialize()
    data_download=ee.ImageCollection(imagecollection)
    
    ##Date Range Creation
    start_date=ee.Date.fromYMD(yearlist[0],start_month,1)
        
    if end_month==12:
        end_date=ee.Date.fromYMD(yearlist[1]+1,1,1)
        
    else:
        end_date=ee.Date.fromYMD(yearlist[1],end_month+1,1)
        
    ##Reducing the ImageCollection
    if select_bandname:
        data_total=data_download.select(bandname).filterDate(start_date,end_date).mean().multiply(factor).toFloat()
    else:
        data_total=data_download.filterDate(start_date,end_date).mean().multiply(factor).toFloat() 
        
    #Creating Output directory
    makedirs(output_dir)
    
    if inputshp_dir:
        shapes=glob(os.path.join(inputshp_dir,search_criteria))
        for shape in shapes:
            ##Define Extent
            minx,miny,maxx,maxy=gpd.read_file(shape).total_bounds
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
              
            #Download URL
            data_url=data_total.getDownloadURL({'name':name,
                                                'crs':"EPSG:4326",
                                                'scale':gee_scale,
                                                'region':gee_extent})
            #dowloading the data
            key_word=shape[shape.rfind(os.sep)+1:shape.rfind("_")]+'_'+dataname
            local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(data_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)  
            
    if shapecsv:
        coords_df=pd.read_csv(shapecsv)
        for index,row in coords_df.iterrows():
            #Define Extent
            minx=row['minx']; miny=row['miny']; maxx=row['maxx']; maxy=row['maxy']
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
            
            #Download URL
            data_url=data_total.getDownloadURL({'name':name,
                                            'crs':"EPSG:4326",
                                            'scale':gee_scale,
                                            'region':gee_extent})  
            #dowloading the data
            key_word=row['shape']
            local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(data_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)     

# =============================================================================
# #Download GRACE ensemble data gradient over the year
# =============================================================================
def download_Grace_gradient(yearlist,start_month,end_month,output_dir,shapecsv=None,inputshp_dir=None,
                            search_criteria="*worldGrid*.shp",gee_scale=5000):
    """
    Download ensembled Grace data gradient from Google Earth Engine 
    ----------
    yearlist : List of years for which data will be downloaded, i.e., [2010,2017]
    start_month : Start month of data.
    end_month : End month of data.
    output_dir : File directory path to downloaded data.
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.
    search_criteria : Search criteria for input shapefiles. Defaults to "*worldGrid*.shp"
    inputshp_dir : File directory path of input shapefiles (data download extent). Defaults to None.
    
    """
    ##Initialize
    ee.Initialize()
    
    ##Getting download url for ensembled grace data
    Grace=ee.ImageCollection("NASA/GRACE/MASS_GRIDS/LAND")
    
    ##Date Range Creation
    start_date=ee.Date.fromYMD(yearlist[0],start_month,1)
        
    if end_month==12:
        end_date=ee.Date.fromYMD(yearlist[1]+1,1,1)
        
    else:
        end_date=ee.Date.fromYMD(yearlist[1],end_month+1,1)

    #a function to add timestamp on the image
    def addTime(image):
        """
        Scale milliseconds by a large constant to avoid very small slopes in the linear regression output.
        """
        return image.addBands(image.metadata('system:time_start').divide(1000*3600*24*365))
    
    #Reducing ImageColeection
    grace_csr=Grace.select("lwe_thickness_csr").filterDate(start_date,end_date).map(addTime)
    grace_csr_trend=grace_csr.select(['system:time_start','lwe_thickness_csr'])\
        .reduce(ee.Reducer.linearFit()).select('scale').toFloat()

    grace_gfz=Grace.select("lwe_thickness_gfz").filterDate(start_date,end_date).map(addTime)
    grace_gfz_trend=grace_gfz.select(['system:time_start','lwe_thickness_gfz'])\
        .reduce(ee.Reducer.linearFit()).select('scale').toFloat()

    grace_jpl=Grace.select("lwe_thickness_jpl").filterDate(start_date,end_date).map(addTime)
    grace_jpl_trend=grace_jpl.select(['system:time_start','lwe_thickness_jpl'])\
        .reduce(ee.Reducer.linearFit()).select('scale').toFloat()
    
    #Ensembling
    grace_ensemble_avg=grace_csr_trend.select(0).add(grace_gfz_trend.select(0)).select(0)\
        .add(grace_jpl_trend.select(0)).select(0).divide(3)
    
    #Creating Output Directory
    makedirs(output_dir)
    
    if inputshp_dir:
        shapes=glob(os.path.join(inputshp_dir,search_criteria))
        for shape in shapes:
            ##Define Extent
            minx,miny,maxx,maxy=gpd.read_file(shape).total_bounds
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
    
            download_url=grace_ensemble_avg.getDownloadURL({'name':'Grace',
                                                                'crs':"EPSG:4326",
                                                                'scale':gee_scale,
                                                                'region':gee_extent})
            #dowloading the data
            key_word=shape[shape.rfind(os.sep)+1:shape.rfind("_")]+'_'+'Grace_'
            local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(download_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)      
            
    if shapecsv:
        coords_df=pd.read_csv(shapecsv)
        for index,row in coords_df.iterrows():
            #Define Extent
            minx=row['minx']; miny=row['miny']; maxx=row['maxx']; maxy=row['maxy']
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
    
            download_url=grace_ensemble_avg.getDownloadURL({'name':'Grace',
                                                                'crs':"EPSG:4326",
                                                                'scale':gee_scale,
                                                                'region':gee_extent})
            #dowloading the data
            key_word=row['shape']+'Grace'
            local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(download_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)   

                    
# =============================================================================
# #Stationary Single Image Download 
# =============================================================================

def download_image_gee(output_dir,bandname,shapecsv=None,inputshp_dir=None,search_criteria="*worldGrid*.shp",factor=1,
                       gee_scale=5000, dataname="DEM_",name="DEM",image="USGS/SRTMGL1_003",Terrain_slope=False):
    """
    Download Stationary/Single Image from Google Earth Engine.
    Parameters
    ----------
    output_dir : File directory path to downloaded data.
    bandname : Bandname of the data.
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.
    search_criteria : Search criteria for input shapefiles. Defaults to "*worldGrid*.shp".
    inputshp_dir : File directory path of input shapefiles (data download extent). Defaults to None.
    factor : Factor to multiply with (if there is scale mentioned in the data) to convert to original scale.
    gee_scale : Pixel size in m. DEfaults to 5000.
    dataname : Name extension added to downloaded file.
    name : Name added to file when downloading.
    image : Image name from Google Earth Engine.
    Terrain_slope : If the image is a SRTM DEM data and slope data download is needed. Defaults to False. 
    """
    
    ##Initialize
    ee.Initialize()
    
    data_download=ee.Image(image).select(bandname).multiply(factor).toFloat()                      
    
    if Terrain_slope:
        data_download=ee.Terrain.slope(data_download)
        
    #Creating Output Directory
    makedirs(output_dir)
    
    if inputshp_dir:
        shapes=glob(os.path.join(inputshp_dir,search_criteria))
        for shape in shapes:
            ##Define Extent
            minx,miny,maxx,maxy=gpd.read_file(shape).total_bounds
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
    
            download_url=data_download.getDownloadURL({'name':'Grace',
                                                        'crs':"EPSG:4326",
                                                        'scale':gee_scale,
                                                        'region':gee_extent})
            #dowloading the data
            key_word=shape[shape.rfind(os.sep)+1:shape.rfind("_")]
            local_file_name=os.path.join(output_dir,key_word+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(download_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)      
        
    if shapecsv:
        coords_df=pd.read_csv(shapecsv)
        for index,row in coords_df.iterrows():
            #Define Extent
            minx=row['minx']; miny=row['miny']; maxx=row['maxx']; maxy=row['maxy']
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
    
            download_url=data_download.getDownloadURL({'name':'Grace',
                                                        'crs':"EPSG:4326",
                                                        'scale':gee_scale,
                                                        'region':gee_extent})
            #dowloading the data
            key_word=row['shape']
            local_file_name=os.path.join(output_dir,key_word+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(download_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)

# =============================================================================
# #Cloudmask function for landsat 8 data
# =============================================================================
def cloudmaskL8sr(image):
    """
    Function to mask clouds based on the pixel_qa band of Landsat 8 SR data. Used in combination with Landasat 
    GEE download function.
    
    param : {ee.Image} image input Landsat 8 SR image
    return : {ee.Image} cloudmasked Landsat 8 image
    """  
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
                   .And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)


# =============================================================================
# #Download Landsat 8 datawith cloudmaking    ## Not Ready
# =============================================================================
# =============================================================================
# def download_Landsat_derived_product(yearlist,start_month,end_month,output_dir,inputshp,
#                                   gee_scale=5000,dataname="Landsat_",name="Landsat",
#                                   imagecollection='LANDSAT/LC08/C01/T1_SR',factor=0.0001,index='NDWI'):
#     """
#     Download Imagecollection mean data of Landsat products (NDVI,NDWI,EVI), with cloudmask applied,
#     from Google Earth Engine for range of years. 
#     
#     ***The code works best for small spatial and temporal scale data (Otherwise takes a lot of time).***
#     Parameters
#     ----------
#     yearlist : list: Year for which data will be downloaded, i.e., [2010,2020]
#     start_month : Integer: Start month of data
#     end_month : Integer: End month of data
#     output_dir : File directory path: Location to downloaded data 
#     inputshp : File directory path: Location of input shapefile (data download extent)
#     gee_scale : Download Scale
#     bandname : Band to download from Google earth engine
#     imagecollection : Imagecollection name
#     dataname : Name extension added to downloaded file
#     name : Name added to file when downloading
#     factor : Factor (if needed) to multiply with the band
#     index : Index to download ('NDVI'/'NDWI','EVI'). Defaults to 'NDWI'
#     """
#     ##Initialize
#     ee.Initialize()
#     data_download=ee.ImageCollection(imagecollection)
#     
#     ##Define Extent
#     minx,miny,maxx,maxy=gpd.read_file(inputshp).total_bounds
#     gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
#     
#     ##Date Range Creation
#     start_date=ee.Date.fromYMD(yearlist[0],start_month,1)
#         
#     if end_month==12:
#         end_date=ee.Date.fromYMD(yearlist[1]+1,1,1)
#         
#     else:
#         end_date=ee.Date.fromYMD(yearlist[1],end_month+1,1)
#     
#     cloudmasked=data_download.filterDate(start_date,end_date).filterBounds(gee_extent).map(cloudmaskL8sr)\
#         .mean().multiply(factor).toFloat()        
#     
#     ##Get URL
#     if index=='NDVI':
#         NDVI=cloudmasked.normalizedDifference(['B5','B4'])
#         data_url=NDVI.getDownloadURL({'name':name,
#                                       'crs':"EPSG:4326",
#                                       'scale':gee_scale})
#     if index=="NDWI":
#         NDWI=cloudmasked.normalizedDifference(['B5','B6'])
#         data_url=NDWI.getDownloadURL({'name':name,
#                                       'crs':"EPSG:4326",
#                                       'scale':gee_scale})
#     if index=="EVI":
#         EVI = cloudmasked.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
#         {'NIR': cloudmasked.select('B5'),
#         'RED': cloudmasked.select('B4'),
#         'BLUE': cloudmasked.select('B2')})
#         
#         data_url=EVI.getDownloadURL({'name':name,
#                                      'crs':"EPSG:4326",
#                                       'scale':gee_scale})
#     #Creating Output directory
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
# 
#     #dowloading the data
#     key_word=inputshp[inputshp.rfind(os.sep)+1:inputshp.rfind("_")]+'_'+dataname
#     local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
#     print('Downloading',local_file_name,'.....')
#     r=requests.get(data_url,allow_redirects=True)
#     open(local_file_name,'wb').write(r.content)     
# =============================================================================

# =============================================================================
# #MODIS CLoudmask    
# =============================================================================
def cloudmask_MODIS09A1(image):
    """
    Removing cloudmask from MODIS (MOD09A1.006 Terra Surface Reflectance 8-Day Global 500m) data.

    param : {ee.Image} image input MODIS09A1 SR image
    return : {ee.Image} cloudmasked MODIS09A1 image
    """
    # Bits 0, 1 and 2 are cloud shadow and cloud, respectively.
    cloudShadow = (1 << 2)
    cloudMask0 = (1 << 0)
    cloudMask1 = (1 << 1)
    #Get the StateQA band.
    qa = image.select('StateQA')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadow).eq(0) \
                   .And(qa.bitwiseAnd(cloudMask0).eq(0))\
                       .And(qa.bitwiseAnd(cloudMask1).eq(0))
    return image.updateMask(mask)

# =============================================================================
# #Download MODIS 09A1  datawith cloudmaking    
# =============================================================================
def download_MODIS_derived_product(yearlist,start_month,end_month,output_dir,shapecsv=None,inputshp_dir=None,
                                  search_criteria="*worldGrid*.shp", gee_scale=5000,
                                  dataname="MODIS_",name="MODIS",
                                  imagecollection="MODIS/006/MOD09A1",factor=0.0001,index='NDWI'):
    """
    Download Imagecollection mean data of Landsat products (NDVI,NDWI,EVI), with cloudmask applied,
    from Google Earth Engine for range of years. 
    
    ***The code works best for small spatial and temporal scale data (Otherwise takes a lot of time).***
    Parameters
    ----------
    yearlist : list: Year for which data will be downloaded, i.e., [2010,2020]
    start_month : Integer: Start month of data
    end_month : Integer: End month of data
    output_dir : File directory path: Location to downloaded data 
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.
    inputshp_dir : File directory path of input shapefiles (data download extent). Defaults to None.
    search_criteria : Search criteria for input shapefiles. Defaults to "*worldGrid*.shp"
    gee_scale : Download Scale
    bandname : Band to download from Google earth engine
    imagecollection : Imagecollection name
    dataname : Name extension added to downloaded file
    name : Name added to file when downloading
    factor : Factor (if needed) to multiply with the band
    index : Index to download ('NDVI'/'NDWI'). Defaults to 'NDWI'
    """    
    ##Initialize
    ee.Initialize()
    
    ##Date Range Creation
    start_date=ee.Date.fromYMD(yearlist[0],start_month,1)
        
    if end_month==12:
        end_date=ee.Date.fromYMD(yearlist[1]+1,1,1)
        
    else:
        end_date=ee.Date.fromYMD(yearlist[1],end_month+1,1)
    
    #Creating Output Directory
    makedirs(output_dir)
    
    dataset=ee.ImageCollection(imagecollection)
    cloudmasked=dataset.filterDate(start_date,end_date).map(cloudmask_MODIS09A1)
    
    ##Get URL
    if index=='NDVI':
        NIR=cloudmasked.select('sur_refl_b02').mean().multiply(factor).toFloat() 
        Red=cloudmasked.select('sur_refl_b01').mean().multiply(factor).toFloat()
        data_download=NIR.subtract(Red).divide(NIR.add(Red))
        
    if index=="NDWI":
        NIR=cloudmasked.select('sur_refl_b02').mean().multiply(factor).toFloat() 
        SWIR=cloudmasked.select('sur_refl_b06').mean().multiply(factor).toFloat()
        data_download=NIR.subtract(SWIR).divide(NIR.add(SWIR))
    
    if inputshp_dir:
        shapes=glob(os.path.join(inputshp_dir,search_criteria))
        for shape in shapes:
            ##Define Extent
            minx,miny,maxx,maxy=gpd.read_file(shape).total_bounds
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
              
            #Download URL
            data_url=data_download.getDownloadURL({'name':name,
                                                'crs':"EPSG:4326",
                                                'scale':gee_scale,
                                                'region':gee_extent})
            #dowloading the data
            key_word=shape[shape.rfind(os.sep)+1:shape.rfind("_")]+'_'+dataname
            local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(data_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)  
            
    if shapecsv:
        coords_df=pd.read_csv(shapecsv)
        for index,row in coords_df.iterrows():
            #Define Extent
            minx=row['minx']; miny=row['miny']; maxx=row['maxx']; maxy=row['maxy']
            gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
            
            #Download URL
            data_url=data_download.getDownloadURL({'name':name,
                                            'crs':"EPSG:4326",
                                            'scale':gee_scale,
                                            'region':gee_extent})  
            #dowloading the data
            key_word=row['shape']
            local_file_name=os.path.join(output_dir,key_word+str(yearlist[0])+'_'+str(yearlist[1])+'.zip')
            print('Downloading',local_file_name,'.....')
            r=requests.get(data_url,allow_redirects=True)
            open(local_file_name,'wb').write(r.content)         

# =============================================================================
# #Extract Data  
# =============================================================================
def extract_data(zip_dir,out_dir,searchby="*.zip",rename_file=True):
    """
    Extract zipped data 
    Parameters
    ----------
    zip_dir : File Location
    out_dir : File Location where data will be extracted
    searchby : Keyword for searching files, default is "*.zip".
    key_word : Extracted files start name, default is 'World_'.
    rename_file : True if file rename is required while extracting
    """
    print('Extracting zip files.....')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for zip_file in glob(os.path.join(zip_dir,searchby)):
        with zipfile.ZipFile(zip_file,'r') as zip_ref:
            if rename_file:
                zip_key=zip_file[zip_file.rfind(os.sep)+1:zip_file.rfind(".")]
                zip_info=zip_ref.infolist()[0]
                zip_info.filename=zip_key+'.tif'
                zip_ref.extract(zip_info,path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)
                

# =============================================================================
# shp="E:\\NGA_Project_Data\\scratch_files\\WorldGrid_35_.shp"
# minx,miny,maxx,maxy=gpd.read_file(shp).total_bounds
# gee_extent=ee.Geometry.Rectangle((minx,miny,maxx,maxy))
# imcollect=ee.ImageCollection("MODIS/006/MOD09A1").filterDate('2013-01-01','2019-12-31').filterBounds(gee_extent)
# masked=imcollect.map(cloudmask_MODIS09A1)
# data=masked.select('sur_refl_b02').mean().multiply(0.0001).toFloat()
# data2=masked.select('sur_refl_b06').mean().multiply(0.0001).toFloat()
# index=data.subtract(data2).divide(data.add(data2))
# data_url=index.getDownloadURL({'name':'M',
# 'crs':'EPSG:4326',
# 'scale':10000})
# =============================================================================

# =============================================================================
# #Download data from URL
# =============================================================================

# =============================================================================
# import os
# import requests
# 
# def download_from_url(out_dir,urllist):    
#     """
#     Download Data from url
#     Parameters
#     ----------
#     out_dir : Location of download
#     urllist : A list of url to download from
#     """
#     
#     #make dir if not created
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     
#     for url in url_list:
#         fname=url.rsplit('/',1)[1]
#         out_fname=os.path.join(out_dir,fname)
#         print("Downloading",fname,"......")
#         r=requests.get(url,allow_redirects=True)
#         open(out_fname,'wb').write(r.content)
#         
# #url list
# url_list=["https://geo.nsstc.nasa.gov/SPoRT/outgoing/crh/4sujay/global_alexi_terra_cfsr_2013.tar"]
# 
# 
# download_from_url(out_dir="F:\\Alexi_Data",urllist=url_list)
# 
# =============================================================================


# =============================================================================
# #For Global Cropland Data
# shape_path=os.path.join("G:\\NGA_Project_Data\\continent_extents","*.shp")
# shapes=glob(shape_path)
# zipdir="E:\\python_files\\Intro to Python GIS data\\GEE CE  Data\\GFSAD_Crop"
# outdir="E:\\python_files\\Intro to Python GIS data\\GEE CE  Data\\GFSAD_Crop"
# 
# for shape in shapes:
#     key_name=shape[shape.rfind(os.sep)+1:shape.rfind('_')]
#     
#     download_image_gee(output_dir="E:\\python_files\\Intro to Python GIS data\\GEE CE  Data\\GFSAD_Crop",
#                        inputshp=shape,gee_scale=5000)
#     
#     extract_data(zip_dir=zipdir,out_dir=outdir,key_word=key_name+"_",rename_file=True)
# =============================================================================

#Download MODIS NDWI (Cloudmasked)
#2013_2019
# =============================================================================
# csv=r"E:\NGA_Project_Data\shapefiles\GEE_Download_coords.csv"
# outdir="E:\\NGA_Project_Data\\NDWI_dataset\\2013_2019\\Raw_NDWI_13_19_Step01"
# mosaic_dir="E:\\NGA_Project_Data\\NDWI_dataset\\2013_2019\\World_NDWI_13_19_Step02"
# 
# download_MODIS_derived_product(yearlist=[2013,2019],start_month=1,end_month=12,output_dir=outdir,
#                                    shapecsv=csv,inputshp_dir=None,
#                                   gee_scale=5000,dataname="MODIS_",name="MODIS",index='NDWI')
# extract_data(zip_dir=outdir, out_dir=outdir)
# 
# mosaic_rasters(input_dir=outdir, output_dir=mosaic_dir, raster_name="NDWI_2013_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# =============================================================================

#2018_2019
# =============================================================================
# outdir2="E:\\NGA_Project_Data\\NDWI_dataset\\2018_2019\\Raw_NDWI_18_19_Step01"
# mosaic_dir2="E:\\NGA_Project_Data\\NDWI_dataset\\2018_2019\\World_NDWI_18_19_Step02"
# for shape in shapes:
#     download_MODIS_derived_product(yearlist=[2018,2019],start_month=1,end_month=12,output_dir=outdir2,
#                                    inputshp=shape,
#                                   gee_scale=5000,dataname="MODIS_",name="MODIS",index='NDWI')
#     extract_data(zip_dir=outdir2, out_dir=outdir2)
# 
# mosaic_rasters(input_dir=outdir2, output_dir=mosaic_dir2, raster_name="NDWI_2018_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True) 
# =============================================================================
   
    
##For TERRACLIMATE precipitation Data
# =============================================================================
# shape_path="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# shapes=glob(os.path.join(shape_path,"*world*.shp"))
# #2013_2019 TERRACLIMATE download
# download_dir="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\2013_2019\\Raw_TRCLM_Step01"
# mosaic_dir="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\2013_2019\\World_TRCLM_Step02"
# 
# for shape in shapes:
#     download_imagecollection_mean([2013,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir,
#                                  inputshp=shape, gee_scale=5000,dataname="TRCLM_",name="TERRACLIMATE",
#                                  bandname="pr",imagecollection="IDAHO_EPSCOR/TERRACLIMATE")
# 
# 
#     extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)  
#     
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="TRCLM_pr_2013_2019.tif",
#                      ref_raster=referenceraster2,create_outdir=True) 
# #2018_2019 TERRACLIMATE download
# download_dir="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\2018_2019\\Raw_TRCLM_Step01"
# mosaic_dir="E:\\NGA_Project_Data\\Rainfall_data\\TERRACLIMATE\\2018_2019\\World_TRCLM_Step02"
# 
# for shape in shapes:
#     download_imagecollection_mean([2018,2019],start_month=1,end_month=12,output_dir=download_dir,
#                                  inputshp=shape, gee_scale=5000,dataname="TRCLM_",name="TERRACLIMATE",
#                                  bandname="pr",imagecollection="IDAHO_EPSCOR/TERRACLIMATE")
# 
#     extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)  
# 
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="TRCLM_pr_2018_2019.tif",
#                      ref_raster=referenceraster2,create_outdir=True) 
# =============================================================================

##Download Modis Landcover Data
# =============================================================================
# shape_path="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# shapes=glob(os.path.join(shape_path,"*worldGrid*.shp"))
# outdir="E:\\NGA_Project_Data\\Land_Use_Data\\Raw\\MCD12Q1_UMD\\MODIS_LC_gridded_Step1"
# 
# for shape in shapes:
#     download_imagecollection_gee(yearlist=range(2019,2020),start_month=1,end_month=12,output_dir=outdir,inputshp=shape,
#                                  gee_scale=5000,dataname="LC_",name="MODIS",
#                                   bandname="LC_Type1",imagecollection="MODIS/006/MCD12Q1")
#     extract_data(zip_dir=outdir, out_dir=outdir)
# =============================================================================

##Download and Mosaic Grace data
# =============================================================================
# shapes="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# csv=r"E:\NGA_Project_Data\shapefiles\GEE_Download_coords.csv"
# outdir="E:\\NGA_Project_Data\\GRACE\\GEE_WorldGrid_Step1"
# mosaic_dir="E:\\NGA_Project_Data\\GRACE\\Global_resampled_Step2"
# download_Grace_gradient(yearlist=[2013,2017],start_month=1,end_month=12,inputshp_dir=None, shapecsv=csv,output_dir=outdir,
#                        gee_scale=5000)
# extract_data(zip_dir=outdir, out_dir=outdir)
# mosaic_rasters(input_dir=outdir,output_dir=mosaic_dir,raster_name="GRACE_2013_2017.tif",ref_raster=referenceraster2,
#                search_by="*.tif",no_data=NO_DATA_VALUE,resolution=0.05)
# =============================================================================

##Download Soil Moisture Data from SMAP
# =============================================================================
# shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee","*worldGrid*.shp"))
# outdir="E:\\NGA_Project_Data\\Soil_Moisture\\Soil_Moisture_SMAP_2015_2019\\GEE_WorldGrid_Step1"
# for shape in shapes:
#     download_imagecollection_mean(yearlist=[2015,2019], start_month=1, end_month=12, output_dir=outdir,
#                                   inputshp=shape,dataname="SM_",name="SMAP",
#                                   bandname="smp",imagecollection="NASA_USDA/HSL/SMAP10KM_soil_moisture")
#     extract_data(zip_dir=outdir, out_dir=outdir)
# 
# #mosacing downloaded soil moisture data to Global data
# indir="E:\\NGA_Project_Data\\Soil_Moisture\\Soil_Moisture_SMAP_2015_2019\\GEE_WorldGrid_Step1" 
# outraster="E:\\NGA_Project_Data\\Soil_Moisture\\Soil_Moisture_SMAP_2015_2019\\Global_resampled_Step2\\SM_SMAP_2015_2019.tif"
# mosaic_rasters(input_dir=indir,output_raster=outraster,ref_raster=referenceraster2,
#                search_by="*.tif",no_data=NO_DATA_VALUE,resolution=0.05)
# =============================================================================


##Download TERRACLIMATE Temperature Data
# =============================================================================
# shape_path="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# shapes=glob(os.path.join(shape_path,"*world*.shp"))
# 
# ##2013_2019 TERRACLIMATE download
# download_dir_tmax="E:\\NGA_Project_Data\\Temperature_data\\Raw_Temp_Step01\\TMAX_2013_2019"
# download_dir_tmin="E:\\NGA_Project_Data\\Temperature_data\\Raw_Temp_Step01\\TMIN_2013_2019"
# mosaic_max="E:\\NGA_Project_Data\\Temperature_data\\World_Temp_Step02\\TMAX_2013_2019"
# mosaic_min="E:\\NGA_Project_Data\\Temperature_data\\World_Temp_Step02\\TMIN_2013_2019"
# 
# #TMAX
# for shape in shapes:
#     download_imagecollection_mean([2013,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir_tmax,
#                                  inputshp=shape, gee_scale=5000,dataname="Tmax_",name="Tmax",
#                                  bandname="tmmx",imagecollection="IDAHO_EPSCOR/TERRACLIMATE",factor=0.1)
# 
# 
#     extract_data(zip_dir=download_dir_tmax,
#                  out_dir=download_dir_tmax,
#                  searchby="*.zip",rename_file=True)
#     
# mosaic_rasters(input_dir=download_dir_tmax, output_dir=mosaic_max, raster_name="Tmax_2013_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# 
# #TMIN    
# for shape in shapes:
#     download_imagecollection_mean([2013,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir_tmin,
#                                  inputshp=shape, gee_scale=5000,dataname="Tmin_",name="Tmin",
#                                  bandname="tmmn",imagecollection="IDAHO_EPSCOR/TERRACLIMATE",factor=0.1)
# 
# 
#     extract_data(zip_dir=download_dir_tmin,
#                  out_dir=download_dir_tmin,
#                  searchby="*.zip",rename_file=True)
# 
# mosaic_rasters(input_dir=download_dir_tmin, output_dir=mosaic_min, raster_name="Tmin_2013_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)   
#     
# ##2018_2019 TERRACLIMATE download
# download_dir_tmax2="E:\\NGA_Project_Data\\Temperature_data\\Raw_Temp_Step01\\TMAX_2018_2019"
# download_dir_tmin2="E:\\NGA_Project_Data\\Temperature_data\\Raw_Temp_Step01\\TMIN_2018_2019"
# mosaic_max2="E:\\NGA_Project_Data\\Temperature_data\\World_Temp_Step02\\TMAX_2018_2019"
# mosaic_min2="E:\\NGA_Project_Data\\Temperature_data\\World_Temp_Step02\\TMIN_2018_2019"
# 
# #TMAX
# for shape in shapes:
#     download_imagecollection_mean([2018,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir_tmax2,
#                                  inputshp=shape, gee_scale=5000,dataname="Tmax_",name="Tmax",
#                                  bandname="tmmx",imagecollection="IDAHO_EPSCOR/TERRACLIMATE",factor=0.1)
# 
# 
#     extract_data(zip_dir=download_dir_tmax2,
#                  out_dir=download_dir_tmax2,
#                  searchby="*.zip",rename_file=True)
# 
# mosaic_rasters(input_dir=download_dir_tmax2, output_dir=mosaic_max2, raster_name="Tmax_2018_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# 
# #TMIN    
# for shape in shapes:
#     download_imagecollection_mean([2018,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir_tmin2,
#                                  inputshp=shape, gee_scale=5000,dataname="Tmin_",name="Tmin",
#                                  bandname="tmmn",imagecollection="IDAHO_EPSCOR/TERRACLIMATE",factor=0.1)
# 
# 
#     extract_data(zip_dir=download_dir_tmin2,
#                  out_dir=download_dir_tmin2,
#                  searchby="*.zip",rename_file=True) 
# 
# mosaic_rasters(input_dir=download_dir_tmin2, output_dir=mosaic_min2, raster_name="Tmin_2018_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# =============================================================================


##Download SRTM DEM Data
# =============================================================================
# csv=r"E:\NGA_Project_Data\shapefiles\GEE_Download_coords.csv"
# download_dir="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM\\Raw_DEM_Step01"
# mosaic_dir="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM\\World_DEM_Step02"
# 
# download_image_gee(output_dir=download_dir, shapecsv=csv,inputshp_dir=None,bandname='elevation',
#                        dataname="DEM_",name="DEM",image="USGS/SRTMGL1_003")
# 
# extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)
#     
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="SRTM_DEM_World.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# =============================================================================

##Download SRTM DEM Slope Data
# =============================================================================
# shape_path="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# shapes=glob(os.path.join(shape_path,"*world*.shp"))
# 
# download_dir="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM_Slope\\Raw_Slope_Step01"
# mosaic_dir="E:\\NGA_Project_Data\\DEM_Landform\\SRTM_DEM_SLope\\World_Slope_Step02"
# 
# 
# for shape in shapes:
#     download_image_gee(output_dir=download_dir, inputshp=shape,bandname='elevation',
#                        dataname="Slope_",name="Slope",image="USGS/SRTMGL1_003",Terrain_slope=True)
# 
# 
#     extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)
#     
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="SRTM_DEM_Slope.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# =============================================================================

##Download ALOS LandForm Data
# =============================================================================
# shape_path="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# shapes=glob(os.path.join(shape_path,"*world*.shp"))
# 
# download_dir="E:\\NGA_Project_Data\\DEM_Landform\\ALOS_Landform\\Raw_ALOS_LF_Step01"
# mosaic_dir="E:\\NGA_Project_Data\\DEM_Landform\\ALOS_Landform\\World_ALOS_LF_Step02"
# 
# 
# for shape in shapes:
#     download_image_gee(output_dir=download_dir, inputshp=shape,bandname='constant',dataname="LF_",name="LF",
#                        image="CSP/ERGo/1_0/Global/ALOS_landforms")
# 
# 
#     extract_data(zip_dir=download_dir,
#                  out_dir=download_dir,
#                  searchby="*.zip",rename_file=True)
#     
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="LF_ALOS_World.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# =============================================================================

##Download EVI MODIS Data (ALready cloudmasked in GEE)
# =============================================================================
# #EVI 2013_2019
# shape_path="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# shapes=glob(os.path.join(shape_path,"*world*.shp"))
# 
# download_dir="E:\\NGA_Project_Data\\Enhanced_Veg_Index\\Raw_EVI_Step01\\EVI_2013_2019"
# mosaic_dir="E:\\NGA_Project_Data\\Enhanced_Veg_Index\\World_EVI_Step02\\EVI_2013_2019"
# 
# for shape in shapes:
#     download_imagecollection_mean([2013,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir,
#                                  inputshp=shape, gee_scale=5000,dataname="EVI_",name="EVI",
#                                  bandname="EVI",imagecollection="MODIS/006/MOD13Q1",factor=0.0001)
# 
# 
#     extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)
# 
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="EVI_2013_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# 
# #EVI 2018_2019
# download_dir="E:\\NGA_Project_Data\\Enhanced_Veg_Index\\Raw_EVI_Step01\\EVI_2018_2019"
# mosaic_dir="E:\\NGA_Project_Data\\Enhanced_Veg_Index\\World_EVI_Step02\\EVI_2018_2019"
# 
# for shape in shapes:
#     download_imagecollection_mean([2018,2019],start_month=1,end_month=12,
#                                   output_dir=download_dir,
#                                  inputshp=shape, gee_scale=5000,dataname="EVI_",name="EVI",
#                                  bandname="EVI",imagecollection="MODIS/006/MOD13Q1",factor=0.0001)
# 
# 
#     extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)
# 
# mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="EVI_2018_2019.tif",
#                     ref_raster=referenceraster2,create_outdir=True)
# =============================================================================

##Download MODIS ET Data

#Modis ET 2013_2019
csv=r'..\Reference_rasters\GEE_Download_coords.csv'
download_dir=r'..\Raw_Data\ET_products\MODIS_ET\ET_2013_2019\Raw_ET_2013_2019'
mosaic_dir=r'..\Resampled Data\ET\MODIS_ET\2013_2019'
download_imagecollection_mean([2013,2019],start_month=1,end_month=12,output_dir=download_dir,shapecsv=csv,gee_scale=2000,
                              dataname='ET_',name='MODIS_ET',bandname='ET', imagecollection='MODIS/006/MOD16A2',
                              factor=0.1)
extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)

mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="MODIS_ET_2013_2019.tif",
                    ref_raster=referenceraster2)

#Modis ET 2018_2019
download_dir=r'..\Raw_Data\ET_products\MODIS_ET\ET_2018_2019\Raw_ET_2018_2019'
mosaic_dir=r'..\Resampled Data\ET\MODIS_ET\2018_2019'

download_imagecollection_mean([2018,2019],start_month=1,end_month=12,output_dir=download_dir,shapecsv=csv,gee_scale=2000,
                              dataname='ET_',name='MODIS_ET',bandname='ET', imagecollection='MODIS/006/MOD16A2',
                              factor=0.1)
extract_data(zip_dir=download_dir,out_dir=download_dir,searchby="*.zip",rename_file=True)

mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name="MODIS_ET_2018_2019.tif",
                    ref_raster=referenceraster2)
# =============================================================================

##Download GPWv411 UN Adjusted POpulation Density 
# =============================================================================
# shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee","*world*.shp"))
# #2010_2020 (This year range is only set up for the code, the data doesn't change based on year range)
# outdir="E:\\NGA_Project_Data\\population_density\\GPWv411 UN-Adjusted Population Density\\2010_2020\\Raw_GEE_data_step01"
# mosaic_dir="E:\\NGA_Project_Data\\population_density\\GPWv411 UN-Adjusted Population Density\\2010_2020\\World_pop_data_step02"
# for shape in shapes:
#     download_imagecollection_mean(yearlist=[2010,2020], start_month=1, end_month=1, output_dir=outdir, inputshp=shape,
#                                   dataname='pop_',name="GPW",gee_scale=5000,
#                                   imagecollection="CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density",
#                                   select_bandname=True,bandname="unwpp-adjusted_population_density")
#     extract_data(zip_dir=outdir, out_dir=outdir)
# mosaic_rasters(input_dir=outdir, output_dir=mosaic_dir, raster_name='Pop_density_GPW_2010_2020.tif')
# =============================================================================

##Download GPWv411 UN Adjusted POpulation Density
# csv=r'..\Reference_rasters\GEE_Download_coords.csv'
# #2010_2020 (This year range is only set up for the code, the data doesn't change based on year range)
# outdir=r'..\Raw_Data\population_density\GPWv411 UN-Adjusted Population Density\2010_2020\Raw_GEE_data_step01'
# mosaic_dir=r'..\Raw_Data\population_density\GPWv411 UN-Adjusted Population Density\2010_2020\World_pop_data_step02'
#
# download_imagecollection_mean(yearlist=[2010,2020], start_month=1, end_month=1, output_dir=outdir,shapecsv=csv,inputshp_dir=None,
#                                   dataname='pop_',name="GPW",gee_scale=2000,
#                                   imagecollection="CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density",
#                                   select_bandname=True,bandname="unwpp-adjusted_population_density")
# extract_data(zip_dir=outdir, out_dir=outdir)
# mosaic_rasters(input_dir=outdir, output_dir=mosaic_dir, raster_name='Pop_density_GPW_2010_2020.tif')

##Download Global Aridity Index
# =============================================================================
# shapes=glob(os.path.join("E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee","*world*.shp"))
# outdir="E:\\NGA_Project_Data\\Global_Aridity\\Raw_Aridity_GEE_step01"  
# mosaic_dir="E:\\NGA_Project_Data\\Global_Aridity\\Global_Aridity_step02"            
# for shape in shapes:
#     download_image_gee(output_dir=outdir,inputshp=shape,bandname='b1',factor=0.0001,gee_scale=5000,
#                        dataname="Aridity_",name="Aridity",image="projects/sat-io/open-datasets/global_ai_et0",
#                        Terrain_slope=False)
#     extract_data(zip_dir=outdir, out_dir=outdir)
# mosaic_rasters(input_dir=outdir, output_dir=mosaic_dir, raster_name="Global_Aridity_Index.tif")
# =============================================================================
