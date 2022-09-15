# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import ee
import pickle
import shutil
import zipfile
import requests
from Raster_operations import *
from PCA import *
from datetime import datetime
from Training_InSAR_processing import process_primary_insar_data, rasterize_coastal_subsidence

No_Data_Value = -9999

referenceraster = '../Data/Reference_rasters_shapes/Global_continents_ref_raster.tif'

csv = '../Data/Reference_rasters_shapes/GEE_Download_coords_modified.csv'
os.chdir('../Codes_Global_GW')


gee_data_list = ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_soil', 'TRCLM_RET', 'MODIS_ET', 'MODIS_EVI',
                 'MODIS_NDWI', 'MODIS_PET', 'GPW_pop', 'SRTM_DEM', 'Aridity_Index', 'Grace',
                 'clay_content_0cm', 'clay_content_10cm', 'clay_content_30cm', 'clay_content_60cm',
                 'clay_content_100cm', 'clay_content_200cm', 'MODIS_Land_Use']


# # Extract Data
def extract_data(zip_dir, out_dir, searchby="*.zip", rename_file=True):
    """
    Extract zipped data
    Parameters
    ----------
    zip_dir : File Location.
    out_dir : File Location where data will be extracted.
    searchby : Keyword for searching files, default is '*.zip'.
    rename_file : True if file rename is required while extracting.
    """
    print('Extracting zip files.....')
    makedirs([out_dir])
    for zip_file in glob(os.path.join(zip_dir, searchby)):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_file:
                zip_key = zip_file[zip_file.rfind(os.sep) + 1:zip_file.rfind(".")]
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_key + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)


# # ImageCollection Data Yearly Sum Download
def download_imagecollection_gee_yearly_sum(yearlist, start_month, end_month, output_dir, shapecsv,
                                            gee_scale=2000, dataname='MODIS_ET', factor=1,
                                            bandname="ET", imagecollection="MODIS/006/MOD16A2"):
    """
    Download Imagecollection GEE data (i.e. MODIS) by yearly sum basis for each download grid for each year.
    Extract and merge the data.

    Parameters:
    yearlist : List of years for which data will be downloaded, i.e., [2013,2019].
    start_month : Start month of data.
    end_month : End month of data.
    output_dir : File directory path to downloaded data.
    shapecsv : Csv of coordinates for download extent.
    gee_scale : Download Scale.
    bandname : Bandname of data to download.
    imagecollection : Imagecollection name.
    dataname : Name extension added to downloaded file.

    Returns: Downloaded data for each download grid. Extracted and merged data for each year.
    """
    # Initialize
    ee.Initialize()
    # Define Extent
    coords_df = pd.read_csv(shapecsv)
    # Date range
    for year in yearlist:
        start_date = ee.Date.fromYMD(year, start_month, 1)
        if end_month == 12:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)
        else:
            end_date = ee.Date.fromYMD(year, end_month + 1, 1)
        # for time series data only
        if start_month <= end_month:
            start_date = ee.Date.fromYMD(year - 1, start_month, 1)

        for index, row in coords_df.iterrows():
            minx = row['minx']
            miny = row['miny']
            maxx = row['maxx']
            maxy = row['maxy']
            gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

            # Get URL
            data_download = ee.ImageCollection(imagecollection).select(bandname).filterDate(start_date, end_date) \
                .sum().multiply(factor).toFloat()  # Change the function sum() based on purpose

            data_url = data_download.getDownloadURL({'name': dataname,
                                                     'crs': "EPSG:4326",
                                                     'scale': gee_scale,
                                                     'region': gee_extent})
            # downloading the data
            download_dir = makedirs([os.path.join(output_dir, str(year))])
            key_word = row['shape']
            local_file_name = os.path.join(download_dir, key_word + str(year) + '.zip')
            print('Downloading', local_file_name, '.....')
            r = requests.get(data_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)

            if index == coords_df.index[-1]:
                extract_data(zip_dir=download_dir, out_dir=download_dir, rename_file=True)
                mosaic_dir = makedirs([os.path.join(output_dir, 'merged_rasters')])
                mosaic_name = dataname + '_' + str(year) + '.tif'
                mosaic_rasters(input_dir=download_dir, output_dir=mosaic_dir, raster_name=mosaic_name,
                               ref_raster=referenceraster, search_by='*.tif', resolution=0.02, no_data=No_Data_Value)


def download_gee_data(yearlist, start_month, end_month, output_dir, dataname, shapecsv=csv,
                      gee_scale=2000, month_conversion=False, nodata=No_Data_Value):
    """
    Download Imagecollection/Image data from Google Earth Engine by range years' mean/median.

    Parameters:
    yearlist : List of years for which data will be downloaded, i.e., [2013,2019].
    start_month : Start month of data.
    end_month : End month of data.
    ***For ee.Image data time parameters should be included just for the code to run properly.
    The data will be the same for whole time period.

    output_dir : File directory path to downloaded data.
    shapecsv : Csv of coordinates for download extent. Defaults to csv (filepath of worldgrid coordinates' csv).
               Set to None if want to use shapefile instead.
    gee_scale : Download Scale.
    dataname : Dataname to download from GEE. The code can download data from the following list-
             ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_RET', 'TRCLM_soil,'MODIS_ET','MODIS_EVI',
             'MODIS_PET', 'GPW_pop','SRTM_DEM', 'Aridity_Index', 'MODIS_Land_Use', 'TRCLM_ET']
    month_conversion : Convert n-day composite data (MOD16 ET) to monthly data.
    nodata : No Data value. Defaults to -9999.

    Returns : Downloaded data from GEE.
    """
    # Initialize
    ee.Initialize()
    data_dict = {'TRCLM_precp': 'IDAHO_EPSCOR/TERRACLIMATE',  # monthly total
                 'TRCLM_tmmx': 'IDAHO_EPSCOR/TERRACLIMATE',  # monthly
                 'TRCLM_tmmn': 'IDAHO_EPSCOR/TERRACLIMATE',  # monthly
                 'TRCLM_soil': 'IDAHO_EPSCOR/TERRACLIMATE',  # monthly total
                 'TRCLM_RET': 'IDAHO_EPSCOR/TERRACLIMATE',  # monthly
                 'MODIS_ET': 'MODIS/006/MOD16A2',  # 8-day composite(sum of 8-day within each composite period)
                 'MODIS_PET': 'MODIS/006/MOD16A2',  # 8-day composite(sum of 8-day within each composite period)
                 'MODIS_EVI': 'MODIS/006/MOD13Q1',  # 16-day composite(composites of best pixels from 16 days)
                 'GPW_pop': 'CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density',  # pop density for modeled years
                 'SRTM_DEM': 'USGS/SRTMGL1_003',
                 'Aridity_Index': 'projects/sat-io/open-datasets/global_ai_et0',
                 'MODIS_Land_Use': 'MODIS/006/MCD12Q1',
                 'TRCLM_ET': 'IDAHO_EPSCOR/TERRACLIMATE'}

    if dataname in ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_RET', 'TRCLM_soil', 'MODIS_ET', 'MODIS_EVI',
                    'MODIS_PET', 'GPW_pop', 'MODIS_Land_Use', 'TRCLM_ET']:
        data_collection = ee.ImageCollection(data_dict[dataname])
    else:
        data_collection = ee.Image(data_dict[dataname])

    # Date Range Creation
    start_date = ee.Date.fromYMD(yearlist[0], start_month, 1)
    if end_month == 12:
        end_date = ee.Date.fromYMD(yearlist[1] + 1, 1, 1)
    else:
        end_date = ee.Date.fromYMD(yearlist[1], end_month + 1, 1)

    # Reducing the ImageCollection
    if dataname == 'TRCLM_precp':
        data = data_collection.select('pr').filterDate(start_date, end_date).mean().toFloat()

    elif dataname == 'TRCLM_tmmx' or dataname == 'TRCLM_tmmn':
        band = dataname.split('_')[1]
        data = data_collection.select(band).filterDate(start_date, end_date).median().multiply(0.1).toFloat()

    elif dataname == 'TRCLM_RET':
        data = data_collection.select('pet').filterDate(start_date, end_date).mean().multiply(0.1).toFloat()

    elif dataname == 'TRCLM_soil':
        band = dataname.split('_')[1]
        data = data_collection.select(band).filterDate(start_date, end_date).mean().multiply(0.1).toFloat()

    elif dataname == 'MODIS_ET':
        data = data_collection.select('ET').filterDate(start_date, end_date).sum().multiply(0.1).toFloat()

    elif dataname == 'MODIS_PET':
        data = data_collection.select('PET').filterDate(start_date, end_date).sum().multiply(0.1).toFloat()

    elif dataname == 'MODIS_EVI':
        data = data_collection.select('EVI').filterDate(start_date, end_date).mean().multiply(0.0001).toFloat()

    elif dataname == 'GPW_pop':
        data = data_collection.select('unwpp-adjusted_population_density').filterDate(start_date,
                                                                                      end_date).mean().toFloat()
    elif dataname == 'SRTM_DEM':
        data = data_collection.select('elevation').toFloat()

    elif dataname == 'Aridity_Index':
        data = data_collection.select('b1').multiply(0.0001).toFloat()

    elif dataname == 'MODIS_Land_Use':
        data = data_collection.select('LC_Type5').filterDate('2018-01-01', '2018-12-31').first().toFloat()

    elif dataname == 'TRCLM_ET':
        data = data_collection.select('aet').filterDate(start_date, end_date).mean().multiply(0.1).toFloat()

    makedirs([output_dir])
    coords_df = pd.read_csv(shapecsv)
    for index, row in coords_df.iterrows():
        # Define Extent
        minx = row['minx']
        miny = row['miny']
        maxx = row['maxx']
        maxy = row['maxy']
        gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

        if dataname == 'MODIS_EVI':
            gee_scale = 3000  # gee_scale 2000 can't download data for some grids, so select it >=3000 for MODIS EVI

        # Download URL
        data_url = data.getDownloadURL({'name': dataname,
                                        'crs': "EPSG:4326",
                                        'scale': gee_scale,
                                        'region': gee_extent})
        # dowloading the data
        key_word = row['shape']
        local_file_name = os.path.join(output_dir, key_word + str(yearlist[0]) + '_' + str(yearlist[1]) + '.zip')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

        if index == coords_df.index[-1]:
            extract_data(zip_dir=output_dir, out_dir=output_dir, rename_file=True)
            mosaic_dir = makedirs([os.path.join(output_dir, 'merged_rasters')])
            mosaic_name = dataname + '_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
            merged_arr, merged_raster = mosaic_rasters(input_dir=output_dir, output_dir=mosaic_dir,
                                                       raster_name=mosaic_name,
                                                       ref_raster=referenceraster, search_by='*.tif', resolution=0.02,
                                                       no_data=No_Data_Value)
            if month_conversion:
                start_date = datetime(yearlist[0], start_month, 1)
                end_date = datetime(yearlist[1], end_month,
                                    31)  # Set to 31 as end month for the project is December
                days_between = (end_date - start_date).days

                merged_arr[merged_arr == nodata] = np.nan
                monthly_arr = merged_arr * 30 / days_between
                monthly_arr[np.isnan(monthly_arr)] = nodata

                output_name = dataname + '_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '_monthly' + '.tif'
                ref_arr, ref_file = read_raster_arr_object(referenceraster)
                write_raster(raster_arr=monthly_arr, raster_file=ref_file, transform=ref_file.transform,
                             outfile_path=os.path.join(mosaic_dir, output_name), no_data_value=nodata)


# Download clay data for different layer from GEE
def download_clay_data(yearlist, output_dir, dataname, shapecsv=csv, gee_scale=2000, nodata=No_Data_Value):
    """
    Download Clay data from Google Earth Engine.

    Parameters:
    yearlist : List of years for which data will be downloaded, i.e., [2013,2019].
    start_month : Start month of data.
    end_month : End month of data.
    ***For ee.Image data time parameters should be included just for the code to run properly.
    The data will be the same for whole time period.

    output_dir : File directory path to downloaded data.
    shapecsv : Csv of coordinates for download extent. Defaults to csv (filepath of worldgrid coordinates' csv).
               Set to None if want to use shapefile instead.
    gee_scale : Download Scale.
    dataname : ['clay_content_0cm', 'clay_content_10cm', 'clay_content_30cm', 'clay_content_60cm', 'clay_content_100cm',
                'clay_content_200cm']
    nodata : No Data value. Defaults to -9999.

    Returns : Downloaded clay data from GEE.
    """

    ee.Initialize()
    data_dict = {'clay_content_0cm': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
                 'clay_content_10cm': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
                 'clay_content_30cm': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
                 'clay_content_60cm': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
                 'clay_content_100cm': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
                 'clay_content_200cm': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02'}

    data_collection = ee.Image(data_dict[dataname])

    if dataname == 'clay_content_0cm':
        data = data_collection.select('b0').toFloat()
    elif dataname == 'clay_content_10cm':
        data = data_collection.select('b10').toFloat()
    elif dataname == 'clay_content_30cm':
        data = data_collection.select('b30').toFloat()
    elif dataname == 'clay_content_60cm':
        data = data_collection.select('b60').toFloat()
    elif dataname == 'clay_content_100cm':
        data = data_collection.select('b100').toFloat()
    elif dataname == 'clay_content_200cm':
        data = data_collection.select('b200').toFloat()

    makedirs([output_dir])
    coords_df = pd.read_csv(shapecsv)
    for index, row in coords_df.iterrows():
        # Define Extent
        minx = row['minx']
        miny = row['miny']
        maxx = row['maxx']
        maxy = row['maxy']
        gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

        # Download URL
        data_url = data.getDownloadURL({'name': dataname,
                                        'crs': "EPSG:4326",
                                        'scale': gee_scale,
                                        'region': gee_extent})
        # dowloading the data
        key_word = row['shape']
        local_file_name = os.path.join(output_dir, key_word + str(yearlist[0]) + '_' + str(yearlist[1]) + '.zip')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

        if index == coords_df.index[-1]:
            extract_data(zip_dir=output_dir, out_dir=output_dir, rename_file=True)
            mosaic_dir = makedirs([os.path.join(output_dir, 'merged_rasters')])
            mosaic_name = dataname + '_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
            merged_arr, merged_raster = mosaic_rasters(input_dir=output_dir, output_dir=mosaic_dir,
                                                       raster_name=mosaic_name,
                                                       ref_raster=referenceraster, search_by='*.tif', resolution=0.02,
                                                       no_data=nodata)

            return merged_raster


# #Download GRACE ensemble data gradient over the year
def download_grace_gradient(yearlist, start_month, end_month, output_dir, shapecsv=csv, gee_scale=2000):
    """
    Download ensembled Grace data gradient from Google Earth Engine 
    ----------
    yearlist : List of years for which data will be downloaded, i.e., [2010,2017]
    start_month : Start month of data.
    end_month : End month of data.
    output_dir : File directory path to downloaded data.
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.

    Returns : Grace time gradient data.
    """
    # Initialize
    ee.Initialize()

    # Getting download url for ensembled grace data
    Grace = ee.ImageCollection("NASA/GRACE/MASS_GRIDS/LAND")

    # Date Range Creation
    start_date = ee.Date.fromYMD(yearlist[0], start_month, 1)

    if end_month == 12:
        end_date = ee.Date.fromYMD(yearlist[1] + 1, 1, 1)

    else:
        end_date = ee.Date.fromYMD(yearlist[1], end_month + 1, 1)

    # a function to sum timestamp on the image
    def addTime(image):
        """
        Scale milliseconds by a large constant to avoid very small slopes in the linear regression output.
        """
        return image.addBands(image.metadata('system:time_start').divide(1000 * 3600 * 24 * 365))

    # Reducing ImageColeection
    grace_csr = Grace.select("lwe_thickness_csr").filterDate(start_date, end_date).map(addTime)
    grace_csr_trend = grace_csr.select(['system:time_start', 'lwe_thickness_csr']) \
        .reduce(ee.Reducer.linearFit()).select('scale').toFloat()

    grace_gfz = Grace.select("lwe_thickness_gfz").filterDate(start_date, end_date).map(addTime)
    grace_gfz_trend = grace_gfz.select(['system:time_start', 'lwe_thickness_gfz']) \
        .reduce(ee.Reducer.linearFit()).select('scale').toFloat()

    grace_jpl = Grace.select("lwe_thickness_jpl").filterDate(start_date, end_date).map(addTime)
    grace_jpl_trend = grace_jpl.select(['system:time_start', 'lwe_thickness_jpl']) \
        .reduce(ee.Reducer.linearFit()).select('scale').toFloat()

    # Ensembling
    grace_ensemble_avg = grace_csr_trend.select(0).add(grace_gfz_trend.select(0)).select(0) \
        .add(grace_jpl_trend.select(0)).select(0).divide(3)

    makedirs([output_dir])
    coords_df = pd.read_csv(shapecsv)
    for index, row in coords_df.iterrows():
        # Define Extent
        minx = row['minx']
        miny = row['miny']
        maxx = row['maxx']
        maxy = row['maxy']
        gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

        download_url = grace_ensemble_avg.getDownloadURL({'name': 'Grace',
                                                          'crs': "EPSG:4326",
                                                          'scale': gee_scale,
                                                          'region': gee_extent})
        # dowloading the data
        key_word = row['shape'] + 'Grace_'
        local_file_name = os.path.join(output_dir, key_word + str(yearlist[0]) + '_' + str(yearlist[1]) + '.zip')
        print('Downloading', local_file_name, '.....')
        r = requests.get(download_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

        if index == coords_df.index[-1]:
            extract_data(zip_dir=output_dir, out_dir=output_dir, rename_file=True)
            mosaic_dir = makedirs([os.path.join(output_dir, 'merged_rasters')])
            mosaic_name = 'Grace' + '_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
            mosaic_rasters(input_dir=output_dir, output_dir=mosaic_dir, raster_name=mosaic_name,
                           ref_raster=referenceraster, search_by='*.tif', resolution=0.02,
                           no_data=No_Data_Value)


# #Stationary Single Image Download
def download_image_gee(output_dir, bandname, shapecsv=None, factor=1, gee_scale=2000,
                       dataname="DEM_", image="USGS/SRTMGL1_003", terrain_slope=False):
    """
    Download Stationary/Single Image from Google Earth Engine.
    Parameters
    ----------
    output_dir : File directory path to downloaded data.
    bandname : Bandname of the data.
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.
    factor : Factor to multiply with (if there is scale mentioned in the data) to convert to original scale.
    gee_scale : Pixel size in m. Defaults to 2000.
    dataname : Name extension added to downloaded file.
    image : Image name from Google Earth Engine.
    terrain_slope : If the image is a SRTM DEM data and slope data download is needed. Defaults to False.
    """

    # Initialize
    ee.Initialize()

    data_download = ee.Image(image).select(bandname).multiply(factor).toFloat()

    if terrain_slope:
        data_download = ee.Terrain.slope(data_download)

    makedirs([output_dir])
    coords_df = pd.read_csv(shapecsv)
    for index, row in coords_df.iterrows():
        # Define Extent
        minx = row['minx']
        miny = row['miny']
        maxx = row['maxx']
        maxy = row['maxy']
        gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

        download_url = data_download.getDownloadURL({'name': dataname,
                                                     'crs': "EPSG:4326",
                                                     'scale': gee_scale,
                                                     'region': gee_extent})
        # dowloading the data
        key_word = row['shape']
        local_file_name = os.path.join(output_dir, key_word + '.zip')
        print('Downloading', local_file_name, '.....')
        r = requests.get(download_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)


# # MODIS Cloudmask
def cloudmask_MODIS09A1(image):
    """
    Removing cloudmask from MODIS (MOD09A1.006 Terra Surface Reflectance 8-Day Global 500m) data.

    param : {ee.Image} image input MODIS09A1 SR image
    return : {ee.Image} cloudmasked MODIS09A1 image
    """
    # Bits 0, 1 are cloud state and Bits 2 is cloud shadow.
    cloudMask0 = (1 << 0)
    cloudMask1 = (1 << 1)
    cloudShadow = (1 << 2)
    # Get the StateQA band.
    qa = image.select('StateQA')
    # Both flags should be set to zero, indicating clear conditions.
    Mask = qa.bitwiseAnd(cloudShadow).eq(0) \
        .And(qa.bitwiseAnd(cloudMask0).eq(0)) \
        .And(qa.bitwiseAnd(cloudMask1).eq(0))
    return image.updateMask(Mask)


# #Download MODIS 09A1  datawith cloudmaking
def download_modis_derived_product(yearlist, start_month, end_month, output_dir, shapecsv=csv, gee_scale=2000,
                                   dataname='MODIS', name='MODIS',
                                   imagecollection='MODIS/006/MOD09A1', factor=0.0001, index_name='NDWI'):
    """
    Download Imagecollection mean data of Landsat products (NDVI,NDWI), with cloudmask applied,
    from Google Earth Engine for range of years. 
    
    ***The code works best for small spatial and temporal scale data (Otherwise takes a lot of time).***

    Parameters:
    yearlist : list: Year for which data will be downloaded, i.e., [2010,2020]
    start_month : Integer: Start month of data
    end_month : Integer: End month of data
    output_dir : File directory path: Location to downloaded data 
    shapecsv : Csv of coordinates for download extent. Set to None if want to use shapefile instead.
    gee_scale : Download Scale
    bandname : Band to download from Google earth engine
    imagecollection : Imagecollection name
    dataname : Name extension added to downloaded file
    name : Name added to file when downloading
    factor : Factor (if needed) to multiply with the band
    index_name : Index to download ('NDVI'/'NDWI'). Defaults to 'NDWI'
    """
    # Initialize
    ee.Initialize()

    # Date Range Creation
    start_date = ee.Date.fromYMD(yearlist[0], start_month, 1)

    if end_month == 12:
        end_date = ee.Date.fromYMD(yearlist[1] + 1, 1, 1)

    else:
        end_date = ee.Date.fromYMD(yearlist[1], end_month + 1, 1)

    # Creating Output Directory
    makedirs([output_dir])

    dataset = ee.ImageCollection(imagecollection)
    cloudmasked = dataset.filterDate(start_date, end_date).map(cloudmask_MODIS09A1)

    # Get URL
    if index_name == 'NDVI':
        NIR = cloudmasked.select('sur_refl_b02').mean().multiply(factor).toFloat()
        Red = cloudmasked.select('sur_refl_b01').mean().multiply(factor).toFloat()
        data_download = NIR.subtract(Red).divide(NIR.add(Red))

    if index_name == 'NDWI':
        NIR = cloudmasked.select('sur_refl_b02').mean().multiply(factor).toFloat()
        SWIR = cloudmasked.select('sur_refl_b06').mean().multiply(factor).toFloat()
        data_download = NIR.subtract(SWIR).divide(NIR.add(SWIR))

    coords_df = pd.read_csv(shapecsv)
    for index, row in coords_df.iterrows():
        # Define Extent
        minx = row['minx']
        miny = row['miny']
        maxx = row['maxx']
        maxy = row['maxy']
        gee_extent = ee.Geometry.Rectangle((minx, miny, maxx, maxy))

        # Download URL
        data_url = data_download.getDownloadURL({'name': name,
                                                 'crs': "EPSG:4326",
                                                 'scale': gee_scale,
                                                 'region': gee_extent})
        # dowloading the data
        key_word = row['shape']
        local_file_name = os.path.join(output_dir, key_word + str(yearlist[0]) + '_' + str(yearlist[1]) + '.zip')
        print('Downloading', local_file_name, '.....')
        r = requests.get(data_url, allow_redirects=True)
        open(local_file_name, 'wb').write(r.content)

        if index == coords_df.index[-1]:
            extract_data(zip_dir=output_dir, out_dir=output_dir, rename_file=True)
            mosaic_dir = makedirs([os.path.join(output_dir, 'merged_rasters')])
            mosaic_name = dataname + '_' + index_name + '_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
            mosaic_rasters(input_dir=output_dir, output_dir=mosaic_dir, raster_name=mosaic_name,
                           ref_raster=referenceraster, search_by='*.tif', resolution=0.02,
                           no_data=No_Data_Value)


# #Download data from URL
def download_from_url(out_dir, url_list):
    """
    Download Data from url
    Parameters:
    out_dir : Filepath to download.
    url_list : A list of url/urls to download.

    Returns: Data downloaded from url.
    """
    makedirs([out_dir])
    for url in url_list:
        fname = url.rsplit('/', 1)[1]
        out_fname = os.path.join(out_dir, fname)
        print("Downloading", fname, "......")
        r = requests.get(url, allow_redirects=True)
        open(out_fname, 'wb').write(r.content)


def download_data(data_list, yearlist, start_month, end_month, shape_csv=csv, gee_scale=2000, skip_download=True):
    """
    Download data from GEE. The code can download data from the following list.
    ['TRCLM_precp','TRCLM_tmmx','TRCLM_tmmn','TRCLM_soil', 'TRCLM_RET', 'MODIS_ET','MODIS_EVI', 'MODIS_NDWI',
    'MODIS_PET', 'GPW_pop', 'SRTM_DEM', 'Aridity_Index','Grace', 'clay_content_0cm', 'clay_content_10cm',
     'clay_content_30cm', 'clay_content_60cm', 'clay_content_100cm', 'clay_content_200cm', 'MODIS_Land_Use',
     'TRCLM_ET']

    Parameters:
    data_list : List of data to download.
    yearlist : List of years for data.
    start_month : Start month of data.
    end_month : End month of data.
    shape_csv : Csv file path containing download grid coordinates. Set None to use inputshp_dir.
    gee_scale : Scale in meter for data download. Defaults to 2000 m.
    skip_download : Set False to download the data.

    Returns : Downloaded data.
    """
    year_string = str(yearlist[0]) + '_' + str(yearlist[1])
    download_dir = '../Data/Raw_Data/GEE_data'
    downdir_NDWI = os.path.join(download_dir, 'MODIS_NDWI', year_string)
    downdir_EVI = os.path.join(download_dir, 'MODIS_EVI', year_string)
    downdir_Grace = os.path.join(download_dir, 'Grace', year_string)
    downdir_TRCLM_precp = os.path.join(download_dir, 'Precipitation', 'Terraclimate', year_string)
    downdir_TRCLM_soil = os.path.join(download_dir, 'Soil_moisture', 'Terraclimate', year_string)
    downdir_TRCLM_Tmin = os.path.join(download_dir, 'Tmin', 'Terraclimate', year_string)
    downdir_TRCLM_Tmax = os.path.join(download_dir, 'Tmax', 'Terraclimate', year_string)
    downdir_TRCLM_RET = os.path.join(download_dir, 'TRCLM_RET', year_string)
    downdir_PopDensity_GPW = os.path.join(download_dir, 'Population_density', 'GPWv411', year_string)
    downdir_MODIS_ET = os.path.join(download_dir, 'MODIS_ET', year_string)
    downdir_MODIS_PET = os.path.join(download_dir, 'MODIS_PET', year_string)
    downdir_SRTM_DEM = os.path.join(download_dir, 'SRTM_DEM')
    downdir_AI = os.path.join(download_dir, 'Aridity_Index')
    downdir_clay_content = os.path.join(download_dir, 'Clay_content_openlandmap')
    downdir_MODIS_land_use = os.path.join(download_dir, 'MODIS_Land_Use')
    downdir_TRCLM_ET = os.path.join(download_dir, 'TRCLM_ET', year_string)

    makedirs([download_dir, downdir_NDWI, downdir_EVI, downdir_Grace, downdir_TRCLM_precp, downdir_TRCLM_soil,
              downdir_TRCLM_Tmin, downdir_TRCLM_Tmax, downdir_TRCLM_RET, downdir_PopDensity_GPW, downdir_MODIS_ET,
              downdir_MODIS_PET, downdir_SRTM_DEM, downdir_AI, downdir_clay_content,
              downdir_MODIS_land_use, downdir_TRCLM_ET])

    if not skip_download:
        for data in data_list:
            if data == 'MODIS_NDWI':
                download_modis_derived_product(yearlist, start_month, end_month, downdir_NDWI, shapecsv=csv,
                                               gee_scale=3500, imagecollection='MODIS/006/MOD09A1', factor=0.0001,
                                               index_name='NDWI')
                # Scale set to 3500 due to having download error at lower scale
            elif data == 'MODIS_EVI':
                download_gee_data(yearlist, start_month, end_month, downdir_EVI, 'MODIS_EVI', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'Grace':
                download_grace_gradient(yearlist, start_month, end_month, downdir_Grace, shape_csv,
                                        gee_scale=gee_scale)
            elif data == 'TRCLM_precp':
                download_gee_data(yearlist, start_month, end_month, downdir_TRCLM_precp, 'TRCLM_precp', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'TRCLM_soil':
                download_gee_data(yearlist, start_month, end_month, downdir_TRCLM_soil, 'TRCLM_soil', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'TRCLM_tmmn':
                download_gee_data(yearlist, start_month, end_month, downdir_TRCLM_Tmin, 'TRCLM_tmmn', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'TRCLM_tmmx':
                download_gee_data(yearlist, start_month, end_month, downdir_TRCLM_Tmax, 'TRCLM_tmmx', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'TRCLM_RET':
                download_gee_data(yearlist, start_month, end_month, downdir_TRCLM_RET, 'TRCLM_RET', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'GPW_pop':
                download_gee_data(yearlist, start_month, end_month, downdir_PopDensity_GPW, 'GPW_pop', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'MODIS_ET':
                download_gee_data(yearlist, start_month, end_month, downdir_MODIS_ET, 'MODIS_ET', shape_csv,
                                  gee_scale=gee_scale, month_conversion=True)
            elif data == 'MODIS_PET':
                download_gee_data(yearlist, start_month, end_month, downdir_MODIS_PET, 'MODIS_PET', shape_csv,
                                  gee_scale=gee_scale, month_conversion=True)
            elif data == 'SRTM_DEM':
                download_gee_data(yearlist, start_month, end_month, downdir_SRTM_DEM, 'SRTM_DEM', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'Aridity_Index':
                download_gee_data(yearlist, start_month, end_month, downdir_AI, 'Aridity_Index', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'clay_content_0cm':
                download_clay_data(yearlist, output_dir=os.path.join(downdir_clay_content, 'claycontent_0cm'),
                                   dataname='clay_content_0cm', shapecsv=csv, gee_scale=2000)
            elif data == 'clay_content_10cm':
                download_clay_data(yearlist, output_dir=os.path.join(downdir_clay_content, 'claycontent_10cm'),
                                   dataname='clay_content_10cm', shapecsv=csv, gee_scale=2000)
            elif data == 'clay_content_30cm':
                download_clay_data(yearlist, output_dir=os.path.join(downdir_clay_content, 'claycontent_30cm'),
                                   dataname='clay_content_30cm', shapecsv=csv, gee_scale=2000)
            elif data == 'clay_content_60cm':
                download_clay_data(yearlist, output_dir=os.path.join(downdir_clay_content, 'claycontent_60cm'),
                                   dataname='clay_content_60cm', shapecsv=csv, gee_scale=2000)
            elif data == 'clay_content_1000cm':
                download_clay_data(yearlist, output_dir=os.path.join(downdir_clay_content, 'claycontent_100cm'),
                                   dataname='clay_content_100cm', shapecsv=csv, gee_scale=2000)
            elif data == 'clay_content_200cm':
                download_clay_data(yearlist, output_dir=os.path.join(downdir_clay_content, 'claycontent_200cm'),
                                   dataname='clay_content_200cm', shapecsv=csv, gee_scale=2000)
            elif data == 'MODIS_Land_Use':
                download_gee_data(yearlist, start_month, end_month, downdir_MODIS_land_use, 'MODIS_Land_Use', shape_csv,
                                  gee_scale=gee_scale)
            elif data == 'TRCLM_ET':
                download_gee_data(yearlist, start_month, end_month, downdir_TRCLM_ET, 'TRCLM_ET', shape_csv,
                                  gee_scale=gee_scale)

    return download_dir, downdir_NDWI, downdir_EVI, downdir_Grace, downdir_TRCLM_precp, downdir_TRCLM_soil, \
           downdir_TRCLM_Tmin, downdir_TRCLM_Tmax, downdir_TRCLM_RET, downdir_PopDensity_GPW, downdir_MODIS_ET, \
           downdir_MODIS_PET, downdir_SRTM_DEM, downdir_AI, downdir_clay_content, \
           downdir_MODIS_land_use, downdir_TRCLM_ET


def prepare_lu_data(gfsad_lu='../Data/Raw_Data/Land_Use_Data/Raw/Global Food Security- GFSAD1KCM/GFSAD1KCM.tif',
                    giam_gw='../Data/Raw_Data/Land_Use_Data/Raw/gmlulca_10classes_global/'
                            'gmlulca_10classes_global.tif',
                    irrigated_meier='../Data/Raw_Data/Land_Use_Data/Raw/global_irrigated_areas/'
                                    'global_irrigated_areas.tif',
                    intermediate_dir='../Data/Intermediate_working_dir',
                    output_dir='../Data/Resampled_Data/Land_Use', skip_processing=True):
    """
    Preparing Land Use Datasets. Works on GFSAD, Irrigated_Meier and GIAM_GW datasets.

    Parameters:
    gfsad_cropextent : GFSAD1KCM Raw dataset path.
              (Data downloaded from https://lpdaac.usgs.gov/products/gfsad1kcmv001/)
    giam_gw : giam_gw Raw dataset path.
              (Data downloaded fromhttp://waterdata.iwmi.org/Applications/GIAM2000/)
    irrigated_meier : Irrigated area dataset path from Meier et al. 2018.
            (Data downloaded from https://doi.pangaea.de/10.1594/PANGAEA.884744)
    intermediate_dir : File path of intermediate directory for processing data.
    output_dir : File path of final resampled data directory.
    skip_processing : Set False to process the rasters. Defaults to True (Raster filepath taken from existing rasters)

    Returns : Processed (resampled and gaussian filtered) GFSAD1KCM and GIAM_GW land use data.
    """

    if not skip_processing:
        makedirs([intermediate_dir, output_dir])
        if gfsad_lu:
            print('Processing GFSAD1KCM Dataset...')
            masked_raster = mask_by_ref_raster(input_raster=gfsad_lu, outdir=intermediate_dir,
                                               raster_name='Global_GFSAD1KCM_raw.tif', ref_raster=referenceraster,
                                               resolution=0.02)
            filtered_raster = filter_specific_values(input_raster=masked_raster, outdir=intermediate_dir,
                                                     raster_name='GFSAD_irrig_only.tif', filter_value=[1],
                                                     new_value=True, value_new=1, paste_on_ref_raster=True,
                                                     ref_raster=referenceraster)
            gfsad_raster = apply_gaussian_filter(input_raster=filtered_raster, outdir=output_dir,
                                                 raster_name='Irrigated_Area_Density.tif', ignore_nan=True,
                                                 normalize=True)
            print('Processed GFSAD1KCM Dataset...')

        if irrigated_meier:
            print('Processing Irrigated Area Meier Dataset...')
            masked_raster = mask_by_ref_raster(input_raster=irrigated_meier, outdir=intermediate_dir,
                                               raster_name='Global_Irrigated_Meier.tif')
            filtered_raster = filter_specific_values(input_raster=masked_raster, outdir=intermediate_dir,
                                                     raster_name='irrigation_meirer.tif', filter_value=[1, 2, 3, 4],
                                                     new_value=True, value_new=1, paste_on_ref_raster=True,
                                                     ref_raster=referenceraster)
            irrigated_meier_raster = apply_gaussian_filter(input_raster=filtered_raster, outdir=output_dir,
                                                           raster_name='Irrigated_Area_Density_meier.tif',
                                                           ignore_nan=True, normalize=True)
            print('Processed Irrigated Area Meier Dataset')

        if giam_gw:
            print('Processing GIAM_GW Dataset...')
            masked_raster = mask_by_ref_raster(input_raster=giam_gw, outdir=intermediate_dir,
                                               raster_name='Global_GIAM_Irrigation.tif')

            filtered_raster = filter_specific_values(input_raster=masked_raster, outdir=intermediate_dir,
                                                     raster_name='GIAM_GW_irrigation_only.tif', filter_value=[2],
                                                     new_value=True, value_new=1, paste_on_ref_raster=True,
                                                     ref_raster=referenceraster)

            giam_gw_raster = apply_gaussian_filter(input_raster=filtered_raster, outdir=output_dir,
                                                   raster_name='GW_Irrigation_Density_giam.tif', ignore_nan=True,
                                                   normalize=True)
            print('Processed GIAM_GW Dataset')

    else:
        gfsad_raster = '../Data/Resampled_Data/Land_Use/Irrigated_Area_Density.tif'
        irrigated_meier_raster = '../Data/Resampled_Data/Land_Use/Irrigated_Area_Density_meier.tif'
        giam_gw_raster = '../Data/Resampled_Data/Land_Use/GW_Irrigation_Density_giam.tif'

    return gfsad_raster, irrigated_meier_raster, giam_gw_raster


def prepare_popdensity_data(
        pop_dataset=r'../Data/Raw_Data/GEE_data/Population_density/GPWv411/2013_2019/merged_rasters\\\n'
                    r'GPW_pop_2013_2019.tif',
        output_dir=r'../Data/Resampled_Data/Pop_Density', skip_processing=True):
    """
    Preparing Population Density Datasets.

    Parameters:
    pop_dataset : Raw population dataset path.
    output_dir : Filepath of final resampled data directory.
    skip_processing : Set False to process the raster. Defaults to True (Raster filepath taken from existing raster)

    Returns : Processed (resampled and gaussian filtered) population density raster.
    """
    if not skip_processing:
        makedirs([output_dir])
        print('Processing Population Density Dataset...')
        sep = pop_dataset.rfind(os.sep)
        dot = pop_dataset.rfind('.')
        split = pop_dataset[sep + 1:dot].split('_')
        rastername = 'Pop_Density_' + split[-2] + '_' + split[-1] + '.tif'
        popdensity_raster = apply_gaussian_filter(input_raster=pop_dataset, outdir=output_dir,
                                                  raster_name=rastername, ignore_nan=True, normalize=True)
        print('Created Population Density Raster')
    else:
        popdensity_raster = r'../Data/Resampled_Data/Pop_Density/Pop_Density_2013_2019.tif'

    return popdensity_raster


def prepare_sediment_thickness_data(input_raster='../Data/Raw_Data/Global_Sediment_Thickness'
                                                 '/average_soil_and_sedimentary-deposit_thickness.tif',
                                    interim_dir='../Data/Intermediate_working_dir',
                                    output_dir='../Data/Resampled_Data/Sediment_Thickness',
                                    raster_name='Global_Sediment_Thickness.tif',
                                    skip_processing=False):
    """
    Processing Global Sediment thickness dataset.

    Parameters:
    input_raster : Input raster filepath with filename.
    output_dir : Output raster directory path.
    interim_dir : Intermediate working directory for storing interdim data.
    output_raster_name : Output raster name.
    skip_processing : Set to True if want to skip processing.

    Returns : Resampled sediment thickness raster.
    """
    makedirs([interim_dir, output_dir])
    if not skip_processing:
        print('Processing Sediment Thickness Dataset...')
        resampled_raster_path = resample_reproject(input_raster, interim_dir, raster_name,
                                                   reference_raster=referenceraster, resample=True)
        raster_arr, raster_file = read_raster_arr_object(resampled_raster_path)
        raster_arr[raster_arr > 50] = raster_file.nodata
        output_raster = output_dir + '/' + raster_name
        sediment_raster = write_raster(raster_arr, raster_file, raster_file.transform,
                                       output_raster)
        print('Processed Sediment Thickness Dataset')
    else:
        sediment_raster = '../Data/Resampled_Data/Sediment_Thickness/Global_Sediment_Thickness.tif'

    return sediment_raster


def prepare_clay_thickness_data(clay_raster='../Data/Resampled_Data/GEE_data_2013_2019/'
                                            'clay_content_200cm_2013_2019.tif',
                                sediment_thickness_raster='../Data/Resampled_Data/Sediment_Thickness/'
                                                          'Global_Sediment_Thickness.tif',
                                output_dir='../Data/Resampled_Data/Clay_Thickness', skip_processing=True):
    """
    Prepare clay thickness data.

    Parameters:
    clay_raster : Filepath of clay thickness raster.
    sediment_thickness_raster : Filepath of sediment thickness raster.
    output_dir : Output raster directory path.
    skip_processing : Set to True if want to skip processing.

    Returns : Clay thickness raster dataset.
    """
    if not skip_processing:
        print('Processing Clay Thickness Dataset...')
        clay_thickness_raster = array_multiply(clay_raster, sediment_thickness_raster, outdir=output_dir,
                                               raster_name='Clay_Thickness.tif', scale=0.01)
        print('Processed Clay Thickness Dataset')
    else:
        clay_thickness_raster = '../Data/Resampled_Data/Sediment_Thickness/Clay_Thickness.tif'

    return clay_thickness_raster


def prepare_river_proximity_data(input_shape='../Data/Raw_Data/Surface_Water/mrb_shp/mrb_rivers.shp',
                                 output_dir='../Data/Resampled_Data/Surface_Water',
                                 ref_raster=referenceraster, skip_processing=True):
    """
    Prepare river proximity (distance) datasets.

    Parameters:
    input_shape : Input river shapefile filepath.
    output_dir : Output raster directory path.
    ref_raster : Reference raster. Default to to referenceraster.
    skip_processing : Set to True if want to skip processing.

    Returns : River distance raster dataset.
    """

    if not skip_processing:
        print('Processing River Dataset')
        makedirs([output_dir])
        river_raster = shapefile_to_raster(input_shape, output_dir, 'River_raster.tif', use_attr=False, burnvalue=1,
                                           resolution=0.02)
        river_arr = read_raster_arr_object(river_raster, get_file=False).flatten()
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_arr = ref_arr.flatten()
        modified_river_arr = np.where(river_arr == 1, river_arr, ref_arr)
        modified_river_arr = modified_river_arr.reshape(ref_file.shape)

        final_river_raster = output_dir + '/River_raster_wgs4326.tif'
        river_raster_wgs4326 = write_raster(modified_river_arr, ref_file, ref_file.transform, final_river_raster)

        river_raster_projected = resample_reproject(river_raster_wgs4326, output_dir, 'River_raster_projected.tif',
                                                    reference_raster=referenceraster, resample=False, reproject=True,
                                                    change_crs_to="EPSG:4087", both=False, resample_algorithm='near',
                                                    nodata=No_Data_Value)

        river_distance_projected = compute_proximity(river_raster_projected, output_dir, 'River_distance_projected.tif',
                                                     target_values=(1,))

        river_distance_wgs4326 = resample_reproject(river_distance_projected, output_dir, 'River_distance_wgs4326.tif',
                                                    reference_raster=referenceraster, resample=False, reproject=True,
                                                    change_crs_to="EPSG:4326", both=False, resample_algorithm='near',
                                                    nodata=No_Data_Value)
        river_distance_prefinal = paste_val_on_ref_raster(river_distance_wgs4326, output_dir,
                                                          'River_distance_prefinal.tif', value=0,
                                                          ref_raster=referenceraster)

        # converting distance values in pixel to km
        river_distance_arr = read_raster_arr_object(river_distance_prefinal, get_file=False)
        river_distance_arr = river_distance_arr / 1000

        final_river_distance = '../Data/Resampled_Data/Surface_Water/River_distance.tif'
        river_distance = write_raster(river_distance_arr, ref_file, ref_file.transform, final_river_distance)

        print('Processed River Dataset')
    else:
        river_distance = '../Data/Resampled_Data/Surface_Water/River_distance.tif'

    return river_distance


def process_global_confining_layer_data(input_raster=
                                        '../Data/Raw_Data/Global_confining_layer/global_confining_layer.tif',
                                        output_dir='../Data/Resampled_Data/Global_confining_layers',
                                        ref_raster=referenceraster):
    """
    Process global confining layer dataset (made by Dr. Ryan Smith).

    Parameters:
    input_raster : Input raster filepath.
    output_dir : Filepath of output directory to save processed raster.
    ref_raster : Reference raster filepath.

    Returns: Confining layers dataset.
    """
    makedirs([output_dir])
    masked_raster = mask_by_ref_raster(input_raster, '../Data/Raw_Data/Global_confining_layer',
                                       'confining_layers_masked.tif', ref_raster=ref_raster)

    confining_layer = paste_val_on_ref_raster(masked_raster, output_dir, 'Confining_layers.tif', value=0)

    return confining_layer


def prepare_modis_landuse_data(output_raster,
                               input_raster='../Data/Raw_Data/GEE_data/MODIS_Land_Use/merged_rasters'
                                            '/MODIS_Land_Use_2013_2019.tif'):
    """
    Reclassifying and processing MODIS Land Use Data.

    Reclassified Classes:
    1 - Forest
    2 - Vegetation
    3 - Cropland
    4 - Urban and built-Up
    5 - Snow and Ice
    6 - Barren land
    7 - Water body

    Parameters:
    output_raster: Output raster filepath.
    input_raster: Input raster filepath.

    Returns: Reclassified MODIS Land Use raster.
    """
    lu_arr, lu_file = read_raster_arr_object(input_raster)

    # Reclassifying classes
    lu_arr = np.where((lu_arr == 2) | (lu_arr == 3) | (lu_arr == 4), 1, lu_arr)
    lu_arr = np.where((lu_arr == 5) | (lu_arr == 6), 2, lu_arr)
    lu_arr = np.where((lu_arr == 7) | (lu_arr == 8), 3, lu_arr)
    lu_arr = np.where(lu_arr == 9, 4, lu_arr)
    lu_arr = np.where(lu_arr == 10, 5, lu_arr)
    lu_arr = np.where(lu_arr == 11, 6, lu_arr)
    lu_arr = np.where(lu_arr == 0, 7, lu_arr)

    write_raster(lu_arr, lu_file, lu_file.transform, output_raster, ref_file=referenceraster)

    return output_raster


def download_process_predictor_datasets(yearlist, start_month, end_month, resampled_gee_dir,
                                        gfsad_cropextent, giam_gw, irrigated_meier, intermediate_dir, outdir_lu,
                                        sediment_thickness, outdir_sed_thickness, outdir_pop, river_shape, outdir_sw,
                                        confining_layer, outdir_confining_layer,
                                        perform_pca=False, skip_download=True, skip_processing=True,
                                        geedatalist=gee_data_list, downloadcsv=csv, gee_scale=2000):
    """
    Download and process (resample) GEE data and other datasets (Land Use, Population, Sediment thickness).

    Parameters:
    yearlist : Year list to use for downloading GEE data.
    start_month : Start month of GEE data.
    end_month : End month of GEE data.
    resampled_gee_dir : Resampled directory for GEE data.
    gfsad_cropextent : Raw GFSAD1KCM crop extent data.
    giam_gw : Unsampled/Raw GIAM land use data.
    irrigated_meier : Irrigated area data from meier et al. 2018.
    intermediate_dir : Intermediate file directory to save intermediate files
    outdir_lu : Output directory for saving processed land use rasters.
    sediment thickness : Unsampled/Raw sediment thickness data.
    outdir_sed_thickness : Output directory for saving processed sediment thickness raster.
    outdir_pop : Output directory for saving processed population raster.
    input_shape : Input river shapefile filepath.
    oudir_sw : Output directory for saving processed river raster.
    confining_layer : Input confining layers dataset.
    outdir_confining_layer : Output directory for saving processed confining layer raster.
    perform_pca : Set to True if to  run PCA.
    skip_download : Set to False if want to download data from GEE. Default set to True.
    skip_processing : Set to False if want to process datasets.
    geedatalist : Data list to download from GEE. Can download data from the following list-
                  ['TRCLM_precp', 'TRCLM_tmmx', 'TRCLM_tmmn', 'TRCLM_ret', 'TRCLM_soil', 'MODIS_ET', 'MODIS_PET',
                  'MODIS_EVI', 'MODIS_NDWI', 'SRTM_DEM', 'Aridity_Index', 'Grace',
                  'clay_content_0cm', 'clay_content_10cm', 'clay_content_30cm', 'clay_content_60cm',
                  'clay_content_100cm', 'clay_content_200cm', 'MODIS_Land_Use', 'TRCLM_ET']
    downloadcsv : Csv (with coordinates) filepath used in downloading data from GEE.
    gee_scale : scale to use in downloading data from GEE in meter. Default set to 2000m.

    Returns : Filepath of processed gee datasets along with land use, population density rasters.
    """

    download_dir, downdir_NDWI, downdir_EVI, downdir_Grace, downdir_TRCLM_precp, downdir_TRCLM_soil, \
    downdir_TRCLM_Tmin, downdir_TRCLM_Tmax, downdir_TRCLM_RET, downdir_PopDensity_GPW, downdir_MODIS_ET, \
    downdir_MODIS_PET, downdir_SRTM_DEM, downdir_AI, downdir_Clay_content, \
    downdir_MODIS_land_use, downdir_TRCLM_ET = download_data(geedatalist, yearlist, start_month, end_month, downloadcsv,
                                                             gee_scale, skip_download)

    EVI = glob(os.path.join(downdir_EVI, 'merged_rasters', '*.tif'))[0]
    NDWI = glob(os.path.join(downdir_NDWI, 'merged_rasters', '*.tif'))[0]
    Grace = glob(os.path.join(downdir_Grace, 'merged_rasters', '*.tif'))[0]
    TRCLM_precp = glob(os.path.join(downdir_TRCLM_precp, 'merged_rasters', '*.tif'))[0]
    TRCLM_soil = glob(os.path.join(downdir_TRCLM_soil, 'merged_rasters', '*.tif'))[0]
    TRCLM_Tmin = glob(os.path.join(downdir_TRCLM_Tmin, 'merged_rasters', '*.tif'))[0]
    TRCLM_Tmax = glob(os.path.join(downdir_TRCLM_Tmax, 'merged_rasters', '*.tif'))[0]
    TRCLM_RET = glob(os.path.join(downdir_TRCLM_RET, 'merged_rasters', '*.tif'))[0]
    PopDensity_GPW = glob(os.path.join(downdir_PopDensity_GPW, 'merged_rasters', '*.tif'))[0]
    MODIS_ET = glob(os.path.join(downdir_MODIS_ET, 'merged_rasters', '*monthly*.tif'))[0]
    MODIS_PET = glob(os.path.join(downdir_MODIS_PET, 'merged_rasters', '*monthly*.tif'))[0]
    SRTM_DEM = glob(os.path.join(downdir_SRTM_DEM, 'merged_rasters', '*.tif'))[0]
    Aridity_Index = glob(os.path.join(downdir_AI, 'merged_rasters', '*.tif'))[0]
    Clay_0cm = os.path.join(downdir_Clay_content, 'claycontent_0cm/merged_rasters/clay_content_0cm_2013_2019.tif')
    Clay_10cm = os.path.join(downdir_Clay_content, 'claycontent_10cm/merged_rasters/clay_content_10cm_2013_2019.tif')
    Clay_30cm = os.path.join(downdir_Clay_content, 'claycontent_30cm/merged_rasters/clay_content_30cm_2013_2019.tif')
    Clay_60cm = os.path.join(downdir_Clay_content, 'claycontent_60cm/merged_rasters/clay_content_60cm_2013_2019.tif')
    Clay_100cm = os.path.join(downdir_Clay_content, 'claycontent_100cm/merged_rasters/clay_content_100cm_2013_2019.tif')
    Clay_200cm = os.path.join(downdir_Clay_content, 'claycontent_200cm/merged_rasters/clay_content_200cm_2013_2019.tif')
    MODIS_LU = os.path.join(downdir_MODIS_land_use, 'merged_rasters/MODIS_Land_Use_2013_2019.tif')
    TRCLM_ET = os.path.join(downdir_TRCLM_ET, 'merged_rasters/TRCLM_ET_2013_2019.tif')

    if yearlist[0] == 2013:
        Alexi_ET = glob(os.path.join(r'../Data/Raw_Data/Alexi_ET/mean_rasters', '*2013*.tif'))[0]
    else:
        Alexi_ET = glob(os.path.join(r'../Data/Raw_Data/Alexi_ET/mean_rasters', '*2018*.tif'))[0]

    Downloaded_list = {'EVI': EVI, 'NDWI': NDWI, 'Grace': Grace, 'TRCLM_precp': TRCLM_precp, 'TRCLM_soil': TRCLM_soil,
                       'TRCLM_Tmin': TRCLM_Tmin, 'TRCLM_Tmax': TRCLM_Tmax, 'TRCLM_RET': TRCLM_RET, 'MODIS_ET': MODIS_ET,
                       'MODIS_PET': MODIS_PET, 'SRTM_Slope': SRTM_DEM, 'Aridity_Index': Aridity_Index,
                       'Alexi_ET': Alexi_ET, 'Clay_content_PCA': None, 'MODIS_Land_Use': MODIS_LU, 'TRCLM_ET': TRCLM_ET,
                       'Clay_200cm': Clay_200cm}

    if not skip_processing:
        resampled_gee_rasters = {}
        for data, path in Downloaded_list.items():
            print('Processing', data, '...')

            if data == 'Alexi_ET':
                name = path[path.rfind(os.sep) + 1:]
                resampled_raster = resample_reproject(Downloaded_list[data], output_dir=resampled_gee_dir,
                                                      raster_name=name, resample=True)
                resampled_gee_rasters[data] = resampled_raster

            elif data == 'SRTM_Slope':
                slope_raster = create_slope_raster(Downloaded_list[data], outdir=resampled_gee_dir,
                                                   raster_name='SRTM_Slope_2013_2019.tif')
                resampled_gee_rasters[data] = slope_raster

            elif data == 'Clay_content_PCA':
                if perform_pca:
                    pca_clay_list = [Clay_0cm, Clay_10cm, Clay_30cm, Clay_60cm, Clay_100cm, Clay_200cm]
                    pca_raster = perform_pca_clay(pca_clay_list, deconstruct_pca=False)
                    resampled_gee_rasters[data] = pca_raster
                else:
                    resampled_gee_rasters[data] = '../Data/Resampled_Data/PCA_Clay/continent_raster/' \
                                                  'pca_clay_content.tif'
                    print('PCA of clay content exists')

            elif data == 'MODIS_Land_Use':
                resampled_raster = prepare_modis_landuse_data(
                    output_raster='../Data/Resampled_Data/GEE_data_2013_2019/MODIS_Land_Use.tif',
                    input_raster=Downloaded_list[data])
                resampled_gee_rasters[data] = resampled_raster

            else:
                resampled_raster = rename_copy_raster(input_raster=Downloaded_list[data], output_dir=resampled_gee_dir,
                                                      rename=False)
                resampled_gee_rasters[data] = resampled_raster

        pickle.dump(resampled_gee_rasters, open(os.path.join(resampled_gee_dir, 'gee_path_dict.pkl'), mode='wb+'))

    else:
        resampled_gee_rasters = pickle.load(open(os.path.join(resampled_gee_dir, 'gee_path_dict.pkl'), mode='rb'))

    gfsad_raster, irrigated_meier_raster, giam_gw_raster = prepare_lu_data(gfsad_cropextent, giam_gw, irrigated_meier,
                                                                           intermediate_dir, outdir_lu, skip_processing)

    sediment_thickness_raster = prepare_sediment_thickness_data(sediment_thickness, intermediate_dir,
                                                                outdir_sed_thickness, 'Global_Sediment_Thickness.tif',
                                                                skip_processing)

    clay_thickness_raster = prepare_clay_thickness_data(Clay_200cm, sediment_thickness_raster, outdir_sed_thickness,
                                                        skip_processing)

    popdensity_raster = prepare_popdensity_data(PopDensity_GPW, outdir_pop, skip_processing)

    river_distance = prepare_river_proximity_data(river_shape, outdir_sw, skip_processing=skip_processing)

    confining_layers = process_global_confining_layer_data(confining_layer, outdir_confining_layer)

    return resampled_gee_rasters, gfsad_raster, irrigated_meier_raster, giam_gw_raster, sediment_thickness_raster, \
           clay_thickness_raster, popdensity_raster, river_distance, confining_layers


def join_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygons, exclude_areas=None,
                                           search_criteria='*Subsidence*.shp'):
    """
    Joining georeferenced subsidence polygons.

    Area processed -'Australia_Perth', 'Bangladesh_GBDelta', 'China_Beijing', 'China_Shanghai', 'China_Tianjin',
    'China_Wuhan', 'China_Xian', 'China_YellowRiverDelta', 'Egypt_NileDelta', 'England_London', 'India_Delhi',
    'Indonesia_Bandung', 'Indonesia_Semarang', 'Iran_MarandPlain', 'Iran_Tehran', 'Iraq_TigrisEuphratesBasin',
    'Italy_PoDelta', 'Italy_VeniceLagoon', 'Mexico_MexicoCity', 'Nigeria_Lagos', 'Spain_Murcia', 'Taiwan_Yunlin',
    'Turkey_Bursa', 'Turkey_Karapinar', 'US_Huston', 'Vietnam_Hanoi', 'Vietnam_HoChiMinh'

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    exclude_areas : Tuple of area names to be excluded from processing. Default set to None.
                    For excluding single area follow tuple pattern ('Bangladesh_GBDelta',)
    search_criteria : Search criteria for input polygons.

    Returns : Joined subsidence polygon.
    """
    # global df
    subsidence_polygons = glob(os.path.join(input_polygons_dir, search_criteria))

    sep = joined_subsidence_polygons.rfind(os.sep)
    makedirs([joined_subsidence_polygons[:sep]])  # creating directory for the prepare_subsidence_raster function

    for each in range(0, len(subsidence_polygons)):
        if each == 0:
            gdf = gpd.read_file(subsidence_polygons[each])
            df = pd.DataFrame(gdf)

        gdf_new = gpd.read_file(subsidence_polygons[each])
        df_new = pd.DataFrame(gdf_new)
        add_to_df = pd.concat([df, df_new], ignore_index=True)
        df = add_to_df
        df['Class_name'] = pd.to_numeric(df['Class_name'], downcast='float')

    if exclude_areas is not None:
        exclude_areas = list(exclude_areas)
        areas = list(df['Area_name'].unique())
        keep_areas = [area for area in areas if area not in exclude_areas]
        df = df.loc[df['Area_name'].isin(keep_areas)]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.to_file(joined_subsidence_polygons)
    return joined_subsidence_polygons


def prepare_subsidence_raster(input_polygons_dir='../InSAR_Data/Georeferenced_subsidence_data',
                              joined_subsidence_polygon='../InSAR_Data/Merged_subsidence_data'
                                                        '/interim_working_dir/georef_subsidence_polygons.shp',
                              insar_data_dir='../InSAR_Data/Merged_subsidence_data/resampled_insar_data',
                              interim_dir='../InSAR_Data/Merged_subsidence_data/interim_working_dir',
                              output_dir='../InSAR_Data/Merged_subsidence_data/final_subsidence_raster',
                              skip_polygon_merge=False, subsidence_column='Class_name', resample_algorithm='near',
                              final_subsidence_raster='Subsidence_training.tif', exclude_georeferenced_areas=None,
                              process_insar_areas=('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin',
                                                   'China_Hebei', 'China_Hefei', 'Colorado'),
                              polygon_search_criteria='*Subsidence*.shp',
                              insar_search_criteria='*reclass_resampled*.tif', already_prepared=False,
                              refraster=referenceraster, merge_coastal_subsidence_data=False):
    """
    Prepare subsidence raster for training data by joining georeferenced polygons and insar data.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    insar_data_dir : InSAR data directory.
    interim_dir : Intermediate working directory for storing interdim data.
    output_dir : Output raster directory.
    skip_polygon_merge : Set to True if polygon merge is not required.
    subsidence_column : Subsidence value column in joined subsidence polygon. Default set to 'Class_name'.
    resample_algorithm : Algorithm for resampling polygon subsidence data. Default set to 'near'.
    final_subsidence_raster : Final subsidence raster including georeferenced and insar data.
    exclude_georeferenced_areas : Tuple of area names to be excluded from processing. Default set to None to include all
                                  gereferenced areas. For excluding single area follow
                                  tuple pattern ('Bangladesh_GBDelta',)
    polygon_search_criteria : Input subsidence polygon search criteria.
    process_insar_areas : Tuple of insar data regions to be included in the model.
                          Default set to ('California', 'Arizona', 'Pakistan_Quetta', 'Iran_Qazvin', 'China_Hebei',
                                          'China_Hefei', 'Colorado')
    insar_search_criteria : InSAR data search criteria.
    already_prepared : Set to True if subsidence raster is already prepared.
    refraster : Global Reference raster.

    Returns : Final subsidence raster to be used as training data.
    """

    if not already_prepared:
        makedirs([interim_dir, output_dir])
        if not skip_polygon_merge:
            print('Processing Subsidence Polygons...')
            subsidene_polygons = join_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygon,
                                                                        exclude_georeferenced_areas,
                                                                        polygon_search_criteria)
        else:
            subsidene_polygons = joined_subsidence_polygon

        interim_georeferenced_subsidence = \
            shapefile_to_raster(subsidene_polygons, interim_dir, resolution=0.005,
                                raster_name='interim_subsidence_raster_0005.tif', use_attr=True,
                                attribute=subsidence_column, ref_raster=refraster, alltouched=False)
        georeferenced_subsidence = resample_reproject(interim_georeferenced_subsidence, interim_dir,
                                                      raster_name='interim_subsidence_raster.tif', resample=True,
                                                      reproject=False, both=False,
                                                      resample_algorithm=resample_algorithm)

        print('Processed Subsidence Polygons')

        print('Processing InSAR Data...')

        process_primary_insar_data(processing_areas=process_insar_areas, output_dir=insar_data_dir)
        insar_arr, merged_insar = mosaic_rasters(insar_data_dir, interim_dir, raster_name='joined_insar_data.tif',
                                                 ref_raster=refraster, search_by=insar_search_criteria, resolution=0.02)

        final_subsidence_arr, subsidence_data = mosaic_two_rasters(merged_insar, georeferenced_subsidence, output_dir,
                                                                   final_subsidence_raster, resolution=0.02)

        if merge_coastal_subsidence_data:
            coastal_raster = rasterize_coastal_subsidence(mean_output_points='../InSAR_Data/Coastal_Subsidence'
                                                                             '/filtered_mean_point.shp',
                                                          output_dir='../InSAR_Data/Coastal_Subsidence',
                                                          input_csv='../InSAR_Data/Coastal_Subsidence/Fig3_data.csv')
            coastal_arr = read_raster_arr_object(coastal_raster, get_file=False)
            ref_arr, ref_file = read_raster_arr_object(refraster)

            # New_classes
            sub_less_1cm = 1
            sub_1cm_to_5cm = 5
            sub_greater_5cm = 10
            other_values = np.nan

            coastal_arr = np.where(coastal_arr >= 0, other_values, coastal_arr)
            coastal_arr = np.where(coastal_arr >= -1, sub_less_1cm, coastal_arr)
            coastal_arr = np.where((coastal_arr < -1) & (coastal_arr >= -5), sub_1cm_to_5cm, coastal_arr)
            coastal_arr = np.where(coastal_arr < -5, sub_greater_5cm, coastal_arr)
            coastal_arr = coastal_arr.flatten()

            final_subsidence_arr = final_subsidence_arr.flatten()
            final_subsidence_arr = np.where(final_subsidence_arr > 0, final_subsidence_arr, coastal_arr)

            # filtering pixels of coastal subsidence that has been added to final subsidence raster
            coastal_arr_used = np.where(np.logical_and(coastal_arr > 0, coastal_arr == final_subsidence_arr),
                                        coastal_arr, np.nan)
            coastal_arr_used = coastal_arr_used.reshape(ref_file.shape[0], ref_file.shape[1])
            coastal_subsidence_raster = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data' \
                                        '/Coastal_subsidence.tif'
            write_raster(coastal_arr_used, ref_file, ref_file.transform, coastal_subsidence_raster, ref_file=refraster)

            final_subsidence_arr = final_subsidence_arr.reshape(ref_file.shape[0], ref_file.shape[1])
            write_raster(final_subsidence_arr, ref_file, ref_file.transform, subsidence_data, ref_file=refraster)

        print('Created Final Subsidence Raster')
        return subsidence_data

    else:
        subsidence_data = os.path.join(output_dir, final_subsidence_raster)
        return subsidence_data


def compile_predictors_subsidence_data(gee_data_dict, gfsad_irrigated_area, giam_gw_data, irrigated_meier_data,
                                       sediment_thickness_data, clay_thickness_data, popdensity_data,
                                       river_distance_data, confining_layer_data, subsidence_data,
                                       output_dir, skip_compiling_predictor_subsidence_data=False):
    """
    Compile predictor datasets and subsidence data in a single folder (to be used for creating predictor database)

    Parameters:
    gee_data_dict : GEE data dictionary consisting resampled gee filepath.
    gfsad_irrigated_area : Resampled GFSAD irrigated area raster filepath.
    giam_gw : Resampled GIAM GW Irrigation filepath.
    irrigated_meier_data: Resampled Irrigated area by meier filepath.
    sediment_thickness_data : Resampled sediment thickness filepath.
    clay_thickness_data : Resampled clay thickness data.
    popdensity_data : Resampled population density filepath.
    river_distance_data : Resampled river distance data filepath.
    confining_layer_data : Resampled confining layer data filepath.
    subsidence_data : Resampled subsidence filepath.
    output_dir : Output directory filepath.
    skip_predictor_subsidence_compilation : Set to True if want to skip compiling all the data again.

    Returns : Output directory filepath.
    """
    if not skip_compiling_predictor_subsidence_data:
        shutil.rmtree(output_dir, ignore_errors=True)
        makedirs([output_dir])

        for key in gee_data_dict.keys():
            rename_copy_raster(gee_data_dict[key], output_dir, rename=True, new_name=(key + '.tif'))

        rename_copy_raster(gfsad_irrigated_area, output_dir, rename=False)
        rename_copy_raster(gfsad_irrigated_area, output_dir, rename=False)
        rename_copy_raster(giam_gw_data, output_dir, rename=False)
        rename_copy_raster(irrigated_meier_data, output_dir, rename=True, new_name='Irrigated_Area_Density2.tif')
        rename_copy_raster(sediment_thickness_data, output_dir, rename=False)
        rename_copy_raster(clay_thickness_data, output_dir, rename=False)
        rename_copy_raster(popdensity_data, output_dir, rename=True, new_name='Population_Density.tif')
        rename_copy_raster(river_distance_data, output_dir, rename=False)
        rename_copy_raster(confining_layer_data, output_dir, rename=False)
        rename_copy_raster(subsidence_data, output_dir, rename=True, new_name='Subsidence.tif')

    return output_dir


