
#converting shapefile to raster

def shapefile_to_raster(input_shape_fp, output_dir,shp_field_pos=1, res=0.05, gdal_path='C:\\OSGeo4W64\\'):
    
    """
    :param input_shape_fp : input shapefile path 
    :param shp_field_pos : column index of shapefile that will be converted to raster value; default 1 
    :param output_dir : output raster file directory
    :param res : cell size of the output raster; default 0.05
    :param gdal_path='C:\\OSGeo4W\' for windows operating system, for linux or Mac-'/usr/bin/gdal/'
    """
    
    import os
    import geopandas as gpd
    
    #loading shapefile
    shapefile=gpd.read_file(input_shape_fp)
    value_field=shapefile.columns[shp_field_pos]
    
    #getting geometric bound
    minx,miny,maxx,maxy=shapefile.geometry.total_bounds
    
    #layer name
    extn_pos=input_shape_fp.rfind ('.')
    layer_name=input_shape_fp[input_shape_fp.rfind(os.sep) + 1:extn_pos]
    
    #naming output file
    out_name=layer_name+'.tif'
    output_fp=os.path.join(output_dir,out_name)
    
    
    #no data value
    no_data_value= -9999
    
    #rasterriziz argument and command
    rasterize_arg=['-l',layer_name,'-a',value_field,'-tr',str(res),str(res),'-te',str(minx),str(miny), str(maxx), str(maxy),
     '-init',str(0.0),'-add','-ot','Float32','-of','GTiff','-a_nodata',str(no_data_value), input_shape_fp,output_fp]
    
    gdal_command='gdal_rasterize'
    
    #adding the batch file of OSGeo with gdal path
    if os.name=='nt':
        gdal_path+='OSGeo4W.bat'
    
    #calling subprocess
    import subprocess
    subprocess.call([gdal_path]+[gdal_command]+rasterize_arg)
    
    
# =============================================================================
# raster_converted=shapefile_to_raster(input_shape_fp="E:\\NGA_Project_Data\\shapefiles\\continent_extents\\Europe_continent.shp"
#                                      ,output_dir="E:\\NGA_Project_Data\\shapefiles\\raster_mod\\Europe_continent.tif")
# =============================================================================

shape=r"E:\NGA_Project_Data\Georeferencing\China_Beijing_Jiao_2018\Tree_based_classification\Subsidence_polygon_Dissolved.shp"
outdir=r"E:\NGA_Project_Data\scratch_files"
shapefile_to_raster(input_shape_fp=shape, output_dir=outdir,res=0.02)

