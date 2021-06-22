import os
import geopandas as gpd

def select_by_attribute(shape,column,value,outshape):
    """
    Select by column attribute

    Parameters:
        shape: Input shapefile path.
        column: Shapefile column to compare.
        value: A list of string or value to make comparison.
        outshape (TYPE): Output shapefile path.

    Returns:None.
    """
    shape=gpd.read_file(shape)
    select_shape=shape[shape[column]==value]
    select_shape.to_file(outshape)

def append_shapefile(shape1,shape2,outshape,ignore_index=True):
    df1=gpd.read_file(shape1)
    df2=gpd.read_file(shape2)
    
    #append
    df_append=df1.append(df2,ignore_index=ignore_index)
    df_append.to_file(outshape)
    
def overlay(shape1,shape2,outshape,how='difference'):
    """
    overlay operations (Intersection, Union, Symetrical Difference, DIfference) on two shapefiles.

    Parameters:
        shape1: Shapefile 1 path.
        shape2: Shapefile 2 path.
        outshape: Output Shapefile after overlay operation with file path.
        how (TYPE, optional): Overlay operation type (Intersection, Union, Symetrical Difference,
                              DIfference). Defaults to 'difference'.
    
    Returns: None.
    """
    shape1=gpd.read_file(shape1)
    shape2=gpd.read_file(shape2)

    #overlay
    overlay_shape=gpd.overlay(shape1,shape2,how=how)
    overlay_shape.to_file(outshape)

def buffer(shape,outshape,Reprojection=True,buffer=1000,projected_epsg_code=8857):
    """
    Fixed Buffer around a shapefile.

    Parameters:
        shape: Shapefile around which buffer will be done.
        outshape: Output Shapefile after buffer operation with file path.
        Reprojection: Defaults to True. Set False if Geographic to Projected conversion is not 
                      needed.
        buffer: Buffer distance. Value should be in meter. Defaults to 1000m.
        projected_epsg_code: Convert to projected coordinate system if input shapefile's 
                             coordinate system is Geographic. Defaults to 8857 
                             (Equal Earth (World))
        
    Returns: None.
    """
    shape=gpd.read_file(shape)
    
    if Reprojection:
        shape_projected=shape.to_crs(epsg=projected_epsg_code)
        
        buffer_shape=shape_projected['geometry'].buffer(distance=buffer)
        buffer_shape_reprojected=buffer_shape.to_crs(epsg=4326)
        buffer_shape_reprojected.to_file(outshape)
    else:
        buffer_shape=shape['geometry'].buffer(distance=buffer)
        buffer_shape.to_file(outshape)
        
def buffer_variable(shape,outshape,buffer_coef=0.0015,Reprojection=True,projected_epsg_code=8857):
    """
    Variable Buffer around a shapefile 
    (buffer distance is a function of area of the corresponding shapefile).

    Parameters:
        shape: Shapefile around which buffer will be done.
        outshape: Output Shapefile after buffer operation with file path.
        Reprojection: Defaults to True. Set False if Geographic to Projected conversion is not 
                      needed.
        buffer_coef: Buffer coefficient to calculate distance. Defaults to 0.00015 % of area.
        projected_epsg_code: Convert to projected coordinate system if input shapefile's 
                             coordinate system is Geographic. Defaults to 8857 
                             (Equal Earth (World))
        
    Returns: None.
    """
    shape=gpd.read_file(shape)
    
    if Reprojection:
        shape_projected=shape.to_crs(epsg=projected_epsg_code)
        
        shape_projected['buffer']=shape_projected['geometry'].area*(buffer_coef/100)
        buffer_shape=shape_projected['geometry'].buffer(distance=shape_projected['buffer'])
        buffer_reprojected=buffer_shape.to_crs(epsg=4326)
        buffer_reprojected.to_file(outshape)
    else:
        shape['buffer']=shape['geometry'].area*(buffer_coef/100)
        buffer_shape=shape['geometry'].buffer(distance=shape['buffer'])
        buffer_shape.to_file(outshape)
    
def separate_shapes(input_shape,output_dir,index_col=True,label='Id'):
    """
    Separate individual shapes from a single shapefile.

    Parameters:
    input_shape : Input shape file.
    output_dir : Output shapefile sirectory..
    index_col : If new index column need to be created. Defaults to True.
    label : Label based on which separation will occur. Defaults to 'Id'.

    Returns: None.
    """
    shape=gpd.read_file(input_shape)
    
    num=1
    if index_col:
        shape['index']=shape.index 
        for each in shape['index']:
           name= input_shape[input_shape.rfind(os.sep)+1:input_shape.rfind(".")]+"_"+str(num)+"_"+".shp"
           new_shape=shape[shape['index']==each]
           new_shape.to_file(os.path.join(output_dir,name))
           num=num+1
    else:
       for each in shape[label]:
           name= input_shape[input_shape.rfind(os.sep)+1:input_shape.rfind(".")]+"_"+str(num)+".shp"
           new_shape=shape[shape[label]==each]
           new_shape.to_file(os.path.join(output_dir,name))
           num=num+1

# =============================================================================
# #Separating World Grid Shapefiles for dowloading GEE data
# shape="E:\\NGA_Project_Data\\scratch_files\\WorldGrid.shp"
# outdir="E:\\NGA_Project_Data\\shapefiles\\world_grid_shapes_for_gee"
# separate_shapes(shape,outdir)
# =============================================================================
