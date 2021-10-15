# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import os
import numpy as np
import pandas as pd
from glob import glob
from Raster_operations import write_raster, clip_resample_raster_cutline, mosaic_rasters
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from System_operations import makedirs

No_Data_Value = -9999


def perform_pca_clay(pca_raster_list, dict_keyword_list=None, output_raster_dir='../Data/Resampled_Data/PCA_Clay',
                     deconstruct_pca=False):
    if dict_keyword_list is None:
        dict_keyword_list = ['clay_0cm', 'clay_10cm', 'clay_30cm', 'clay_60cm', 'clay_100cm', 'clay_200cm']

    pca_rasters_dict = {}
    for i in range(0, len(pca_raster_list)):
        pca_rasters_dict[dict_keyword_list[i]] = pca_raster_list[i]

    continent_shapes = glob(os.path.join('../Data/Reference_rasters_shapes/continent_extents', '*continent.shp'))

    for shape in continent_shapes:
        continent_name = shape[shape.rfind(os.sep) + 1:shape.rfind('_')]
        print('performing PCA for', continent_name)

        clay_0cm_arr, clay_0cm_file = clip_resample_raster_cutline(pca_rasters_dict['clay_0cm'],
                                                                   output_raster_dir, shape)
        clay_10cm_arr, clay_10cm_file = clip_resample_raster_cutline(pca_rasters_dict['clay_10cm'],
                                                                     output_raster_dir, shape)
        clay_30cm_arr, clay_30cm_file = clip_resample_raster_cutline(pca_rasters_dict['clay_30cm'],
                                                                     output_raster_dir, shape)
        clay_60cm_arr, clay_60cm_file = clip_resample_raster_cutline(pca_rasters_dict['clay_60cm'],
                                                                     output_raster_dir, shape)
        clay_100cm_arr, clay_100cm_file = clip_resample_raster_cutline(pca_rasters_dict['clay_100cm'],
                                                                       output_raster_dir, shape)
        clay_200cm_arr, clay_200cm_file = clip_resample_raster_cutline(pca_rasters_dict['clay_200cm'],
                                                                       output_raster_dir, shape)

        raster_shape = clay_0cm_arr.shape

        data_dict = {'clay_0cm': clay_0cm_arr.flatten(), 'clay_10cm': clay_10cm_arr.flatten(),
                     'clay_30cm': clay_30cm_arr.flatten(), 'clay_60cm': clay_60cm_arr.flatten(),
                     'clay_100cm': clay_100cm_arr.flatten(), 'clay_200cm': clay_200cm_arr.flatten()}

        nan_pos_dict = {}
        modified_data_dict = {}
        for key, array in data_dict.items():
            nan_pos_dict[key] = np.isnan(array)
            array[nan_pos_dict[key]] = 0
            modified_data_dict[key] = array

        clay_0cm = pd.Series(modified_data_dict['clay_0cm'])
        clay_10cm = pd.Series(modified_data_dict['clay_10cm'])
        clay_30cm = pd.Series(modified_data_dict['clay_30cm'])
        clay_60cm = pd.Series(modified_data_dict['clay_60cm'])
        clay_100cm = pd.Series(modified_data_dict['clay_100cm'])
        clay_200cm = pd.Series(modified_data_dict['clay_200cm'])

        final_data_dict = {'clay_0cm': clay_0cm, 'clay_10cm': clay_10cm, 'clay_30cm': clay_30cm, 'clay_60cm': clay_60cm,
                           'clay_100cm': clay_100cm, 'clay_200cm': clay_200cm}

        df = pd.DataFrame(final_data_dict)
        x_df = df.values
        x = StandardScaler().fit_transform(x_df)
        pca = PCA()
        pca_result = pca.fit_transform(x)
        print(pca.explained_variance_ratio_[0])

        pca_component1 = pca_result[:, [0]].flatten()
        for key in nan_pos_dict.keys():
            pca_component1[nan_pos_dict[key]] = clay_0cm_file.nodata

        pca_component1 = pca_component1.reshape(raster_shape[0], raster_shape[1])

        pca_raster1 = os.path.join(output_raster_dir, continent_name + '_pca1.tif')

        write_raster(pca_component1, clay_0cm_file, clay_0cm_file.transform, pca_raster1)

        # if deconstruct_pca:
        #     # # concept from https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-
        #     # original-variables-from-several-principal-com
        #
        #     mean = np.mean(x_df, axis=0)
        #     ncomp = 1
        #     pca_deconstruct_arr = np.dot(pca_result[:, :ncomp], pca.components_[:ncomp, :])
        #     pca_deconstruct_arr += mean
        #     pca_deconstruct_arr = np.amax(pca_deconstruct_arr, axis=1)
        #     pca_deconstruct_arr = pca_deconstruct_arr.flatten()
        #
        #     for key in nan_pos_dict.keys():
        #         pca_deconstruct_arr[nan_pos_dict[key]] = clay_0cm_file.nodata
        #
        #     pca_deconstruct_arr = pca_deconstruct_arr.reshape(raster_shape[0], raster_shape[1])
        #     pca_deconstruct_raster = os.path.join(output_raster_dir, continent_name + '_pca_deconstruct.tif')
        #     write_raster(pca_deconstruct_arr, clay_0cm_file, clay_0cm_file.transform, pca_deconstruct_raster)

    input_dir = output_raster_dir
    continent_raster_dir = os.path.join(input_dir, 'continent_raster')
    makedirs(continent_raster_dir)

    if not deconstruct_pca:
        pca_arr, pca_raster = mosaic_rasters(output_raster_dir, continent_raster_dir, 'pca_clay_content.tif',
                                             search_by='*pca1.tif')

        print('PCA Clay results saved as raster')

        return pca_raster

    # else:
    #     pca_arr, pca_raster = mosaic_rasters(output_raster_dir, continent_raster_dir, 'pca_clay_content.tif',
    #                                          search_by='*pca1.tif')
    #
    #     print('PCA Clay results saved as raster')
    #
    #     pca_decons_arr, pca_decons_raster = mosaic_rasters(output_raster_dir, continent_raster_dir,
    #                                                        'pca_clay_content_deconstructed.tif',
    #                                                        search_by='*pca_deconstruct.tif')
    #     return pca_raster, pca_decons_raster


# Clay_0cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_0cm/merged_rasters/clay_content_0cm_2013_2019.tif'
# Clay_10cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_10cm/merged_rasters/clay_content_10cm_2013_2019.tif'
# Clay_30cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_30cm/merged_rasters/clay_content_30cm_2013_2019.tif'
# Clay_60cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_60cm/merged_rasters/clay_content_60cm_2013_2019.tif'
# Clay_100cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_100cm/merged_rasters/clay_content_100cm_2013_2019.tif'
# Clay_200cm = '../Data/Raw_Data/GEE_data/Clay_content_openlandmap/claycontent_200cm/merged_rasters/clay_content_200cm_2013_2019.tif'
# raster_list = [Clay_0cm, Clay_10cm, Clay_30cm, Clay_60cm, Clay_100cm, Clay_200cm]
#
# perform_pca_clay(raster_list, dict_keyword_list=None, output_raster_dir='../Data/Resampled_Data/PCA_Clay',
#                  deconstruct_pca=True)