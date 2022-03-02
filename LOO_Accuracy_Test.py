# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import os
import pickle
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, \
    precision_score, recall_score, f1_score
from System_operations import makedirs
from Raster_operations import shapefile_to_raster, mosaic_rasters, mosaic_two_rasters, read_raster_arr_object, \
    write_raster, clip_resample_raster_cutline

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # to ignore future warning coming from pandas
warnings.filterwarnings(action='ignore')

referenceraster = '../Data/Reference_rasters_shapes/Global_continents_ref_raster.tif'


def combine_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygons,
                                              search_criteria='*Subsidence*.shp', skip_polygon_processing=True):
    """
    Combining georeferenced subsidence polygons.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    search_criteria : Search criteria for input polygons.
    skip_polygon_processing : Set False if want to process georeferenced subsidence polygons.

    Returns : Joined subsidence polygon.
    """

    if not skip_polygon_processing:
        subsidence_polygons = glob(os.path.join(input_polygons_dir, search_criteria))

        sep = joined_subsidence_polygons.rfind(os.sep)
        makedirs([joined_subsidence_polygons[:sep]])  # creating directory for the  prepare_subsidence_raster function

        for each in range(1, len(subsidence_polygons) + 1):
            if each == 1:
                gdf = gpd.read_file(subsidence_polygons[each - 1])

            gdf_new = gpd.read_file(subsidence_polygons[each - 1])
            add_to_gdf = pd.concat([gdf, gdf_new], ignore_index=True)
            gdf = add_to_gdf
            gdf['Class_name'] = gdf['Class_name'].astype(float)

        unique_area_name = gdf['Area_name'].unique().tolist()
        unique_area_name_code = [i + 1 for i in range(len(unique_area_name))]

        polygon_area_name_dict = {}
        for name, code in zip(unique_area_name, unique_area_name_code):
            polygon_area_name_dict[name] = code

        Area_code = []
        for index, row in gdf.iterrows():
            Area_code.append(polygon_area_name_dict[row['Area_name']])

        gdf['Area_code'] = pd.Series(Area_code)

        gdf.to_file(joined_subsidence_polygons)

        pickle.dump(polygon_area_name_dict, open('../Model Run/LOO_Test/InSAR_Data/polygon_area_name_dict.pkl',
                                                 mode='wb+'))

    else:
        joined_subsidence_polygons = '../Model Run/LOO_Test/InSAR_Data/georef_subsidence_polygons.shp'
        polygon_area_name_dict = pickle.load(open('../Model Run/LOO_Test/InSAR_Data/polygon_area_name_dict.pkl',
                                                  mode='rb'))

    return joined_subsidence_polygons, polygon_area_name_dict


def substitute_area_code_on_raster(input_raster, value_to_substitute, output_raster):
    """
    Substitute raster values with area code for InSAR produced subsidence rasters (California, Arizona, Quetta, Qazvin,
    China_Hebei, Coastal_subsidence etc.)

    Parameters:
    input_raster : Input subsidence raster filepath.
    value_to_substitute : Area code that will substitute raster values.
    output_raster : Filepath of output raster.

    Returns : Raster with values substituted with area code.
    """
    raster_arr, raster_file = read_raster_arr_object(input_raster)

    raster_arr = np.where(np.isnan(raster_arr), raster_arr, value_to_substitute)

    area_coded_raster = write_raster(raster_arr, raster_file, raster_file.transform, output_raster)

    return area_coded_raster


def combine_georef_insar_subsidence_raster(input_polygons_dir='../InSAR_Data/Georeferenced_subsidence_data',
                                           joined_subsidence_polygon='../Model Run/LOO_Test/InSAR_Data/'
                                                                     'georef_subsidence_polygons.shp',
                                           insar_data_dir='../Model Run/LOO_Test/InSAR_Data/'
                                                          'interim_working_dir',
                                           interim_dir='../Model Run/LOO_Test/InSAR_Data/'
                                                       'interim_working_dir',
                                           output_dir='../Model Run/LOO_Test/InSAR_Data/'
                                                      'final_subsidence_raster',
                                           skip_polygon_processing=False,
                                           area_code_column='Area_code',
                                           final_subsidence_raster='Subsidence_area_coded.tif',
                                           polygon_search_criteria='*Subsidence*.shp', already_prepared=False,
                                           refraster=referenceraster):
    """
    Prepare area coded subsidence raster for training data by joining georeferenced polygons and insar data.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    insar_data_dir : InSAR data directory.
    interim_dir : Intermediate working directory for storing interim data.
    output_dir : Output raster directory.
    skip_polygon_processing : Set to True if polygon merge is not required.
    final_subsidence_raster : Final subsidence raster including georeferenced and insar data.
    polygon_search_criteria : Input subsidence polygon search criteria.
    insar_search_criteria : InSAR data search criteria.
    already_prepared : Set to True if subsidence raster is already prepared.
    refraster : Global Reference raster.

    Returns : Final subsidence raster to be used as training data and a subsidence area code dictionary.
    """

    if not already_prepared:
        makedirs([interim_dir, output_dir])

        print('Processing area coded subsidence polygons...')
        subsidence_polygons, subsidence_areaname_dict = \
            combine_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygon,
                                                      polygon_search_criteria, skip_polygon_processing)

        print('Processed area coded subsidence polygons')
        georeferenced_raster_area_coded = shapefile_to_raster(subsidence_polygons, interim_dir,
                                                              raster_name='interim_georef_subsidence_raster_areacode'
                                                                          '.tif',
                                                              use_attr=True, attribute=area_code_column,
                                                              ref_raster=refraster, alltouched=False)
        georef_arr = read_raster_arr_object(georeferenced_raster_area_coded, get_file=None)

        print('Processing area coded InSAR data...')
        georef_subsidence_gdf = gpd.read_file(joined_subsidence_polygon)
        num_of_georef_subsidence = len(georef_subsidence_gdf['Area_code'].unique())

        california_area_code = num_of_georef_subsidence + 1
        arizona_area_code = california_area_code + 1
        quetta_area_code = arizona_area_code + 1
        qazvin_area_code = quetta_area_code + 1
        hebei_area_code = qazvin_area_code + 1
        coastal_area_code = hebei_area_code + 1

        subsidence_areaname_dict['California'] = california_area_code
        subsidence_areaname_dict['Arizona'] = arizona_area_code
        subsidence_areaname_dict['Pakistan_Quetta'] = quetta_area_code
        subsidence_areaname_dict['Iran_Qazvin'] = qazvin_area_code
        subsidence_areaname_dict['China_Hebei'] = hebei_area_code
        subsidence_areaname_dict['Coastal'] = coastal_area_code

        california_subsidence = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data' \
                                '/California_reclass_resampled.tif'
        arizona_subsidence = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data/' \
                             'Arizona_reclass_resampled.tif'
        quetta_subsidence = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data' \
                            '/Pakistan_Quetta_reclass_resampled.tif'
        qazvin_subsidence = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data/' \
                            'Iran_Qazvin_reclass_resampled.tif'
        hebei_subsidence = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data/' \
                           'China_Hebei_reclass_resampled.tif'
        coastal_subsidence = '../InSAR_Data/Merged_subsidence_data/resampled_insar_data' \
                             '/Coastal_subsidence.tif'

        substitute_area_code_on_raster(california_subsidence, california_area_code,
                                       '../Model Run/LOO_Test/InSAR_Data/'
                                       'interim_working_dir/California_area_raster.tif')
        substitute_area_code_on_raster(arizona_subsidence, arizona_area_code,
                                       '../Model Run/LOO_Test/InSAR_Data/'
                                       'interim_working_dir/Arizona_area_raster.tif')
        substitute_area_code_on_raster(quetta_subsidence, quetta_area_code,
                                       '../Model Run/LOO_Test/InSAR_Data/'
                                       'interim_working_dir/Pakistan_Quetta_area_raster.tif')
        substitute_area_code_on_raster(qazvin_subsidence, qazvin_area_code,
                                       '../Model Run/LOO_Test/InSAR_Data/'
                                       'interim_working_dir/Iran_Qazvin_area_raster.tif')
        substitute_area_code_on_raster(hebei_subsidence, hebei_area_code,
                                       '../Model Run/LOO_Test/InSAR_Data/'
                                       'interim_working_dir/China_Hebei_area_raster.tif')
        coastal_raster_area_coded = substitute_area_code_on_raster(coastal_subsidence, coastal_area_code,
                                                                   '../Model Run/LOO_Test/InSAR_Data/'
                                                                   'interim_working_dir/Coastal_raster.tif')

        mosaic_rasters(insar_data_dir, output_dir=insar_data_dir, raster_name='interim_insar_Area_data.tif',
                       ref_raster=refraster, search_by='*area_raster.tif', resolution=0.02)
        insar_arr, merged_insar_file = read_raster_arr_object(os.path.join(insar_data_dir,
                                                                           'interim_insar_Area_data.tif'))
        # adding coastal area coded raster
        row, col = insar_arr.shape[0], insar_arr.shape[1]
        coastal_arr = read_raster_arr_object(coastal_raster_area_coded, get_file=None).ravel()
        insar_arr = insar_arr.ravel()
        insar_arr = np.where(coastal_arr == coastal_area_code, coastal_arr, insar_arr).reshape((row, col))
        merged_insar = os.path.join(insar_data_dir, 'final_insar_Area_data.tif')
        write_raster(insar_arr, merged_insar_file, merged_insar_file.transform, merged_insar)

        # merging georeferenced and insar subsidence data
        ref_arr, ref_file = read_raster_arr_object(referenceraster)
        final_subsidence_arr = np.where(insar_arr > 0, insar_arr, georef_arr)
        subsidence_data = os.path.join(output_dir, final_subsidence_raster)
        write_raster(final_subsidence_arr, ref_file, ref_file.transform, subsidence_data)

        print('Created final area coded subsidence raster')
        pickle.dump(subsidence_areaname_dict, open(os.path.join(output_dir, 'subsidence_areaname_dict.pkl'),
                                                   mode='wb+'))
        return subsidence_data, subsidence_areaname_dict

    else:
        subsidence_data = os.path.join(output_dir, final_subsidence_raster)
        subsidence_areaname_dict = pickle.load(open(os.path.join(output_dir, 'subsidence_areaname_dict.pkl'),
                                                    mode='rb'))
        return subsidence_data, subsidence_areaname_dict


def create_traintest_df_loo_accuracy(input_raster_dir, subsidence_areacode_dict, exclude_columns,
                                     output_dir='../Model Run/LOO_Test/Predictors_csv',
                                     search_by='*.tif', skip_dataframe_creation=False):
    """
    create dataframe from predictor rasters along with area code.

    Parameters:
    input_raster_dir : Input rasters directory.
    subsidence_areacode_dict : subsidence area code dictionary (output from 'combine_georef_insar_subsidence_raster'
                                                                function)
    exclude_columns List of predictors to be excluded from fitted_model training.
    output_dir : Output directory path.
    search_by : Input raster search criteria. Defaults to '*.tif'.
    skip_predictor_subsidence_compilation : Set to True if want to skip processing.

    Returns: predictor_df dataframe created from predictor rasters.
    """
    print('Creating area coded predictors csv...')
    if not skip_dataframe_creation:
        predictors = glob(os.path.join(input_raster_dir, search_by))

        predictor_dict = {}
        for predictor in predictors:
            variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind('.')]
            raster_arr, file = read_raster_arr_object(predictor, get_file=True)
            raster_arr = raster_arr.flatten()
            predictor_dict[variable_name] = raster_arr

        subsidence_area_arr, subsidence_area_file = \
            read_raster_arr_object('../Model Run/LOO_Test/InSAR_Data/final_subsidence_raster/Subsidence_area_coded.tif')

        predictor_dict['Area_code'] = subsidence_area_arr.flatten()
        predictor_df = pd.DataFrame(predictor_dict)

        predictor_name_dict = {'Alexi_ET': 'Alexi ET', 'Aridity_Index': 'Aridity Index', 'ALOS_Landform': 'Landform',
                               'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI', 'Grace': 'Grace',
                               'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                               'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                               'Irrigated_Area_Density': 'Irrigated Area Density (gfsad)',
                               'MODIS_ET': 'MODIS ET (kg/m2)', 'Irrigated_Area_Density2': 'Irrigated Area Density',
                               'MODIS_PET': 'MODIS PET (kg/m2)', 'NDWI': 'NDWI',
                               'Population_Density': 'Population Density', 'SRTM_Slope': '% Slope',
                               'Subsidence': 'Subsidence', 'TRCLM_RET': 'TRCLM RET (mm)',
                               'TRCLM_precp': 'Precipitation (mm)', 'TRCLM_soil': 'Soil moisture (mm)',
                               'TRCLM_Tmax': 'Tmax (°C)', 'TRCLM_Tmin': 'Tmin (°C)', 'MODIS_Land_Use': 'MODIS Land Use',
                               'TRCLM_ET': 'TRCLM ET (mm)'}
        predictor_df = predictor_df.rename(columns=predictor_name_dict)
        predictor_df = predictor_df.drop(columns=exclude_columns)
        predictor_df = predictor_df.dropna(axis=0)
        print(predictor_df['Area_code'].unique().sort())
        area_code = predictor_df['Area_code'].tolist()

        area_name_list = list(subsidence_areacode_dict.keys())
        area_code_list = list(subsidence_areacode_dict.values())

        area_name = []
        for code in area_code:
            position = area_code_list.index(code)
            name = area_name_list[position]
            area_name.append(name)

        predictor_df['Area_name'] = area_name
        makedirs([output_dir])
        output_csv = output_dir + '/' + 'train_test_area_coded_2013_2019.csv'
        predictor_df.to_csv(output_csv, index=False)

        print('Area coded predictors csv created')
        return predictor_df, output_csv
    else:
        output_csv = output_dir + '/' + 'train_test_area_coded_2013_2019.csv'
        predictor_df = pd.read_csv(output_csv)
        return predictor_df, output_csv


def train_test_split_loo_accuracy(predictor_csv, loo_test_area_name, pred_attr='Subsidence',
                                  outdir='../Model Run/LOO_Test/Predictors_csv'):
    """
    Create x_train, y_train, x_test, y_test arrays for machine learning fitted_model.

    Parameters:
    predictor_dataframe_csv : Predictor csv filepath.
    loo_test_area_name : Area name which will be used as test data.
    pred_attr : Prediction attribute column name.  Default set to 'Subsidence'.
    outdir : Output directory where train and test csv will be saved.

    Returns : x_train_csv_path, x_train, y_train, x_test, y_test arrays.
    """
    predictor_df = pd.read_csv(predictor_csv)
    train_df = predictor_df[predictor_df['Area_name'] != loo_test_area_name]
    x_train_df = train_df.drop(columns=['Area_name', 'Area_code', pred_attr])
    y_train_df = train_df[pred_attr]

    test_df = predictor_df[predictor_df['Area_name'] == loo_test_area_name]
    x_test_df = test_df.drop(columns=['Area_name', 'Area_code', pred_attr])
    y_test_df = test_df[[pred_attr]]

    x_train_arr = np.array(x_train_df)
    y_train_arr = np.array(y_train_df)
    x_test_arr = np.array(x_test_df)
    y_test_arr = np.array(y_test_df)

    x_train_df_path = os.path.join(outdir, 'x_train_loo_test.csv')
    x_train_df.to_csv(x_train_df_path, index=False)
    y_train_df.to_csv(os.path.join(outdir, 'y_train_loo_test.csv'), index=False)
    x_test_df.to_csv(os.path.join(outdir, 'x_test_loo_test.csv'), index=False)
    y_test_df.to_csv(os.path.join(outdir, 'y_test_loo_test.csv'), index=False)

    return x_train_df_path, x_train_arr, y_train_arr, x_test_arr, y_test_arr


def build_ml_classifier(predictor_csv, loo_test_area_name, model='RF', random_state=0,
                        n_estimators=300, max_depth=20, max_features=10, min_samples_leaf=1e-05, min_samples_split=2,
                        class_weight='balanced', bootstrap=True, oob_score=True, n_jobs=-1,
                        accuracy_dir=r'../Model Run/Accuracy_score_loo_test',
                        modeldir='../Model Run/LOO_Test/Model_Loo_test'):
    """
    Build ML 'Random Forest' Classifier.

    Parameters:
    predictor_csv : Predictor csv (with filepath) containing all the predictors.
    loo_test_area_name : Area name which will be used as test data.
    fitted_model : Machine learning fitted_model to run.Can only run random forest 'RF' fitted_model.
    random_state : Seed value. Defaults to 0.
    n_estimators : The number of trees in the forest. Defaults to 500.
    max_depth : Depth of each tree. Default set to 20.
    min_samples_leaf : Minimum number of samples required to be at a leaf node. Defaults to 1.
    min_samples_split : Minimum number of samples required to split an internal node. Defaults to 2.
    max_features : The number of features to consider when looking for the best split. Defaults to 'log2'.
    class_weight : To assign class weight. Default set to 'balanced'.
    bootstrap : Whether bootstrap samples are used when building trees. Defaults to True.
    oob_score : Whether to use out-of-bag samples to estimate the generalization accuracy. Defaults to True.
    n_jobs : The number of jobs to run in parallel. Defaults to -1(using all processors).
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    modeldir : Model directory to store/load fitted_model. Default is '../Model Run/Model/Model_Loo_test'.

    Returns: rf_classifier (A fitted random forest fitted_model)
    """

    x_train_csv, x_train, y_train, x_test, y_test = train_test_split_loo_accuracy(predictor_csv, loo_test_area_name,
                                                                                  pred_attr='Subsidence',
                                                                                  outdir='../Model Run/LOO_test/'
                                                                                         'Predictors_csv')

    makedirs([modeldir])
    model_file = os.path.join(modeldir, model)

    if model == 'RF':
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                            max_features=max_features, class_weight=class_weight,
                                            random_state=random_state, bootstrap=bootstrap,
                                            n_jobs=n_jobs, oob_score=oob_score, )

    classifier = classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    pickle.dump(classifier, open(model_file, mode='wb+'))

    classification_accuracy(y_test, y_pred, loo_test_area_name, accuracy_dir)

    return classifier, loo_test_area_name


def classification_accuracy(y_test, y_pred, loo_test_area_name,
                            accuracy_dir=r'../Model Run/LOO_Test/Accuracy_score'):
    """
    Classification accuracy assessment.

    Parameters:
    y_test : y_test array from train_test_split_loo_accuracy() function.
    y_pred : y_pred data from build_ML_classifier() function.
    classifier : ML classifier from build_ML_classifier() function.
    x_train_csv : path of x train csv from 'train_test_split_loo_accuracy' function.
    loo_test_area_name : test area name for which to create confusion matrix.
    area_index : Index that will help saving accuracy score. (don't need to added manually, fitted_model will take from
                 run_loo_accuracy_test function)
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    predictor_importance : Default set to True to plot predictor importance plot.
    predictor_imp_keyword : Keyword to save predictor important plot.

    Returns: Confusion matrix, score and predictor importance graph.
    """
    subsidence_training_area_list = [
        'Arizona', 'Australia_Perth', 'Bangladesh_GBDelta', 'California', 'China_Beijing',
        'China_Hebei', 'China_Shanghai', 'China_Tianjin', 'China_Wuhan', 'China_Xian',
        'China_YellowRiverDelta', 'Coastal', 'Egypt_NileDelta', 'England_London',
        'Indonesia_Bandung', 'Indonesia_Semarang', 'Iran_MarandPlain''Iran_Qazvin',
        'Iran_Tehran', 'Iraq_TigrisEuphratesBasin', 'Italy_PoDelta', 'Italy_VeniceLagoon',
        'Mexico_MexicoCity', 'Nigeria_Lagos', 'Pakistan_Quetta', 'Spain_Murcia',
        'Taiwan_Yunlin', 'Turkey_Bursa' 'Turkey_Karapinar', 'US_Huston', 'Vietnam_HoChiMinh'
    ]
    subsidence_training_area_list = sorted(subsidence_training_area_list)

    makedirs([accuracy_dir])

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_name = loo_test_area_name + '_cmatrix.csv'
    csv = os.path.join(accuracy_dir, cm_name)
    cm_df.to_csv(csv, index=True)

    overall_accuracy = round(accuracy_score(y_test, y_pred), 2)

    # generating classification report
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    del classification_report_dict['accuracy']
    classification_report_df = pd.DataFrame(classification_report_dict)
    classification_report_df.drop(labels='support', inplace=True)
    micro_precision = round(precision_score(y_test, y_pred, average='micro'), 2)
    micro_recall = round(recall_score(y_test, y_pred, average='micro'), 2)
    micro_f1 = round(f1_score(y_test, y_pred, average='micro'), 2)

    classification_report_df['micro avg'] = [micro_precision, micro_recall, micro_f1]
    cols = classification_report_df.columns.tolist()
    if '1.0' not in cols:
        classification_report_df['1.0'] = [np.nan, np.nan, np.nan]
    if '5.0' not in cols:
        classification_report_df['5.0'] = [np.nan, np.nan, np.nan]
    if '10.0' not in cols:
        classification_report_df['10.0'] = [np.nan, np.nan, np.nan]

    classification_report_df = classification_report_df[['1.0', '5.0', '10.0', 'micro avg', 'macro avg',
                                                         'weighted avg']]
    classification_report_df.rename(columns={'1.0': '<1cm/yr', '5.0': '1-5cm/yr', '10.0': '>5cm/yr'}, inplace=True)
    classification_report_df = classification_report_df[classification_report_df.columns].round(2)
    classification_report_csv_name = accuracy_dir + '/' + loo_test_area_name + '_classification_report.csv'
    classification_report_df.to_csv(classification_report_csv_name)

    print('Accuracy Score for {} : {}'.format(loo_test_area_name, overall_accuracy))
    path = accuracy_dir + '/' + 'Accuracy_Reports_Joined' + '/' + 'Accuracy_scores.txt'
    if loo_test_area_name == subsidence_training_area_list[0]:
        os.remove(path)
        txt_object = open(path, 'w+')
    else:
        txt_object = open(path, 'a')
    txt_object.write('Accuracy Score for {} : {} \n'.format(loo_test_area_name, overall_accuracy))
    txt_object.close()

    return overall_accuracy


def create_prediction_raster(predictors_dir, fitted_model, yearlist=[2013, 2019], search_by='*.tif',
                             continent_search_by='*continent.shp',
                             continent_shapes_dir='../Data/Reference_rasters_shapes/continent_extents',
                             prediction_raster_dir='../Model Run/LOO_Test/Prediction_rasters',
                             exclude_columns=(), pred_attr='Subsidence',
                             prediction_raster_keyword='RF', predictor_csv_exists=False):
    """
    Create predicted raster from random forest fitted_model.

    Parameters:
    predictors_dir : Predictor rasters' directory.
    fitted_model : A fitted_model obtained from random_forest_classifier function.
    yearlist : List of years for the prediction.
    search_by : Predictor rasters search criteria. Defaults to '*.tif'.
    continent_search_by : Continent shapefile search criteria. Defaults to '*continent.tif'.
    continent_shapes_dir : Directory path of continent shapefiles.
    prediction_raster_dir : Output directory of prediction raster.
    exclude_columns : Predictor rasters' name that will be excluded from the fitted_model. Defaults to ().
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    prediction_raster_keyword : Keyword added to final prediction raster name.
    predictor_csv_exists : Set to True if predictor csv for each continent exists. Default set to False to create
                           create new predcitor csv (also needed if predictor combinations are changed).

    Returns: Subsidence prediction raster and
             Subsidence prediction probability raster (if prediction_probability=True).
    """
    predictor_rasters = glob(os.path.join(predictors_dir, search_by))
    continent_shapes = glob(os.path.join(continent_shapes_dir, continent_search_by))
    drop_columns = list(exclude_columns) + [pred_attr]

    continent_prediction_raster_dir = os.path.join(prediction_raster_dir, 'continent_prediction_rasters_'
                                                   + str(yearlist[0]) + '_' + str(yearlist[1]))
    makedirs([prediction_raster_dir])
    makedirs([continent_prediction_raster_dir])

    predictor_name_dict = {'Alexi_ET': 'Alexi ET', 'Aridity_Index': 'Aridity Index', 'ALOS_Landform': 'Landform',
                           'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI', 'Grace': 'Grace',
                           'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                           'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                           'Irrigated_Area_Density': 'Irrigated Area Density (gfsad)',
                           'MODIS_ET': 'MODIS ET (kg/m2)', 'Irrigated_Area_Density2': 'Irrigated Area Density',
                           'MODIS_PET': 'MODIS PET (kg/m2)', 'NDWI': 'NDWI',
                           'Population_Density': 'Population Density', 'SRTM_Slope': '% Slope',
                           'Subsidence': 'Subsidence', 'TRCLM_RET': 'TRCLM RET (mm)',
                           'TRCLM_precp': 'Precipitation (mm)', 'TRCLM_soil': 'Soil moisture (mm)',
                           'TRCLM_Tmax': 'Tmax (°C)', 'TRCLM_Tmin': 'Tmin (°C)', 'MODIS_Land_Use': 'MODIS Land Use',
                           'TRCLM_ET': 'TRCLM ET (mm)'}

    for continent in continent_shapes:
        continent_name = continent[continent.rfind(os.sep) + 1:continent.rfind('_')]

        predictor_csv_dir = '../Model Run/LOO_Test/Predictors_csv/continent_csv'
        makedirs([predictor_csv_dir])
        predictor_csv_name = continent_name + '_predictors.csv'
        predictor_csv = os.path.join(predictor_csv_dir, predictor_csv_name)

        nan_pos_dict_name = predictor_csv_dir + '/nanpos_' + continent_name  # name to save nan_position_dict

        clipped_dir = '../Model Run/LOO_Test/Predictors_csv/Predictors_2013_2019'
        makedirs([clipped_dir])
        clipped_predictor_dir = os.path.join(clipped_dir, continent_name + '_predictors_' + str(yearlist[0]) +
                                             '_' + str(yearlist[1]))
        if not predictor_csv_exists:
            predictor_dict = {}
            nan_position_dict = {}
            raster_shape = None

            for predictor in predictor_rasters:
                variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind('.')]
                variable_name = predictor_name_dict[variable_name]

                if variable_name not in drop_columns:
                    raster_arr, raster_file = clip_resample_raster_cutline(predictor, clipped_predictor_dir, continent,
                                                                           naming_from_both=False)
                    raster_shape = raster_arr.shape
                    raster_arr = raster_arr.reshape(raster_shape[0] * raster_shape[1])
                    nan_position_dict[variable_name] = np.isnan(raster_arr)
                    raster_arr[nan_position_dict[variable_name]] = 0
                    predictor_dict[variable_name] = raster_arr

            pickle.dump(nan_position_dict, open(nan_pos_dict_name, mode='wb+'))

            predictor_df = pd.DataFrame(predictor_dict)
            predictor_df = predictor_df.dropna(axis=0)
            predictor_df.to_csv(predictor_csv, index=False)

        else:
            predictor_df = pd.read_csv(predictor_csv)

            nan_position_dict = pickle.load(open(nan_pos_dict_name, mode='rb'))

            raster_arr, raster_file = clip_resample_raster_cutline(predictor_rasters[1], clipped_predictor_dir,
                                                                   continent, naming_from_both=False)
            raster_shape = raster_arr.shape

        x = predictor_df.values
        y_pred = fitted_model.predict(x)

        for nan_pos in nan_position_dict.values():
            y_pred[nan_pos] = raster_file.nodata
        y_pred_arr = y_pred.reshape(raster_shape)

        prediction_raster_name = continent_name + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
        predicted_raster = os.path.join(continent_prediction_raster_dir, prediction_raster_name)
        write_raster(raster_arr=y_pred_arr, raster_file=raster_file, transform=raster_file.transform,
                     outfile_path=predicted_raster)
        print('Prediction raster created for', continent_name)

    raster_name = prediction_raster_keyword + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
    mosaic_rasters(continent_prediction_raster_dir, prediction_raster_dir, raster_name, search_by='*prediction*.tif')
    print('Global prediction raster created')


def run_loo_accuracy_test(predictor_dataframe_csv, exclude_predictors_list, n_estimators=300, max_depth=20,
                          max_features=10, min_samples_leaf=1e-05, min_samples_split=2, class_weight='balanced',
                          predictor_raster_directory='../Model Run/Predictors_2013_2019',
                          skip_create_prediction_raster=False, predictor_csv_exists=False):
    """
    Driver code for running Loo Accuracy Test.

    Parameters:
    predictor_dataframe_csv : filepath of predictor csv.
    exclude_predictors_list : List of predictors to exclude for training the fitted_model.
    n_estimators : The number of trees in the forest. Defaults to 500.
    max_depth : Depth of each tree. Default set to 20.
    min_samples_leaf : Minimum number of samples required to be at a leaf node. Defaults to 1.
    min_samples_split : Minimum number of samples required to split an internal node. Defaults to 2.
    max_features : The number of features to consider when looking for the best split. Defaults to 'log2'.
    class_weight : To assign class weight. Default set to 'balanced'.
    predictor_raster_directory : Original predictor raster directory. Default set to
                                 '../Model Run/Predictors_2013_2019'.
    skip_create_prediction_raster : Set to True if want to skip prediction raster creation.
    predictor_csv_exists : Set to True if predictor csv for each continent exists. Default set to False to create
                           create new predictor csv (also needed if predictor combinations are changed).

    Returns : Classification reports and confusion matrix for individual fitted_model training, Overall accuracy result for
              each fitted_model as a single text file,
              prediction rasters for each fitted_model (if skip_create_prediction_raster=False)
    """
    subsidence_training_area_list = [
        'Arizona', 'Australia_Perth', 'Bangladesh_GBDelta', 'California', 'China_Beijing',
        'China_Hebei', 'China_Shanghai', 'China_Tianjin', 'China_Wuhan', 'China_Xian',
        'China_YellowRiverDelta', 'Coastal', 'Egypt_NileDelta', 'England_London',
        'Indonesia_Bandung', 'Indonesia_Semarang', 'Iran_MarandPlain', 'Iran_Qazvin',
        'Iran_Tehran', 'Iraq_TigrisEuphratesBasin', 'Italy_PoDelta', 'Italy_VeniceLagoon',
        'Mexico_MexicoCity', 'Nigeria_Lagos', 'Pakistan_Quetta', 'Spain_Murcia',
        'Taiwan_Yunlin', 'Turkey_Bursa', 'Turkey_Karapinar', 'US_Huston', 'Vietnam_HoChiMinh'
    ]
    subsidence_training_area_list = sorted(subsidence_training_area_list)

    for area in subsidence_training_area_list:
        print('Running without', area)
        trained_rf, loo_area = build_ml_classifier(predictor_dataframe_csv, area, model='RF', random_state=0,
                                                   n_estimators=n_estimators, max_depth=max_depth,
                                                   max_features=max_features, min_samples_leaf=min_samples_leaf,
                                                   min_samples_split=min_samples_split, class_weight=class_weight,
                                                   bootstrap=True, oob_score=True, n_jobs=-1,
                                                   accuracy_dir=r'../Model Run/LOO_Test/Accuracy_score',
                                                   modeldir='../Model Run/LOO_Test/Model_Loo_test')

        if not skip_create_prediction_raster:
            create_prediction_raster(predictor_raster_directory, trained_rf, yearlist=[2013, 2019], search_by='*.tif',
                                     continent_search_by='*continent.shp',
                                     continent_shapes_dir='../Data/Reference_rasters_shapes/continent_extents',
                                     prediction_raster_dir='../Model Run/LOO_Test/Prediction_rasters',
                                     exclude_columns=exclude_predictors_list, pred_attr='Subsidence',
                                     prediction_raster_keyword='Trained_without_' + area,
                                     predictor_csv_exists=predictor_csv_exists)


def difference_from_model_prediction(original_model_prediction_raster,
                                     loo_test_prediction_dir='../Model Run/LOO_Test/Prediction_rasters'):
    """
    Find mismatch % between original fitted_model predictions and loo test predictions.

    Parameters:
    original_model_prediction_raster : Filepath of original fitted_model prediction raster.
    loo_test_prediction_dir : directory path of loo test predictions.

    Returns : A text file containing mismatch % values.
    """
    pred_arr = read_raster_arr_object(original_model_prediction_raster, get_file=False).ravel()
    pred_arr = pred_arr[~np.isnan(pred_arr)]
    loo_rasters = glob(os.path.join(loo_test_prediction_dir, '*.tif'))

    i = 0
    for loo_prediction in loo_rasters:

        loo_arr = read_raster_arr_object(loo_prediction, get_file=False).ravel()
        loo_arr = loo_arr[~np.isnan(loo_arr)]
        name = loo_prediction[loo_prediction.rfind(os.sep) + 1: loo_prediction.find('prediction') - 1]
        mismatch = round((np.sum(pred_arr != loo_arr) * 100 / len(pred_arr)), 2)
        txt_path = '../Model Run/LOO_Test/Accuracy_score/Accuracy_Reports_Joined/mismatch.txt'
        if i == 0:
            os.remove(txt_path)
            text = open(txt_path, 'w+')
            text.write('{} mismatch : {} % \n'.format(name, mismatch))
            text.close()
        else:
            text = open(txt_path, 'a+')
            text.write('{} mismatch : {} % \n'.format(name, mismatch))
            text.close()
        i += 1
        print('{} mismatch : {} % \n'.format(name, mismatch))


def concat_classification_reports(classification_csv_dir='../Model Run/LOO_Test/Accuracy_score'):
    """
    Merge classification reports from all fitted_model runs.

    Parameters:
    classification_csv_dir : Directory of individual classification reports. Default set to
                             '../Model Run/LOO_Test/Accuracy_score'

    Returns : A joined classification report.
    """
    reports = glob(classification_csv_dir + '/' + '*classification_report*.csv')
    report_df = [pd.read_csv(report) for report in reports]

    area_name = []
    for report in reports:
        area = report[report.rfind(os.sep) + 1:report.find('classification') - 1]
        area_name.append(area)
    merged_reports_df = pd.concat(report_df, keys=area_name, ignore_index=False)
    merged_reports_df = merged_reports_df.reset_index(level=1, drop=True)
    merged_reports_df = merged_reports_df.rename(columns={'Unnamed: 0': 'metrics'})
    merged_reports_df = merged_reports_df[['metrics', '<1cm/yr', '1-5cm/yr', '>5cm/yr', 'micro avg', 'macro avg',
                                           'weighted avg']]
    merged_reports_df.to_csv('../Model Run/LOO_Test/Accuracy_score/Accuracy_Reports_Joined/'
                             'Classification_reports_joined.csv')


# LOO Accuracy Test Run
run_loo_test = True

if run_loo_test:
    subsidence_raster, areaname_dict = \
        combine_georef_insar_subsidence_raster(already_prepared=True,  # #
                                               skip_polygon_processing=True)  # #

    predictor_raster_dir = '../Model Run/Predictors_2013_2019'
    exclude_predictors = ['Alexi ET', 'Grace', 'MODIS ET (kg/m2)', 'Irrigated Area Density (gfsad)',
                          'GW Irrigation Density giam', 'Landform', 'MODIS PET (kg/m2)', 'MODIS Land Use']

    df, predictor_csv = create_traintest_df_loo_accuracy(predictor_raster_dir, areaname_dict, exclude_predictors,
                                                         skip_dataframe_creation=True)  # #

    run_loo_accuracy_test(predictor_dataframe_csv=predictor_csv, exclude_predictors_list=exclude_predictors,
                          n_estimators=200, max_depth=20, max_features=5, min_samples_leaf=1e-05,
                          class_weight='balanced',
                          predictor_raster_directory='../Model Run/Predictors_2013_2019',
                          skip_create_prediction_raster=False,  # #
                          predictor_csv_exists=False)  # #

    concat_classification_reports(classification_csv_dir='../Model Run/LOO_Test/Accuracy_score')

# Perform Mismatch estimation

# Set run_loo_test=False and mismatch_estimation=True to perform mismatch estimation. Input is origin fitted_model
# prediction raster which will work as the baseline fitted_model to compare loo test prediction models.

mismatch_estimation = True

original_model_prediction = '../Model Run/Prediction_rasters/RF112_prediction_2013_2019.tif'

if mismatch_estimation:
    difference_from_model_prediction(original_model_prediction,
                                     loo_test_prediction_dir='../Model Run/LOO_Test/Prediction_rasters')
