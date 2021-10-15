# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import os
import pickle
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from System_operations import makedirs
from Raster_operations import shapefile_to_raster, mosaic_rasters, mosaic_two_rasters, read_raster_arr_object, \
    write_raster

referenceraster2 = '../Data/Reference_rasters_shapes/Global_continents_ref_raster_002.tif'


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
    global gdf

    if not skip_polygon_processing:
        subsidence_polygons = glob(os.path.join(input_polygons_dir, search_criteria))

        sep = joined_subsidence_polygons.rfind(os.sep)
        makedirs([joined_subsidence_polygons[:sep]])  # creating directory for the  prepare_subsidence_raster function

        for each in range(1, len(subsidence_polygons) + 1):
            if each == 1:
                gdf = gpd.read_file(subsidence_polygons[each - 1])

            gdf_new = gpd.read_file(subsidence_polygons[each - 1])
            add_to_gdf = gdf.append(gdf_new, ignore_index=True)
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

        pickle.dump(polygon_area_name_dict, open('../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                 'polygon_area_name_dict.pkl', mode='wb+'))

        return joined_subsidence_polygons, polygon_area_name_dict

    else:
        joined_subsidence_polygons = '../InSAR_Data/Resampled_subsidence_data/' \
                                     'LOO_test_dir/georef_subsidence_polygons.shp'
        polygon_area_name_dict = pickle.load(open('../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                  'polygon_area_name_dict.pkl', mode='rb'))

        return joined_subsidence_polygons, polygon_area_name_dict


def substitute_area_code_on_raster(input_raster, value_to_substitute, output_raster):
    """
    Substitute raster values with area code for InSAR produced subsidence rasters (California, Arizona, Quetta etc.)

    Parameters:
    input_raster : Input subsidence raster filepath.
    value_to_substitute : Area code that will substitute raster values.
    output_raster : Filepath of output raster.

    Returns : Raster with values subsituted with area code.
    """
    raster_arr, raster_file = read_raster_arr_object(input_raster)

    raster_arr = np.where(np.isnan(raster_arr), raster_arr, value_to_substitute)

    area_coded_raster = write_raster(raster_arr, raster_file, raster_file.transform, output_raster)

    return area_coded_raster


def combine_georef_insar_subsidence_raster(input_polygons_dir='../InSAR_Data/Georeferenced_subsidence_data',
                                           joined_subsidence_polygon='../InSAR_Data/Resampled_subsidence_data/'
                                                                     'LOO_test_dir/georef_subsidence_polygons.shp',
                                           insar_data_dir='../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                          'interim_working_dir',
                                           interim_dir='../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                       'interim_working_dir',
                                           output_dir='../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                                      'final_subsidence_raster',
                                           skip_polygon_processing=False,
                                           area_code_column='Area_code',
                                           final_subsidence_raster='Subsidence_area_coded.tif',
                                           polygon_search_criteria='*Subsidence*.shp', already_prepared=False,
                                           refraster=referenceraster2):
    """
    Prepare area coded subsidence raster for training data by joining georeferenced polygons and insar data.

    Parameters:
    input_polygons_dir : Input subsidence polygons' directory.
    joined_subsidence_polygons : Output joined subsidence polygon filepath.
    insar_data_dir : InSAR data directory.
    interim_dir : Intermediate working directory for storing interdim data.
    output_dir : Output raster directory.
    skip_polygon_processing : Set to True if polygon merge is not required.
    final_subsidence_raster : Final subsidence raster including georeferenced and insar data.
    polygon_search_criteria : Input subsidence polygon search criteria.
    insar_search_criteria : InSAR data search criteria.
    already_prepared : Set to True if subsidence raster is already prepared.
    refraster : Global Reference raster.

    Returns : Final subsidence raster to be used as training data.
    """

    global subsidence_areaname_dict
    if not already_prepared:
        makedirs([interim_dir, output_dir])

        print('Processing area coded subsidence polygons...')
        subsidence_polygons, subsidence_areaname_dict = \
            combine_georeferenced_subsidence_polygons(input_polygons_dir, joined_subsidence_polygon,
                                                      polygon_search_criteria, skip_polygon_processing)

        print('Processed area coded subsidence polygons')
        subsidence_raster_area_coded = shapefile_to_raster(subsidence_polygons, interim_dir,
                                                           raster_name='interim_georef_subsidence_raster_areacode.tif',
                                                           burn_attr=True, attribute=area_code_column,
                                                           ref_raster=refraster, alltouched=False)

        print('Processing area coded InSAR data...')
        georef_subsidence_gdf = gpd.read_file(joined_subsidence_polygon)
        num_of_georef_subsidence = len(georef_subsidence_gdf)

        california_area_code = num_of_georef_subsidence + 1
        arizona_area_code = california_area_code + 1
        quetta_area_code = arizona_area_code + 1
        subsidence_areaname_dict['California'] = california_area_code
        subsidence_areaname_dict['Arizona'] = arizona_area_code
        subsidence_areaname_dict['Quetta'] = quetta_area_code

        california_subsidence = '../InSAR_Data/Resampled_subsidence_data/California_reclass_resampled.tif'
        arizona_subsidence = '../InSAR_Data/Resampled_subsidence_data/Arizona_reclass_resampled.tif'
        quetta_subsidence = '../InSAR_Data/Resampled_subsidence_data/Quetta_reclass_resampled.tif'

        substitute_area_code_on_raster(california_subsidence, california_area_code,
                                       '../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                       'interim_working_dir/California_area_raster.tif')
        substitute_area_code_on_raster(arizona_subsidence, arizona_area_code,
                                       '../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                       'interim_working_dir/Arizona_area_raster.tif')
        substitute_area_code_on_raster(quetta_subsidence, quetta_area_code,
                                       '../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                       'interim_working_dir/Quetta_area_raster.tif')

        insar_arr, merged_insar = mosaic_rasters(insar_data_dir, output_dir=insar_data_dir,
                                                 raster_name='joined_insar_Area_data.tif',
                                                 ref_raster=refraster, search_by='*area*.tif', resolution=0.02)

        final_subsidence_arr, subsidence_data = mosaic_two_rasters(merged_insar, subsidence_raster_area_coded,
                                                                   output_dir, final_subsidence_raster, resolution=0.02)
        print('Created final area coded subsidence raster')
        pickle.dump(subsidence_areaname_dict, open(os.path.join(output_dir, 'subsidence_areaname_dict.pkl'),
                                                   mode='wb+'))

        return subsidence_data, subsidence_areaname_dict

    else:
        subsidence_data = os.path.join(output_dir, final_subsidence_raster)
        subsidence_areaname_dict = pickle.load(open(os.path.join(output_dir, 'subsidence_areaname_dict.pkl'),
                                                    mode='rb'))
        return subsidence_data, subsidence_areaname_dict


def create_dataframe_for_loo_accuracy(input_raster_dir, output_csv, subsidence_areacode_dict,
                                      search_by='*.tif', skip_dataframe_creation=False):
    """
    create dataframe from predictor rasters along with area code.

    Parameters:
    input_raster_dir : Input rasters directory.
    output_csv : Output csv file with filepath.
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
            read_raster_arr_object('../InSAR_Data/Resampled_subsidence_data/LOO_test_dir/'
                                   'final_subsidence_raster/Subsidence_area_coded.tif')
        predictor_dict['Area_code'] = subsidence_area_arr.flatten()

        predictor_df = pd.DataFrame(predictor_dict)
        predictor_df = predictor_df.dropna(axis=0)

        area_code = predictor_df['Area_code'].tolist()

        area_name_list = list(subsidence_areacode_dict.keys())
        area_code_list = list(subsidence_areacode_dict.values())

        area_name = []
        for code in area_code:
            position = area_code_list.index(code)
            name = area_name_list[position]
            area_name.append(name)

        predictor_df['Area_name'] = area_name
        predictor_df.to_csv(output_csv, index=False)

        print('Area coded predictors csv created')
        return predictor_df, output_csv
    else:
        predictor_df = pd.read_csv(output_csv)
        return predictor_df, output_csv


def create_train_test_data(predictor_csv, loo_test_area_name, exclude_columns, pred_attr='Subsidence',
                           outdir='../Model Run/Predictors_csv/Predictors_csv_Loo_test'):
    """
    Create x_train, y_train, x_test, y_test arrays for machine learning model.

    Parameters:
    predictor_csv : Predictor csv filepath.
    loo_test_area_name : Area name which will be used as test data.
    pred_attr : Prediction attribute column name.  Default set to 'Subsidence'.
    outdir : Output directory where train and test csv will be saved.

    Returns : x_train, y_train, x_test, y_test arrays.
    """
    predictor_df = pd.read_csv(predictor_csv)

    train_df = predictor_df[predictor_df['Area_name'] != loo_test_area_name]
    drop_columns = exclude_columns + ['Area_name', 'Area_code', pred_attr]
    print('Dropping Columns-', exclude_columns)
    x_train_df = train_df.drop(columns=drop_columns)
    y_train_df = train_df[pred_attr]
    print('Predictors:', x_train_df.columns)

    test_df = predictor_df[predictor_df['Area_name'] == loo_test_area_name]
    x_test_df = test_df.drop(columns=drop_columns)
    y_test_df = test_df[[pred_attr]]

    x_train = np.array(x_train_df)
    y_train = np.array(y_train_df)
    x_test = np.array(x_test_df)
    y_test = np.array(y_test_df)

    x_train_df.to_csv(os.path.join(outdir, 'x_train_loo_test.csv'), index=False)
    y_train_df.to_csv(os.path.join(outdir, 'y_train_loo_test.csv'), index=False)
    x_test_df.to_csv(os.path.join(outdir, 'x_test_loo_test.csv'), index=False)
    y_test_df.to_csv(os.path.join(outdir, 'y_test_loo_test.csv'), index=False)

    return x_train, y_train, x_test, y_test


def build_ml_classifier(predictor_csv, loo_test_area_name, exclude_columns=(), model='RF', load_model=False,
                        random_state=0, n_estimators=500, bootstrap=True, oob_score=True, n_jobs=-1,
                        max_features='auto',
                        accuracy=True, save=True, accuracy_dir=r'../Model Run/Accuracy_score', cm_name='cmatrix.csv',
                        predictor_importance=False,
                        plot_pdp=False, modeldir='../Model Run/Model/Model_Loo_test'):
    """
    Build Machine Learning Classifier. Can run 'Random Forest', 'Extra Trees Classifier' and 'XGBClassifier'.

    Parameters:
    predictor_csv : Predictor csv (with filepath) containing all the predictors.
    exclude_columns : Tuple of columns not included in training the model.
    model : Machine learning model to run. Choose from 'RF'/ETC'/'XGBC'. Default set to 'RF'.
    load_model : Set True to load existing model. Default set to False for new model creation.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    shuffle : Whether or not to shuffle data before spliting. Defaults to True.
    output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    n_estimators : The number of trees in the forest.. Defaults to 500.
    bootstrap : Whether bootstrap samples are used when building trees. Defaults to True.
    oob_score : Whether to use out-of-bag samples to estimate the generalization accuracy. Defaults to True.
    n_jobs : The number of jobs to run in parallel. Defaults to -1(using all processors).
    max_features : The number of features to consider when looking for the best split. Defaults to None.
    multiclass : If multiclass classification, set True for getting model performance. Defaults to False.
    save : Set True to save confusion matrix as csv. Defaults to False.
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    cm_name : Confusion matrix name. Defaults to 'cmatrix.csv'.
    predictor_importance : Set True if predictor importance plot is needed. Defaults to False.
    predictor_imp_keyword : Keyword to save predictor important plot.
    modeldir : Model directory to store/load model. Default is '../Model Run/Model/Model_Loo_test'.
    # ADD PDP VARIABLES

    Returns: rf_classifier (A fitted random forest model)
    """

    x_train, y_train, x_test, y_test = create_train_test_data(predictor_csv, loo_test_area_name, exclude_columns,
                                                              pred_attr='Subsidence',
                                                              outdir='../Model Run/Predictors_csv_Loo_test')

    makedirs([modeldir])
    model_file = os.path.join(modeldir, model)

    if not load_model:
        if model == 'RF':
            classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                                bootstrap=bootstrap,
                                                n_jobs=n_jobs, oob_score=oob_score, max_features=max_features)

        classifier = classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        pickle.dump(classifier, open(model_file, mode='wb+'))

    else:
        classifier = pickle.load(open(model_file, mode='rb'))

    predictor_imp_keyword = 'RF_teston_' + loo_test_area_name + '_'

    if accuracy:
        classification_accuracy(y_test, y_pred, classifier, x_train, save, accuracy_dir, cm_name,
                                predictor_importance, predictor_imp_keyword)
    if plot_pdp:
        pdp_plot(classifier, x_train, accuracy_dir, plot_save_keyword=predictor_imp_keyword)

    return classifier


def classification_accuracy(y_test, y_pred, classifier, x_train, save=True,
                            accuracy_dir=r'../Model Run/LOO_Test_Accuracy_score', cm_name='cmatrix.csv',
                            predictor_importance=False, predictor_imp_keyword='RF'):
    """
    Classification accuracy assessment.

    Parameters:
    y_test : y_test data from split_train_test_ratio() function.
    y_pred : y_pred data from build_ML_classifier() function.
    classifier : ML classifier from build_ML_classifier() function.
    x_train : x train from 'split_train_test_ratio' function.
    save : Set True to save confusion matrix.
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    cm_name : Confusion matrix name. Defaults to 'cmatrix.csv'.
    predictor_importance : Set True if predictor importance plot is needed. Defaults to False.
    predictor_imp_keyword : Keyword to save predictor important plot.

    Returns: Confusion matrix, score and predictor importance graph.
    """
    column_labels = [np.array(['predictied', 'predictied', 'predictied']),
                     np.array(['<1cm subsidence', '1-5 cm Subsidence', '>5cm Subsidence'])]
    index_labels = [np.array(['Actual', 'Actual', 'Actual']),
                    np.array(['<1cm subsidence', '1-5 cm Subsidence', '>5cm Subsidence'])]
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=column_labels, index=index_labels)

    if save:
        makedirs([accuracy_dir])
        csv = os.path.join(accuracy_dir, cm_name)
        cm_df.to_csv(csv, index=True)

    print(cm_df, '\n')
    print('Recall Score {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Precision Score {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Accuracy Score {:.2f}'.format(accuracy_score(y_test, y_pred)))

    if predictor_importance:
        x_train_df = pd.DataFrame(x_train)
        col_labels = np.array(x_train_df.columns)
        importance = np.array(classifier.feature_importances_)
        imp_dict = {'feature_names': col_labels, 'feature_importance': importance}
        imp_df = pd.DataFrame(imp_dict)
        imp_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=imp_df['feature_importance'], y=imp_df['feature_names'])
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        plt.tight_layout()
        plt.savefig((accuracy_dir + '/' + predictor_imp_keyword + '_pred_importance.png'))
        print('Feature importance plot saved')


def pdp_plot(classifier, x_train, output_dir, title1='PDP less 1cm Subsidence.png',
             title2='PDP 1 to 5cm Subsidence.png', title3='PDP greater 5cm Subsidence.png', plot_save_keyword='RF'
             ):
    plot_names = x_train.columns.tolist()
    feature_indices = range(len(plot_names))

    # Class <1cm
    plot_partial_dependence(classifier, x_train, target=1, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            grid_resolution=20)
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.suptitle('Partial Dependence Plot for <1cm Subsidence')
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + title1), dpi=300, bbox_inches='tight')
    print('pdp for <1cm saved')

    # Class 1-5cm
    plot_partial_dependence(classifier, x_train, target=5, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            grid_resolution=20)
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.suptitle('Partial Dependence Plot for 1-5cm Subsidence')
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + title2), dpi=300, bbox_inches='tight')
    print('pdp for 1-5cm saved')

    # Class >5cm
    plot_partial_dependence(classifier, x_train, target=10, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            grid_resolution=20)
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.suptitle('Partial Dependence Plot for >5cm Subsidence')
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + title3), dpi=300, bbox_inches='tight')
    print('pdp for >5cm saved')

Subsidence_training_area_list = ['England_London', 'Italy_VeniceLagoon', 'Italy_PoDelta', 'California', 'China_Beijing',
                                 'Iran_MarandPlain', 'Spain_Murcia', 'China_YellowRiverDelta',
                                 'Iraq_TigrisEuphratesBasin', 'China_Xian', 'Arizona', 'Egypt_NileDelta',
                                 'China_Shanghai', 'China_Wuhan', 'Quetta', 'Bangladesh_GBDelta', 'Taiwan_Yunlin',
                                 'Mexico_MexicoCity', 'Vietnam_HoChiMinh', 'Nigeria_Lagos', 'Indonesia_Semarang',
                                 'Indonesia_Bandung', 'Australia_Perth']



x_train, y_train, x_test, y_test = create_train_test_data(
    predictor_csv='../Model Run/Predictors_csv/train_test_area_coded_2013_2019.csv',
    loo_test_area_name='England_London')

# subsidence_raster, areaname_dict = combine_georef_insar_subsidence_raster(already_prepared=True,
#                                                                           skip_polygon_processing=True)
#
# predictor_raster_dir = '../Model Run/Predictors_2013_2019'
# train_test_csv = '../Model Run/Predictors_csv/train_test_area_coded_2013_2019.csv'
# create_dataframe_for_loo_accuracy(predictor_raster_dir, train_test_csv, areaname_dict)
