import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence
from Raster_operations import *
from System_operations import *

referenceraster2 = r'../Data/Reference_rasters_shapes/Global_continents_ref_raster_002.tif'


def create_dataframe(input_raster_dir, output_csv, search_by='*.tif', skip_processing=False):
    """
    create dataframe from predictor rasters.

    Parameters:
    input_raster_dir : Input rasters directory.
    output_csv : Output csv file with filepath.
    search_by : Input raster search criteria. Defaults to '*.tif'.
    skip_processing : Set to True if want to skip processing.

    Returns: predictor_df dataframe created from predictor rasters.
    """
    print('Creating Predictors csv...')
    if not skip_processing:
        predictors = glob(os.path.join(input_raster_dir, search_by))

        predictor_dict = {}
        for predictor in predictors:
            variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind(".")]
            raster_arr, file = read_raster_arr_object(predictor, get_file=True)
            raster_arr = raster_arr.flatten()
            predictor_dict[variable_name] = raster_arr

        predictor_df = pd.DataFrame(predictor_dict)
        predictor_df = predictor_df.dropna(axis=0)
        predictor_df.to_csv(output_csv, index=False)

        print('Predictors csv created')
        return predictor_df
    else:
        predictor_df = pd.read_csv(output_csv)
        return predictor_df


def split_train_test_ratio(predictor_csv, exclude_columns=(), pred_attr='Subsidence', test_size=0.3, random_state=0,
                           shuffle=True, outdir=None):
    """
    Split dataset into train and test data based on a ratio

    parameters:
    input_csv : Input csv (with filepath) containing all the predictors.
    exclude_columns : Tuple of columns not included in training the model.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    shuffle : Whether or not to shuffle data before spliting. Defaults to True.
    output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.

    Returns: X_train, X_test, y_train, y_test
    """
    input_df = pd.read_csv(predictor_csv)
    drop_columns = list(exclude_columns) + [pred_attr]
    x = input_df.drop(columns=drop_columns)
    y = input_df[pred_attr]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        shuffle=shuffle)

    if outdir:
        x_train_df = pd.DataFrame(x_train)
        x_train_df.to_csv(os.path.join(outdir, 'X_train.csv'), index=False)

        y_train_df = pd.DataFrame(y_train)
        y_train_df.to_csv(os.path.join(outdir, 'y_train.csv'), index=False)

        x_test_df = pd.DataFrame(x_test)
        x_test_df.to_csv(os.path.join(outdir, 'X_test.csv'), index=False)

        y_test_df = pd.DataFrame(y_test)
        y_test_df.to_csv(os.path.join(outdir, 'y_test.csv'), index=False)

    return x_train, x_test, y_train, y_test


def build_ml_classifier(predictor_csv, modeldir, exclude_columns=(), model='RF', load_model=False,
                        pred_attr='Subsidence', test_size=0.3, random_state=0, shuffle=True, output_dir=None,
                        n_estimators=500, bootstrap=True, oob_score=True, n_jobs=-1, max_features='auto',
                        accuracy=True, save=True, accuracy_dir=r'../Model Run/Accuracy_score', cm_name='cmatrix.csv',
                        predictor_importance=False, predictor_imp_keyword='RF'):
    """
    Build Machine Learning Classifier. Can run 'Random Forest', 'Extra Trees Classifier' and 'XGBClassifier'.

    Parameters:
    predictor_csv : Predictor csv (with filepath) containing all the predictors.
    modeldir : Model directory to store/load model.
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
    # ADD PDP VARIABLES

    Returns: rf_classifier (A fitted random forest model)
    """

    # Spliting Training and Tesing Data
    x_train, x_test, y_train, y_test = split_train_test_ratio(predictor_csv=predictor_csv,
                                                              exclude_columns=exclude_columns, pred_attr=pred_attr,
                                                              test_size=test_size, random_state=random_state,
                                                              shuffle=shuffle, outdir=output_dir)
    # Making directory for model
    makedirs([modeldir])
    model_file = os.path.join(modeldir, model)

    # Machinle Learning Models
    if not load_model:
        if model == 'RF':
            classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                                bootstrap=bootstrap,
                                                n_jobs=n_jobs, oob_score=oob_score, max_features=max_features)

        if model == 'ETC':
            classifier = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state,
                                              bootstrap=bootstrap,
                                              n_jobs=n_jobs, oob_score=oob_score, max_features=max_features)

        if model == 'XGBC':
            classifier = XGBClassifier(n_estimators=n_estimators, random_state=random_state, learning_rate=0.0098,
                                       grow_policy='lossguide', booster='gbtree', objective='multi:softmax',
                                       subsample=0.75, n_jobs=n_jobs,
                                       colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1)
        classifier = classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        pickle.dump(classifier, open(model_file, mode='wb+'))

    else:
        classifier = pickle.load(open(model_file, mode='rb'))

    if accuracy:
        classification_accuracy(y_test, y_pred, classifier, x_train, save, accuracy_dir, cm_name,
                                predictor_importance, predictor_imp_keyword)
    # if pdp:
    #     pdp_plot(classifier=classifier, X_train=X_train, output_dir=plot_save)

    return classifier


def classification_accuracy(y_test, y_pred, classifier, x_train, save=True,
                            accuracy_dir=r'../Model Run/Accuracy_score', cm_name='cmatrix.csv',
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
    labels = ['<1cm subsidence', '1-5 cm Subsidence', '>5cm Subsidence']
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=labels, index=labels)

    if save:
        makedirs([accuracy_dir])
        csv = os.path.join(accuracy_dir, cm_name)
        cm_df.to_csv(csv, index=False)

    print(cm_df, '\n')
    print('Recall Score {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Precision Score {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Accuracy Score {:.2f}'.format(accuracy_score(y_test, y_pred)))

    if predictor_importance:
        x_train_df = pd.DataFrame(x_train)
        col_labels = np.array(x_train_df.columns)
        importance = np.array(classifier.feature_importances_)
        imp_dict = {'feature_names':col_labels, 'feature_importance':importance}
        imp_df = pd.DataFrame(imp_dict)
        imp_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=imp_df['feature_importance'], y=imp_df['feature_names'])
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        plt.tight_layout()
        plt.savefig((accuracy_dir + '/' + predictor_imp_keyword + '_pred_importance.png'))


def pdp_plot(x_train, classifier, outdir,  # title1='PDP <5cm Subsidence.png',
             # title2='PDP >5cm Subsidence.png'
             ):
    plot_names = x_train.columns.tolist()
    feature_indices = range(len(plot_names))

    # Class 5
    fig, ax = plt.subplots()
    ax.set_title('Partial Dependence Plot for <5cm Subsidence')
    plot_partial_dependence(classifier, x_train, target=5, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            ax=ax)
    plt.rcParams['font.size'] = '8'
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=3)
    # fig.savefig(os.path.join(output_dir,title1),dpi=300, bbox_inches='tight')

    # Class 10
    fig, ax = plt.subplots()
    ax.set_title('Partial Dependence Plot for >5cm Subsidence')
    plot_partial_dependence(classifier, x_train, target=10, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            ax=ax)
    plt.rcParams['font.size'] = '8'
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=3)
    # fig.savefig(os.path.join(output_dir,title2),dpi=300, bbox_inches='tight')


def create_prediction_raster(predictors_dir, model, yearlist=[2013, 2019], search_by='*.tif',
                             continent_shapes_dir=r'../Data/Reference_rasters_shapes/continent_extents',
                             prediction_raster_dir=r'../Model Run/Prediction_rasters',
                             exclude_columns=(), pred_attr='Subsidence',
                             prediction_raster_keyword='RF'):
    """
    Create predicted raster from random forest model.

    Parameters:
    predictors_dir : Predictor rasters' directory.
    model : A fitted model obtained from randon_forest_classifier function.
    yearlist : List of years for the prediction.
    search_by : Predictor rasters search search_by. Defaults to '*.tif'.
    continent_shapes_dir : Directory path of continent shapefiles.
    prediction_raster_dir : Output directory of prediction raster.
    exclude_columns : Predictor rasters' name that will be excluded from the model. Defaults to ().
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    prediction_raster_keyword : Keyword added to final prediction raster name.

    Returns: None.
    """
    predictor_rasters = glob(os.path.join(predictors_dir, search_by))
    continent_shapes = glob(os.path.join(continent_shapes_dir, '*continent.shp'))
    drop_columns = list(exclude_columns) + [pred_attr]

    for continent in continent_shapes:
        continent_name = continent[continent.rfind(os.sep) + 1:continent.rfind('_')]
        predictor_dict = {}
        nan_position_dict = {}
        raster_shape = None
        for predictor in predictor_rasters:
            variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind(".")]
            if variable_name not in drop_columns:
                clipped_predictor_dir = os.path.join(r'../Model Run/Predictors_2013_2019', continent_name+'_predictors_'
                                                     + str(yearlist[0]) + '_' + str(yearlist[1]))
                raster_arr, raster_file = clip_resample_raster_cutline(predictor, clipped_predictor_dir, continent,
                                                                       naming_from_both=False)
                raster_shape = raster_arr.shape
                raster_arr = raster_arr.reshape(raster_shape[0]*raster_shape[1])
                nan_position_dict[variable_name] = np.isnan(raster_arr)
                raster_arr[nan_position_dict[variable_name]] = 0
                predictor_dict[variable_name] = raster_arr

        predictor_df = pd.DataFrame(predictor_dict)
        predictor_df = predictor_df.dropna(axis=0)

        predictor_csv_dir = r'../Model Run/Predictors_csv/continent_csv'
        makedirs([predictor_csv_dir])
        predictor_csv_name = continent_name + '_predictors.csv'
        predictor_csv = os.path.join(predictor_csv_dir, predictor_csv_name)
        predictor_df.to_csv(predictor_csv, index=False)

        x = predictor_df.values
        y_pred = model.predict(x)

        for nan_pos in nan_position_dict.values():
            y_pred[nan_pos] = raster_file.nodata
        y_pred_arr = y_pred.reshape(raster_shape)
        continent_prediction_raster_dir = os.path.join(prediction_raster_dir, 'continent_prediction_rasters_'
                                                       + str(yearlist[0]) + '_' + str(yearlist[1]))
        makedirs([prediction_raster_dir])
        makedirs([continent_prediction_raster_dir])
        prediction_raster_name = continent_name + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
        predicted_raster = os.path.join(continent_prediction_raster_dir, prediction_raster_name)
        write_raster(raster_arr=y_pred_arr, raster_file=raster_file, transform=raster_file.transform,
                     outfile_path=predicted_raster)
        print('Prediction raster created for', continent_name)

    mosaic_rasters(continent_prediction_raster_dir, prediction_raster_dir,
                   raster_name=prediction_raster_keyword + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif')
    print('Global prediction raster created')
