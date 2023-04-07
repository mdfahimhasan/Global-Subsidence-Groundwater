# Author: Md Fahim Hasan
# Email: Fahim.Hasan@colostate.edu

import pickle
import pandas as pd
import seaborn as sns
from pprint import pprint
import dask.dataframe as ddf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from lightgbm import LGBMClassifier
from Raster_operations import *
from System_operations import *

referenceraster = '../Data/Reference_rasters_shapes/Global_continents_ref_raster.tif'


def reindex_df(df):
    """
    Reindex dataframe based on column names.
    Parameters:
    df : Pandas dataframe.
    Returns: Reindexed dataframe.
    """
    sorted_columns = sorted(df.columns)
    df = df.reindex(sorted_columns, axis=1)

    return df


def create_dataframe(input_raster_dir, output_csv, search_by='*.tif', skip_dataframe_creation=False):
    """
    create dataframe from predictor rasters.
    Parameters:
    input_raster_dir : Input rasters directory.
    output_csv : Output csv file with filepath.
    search_by : Input raster search criteria. Defaults to '*.tif'.
    skip_predictor_subsidence_compilation : Set to True if want to skip processing.
    Returns: predictor_df dataframe created from predictor rasters.
    """
    print('Creating Predictors csv...')

    predictor_rename_dict = {'Alexi_ET': 'Alexi ET', 'Aridity_Index': 'Aridity Index',
                             'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI', 'Grace': 'Grace',
                             'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                             'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                             'Irrigated_Area_Density_gfsad': 'Irrigated Area Density (gfsad)',
                             'MODIS_ET': 'MODIS ET (kg/m2)', 'MODIS_PET': 'MODIS PET (kg/m2)', 'NDWI': 'NDWI',
                             'Irrigated_Area_Density_meier': 'Normalized Irrigated Area Density',
                             'Population_Density': 'Normalized Population Density', 'SRTM_Slope': '% Slope',
                             'Subsidence': 'Subsidence', 'TRCLM_RET': 'RET (mm)',
                             'TRCLM_precp': 'Precipitation (mm)', 'TRCLM_soil': 'Soil moisture (mm)',
                             'TRCLM_Tmax': 'Tmax (°C)', 'TRCLM_Tmin': 'Tmin (°C)', 'MODIS_Land_Use': 'MODIS Land Use',
                             'TRCLM_ET': 'ET (mm)', 'Clay_Thickness': 'Clay Thickness (m)',
                             'Normalized_clay_indicator': 'Normalized Clay Indicator', 'Clay_200cm': 'Clay % 200cm',
                             'River_gaussian': 'River Gaussian', 'River_distance': 'River Distance (km)',
                             'Confining_layers': 'Confining Layers'}

    if not skip_dataframe_creation:
        predictors = glob(os.path.join(input_raster_dir, search_by))

        predictor_dict = {}
        for predictor in predictors:
            variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind('.')]
            raster_arr, file = read_raster_arr_object(predictor, get_file=True)
            raster_arr = raster_arr.flatten()
            predictor_dict[variable_name] = raster_arr

        predictor_df = pd.DataFrame(predictor_dict)

        # converting to dask dataframe as this is huge data and might throw memory error
        predictor_df = ddf.from_pandas(predictor_df, npartitions=60)
        predictor_df = predictor_df.dropna()  # drops nan by rows by default (axis=0 throws error in dask dataframe)

        # converting to pandas dataframe again for applying reindex function
        predictor_df = predictor_df.compute()
        predictor_df = predictor_df.rename(columns=predictor_rename_dict)
        predictor_df = reindex_df(predictor_df)
        predictor_df.to_csv(output_csv, index=False)

        print('Predictors csv created')
        return predictor_df
    else:
        predictor_df = pd.read_csv(output_csv)
        return predictor_df


def split_train_test_ratio(predictor_csv, exclude_columns=[], pred_attr='Subsidence', test_size=0.3, random_state=0,
                           outdir=None, verbose=True):
    """
    Split dataset into train and test data based on a ratio
    parameters:
    input_csv : Input csv (with filepath) containing all the predictors.
    exclude_columns : Tuple of columns not included in training the fitted_model.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    verbose : Set to True if want to print which columns are being dropped and which will be included in the model.
    Returns: X_train, X_test, y_train, y_test
    """
    input_df = pd.read_csv(predictor_csv)
    predictor_name_dict = {'Alexi_ET': 'Alexi ET', 'Aridity_Index': 'Aridity Index',
                           'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI', 'Grace': 'Grace',
                           'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                           'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                           'Irrigated_Area_Density_gfsad': 'Irrigated Area Density (gfsad)',
                           'MODIS_ET': 'MODIS ET (kg/m2)', 'MODIS_PET': 'MODIS PET (kg/m2)', 'NDWI': 'NDWI',
                           'Irrigated_Area_Density_meier': 'Normalized Irrigated Area Density',
                           'Population_Density': 'Normalized Population Density', 'SRTM_Slope': '% Slope',
                           'Subsidence': 'Subsidence', 'TRCLM_RET': 'RET (mm)',
                           'TRCLM_precp': 'Precipitation (mm)', 'TRCLM_soil': 'Soil moisture (mm)',
                           'TRCLM_Tmax': 'Tmax (°C)', 'TRCLM_Tmin': 'Tmin (°C)', 'MODIS_Land_Use': 'MODIS Land Use',
                           'TRCLM_ET': 'ET (mm)', 'Clay_Thickness': 'Clay Thickness (m)',
                           'Normalized_clay_indicator': 'Normalized Clay Indicator', 'Clay_200cm': 'Clay % 200cm',
                           'River_gaussian': 'River Gaussian', 'River_distance': 'River Distance (km)',
                           'Confining_layers': 'Confining Layers'}

    input_df = input_df.rename(columns=predictor_name_dict)
    drop_columns = exclude_columns + [pred_attr]
    x = input_df.drop(columns=drop_columns)
    y = input_df[pred_attr]
    if verbose:
        print('Dropping Columns-', exclude_columns)
        print('Predictors:', x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        shuffle=True, stratify=y)

    if outdir:
        x_train_df = pd.DataFrame(x_train)
        x_train_df.to_csv(os.path.join(outdir, 'X_train.csv'), index=False)

        y_train_df = pd.DataFrame(y_train)
        y_train_df.to_csv(os.path.join(outdir, 'y_train.csv'), index=False)

        x_test_df = pd.DataFrame(x_test)
        x_test_df.to_csv(os.path.join(outdir, 'X_test.csv'), index=False)

        y_test_df = pd.DataFrame(y_test)
        y_test_df.to_csv(os.path.join(outdir, 'y_test.csv'), index=False)

    return x_train, x_test, y_train, y_test, predictor_name_dict


def hyperparameter_optimization(x_train, y_train, model='rf', folds=10, n_iter=70, random_search=True,
                                repeatedstratified=False):
    """
    Hyperparameter optimization using RandomizedSearchCV/GridSearchCV.
    Parameters:
    x_train, y_train : x_train (predictor) and y_train (target) arrays from split_train_test_ratio function.
    mode : Model for which hyperparameters will be tuned. Should be 'rf'/'gbdt'. Default set to 'rf'.
    folds : Number of folds in K Fold CV. Default set to 5.
    n_iter : Number of parameter combinations to be tested in RandomizedSearchCV.
    random_search : Set to False if want to perform GridSearchCV. Default set to True to perform RandomizedSearchCV.
    repeatedstratified : Set to False to perform Stratified CV.
    Returns : Optimized Hyperparameters.
    """
    global classifier
    param_dict = {'rf':
                      {'n_estimators': [100, 200, 300, 400, 500],
                       'max_depth': [8, 12, 13, 14],
                       'max_features': [6, 7, 9, 10],
                       'min_samples_leaf': [5e-4, 1e-5, 1e-3, 6, 12, 20, 25],
                       'min_samples_split': [6, 7, 8, 10]
                       },
                  'gbdt':
                      {'num_leaves': [31, 63, 100, 200],
                       'max_depth': [10, 12, 15, 20],
                       'learning_rate': [0.01, 0.05],
                       'n_estimators': [100, 200, 300],
                       'subsample': [1, 0.9],
                       'min_child_samples': [20, 25, 30, 35, 50]}
                  }

    print('Classifier Name:', model)
    pprint(param_dict[model])

    if model == 'rf':
        classifier = RandomForestClassifier(random_state=0, n_jobs=-1, bootstrap=True, oob_score=True,
                                            class_weight='balanced')
    elif model == 'gbdt':
        classifier = LGBMClassifier(boosting_type='gbdt', objective='multiclass', class_weight='balanced',
                                    importance_type='split', random_state=0, n_jobs=-1)

    # creating scorer that gives score of precision and recall of 1-5 cm/year class (works, storing as a reference
    # function to create own score)

    # def confusion_matrix_scorer(clf, x_train, y_train):
    #
    #     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_ratio,
    #                                                       random_state=0, shuffle=True, stratify=y_train)
    #     y_val_pred = clf.predict(x_val)
    #     class_report_dict = classification_report(y_val, y_val_pred, target_names=['<1cm/yr', '1-5cm/yr',
    #                                                                                '>5cm/yr'],
    #                                               output_dict=True)
    #
    #     class_report_df = pd.DataFrame(class_report_dict)
    #     macro_f1_score = class_report_df.loc['f1-score']['macro avg']
    #
    #     y_train_pred = clf.predict(x_train)
    #     cf = confusion_matrix(y_train, y_train_pred)
    #     fn_1cm = [cf[0][1], cf[0][2]]
    #     fn_1_5cm = [cf[1][0], cf[1][2]]
    #     fn_5cm = [cf[2][0], cf[2][1]]
    #     if all(fn_1cm) > 0 and all(fn_1_5cm) > 0 and any(fn_5cm):
    #         return {'macro_f1_score': macro_f1_score}
    #     else:
    #         return {'macro_f1_score': 0}

    if repeatedstratified:
        kfold = RepeatedStratifiedKFold(n_splits=folds, n_repeats=10, random_state=0)
    else:
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    if random_search:
        CV = RandomizedSearchCV(estimator=classifier, param_distributions=param_dict[model], n_iter=n_iter,
                                cv=kfold, verbose=1, random_state=0, n_jobs=-1,
                                scoring='f1_macro', refit=True, return_train_score=True)
    else:
        CV = GridSearchCV(estimator=classifier, param_grid=param_dict[model], cv=kfold, verbose=1, n_jobs=-1,
                          scoring='f1_macro', refit=True, return_train_score=True)

    CV.fit(x_train, y_train)

    print('\n')
    print('best parameters for macro f1 value ', '\n')
    pprint(CV.best_params_)
    print('\n')
    print('mean_test_macro_f1_score', round(CV.cv_results_['mean_test_score'][CV.best_index_], 2))
    print('mean_train_macro_f1_score', round(CV.cv_results_['mean_train_score'][CV.best_index_], 2))

    if model == 'rf':
        optimized_param_dict = {'n_estimators': CV.best_params_['n_estimators'],
                                 'max_depth': CV.best_params_['max_depth'],
                                 'max_features': CV.best_params_['max_features'],
                                 'min_samples_leaf': CV.best_params_['min_samples_leaf'],
                                'min_samples_split': CV.best_params_['min_samples_split']
                                }

        return optimized_param_dict

    elif model == 'gbdt':
        optimized_param_dict = {'num_leaves': CV.best_params_['num_leaves'],
                                'max_depth': CV.best_params_['max_depth'],
                                'learning_rate': CV.best_params_['learning_rate'],
                                'n_estimators': CV.best_params_['n_estimators'],
                                'subsample': CV.best_params_['subsample'],
                                'min_child_samples': CV.best_params_['min_child_samples']}

        return optimized_param_dict


def build_ml_classifier(predictor_csv, modeldir, exclude_columns=(), model='rf', load_model=False,
                        pred_attr='Subsidence', test_size=0., random_state=0, output_dir=None,
                        n_estimators=300, min_samples_leaf=1, min_samples_split=2, max_depth=20, max_features='auto',
                        max_samples=None, max_leaf_nodes=None,  # #
                        bootstrap=True, oob_score=True, n_jobs=-1, class_weight='balanced',
                        num_leaves=31, max_depth_gbdt=-1, learning_rate=0.01, n_estimators_gbdt=200, subsample=0.9,
                        colsample_bytree=1, min_child_samples=20,
                        estimate_accuracy=True, accuracy_dir=r'../Model Run/Accuracy_score',
                        predictor_importance=False, predictor_imp_keyword='RF',
                        plot_pdp=False, variables_pdp=('Normalized Irrigated Area Density',
                                                       'Normalized Population Density',
                                                       'Precipitation (mm)', 'Sediment Thickness (m)',
                                                       'Soil moisture (mm)', 'TRCLM ET (mm)'),
                        pdp_combinations=(('Normalized Irrigated Area Density', 'Normalized Clay Indicator'),
                                          ('Normalized Irrigated Area Density', 'Soil moisture (mm)')),
                        plot_confusion_matrix=True, cm_name='cmatrix.csv',
                        tune_hyperparameter=False, repeatedstratified=False, k_fold=10, n_iter=70, random_searchCV=True):
    """
    Build Machine Learning Classifier. Can run 'Random Forest', 'Gradient Boosting Decision Tree'.
    Parameters:
    predictor_dataframe_csv : Predictor csv (with filepath) containing all the predictors.
    modeldir : Model directory to store/load fitted_model.
    exclude_columns : Tuple of columns not included in training the fitted_model.
    fitted_model : Machine learning fitted_model to run. Choose from 'rf'/'gdbt'. Default set to 'rf'.
    load_model : Set True to load existing fitted_model. Default set to False for new fitted_model creation.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    shuffle : Whether or not to shuffle data before splitting. Defaults to True.
    output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    n_estimators (rf param) : Number of trees in the random forest. Defaults to 300.
    min_samples_leaf (rf param): Minimum number of samples required to be at a leaf node. Defaults to 1.
    min_samples_split (rf param): Minimum number of samples required to split an internal node. Defaults to 2.
    max_features (rf param): The number of features to consider when looking for the best split. Defaults to 'log2'.
    max_depth (rf param): max_length of tree. Default set to 76.
    bootstrap (rf param): Whether bootstrap samples are used when building trees. Defaults to True.
    oob_score (rf param): Whether to use out-of-bag samples to estimate the generalization accuracy. Defaults to True.
    n_jobs (rf/gbdt param): The number of jobs to run in parallel. Defaults to -1(using all processors).
    class_weight (rf/gbdt param): To assign class weight. Default set to 'balanced'.
    num_leaves (gbdt param): Maximum tree leaves for base learners. Default set to 31.
    max_depth_gbdt (gbdt param): Maximum tree depth for base learners. Default set to -1 (<=0 means no limit).
    learning_rate (gbdt param): Boosting learning rate. Default set to 0.01.
    n_estimators_gbdt (gbdt param): Boosting learning rate. Default set to 200.
    subsample (gbdt param): Subsample ratio of the training instance. Default set to 0.9.
    colsample_bytree (gbdt param): Subsample ratio of columns when constructing each tree. Default set to 1.
    min_child_samples (gbdt param):  Minimum number of data needed in a child (leaf). Default set to 20.
    estimate_accuracy : Set to True if want to estimate model accuracy metrices.
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    predictor_importance : Set True if predictor importance plot is needed. Defaults to False.
    predictor_imp_keyword : Keyword to save predictor important plot.
    plot_save_keyword : Keyword to sum before saved PDP plots.
    plot_pdp : Set to True if want to plot PDP.
    variables_pdp : Tuple of variable names to plot in pdp plot.
    plot_confusion_matrix : Set to True if want to plot confusion matrix.
    cm_name : Confusion matrix name. Defaults to 'cmatrix.csv'.
    tune_hyperparameter : Set to True to tune hyperparameter. Default set to False.
    repeatedstratified : Set to False to perform Stratified CV. Default set to False.
    k_fold : number of folds in K-fold CV. Default set to 5.
    n_iter : Number of parameter combinations to be tested in RandomizedSearchCV. Default set to 70.
    random_searchCV : Set to False if want to perform GridSearchCV. Default set to True to perform RandomizedSearchCV.
    Returns: rf_classifier (A fitted random forest fitted_model)
    """

    # Splitting Training and Testing Data
    global classifier
    x_train, x_test, y_train, y_test, predictor_name_dict = \
        split_train_test_ratio(predictor_csv=predictor_csv, exclude_columns=exclude_columns, pred_attr=pred_attr,
                               test_size=test_size, random_state=random_state, outdir=output_dir)

    # Making directory for fitted_model
    makedirs([modeldir])
    model_file = os.path.join(modeldir, model)

    # Hyperparamter Tuning
    if tune_hyperparameter:
        optimized_param_dict = hyperparameter_optimization(x_train, y_train, model=model, folds=k_fold, n_iter=n_iter,
                                                           random_search=random_searchCV,
                                                           repeatedstratified=repeatedstratified)
        if model == 'rf':
            n_estimators = optimized_param_dict['n_estimators']
            max_depth = optimized_param_dict['max_depth']
            max_features = optimized_param_dict['max_features']
            min_samples_leaf = optimized_param_dict['min_samples_leaf']
            # max_leaf_nodes = optimized_param_dict['max_leaf_nodes']
            # max_samples = optimized_param_dict['max_samples']
            min_samples_split = optimized_param_dict['min_samples_split']

        elif model == 'gbdt':
            num_leaves = optimized_param_dict['num_leaves']
            max_depth_gbdt = optimized_param_dict['max_depth']
            learning_rate = optimized_param_dict['learning_rate']
            n_estimators_gbdt = optimized_param_dict['n_estimators']
            subsample = optimized_param_dict['subsample']
            # colsample_bytree = optimized_param_dict['colsample_bytree']
            min_child_samples = optimized_param_dict['min_child_samples']

    # Machine Learning Models
    if not load_model:
        if model == 'rf':
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                max_depth=max_depth, max_samples=max_samples,
                                                max_leaf_nodes=max_leaf_nodes, random_state=random_state,
                                                bootstrap=bootstrap, class_weight=class_weight, n_jobs=n_jobs,
                                                oob_score=oob_score)
        elif model == 'gbdt':
            classifier = LGBMClassifier(boosting_type='gbdt', objective='multiclass', class_weight='balanced',
                                        num_leaves=num_leaves, max_depth_gbdt=max_depth_gbdt,
                                        learning_rate=learning_rate, n_estimators_gbdt=n_estimators_gbdt,
                                        subsample=subsample, colsample_bytree=colsample_bytree,
                                        min_child_samples=min_child_samples, importance_type='split', random_state=0,
                                        n_jobs=-1)

        classifier = classifier.fit(x_train, y_train)

        pickle.dump(classifier, open(model_file, mode='wb+'))

    else:
        classifier = pickle.load(open(model_file, mode='rb'))

    if estimate_accuracy:
        classification_accuracy(x_train, x_test, y_train, y_test, classifier, accuracy_dir, cm_name,
                                predictor_importance, predictor_imp_keyword, plot_confusion_matrix)
    if plot_pdp:
        pdp_plot(classifier, x_train, accuracy_dir, plot_save_keyword=predictor_imp_keyword,
                 feature_names=variables_pdp)
        pdp_plot_combinations(classifier, x_train, accuracy_dir, plot_save_keyword=predictor_imp_keyword,
                              feature_names=pdp_combinations)

    return classifier, predictor_name_dict


def classification_accuracy(x_train, x_test, y_train, y_test, classifier,
                            accuracy_dir=r'../Model Run/Accuracy_score', cm_name='cmatrix.csv',
                            predictor_importance=False, predictor_imp_keyword='RF', plot_confusion_matrix=True):
    """
    Classification accuracy assessment.
    Parameters:
    x_train : x_train from 'split_train_test_ratio' function.
    x_test :  x_test from 'split_train_test_ratio' function.
    y_train : y_train data from split_train_test_ratio() function.
    y_test : y_test data from split_train_test_ratio() function.
    classifier : ML classifier from build_ML_classifier() function.
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    cm_name : Confusion matrix name. Defaults to 'cmatrix.csv'.
    predictor_importance : Set True if predictor importance plot is needed. Defaults to False.
    predictor_imp_keyword : Keyword to save predictor important plot.
    Returns: Confusion matrix, score and predictor importance graph.
    """
    makedirs([accuracy_dir])
    y_train_pred = classifier.predict(x_train)
    y_pred = classifier.predict(x_test)

    # Plotting and saving confusion matrix
    column_labels = [np.array(['Predicted', 'Predicted', 'Predicted']),
                     np.array(['<1cm/yr', '1-5cm/yr', '>5cm/yr'])]
    index_labels = [np.array(['Actual', 'Actual', 'Actual']),
                    np.array(['<1cm/yr', '1-5cm/yr', '>5cm/yr'])]

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_df_train = pd.DataFrame(cm_train, columns=column_labels, index=index_labels)
    cm_name_train = predictor_imp_keyword + '_train_' + cm_name
    csv_train = os.path.join(accuracy_dir, cm_name_train)
    cm_df_train.to_csv(csv_train)

    column_labels = [np.array(['Predicted', 'Predicted', 'Predicted']),
                     np.array(['<1cm/yr', '1-5cm/yr', '>5cm/yr'])]
    cm_test = confusion_matrix(y_test, y_pred)
    cm_df_test = pd.DataFrame(cm_test, columns=column_labels, index=index_labels)
    cm_name_test = predictor_imp_keyword + '_test_' + cm_name
    csv_test = os.path.join(accuracy_dir, cm_name_test)
    cm_df_test.to_csv(csv_test, index=True)

    pd.options.display.width = 0
    print(cm_df_train, '\n')
    print(cm_df_test, '\n')

    if plot_confusion_matrix:
        font = 11
        label = np.array(['<1cm', '1-5 cm', '>5cm'])
        disp = ConfusionMatrixDisplay(cm_test, display_labels=label)
        disp.plot(cmap='YlGn')
        for labels in disp.text_.ravel():
            labels.set_fontsize(font)
        disp.ax_.set_ylabel('True Class', fontsize=font)
        disp.ax_.set_xlabel('Predicted Class', fontsize=font)
        plot_name = cm_name_test[:cm_name_test.rfind('.')] + '.png'
        plt.savefig((accuracy_dir + '/' + plot_name), dpi=400)
        print('Test confusion matrix saved')

    # Saving fitted_model accuracy for individual classes
    overall_accuracy = round(accuracy_score(y_test, y_pred), 2)
    print('Accuracy Score {}'.format(overall_accuracy))
    accuracy_csv_name = accuracy_dir + '/' + predictor_imp_keyword + '_accuracy.csv'
    save_model_accuracy(cm_df_test, overall_accuracy, accuracy_csv_name)

    # generating classification report
    label_names = ['<1cm/yr', '1-5cm/yr', '>5cm/yr']
    classification_report_dict_train = classification_report(y_train, y_train_pred, target_names=label_names,
                                                             output_dict=True)
    del classification_report_dict_train['accuracy']
    classification_report_df_train = pd.DataFrame(classification_report_dict_train)
    classification_report_df_train.drop(labels='support', inplace=True)
    micro_precision = round(precision_score(y_train, y_train_pred, average='micro'), 2)
    micro_recall = round(recall_score(y_train, y_train_pred, average='micro'), 2)
    micro_f1 = round(f1_score(y_train, y_train_pred, average='micro'), 2)

    classification_report_df_train['micro avg'] = [micro_precision, micro_recall, micro_f1]
    cols = classification_report_df_train.columns.tolist()
    cols = cols[:3] + cols[-1:] + cols[3:5]  # rearranging columns
    classification_report_df_train = classification_report_df_train[cols].round(2)
    print(classification_report_df_train)
    classification_report_csv_name = accuracy_dir + '/' + predictor_imp_keyword + '_train classification report.csv'
    classification_report_df_train.to_csv(classification_report_csv_name)

    label_names = ['<1cm/yr', '1-5cm/yr', '>5cm/yr']
    classification_report_dict_test = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    del classification_report_dict_test['accuracy']
    classification_report_df_test = pd.DataFrame(classification_report_dict_test)
    classification_report_df_test.drop(labels='support', inplace=True)
    micro_precision = round(precision_score(y_test, y_pred, average='micro'), 2)
    micro_recall = round(recall_score(y_test, y_pred, average='micro'), 2)
    micro_f1 = round(f1_score(y_test, y_pred, average='micro'), 2)

    classification_report_df_test['micro avg'] = [micro_precision, micro_recall, micro_f1]
    cols = classification_report_df_test.columns.tolist()
    cols = cols[:3] + cols[-1:] + cols[3:5]  # rearranging columns
    classification_report_df = classification_report_df_test[cols].round(2)
    print(classification_report_df)
    classification_report_csv_name = accuracy_dir + '/' + predictor_imp_keyword + '_test classification report.csv'
    classification_report_df.to_csv(classification_report_csv_name)

    # predictor importance plot
    if predictor_importance:
        predictor_dict = {'Alexi_ET': 'Alexi ET', 'Aridity_Index': 'Aridity Index',
                          'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI', 'Grace': 'Grace',
                          'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                          'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                          'Irrigated_Area_Density_gfsad': 'Irrigated Area Density (gfsad)',
                          'MODIS_ET': 'MODIS ET (kg/m2)', 'MODIS_PET': 'MODIS PET (kg/m2)', 'NDWI': 'NDWI',
                          'Irrigated_Area_Density_meier': 'Normalized Irrigated Area Density',
                          'Population_Density': 'Normalized Population Density', 'SRTM_Slope': '% Slope',
                          'Subsidence': 'Subsidence', 'TRCLM_RET': 'RET (mm)',
                          'TRCLM_precp': 'Precipitation (mm)', 'TRCLM_soil': 'Soil moisture (mm)',
                          'TRCLM_Tmax': 'Tmax (°C)', 'TRCLM_Tmin': 'Tmin (°C)', 'MODIS_Land_Use': 'MODIS Land Use',
                          'TRCLM_ET': 'ET (mm)', 'Clay_Thickness': 'Clay Thickness (m)',
                          'Normalized_clay_indicator': 'Normalized Clay Indicator', 'Clay_200cm': 'Clay % 200cm',
                          'River_gaussian': 'River Gaussian', 'River_distance': 'River Distance (km)',
                          'Confining_layers': 'Confining Layers'}
        x_train_df = pd.DataFrame(x_train)
        x_train_df = x_train_df.rename(columns=predictor_dict)
        col_labels = np.array(x_train_df.columns)
        importance = np.array(classifier.feature_importances_)
        imp_dict = {'feature_names': col_labels, 'feature_importance': importance}
        imp_df = pd.DataFrame(imp_dict)
        imp_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        plt.figure(figsize=(10, 8))
        plt.rcParams['font.size'] = 14
        sns.barplot(x=imp_df['feature_names'], y=imp_df['feature_importance'], palette='rocket')
        plt.xticks(rotation=90)
        plt.ylabel('Variable Importance')
        plt.xlabel('Variables')
        plt.tight_layout()
        plt.savefig((accuracy_dir + '/' + predictor_imp_keyword + '_pred_importance.png'), dpi=600)
        print('Feature importance plot saved')

    return cm_df_test, overall_accuracy


def save_model_accuracy(cm_df_test, overall_accuracy, accuracy_csv_name):
    """
    Save fitted_model accuracy parameters as csv.
    Parameters:
    cm_df_test : Confusion matrix dataframe (input from 'classification_accuracy' function).
    overall_accuracy : Overall accuracy value (input from 'classification_accuracy' function).
    accuracy_csv_name : Name of the csv file to save.
    Returns : Saved csv with fitted_model accuracy values.
    """
    from operator import truediv
    act_pixel_less_1cm = sum(cm_df_test.loc[('Actual', '<1cm/yr')])
    act_pixel_1cm_to_5cm = sum(cm_df_test.loc[('Actual', '1-5cm/yr')])
    act_pixel_greater_5cm = sum(cm_df_test.loc[('Actual', '>5cm/yr')])
    pred_pixel_less_1cm = cm_df_test.loc[('Actual', '<1cm/yr'), ('Predicted', '<1cm/yr')]
    pred_pixel_1cm_to_5cm = cm_df_test.loc[('Actual', '1-5cm/yr'), ('Predicted', '1-5cm/yr')]
    pred_pixel_greater_1cm = cm_df_test.loc[('Actual', '>5cm/yr'), ('Predicted', '>5cm/yr')]

    actual_no_pixels = [act_pixel_less_1cm, act_pixel_1cm_to_5cm, act_pixel_greater_5cm]
    accurately_pred_pixel = [pred_pixel_less_1cm, pred_pixel_1cm_to_5cm, pred_pixel_greater_1cm]
    accuracy = list(map(truediv, accurately_pred_pixel, actual_no_pixels))
    accuracy = [round(i, 2) for i in accuracy]

    total_accuracy = np.array([overall_accuracy, overall_accuracy, overall_accuracy])
    accuracy_dataframe = pd.DataFrame({'Actual No. of Pixels': actual_no_pixels,
                                       'Accurately Predicted Pixels': accurately_pred_pixel, 'Accuracy': accuracy,
                                       'Overall Accuracy': total_accuracy},
                                      index=['<1cm/yr', '1-5cm/yr', '>5cm/yr'])
    accuracy_dataframe.to_csv(accuracy_csv_name)


def pdp_plot(classifier, x_train, output_dir, plot_save_keyword='rf',
             feature_names=('Sediment Thickness (m)', 'Normalized Irrigated Area Density',
                            'Normalized Population Density', 'Precipitation (mm)', 'Clay Thickness (m)',
                            'Soil moisture (mm)', 'TRCLM ET (mm)')):
    """
    Plot Partial Dependence Plot for the fitted_model.
    Parameters:
    classifier :ML fitted_model classifier.
    x_train : X train array.
    output_dir : Output directory path to save the plots.
    plot_save_keyword : Keyword to sum before saved PDP plots.
    feature_names : Tuple of variable names to plot in pdp plot.
    Returns : PDP plots.
    """
    global vals, probability
    plt.rcParams['font.size'] = 18

    classes = [1, 5, 10]

    for each in classes:
        pdisp = PartialDependenceDisplay.from_estimator(classifier, x_train, features=feature_names, target=each,
                                                        response_method='predict_proba', percentiles=(0, 1),
                                                        n_jobs=-1, random_state=0, grid_resolution=20)
        for row_idx in range(0, pdisp.axes_.shape[0]):
            pdisp.axes_[row_idx][0].set_ylabel('Partial Dependence')
        fig = plt.gcf()
        fig.set_size_inches(20, 15)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdp_plot_name = {1: 'PDP less 1cm Subsidence.png', 5: 'PDP 1 to 5cm Subsidence.png',
                         10: 'PDP greater 5cm Subsidence.png'}
        fig.savefig((output_dir + '/' + plot_save_keyword + '_' + pdp_plot_name[each]),
                    dpi=300, bbox_inches='tight')
        print(pdp_plot_name[each].split('.')[0], 'saved')

    if 'Confining Layers' in feature_names:
        pdp = partial_dependence(classifier, x_train, features='Confining Layers',
                                               response_method='predict_proba')
    for i in range(len(classes)):
        y_val = list(pdp['average'][i])

        plt.figure(figsize=(10, 7.5))
        plt.bar(['0', '1'], y_val, color='tab:blue', width=0.3)
        plt.ylim([0, 0.25])
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.savefig((output_dir + '/' + 'pdp_confining' + '_' + str(classes[i]) + '.png'), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(25, 10))
    plt.subplots_adjust(bottom=0.15, top=0.96, left=0.4, right=0.99, wspace=0.2,
                        hspace=0.27)  # wspace and hspace adjust the horizontal and vertical spaces, respectively.

    class_names = ['<1 cm/year', '1-5 cm/year', '>5 cm/year']
    serial = ['(a)', '(b)', '(c)']
    for i in range(len(classes)):
        y_val = list(pdp['average'][i])

        plt.subplot(1, 3, i+1)
        plt.bar(['0', '1'], y_val, color='tab:blue', width=0.3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(f'Confining Layers\n {serial[i]} {class_names[i]}', fontsize=20)
        if i == 0:
            plt.ylabel('Partial Dependence', fontsize=20)
    plt.tight_layout()
    plt.savefig((output_dir + '/' + 'pdp_confining' + '_all' + '.png'), dpi=300, bbox_inches='tight')


def pdp_plot_combinations(classifier, x_train, output_dir, plot_save_keyword='rf',
                          feature_names=(['Normalized Irrigated Area Density', 'Normalized Clay Indicator'],
                                         ['Precipitation (mm)', 'Soil moisture (mm)'])):
    """
    PDP of 2*2 = 4 variables. Don't include 'Confining Layers'
    Parameters:
    classifier :ML fitted_model classifier.
    x_train : X train array.
    output_dir : Output directory path to save the plots.
    plot_save_keyword : Keyword to sum before saved PDP plots.
    feature_names : Tuple of variable names to plot in pdp plot. For combined PDP of 2 variables put the variables in a
                    tuple like ('Precipitation (mm)', 'Soil moisture (mm)').
    Returns : PDP plots.
    """
    prediction_class = [5]
    pdisp = PartialDependenceDisplay.from_estimator(classifier, x_train, features=feature_names,
                                                       target=prediction_class[0], response_method='predict_proba',
                                                       percentiles=(0.01, 0.999), n_jobs=-1, random_state=0,
                                                       grid_resolution=20)
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    pdisp.plot(ax=ax)
    ax[0].set_title('(a)', y=-0.2)
    ax[1].set_title('(b)', y=-0.2)
    fig = plt.gcf()
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.subplots_adjust(wspace=0.35, hspace=0.3)

    # Formatting colorbar
    import matplotlib.ticker as tick
    cbar = fig.colorbar((pdisp.contours_[0]), ax=ax)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    cbar.set_label('Probability of Subsidence between 1-5 cm/year', rotation=90, labelpad=15)

    pdp_plot_name = 'PDP 1 to 5cm Subsidence_combinations.png'
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + pdp_plot_name),
                dpi=400, bbox_inches='tight')

    pdp_plot_name_pdf = 'PDP 1 to 5cm Subsidence_combinations.pdf'
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + pdp_plot_name_pdf),
                dpi=400, bbox_inches='tight')

    print(pdp_plot_name.split('.')[0], 'saved')

    # # # # # # # # # # # # # # # #
    # # # 3D Plot (works, need to formatted properly for better visualization)
    # plt.rcParams['font.size'] = 13
    # fig = plt.figure()
    #
    # pdp = partial_dependence(classifier, x_train, features=('Irrigated Area Density', 'Clay Thickness (m)'),
    #                          grid_resolution=20, response_method='predict_proba')
    # XX, YY = np.meshgrid(pdp['values'][0], pdp['values'][1])
    # Z = pdp['average'][1].T  # index 1 for class 1-5 cm (class 5)
    # ax = fig.add_subplot(projection="3d")
    # fig.add_axes(ax)
    # surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.YlGnBu, edgecolor='k')
    #
    # ax.set_xlabel('Irrigated Area Density')
    # ax.set_ylabel('Clay Thickness (m)')
    # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    # ax.set_zlabel('Partial dependence', fontsize=20, rotation=90)
    #
    # #  pretty init view
    # ax.view_init(elev=22, azim=150)
    # plt.colorbar(surf)
    #
    # pdp_plot_name = 'PDP 1 to 5cm Subsidence_3D.png'
    # fig.savefig((output_dir + '/' + plot_save_keyword + '_' + pdp_plot_name),
    #             dpi=400, bbox_inches='tight')
    # print('3D PDP plot saved')
    # # # # # # # # # # # # # # # #


def create_prediction_raster(predictors_dir, model, predictor_name_dict, yearlist=(2013, 2019), search_by='*.tif',
                             continent_search_by='*continent.shp', predictor_csv_exists=False,
                             continent_shapes_dir='../Data/Reference_rasters_shapes/continent_extents',
                             prediction_raster_dir='../Model Run/Prediction_rasters',
                             exclude_columns=(), pred_attr='Subsidence',
                             prediction_raster_keyword='rf', predict_probability_greater_1cm=True):
    """
    Create predicted raster from random forest fitted_model.
    Parameters:
    predictors_dir : Predictor rasters' directory.
    fitted_model : A fitted fitted_model obtained from random_forest_classifier function.
    predictor_name_dict : Predictor name dictionary (comes from split_train_test_ratio > build_ml_classifier function).
    yearlist :Tuple of years for the prediction. Default set to (2013, 2019).
    search_by : Predictor rasters search criteria. Defaults to '*.tif'.
    continent_search_by : Continent shapefile search criteria. Defaults to '*continent.tif'.
    predictor_csv_exists : Set to True if predictor csv already exists. Defaults set to False. Should be False if
                           list of drop columns changes.
    continent_shapes_dir : Directory path of continent shapefiles.
    prediction_raster_dir : Output directory of prediction raster.
    exclude_columns : Predictor rasters' name that will be excluded from the fitted_model. Defaults to ().
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    prediction_raster_keyword : Keyword added to final prediction raster name.
    predict_probability_greater_1cm : Set to False if probability of prediction of each classes (<1cm, 1-5cm, >5cm)
                                      is required. Default set to True to predict probability of prediction for >1cm.
    Returns: Subsidence prediction raster and
             Subsidence prediction probability raster (if prediction_probability=True).
    """
    global raster_file
    predictor_rasters = glob(os.path.join(predictors_dir, search_by))
    continent_shapes = glob(os.path.join(continent_shapes_dir, continent_search_by))
    drop_columns = list(exclude_columns) + [pred_attr]

    continent_prediction_raster_dir = os.path.join(prediction_raster_dir, 'continent_prediction_rasters_'
                                                   + str(yearlist[0]) + '_' + str(yearlist[1]))
    makedirs([prediction_raster_dir])
    makedirs([continent_prediction_raster_dir])

    for continent in continent_shapes:
        continent_name = continent[continent.rfind(os.sep) + 1:continent.rfind('_')]

        predictor_csv_dir = '../Model Run/Predictors_csv/continent_csv'
        makedirs([predictor_csv_dir])
        predictor_csv_name = continent_name + '_predictors.csv'
        predictor_csv = os.path.join(predictor_csv_dir, predictor_csv_name)

        dict_name = predictor_csv_dir + '/nanpos_' + continent_name  # name to save nan_position_dict

        clipped_predictor_dir = os.path.join('../Model Run/Predictors_2013_2019', continent_name +
                                             '_predictors_' + str(yearlist[0]) + '_' + str(yearlist[1]))

        if not predictor_csv_exists:
            predictor_dict = {}
            nan_position_dict = {}
            raster_shape = None
            for predictor in predictor_rasters:
                variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind('.')]
                variable_name = predictor_name_dict[variable_name]

                raster_arr, raster_file = clip_resample_raster_cutline(predictor, clipped_predictor_dir, continent,
                                                                       naming_from_both=False,
                                                                       naming_from_raster=True, assigned_name=None)
                raster_shape = raster_arr.shape
                raster_arr = raster_arr.reshape(raster_shape[0] * raster_shape[1])
                nan_position_dict[variable_name] = np.isnan(raster_arr)
                raster_arr[nan_position_dict[variable_name]] = 0
                predictor_dict[variable_name] = raster_arr

            pickle.dump(nan_position_dict, open(dict_name, mode='wb+'))

            predictor_df = pd.DataFrame(predictor_dict)
            predictor_df = predictor_df.dropna(axis=0)
            predictor_df = reindex_df(predictor_df)
            # this predictor df consists all input variables including the ones to drop
            predictor_df.to_csv(predictor_csv, index=False)

        else:
            predictor_df = pd.read_csv(predictor_csv)

            nan_position_dict = pickle.load(open(dict_name, mode='rb'))

            raster_arr, raster_file = clip_resample_raster_cutline(predictor_rasters[1], clipped_predictor_dir,
                                                                   continent, naming_from_both=False,
                                                                   naming_from_raster=True, assigned_name=None)
            raster_shape = raster_arr.shape

        # selecting only variables for which the model was trained for
        predictor_df = predictor_df.drop(columns=drop_columns)
        y_pred = model.predict(predictor_df)

        for variable_name, nan_pos in nan_position_dict.items():
            if variable_name not in drop_columns:
                y_pred[nan_pos] = raster_file.nodata

        y_pred_arr = y_pred.reshape(raster_shape)

        prediction_raster_name = continent_name + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
        predicted_raster = os.path.join(continent_prediction_raster_dir, prediction_raster_name)
        write_raster(raster_arr=y_pred_arr, raster_file=raster_file, transform=raster_file.transform,
                     outfile_path=predicted_raster)
        print('Prediction raster created for', continent_name)

        if predict_probability_greater_1cm:
            y_pred_proba = model.predict_proba(predictor_df)
            y_pred_proba = y_pred_proba[:, 1] + y_pred_proba[:, 2]

            for variable_name, nan_pos in nan_position_dict.items():
                if variable_name not in drop_columns:
                    y_pred_proba[nan_pos] = raster_file.nodata

            y_pred_proba_arr = y_pred_proba.reshape(raster_shape)

            probability_raster_name = continent_name + '_proba_greater_1cm_' + str(yearlist[0]) + '_' + \
                                      str(yearlist[1]) + '.tif'
            probability_raster = os.path.join(continent_prediction_raster_dir, probability_raster_name)
            write_raster(raster_arr=y_pred_proba_arr, raster_file=raster_file, transform=raster_file.transform,
                         outfile_path=probability_raster)
            print('Prediction probability for >1cm created for', continent_name)

    raster_name = prediction_raster_keyword + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
    subsidence_arr, path = mosaic_rasters(continent_prediction_raster_dir, prediction_raster_dir, raster_name,
                                          search_by='*prediction*.tif')

    print('Global prediction raster created')

    if predict_probability_greater_1cm:
        proba_raster_name = prediction_raster_keyword + '_proba_greater_1cm_' + str(yearlist[0]) + '_' + \
                            str(yearlist[1]) + '.tif'
        mosaic_rasters(continent_prediction_raster_dir, prediction_raster_dir, proba_raster_name,
                       search_by='*proba_greater_1cm*.tif')
        print('Global prediction probability raster created')
