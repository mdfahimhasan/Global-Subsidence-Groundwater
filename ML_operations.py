# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, \
    precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.inspection import plot_partial_dependence
from Raster_operations import *
from System_operations import *

referenceraster = '../Data/Reference_rasters_shapes/Global_continents_ref_raster_002.tif'


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
    if not skip_dataframe_creation:
        predictors = glob(os.path.join(input_raster_dir, search_by))

        predictor_dict = {}
        for predictor in predictors:
            variable_name = predictor[predictor.rfind(os.sep) + 1:predictor.rfind('.')]
            raster_arr, file = read_raster_arr_object(predictor, get_file=True)
            raster_arr = raster_arr.flatten()
            predictor_dict[variable_name] = raster_arr

        predictor_df = pd.DataFrame(predictor_dict)
        predictor_df = predictor_df.dropna(axis=0)
        predictor_df = reindex_df(predictor_df)
        predictor_df.to_csv(output_csv, index=False)

        print('Predictors csv created')
        return predictor_df
    else:
        predictor_df = pd.read_csv(output_csv)
        return predictor_df


def split_train_test_ratio(predictor_csv, exclude_columns=[], pred_attr='Subsidence', test_size=0.3, random_state=0,
                           outdir=None):
    """
    Split dataset into train and test data based on a ratio

    parameters:
    input_csv : Input csv (with filepath) containing all the predictors.
    exclude_columns : Tuple of columns not included in training the model.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.

    Returns: X_train, X_test, y_train, y_test
    """
    input_df = pd.read_csv(predictor_csv)
    drop_columns = exclude_columns + [pred_attr]
    print('Dropping Columns-', exclude_columns)
    x = input_df.drop(columns=drop_columns)
    y = input_df[pred_attr]
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

    return x_train, x_test, y_train, y_test


def hyperparameter_optimization(x_train, y_train, folds=5, n_iter=50, random_search=True):
    """
    Hyperparameter optimization using RandomizedSearchCV/GridSearchCV.

    Parameters:
    x_train, y_train : x_train (predictor) and y_train (target) arrays from split_train_test_ratio function.
    folds : Number of folds in K Fold CV. Default set to 5.
    n_iter : Number of parameter combinations to be tested in RandomizedSearchCV.
    random_search : Set to False if want to perform GridSearchCV. Default set to True to perform RandomizedSearchCV.

    Returns : Optimized Hyperparameters (currently n_estimator and max_depth).
    """

    n_estimators = [200]
    max_depth = [5, 11, 13, 15, 17, 19, 20, 25]
    max_features = [4, 5, 6, 7, 8, 10]
    min_samples_split = [0.9, 0.8, 0.7, 2]
    min_samples_leaf = [5e-4, 1e-5, 2, 6, 12, 20, 25, 30, 50]

    param_dict = {
                   'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'max_features': max_features,
                   'min_samples_leaf': min_samples_leaf,
                   # 'min_samples_split': min_samples_split
                    }

    pprint(param_dict)

    rf_classifier = RandomForestClassifier(random_state=0)

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

    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    if random_search:
        CV = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dict, n_iter=n_iter,
                                cv=kfold, verbose=1, random_state=0, n_jobs=-1,
                                scoring='f1_macro', refit=True, return_train_score=True)
    else:
        CV = GridSearchCV(estimator=rf_classifier, param_grid=param_dict, cv=kfold, verbose=1, n_jobs=-1,
                          scoring='f1_macro', refit=True, return_train_score=True)
    CV.fit(x_train, y_train)

    print('\n')
    print('best parameters for macro f1 value ', '\n')
    pprint(CV.best_params_)
    print('best macro f1 score', CV.best_score_)
    print('\n')
    print('mean_test_macro_f1_score', CV.cv_results_['mean_test_score'][CV.best_index_])
    print('mean_train_macro_f1_score', CV.cv_results_['mean_train_score'][CV.best_index_])

    optimized_estimators = CV.best_params_['n_estimators']
    optimized_depth = CV.best_params_['max_depth']
    optimized_max_features = CV.best_params_['max_features']
    optimized_samples_leaf = CV.best_params_['min_samples_leaf']
    # optimized_samples_split = CV.best_params_['min_samples_split']

    return optimized_estimators, optimized_depth, optimized_max_features, optimized_samples_leaf


def build_ml_classifier(predictor_csv, modeldir, exclude_columns=(), model='RF', load_model=False,
                        pred_attr='Subsidence', test_size=0., random_state=0, output_dir=None,
                        n_estimators=300, min_samples_leaf=1, min_samples_split=2, max_depth=20, max_features='auto',
                        bootstrap=True, oob_score=True, n_jobs=-1, class_weight='balanced',
                        accuracy_dir=r'../Model Run/Accuracy_score', cm_name='cmatrix.csv',
                        predictor_importance=False, predictor_imp_keyword='RF',
                        plot_pdp=False, plot_confusion_matrix=True,
                        tune_hyperparameter=False, k_fold=5, n_iter=70, random_searchCV=True):
    """
    Build Machine Learning Classifier. Can run 'Random Forest', 'Extra Trees Classifier' and 'XGBClassifier'.

    Parameters:
    predictor_dataframe_csv : Predictor csv (with filepath) containing all the predictors.
    modeldir : Model directory to store/load model.
    exclude_columns : Tuple of columns not included in training the model.
    model : Machine learning model to run. Choose from 'RF'/ETC'/'XGBC'. Default set to 'RF'.
    load_model : Set True to load existing model. Default set to False for new model creation.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    shuffle : Whether or not to shuffle data before splitting. Defaults to True.
    output_dir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    n_estimators : Number of trees in the forest. Defaults to 450.
    min_samples_leaf : Minimum number of samples required to be at a leaf node. Defaults to 1.
    min_samples_split : Minimum number of samples required to split an internal node. Defaults to 2.
    max_features : The number of features to consider when looking for the best split. Defaults to 'log2'.
    max_depth : max_length of tree. Default set to 76.
    bootstrap : Whether bootstrap samples are used when building trees. Defaults to True.
    oob_score : Whether to use out-of-bag samples to estimate the generalization accuracy. Defaults to True.
    n_jobs : The number of jobs to run in parallel. Defaults to -1(using all processors).
    class_weight : To assign class weight. Default set to 'balanced'.
    accuracy_dir : Confusion matrix directory. If save=True must need a accuracy_dir.
    cm_name : Confusion matrix name. Defaults to 'cmatrix.csv'.
    predictor_importance : Set True if predictor importance plot is needed. Defaults to False.
    predictor_imp_keyword : Keyword to save predictor important plot.
    plot_save_keyword : Keyword to sum before saved PDP plots.
    plot_pdp : Set to True if want to plot PDP.
    plot_confusion_matrix : Set to True if want to plot confusion matrix.
    tune_hyperparameter : Set to True to tune hyperparameter. Default set to False.
    k_fold : number of folds in K-fold CV. Default set to 5.
    n_iter : Number of parameter combinations to be tested in RandomizedSearchCV. Default set to 70.
    random_searchCV : Set to False if want to perform GridSearchCV. Default set to True to perform RandomizedSearchCV.

    Returns: rf_classifier (A fitted random forest model)
    """

    # Splitting Training and Testing Data
    x_train, x_test, y_train, y_test = split_train_test_ratio(predictor_csv=predictor_csv,
                                                              exclude_columns=exclude_columns, pred_attr=pred_attr,
                                                              test_size=test_size, random_state=random_state,
                                                              outdir=output_dir)
    # Making directory for model
    makedirs([modeldir])
    model_file = os.path.join(modeldir, model)

    if tune_hyperparameter:
        n_estimators, max_depth, max_features, min_samples_leaf = \
            hyperparameter_optimization(x_train, y_train, folds=k_fold, n_iter=n_iter, random_search=random_searchCV)

    # Machine Learning Models
    if not load_model:
        if model == 'RF':
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                max_depth=max_depth, random_state=random_state,
                                                bootstrap=bootstrap, class_weight=class_weight, n_jobs=n_jobs,
                                                oob_score=oob_score)


        classifier = classifier.fit(x_train, y_train)

        pickle.dump(classifier, open(model_file, mode='wb+'))

    else:
        classifier = pickle.load(open(model_file, mode='rb'))

    classification_accuracy(x_train, x_test, y_train, y_test, classifier, accuracy_dir, cm_name,
                            predictor_importance, predictor_imp_keyword, plot_confusion_matrix)
    if plot_pdp:
        pdp_plot(classifier, x_train, accuracy_dir, plot_save_keyword=predictor_imp_keyword)

    return classifier


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
        font = 18
        label = np.array(['<1cm', '1-5 cm', '>5cm'])
        disp = ConfusionMatrixDisplay(cm_test, display_labels=label)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap='YlGn', ax=ax)
        for labels in disp.text_.ravel():
            labels.set_fontsize(font)
        ax.set_xlabel('Predicted Class', fontsize=font)
        ax.set_ylabel('True Class', fontsize=font)
        ax.set_xticklabels(label, fontsize=font)
        ax.set_yticklabels(label, fontsize=font)

        plot_name = cm_name_test[:cm_name_test.rfind('.')] + '.png'
        plt.savefig((accuracy_dir + '/' + plot_name), dpi=600)

    # print Overall accuracy
    overall_accuracy = round(accuracy_score(y_test, y_pred), 2)
    print('Accuracy Score {}'.format(overall_accuracy))

    # Saving model accuracy for individual classes
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
                          'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI',
                          'Global_Sediment_Thickness': 'Sediment Thickness',
                          'Global_Sed_Thickness_Exx': 'Sediment Thickness Exxon',
                          'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                          'Irrigated_Area_Density': 'Irrigated Area Density (gfsad)', 'MODIS_ET': 'MODIS ET',
                          'Irrigated_Area_Density2': 'Irrigated Area Density',
                          'MODIS_PET': 'MODIS PET', 'NDWI': 'NDWI', 'Population_Density': 'Population Density',
                          'SRTM_Slope': 'Slope', 'Subsidence': 'Subsidence',
                          'TRCLM_RET': 'TRCLM RET (mm)', 'TRCLM_precp': 'Precipitation',
                          'TRCLM_soil': 'Soil moisture', 'TRCLM_Tmax': 'Tmax',
                          'TRCLM_Tmin': 'Tmin', 'MODIS_Land_Use': 'MODIS Land Use', 'TRCLM_ET': 'TRCLM ET (mm)'}
        x_train_df = pd.DataFrame(x_train)
        x_train_df = x_train_df.rename(columns=predictor_dict)
        col_labels = np.array(x_train_df.columns)
        importance = np.array(classifier.feature_importances_)
        imp_dict = {'feature_names': col_labels, 'feature_importance': importance}
        imp_df = pd.DataFrame(imp_dict)
        imp_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        plt.figure(figsize=(10, 8))
        plt.rcParams['font.size'] = 20
        sns.barplot(x=imp_df['feature_importance'], y=imp_df['feature_names'], palette='rocket_r')
        plt.xlabel('Predictor Importance')
        plt.ylabel('Predictor Names')
        plt.tight_layout()
        plt.savefig((accuracy_dir + '/' + predictor_imp_keyword + '_pred_importance.png'), dpi=600)
        print('Feature importance plot saved')

    return cm_df_test, overall_accuracy


def save_model_accuracy(cm_df_test, overall_accuracy, accuracy_csv_name):
    """
    Save model accuracy parameters as csv.

    Parameters:
    cm_df_test : Confusion matrix dataframe (input from 'classification_accuracy' function).
    overall_accuracy : Overall accuracy value (input from 'classification_accuracy' function).
    accuracy_csv_name : Name of the csv file to save.

    Returns : Saved csv with model accuracy values.
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


def pdp_plot(classifier, x_train, output_dir, plot_save_keyword='RF'
             ):
    """
    Plot Partial Dependence Plot for the model.

    Parameters:
    classifier :ML model classifier.
    x_train : X train array.
    output_dir : Output directory path to save the plots.
    plot_save_keyword : Keyword to sum before saved PDP plots.

    Returns : PDP plots.
    """
    predictor_dict = {'Alexi_ET': 'Alexi ET (mm)', 'Aridity_Index': 'Aridity Index',
                      'Clay_content_PCA': 'Clay content PCA', 'EVI': 'EVI',
                      'Global_Sediment_Thickness': 'Sediment Thickness (m)',
                      'Global_Sed_Thickness_Exx': 'Sediment Thickness Exxon (km)',
                      'GW_Irrigation_Density_fao': 'GW Irrigation Density fao',
                      'GW_Irrigation_Density_giam': 'GW Irrigation Density giam',
                      'Irrigated_Area_Density': 'Irrigated Area Density', 'MODIS_ET': 'MODIS ET (mm)',
                      'MODIS_PET': 'MODIS PET (mm)', 'NDWI': 'NDWI', 'Population_Density': 'Population Density',
                      'SRTM_Slope': 'Slope (%)', 'Subsidence': 'Subsidence (cm/yr)',
                      'TRCLM_RET': 'TRCLM RET (mm)', 'TRCLM_precp': 'Precipitation (mm)',
                      'TRCLM_soil': 'Soil moisture (mm)', 'TRCLM_Tmax': 'Tmax (deg C)', 'TRCLM_Tmin': 'Tmin (deg C)',
                      'MODIS_Land_Use': 'MODIS Land Use', 'TRCLM_ET': 'TRCLM ET (mm)'}

    x_train = x_train.rename(columns=predictor_dict)
    plot_names = x_train.columns.tolist()
    feature_indices = range(len(plot_names))

    plt.rcParams['font.size'] = 18

    # Class <1cm

    plot_partial_dependence(classifier, x_train, target=1, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            grid_resolution=20)
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.suptitle('Partial Dependence Plot for <1cm/yr Subsidence', size=22, y=1)
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + 'PDP less 1cm Subsidence.png'),
                dpi=300, bbox_inches='tight')
    print('pdp for <1cm saved')

    # Class 1-5cm
    plot_partial_dependence(classifier, x_train, target=5, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            grid_resolution=20)
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.suptitle('Partial Dependence Plot for 1-5cm/yr Subsidence', size=22, y=1)
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + 'PDP 1 to 5cm Subsidence.png'),
                dpi=300, bbox_inches='tight')
    print('pdp for 1-5cm saved')

    # Class >5cm
    plot_partial_dependence(classifier, x_train, target=10, features=feature_indices, feature_names=plot_names,
                            response_method='predict_proba', percentiles=(0, 1), n_jobs=-1, random_state=0,
                            grid_resolution=20)
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    fig.suptitle('Partial Dependence Plot for >5cm/yr Subsidence', size=22, y=1)
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig((output_dir + '/' + plot_save_keyword + '_' + 'PDP greater 5cm Subsidence.png'),
                dpi=300, bbox_inches='tight')
    print('pdp for >5cm saved')


def create_prediction_raster(predictors_dir, model, yearlist=[2013, 2019], search_by='*.tif',
                             continent_search_by='*continent.shp', predictor_csv_exists=False,
                             continent_shapes_dir='../Data/Reference_rasters_shapes/continent_extents',
                             prediction_raster_dir='../Model Run/Prediction_rasters',
                             exclude_columns=(), pred_attr='Subsidence',
                             prediction_raster_keyword='RF', predict_probability_greater_1cm=True):
    """
    Create predicted raster from random forest model.

    Parameters:
    predictors_dir : Predictor rasters' directory.
    model : A fitted model obtained from random_forest_classifier function.
    yearlist : List of years for the prediction.
    search_by : Predictor rasters search criteria. Defaults to '*.tif'.
    continent_search_by : Continent shapefile search criteria. Defaults to '*continent.tif'.
    predictor_csv_exists : Set to True if predictor csv already exists. Defaults set to False. Should be False if
                           list of drop columns changes.
    continent_shapes_dir : Directory path of continent shapefiles.
    prediction_raster_dir : Output directory of prediction raster.
    exclude_columns : Predictor rasters' name that will be excluded from the model. Defaults to ().
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    prediction_raster_keyword : Keyword added to final prediction raster name.
    predict_probability_greater_1cm : Set to False if probability of prediction of each classes (<1cm, 1-5cm, >5cm)
                                      is required. Default set to True to predict probability of prediction for >1cm.

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
                if variable_name not in drop_columns:
                    raster_arr, raster_file = clip_resample_raster_cutline(predictor, clipped_predictor_dir, continent,
                                                                           naming_from_both=False)
                    raster_shape = raster_arr.shape
                    raster_arr = raster_arr.reshape(raster_shape[0] * raster_shape[1])
                    nan_position_dict[variable_name] = np.isnan(raster_arr)
                    raster_arr[nan_position_dict[variable_name]] = 0
                    predictor_dict[variable_name] = raster_arr

            pickle.dump(nan_position_dict, open(dict_name, mode='wb+'))

            predictor_df = pd.DataFrame(predictor_dict)
            predictor_df = predictor_df.dropna(axis=0)
            predictor_df = reindex_df(predictor_df)
            predictor_df.to_csv(predictor_csv, index=False)

        else:
            predictor_df = pd.read_csv(predictor_csv)

            nan_position_dict = pickle.load(open(dict_name, mode='rb'))

            raster_arr, raster_file = clip_resample_raster_cutline(predictor_rasters[1], clipped_predictor_dir,
                                                                   continent, naming_from_both=False)
            raster_shape = raster_arr.shape

        x = predictor_df.values
        y_pred = model.predict(x)

        for nan_pos in nan_position_dict.values():
            y_pred[nan_pos] = raster_file.nodata
        y_pred_arr = y_pred.reshape(raster_shape)

        prediction_raster_name = continent_name + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
        predicted_raster = os.path.join(continent_prediction_raster_dir, prediction_raster_name)
        write_raster(raster_arr=y_pred_arr, raster_file=raster_file, transform=raster_file.transform,
                     outfile_path=predicted_raster)
        print('Prediction raster created for', continent_name)

        if predict_probability_greater_1cm:
            y_pred_proba = model.predict_proba(x)
            y_pred_proba = y_pred_proba[:, 1] + y_pred_proba[:, 2]

            for nan_pos in nan_position_dict.values():
                y_pred_proba[nan_pos] = raster_file.nodata
            y_pred_proba_arr = y_pred_proba.reshape(raster_shape)

            probability_raster_name = continent_name + '_proba_greater_1cm_' + str(yearlist[0]) + '_' + \
                                      str(yearlist[1]) + '.tif'
            probability_raster = os.path.join(continent_prediction_raster_dir, probability_raster_name)
            write_raster(raster_arr=y_pred_proba_arr, raster_file=raster_file, transform=raster_file.transform,
                         outfile_path=probability_raster)
            print('Prediction probability for >1cm created for', continent_name)

    raster_name = prediction_raster_keyword + '_prediction_' + str(yearlist[0]) + '_' + str(yearlist[1]) + '.tif'
    mosaic_rasters(continent_prediction_raster_dir, prediction_raster_dir, raster_name, search_by='*prediction*.tif')
    print('Global prediction raster created')

    if predict_probability_greater_1cm:
        proba_raster_name = prediction_raster_keyword + '_proba_greater_1cm_' + str(yearlist[0]) + '_' + \
                            str(yearlist[1]) + '.tif'
        mosaic_rasters(continent_prediction_raster_dir, prediction_raster_dir, proba_raster_name,
                       search_by='*proba_greater_1cm*.tif')
        print('Global prediction probability raster created')

