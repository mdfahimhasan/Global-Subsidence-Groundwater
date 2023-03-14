# Author: Md Fahim Hasan
# Email: Fahim.Hasan@colostate.edu

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from System_operations import makedirs
from ML_operations import split_train_test_ratio, build_ml_classifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def country_subsidence_barplot(country_stat_excel, number_of_countries=30):
    """
    Create bar plots of subsidence stats for countries.

    Parameters:
    country_stat_excel : Filepath of excel file with countries' subsidence stats.
    number_of_countries : Number of country to plot in bar plot. Default set to 26.

    Returns: Bar plots showing countries' stats.
    """
    stat = pd.read_excel(country_stat_excel, sheet_name='countryarea_corrected')
    stat = stat.dropna(axis=0, how='any')
    stat['area subsidence >1cm/yr'] = stat['area subsidence >1cm/yr'].astype('int')
    stat = stat.drop(columns=['perc_subsidence_of_cntry_area'])
    stat['perc_subsidence_of_cntry_area'] = round((stat['area subsidence >1cm/yr'] * 100 / stat['area_sqkm_google']), 2)
    stat_1 = stat.sort_values('perc_subsidence_of_cntry_area', ascending=False)
    stat_highest_1 = stat_1.iloc[0: number_of_countries - 1, :]

    fig, axs = plt.subplots(2, figsize=(16, 10))

    sns.barplot(x='country_name', y='perc_subsidence_of_cntry_area', data=stat_highest_1, palette='Blues_r', ax=axs[0])
    axs[0].bar_label(axs[0].containers[0], fmt='%.2f', fontsize=10, padding=0)
    axs[0].set_xticks(range(len(stat_highest_1['country_name'])), list(stat_highest_1['country_name']), rotation=90)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].set_xlabel('(a)', fontsize=20)
    axs[0].set_ylabel('% area of country \n subsiding >1cm/year', labelpad=15, fontsize=18)

    stat_2 = stat.sort_values('area subsidence >1cm/yr', ascending=False)
    stat_highest_2 = stat_2.iloc[0: number_of_countries - 1, :]
    sns.barplot(x='country_name', y='area subsidence >1cm/yr', data=stat_highest_2, palette='Purples_r', ax=axs[1])
    axs[1].bar_label(axs[1].containers[0], fmt='%.f', fontsize=10, padding=0)
    axs[1].set_yscale('log')
    axs[1].set_xticks(range(len(stat_highest_2['country_name'])), list(stat_highest_2['country_name']), rotation=90)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].set_xlabel('(b)', fontsize=20)
    axs[1].set_ylabel('area (sqkm) of country \n subsiding >1cm/year \n (log-scale)', fontsize=18)

    plt.tight_layout(pad=1, w_pad=1, h_pad=1)

    plot_name = '../Model Run/Stats' + '/' + 'top_subsidence_stat_by_countries.png'
    plt.savefig(plot_name, dpi=500, bbox_inches='tight')


country_subsidence_barplot(country_stat_excel='../Model Run/Stats/country_area_record_google.xlsx',
                           number_of_countries=30)


def country_subsidence_barplot_type_02(country_stat_excel, gw_loss_excel, number_of_countries=10):
    """
    Create bar plots of subsidence stats for countries.

    Parameters:
    country_stat_excel : Filepath of excel file with countries' subsidence stats.
    gw_loss_excel : Filepath of excel file with countries' groundwater volume loss stats.
    number_of_countries : Number of country to plot in bar plot. Default set to 10.

    Returns: Bar plots showing countries' stats.
    """

    stat = pd.read_excel(country_stat_excel, sheet_name='countryarea_corrected')
    stat = stat.dropna(axis=0, how='any')
    stat['area subsidence >1cm/yr'] = stat['area subsidence >1cm/yr'].astype('int')
    stat = stat.drop(columns=['perc_subsidence_of_cntry_area'])
    stat['perc_subsidence_of_cntry_area'] = round((stat['area subsidence >1cm/yr'] * 100 / stat['area_sqkm_google']), 2)

    gw_stat = pd.read_excel(gw_loss_excel, sheet_name='Sheet1')
    gw_stat = gw_stat.dropna(axis=0, how='any')

    fig, axs = plt.subplots(2, figsize=(12, 10))

    stat_1 = stat.sort_values('perc_subsidence_of_cntry_area', ascending=False)
    stat_highest_1 = stat_1.iloc[0: number_of_countries, :]
    sns.barplot(x='country_name', y='perc_subsidence_of_cntry_area', data=stat_highest_1, palette='Blues_r', ax=axs[0])
    axs[0].bar_label(axs[0].containers[0], fmt='%.2f', fontsize=10, padding=0.5)
    axs[0].margins(y=0.1)  # make room for the labels
    axs[0].set_xticks(range(len(stat_highest_1['country_name'])), list(stat_highest_1['country_name']), rotation=90)
    axs[0].tick_params(axis='both', which='major', labelsize=18)
    axs[0].set_xlabel('(a)', fontsize=18)
    axs[0].set_ylabel('% area of country \n subsiding >1cm/year', labelpad=15, fontsize=18)

    stat_2 = gw_stat.sort_values('volume avg total gw loss (km3/yr)', ascending=False)
    stat_highest_2 = stat_2.iloc[0: number_of_countries, :]
    sns.barplot(x='CNTRY_NAME', y='volume avg total gw loss (km3/yr)', data=stat_highest_2, palette='Purples_r',
                ax=axs[1])
    axs[1].bar_label(axs[1].containers[0], fmt='%.2f', fontsize=10, padding=0.1)
    axs[1].margins(y=0.1)  # make room for the labels
    axs[1].set_yscale('log')
    axs[1].set_xticks(range(len(stat_highest_2['CNTRY_NAME'])), list(stat_highest_2['CNTRY_NAME']), rotation=90)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    axs[1].set_xlabel('(b)', fontsize=18)
    axs[1].set_ylabel('Groundwater Storage Loss \n due to Consolidation \n (km3/year) (log-scale)', fontsize=18)

    fig.tight_layout(pad=1, w_pad=1, h_pad=1)

    plot_name = '../Model Run/Stats' + '/' + 'top_subsidence_stat_by_countries_type_02.png'
    plt.savefig(plot_name, dpi=500, bbox_inches='tight')


# country_subsidence_barplot_type_02(country_stat_excel='../Model Run/Stats/country_area_record_google.xlsx',
#                                    gw_loss_excel='../Model Run/Stats/country_gw_volume_loss.xlsx',
#                                    number_of_countries=10)


def variable_correlation_plot(variables_to_include, method='spearman',
                              training_data_csv='../Model Run/Predictors_csv/train_test_2013_2019.csv',
                              output_dir='../Model Run/Stats'):
    """
    Makes correlation heatmap of variables (predictors) used in the model using pearson's/spearman's correlation method.
    *** pearson's method consider linear relationship between variables while spearman's method consider non-linear
    relationship.

    Parameters:
    variables_to_include: A list  of variables. Variables are those what were used in the final model.
    method: pearson or spearman.
    training_data_csv: Filepath of training data csv.
    output_dir: Filepath of output dir to save the plot.

    Returns: A heatmap of correlation between variables.
    """
    training_df = pd.read_csv(training_data_csv)

    training_df = training_df[variables_to_include]
    corr_coef = round(training_df.corr(method=method), 2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_coef, cmap='coolwarm', annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_correlation.jpeg'), dpi=200)

    # Calculating total value of absolute
    corr_coef = round(training_df.corr(method=method).abs(), 2)
    corr_coef['sum'] = corr_coef.sum() - 1  # deleting 1 to remove self correlation
    corr_coef.to_csv('../Model Run/Stats/variable_correlation.csv')


# # Give list of predictors used in the model
# columns_to_plot = ['% Slope', 'Aridity Index', 'Clay Thickness (m)', 'Confining Layers',
#                    'EVI', 'Irrigated Area Density', 'NDWI', 'Population Density',
#                    'Precipitation (mm)', 'River Distance (km)', 'Soil moisture (mm)',
#                    'TRCLM ET (mm)', 'Tmax (°C)']

# variable_correlation_plot(variables_to_include=columns_to_plot,
#                           training_data_csv='../Model Run/Predictors_csv/train_test_2013_2019.csv',
#                           output_dir='../Model Run/Stats')


def plot_permutation_importance(train_test_csv='../Model Run/Predictors_csv/train_test_2013_2019.csv',
                                output_dir='../Model Run/permutation_importance',
                                exclude_columns=('Alexi ET', 'MODIS ET (kg/m2)', 'Irrigated Area Density (gfsad)',
                                                 'GW Irrigation Density giam', 'MODIS PET (kg/m2)',
                                                 'Clay content PCA', 'MODIS Land Use', 'Sediment Thickness (m)',
                                                 'Grace', 'Clay % 200cm', 'Tmin (°C)', 'TRCLM RET (mm)'),
                                plot_keyword='RF'):
    """
    Plot permutation importance for model predictors.

    resources:
    # https://explained.ai/rf-importance/
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    # https://towardsdatascience.com/from-scratch-permutation-feature-importance-for-ml-interpretability-b60f7d5d1fe9
    # https://christophm.github.io/interpretable-ml-book/feature-importance.html

    Parameters:
    train_test_csv: Filepath of training data csv.
    output_dir: Filepath of output directory to save results/plots.
    exclude_columns: Tuple of columns to exclude from training dataset. Use the same excluded columns that were dropped
                     during model training.
    plot_keyword: keyword to add in saved plot. Default set to 'RF'.

    Returns: None.
    """
    modeldir = '../Model Run/Model'
    makedirs([output_dir])
    model = 'rf'

    x_train, x_test, y_train, y_test, _ = \
        split_train_test_ratio(predictor_csv=train_test_csv, exclude_columns=exclude_columns,
                               pred_attr='Subsidence', test_size=0.3, random_state=0,
                               outdir=output_dir, verbose=False)

    trained_rf, predictor_name_dict = \
        build_ml_classifier(train_test_csv, modeldir, exclude_columns, model, load_model=False,
                            pred_attr='Subsidence', test_size=0.3, random_state=0, output_dir=output_dir,
                            n_estimators=300, min_samples_leaf=1e-05, min_samples_split=7, max_depth=14,
                            max_features=7, class_weight='balanced',
                            max_samples=None, max_leaf_nodes=None,
                            estimate_accuracy=False, predictor_imp_keyword=None, predictor_importance=False,
                            variables_pdp=None, plot_pdp=False,
                            plot_confusion_matrix=True)

    predictor_cols = pd.read_csv('../Model Run/permutation_importance/X_test.csv').columns

    # Permutation importance on test set
    result_test = permutation_importance(
        trained_rf, x_test, y_test, n_repeats=30, random_state=0, n_jobs=-1)

    sorted_importances_idx = result_test.importances_mean.argsort()
    importances = pd.DataFrame(result_test.importances[sorted_importances_idx].T,
                               columns=predictor_cols[sorted_importances_idx])
    plt.figure(figsize=(10, 8))
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importance (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Relative change in accuracy")
    ax.figure.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{plot_keyword}_permutation_imp_test.jpeg'), dpi=200)

    # Permutation importance on train set
    result_train = permutation_importance(
        trained_rf, x_train, y_train, n_repeats=30, random_state=0, n_jobs=-1)

    sorted_importances_idx = result_train.importances_mean.argsort()
    importances = pd.DataFrame(result_train.importances[sorted_importances_idx].T,
                               columns=predictor_cols[sorted_importances_idx])
    plt.figure(figsize=(10, 8))
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importance (train set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Relative change in accuracy")
    ax.figure.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{plot_keyword}_permutation_imp_train.jpeg'), dpi=200)


# # Plot permutation importance
# # change for fitted_model run
# drop_columns = ['Alexi ET', 'MODIS ET (kg/m2)', 'Irrigated Area Density (gfsad)', 'GW Irrigation Density giam',
#                 'MODIS PET (kg/m2)', 'Clay content PCA', 'MODIS Land Use', 'Grace', 'Sediment Thickness (m)',
#                 'Clay % 200cm', 'Tmin (°C)', 'TRCLM RET (mm)']
# plot_permutation_importance(exclude_columns=drop_columns, plot_keyword='RF_132')


def plot_soil_pdp_combinations(train_test_csv='../Model Run/Predictors_csv/train_test_2013_2019.csv',
                               output_dir='../Model Run/PDP_combinations',
                               exclude_columns=('Alexi ET', 'MODIS ET (kg/m2)', 'Irrigated Area Density (gfsad)',
                                                'GW Irrigation Density giam', 'MODIS PET (kg/m2)',
                                                'Clay content PCA', 'MODIS Land Use', 'Grace',
                                                'Sediment Thickness (m)', 'Clay % 200cm', 'Tmin (°C)',
                                                'TRCLM RET (mm)'),
                               plot_combinations=(('TRCLM ET (mm)', 'Soil moisture (mm)'),
                                                  ('Precipitation (mm)', 'Soil moisture (mm)'),
                                                  ('Irrigated Area Density', 'Soil moisture (mm)')),
                               plot_keyword='soil'):
    """
    Plots partial dependence plot for combinations of soil mistures and related variables.

    Parameters:
    train_test_csv: Filepath of training data csv.
    output_dir: Filepath of output directory to save results/plots.
    exclude_columns: Tuple of columns to exclude from training dataset. Use the same excluded columns that were dropped
                     during model training.
    plot_combinations: Tuple of tuples of predictor combinations.
    plot_keyword: Keyword to use in saving plot. Default set to 'soil'.

    Returns: None
    """
    modeldir = '../Model Run/Model'
    makedirs([output_dir])
    model = 'rf'

    x_train, x_test, y_train, y_test, _ = \
        split_train_test_ratio(predictor_csv=train_test_csv, exclude_columns=exclude_columns,
                               pred_attr='Subsidence', test_size=0.3, random_state=0,
                               outdir=output_dir, verbose=False)

    trained_rf, predictor_name_dict = \
        build_ml_classifier(train_test_csv, modeldir, exclude_columns, model, load_model=False,
                            pred_attr='Subsidence', test_size=0.3, random_state=0, output_dir=output_dir,
                            n_estimators=300, min_samples_leaf=1e-05, min_samples_split=7, max_depth=14,
                            max_features=7, class_weight='balanced',
                            max_samples=None, max_leaf_nodes=None,
                            estimate_accuracy=False, predictor_imp_keyword=None, predictor_importance=False,
                            variables_pdp=None, plot_pdp=False,
                            plot_confusion_matrix=True)

    prediction_class = [5]
    pdisp = PartialDependenceDisplay.from_estimator(trained_rf, x_train, features=plot_combinations,
                                                    target=prediction_class[0], response_method='predict_proba',
                                                    percentiles=(0.05, 0.95), n_jobs=-1, random_state=0,
                                                    grid_resolution=20)
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    pdisp.plot(ax=ax)
    ax[0].set_title('(a)', y=-0.2)
    ax[1].set_title('(b)', y=-0.2)
    ax[2].set_title('(c)', y=-0.2)
    fig = plt.gcf()
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.subplots_adjust(wspace=0.35, hspace=0.3)

    # Formatting colorbar
    import matplotlib.ticker as tick
    cbar = fig.colorbar((pdisp.contours_[0]), ax=ax)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    cbar.set_label('Probability of Subsidence between 1-5 cm/year', rotation=90, labelpad=15)

    pdp_plot_name = '1_5cm_subsidence_proba.png'
    fig.savefig((os.path.join(output_dir, f'{plot_keyword}_PDP_{pdp_plot_name}')), dpi=400, bbox_inches='tight')
    print(pdp_plot_name.split('.')[0], 'saved')

# # Plot PDP combinations for soil moisture
# drop_columns = ['Alexi ET', 'MODIS ET (kg/m2)', 'Irrigated Area Density (gfsad)', 'GW Irrigation Density giam',
#                 'MODIS PET (kg/m2)', 'Clay content PCA', 'MODIS Land Use', 'Grace', 'Sediment Thickness (m)',
#                 'Clay % 200cm', 'Tmin (°C)', 'TRCLM RET (mm)']
# plot_soil_pdp_combinations(exclude_columns=drop_columns)
