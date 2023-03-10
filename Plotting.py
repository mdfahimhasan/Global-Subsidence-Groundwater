# Author: Md Fahim Hasan
# Email: Fahim.Hasan@colostate.edu

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    sns.barplot(x='CNTRY_NAME', y='volume avg total gw loss (km3/yr)', data=stat_highest_2, palette='Purples_r', ax=axs[1])
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


def variable_correlation_plot(variables_to_include,
                              training_data_csv='../Model Run/Predictors_csv/train_test_2013_2019.csv',
                              output_dir='../Model Run/Stats'):
    """
    Makes correlation heatmap of variables (predictors) used in the model.

    Parameters
    ----------
    variables_to_include: A list  of variables. Variables are those what were used in the final model.
    training_data_csv: Filepath of training data csv.
    output_dir: Filepath of output dir to save the plot.

    Returns: A heatmap of correlation between variables.
    """
    training_df = pd.read_csv(training_data_csv)

    training_df = training_df[variables_to_include]
    corr_coef = round(training_df.corr(method='pearson'), 2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_coef, cmap='coolwarm', annot=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_corr.jpeg'), dpi=200)


# # Give list of predictors used in the model
# columns_to_plot = ['% Slope', 'Aridity Index', 'Clay Thickness (m)', 'Confining Layers',
#                    'EVI', 'Grace', 'Irrigated Area Density', 'NDWI', 'Population Density',
#                    'Precipitation (mm)', 'River Distance (km)', 'Soil moisture (mm)',
#                    'TRCLM ET (mm)', 'TRCLM RET (mm)', 'Tmax (°C)']
#
# variable_correlation_plot(variables_to_include=columns_to_plot,
#                               training_data_csv='../Model Run/Predictors_csv/train_test_2013_2019.csv',
#                               output_dir='../Model Run/Stats')