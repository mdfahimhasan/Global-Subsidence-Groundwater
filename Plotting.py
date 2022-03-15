# Author: Md Fahim Hasan
# Email: mhm4b@mst.edu

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def country_subsidence_barplot(country_stat_excel, number_of_countries=26):
    """
    Create bar plots of subsidence stats for countries.

    Parameters:
    country_stat_excel : Filepath of excel file with countries' subsidence stats.
    number_of_countries : Number of country to plot in bar plot. Default set to 26.

    Returns: Bar plots showing countries' stats.
    """
    stat = pd.read_excel(country_stat_excel, sheet_name=0)
    stat['area subsidence >1cm/yr'] = stat['area subsidence >1cm/yr'].astype('int')
    stat_1 = stat.sort_values('perc_subsidence_of_cntry_area', ascending=False)
    stat_highest_1 = stat_1.iloc[0: number_of_countries-1, :]

    fig, axs = plt.subplots(2, figsize=(16, 8))
    sns.barplot(x='country_name', y='perc_subsidence_of_cntry_area', data=stat_highest_1, palette='Blues_r', ax=axs[0])
    axs[0].bar_label(axs[0].containers[0], fmt='%.2f', fontsize=6)
    axs[0].set_xticks(range(len(stat_highest_1['country_name'])), list(stat_highest_1['country_name']), rotation=90)
    axs[0].set_xlabel('(a)', fontsize=12)
    axs[0].set_ylabel('% area of country subsiding >1cm/year', labelpad=15)

    stat_2 = stat.sort_values('area subsidence >1cm/yr', ascending=False)
    stat_highest_2 = stat_2.iloc[0: number_of_countries-1, :]
    sns.barplot(x='country_name', y='area subsidence >1cm/yr', data=stat_highest_2, palette='Purples_r', ax=axs[1])
    axs[1].bar_label(axs[1].containers[0], fmt='%.f', fontsize=6)
    axs[1].set_yscale('log')
    axs[1].set_xticks(range(len(stat_highest_2['country_name'])), list(stat_highest_2['country_name']), rotation=90)
    axs[1].set_xlabel('(b)', fontsize=12)
    axs[1].set_ylabel('area (sqkm) of country subsiding >1cm/year \n (log-scale)')

    plt.tight_layout(pad=1, w_pad=1, h_pad=1)

    plot_name = '../Model Run/Stats' + '/' + 'top_subsidence_stat_by_countries.png'
    plt.savefig(plot_name, dpi=400, bbox_inches='tight')


# country_subsidence_barplot(country_stat_excel='../Model Run/Stats/subsidence_area_by_country.xlsx',
#                            number_of_countries=26)


