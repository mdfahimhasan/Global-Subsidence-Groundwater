# Introduction

Groundwater overdraft gives rise to multiple adverse impacts including land subsidence and permanent groundwater storage loss. Existing methods are unable to characterize groundwater storage loss at the global scale with sufficient resolution to be relevant for local studies. Here we explore the interrelation between groundwater stress, aquifer depletion, and land subsidence using remote sensing and model-based datasets with a machine learning approach. The developed model predicts global land subsidence magnitude at high spatial resolution (~2 km), provides a first-order estimate of aquifer storage loss due to consolidation of ~17 km3/year globally, and quantifies key drivers of subsidence. Roughly 73% of the mapped subsidence occurs over cropland and urban areas, highlighting the need for sustainable groundwater management practices over these areas. The results of this study aid in assessing the spatial extents of subsidence in known subsiding areas, and in locating unknown groundwater stressed regions.

This repository consists of python scripts that executes the *Global-Subsidence-Groundwater* model. It includes codes for automatic data downloading & preprocessing (Data_operations), raster processing (Raster_operations), machine learning model (ML_operations), training data processing (Training_InSAR_processing), result analysis (Result_Analysis), etc. The Model_Driver scripts runs the entire model. Some scripts like Training_InSAR_processing, Result_Analysis, Plotting have to be run separately. High resolution maps are provided in the Maps folder.

The results of the *Global-Subsidence-Groundwater* model has been published in a research artcile in __Nature Communications__. The manuscript can be found in this link: __https://rdcu.be/dnIKQ__.

The datasets used for training the model, the input variables, and the model-generated datasets are avaiable at this HydroShare repository: 
https://www.hydroshare.org/resource/dc7c5bfb3a86479b889d3b30ab0e4ef7/


# Model Prediction
![prediction_map_region_based](https://github.com/mdfahimhasan/GlobalGW_Subsidence/assets/77580408/0d6f25ec-c2e4-4af2-8632-ded375c284a3)


# Workflow

![Nature Global GW Workflow_with LU filter](https://github.com/mdfahimhasan/GlobalGW_Subsidence/assets/77580408/7cc10f43-5a8f-463b-8e84-114f6aa0d473)

# Setting up model environment
We recommend using [Anaconda](https://www.anaconda.com/products/individual) environment for running the model. The environment will need to be set up with Python 3 and some dependencies. The dependencies are listed in the GlobalGW.yml file.

# To cite the Article
Hasan, M.F., Smith, R., Vajedian, S. et al. Global land subsidence mapping reveals widespread loss of aquifer storage capacity. Nat Commun 14, 6180 (2023). https://doi.org/10.1038/s41467-023-41933-z

# To cite the Code/Model
Hasan, M. F., Smith, R., Vajedian, S., Pommerenke, R., Majumdar, S., Global Land Subsidence Mapping Reveals Widespread Loss of Aquifer Storage Capacity, GitHub (2023) doi: 10.5281/zenodo.8280482


# To cite the Datasets
Hasan, M. F., Smith, R., Vajedian, S., Pommerenke, R., Majumdar, S., Global Land Subsidence Mapping Reveals Widespread Loss of Aquifer Storage Capacity Datasets, HydroShare (2023) doi: https://doi.org/10.4211/hs.dc7c5bfb3a86479b889d3b30ab0e4ef7

# Affiliation

<img src="https://user-images.githubusercontent.com/77580408/216176949-71a889cd-8926-4c19-8cd4-cece55303931.png" width="8%" height="8%" /> &nbsp;   <img src="https://user-images.githubusercontent.com/77580408/216177156-66d191fb-6c7a-4e84-ba1b-4291767864bb.png" width="10%" height="10%" /> &nbsp;  <img src="https://www.dri.edu/wp-content/uploads/Official-DRI-Logo-for-Web.png" width="12%" height="12%" /> 



# Funding Agency

<img src="https://engineering.ucsb.edu/sites/default/files/styles/large/public/images/events/NGIALogo-277siiq.png?itok=c0wrYb1A" width="15%" height="15%" />
