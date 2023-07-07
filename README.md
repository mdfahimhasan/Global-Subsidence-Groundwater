# Introduction

This project employs a Random Forest Machine Learning model to map land subsidence globally at high resolution (~2 km) that is induced by groundwater pumping. The datasets used in the model are publicly available remote sensing and model-based data. The datasets are available at the HydroShare repository: http://www.hydroshare.org/resource/db187b7e328c4158879926d8f9a6dccd

The python scripts include codes for automatic data downloading & preprocessing (Data_operations), raster processing (Raster_operations), machine learning model (ML_operations), training data processing (Training_InSAR_processing), result analysis (Result_Analysis), etc. The Model_Driver scripts runs the entire model. Some scripts like Training_InSAR_processing, Result_Analysis, Plotting have to be run separately. High resolution maps are provided in the Maps folder.


# Setting up model environment
We recommend using [Anaconda](https://www.anaconda.com/products/individual) environment for running the model. The environment will need to be set up with Python 3 and some dependencies. The dependencies are listed in the GlobalGW.yml file.



# Model Prediction
![prediction_map_region_based](https://github.com/mdfahimhasan/GlobalGW_Subsidence/assets/77580408/0d6f25ec-c2e4-4af2-8632-ded375c284a3)





# Workflow

![Nature Global GW Workflow_with LU filter](https://github.com/mdfahimhasan/GlobalGW_Subsidence/assets/77580408/7cc10f43-5a8f-463b-8e84-114f6aa0d473)



# Affiliation

<img src="https://user-images.githubusercontent.com/77580408/216176949-71a889cd-8926-4c19-8cd4-cece55303931.png" width="8%" height="8%" /> &nbsp;   <img src="https://user-images.githubusercontent.com/77580408/216177156-66d191fb-6c7a-4e84-ba1b-4291767864bb.png" width="10%" height="10%" />


# Funding Agency

<img src="https://engineering.ucsb.edu/sites/default/files/styles/large/public/images/events/NGIALogo-277siiq.png?itok=c0wrYb1A" width="15%" height="15%" />
