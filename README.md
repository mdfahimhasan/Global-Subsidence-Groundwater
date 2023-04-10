# Introduction

This project employs a Random Forest Machine Learning model to map land subsidence globally at high resolution (~2 km resolution) that is induced by groundwater pumping. The datasets used in the model are publicly available remote sensing and model-based data. The datasets are available at the HydroShare repository: http://www.hydroshare.org/resource/f25fc4c9b1224314a6a037e85052bcbb 

The python scripts include codes for automatic data downloading & preprocessing (Data_operations), raster processing (Raster_operations), machine learning model (ML_operations), training data processing (Training_InSAR_processing), result analysis (Result_Analysis), etc. The Model_Driver scripts runs the entire model. Some scripts like Training_InSAR_processing, Result_Analysis, Plotting have to be run separately. High resolution maps are provided in the Maps folder.


# Setting up model environment
We recommend using [Anaconda](https://www.anaconda.com/products/individual) environment for running the model. The environment will need to be set up with Python 3 and some dependencies. The dependencies are listed in the GlobalGW.yml file.



# Model Prediction
![prediction_map_region_based](https://user-images.githubusercontent.com/77580408/230847670-ee91158c-e49d-416a-a75f-0d72c197974e.png)




# Workflow

![Nature Global GW Workflow](https://user-images.githubusercontent.com/77580408/229738011-070d6c82-8cfe-4960-9239-f2464b2845b3.png)



# Affiliation

<img src="https://user-images.githubusercontent.com/77580408/216176949-71a889cd-8926-4c19-8cd4-cece55303931.png" width="8%" height="8%" /> &nbsp;   <img src="https://user-images.githubusercontent.com/77580408/216177156-66d191fb-6c7a-4e84-ba1b-4291767864bb.png" width="10%" height="10%" />


# Funding Agency

<img src="https://engineering.ucsb.edu/sites/default/files/styles/large/public/images/events/NGIALogo-277siiq.png?itok=c0wrYb1A" width="15%" height="15%" />
