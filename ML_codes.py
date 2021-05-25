import os
from glob import glob
import Raster_operations as rops
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
#from sklearn.svm import SVC
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence

def create_dataframe(input_raster_dir,output_csv,exclude_columns=(),pattern="*.tif"):
    """
    create dataframe from predictor rasters.

    Parameters:
    input_raster_dir : Input rasters directory.
    output_csv : Output csv file with filepath.
    exclude_columns : Tuple of columns not included in the final csv.
    pattern : Input raster search pattern. Defaults to "*.tif".

    Returns: 
    predictor_df : Dataframe created from predictor rasters.
    """
    rasters=glob(os.path.join(input_raster_dir,pattern))
    predictor_dict={}
    for raster in rasters:
        variable_name=raster[raster.rfind(os.sep)+1:raster.rfind(".")]
        raster_arr,file=rops.read_raster_arr_object(raster,get_file=True)
        raster_arr=raster_arr.flatten()
        predictor_dict[variable_name]=raster_arr
        
    drop_columns=list(exclude_columns) 
    predictor_df=pd.DataFrame(predictor_dict)
    predictor_df=predictor_df.dropna(axis=0) 
    predictor_df=predictor_df.drop(columns=drop_columns)
    predictor_df.to_csv(output_csv,index=False)
    
    return predictor_df


def split_train_test_ratio(input_csv,pred_attr='Subsidence_G5_L5',test_size=0.3,random_state=0,
                                                      shuffle=True,outdir=None):
    """
    Split dataset into train and test data based on a ratio

    parameters:
    input_csv : Input csv (with filepath) containing all the predictors.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    shuffle : Whether or not to shuffle data before spliting. Defaults to True.
    outdir : Set a output directory if training and test dataset need to be saved. Defaults to None.

    Returns: X_train, X_test, y_train, y_test
    """
    input_df=pd.read_csv(input_csv)
    X=input_df.drop(pred_attr,axis=1)
    y=input_df[pred_attr]
    
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,
                                                      shuffle=shuffle)
                                                      
    if outdir:
        X_train_df=pd.DataFrame(X_train)
        X_train_df.to_csv(os.path.join(outdir,'X_train.csv'),index=False)
        
        y_train_df=pd.DataFrame(y_train)
        y_train_df.to_csv(os.path.join(outdir,'y_train.csv'),index=False)        
        
        X_test_df=pd.DataFrame(X_test)
        X_test_df.to_csv(os.path.join(outdir,'X_test.csv'),index=False)
        
        y_test_df=pd.DataFrame(y_test)
        y_test_df.to_csv(os.path.join(outdir,'y_test.csv'),index=False)  
        
    return X_train, X_test, y_train, y_test


def pdp_plot(X_train,classifier,outdir,multiclass_pdp=True,#title1='PDP <5cm Subsidence.png',
             #title2='PDP >5cm Subsidence.png'
             ):
    plot_names=X_train.columns.tolist()
    feature_indices=range(len(plot_names))
    if multiclass_pdp:            
        #Class 5
        fig,ax=plt.subplots()
        ax.set_title('Partial Dependence Plot for <5cm Subsidence')
        plot_partial_dependence(classifier,X_train,target=5,features=feature_indices,feature_names=plot_names,
                                response_method='predict_proba',percentiles=(0,1),n_jobs=-1,random_state=0,
                                ax=ax) 
        plt.rcParams['font.size'] = '8'
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=3)
        #fig.savefig(os.path.join(outdir,title1),dpi=300, bbox_inches='tight')
        
        #Class 10
        fig,ax=plt.subplots()
        ax.set_title('Partial Dependence Plot for >5cm Subsidence')
        plot_partial_dependence(classifier,X_train,target=10,features=feature_indices,feature_names=plot_names,
                                response_method='predict_proba',percentiles=(0,1),n_jobs=-1,random_state=0,
                                ax=ax) 
        plt.rcParams['font.size'] = '8'
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=3)
        #fig.savefig(os.path.join(outdir,title2),dpi=300, bbox_inches='tight')
            
    else:
        fig,ax=plt.subplots()
        plot_partial_dependence(classifier,X_train,features=feature_indices,feature_names=plot_names,
                                response_method='Predict_proba',percentiles=(0,1),n_jobs=-1,random_state=0,
                                ax=ax) 
        plt.rcParams['font.size'] = '8'
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=3)


def random_forest_classifier(input_csv,plot_save=None,pred_attr='Subsidence_G5_L5',test_size=0.3,random_state=0,shuffle=True,
                             outdir=None,n_estimators=500,bootstrap=True,oob_score=True,
                             n_jobs=-1,max_features=None,multiclass=False,save=False,cm_csv=None,
                             predictor_importance=False,pdp=False,multiclass_pdp=True):
    """
    Random Forest Classifier.

    Parameters:
    input_csv : Input csv (with filepath) containing all the predictors.
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.
    test_size : The percentage of test dataset. Defaults to 0.3.
    random_state : Seed value. Defaults to 0.
    shuffle : Whether or not to shuffle data before spliting. Defaults to True.
    outdir : Set a output directory if training and test dataset need to be saved. Defaults to None.
    n_estimators : The number of trees in the forest.. Defaults to 500.
    bootstrap : Whether bootstrap samples are used when building trees. Defaults to True.
    oob_score : Whether to use out-of-bag samples to estimate the generalization accuracy. Defaults to True.
    n_jobs : The number of jobs to run in parallel. Defaults to -1(using all processors).
    max_features : The number of features to consider when looking for the best split. Defaults to None.
    multiclass : If multiclass classification, set True for getting model performance. Defaults to False.
    save : Set True to save confusion matrix as csv. Defaults to False.
    cm_csv : Confusion matrix csv filepath. If save=True must need a cm_csv. Defaults to None.
    predictor_importance : Set True if predictor importance plot is needed. Defaults to False.

    Returns: rf_classifier (A fitted random forest model)

    """
    
    X_train, X_test, y_train, y_test=split_train_test_ratio(input_csv=input_csv,pred_attr=pred_attr,
                                                            test_size=test_size,random_state=random_state,
                                                            shuffle=shuffle,outdir=outdir)
    rf_classifier=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state,bootstrap=bootstrap,
                              n_jobs=n_jobs,oob_score=oob_score,max_features=max_features)
    rf_classifier.fit(X_train,y_train)
    y_pred=rf_classifier.predict(X_test)
      
    if multiclass:
        labels=['No Subsidence','<5 cm Subsidence','>5cm Subsidence'] 
        cm=confusion_matrix(y_test,y_pred)
        cm_df=pd.DataFrame(cm,columns=labels,index=labels)
        if save:
            cm_df.to_csv(cm_csv)
        
        print(cm_df,'\n')
        print('Recall Score {:.2f}'.format(recall_score(y_test,y_pred,average='micro')))
        print('Precision Score {:.2f}'.format(precision_score(y_test,y_pred,average='micro')))
        print('Accuracy Score {:.2f}'.format(accuracy_score(y_test,y_pred)))
    else:
        labels=['No Subsidence','>5cm Subsidence']
        cm=confusion_matrix(y_test,y_pred)
        cm_df=pd.DataFrame(cm,columns=labels,index=labels)
        if save:
            cm_df.to_csv(cm_csv)
        print(cm_df,'\n')
        print('Recall Score {:.2f}'.format(recall_score(y_test,y_pred)))
        print('Precision Score {:.2f}'.format(precision_score(y_test,y_pred)))
        print('Accuracy Score {:.2f}'.format(accuracy_score(y_test,y_pred)))
    if predictor_importance:
        col_labels=pd.read_csv(input_csv).columns
        importance=rf_classifier.feature_importances_
        plt.bar([x for x in range(len(importance))],importance,tick_label=col_labels[:-1],color='tomato')
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        
    if pdp:
        pdp_plot(classifier=rf_classifier, X_train=X_train,outdir=plot_save,multiclass_pdp=multiclass_pdp)

    return rf_classifier


def create_prediction_raster(predictor_raster_dir,prediction_raster,model,pattern='*.tif',exclude_columns=(),
                             pred_attr='Subsidence_G5_L5'):
    """
    Create predicted raster from random forest model.

    Parameters:
    predictor_raster_dir : Predictor rasters' directory.
    prediction_raster : Final predicted raster filename (withfilepath).
    model : A fitted model obtained from randon_forest_classifier function.
    pattern : Predictor rasters search pattern. Defaults to '*.tif'.
    exclude_columns : Predictor rasters' name that will be excluded from the model. Defaults to ().
    pred_attr : Variable name which will be predicted. Defaults to 'Subsidence_G5_L5'.

    Returns: None.
    """
    rasters=glob(os.path.join(predictor_raster_dir,pattern))
    predictor_dict={}
    nan_position_dict={}
    for raster in rasters:
        variable_name=raster[raster.rfind(os.sep)+1:raster.rfind(".")]
        raster_arr,file=rops.read_raster_arr_object(raster,get_file=True)
        raster_arr=raster_arr.flatten()
        nan_position_dict[variable_name]=np.isnan(raster_arr)
        raster_arr[nan_position_dict[variable_name]]=0
        predictor_dict[variable_name]=raster_arr
        
    drop_columns=[pred_attr]+list(exclude_columns) 
    predictor_df=pd.DataFrame(predictor_dict)
    predictor_df=predictor_df.dropna(axis=0) 
    
    #predictor_df.to_csv(output_csv,index=False)
    
    X=predictor_df.drop(columns=drop_columns).values
    #y=predictor_df[pred_attr].values
    
    y_pred=model.predict(X)
    
    for nan_pos in nan_position_dict.values():
        y_pred[nan_pos]=file.nodata
    y_pred_arr=y_pred.reshape((file.shape[0],file.shape[1]))
    rops.write_raster(raster_arr=y_pred_arr, raster_file=file, transform=file.transform, 
                     outfile_path=prediction_raster)



##Model run for Subsidence G5 training data
# Dataframe creation for train-test rasters
# =============================================================================
# indir="E:\\NGA_Project_Data\\Model Run\\Train_test_raster_G5"
# outcsv="E:\\NGA_Project_Data\\Model Run\\Predictors_csv\\Train_test_G5.csv"
# 
# df2=create_dataframe(input_raster_dir=indir,output_csv=outcsv,
#                     exclude_columns=('Alexi_ET_18_19','MODIS_ET_18_19','NDWI_18_19',
#                                      'Rainfall_TRCLM_18_19','NDWI_13_19'),pattern="*.tif")
# 
# #Model run for global data
# train_test_csv="E:\\NGA_Project_Data\\Model Run\\Predictors_csv\\Train_test_G5.csv"
# csv="E:\\NGA_Project_Data\\Model Run\\Final_prediction_raster\\Subsidence_G5_02.csv"
# rf_classifier=random_forest_classifier(input_csv=train_test_csv,pred_attr='Subsidence_G5',multiclass=False,
#                                        save=True,cm_csv=csv,predictor_importance=True)
# 
# global_predictor="E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster_G5"
# prediction_raster="E:\\NGA_Project_Data\\Model Run\\Final_prediction_raster\\Subsidence_G5_02.tif"
# create_prediction_raster(predictor_raster_dir=global_predictor,prediction_raster=prediction_raster,
#                          model=rf_classifier,pattern='*.tif',exclude_columns=('Alexi_ET_18_19','MODIS_ET_18_19','NDWI_18_19',
#                                      'Rainfall_TRCLM_18_19','NDWI_13_19'))
# =============================================================================


#Model run for Subsidence G5 and L5 training data
#Dataframe creation for train-test rasters
# =============================================================================
# indir="E:\\NGA_Project_Data\\Model Run\\Train_test_raster_G5_L5"
# outcsv="E:\\NGA_Project_Data\\Model Run\\Predictors_csv\\Train_test_G5_L5.csv"
# without_columns=('Slope','DEM','Landform', 'Temp_mean_13_19','z_soiltype','EVI_13_19')
# 
# df2=create_dataframe(input_raster_dir=indir,output_csv=outcsv,
#                     exclude_columns=without_columns,pattern="*.tif")
# 
# #Model run for global data
# train_test_csv="E:\\NGA_Project_Data\\Model Run\\Predictors_csv\\Train_test_G5_L5.csv"
# csv="E:\\NGA_Project_Data\\Model Run\\Final_prediction_raster\\Subsidence_G5_L5_01.csv"
# #outdir="E:\\NGA_Project_Data\\Model Run\\Predictors_csv"
# rf_classifier=random_forest_classifier(input_csv=train_test_csv,pred_attr='Subsidence_G5_L5',multiclass=True,
#                                         save=True,cm_csv=csv,predictor_importance=True,outdir=None,
#                                         pdp=True,multiclass_pdp=True,plot_save=None)
# 
# global_predictor="E:\\NGA_Project_Data\\Model Run\\Global_Predictors_raster_G5_L5"
# prediction_raster="E:\\NGA_Project_Data\\Model Run\\Final_prediction_raster\\Subsidence_G5_L5_01.tif"
# create_prediction_raster(predictor_raster_dir=global_predictor,prediction_raster=prediction_raster,
#                          model=rf_classifier,pattern='*.tif',exclude_columns=without_columns,
#                              pred_attr='Subsidence_G5_L5')
# =============================================================================


