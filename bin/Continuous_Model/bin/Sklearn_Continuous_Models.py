import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########################################
# Feature Imporatance (Sklearn models)
########################################


def feature_importance(model, column_names):

    # Variable Importance Bar Graph

    feature_importance=pd.DataFrame({'xgboost_model':model.feature_importances_},index=column_names)
    feature_importance.sort_values(by='xgboost_model',ascending=True,inplace=True)

    index = np.arange(len(feature_importance))
    fig, ax = plt.subplots(figsize=(12,8))
    rfc_feature=ax.barh(index,feature_importance['xgboost_model'],0.4,color='rebeccapurple',label='XGBoost Model')
    ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

    ax.legend()
    plt.show()
    
    
    
########################################
#  SHAP Analysis                       #
########################################

# Code currently located in the two XGBoost - Copy1 files in Python Packages folder
# https://towardsdatascience.com/explaining-scikit-learn-models-with-shap-61daff21b12a

# Create SHAP values
def create_shap_values():
    pass

# Create global plots of SHAP values
def global_shap_analysis():
    pass

# Create local plots of SHAP values
def local_shap_analysis():
    pass






