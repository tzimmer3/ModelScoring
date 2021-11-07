import pandas as pd
import numpy as np


def summary_table(regressor, column_names):
    
    # Collate feature names
    feature_names = ["Intercept"]
    feature_names.extend(column_names)
    
    # Collate intercept and coefficients
    coefficients=[]
    coefficients.extend(regressor.intercept_.tolist())
    coefficients.extend(regressor.coef_[0].tolist())
    
    # T Stat
    
    # P Value
    
    # VIF
    
    # Confidence Intervals

    summary_table = pd.DataFrame({'Feature Name': feature_names, 'Coefficient': coefficients})
    
    return summary_table


 #########################################################
    
#FURTHER FEATURES TO ADD:
    
 # GENERAL

    #t_stats = 
    #p_value = 
    #confidence_025 =
    #confidence_975 = 

        # Standard Error Equation
        # [1 - R2 / (1-Tolerance) * (N - K - 1)]   *   [SE(y) / SE(x)]
        # Tolerance is the biggest problem child
        # SE(y) is the standard deviation of Y (col.std())
        # SE(x) is the standard deviation of X (each variable)
        # Tolerance is the R2 value of variable in question vs the rest of the variables


#summary_table = pd.DataFrame({'Column': XtrainA.columns, 
#                              'Coefficient': regressor.coef_, 
#                              'Standard Error': standard_errors, 
#                              "T Statistic": t_stats, 
#                              "P Value": p_values, 
#                              "Confidence .025": confidence_025, 
#                              "Confidence .975": confidence_975})