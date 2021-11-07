import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score, mean_squared_error

# X_test, y_test[target],y_test['Prediction']
def accuracy_table(df, target, prediction):
    # Global model accuracy metrics
    measures = ['MSE','RMSE','R2', 'Adj R2']

    n = len(df)
    p = len(df.columns)

    mse_value = mean_squared_error(target, prediction)
    rmse_value = math.sqrt(mse_value) 
    R2 = r2_score(target, prediction)
    AdjR2 = 1-(1-R2)*(n-1)/(n-p-1)

    scores = [mse_value, rmse_value, R2, AdjR2]
    accuracytable = pd.DataFrame({'Measure': measures, 'Value': scores})

    return accuracytable