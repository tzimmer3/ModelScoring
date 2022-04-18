import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


########################################
# Continuous Accuracy Table
########################################

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

########################################
# Residuals vs Fitted Plot
########################################
def residuals_vs_fitted(residuals, prediction):
    smoothed = lowess(residuals,prediction)
    top3 = abs(residuals).sort_values(ascending = False)[:3]

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,7)
    fig, ax = plt.subplots()
    ax.scatter(prediction, residuals, edgecolors = 'k', facecolors = 'none')
    ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted Values')
    ax.set_title('Residuals vs. Fitted')
    ax.plot([min(prediction),max(prediction)],[0,0],color = 'k',linestyle = ':', alpha = .3)

    #Annotate the max and min values
    for i in top3.index:
        ax.annotate(i,xy=(predictions_df['Prediction'][i],predictions_df['Residuals'][i]))
    
    
    # In this case, you only learn what the predictions look like to potentially see outliers, or trends.
    # Heteroskedasticity is not an assumption to check
    # Does not show how a particular observation fell in the tree.