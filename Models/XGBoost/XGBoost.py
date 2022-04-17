
# Import Packages
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#XGBoost
import xgboost as xgb

# Model packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder


# Import Data
filepath = "C:\\Users\\hlmq\\OneDrive - Chevron\\Data\\DSDP\\Ames\\"

df = pd.read_csv(str(filepath)+"AmesHousing.csv")


# Subset dataset to only columns for modeling.
X = df[independent]
y = df[target]
y.columns = ['Target']


# This will DROP the original column and create dummies for categorical variables

for col in dummy_code_columns:
    X = pd.get_dummies(X, columns=[col])
    X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)


# Train,Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# XGBoost Modeling.  Cross fold validation and grid search to find best parameters.

xgboost_model = xgb.XGBRegressor()

# Save column names for future visualization
independent_columns = X_test.columns

# Initialize parameter grid for XGBoost. Shortened for brevity



# Initialize grid_search.  If it takes too long, lower the crossvalidation number
grid_search = GridSearchCV(estimator=xgboost_model,
                           param_grid=param_grid,  # parameters to be tuned
                           cv=2,
                           n_jobs=-1,  # -1 means use all available cores
                           verbose=2,
                           )

# Perform CV search over grid of hyperparameters
grid_search.fit(X_train, y_train)

print("Best CV accuracy: {}, with parameters: {}".format(
    grid_search.best_score_, grid_search.best_params_))



cv_best_model = grid_search.best_estimator_


# Add predictions to test dataset
y_test['Prediction'] = cv_best_model.predict(X_test)

# Create residuals
y_test['Residual'] = y_test['Target']-y_test['Prediction']


# Make sure to modify the file name!!!!!

# Use this if you want to export any data
filepath = 'Models\\XGBoost\\out\\'
now = datetime.now()
current_time = now.strftime("%Y_%m_%d-%H_%M_%S")
filename_submission = current_time + '_XGBoost_Regressor_Results.csv'
output_data = y_test

output_data.to_csv(filepath+filename_submission, sep=',', index = False)