import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import linear_model
from scipy import stats
import numpy as np


########################################
# Extend sklearn.linear_model
########################################

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = True
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.se = se
        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

########################################
# Data Collation for Summary Table
########################################

def collate_values(column):
    list_of_stats = np.concatenate(column)
    list_of_stats = np.append(0, list_of_stats)
    list_of_stats = np.round(list_of_stats, decimals=2)
    return list_of_stats

########################################
# Summary Table
########################################
def summary_table(regressor, column_names):
    
    # Collate feature names
    feature_names = ["Intercept"]
    feature_names.extend(column_names)
    
    # Collate intercept and coefficients
    coefficients=[]
    coefficients.extend(regressor.intercept_.tolist())
    coefficients.extend(regressor.coef_[0].tolist())
    coefficients = np.round(coefficients, decimals=3)
    
    # Standard Error
    standard_errors = collate_values(regressor.se)
    # T Stat
    t_stats = collate_values(regressor.t)
    # P Value
    p_values = collate_values(regressor.p)
    # VIF
    
    # Confidence Intervals

    #coefficient - 1.96 * se

    #coefficient + 1.96 * se

    summary_table = pd.DataFrame({'Feature Name': feature_names, 
                                   'Coefficient': coefficients,
                                'Standard Error': standard_errors, 
                                   "T Statistic": t_stats, 
                                       "P Value": p_values,
                                   })
    
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



########################################
# Diagnostic Plots
########################################
def create_diagnostic_plots(target,residual):

    
    # Create figure
    fig = plt.figure(figsize=(20,10))
    # Create gridspec
    gs = fig.add_gridspec(7,7)

    # Create charts using gridspec
    top_left = fig.add_subplot(gs[0:3,0:3])
    top_right = fig.add_subplot(gs[0:3, 4:7])
    bottom_left = fig.add_subplot(gs[4:7,0:3])
    bottom_right = fig.add_subplot(gs[4:7,4:7])

    # Chart Titles
    top_left.set_title("Histogram of Target Variable")
    top_right.set_title("Histogram of Residuals")
    bottom_left.set_title("Scatter Plot of Target vs Residuals")
    bottom_right.set_title("Q-Q Plot to assess Normality")
    
    # Y Axis Labels
    top_left.set_ylabel("Frequency")
    top_right.set_ylabel("Frequency")
    bottom_left.set_ylabel("Residual Values")
    bottom_right.set_ylabel("Ordered Values")
    
    # X Axis Labels
    top_left.set_xlabel("Target Values")
    top_right.set_xlabel("Fitted Values")
    bottom_left.set_xlabel("Fitted Values")
    bottom_right.set_xlabel("Theoretical Quantiles")    

    # 1. What is the distribution of the Target?
    top_left.hist(target, color='rebeccapurple')
    # 2. What is the distribution of the Residuals?
    top_right.hist(residual, color='gold')
    # 3. Do we satisfy the assumption of Heteroskedasticity?
    x_axis = [target.min(),target.max()]
    y_axis = [0,0]
    bottom_left.scatter(x=target, y=residual, color='rebeccapurple')
    bottom_left.plot(x_axis,y_axis, color='gold', linewidth=2)
    # 4. Do we satisfy the assumption of Normality?
    sp.stats.probplot(residual, dist="norm")
    
    
    
    #########################################################
    
#FURTHER FEATURES TO ADD:
    
 # GENERAL
    # Make titles larger
    # Make axis larger
    # Fix QQ-Plot functionality
    
 # HISTOGRAMS
    # Include normal distribution overlayed on top of Target and Residuals Histograms
    # Add outlines on value bars 
    