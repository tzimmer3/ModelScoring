import sys

sys.path.append("Models//")

from python_packages.Continuous_Model import functions

functions.accuracy_table()
functions.residuals_vs_fitted()
functions.feature_importance()