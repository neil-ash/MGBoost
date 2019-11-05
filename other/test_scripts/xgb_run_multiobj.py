"""
INCOMPLETE, REWRITTEN AS JUPYTER NOTEBOOK
Testing MGBoost on a synthetically generated multiobjective dataset
"""

# Imports
import numpy as np
from xgboost import XGBRegressor

# For reproducible results (should change to use RandomState instead?)
np.random.seed(1)

# Dimensions of generated data
m = 10000   # Training examples
n = 3       # Features
k = 4       # Distinct objectives

# Generate features randomly (gaussian)
X = np.random.normal(loc=0, scale=1, size=(m, n))

# Generate random integer weights
W = np.random.randint(low=-10, high=10, size=(k, n + 1))

# Generate labels using various weights multiplied by feature values
# Include bias term to match linear regression (?)
y = np.matmul(W, np.hstack((np.ones(m, 1), X)))

# Fit xgboost model on each label individually

# Fit mgboost model on all labels together

# Compare mgboost performance vs each label individually

# Compare mgboost performance to individual xgboost models
