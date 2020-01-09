import numpy as np
import pandas as pd
from xgboost import XGBRegressor

data = pd.read_csv('linear_regression_data.csv')
X = data.X.values.reshape(-1, 1)
y = data.y.values
m = y.size

def fobj_MSE(y_true, y_pred):
    """
    'Custom' objective function (simply squared error)
    Needed in order to execute correct code in Booster.update()
    """
    print('Custom MSE objective function executing')
    grad = 2 * (y_pred - y_true)
    hess = 2 * np.ones(y_true.shape)
    return grad, hess

xgbr = XGBRegressor(n_estimators=10,
                    max_depth=6,
                    objective=fobj_MSE)

# Use with original XGBoost package
xgbr.fit(X, y)

# Use with MGBoost to ensure that labels passed into fit() do not matter
# xgbr.fit(X, np.random.randint(2, size=y.size))

y_pred = xgbr.predict(X)

MSE = np.sum((y - y_pred) ** 2) / m

print('\nMSE: %.2f\n' % MSE)
