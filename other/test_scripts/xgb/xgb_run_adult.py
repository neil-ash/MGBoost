""" Script to test changes to xgboost code """
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = '/Users/Ashtekar15/Desktop/Thesis/MGBoost/Numpy_Data/'

X_full = np.load(path + 'X_full.npy')
y_full = np.load(path + 'y_full.npy')

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,
                                                    test_size=0.20,
                                                    random_state=1)
xgb = XGBClassifier(max_depth=6,
                    learning_rate=0.3,
                    n_estimators=10)

# Need both 0s and 1s in training set, or will get:
# 'ValueError: y contains previously unseen labels: [1]'
# Fit on random selection of 0s and 1s since the labels passed in should not
# matter
xgb.fit(X_train, np.random.randint(2, size=y_train.size))

# Test if features matter -- it seems they do matter. Should they matter?
# When the model is trained on random features, it predicts all 1s
# xgb.fit(np.random.randint(2, size=X_train.size).reshape(X_train.shape),
#         np.random.randint(2, size=y_train.size))

#xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test).astype(int)

print('\nAccuracy: %.2f\n'
      % accuracy_score(y_test, y_pred))

print('Predicts all 1s:', str((y_pred == 1).all()))
print('Predicts all 0s:', str((y_pred == 0).all()))
