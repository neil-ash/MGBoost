#!/usr/bin/python
# this is the example script to use xgboost to train
import numpy as np

import xgboost as xgb

test_size = 550000

# path to where the data lies
dpath = '/Users/Ashtekar15/Desktop/Thesis/higgs-boson/data'

# load in training data, directly use numpy
dtrain = np.loadtxt( dpath+'/training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
print ('finish loading from csv ')

label  = dtrain[:,32]
data   = dtrain[:,1:31]
# rescale weight to make it same as test set
weight = dtrain[:,31] * float(test_size) / len(label)

sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

# print weight statistics
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix( data, label=label, missing = -999.0, weight=weight )

################################################################################
# # setup parameters for xgboost
# param = {}
# # use logistic regression loss, use raw prediction before logistic transformation
# # since we only need the rank
# param['objective'] = 'binary:logitraw'
# # scale weight of positive examples
# param['scale_pos_weight'] = sum_wneg/sum_wpos
# param['eta'] = 0.1
# param['max_depth'] = 6
# param['eval_metric'] = 'auc'
# param['silent'] = 1
# param['nthread'] = 16
#
# # you can directly throw param in, though we want to watch multiple metrics here
# plst = list(param.items())+[('eval_metric', 'ams@0.15')]
#
# watchlist = [ (xgmat,'train') ]
# # boost 120 trees
# num_round = 10
#
# print ('loading data end, start to boost trees')
# bst = xgb.train( plst, xgmat, num_round, watchlist );

################################################################################
def fobj_MSE(y_true, y_pred):
    """
    'Custom' objective function (simply squared error)
    Needed in order to execute correct code in Booster.update()
    """
    print('Custom MSE objective function executing')
    grad = 2 * (y_pred - y_true)
    hess = 2 * np.ones(y_true.shape)
    return grad, hess

def fobj_logistic(y_true, y_pred):
    print('Custom logistic objective function executing')
    grad = -((y_true -  1) * np.exp(y_pred) + y_true) / (np.exp(y_pred) + 1)
    hess = np.exp(y_pred) / np.square(np.exp(y_pred) + 1)
    return grad, hess

# LogLoss functions from linxgb.py
# def dlogisticloss(self, X, y, y_hat):
#     """Return the first-order derivative of the logistic loss
#     w.r.t. its second argument evaluated at \f$(y, \hat{y}^{(t-1)})\f$.
#     """
#
#     return -( (y-1.)*np.exp(y_hat)+y)/(np.exp(y_hat)+1.)
#
# def ddlogisticloss(self, X, y, y_hat):
#     """Return the second-order derivative of the logistic loss
#     w.r.t. its second argument evaluated at \f$(y, \hat{y}^{(t-1)})\f$.
#     """
#
#     return np.exp(y_hat)/np.square(np.exp(y_hat)+1.)

xgbr = xgb.XGBRegressor(n_estimators=120,
                        max_depth=6,
                        learning_rate=0.1,
                        scale_pos_weight=sum_wneg/sum_wpos,
                        silent=1,
                        objective=fobj_MSE)

# eval_metric in fit()
xgbr.fit(data, label, eval_metric='auc')

################################################################################

# save out model
xgbr.save_model('higgs.model')

print ('finish training')
