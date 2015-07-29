__author__ = 'michael.pearmain'

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

DATA_TRAIN_PATH = 'data/otto/train.csv'
DATA_TEST_PATH = 'data/otto/test.csv'

def load_data(path_train = DATA_TRAIN_PATH, path_test = DATA_TEST_PATH):
    train = pd.read_csv(path_train)
    train_labels = [int(v[-1])-1 for v in train.target.values]
    train_ids = train.id.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)

    test = pd.read_csv(path_test)
    test_ids = test.id.values
    test = test.drop('id', axis=1)

    return np.array(train, dtype=float), \
           np.array(train_labels), \
           np.array(test, dtype=float),\
           np.array(train_ids), \
           np.array(test_ids)


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              silent,
              objective,
              nthread,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              seed = 1234):
    return cross_val_score(XGBClassifier(max_depth = max_depth,
                                         learning_rate = learning_rate,
                                         n_estimators = n_estimators,
                                         silent = silent,
                                         objective = objective,
                                         nthread = nthread,
                                         gamma = gamma,
                                         min_child_weight = min_child_weight,
                                         max_delta_step = max_delta_step,
                                         subsample = subsample,
                                         colsample_bytree = colsample_bytree,
                                         seed = seed),
                           train,
                           labels,
                           'log_loss',
                           cv=5).mean()

if __name__ == "__main__":
    # Load data set and target values
    train, labels, test, _, _ = load_data()

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (1, 10),
                                      'learning_rate': (0.1, 0.3),
                                      'n_estimators': (100, 250),
                                      'silent': (False),
                                      'objective': ("binary:logistic"),
                                      'nthread': (-1),
                                      'gamma': (1., 0.1),
                                      'min_child_weight': (2, 3),
                                      'max_delta_step': (0, 0.1),
                                      'subsample': (0.7, 0.8),
                                      'colsample_bytree': (0.7, 0.8)
                                     })
    xgboostBO.explore( {'max_depth': (1, 10),
                        'learning_rate': (0.1, 0.3),
                        'n_estimators': (100, 250),
                        'silent': (False),
                        'objective': ("binary:logistic"),
                        'nthread': (-1),
                        'gamma': (1., 0.1),
                        'min_child_weight': (2, 3),
                        'max_delta_step': (0, 0.1),
                        'subsample': (0.7, 0.8),
                        'colsample_bytree': (0.7, 0.8)
    })
    xgboostBO.maximize()
    print('-'*53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])