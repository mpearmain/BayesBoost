from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

DATA_TRAIN_PATH = '../data/otto/train.csv'
DATA_TEST_PATH = '../data/otto/test.csv'

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
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent =True,
              nthread = -1,
              seed = 1234):
    return cross_val_score(XGBClassifier(max_depth = int(max_depth),
                                         learning_rate = learning_rate,
                                         n_estimators = int(n_estimators),
                                         silent = silent,
                                         nthread = nthread,
                                         gamma = gamma,
                                         min_child_weight = min_child_weight,
                                         max_delta_step = max_delta_step,
                                         subsample = subsample,
                                         colsample_bytree = colsample_bytree,
                                         seed = seed,
                                         objective = "multi:softprob"),
                           train,
                           labels,
                           "log_loss",
                           cv=5).mean()

if __name__ == "__main__":
    # Load data set and target values
    train, labels, test, _, _ = load_data()

    xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (5, 10),
                                      'learning_rate': (0.01, 0.3),
                                      'n_estimators': (50, 1000),
                                      'gamma': (1., 0.01),
                                      'min_child_weight': (2, 10),
                                      'max_delta_step': (0, 0.1),
                                      'subsample': (0.7, 0.8),
                                      'colsample_bytree' :(0.5, 0.99)
                                     })

    xgboostBO.maximize()
    print('-'*53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])