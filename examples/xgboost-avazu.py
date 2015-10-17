from __future__ import print_function
from __future__ import division

__author__ = 'michael.pearmain'

import pandas as pd
import xgboost as xgb
import random
import os
from subprocess import Popen
from sklearn.metrics import log_loss
from bayes_opt import bayesian_optimization


# Driver python script for using xgboost on cluster with bayesian optimzation.

# First define the function to pass the 
def xgboostcv(max_depth,
              eta,
              num_rounds,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent=True,
              seed=1234):
    
    print ('\nRunning XGBOOST on the cluster')
    
    # Call xgboost in distributed mode (CLI input for params)
    xgb_run = ['max_depth=%s' % int(max_depth),
               'eta=%s' % eta,
               'silent=%s' % silent,
               'gamma=%s' % gamma,
               'min_child_weight=%s' % int(min_child_weight),
               'max_delta_step=%s' % max_delta_step,
               'subsample=%s' % subsample,
               'eval_metric=logloss',
               'colsample_bytree=%s' % colsample_bytree,
               'seed=%s' % seed,
               'objective=binary:logistic',
               'eval[eval_set]=%s' % deval,
               'eval[train_set]=%s' % dtrain,
               'num_round=%s' % int(num_rounds),
               'data=%s' % dtrain,
               'model_out=%s' % model_ouput]
    argv = ['wormhole/repo/dmlc-core/tracker/dmlc_yarn.py', # Where your instance is found!!
            '-n',
            '16',
            'wormhole/bin/xgboost.dmlc', # Where your instance is found!!
            './examples/xgboost-avazu.txt'] + xgb_run
    print(' '.join(argv))
    # Cluster specific ENV VARS.
    Popen(argv,
          env = {'JAVA_HOME': '/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.45-28.b13.el6_6.x86_64/',
                 'HADOOP_HOME': '/usr/',
                 'HADOOP_HDFS_HOME': '/usr/lib/hadoop-hdfs/',
                 'PATH': os.getenv('PATH')}).communicate()

    # Export model to local filesystem
    try:
      os.remove("avazu.model")
    except OSError:
      pass
    Popen(["hadoop","fs","-copyToLocal","/tmp/avazu.model", "."]).communicate()
    # Delete stored model.
    Popen(["hadoop","fs","-rm","/tmp/avazu.model", "."]).communicate()
    
    # Load Model file
    bst = xgb.Booster(model_file='avazu.model')
    preds = bst.predict(dtest)
    y_pred = bst.predict(dtest)
    y_valid = dtest.get_label()
    print('logloss = ', log_loss(y_valid, y_pred))
    # We are maximizing the function.
    return -log_loss(y_valid, y_pred)
    
if __name__ == "__main__":
    # Load data sets (for ease use unix 'split' data for train and eval)
    # sort -R train.csv | split -l 6000000  ## where 6,000,000 is actually train set size
    # Rename the resulting files but keep the eval set for prediction.
    # give absolute paths to pass to 'trainer'
    dtrain = 'hdfs://tmp/train-svm'
    deval  = 'hdfs://tmp/eval-svm'
    model_ouput = "hdfs://das/tmp/avazu.model"
    # Local copy of the eval data (very hacky --> need to get the labels)
    dtest = xgb.DMatrix('avazu/eval-svm')
    
    xgboostBO = bayesian_optimization.BayesianOptimization(xgboostcv,
                                     {'max_depth': (15, 20),
                                      'eta': (1.0, 0.1),
                                      'num_rounds': (20, 50),
                                      'gamma': (1., 0.01),
                                      'min_child_weight': (6, 20),
                                      'max_delta_step': (0., 3),
                                      'subsample': (0.7, 0.85),
                                      'colsample_bytree': (0.7, 0.85)
                                     })

    xgboostBO.maximize(init_points=5, restarts=50, n_iter=50)
    print('-' * 53)

    print('Final Results')
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
