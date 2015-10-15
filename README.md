# BayesBoost
Bayesian Optimization using xgboost and sklearn API

Simple test scripts for optimal hyperparameter of xgboost using bayesian optimization.

Original bayesian optimization code is from https://github.com/fmfn/BayesianOptimization and all credit for this work
goes to the original author.  


Example 1. is based on the otto dataset from Kaggle, this remains in memory.
(https://www.kaggle.com/c/otto-group-product-classification-challenge)

Example 2. is based on Avazu click prediction dataset from Kaggle and requires the 'distributed' version of xgboost.
(https://www.kaggle.com/c/avazu-ctr-prediction)

### Run
To get this running create a data/otto and data/avazu dir and download the datasets into the respective directories and unzip / untar the files.

Dependencies:
* Scipy
* Numpy
* Scikit-Learn
* xgboost (https://github.com/dmlc/xgboost)

References:
* http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
* http://arxiv.org/pdf/1012.2599v1.pdf
* http://www.gaussianprocess.org/gpml/
* https://www.youtube.com/watch?v=vz3D36VXefI&index=10&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6

