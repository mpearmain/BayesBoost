# Running examples.
All examples should run without a problem and all except the Avazu script should be a little obvious as to what is happening.

## Avazu specific example using HDFS 
Running the bayesian optimization is especially useful if you are able to use a cluster as it means more threads for XGBoost to use.
There are various intricacies to be aware of when running on your own cluster, however the main idea is to create a 'generic' template 
file, and then via python run via the shell passing the bayes_opt arguments.

So steps:

1. Build XGBoost in distributed mode (https://github.com/dmlc/wormhole/tree/master/learn/xgboost)
2. Create a template file - `xgboost-avazu.txt`
3. Create a python run file. `xgboost-avazu-dist.py`
4. Run in python and wait for the results.
