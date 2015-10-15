# Running examples.
All examples should run without a problem and all except the Avazu script should be obvious as to what is happening.

## Avazu specific example using HDFS 
Running the bayesian optimization is especially useful if you are able to use a cluster as it means more threads for XGBoost to use.
There are various intricacies to be aware of when running on your own cluster, however the main idea is to create a 'generic' template 
file, and then via python run via the shell passing the bayes_opt arguments.

(This is untested at the minute)

So steps:

1. Build XGBoost in distributed mode (https://github.com/dmlc/wormhole/tree/master/learn/xgboost)
2. Create a template file - `xgboost-avazu.txt`
3. Create a python run file. `xgboost-avazu-dist.py`
4. Convert csv file into LIBSVM format. (I reccomend the helpers from teh excellent Phraug2 repo - https://github.com/zygmuntz/phraug2/blob/master/csv2libsvm.py)
5. split the data into training and validation `sort -R train.csv | split -l 6000000` where 6,000,000 is actually train set size and rename
6. Load data into HDFS (`hadoop fs -put avazu-train.svm /tmp`)
7. Run in python and wait for the results.
