# Databricks notebook source
# MAGIC %md
# MAGIC ![](/files/Blueprint_logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Optimizing Advertising Spend with Machine Learning & Databricks
# MAGIC * This is a new accelerator from Blueprint
# MAGIC 
# MAGIC * For additional information how to run these notebooks go to notebook 00_START_HERE
# MAGIC 
# MAGIC The diagram below will guide you so that you know where you are
# MAGIC 
# MAGIC **Please Note**: 
# MAGIC 
# MAGIC * This new accelerator is based on original Databricks Accelerator [Solution Accelerator: Multi-touch Attribution](https://databricks.com/blog/2021/08/23/solution-accelerator-multi-touch-attribution.html)
# MAGIC 
# MAGIC * As of this X-Challenge version the original Databricks data is still required, so the original Databricks notebooks are still required to create the BRONZE and SILVER raw data tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### PLEASE NOTE:
# MAGIC 
# MAGIC * This notebook is work in progress JUST A PLACEHOLDER for now
# MAGIC   * One Boosted Tree Model for Impressions is provided as an **Example**
# MAGIC   * There are several TO DOs
# MAGIC   
# MAGIC TODO. Additional machine learning models:
# MAGIC * Click Through Rate 
# MAGIC * Conversion Rate
# MAGIC * Cost Per Mille 
# MAGIC * Cost Per Click
# MAGIC * Revenue per Transaction

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/ad_spend_flow_03.png)

# COMMAND ----------

# MAGIC %pip install hyperopt

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

params = get_params()
database_name = params['database_name']
raw_data_path = params['raw_data_path']
bronze_tbl_path = params['bronze_tbl_path']
print(database_name)
print(raw_data_path)

# COMMAND ----------

print(database_name)
_ = spark.sql('USE {}'.format(database_name))

# COMMAND ----------

from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost

# COMMAND ----------

# DBTITLE 1,Load Data

data = (spark.sql("select * from train_impr_clicks_and_cost")
                    .withColumnRenamed('visit_impr', 'impr')
                    .withColumnRenamed('sum_visit_impr', 'user_visits')
                    .withColumnRenamed('visit_clicks', 'clicks')
                    .withColumnRenamed('visit_txns', 'txns')
                    .withColumnRenamed('max_cpm', 'cpm')
                    .withColumnRenamed('avg_cpi', 'cpi')
                    .withColumnRenamed('bid_cpc', 'cpc')
       )
    
print(data.columns)

# COMMAND ----------

from pyspark.sql.functions import col
display(data.sort(col('gpt').desc()))

# COMMAND ----------

# DBTITLE 1,Reduce train size (for Demo purposes)
split_perc = .2
if split_perc is not None:
    train_data, val_data = data.randomSplit([split_perc, 1-split_perc], seed=1234)
else:
    train_data, val_data = data, None
from_nrows = data.count()
to_nrows = train_data.count()
print(f"keeping {to_nrows} rows from total {from_nrows} rows")
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Impressions

# COMMAND ----------

#from pyspark.ml.linalg import Vectors
#from pyspark.ml.feature import VectorAssembler, StringIndexer

#inputs = ["label1", "label2"]
#outputs = ["index1", "index2"]
#stringIndexer = StringIndexer(inputCols=inputs, outputCols=outputs)
#model = stringIndexer.fit(multiRowDf)


# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/_static/notebooks/gbt-regression.html

# COMMAND ----------

train_data.printSchema()

# COMMAND ----------

columnList = [item[0] for item in train_data.dtypes if item[1].startswith('string')]
columnList

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

train_data2 = train_data
numeric_cols= [ 'impr', 'timestamp', 'month', 'dayofweek', 'weekofyear', 'product_stars', 'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]
for col_name in numeric_cols:
    train_data2 = train_data2.withColumn(col_name, col(col_name).cast(DoubleType()))
train_data2.printSchema()

# COMMAND ----------

display(train_data2)

# COMMAND ----------

# DBTITLE 1,Prepare data for Spark models
# Spark Mlib API requires two columns: 
#("label","features")

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer

impr_feature_cols = [ 'timestamp', 'month', 'dayofweek', 'weekofyear', 'channelIndex', 'product_stars', 'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]
impr_target_col =  ['impr']

drop_cols = ['time', 'gpt']
key_cols = ['uid']

data2 = train_data2.fillna(0)
#index string cols
inputs = ['channel']
outputs = [ 'channelIndex']
data3 = (StringIndexer(inputCols=inputs, outputCols=outputs)
                  .fit(data2)
                  .transform(data2)
                )

# drop unnecessary columns
data4 = data3.select(key_cols + impr_target_col + impr_feature_cols)

#cast to double
for col_name in impr_feature_cols:
    data4.withColumn(col_name, col(col_name).cast(DoubleType()))

#collect features into a vector
vec_assembler = VectorAssembler(
        inputCols = impr_feature_cols,
        outputCol="features")

data5 = vec_assembler.transform(data4)


data6 = data5.select("features",'impr')
display(data6)


# COMMAND ----------

# MAGIC %md
# MAGIC This model is presented as an example

# COMMAND ----------

# DBTITLE 1,Impressions - Model
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

#build model
impr_gbt = GBTRegressor(labelCol='impr',featuresCol='features')

print(f"label col: {impr_gbt.getLabelCol()}")
print(f"prediction col: {impr_gbt.getPredictionCol()}")
print(f"features col: {impr_gbt.getFeaturesCol()}")

#Spark GBTRegressor hyper-parameters  https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GBTRegressor.html
#lossType='squared'
#stepSize=0.1. Learning rate
#MaxDepth. Maximum depth of the tree.
#MaxIter. Maximum Number of Iterations.
#MaxBins. Maximum number of bins used for discretizing continuous features and for choosing how to split on features at each node.
#minInstancesPerNode. Minimum number of instances each child must have after split
#minWeightFractionPerNode. Minimum fraction of the weighted sample count that each child must have after split
#minInfoGain. Minimum information gain for a split to be considered at a tree node

impr_paramGrid = ParamGridBuilder()\
  .addGrid(impr_gbt.maxDepth, [2, 5, 7])\
  .addGrid(impr_gbt.maxIter, [5, 10, 15])\
  .build()

impr_evaluator = RegressionEvaluator(metricName="rmse", labelCol=impr_gbt.getLabelCol(), predictionCol=impr_gbt.getFeaturesCol())

#impr_gbt.fit(data6)

# COMMAND ----------

display(data6)

# COMMAND ----------

# DBTITLE 1,Fit Model individually
impr_gbt.fit(data6)

# COMMAND ----------

# DBTITLE 1,Hyper-Parameter Tuning
# MAGIC %md
# MAGIC 
# MAGIC K-fold cross validation performs model selection by splitting the dataset into a set of non-overlapping randomly partitioned folds 
# MAGIC which are used as separate training and test datasets
# MAGIC 
# MAGIC For example:
# MAGIC      - with k=3 folds, 
# MAGIC      - K-fold cross validation will generate 3 dataset pairs (training, test)
# MAGIC      - each of which uses 2/3 of the data for training and 1/3 for testing
# MAGIC      - Each fold is used to evaluate exactly once.
# MAGIC      
# MAGIC attributes:
# MAGIC  average error across the folds = cv.avgMetrics (e.g. avg RMSE)

# COMMAND ----------

impr_cv = CrossValidator(estimator=impr_gbt,
                    estimatorParamMaps=impr_paramGrid,
                    evaluator=impr_evaluator)

#impr_cv.fit(data6)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the Cross-validation
# MAGIC 
# MAGIC If using CrossValidator MLlib will [automatically track trials in MLflow](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/mllib-mlflow-integration.html). After your tuning fit() call has completed, view the MLflow UI to see logged runs.
# MAGIC 
# MAGIC Now that you have defined the pipeline, you can run the cross validation to tune the model's hyperparameters. During this process, [MLflow automatically tracks the models produced by CrossValidator](https://docs.microsoft.com/en-us/azure/databricks/_static/notebooks/mllib-mlflow-integration.html), along with their evaluation metrics. This allows you to investigate how specific hyperparameters affect the model's performance.
# MAGIC 
# MAGIC In this example, you examine two hyperparameters in the cross-validation:
# MAGIC 
# MAGIC maxDepth. This parameter determines how deep, and thus how large, the tree can grow.
# MAGIC maxBins. For efficient distributed training of Decision Trees, MLlib discretizes (or "bins") continuous features into a finite number of values. The number of bins is controlled by maxBins. In this example, the number of bins corresponds to the number of grayscale levels; m
