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

# MAGIC %md
# MAGIC ### Predictive Models for Optimizing Advertising Spend
# MAGIC 
# MAGIC * We want to predict return on ad spend on any ad before we buy it so we can take decision not to.
# MAGIC 
# MAGIC * Natural way to do this is with a Machine Learning model to predict ROAS, user behavior, an [ad performance](https://www.newbreedrevenue.com/blog/how-to-measure-the-performance-of-your-paid-ad-campaigns):
# MAGIC   * Random Forests, Neural Networks, Boosted Trees, Time Series Forecasting, are well known methods and APIs to do this.
# MAGIC   * Some of the metrics traditionally used to measure ad performance are the following:
# MAGIC     1. Impressions
# MAGIC     2. Clicks
# MAGIC     3. Conversions
# MAGIC     4. Click Through Rate
# MAGIC     5. Cost per Click
# MAGIC     6. Quality Score
# MAGIC 
# MAGIC We can leverage Spark ML and Databricks ML Runtime for building a set of predictive models to predict ad performance
# MAGIC * For example: We can use [Gradient-Boosted Trees (GBTs)](https://en.wikipedia.org/wiki/Gradient_boosting) for regression, which is available off-the-shelf from Spark ML Library.
# MAGIC 
# MAGIC 
# MAGIC As an example, in this notebook we provide an implementation of a [gradient boosted tree](https://docs.databricks.com/_static/notebooks/gbt-regression.html) for predicting [Ad Impressions](https://support.google.com/google-ads/answer/6320?hl=en)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install required libraries

# COMMAND ----------

# MAGIC %pip install hyperopt

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

# DBTITLE 1,Set Environment 
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
from pyspark.sql.functions import col

# COMMAND ----------

# DBTITLE 1,Load Train Data
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
columnList = [item[0] for item in train_data.dtypes if item[1].startswith('string')]
print(f"string columns: {columnList}")
train_data.printSchema()
#display(train_data)

# COMMAND ----------

# DBTITLE 1,Set aside some data to Test (this is different from Validation set)
#best practice - even when using methods such as cross-validation 
train_data, test_data = train_data.randomSplit([.8,.2], seed=1234)
print(train_data.count())
print(test_data.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Impressions - Predictive Model
# MAGIC 
# MAGIC Ad impressions (IMPR) are counted when an add displays on a screen.
# MAGIC 
# MAGIC 
# MAGIC Gradient-boosted trees (GBTs) are a popular regression method using ensembles of decision trees. More information about the spark.ml implementation can be found further in the section on [GBTs](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-trees-gbts).
# MAGIC 
# MAGIC Input: (for training)
# MAGIC 
# MAGIC | Param name	| Type(s)	| Default	| Description | 
# MAGIC | --------------| ----------| ----------| ------------|
# MAGIC | labelCol	| Double	| "label"	| Label to predict |
# MAGIC | featuresCol	| Vector	| "features"| feature vector |	
# MAGIC 
# MAGIC Output: (for prediction)
# MAGIC 
# MAGIC | Param name	| Type(s)	| Default	| Description |
# MAGIC | ------------- | --------- | --------- | ------------|
# MAGIC | predictionCol	| Double	| "prediction"	| Predicted label | 

# COMMAND ----------

# DBTITLE 1,Prepare data for Spark models
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

# Spark Mlib API requires two columns: 
#("label","features")

#define feature columns
key_cols = ['uid', 'channel', 'product_stars']
impr_target_col =  ['impr']
drop_cols = ['time', 'gpt']
#impr_feature_cols = train_data.columns - set(key_cols) - set(impr_target_cols) - set(drop_cols)
impr_feature_cols = [ 'timestamp', 'month', 'dayofweek', 'weekofyear', 'channelIndex', 'product_stars', 'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]

#cast numeric cols to Double
numeric_cols= [ 'impr', 'timestamp', 'month', 'dayofweek', 'weekofyear', 'product_stars', 'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]
train_data2 = train_data
for col_name in numeric_cols:
    train_data2 = train_data2.withColumn(col_name, col(col_name).cast(DoubleType()))
#train_data2.printSchema()

train_data3 = train_data2.fillna(0)

#index string cols
inputs = ['channel']
outputs = [ 'channelIndex']
train_data4 = (StringIndexer(inputCols=inputs, outputCols=outputs)
                  .fit(train_data3)
                  .transform(train_data3)
                )

# drop unnecessary columns
train_data5 = train_data4.select(key_cols + impr_target_col + impr_feature_cols)
    
#collect features into a vector column "features"
impr_assembler = VectorAssembler(
        inputCols = impr_feature_cols,
        outputCol="features")

train_features_df = impr_assembler.transform(train_data5)

train_features_df2 = train_features_df.select(bid_unit_cols + impr_target_col + ["features"])
display(train_features_df2)


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Build Impressions Model
# MAGIC 
# MAGIC This model is presented as an example.
# MAGIC 
# MAGIC Spark GBTRegressor hyper-parameters  https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GBTRegressor.html
# MAGIC * lossType='squared'
# MAGIC * stepSize=0.1. Learning rate
# MAGIC * MaxDepth. Maximum depth of the tree.
# MAGIC * MaxIter. Maximum Number of Iterations.
# MAGIC * MaxBins. Maximum number of bins used for discretizing continuous features and for choosing how to split on features at each node.
# MAGIC * minInstancesPerNode. Minimum number of instances each child must have after split
# MAGIC * minWeightFractionPerNode. Minimum fraction of the weighted sample count that each child must have after split
# MAGIC * minInfoGain. Minimum information gain for a split to be considered at a tree node

# COMMAND ----------

# DBTITLE 1,Train Impressions - Model
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

#build model
impr_gbt = GBTRegressor(labelCol='impr',featuresCol='features')

print(f"label col: {impr_gbt.getLabelCol()}")
print(f"prediction col: {impr_gbt.getPredictionCol()}")
print(f"features col: {impr_gbt.getFeaturesCol()}")

impr_paramGrid = ParamGridBuilder()\
  .addGrid(impr_gbt.maxDepth, [2, 5, 7])\
  .addGrid(impr_gbt.maxIter, [5, 10, 15])\
  .build()

impr_evaluator = RegressionEvaluator(metricName="rmse", labelCol=impr_gbt.getLabelCol(), predictionCol=impr_gbt.getPredictionCol())

#train model individually
impr_gbt.fit(train_features_df2)

# COMMAND ----------

# DBTITLE 1,Spark Pipeline
# MAGIC %md
# MAGIC A Spark Pipeline is a sequence of stages. 
# MAGIC * Each stage can be run in a sequence
# MAGIC * Similar to a ['Sequential' Model](https://www.tensorflow.org/guide/keras/sequential_model) in Tensorflow
# MAGIC * for more information [Spark ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html#how-it-works)

# COMMAND ----------

from pyspark.ml import Pipeline
#wrap inside a Spark Pipeline
impr_gbt_pipeline = Pipeline(stages=[impr_assembler, impr_gbt])

#train the model in a Pipeline
impr_pipelineModel = impr_gbt_pipeline.fit(train_data5)

#show model summary
impr_gbt_model = impr_pipelineModel.stages[1]
print(impr_gbt_model)  

# COMMAND ----------

# DBTITLE 1,Load and Prepare Test Data
#cast numeric cols to Double
numeric_cols= [ 'impr', 'timestamp', 'month', 'dayofweek', 'weekofyear', 'product_stars', 'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]
test_data2 = test_data
for col_name in numeric_cols:
    test_data2 = test_data2.withColumn(col_name, col(col_name).cast(DoubleType()))
#train_data2.printSchema()

test_data3 = test_data2.fillna(0)

#index string cols
inputs = ['channel']
outputs = [ 'channelIndex']
test_data4 = (StringIndexer(inputCols=inputs, outputCols=outputs)
                  .fit(test_data3)
                  .transform(test_data3)
                )

# drop unnecessary columns
test_data5 = test_data4.select(key_cols + impr_target_col + impr_feature_cols)

# COMMAND ----------

# DBTITLE 1,Train Impressions - Predict and Evaluate
#train
impr_pred1 = impr_pipelineModel.transform(train_data5)
impr_pred2 = impr_pipelineModel.transform(test_data5)

#evaluate
print(f"rmse with training data: {impr_evaluator.evaluate(impr_pred1)}")
print(f"rmse with test data: {impr_evaluator.evaluate(impr_pred2)}")
display(impr_pred1)

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

# DBTITLE 1,Run Cross-Validation
impr_cv = CrossValidator(estimator=impr_gbt,
                    estimatorParamMaps=impr_paramGrid,
                    evaluator=impr_evaluator)

impr_cv_model = impr_cv.fit(train_features_df2)

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
# MAGIC maxBins. For efficient distributed training of Decision Trees, MLlib discretizes (or "bins") continuous features into a finite number of values. The number of bins is controlled by maxBins. 

# COMMAND ----------

# DBTITLE 1,Predict with CV Model
impr_cv_prediction1 = impr_cv_model.transform(train_features_df2)

impr_gbt_pipeline2 = Pipeline(stages=[impr_assembler, impr_cv_model.bestModel])
impr_pipelineModel2 = impr_gbt_pipeline2.fit(train_data5)
impr_cv_prediction2 = impr_pipelineModel2.transform(test_data5)

print(f"rmse with cross validation train data: {impr_evaluator.evaluate(impr_cv_prediction1)}")
print(f"rmse with cross validation test data: {impr_evaluator.evaluate(impr_cv_prediction2 )}")
