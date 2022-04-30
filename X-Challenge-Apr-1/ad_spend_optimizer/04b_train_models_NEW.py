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
# MAGIC #### PLEASE NOTE:
# MAGIC 
# MAGIC * In this notebook we are presenting an **Example** of a predictive model using Spark MLlib 
# MAGIC 
# MAGIC * After working through this series of notebooks for the first time, you may want to customize the settings and add additional models 
# MAGIC 
# MAGIC * Here are some ideas of additional models to try:
# MAGIC      * Revenue per Transaction
# MAGIC      * Cost per ad impression or user click
# MAGIC      * Click-through or Conversion rate    
# MAGIC      * ROAS per ad impression

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/ad_spend_flow_03.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predictive Models for Optimizing Advertising Spend
# MAGIC 
# MAGIC * Predictive Modeling is the process of making predictions on makerting campaign and ad performance future outcomes to optimize ROI and ROAS.
# MAGIC   
# MAGIC     * We want to predict return on ad spend on any ad before we buy it so we can take decision not to.
# MAGIC 
# MAGIC     * Natural way to do this is with a Machine Learning model to predict ROAS, user behavior, an [ad performance](https://www.newbreedrevenue.com/blog/how-to-  
# MAGIC       measure-the-performance-of-your-paid-ad-campaigns):
# MAGIC       * Random Forests, Neural Networks, Boosted Trees, Time Series Forecasting, are well known methods and APIs to do this.
# MAGIC 
# MAGIC   * Some of the metrics traditionally used to measure ad performance are the following:
# MAGIC         1. Impressions
# MAGIC         2. Clicks
# MAGIC         3. Conversions
# MAGIC         4. Quality Score
# MAGIC         5. Click Through Rate
# MAGIC 
# MAGIC * We can leverage Spark ML and Databricks ML Runtime for building a set of predictive models to predict ad performance
# MAGIC   * For example: We can use [Gradient-Boosted Trees (GBTs)](https://en.wikipedia.org/wiki/Gradient_boosting) for regression, which is available off-the-shelf from Spark ML Library.
# MAGIC   * in this notebook we provide an implementation of a [gradient boosted tree](https://docs.databricks.com/_static/notebooks/gbt-regression.html) for predicting [Ad Impressions](https://support.google.com/google-ads/answer/6320?hl=en)
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the Environment
# MAGIC 
# MAGIC In the following cells, we will:
# MAGIC   1. Install and import libraries
# MAGIC   2. Run the `99_utils` notebook to gain access to the function `get_params`
# MAGIC   3. View the parameters returned by `get_params`
# MAGIC   4. `get_params` and store values in variables  
# MAGIC   5. set to use the default database

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install required libraries
# MAGIC 
# MAGIC 
# MAGIC [Hyperopt: Distributed Hyperparameter Optimization](https://github.com/hyperopt/hyperopt)
# MAGIC 
# MAGIC %pip install hyperopt
# MAGIC 
# MAGIC [MLflow](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-python.html)
# MAGIC 
# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC The original accelerator uses three separate notebooks to configure the environment:
# MAGIC * 99_cleanup
# MAGIC * 99_config
# MAGIC * 99_utils
# MAGIC 
# MAGIC They can be found in the same folder as this notebook

# COMMAND ----------

# DBTITLE 1,Set Environment 
# MAGIC %run ./99_utils

# COMMAND ----------

# DBTITLE 1,Get database and paths names (for this accelerator environment)
params = get_params()
database_name = params['database_name']
raw_data_path = params['raw_data_path']
bronze_tbl_path = params['bronze_tbl_path']
print(database_name)
print(raw_data_path)

# COMMAND ----------

# DBTITLE 1,Select database
print(database_name)
_ = spark.sql('USE {}'.format(database_name))

# COMMAND ----------

# DBTITLE 1,Import libraries
from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK
import mlflow

from pyspark.ml.pipeline import Transformer
from pyspark.ml.util import Identifiable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import DoubleType

from pyspark.sql.functions import col

# COMMAND ----------

# DBTITLE 1,Load Train Data
data = (spark.table("train_impr_clicks_and_cost")
                    .withColumnRenamed('visit_impr', 'impr')
                    .withColumnRenamed('sum_visit_impr', 'user_visits')
                    .withColumnRenamed('visit_clicks', 'clicks')
                    .withColumnRenamed('visit_txns', 'txns')
                    .withColumnRenamed('max_cpm', 'cpm')
                    .withColumnRenamed('avg_cpi', 'cpi')
                    .withColumnRenamed('bid_cpc', 'cpc')
       )
    
display(data.sort(col('gpt').desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reduce train size
# MAGIC 
# MAGIC For demo purposes only, we'll provide a demo option to just keep 20% of the training data 
# MAGIC 
# MAGIC Note: This is not what we would do in a production environment, this is just so that this entire notebook can run end-to-end fast enough to demonstrate on a quick demo session. Kind of smoke test and for illustration purposes.

# COMMAND ----------

# DBTITLE 1,Reduce train size (for Demo purposes)
#To make this notebook run fast: keep 20% of data
split_perc = .2

#uncomment and set to None if you want to run on entire dataset 
#split_perc = None

#this will choose between the two options
if split_perc is not None:
    train_data, val_data = data.randomSplit([split_perc, 1-split_perc], seed=1234)
else:
    train_data, val_data = data, None
    
#just for imformative purposes, print how many records we are keeping 
from_nrows = data.count()
to_nrows = train_data.count()
print(f"keeping {to_nrows} rows from total {from_nrows} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check train data format and schema
# MAGIC 
# MAGIC Spark MLib models require data in specific format.
# MAGIC 
# MAGIC For example:
# MAGIC * Categorical variables might need to be encoded
# MAGIC * Certain Spark data types might be mandatory for certain Spark models (Ex. DoubleType)
# MAGIC * A feature vector needs to be created, and named "features" column
# MAGIC * All of these needs to be repeatable, as you would want to perform same data preparation to train, test, and prediction data.
# MAGIC 
# MAGIC This next cell is just informative; the actual data preparation happens in subsequent cells below

# COMMAND ----------

# DBTITLE 1,Sanity Check on String Columns and Train Data Schema and Data Types
#show list of string columns
columnList = [item[0] for item in train_data.dtypes if item[1].startswith('string')]
print(f"string columns: {columnList}")
train_data.printSchema()
#display(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, Test, and, Validation split
# MAGIC 
# MAGIC "The standard procedure is to split any training data into three sets: The first is the training data, used to train any model. The
# MAGIC second is used to assess which model has a better test performance. Once we have chosen our optimal model on the basis of
# MAGIC using validation data, we can get an unbiased estimate of the expected performance of this model by using a third set of independent test data."
# MAGIC 
# MAGIC source: Bayesian Reasoning and Machine Learning. David Barber. 2012
# MAGIC 
# MAGIC In this notebook we'll use a slightly modified standard procedure use of a three set split best practice:
# MAGIC * **Train** - data used to train and crossvalidate and hyper-parameter tune our model using MLFlow and Hyperopt and other platforms available
# MAGIC * **Test** - data used to evaluate predictions and apply multiple error measures and potentially confidence intervals. Ex. potentially using boostrapping or other methods
# MAGIC * **Validation** - data used to compare our forecast with "actual" data (not used in any of the two previous steps)
# MAGIC 
# MAGIC Just for context and for the interested reader:
# MAGIC 
# MAGIC In the Machine Learning and Deep Learning communities, the terms "test" and "validation" sets are sometimes used interchangeably.
# MAGIC   * For [additional information](https://machinelearningmastery.com/difference-test-validation-datasets/) on controversies and discussion
# MAGIC   * the key point to remember about validation and test sets are that they are both hold-out sets - [that should not be used in training](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html)
# MAGIC   * In my view, the term "test" just feels more natural than the term "validation" (the word test is shorter than the word validation)
# MAGIC   * many things have changed since the 1990s that is when the original defintiion of train, test, and validation sets was defined
# MAGIC   * overfitting and generalization are the real problem: a forecast is just as good as it performs with real data
# MAGIC 
# MAGIC 
# MAGIC The classical "canonical" definition is as following:
# MAGIC 
# MAGIC * Training set: A set of examples used for learning, that is to fit the parameters of the classifier.
# MAGIC * Validation set: A set of examples used to tune the parameters of a classifier, for example to choose the number of hidden units in a neural network.
# MAGIC * Test set: A set of examples used only to assess the performance of a fully-specified classifier.
# MAGIC 
# MAGIC source: Brian Ripley, page 354, Pattern Recognition and Neural Networks, 1996
# MAGIC 
# MAGIC 
# MAGIC The modern definition is continuously changing:
# MAGIC 
# MAGIC "The validation dataset is different from the test dataset that is also held back from the training of the model, but is instead used to give an unbiased estimate of the skill of the final tuned model when comparing or selecting between final models" 
# MAGIC 
# MAGIC source: What is the Difference Between Test and Validation Datasets? Jason Brownlee, 2017 

# COMMAND ----------

# DBTITLE 1,Set aside some "test" data to Evaluate Performance (this is different from held-out Validation set)
#best practice - even when using methods such as cross-validation 
train_data, test_data = train_data.randomSplit([.8,.2], seed=1234)
print(f"data to train: {train_data.count()} rows")
print(f"data to evaluate/test: {test_data.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Impressions - Predictive Model
# MAGIC 
# MAGIC Ad impressions (IMPR) are counted when an add displays on a screen.
# MAGIC * predicting the number of times an ad will be shown will be treated as a regression problem
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

# MAGIC %md
# MAGIC ### Prepare Data for Spark Models
# MAGIC 
# MAGIC Spark MLib models require data in specific format
# MAGIC 
# MAGIC In the following cell:
# MAGIC 1. Identify feature columns
# MAGIC 2. prepare data for Spark MLib model
# MAGIC 3. Cast all numeric columns to DoubleType
# MAGIC 4. Remove null values
# MAGIC 5. Encode String columns

# COMMAND ----------

# DBTITLE 1,Prepare data for Spark models - Part 1
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

#identify feature columns
key_cols = ['uid', 'channel', 'product_stars']
impr_target_col =  ['impr']
drop_cols = ['time', 'gpt']
#impr_feature_cols = train_data.columns - set(key_cols) - set(impr_target_cols) - set(drop_cols)
impr_feature_cols = [ 'timestamp', 'month', 'dayofweek', 'weekofyear', 'channelIndex', 'product_stars', 
                      'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]

# Spark Mlib API requires cast all numeric cols to Double
numeric_cols= [ 'impr', 'timestamp', 'month', 'dayofweek', 'weekofyear', 'product_stars', 
                'user_visits',  'clicks', 'txns', 'cpm', 'cpi', 'cpc', 'VISIT_RANK2' ]

#data prep function to be re-used for test and training pipeline
def prep_data(train_data):

    def cast_to_double(df, numeric_cols):
        for col_name in numeric_cols:
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
        return df

    train_data2 = cast_to_double(train_data,
                                numeric_cols)
    #train_data2.printSchema()

    #removing any null values which could have not been removed before
    #the assumption is that the data is already clean
    #TODO. investigate if there are any null values downstream
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
    return train_data5


# COMMAND ----------

# DBTITLE 1,Prepare data for Spark models - Part 2
# Spark Mlib API requires specific format
clean_train_data = prep_data(train_data)

#collect features into a "features" vector column 
impr_assembler = VectorAssembler(
        inputCols = impr_feature_cols,
        outputCol="features")

# Spark Mlib API requires two columns: 
#("label","features")
train_features_df = impr_assembler.transform(clean_train_data)

train_features_df2 = train_features_df.select(key_cols + impr_target_col + ["features"])
display(train_features_df2)

# COMMAND ----------

# DBTITLE 1,Custom Transformer for Spark Pipeline
from pyspark.ml.pipeline import Transformer
from pyspark.ml.util import Identifiable

class PrepDataTransformer(Transformer):
    def _prep_data(self, train_data):
        def cast_to_double(df, numeric_cols):
            for col_name in numeric_cols:
                df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
            return df

        train_data2 = cast_to_double(train_data,
                                    numeric_cols)
        #train_data2.printSchema()

        #removing any null values which could have not been removed before
        #the assumption is that the data is already clean
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
        return train_data5
        
    def __init__(self):
        pass
        #self.inputCol = inputCol #the name of your columns
        #self.outputCol = outputCol #the name of your output column
    def this():
        #define an unique ID
        #make this transformer object identifiable and immutable within our pipeline by assigning it a unique ID
        this(Identifiable._randomUID())
    def copy(extra):
        defaultCopy(extra)
    #def check_input_type(self, schema):
    #    field = schema[self.inputCol]
    #    #assert that field is a datetype 
    #    if (field.dataType != DateType()):
    #        raise Exception('DayExtractor input type %s did not match input type DateType' % field.dataType)
    def _transform(self, df):
        return self._prep_data(df)
    
    
prep_data = PrepDataTransformer()
dummy = prep_data.transform(train_data)
display(dummy)

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
impr_gbt_pipeline = Pipeline(stages=[prep_data, impr_assembler, impr_gbt])

#train the model in a Pipeline
impr_pipelineModel = impr_gbt_pipeline.fit(train_data)

#show model summary
impr_gbt_model = impr_pipelineModel.stages[2]
print(impr_gbt_model)  

# COMMAND ----------

# DBTITLE 1,Train Impressions - Predict and Evaluate
#train
impr_pred1 = impr_pipelineModel.transform(train_data)
impr_pred2 = impr_pipelineModel.transform(test_data)

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

impr_gbt_pipeline2 = Pipeline(stages=[prep_data, impr_assembler, impr_cv_model.bestModel])
impr_pipelineModel2 = impr_gbt_pipeline2.fit(train_data)
impr_cv_prediction2 = impr_pipelineModel2.transform(test_data)

print(f"rmse with cross validation train data: {impr_evaluator.evaluate(impr_cv_prediction1)}")
print(f"rmse with cross validation test data: {impr_evaluator.evaluate(impr_cv_prediction2 )}")
