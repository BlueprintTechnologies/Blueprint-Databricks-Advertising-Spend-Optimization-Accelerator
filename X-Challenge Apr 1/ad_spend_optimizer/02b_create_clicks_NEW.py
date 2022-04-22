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
# MAGIC * As of this X-Challenge version the original Databricks data is still required.
# MAGIC    * if not already done, please run the original Databricks notebooks to create the BRONZE and SILVER attibution tables

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/ad_spend_flow_01.png)

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC ### In this notebook you:
# MAGIC * Add Impressions and Click Share data
# MAGIC * Add Cost data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure the Environment
# MAGIC 
# MAGIC In this step:
# MAGIC   1. Import libraries
# MAGIC   2. Run `utils` notebook to gain access to the function `get_params`
# MAGIC   3. `get_params` and store values in variables

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.1: Import libraries

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import *
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.2: Run `utils` notebook to gain access to the function `get_params`
# MAGIC * `%run` is magic command provided within Databricks that enables you to run notebooks from within other notebooks.
# MAGIC * `get_params` is a helper function that returns a few parameters used throughout this solution accelerator. Usage of these parameters will be made explicit when used.

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.3: `get_params` and store values in variables
# MAGIC * Three of the parameters returned by `get_params` are used in this notebook. For convenience, we will store the values for these parameters in new variables.
# MAGIC   * **database_name:** the name of the database created in notebook `02_load_data`. The default value can be overridden in the notebook `99_config`.
# MAGIC   * **raw_data_path:** the path used when reading in the data generated in the notebook `01_intro`.
# MAGIC   * **bronze_tbl_path:** the path used in `02_load_data` to write out bronze-level data in delta format.

# COMMAND ----------

params = get_params()
database_name = params['database_name']
raw_data_path = params['raw_data_path']
bronze_tbl_path = params['bronze_tbl_path']
print(database_name)
print(raw_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.4: Select Database

# COMMAND ----------

print(database_name)
_ = spark.sql('USE {}'.format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load bronze-level table in Delta Lake
# MAGIC   
# MAGIC * bronze-level tables for raw data are crated by original Databricks Accelerator (notebooks marked with ORIGINAL notebook name)
# MAGIC 
# MAGIC * The nomenclature of bronze, silver, and gold tables correspond with a commonly used data modeling approach known as multi-hop architecture. 
# MAGIC   * Additional information about this pattern can be found [here](https://databricks.com/blog/2019/08/14/productionizing-machine-learning-with-delta-lake.html).

# COMMAND ----------

# DBTITLE 1,Load synthetic data created by Original Databricks Accelerator
bronze_data = spark.sql(f"select * from {database_name}.bronze")
display(bronze_data)

# COMMAND ----------

# DBTITLE 1,Quick Data Exploration
display(bronze_data.select("interaction").groupby("interaction").count())

# COMMAND ----------

# DBTITLE 1,Impression Counts Percentage per Channel
from pyspark.sql.functions import lit
nrows = bronze_data.count()
print(f"{nrows} rows")
per_channel = (bronze_data.select("channel")
               .groupby("channel")
               .count()
               .withColumn("total_cnt", lit(nrows))
               .withColumn("p_channel", col('count') / col('total_cnt') )
              )

display(per_channel)

# COMMAND ----------

# DBTITLE 1,Average number of interactions per user (created by Original Databricks Accelerator)
users = bronze_data.select('uid').distinct()
nusers = users.count()
print(f"number of users: {nusers} total number of interactions: {nrows}")
per_user = (bronze_data.select("uid")
               .groupby("uid")
               .count()
               .withColumn("total_cnt", lit(nrows))
               .withColumn("p_user", col('count') / col('total_cnt') )
              )
display(per_user.select("count").distinct().sort('count', ascending = False))
#display(per_user)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Clicks
# MAGIC 
# MAGIC for simulating purposes will assume a click through rate for each ad ctr according to a simulated position on a google search like query
# MAGIC * Ex. first position on the search will get the majority of clicks 
# MAGIC * last position on the search gets very few clicks
# MAGIC 
# MAGIC This is only for the purpose of generating synthetic data
# MAGIC * on a real life scenario, the number of clicks should come directly from the ad platform or website (such as Google Ads)

# COMMAND ----------

# DBTITLE 1,Simulate Clicks
import math
import matplotlib.pyplot as plt
import random
#define click through rates - for simulating purposes 10 ctr buckets
ctr_ranks = list(range(10))
max_ctr_bucket = max(ctr_ranks)
zctr_buckets = [ x/max_ctr_bucket * 2.5 for x in ctr_ranks ]
#for simulating purposes consider the ctr rank to be approximately distributed as an exponential or "normal" distribution
ctr_perc_levels = [round(math.exp(-(x**2)),2) for x in zctr_buckets]
max_ctr = .12
ctr_levels = [max_ctr * x for x in ctr_perc_levels]
#create a dictionary with ctrs - this is going to be used later
ctrs = dict(zip(ctr_ranks, ctr_levels))
print(ctr_ranks)
print(ctr_perc_levels )
print(ctr_levels)
print(ctrs)
plt.plot(ctr_perc_levels)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Likes
# MAGIC 
# MAGIC On a similar way, for simulation purposes we'll simualte the number of "likes" on a product on our website or add
# MAGIC * Ex. Product with 5 stars product reviews tend to have more likes
# MAGIC * products with less stars tend to have less

# COMMAND ----------

# DBTITLE 1,Simulate Likes
nproduct_review_categories = 5
stars_per_product = list(range(1,nproduct_review_categories+1))
max_stars = max(stars_per_product)
max_likes = 1261 
zstars_buckets = [ x/max_stars*2 for x in stars_per_product ]
like_perc_levels = [round(math.exp(-(x**2)),2) for x in zstars_buckets]
like_buckets = [int(max_likes * perc) for perc in like_perc_levels]
#create a view items table or dictionary 
#in a real project, this will normally include statistics of top items a user visited or was interested in his customer journey
view_items_like_perc = dict(zip(reversed(stars_per_product), like_perc_levels))
print(stars_per_product)
print(zstars_buckets)
print(like_perc_levels)
print(like_buckets)
print(view_items_like_perc)

# COMMAND ----------

# DBTITLE 1,Simulate Click Through Rates Score per User
import pyspark.sql.functions as F
from pyspark.sql.functions import col, create_map, lit
from itertools import chain

#use ctrs dictionary created before
mapping_expr = create_map([lit(x) for x in chain(*ctrs.items())])

#select all users
users = bronze_data.select('uid').distinct()
users = (users
         .withColumn("ctr_rank", (F.rand()*10).cast("int"))
         .withColumn("ctr_perc", mapping_expr[col("ctr_rank")])
         .withColumn("star_array", F.array(lit(1),
                                           lit(2),
                                           lit(3),
                                            lit(4),
                                          lit(5))
                    )
        )
#ctr_buckets = sorted(row[0] for row in users.select("ctr_rank").distinct().collect())
#print(ctr_buckets)
display(users)

# COMMAND ----------

# DBTITLE 1,Simulate User Behavior
from pyspark.sql.functions import explode

#use view items table or dictionary 
mapping_expr2 = create_map([lit(x) for x in chain(*view_items_like_perc.items())])

#select all users and find distribution of clicks per type of product review
user_preferences = (users.select(["uid", "ctr_rank", "ctr_perc", explode(col("star_array")).alias("product_stars")])
                    .withColumn("like_perc", mapping_expr2[col("product_stars")])
                    .withColumn("ctr", col("ctr_perc") * 1.3 * col("like_perc"))
                   )
display(user_preferences)

# COMMAND ----------

# DBTITLE 1,Save Simulated Click Through Rate by User to Delta Lake
ctrs_table_name = "ctrs_by_user_product_stars_and_likes"

(user_preferences
   .write
   .format("delta")
   .mode("overwrite")
  .saveAsTable(ctrs_table_name)
)


# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from ctrs_by_user_product_stars_and_likes

# COMMAND ----------

# DBTITLE 1,Load Bronze data again (we would need to further enrich)
#no need to reload as we are already in spark
display(bronze_data)

# COMMAND ----------

# DBTITLE 1,Simulate Product Bundles and Product Reviews
#add product stars column
clicks_data = (bronze_data
               .withColumn("product_stars",  ((F.rand()*5)+1).cast("int"))
              )

#join with ctr and stars per user
user_likes = spark.sql("Select * from ctrs_by_user_product_stars_and_likes")
bronze_data2 = (clicks_data.join(user_likes, on = ['uid', 'product_stars'], how = 'left')
               )

#star_ids = temp.select("product_stars").distinct()
display(bronze_data2)

# COMMAND ----------

# DBTITLE 1,Simulate User Clicks
from pyspark.sql.functions import pandas_udf, PandasUDFType, when
from pyspark.sql.types import IntegerType
import pandas as pd
#from numpy.random import choice

#use pandas udf to simulate user click
@pandas_udf("float", PandasUDFType.SCALAR)
def user_click(ctrs: float) -> int:
    import numpy as np
    #return random.choices([0, 1 ], 
    #                         (1-ctr_val, ctr_val), k=1)[0]
    user_clicks = []
    for ctr_val in ctrs:
       adj_ctr_val = 1.3 * ctr_val
       click = np.random.choice([0, 1 ], 1,
              p=(1-adj_ctr_val, adj_ctr_val))
       user_clicks.append(click.item())
    return pd.Series(user_clicks)

  
print(bronze_data2.count())
bronze_data3 = (bronze_data2.withColumn("click", user_click("ctr"))
                .withColumn("click", when(col("conversion") == 1, 1).otherwise(col("click"))) 
               )
display(bronze_data3)

# COMMAND ----------

# DBTITLE 1,Number of Clicks per Type of Product and CTR Rank
print(bronze_data3.count())
temp2 = bronze_data3.select(["ctr_rank", "product_stars", "click"]).groupby("ctr_rank", "product_stars").sum("click")
display(temp2)

# COMMAND ----------

# DBTITLE 1,Define Bronze Clicks Output Columns
out_cols1 = ['uid', 'time', 'interaction', 'channel', 'conversion', 'click', 'product_stars' ] 
out_cols2 = ['ctr_rank', 'ctr_perc', 'like_perc', 'ctr']
out_cols3 = out_cols1 + out_cols2
print(len(bronze_data.columns), bronze_data.columns)
print(len(bronze_data3.columns), bronze_data3.columns)
print(len(set(out_cols3)), out_cols3)
bronze_with_clicks = bronze_data3.select(out_cols1)
display(bronze_with_clicks)

# COMMAND ----------

ctrs_per_transaction = bronze_data3.select(out_cols3)
display(ctrs_per_transaction)

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
ctrs_per_transaction_table_name = "ctr_per_txn_with_clicks"

(ctrs_per_transaction
   .write
   .format("delta")
   .mode("overwrite")
  .saveAsTable(ctrs_per_transaction_table_name)
)


# COMMAND ----------

#spark.sql("DROP TABLE IF EXISTS bronze_with_clicks")

# COMMAND ----------

bronze_clicks_table_name = "bronze_with_clicks"

(bronze_with_clicks
   .write
   .format("delta")
   .mode("overwrite")
  .saveAsTable(bronze_clicks_table_name)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC * In the next notebook, we will prepare this data so that it can be used for attribution modeling with Markov Chains.

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Matplotlib|Python Software Foundation (PSF) License |https://matplotlib.org/stable/users/license.html|https://github.com/matplotlib/matplotlib|
# MAGIC |Numpy|BSD-3-Clause License|https://github.com/numpy/numpy/blob/master/LICENSE.txt|https://github.com/numpy/numpy|
# MAGIC |Pandas|BSD 3-Clause License|https://github.com/pandas-dev/pandas/blob/master/LICENSE|https://github.com/pandas-dev/pandas|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Seaborn|BSD-3-Clause License|https://github.com/mwaskom/seaborn/blob/master/LICENSE|https://github.com/mwaskom/seaborn|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
