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
# MAGIC ![](/files/ad_spend_flow_02.png)

# COMMAND ----------

# DBTITLE 1,Low environment and table names
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

# DBTITLE 1,Load data
#input_table_name = "bronze_with_clicks_and_cost"
input_table_name = 'silver_with_clicks_and_cost'
data = spark.table(input_table_name)
print(data.count())
display(data)

# COMMAND ----------

# DBTITLE 1,Train, Validation Split
train_data, test_data = data.randomSplit([.8,.2], seed=1234)
print(train_data.count())
print(test_data.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_ad_spend

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS train_impr_clicks_and_cost")

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
train_table_name = "train_impr_clicks_and_cost"

(train_data
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(train_table_name)
)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS val_impr_clicks_and_cost")

# COMMAND ----------

test_table_name = "val_impr_clicks_and_cost"

(test_data
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(test_table_name)
)
