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

# DBTITLE 1,Score Data with Machine Learning and Databricks
# MAGIC %md
# MAGIC ### Generate Predictions for
# MAGIC * Impressions per client per channel
# MAGIC * CTR per client per channel
# MAGIC * CVR per client per channel
# MAGIC * Txns per client
# MAGIC * Cost per Impressions per channel
# MAGIC * Cost per click per client

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/ad_spend_flow_05.png)

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

from pyspark.sql.functions import lit, when, col

# COMMAND ----------

scoring_data = (spark.sql("""select *, NTILE(10) OVER (ORDER BY VISIT_RANK2) AS user_rank
                             from val_impr_clicks_and_cost""")
                    .withColumnRenamed('visit_impr', 'impr')
                    .withColumnRenamed('sum_visit_impr', 'user_visits')
                    .withColumnRenamed('visit_clicks', 'clicks')
                    .withColumnRenamed('visit_txns', 'txns')
                    .withColumnRenamed('max_cpm', 'cpm')
                    .withColumnRenamed('avg_cpi', 'cpi')
                    .withColumnRenamed('bid_cpc', 'cpc')
                    .withColumnRenamed('impr', 'impr_pred')
                    .withColumnRenamed('clicks', 'clicks_pred')
                    .withColumnRenamed('txns', 'txns_pred')
                    .withColumn('ctr_pred', when( col("impr_pred") > 0 , col("clicks_pred") / col("impr_pred") ).otherwise(0)  )      
                    .withColumn('cvr_pred', when( col("clicks_pred") > 0 , col("txns_pred") / col("clicks_pred") ).otherwise(0) )
                    .withColumnRenamed('gpt', 'gpt_pred')
                    .withColumnRenamed('cpi', 'cpi_pred')
                    .withColumnRenamed('cpc', 'cpc_pred') 
       )

pred_cols = [col_name for col_name in scoring_data.columns if 'pred' in col_name]
#['impr_pred', 'clicks_pred', 'txns_pred', 'cpi_pred', 'cpc_pred', 'gpt_pred', 'ctr_pred', 'cvr_pred']
user_cols = ['uid', 'user_rank']
ts_cols = [ 'time', 'month', 'dayofweek', 'weekofyear', 'timestamp']
bid_unit_cols = ['user_rank', 'channel', 'product_stars']
group_by_cols = ['month', 'weekofyear'] + bid_unit_cols
key_cols_str = ", ".join(group_by_cols)
print(bid_unit_cols)
print(scoring_data.columns)
print(key_cols_str)
print(pred_cols)
scoring_data_raw = scoring_data.select(group_by_cols + pred_cols)
scoring_data_raw.createOrReplaceTempView('scoring_data')

scoring_data2 = (spark.sql("""
            SELECT *, 
                  A.sum_ctr_pred / A.cnt AS ctr_pred,
                  A.sum_cvr_pred / A.cnt AS cvr_pred
            FROM (
            SELECT month, weekofyear, user_rank, channel, product_stars,
                  sum(impr_pred) AS impr_pred,
                  sum(clicks_pred) AS clicks_pred,
                  sum(txns_pred) AS txns_pred,
                  avg(cpi_pred) AS cpi_pred,
                  avg(cpc_pred) AS cpc_pred,
                  avg(gpt_pred) AS gpt_pred,
                  sum(ctr_pred) AS sum_ctr_pred,
                  sum(cvr_pred) AS sum_cvr_pred,
                  count(*) AS cnt
            FROM scoring_data
            GROUP BY month, weekofyear, user_rank, channel, product_stars
            ) AS A 
        """).drop(*['sum_ctr_pred', 'sum_cvr_pred', 'cnt'])
        )

display(scoring_data2)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *, 
# MAGIC       A.sum_ctr_pred / A.cnt AS ctr_pred,
# MAGIC       A.sum_cvr_pred / A.cnt AS cvr_pred
# MAGIC FROM (
# MAGIC SELECT month, weekofyear, user_rank, channel, product_stars,
# MAGIC       sum(impr_pred) AS impr_pred,
# MAGIC       sum(clicks_pred) AS clicks_pred,
# MAGIC       sum(txns_pred) AS txns_pred,
# MAGIC       avg(cpi_pred) AS cpi_pred,
# MAGIC       avg(cpc_pred) AS cpc_pred,
# MAGIC       avg(gpt_pred) AS gpt_pred,
# MAGIC       sum(ctr_pred) AS sum_ctr_pred,
# MAGIC       sum(cvr_pred) AS sum_cvr_pred,
# MAGIC       count(*) AS cnt
# MAGIC FROM scoring_data
# MAGIC GROUP BY month, weekofyear, user_rank, channel, product_stars
# MAGIC ) AS A 

# COMMAND ----------

pred_cols = [col_name for col_name in scoring_data.columns if 'pred' in col_name]
print(pred_cols)

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
score_table_name = "score_data_with_pred"
out_score_df = scoring_data2

(out_score_df
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(score_table_name)
)

# COMMAND ----------


