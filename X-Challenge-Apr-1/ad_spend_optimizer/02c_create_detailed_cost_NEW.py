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
# MAGIC ![](/files/ad_spend_flow_01.png)

# COMMAND ----------

# DBTITLE 1,Load Original Databricks Database Names and Environment Definitions
# MAGIC %run ./99_utils

# COMMAND ----------

params = get_params()
database_name = params['database_name']
raw_data_path = params['raw_data_path']
bronze_tbl_path = params['bronze_tbl_path']
print(database_name)
print(raw_data_path)

# COMMAND ----------

# DBTITLE 1,Use new database
print(database_name)
_ = spark.sql('USE {}'.format(database_name))

# COMMAND ----------

# DBTITLE 1,Load simulated clicks data
clicks_table_name = "bronze_with_clicks"
clicks_data = spark.sql(f"SELECT * FROM {clicks_table_name}")
display(clicks_data)

# COMMAND ----------

# DBTITLE 1,Show Marketing Spend Table from Original Databricks Accelerator
# MAGIC %sql
# MAGIC SELECT * FROM gold_ad_spend

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC ###  Simualte Costs for Digital Advertising
# MAGIC 
# MAGIC There are two common types of pricing models:
# MAGIC 
# MAGIC     
# MAGIC * **cost-per-mille (CPM)**
# MAGIC   * cost-per-thousand-impressions
# MAGIC * **cost-per-click (CPC)**
# MAGIC   * cost per each time a user clicks on an ad
# MAGIC 
# MAGIC 
# MAGIC In general, different advertising channels may have different princing models.
# MAGIC 
# MAGIC * For example, Search Engine Marketing or SEM could charge a cost per click and Social Media Advertising may charge a cost per number of impressions. 
# MAGIC * As a general rule, the same principles that apply to multi-touch attributions apply to assigning costs attributions. 
# MAGIC 
# MAGIC In this accelerator, we'll handle both types of pricing models 
# MAGIC 
# MAGIC * by defining a dictionary of synthetic CPC and CPM rates
# MAGIC * by simulating user behavior (clicks, view items, and likes) and preferences (likes, revies, stars, and conversions) according to pre-specified user and product categories

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Cost-per-thousand impressions (CPM)
# MAGIC Definition: A way to bid where you pay per one thousand views (impressions) on the Google Display Network.
# MAGIC 
# MAGIC Viewable CPM (vCPM) bidding ensures that you only pay when your ads can be seen. Existing CPM bids will be converted to vCPM automatically, but it's best to update your bids since viewable impressions are potentially more valuable. Learn more about using viewable CPM bids.
# MAGIC 
# MAGIC * 1000 pageviews * 5 ad units = 5000 ad impressions
# MAGIC 
# MAGIC     * Ex: 5000 Ads = $15
# MAGIC     * 1000 Ads = 15/5 = $3
# MAGIC     * Hence, your CPM is $3

# COMMAND ----------

# DBTITLE 1,Define CPM and Cost per Clicks per channel
# CPMs per channel
cpm_per_channel = {'Social Network': 8.0, 
                   'Search Engine Marketing':2.43, 
                   'Google Display Network':3.12, 
                   'Affiliates':20.0, 
                   'Email':15.0}
print(f"cpm_rates: {list(cpm_per_channel.items())}")

# CPCs per channel
cpc_per_channel = {'Social Network': 0.92, 
                   'Search Engine Marketing':2.62, 
                   'Google Display Network':0.63, 
                   'Affiliates':0.88, 
                   'Email':0.91}
print(f"cpc_rates: {list(cpc_per_channel.items())}")

# COMMAND ----------

# DBTITLE 1,Simulate Impressions and CPM data
import pyspark.sql.functions as F
from pyspark.sql.functions import col, create_map, lit, when
from itertools import chain
mapping_expr = create_map([lit(x) for x in chain(*cpm_per_channel.items())])
detailed_costs = (clicks_data
                  .withColumn("impression", 
                              when(col("interaction") == 'impression',1).otherwise(0))
                  .withColumn("cpm", mapping_expr[col("channel")])
                  .withColumn("cpi", col("cpm")/1000)
    )

display(detailed_costs)

# COMMAND ----------

# DBTITLE 1,Simulate Detailed Spend per Channel 
mapping_expr2 = create_map([lit(x) for x in chain(*cpc_per_channel.items())])
detailed_costs2 = (detailed_costs
                   .withColumn("cpc", mapping_expr2[col("channel")])
                  )
display(detailed_costs2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate Revenue per Ad

# COMMAND ----------

# DBTITLE 1,Define Products Category and Product Segments
# revenue per product category
avg_price = 139.39
gpt_per_product = { 5 : avg_price * 3.0/2, 
                    4 : avg_price * 2.5/2, 
                    3 : avg_price/2, 
                    2 : avg_price * 0.5/2, 
                    1 : avg_price * 0.15/2}

print(gpt_per_product)
display(spark.createDataFrame(gpt_per_product.items(), schema = ["stars", "avg_order_value"]))

# COMMAND ----------

# DBTITLE 1,Simulate Revenue per Product Category
mapping_expr3 = create_map([lit(x) for x in chain(*gpt_per_product.items())])
detailed_costs3 = (detailed_costs2
                    .withColumn("potential_gpt", mapping_expr3[col("product_stars") ])
                    .withColumn("gpt", when(col("conversion")==1, col("potential_gpt")).otherwise(0))
                  )

display(detailed_costs3)

# COMMAND ----------

# DBTITLE 1,Detailed Marketing Contribution per Channel
contribution_per_channel = (detailed_costs3.groupby("channel").sum(*["impression", "click", "conversion", "cpi", "cpc", "gpt"])
                   .withColumn("marketing_cost", col("sum(cpi)") + col('sum(cpc)'))
                   .withColumn("marketing_contribution", col("sum(gpt)") - col("marketing_cost"))
                   .withColumn("ROAS", col("sum(gpt)") / col("marketing_cost"))
                  )
display(contribution_per_channel)

# COMMAND ----------

#spark.sql("DROP TABLE IF EXISTS bronze_with_clicks_and_cost")

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
detailed_cost_table_name = "bronze_with_clicks_and_cost"
bronze_with_cost = detailed_costs3.drop("potential_gpt")

(bronze_with_cost
   .write
   .format("delta")
   .mode("overwrite")
  .saveAsTable(detailed_cost_table_name)
)


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bronze_with_clicks_and_cost
