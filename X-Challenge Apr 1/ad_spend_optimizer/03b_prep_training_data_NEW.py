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

# DBTITLE 1,Load environment and table names
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

# DBTITLE 1,Load Data
input_table_name = "bronze_with_clicks_and_cost"
bronze_clicks = spark.table(input_table_name)
bronze_clicks.createOrReplaceTempView("bronze_clicks")
print(bronze_clicks)
print(bronze_clicks.count())
display(bronze_clicks)

# COMMAND ----------

# DBTITLE 1,Summarize user behavior
user_visits= spark.sql("""SELECT uid, time, channel, product_stars, 
                         count(interaction) OVER (PARTITION BY uid, channel) AS visit_impr,
                         sum(click) OVER (PARTITION BY uid, channel, product_stars) AS visit_clicks,
                         sum(conversion) OVER (PARTITION BY uid,channel, product_stars) AS visit_txns,
                         max(cpm) OVER (PARTITION BY uid,channel, product_stars) AS max_cpm,
                         avg(cpi) OVER (PARTITION BY uid,channel, product_stars) AS avg_cpi,
                         max(cpc) OVER (PARTITION BY uid,channel, product_stars) AS bid_cpc,
                         avg(gpt) OVER (PARTITION BY uid,channel, product_stars) AS gpt
                    FROM bronze_clicks""").drop_duplicates()
#display(visits.filter(" uid = '00032268f59b4867bc1c4143de39de3c'"))
#print(user_visits.count())
display(user_visits)

# COMMAND ----------

# DBTITLE 1,Show Marketing Spend Table from Original Databricks Accelerator
# MAGIC %sql
# MAGIC SELECT * FROM gold_ad_spend

# COMMAND ----------

# DBTITLE 1,Feature Engineer Dates
from pyspark.sql.functions import month, col, dayofweek, weekofyear, unix_timestamp #minute, second

bronze_data2 = (user_visits
            .withColumn("month", month(col("time")))
            .withColumn("dayofweek", dayofweek(col("time")))
            .withColumn("weekofyear", weekofyear(col("time")))
            .withColumn("timestamp", unix_timestamp(col("time")))
        )

bronze_data2.createOrReplaceTempView('bronze_data2')
display(bronze_data2)

# COMMAND ----------

# DBTITLE 1,Rank Customers
import pyspark.sql.functions as F
bid_cols = ["visit_impr", "visit_clicks", "visit_txns", "avg_cpi", "bid_cpc", "gpt"]
funcs = [ ('sum', F.sum) ]     #('avg', F.avg), ('stddev', F.stddev), ('min', F.min),('max', F.max) ]
expr_cv = [f[1](F.col(c)).alias(f"{f[0]}_{c}") for f in funcs for c in bid_cols]

contribution_per_customer = (bronze_data2.groupby('uid').agg(*expr_cv)
                               .withColumn("marketing_cost", col("sum_avg_cpi") + col("sum_bid_cpc"))
                               .withColumn("marketing_contribution", col("sum_gpt") - col("marketing_cost"))
                               .withColumn("ROAS", col("sum_gpt") / col("marketing_cost"))
                            )

contribution_per_customer.createOrReplaceTempView("contribution_per_customer")
display(contribution_per_customer)

# COMMAND ----------

contribution_per_customer.select('sum_visit_impr').toPandas().hist()

# COMMAND ----------

contribution_per_customer.printSchema()

# COMMAND ----------

# DBTITLE 1,Rank Customers by how active/visits - Part 1
#%sql
df = spark.sql("""
    (SELECT 
      NTILE(100) OVER (ORDER BY sum_visit_impr) AS VISIT_RANK,
      --NTILE(10) OVER (ORDER BY sum_visit_clicks DESC) AS CLICKS_RANK,
      *
    FROM  contribution_per_customer)
"""
)
print(df.count())
max_impr = df.agg({"sum_visit_impr": "max"}).collect()[0]
min_impr = df.agg({"sum_visit_impr": "min"}).collect()[0]
max_rank = df.agg({"VISIT_RANK": "max"}).collect()[0]
min_rank = df.agg({"VISIT_RANK": "min"}).collect()[0]
print(max_impr,min_impr)
print(max_rank, min_rank)
display(df)

# COMMAND ----------

# DBTITLE 1,Rank Customers by how active/visits - Part 2
 df2 = spark.sql("""
 SELECT VISIT_RANK, visits, DENSE_RANK() OVER (ORDER BY visits) AS VISIT_RANK2 
 FROM 
 (SELECT 
   VISIT_RANK, sum(sum_visit_impr) as visits
   FROM 
   (SELECT 
      NTILE(100) OVER (ORDER BY sum_visit_impr) AS VISIT_RANK,
      --NTILE(10) OVER (ORDER BY sum_visit_clicks DESC) AS CLICKS_RANK,
      *
    FROM  contribution_per_customer) as A
   GROUP BY VISIT_RANK) as B   
 """
 )
display(df2)

# COMMAND ----------

# DBTITLE 1,Rank Customers by how active/visits - Part 3
df3 = df.join(df2.select('VISIT_RANK', 'VISIT_RANK2'), on = ["VISIT_RANK"], how = 'left')
display(df3.select('VISIT_RANK', 'VISIT_RANK2').distinct().sort(col('VISIT_RANK2'), col('VISIT_RANK')))

# COMMAND ----------

# DBTITLE 1,Rank Customers by how active/visits - Part 4
user_visit_rankings = df3.select('uid', 'sum_visit_impr', 'VISIT_RANK', 'VISIT_RANK2').distinct()
max_impr = user_visit_rankings.agg({"sum_visit_impr": "max"}).collect()[0]
min_impr = user_visit_rankings.agg({"sum_visit_impr": "min"}).collect()[0]
max_rank = user_visit_rankings.agg({"VISIT_RANK2": "max"}).collect()[0]
min_rank = user_visit_rankings.agg({"VISIT_RANK2": "min"}).collect()[0]
print(max_impr,min_impr)
print(max_rank, min_rank)
display(user_visit_rankings.sort(col('VISIT_RANK2').desc()))
#display(user_visit_rankings.select('VISIT_RANK', 'VISIT_RANK2').distinct().sort(col('VISIT_RANK2'), col('VISIT_RANK')))

# COMMAND ----------

# DBTITLE 1,Save to Delta
user_visit_rank_table_name = 'user_visit_rankings'

(user_visit_rankings
      .write
      .format("delta")
      .mode("overwrite")
      .saveAsTable(user_visit_rank_table_name)
)

# COMMAND ----------

# DBTITLE 1,Rank Products
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
from pyspark.sql.window import Window

bid_cols = ["visit_impr", "visit_clicks", "visit_txns", "avg_cpi", "bid_cpc", "gpt"]
funcs = [ ('sum', F.sum) ]     #('avg', F.avg), ('stddev', F.stddev), ('min', F.min),('max', F.max) ]
expr_cv = [f[1](F.col(c)).alias(f"{f[0]}_{c}") for f in funcs for c in bid_cols]

#prod_win_spec = 

contribution_per_product_detail = (bronze_data2.groupby('product_stars').agg(*expr_cv)
                               .withColumn("marketing_cost", col("sum_avg_cpi") + col("sum_bid_cpc"))
                               .withColumn("marketing_contribution", col("sum_gpt") - col("marketing_cost"))
                               .withColumn("ROAS", col("sum_gpt") / col("marketing_cost"))
                               .withColumn("product_rank", F.dense_rank().over(Window.orderBy(col("ROAS").desc())))
                            )

contribution_per_product = contribution_per_product_detail.select("product_stars", "marketing_contribution", "ROAS", "product_rank")

display(contribution_per_product)

# COMMAND ----------

# DBTITLE 1,Rank Channels
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
from pyspark.sql.window import Window

bid_cols = ["visit_impr", "visit_clicks", "visit_txns", "avg_cpi", "bid_cpc", "gpt"]
funcs = [ ('sum', F.sum) ]     #('avg', F.avg), ('stddev', F.stddev), ('min', F.min),('max', F.max) ]
expr_cv = [f[1](F.col(c)).alias(f"{f[0]}_{c}") for f in funcs for c in bid_cols]

#prod_win_spec = 

contribution_per_channel_detail = (bronze_data2.groupby('channel').agg(*expr_cv)
                               .withColumn("marketing_cost", col("sum_avg_cpi") + col("sum_bid_cpc"))
                               .withColumn("marketing_contribution", col("sum_gpt") - col("marketing_cost"))
                               .withColumn("ROAS", col("sum_gpt") / col("marketing_cost"))
                               .withColumn("channel_rank", F.dense_rank().over(Window.orderBy(col("ROAS").desc())))
                            )

#contribution_per_channel = contribution_per_channel_detail.select("product_stars", "marketing_contribution", "ROAS", "product_rank")
contribution_per_channel_detail.createOrReplaceTempView('per_channel_contribution')
display(contribution_per_channel_detail)

# COMMAND ----------

# DBTITLE 1,Number of Interactions and Customers per Channel
# MAGIC %sql
# MAGIC SELECT A.channel, A.sum_visit_clicks, B.ncust
# MAGIC FROM
# MAGIC (SELECT channel, sum_visit_clicks
# MAGIC FROM per_channel_contribution) A
# MAGIC JOIN
# MAGIC (select channel, count(*) as ncust
# MAGIC from bronze_clicks
# MAGIC group by channel) B
# MAGIC ON A.channel = B.channel 

# COMMAND ----------

# DBTITLE 1,Rank users according to Click Through Rate
# MAGIC %sql
# MAGIC SELECT CLICKS_RANK, SUM(sum_visit_clicks)
# MAGIC FROM 
# MAGIC (SELECT 
# MAGIC       --NTILE(10) OVER (ORDER BY sum_visit_impr DESC) AS IMPR_RANK,
# MAGIC       NTILE(10) OVER (ORDER BY sum_visit_clicks DESC) AS CLICKS_RANK,
# MAGIC       *
# MAGIC FROM  contribution_per_customer) AS A
# MAGIC GROUP BY CLICKS_RANK

# COMMAND ----------

# DBTITLE 1,Join User Rankings with Click Data
user_rankings = user_visit_rankings.select(['uid', 'sum_visit_impr', 'VISIT_RANK2'])
#print(user_visit_rankings.select(['uid', 'sum_visit_impr', 'VISIT_RANK2']).columns)
silver_data = bronze_data2.join(user_rankings, on = 'uid', how = 'left')
print(silver_data.count())
display(silver_data)

# COMMAND ----------

# DBTITLE 1,Summarize User Behavior - Part 2
user_visits2= (spark.sql("""SELECT uid, 
                         count(interaction) OVER (PARTITION BY uid) AS user_impr,
                         sum(click) OVER (PARTITION BY uid) AS user_clicks,
                         sum(conversion) OVER (PARTITION BY uid) AS user_txns,
                         max(cpm) OVER (PARTITION BY uid) AS user_cpm,
                         avg(cpi) OVER (PARTITION BY uid) AS user_cpi,
                         max(cpc) OVER (PARTITION BY uid) AS user_cpc,
                         avg(gpt) OVER (PARTITION BY uid) AS user_gpt
                    FROM bronze_clicks""").drop_duplicates()
                )

user_visits2.createOrReplaceTempView("user_visits2")

user_visits3 = spark.sql("""
                         SELECT uid, 
                             sum(user_impr) as user_impr,
                             sum(user_clicks) as user_clicks,
                             sum(user_txns) as user_txns,
                             avg(user_cpi) as user_cpi,
                             max(user_cpc) as user_cpc,
                             avg(user_gpt) as user_gpt
                             --sum(user_gpt) as sum_user_gpt
                             --std(user_gpt) as std_gpt
                         FROM
                             user_visits2
                         GROUP BY 
                             uid
                            """)

user_visits3.createOrReplaceTempView('user_visits3')
#display(visits.filter(" uid = '00032268f59b4867bc1c4143de39de3c'"))
display(user_visits3)

# COMMAND ----------

# DBTITLE 1,Summarize User Behavior - Part 3
user_visits4 = user_visits3.join(user_visit_rankings.select('uid', 'VISIT_RANK2'), on = 'uid', how = 'left')
user_visits4.createOrReplaceTempView("user_visits4")
display(user_visits4)

# COMMAND ----------

# DBTITLE 1,Summarize User Behavior - Part 4
user_visits5 = spark.sql(
"""
select VISIT_RANK2, sum(user_impr) impr, sum(user_clicks) clicks, sum(user_txns) txns,
        avg(user_cpi) as cpm, avg(user_cpc) as cpc, sum(user_gpt) as potential_gp, avg(user_gpt) avg_gpt, std(user_gpt) std_gpt 
from 
user_visits4
group by VISIT_RANK2
order by VISIT_RANK2
"""
)
display(user_visits5)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS users_revenue_per_segment")

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
out_table_name = 'users_revenue_per_segment'
(user_visits5
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(out_table_name)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from users_revenue_per_segment
# MAGIC order by potential_gp

# COMMAND ----------

# DBTITLE 1,Summarize User Behavior - Part 5
#%sql
df10 = spark.sql("""
    (SELECT 
      NTILE(100) OVER (ORDER BY user_impr) AS VISIT_RANK3,
      --NTILE(10) OVER (ORDER BY sum_visit_clicks DESC) AS CLICKS_RANK,
      *
    FROM  user_visits3)
"""
)
print(df10.count())
max_impr = df10.agg({"user_impr": "max"}).collect()[0]
min_impr = df10.agg({"user_impr": "min"}).collect()[0]
max_rank = df10.agg({"VISIT_RANK3": "max"}).collect()[0]
min_rank = df10.agg({"VISIT_RANK3": "min"}).collect()[0]
print(max_impr,min_impr)
print(max_rank, min_rank)
display(df10)

# COMMAND ----------

display(df10.groupby('VISIT_RANK3').sum('user_impr'))

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
silver_table_name = "silver_with_clicks_and_cost"
adsf
(silver_data
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(silver_table_name)
)

# COMMAND ----------

silver_clicks_raw = spark.sql("select * from silver_with_clicks_and_cost")
print(silver_clicks_raw.columns)

# COMMAND ----------

# DBTITLE 1,Summarize User Behavior - Part 6
silver_clicks2 = spark.sql("""
SELECT uid, channel, product_stars,
       month, dayofweek, weekofyear,
       --'time', 
       sum(visit_impr) as impr, 
       sum(sum_visit_impr) as total_impr,
       sum(visit_clicks) as clicks,
       sum(visit_txns) as txns,
       max(max_cpm) as cpm,
       avg(bid_cpc) as cpc,
       avg(gpt) as gpt
FROM 
    silver_with_clicks_and_cost
GROUP BY 
    uid, channel, product_stars, month, dayofweek, weekofyear
    """
    )
out_cols = ['month', 'weekofyear', 'dayofweek', 'uid', 'channel', 'product_stars', 'impr', 'total_impr', 'clicks', 'txns', 'cpm', 'cpc', 'gpt']
silver_clicks2 = silver_clicks2.select(out_cols)
silver_clicks2.createOrReplaceTempView('silver_clicks2')
print(silver_clicks2.count())
display(silver_clicks2.sort('uid', 'month', 'weekofyear', 'dayofweek' ))

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
silver_table_name2 = "silver_with_clicks_and_cost2"

(silver_clicks2
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(silver_table_name2)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM silver_with_clicks_and_cost

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM silver_with_clicks_and_cost2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     month, weekofyear, dayofweek, uid, first(channel) as channel, avg(product_stars), sum(impr), first(total_impr), sum(clicks), sum(txns), avg(cpm), avg(cpc), avg(gpt)
# MAGIC FROM
# MAGIC    silver_with_clicks_and_cost2
# MAGIC GROUP BY 
# MAGIC     month, weekofyear, dayofweek, uid
# MAGIC ORDER BY 
# MAGIC     uid

# COMMAND ----------

display(silver_clicks2.filter("clicks > impr"))
