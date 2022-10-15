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
# MAGIC * As of this X-Challenge version the original Databricks data is still required. The original Databricks notebooks are used to create the BRONZE and SILVER raw data tables

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/ad_spend_flow_06.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ad Spend Optimization Context
# MAGIC 
# MAGIC Modern digital ad solutions help you engage potentially millions of customers.
# MAGIC 
# MAGIC According to some statistics:
# MAGIC 
# MAGIC    * millions if not billions of people log in into their favorite social app everyday (for example, facebook reports mind blowing 1.8 billion active user Q4 of 2021)
# MAGIC    * millions if not billions of products are accessible to a user on the palm of their hands (eBay reports 1.6 billion active products listings)
# MAGIC    
# MAGIC    * there are no official figures some sources estimate the average person encounters between 100 to 300 online ads every day.
# MAGIC 
# MAGIC 
# MAGIC With this sheer amount of options, the name of the game is to attract high quality users as cost effectively as possible, and help your business connect with customers at every stage of their journey.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is Mathematical Optimization?
# MAGIC 
# MAGIC Mathematical optimization is a branch of applied mathematics that has applications in many different fields. From Manufacturing, to Inventory Control, to Engineering.
# MAGIC 
# MAGIC * A basic optimization process consist of:
# MAGIC 
# MAGIC     * an objection function f(x) 
# MAGIC     * a vector x of decision variables 
# MAGIC     * a set of contraints Ci
# MAGIC     
# MAGIC     
# MAGIC 
# MAGIC In classical operations research and optimization literature, objective function tipically reflects the **total cost** or **total profit** 
# MAGIC 
# MAGIC Constraints are logical or mathematical conditions to a solution, which an optimization problem has to satisify to be a candidate to a valid solution.
# MAGIC * Usually, reflect real-world constraints or limits such as stock of raw materials, quantities, or budget available, among others.

# COMMAND ----------

# MAGIC %md
# MAGIC ### PuLP 
# MAGIC 
# MAGIC 
# MAGIC * PuLP is a python library which can be used to solve linear programming problems.
# MAGIC 
# MAGIC * PuLP is a high level API, similar to other high level APIs like Keras, which tries to present consistent and simple methods, which user may already being familiar, generating in the backend input files for various LP solvers.
# MAGIC 
# MAGIC PROs: relatively intuitive and easy to use (notation similar to how classic mathematical problems are solved) 
# MAGIC 
# MAGIC CONs: not easy to integrate with other linear algebra engines (for example: scipy, or numpy) No distributed version for Spark as far as we know.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Optimization Process 
# MAGIC 
# MAGIC Seven steps, according to https://coin-or.github.io/pulp/:
# MAGIC 
# MAGIC 1. Analyze the problem - Identify key Decision Variables or "features".
# MAGIC 
# MAGIC 2. Formulate "The Problem" -  Capture interdependencies in a mathematical model
# MAGIC 
# MAGIC 3. Formulate the Objective Function
# MAGIC 4. Formulate the Constraints
# MAGIC 5. Solve the optimization Problem
# MAGIC 6. Perform some post-optimal analysis and validations
# MAGIC 7. Present the solution and analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Solving a mathematical optimization problems means to find a valid mathematical solution that miminize (or maximize) the objective function given the constrants. 
# MAGIC 
# MAGIC * Traditionally:
# MAGIC    * Assuming the objective function is differentiable.  
# MAGIC    * Using classical and modern optimization algorithms such as the Revised Simplex Method, Interior Point Methods, Stochastic Gradient Descent, or Lagrange Multipliers.

# COMMAND ----------

# DBTITLE 1,Setup Environment
# MAGIC %pip install PuLP

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

# DBTITLE 1,Import Python Libraries
import pulp 
import numpy as np
import pandas as pd

from pyspark.sql.functions import lit, when, col

# COMMAND ----------

# DBTITLE 1,Optimization Problem Definition
# MAGIC %md
# MAGIC * Goal: Maximize Profitable Revenue and Gross Margin in advertising spend
# MAGIC * Subject to Operational, Spend and ROI constraints
# MAGIC 
# MAGIC 
# MAGIC * Key Decision variables:
# MAGIC   * User Behavior (for example: Risk Aversion)
# MAGIC   * Marketing Campaign Targets = Boost in relevant target audience, clicks or sales conversions.
# MAGIC   * Mean and Variance of Return of Investments or Rewards per user segment.
# MAGIC   * Marketing Contribution per Channel
# MAGIC   * Sales Gross Margin = Revenue - Spend
# MAGIC   * Spend  = Cost_per_Mille_Impressions * Number_of_Impressions + Cost_per_Click * Number_of_Clicks
# MAGIC   * Revenue = Revenue_per_Transaction * Number_of_Transactions
# MAGIC   * Number_of_Sales_Transactions = Number_of_Clicks * Coversion_Rate
# MAGIC   * Number_of Clicks = Number_of_Impressions * Click_Through_Rate
# MAGIC   * ROAS = Revenue / Spend
# MAGIC   
# MAGIC   
# MAGIC ROI or ROAS?
# MAGIC 
# MAGIC "ROI vs. ROAS is not an either/or decision. ROIs are best for long-term profitability, and ROAS might be more helpful in optimizing for short-term or very specific strategies" https://www.adjust.com/glossary/roas-definition/
# MAGIC ROAS is the amount of revenue that is earned for every dollar spent on a campaign.
# MAGIC ROI can be applied high-level to measure overall profits, ROAS will help you determine how much a campaign is contributing to those overall profits.
# MAGIC ROAS is used to estimate the profit potential of a marketing campaign (on a similar way than EBITDA can be used to estimate the profit making ability of a company)
# MAGIC ROAS is not ROI in the sense that it does not include any other operating expenses. Therefore, many times advertising ROAS is regarded as "marketing contribution", which is a revenue or gross margin that ignores all costs except for marketing costs and advertising spend.

# COMMAND ----------

# define the problem
prob = pulp.LpProblem("optimize_ad_spend", pulp.LpMaximize)

# COMMAND ----------

# DBTITLE 1,Objective Function
# MAGIC %md
# MAGIC Linearized utility function:
# MAGIC 
# MAGIC 
# MAGIC $$U(w) = (1-alpha) predictedrevenue + (alpha) [ -beta * C_1 std(predictedvisits) - C_2std(predictedclicks) + C_3std(predictedtxns)+ epsilon]$$ 
# MAGIC 
# MAGIC 
# MAGIC Where:
# MAGIC 
# MAGIC $$ alpha $$ 
# MAGIC 
# MAGIC - Is an **exploration** vs **exploitation** factor 
# MAGIC     - alpha of zero will be very conservative and ensure an optimal ROAS based on **previous customers** demand and orders
# MAGIC     - alpha of one would result in semi-random search of **new customers** - subject only to the user showing interest in our site or brand, while maximizing marketing budget allocation and target ROAS.
# MAGIC     
# MAGIC 
# MAGIC $$C_1, C_2 and C_3$$
# MAGIC 
# MAGIC * are scaling constants (so that all the units are in marketing contribution e.g. dollars
# MAGIC 
# MAGIC $$ epsilon $$ 
# MAGIC 
# MAGIC * is a small value provided for numerical stability and to avoid vanishing returns issue when there are too few visits, clicks, or transactions

# COMMAND ----------

# MAGIC %md
# MAGIC How a linear optimizer if we have a potentialy non-linear objective function?
# MAGIC 
# MAGIC * For Proof of concept and demostration purposes this ad spend optimizer uses a first order Taylor approximation of a Quadratic utility function.
# MAGIC * This is not uncommon or unheard: Many fields use linear programming techniques to make their processes more efficient.
# MAGIC * Linear programming requires linear equations, and despite this limitations of being forced to model only linear equations, classic linear programming has proven to provide most economical solution in multiple real-world applications.
# MAGIC * One advantage of linear optimization formulation is the ease to explain to a business user and the simplicity in adding linear parameters that a business user can understand and control

# COMMAND ----------

# MAGIC %md 
# MAGIC Ways to improve ROAS:
# MAGIC 1. maintain or increase the price of the goods sold, considering the elasticity demand, sometimes called "price optimization"
# MAGIC 2. keep the prices fixed and increase the volume of transactions
# MAGIC 3. offering discounts can drive volumen or re-direct ROAS 

# COMMAND ----------

# DBTITLE 1,Load Predictive Models Scores
pred_data = (spark.sql("SELECT * FROM score_data_with_pred")
             .withColumn("cvr_pred2", when( col('clicks_pred') > 0, col('txns_pred') / col('clicks_pred') ).otherwise(0)  )
            )

print(pred_data.agg({"user_rank": "max"}).collect()[0])
display(pred_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Quadratic Utility Funciton modeling
# MAGIC 
# MAGIC * In retail: “Utility is the happiness a person gets by consuming goods and services”
# MAGIC 
# MAGIC * In portfolio management: 
# MAGIC   * Utility Function can be used both to score product bundles and to determine Optimal Portfolio and Assortment Planning
# MAGIC   * degrees of risk aversion are defined by the additional marginal return an investor needs to accept more risk
# MAGIC   * the additional marginal return is calculated as the standard deviation of the return on investment (ROI)
# MAGIC 
# MAGIC * Quadratic Utility function:
# MAGIC   Expected Return on investment - standard deviation^2

# COMMAND ----------

# DBTITLE 1,Risk Seeker Score (opposite of risk aversion)
prob_to_browse = spark.sql(
"""
SELECT VISIT_RANK2, sum(prob_to_visit) logit_to_visit
FROM 
(
select A.uid, B.VISIT_RANK2, A.prob_to_browse prob_to_visit
from user_prob_table A
LEFT JOIN
(
    select * from 
    user_visit_rankings
) B 
ON A.uid = B.uid
)
group by VISIT_RANK2
--order by VISIT_RANK2
"""
)

prob_to_browse.createOrReplaceTempView("logits")

total_logits_df = spark.sql("select sum(logit_to_visit) total_logits, avg(logit_to_visit) avg_logits from logits")

prob_to_browse2 = (prob_to_browse.join(total_logits_df, how = "left")
                     .withColumn("prob_visit_score", col("logit_to_visit") / col("avg_logits"))
                  )

#prob_to_browse2.createOrReplaceTempView("logits2")

user_risk_aversion_score = spark.sql("""
SELECT * from user_risk_aversion
-- order by transitions
"""
)
risk_cols = ['VISIT_RANK2', 'transitions', 'max_risk_score' , 'prob_visit_score'] # 'std_prob', 'std_prob2',  'risk_score']
user_risk_aversion_score2 = (user_risk_aversion_score.join(prob_to_browse2, on = ["VISIT_RANK2"], how = 'left').select(risk_cols)
                            .withColumn("risk_seeker_score", col("max_risk_score") * col('prob_visit_score'))
                             #.withColumnRenamed('VISIT_RANK2', 'user_rank')
                            )


user_risk_aversion_score2.createOrReplaceTempView("logits3")

user_risk_aversion_score3 = spark.sql(
"""
select user_rank, avg(A.transitions) avg_transitions, std(A.transitions) std_transitions,
max(risk_seeker_score) max_risk_seeker_sore, avg(risk_seeker_score) avg_risk_seeker_score
from
(
select NTILE(10) OVER (ORDER BY VISIT_RANK2) as user_rank, * from logits3
) A
group by user_rank
"""
)

display(user_risk_aversion_score3)

# COMMAND ----------

pred_data2 = pred_data.join(user_risk_aversion_score3, on = ['user_rank'], how = 'left')
display(pred_data2)

# COMMAND ----------

# DBTITLE 1,Convert to Pandas
#TODO. Find a distributed version 
preds = pred_data2.toPandas()
print(preds.shape)
preds.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Core Predictions
# MAGIC * Risk Aversion
# MAGIC * Impressions per client per channel
# MAGIC * CTR per client per channel
# MAGIC * CVR per client per channel
# MAGIC * Txns per client
# MAGIC * Cost per Impressions per channel
# MAGIC * Cost per click per client

# COMMAND ----------

# DBTITLE 1,Load Optimization Parameters
#impr = pred_impr
clicks = preds.impr_pred * preds.ctr_pred
#risk aversion
risk_seeker = preds.avg_risk_seeker_score
#clicks.head()
txns = clicks * preds.cvr_pred
#print(txns.head())
revenue = txns * preds.gpt_pred
#revenue.head()
spend = (preds.impr_pred * preds.cpi_pred) + (clicks * preds.cpc_pred)
#spend.head()
std_visits = preds.std_transitions

# COMMAND ----------

# DBTITLE 1,Scale constants
#C1
total_revenue = sum(revenue)
total_transitions = sum(preds.impr_pred)
c1 = total_revenue / total_transitions
print(total_revenue, total_transitions, c1)
#C2
total_revenue = sum(revenue)
total_clicks = sum(clicks)
c2 = total_revenue / total_clicks
print(total_revenue, total_clicks, c2)
#C3
total_revenue = sum(revenue)
total_txns = sum(txns)
c3 = total_revenue / total_txns
print(total_revenue, total_clicks, c3)

# COMMAND ----------

# DBTITLE 1,Define PuLP variables
row_ids = list(preds.index)
print( min(row_ids), max(row_ids))
bid_units = pulp.LpVariable.dicts('bid_unit', row_ids, lowBound =0, upBound =1, cat = "Continuous")
print(len(bid_units))
print(bid_units[0])

# COMMAND ----------

# MAGIC %md
# MAGIC How can we use ROAS as a linear constraint?
# MAGIC 
# MAGIC * Linear Programming cannot have divisions in its constraints.
# MAGIC * therefore we reformulate the ROAS = Revenue / Spend to Revenue = ROAS * Spend
# MAGIC * or 1/ROAS = Spend / Revenue  => Spend = 1/ROAS * Revenue
# MAGIC * If we assume Revenue is the predicted or budgeted revenue (assumed fixed afer a prediction is made)
# MAGIC * Then, our ROAS selection/optimization will choose an upper bound for spend < upper_bound
# MAGIC * Above that upper bound of spend and our ROAS constraint will be broken
# MAGIC 
# MAGIC https://stackoverflow.com/questions/33929353/how-to-use-a-variable-as-a-divisor-in-pulp

# COMMAND ----------

revenues = [ ((revenue[i])) * bid_units[i]  for i in row_ids]
prob+= pulp.lpSum(revenues)
prob

# COMMAND ----------

# DBTITLE 1,Define PuLP Constraints
# Define constraints
# ROAS = Revenue / Advertising Costs 
# Efficiency = Spend / Revenue = 1/ROAS

target_ROAS = 0.7
target_eff = 1/target_ROAS
print(target_ROAS)
print(target_eff)
contributions = [(spend[i] - target_eff * revenue[i]) * bid_units[i] for i in row_ids]
#print(contributions)
prob+= pulp.lpSum(contributions) <= 0
prob

# COMMAND ----------

prob.solve()
print("Status:", pulp.LpStatus[prob.status])
print(f"objective: {prob.objective.value()}")
for var in prob.variables():
     print(f"{var.name}: {var.value()}")

# COMMAND ----------

type(prob.variables())

# COMMAND ----------

preds["opt_result"] = [round(var.value()) for var in prob.variables()]
preds_df = spark.createDataFrame(preds)
preds_df.createOrReplaceTempView('opt_results_table')
display(preds_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT channel,  opt_result,
# MAGIC       sum(impr_pred) as impr,
# MAGIC       sum(clicks_pred) as clicks,
# MAGIC       sum(txns_pred) as txns,
# MAGIC       avg(cpi_pred) as cpm,
# MAGIC       avg(cpc_pred) as cpc,
# MAGIC       avg(gpt_pred) as gpt,
# MAGIC       avg(ctr_pred) as ctr,
# MAGIC       avg(cvr_pred) as cvr
# MAGIC FROM opt_results_table
# MAGIC group by channel, opt_result
# MAGIC order by channel, opt_result

# COMMAND ----------

# DBTITLE 1,Define PuLP Objective Function
revenues = [ ((revenue[i]) + risk_seeker[i] * c1 * std_visits[i]) * bid_units[i]  for i in row_ids]
prob+= pulp.lpSum(revenues)
prob

# COMMAND ----------

# DBTITLE 1,Solve Optimization to Optimize Ad Spend
prob.solve()
print("Status:", pulp.LpStatus[prob.status])
print(f"objective: {prob.objective.value()}")
for var in prob.variables():
     print(f"{var.name}: {var.value()}")

# COMMAND ----------


