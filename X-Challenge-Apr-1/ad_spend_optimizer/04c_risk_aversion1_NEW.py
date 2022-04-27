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
# MAGIC * This notebook re-uses the Markov Chain model transition probability matrix created in original Databricks Accelerator
# MAGIC 
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://cme-solution-accelerators-images.s3.us-west-2.amazonaws.com/multi-touch-attribution/mta-dag-1.png"; width="60%">
# MAGIC </div>
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/files/ad_spend_flow_04.png)

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC ### In this notebook you:
# MAGIC * Review how markov chain attribution models work
# MAGIC * Construct a transition probability matrix
# MAGIC * Calculate total conversion probability
# MAGIC * Use the removal effect to calculate attribution
# MAGIC * Compare channel performance across methods

# COMMAND ----------

# MAGIC %md
# MAGIC ### Intro to Multi-Touch Attribution with Markov Chains

# COMMAND ----------

# MAGIC %md
# MAGIC **Overview**
# MAGIC * Heuristic-based attribution methods like first-touch, last-touch, and linear are relatively easy to implement but are less accurate than data-driven methods. With marketing dollars at stake, data-driven methods are highly recommended.
# MAGIC 
# MAGIC * There are three steps to take when using Markov Chains to calculate attribution:
# MAGIC   * Step 1: Construct a transition probablity matrix
# MAGIC   * Step 2: Calculate the total conversion probability
# MAGIC   * Step 3: Use the removal effect to calculate attribution
# MAGIC   
# MAGIC * As the name suggests, a transition probability matrix is a matrix that contains the probabilities associated with moving from one state to another state. This is calculated using the data from all available customer journeys. With this matrix in place, we can then easily calculate the total conversion probability, which represents, on average, the likelihood that a given user will experience a conversion event. Lastly, we use the total conversion probability as an input for calculating the removal effect for each channel. The way that the removal effect is calculated is best illustrated with an example.
# MAGIC 
# MAGIC **An Example**
# MAGIC 
# MAGIC In the image below, we have a transition probability graph that shows the probabilty of going from one state to another state. In the context of a customer journey, states can be non-terminal (viewing an impression on a given channel) or terminal (conversion, no conversion).
# MAGIC 
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://cme-solution-accelerators-images.s3.us-west-2.amazonaws.com/multi-touch-attribution/mta-dag-1.png"; width="60%">
# MAGIC </div>
# MAGIC 
# MAGIC This image, which is simply a visual representation of a transition probability matrix, can be used to calculate the total conversion probability. The total conversion probability can be calculated by summing the probability of every path that leads to a conversion. For example, in the image above, we have 5 paths that lead to conversion. The paths and conversion probabilities are: 
# MAGIC 
# MAGIC | Path | Conversion Probability |
# MAGIC |---|---|
# MAGIC | State --> Facebook --> Conversion| 0.2 x 0.8|
# MAGIC | Start --> Facebook --> Email --> Conversion | 0.2 x 0.2 x 0.1 | 
# MAGIC | Start --> Google Display / Search --> Conversion | 0.8 x 0.6 | 
# MAGIC | Start --> Google Display / Search --> Facebook / Social --> Conversion | 0.8 x 0.4 x 0.8 |
# MAGIC | Start --> Google Display / Search --> Facebook / Social -- Email --> Conversion | 0.8 x 0.4 x 0.2 x 0.1 |
# MAGIC 
# MAGIC Therefore, the total probability of conversion is `0.90`:
# MAGIC 
# MAGIC ```P(Conversion) = (0.2 X 0.8) + (0.2 X 0.2 X 0.1) + (0.8 X 0.6) + (0.8 X 0.4 X 0.8) + (0.8 X 0.4 X 0.2 X 0.1)  = 0.90```
# MAGIC 
# MAGIC Now, let's calculate the removal effect for one of our channels: Facebook/Social. For this, we will set the conversion for Facebook/Social to 0% and then recalculate the total conversion probabilty. Now we have `0.48`.
# MAGIC 
# MAGIC ```P(Conversion) = (0.2 X 0.0) + (0.2 X 0.0 X 0.1) + (0.8 X 0.6) + (0.8 X 0.4 X 0) +(0.8 X 0.4 X 0.0 X 0.1)  = 0.48```
# MAGIC 
# MAGIC 
# MAGIC <div style="text-align: left">
# MAGIC   <img src="https://cme-solution-accelerators-images.s3.us-west-2.amazonaws.com/multi-touch-attribution/mta-dag-2.png"; width="60%">
# MAGIC </div>
# MAGIC 
# MAGIC With these two probabilities, we can now calculate the removal effect for Facebook/Social. The removal effect can be calculated as the difference between the total conversion probability (with all channels) and the conversion probability when the conversion for Facebook/Social is set to 0%.
# MAGIC 
# MAGIC ```Removal Effect(Facebook/ Social media) = 0.90 - 0.48 = 0.42```
# MAGIC 
# MAGIC Similarly, we can calculate the removal effect for each of the other channels and calculate attribution accordingly.
# MAGIC 
# MAGIC An excellent visual explanation of Markov Chains is available in this [article](https://setosa.io/ev/markov-chains/).

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 1: Configure the Environment
# MAGIC 
# MAGIC In this step, we will:
# MAGIC 1. Import libraries
# MAGIC 2. Run the utils notebook to gain acces to the get_params function
# MAGIC 3. get_params and store the relevant values in variables
# MAGIC 4. Set the current database so that it doesn't need to be manually specified each time it's used

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.1: Import libraries

# COMMAND ----------

from pyspark.sql.types import StringType, ArrayType
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.2: Run the `utils` notebook to gain access to the function `get_params`
# MAGIC * `%run` is a magic command provided within Databricks that enables you to run notebooks from within other notebooks.
# MAGIC * `get_params` is a helper function that returns a few parameters used throughout this solution accelerator. Usage of these parameters will be explicit.

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.3: `get_params` and store values in variables
# MAGIC * Two of the parameters returned by `get_params` are used in this notebook. For convenience, we will store the values for these parameters in new variables. 
# MAGIC   * **project_directory:** the directory used to store the files created in this solution accelerator. The default value can be overridden in the notebook 99_config.
# MAGIC   * **database_name:** the name of the database created in notebook `02_load_data`. The default value can be overridden in the notebook `99_config`

# COMMAND ----------

params = get_params()
project_directory = params['project_directory']
database_name = params['database_name']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 1.4: Set the current database so that it doesn't need to be manually specified each time it's used.
# MAGIC * Please note that this is a completely optional step. An alternative approach would be to use the syntax `database_name`.`table_name` when querying the respective tables.

# COMMAND ----------

_ = spark.sql("use {}".format(database_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Construct the Transition Probability Matrix
# MAGIC 
# MAGIC As discussed above, the transition probability matrix contains the probablities associated with moving from one state to another state. This is calculated using the data from all customer journeys.
# MAGIC 
# MAGIC In this step, we will:
# MAGIC 1. Define a user-defined function (UDF), `get_transition_array`, that takes a customer journey and enumerates each of the corresponding channel transitions
# MAGIC 2. Register the `get_transition_array` udf as a Spark UDF so that it can be utilized in Spark SQL
# MAGIC 3. Use `get_transition_array` to enumerate all channel transitions in a customer's journey
# MAGIC 4. Construct the transition probability matrix
# MAGIC 5. Validate that the state transition probabilities are calculated correctly
# MAGIC 6. Display the transition probability matrix

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gold_user_journey

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Step 2.1: Define a user-defined function (UDF) that takes a customer journey and enumerates each of the corresponding channel transitions

# COMMAND ----------

 def get_transition_array(path):
  '''
    This function takes as input a user journey (string) where each state transition is marked by a >. 
    The output is an array that has an entry for each individual state transition.
  '''
  state_transition_array = path.split(">")
  initial_state = state_transition_array[0]
  
  state_transitions = []
  for state in state_transition_array[1:]:
    state_transitions.append(initial_state.strip()+' > '+state.strip())
    initial_state =  state
  
  return state_transitions

# COMMAND ----------

foo = get_transition_array(
"Start > Affiliates > Search Engine Marketing > Affiliates > Social Network > Social Network > Social Network > Social Network > Search Engine Marketing > Null")
for s in foo:
  print(s)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Step 2.2: Register the `get_transition_array` udf as a Spark UDF so that it can be utilized in Spark SQL
# MAGIC * Note: this is an optional step that enables cross-language support.

# COMMAND ----------

spark.udf.register("get_transition_array", get_transition_array, ArrayType(StringType()))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Step 2.3: Use the `get_transition_array` to enumerate all channel transitions in a customer's journey

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW markov_state_transitions_detail AS
# MAGIC SELECT uid, path,
# MAGIC   explode(get_transition_array(path)) as transition,
# MAGIC   1 AS cnt
# MAGIC FROM
# MAGIC   gold_user_journey

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from markov_state_transitions_detail

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Step 2.4: Construct the transition probability matrix

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW transition_matrix_detail AS
# MAGIC SELECT
# MAGIC   left_table.uid,
# MAGIC   left_table.start_state,
# MAGIC   left_table.end_state,
# MAGIC   left_table.total_transitions nstate_trainsitions,
# MAGIC   right_table.total_state_transitions_initiated_from_start_state,
# MAGIC   left_table.total_transitions / right_table.total_state_transitions_initiated_from_start_state AS transition_probability
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       uid, transition,
# MAGIC       sum(cnt) total_transitions,
# MAGIC       trim(SPLIT(transition, '>') [0]) start_state,
# MAGIC       trim(SPLIT(transition, '>') [1]) end_state
# MAGIC     FROM
# MAGIC       markov_state_transitions_detail
# MAGIC     GROUP BY
# MAGIC       uid,transition
# MAGIC     ORDER BY
# MAGIC       uid, transition
# MAGIC   ) left_table
# MAGIC   JOIN (
# MAGIC     SELECT
# MAGIC       a.uid,
# MAGIC       a.start_state,
# MAGIC       sum(a.cnt) total_state_transitions_initiated_from_start_state
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           uid,
# MAGIC           trim(SPLIT(transition, '>') [0]) start_state,
# MAGIC           cnt
# MAGIC         FROM
# MAGIC           markov_state_transitions_detail
# MAGIC       ) AS a
# MAGIC     GROUP BY
# MAGIC       uid, a.start_state
# MAGIC     ) right_table 
# MAGIC   ON  left_table.uid = right_table.uid AND
# MAGIC       left_table.start_state = right_table.start_state
# MAGIC ORDER BY
# MAGIC   end_state DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from transition_matrix_detail
# MAGIC order by uid, start_state, end_state

# COMMAND ----------

transitions_detail = spark.sql("""
SELECT *, start_state_probability * transition_probability end_state_probability FROM
(
SELECT C.*, C.nstart_states / C.total_transitions_per_user start_state_probability, C.conversion, D.total_nrows
FROM
(
SELECT *, sum(nstate_trainsitions) OVER (PARTITION BY uid, start_state) nstart_states,
     CASE WHEN end_state = 'Conversion' THEN 1 ELSE 0 END conversion
FROM
(
SELECT A.*, B.total_transitions_per_user from transition_matrix_detail A
left join (
select uid, 
       sum(nstate_trainsitions) total_transitions_per_user
from transition_matrix_detail
group by uid
) B
ON A.uid = B.uid
)
) C
left join (
  SELECT count(*) total_nrows
  from transition_matrix_detail
) D
)
order by uid, start_state
"""
)
transitions_detail.createOrReplaceTempView("transition_matrix_detail2")
display(transitions_detail)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from transition_matrix_detail2
# MAGIC where uid = '000016cf3fa24ce2a97d74a0f66057f4'

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct uid, start_state,
# MAGIC        total_state_transitions_initiated_from_start_state,
# MAGIC        total_transitions_per_user,
# MAGIC        start_state_probability,
# MAGIC        sum(conversion) conversions,
# MAGIC        sum(end_state_probability) to_browse_prob
# MAGIC FROM
# MAGIC transition_matrix_detail2
# MAGIC where end_state != 'Null'
# MAGIC group by 
# MAGIC        uid, start_state, 
# MAGIC        total_state_transitions_initiated_from_start_state,
# MAGIC        total_transitions_per_user,
# MAGIC        start_state_probability
# MAGIC 
# MAGIC order by uid, start_state

# COMMAND ----------

# DBTITLE 1,Probability of staying in the user journey
transition_details2 = spark.sql(
"""
select distinct uid, start_state,
       total_state_transitions_initiated_from_start_state,
       total_transitions_per_user,
       start_state_probability,
       sum(conversion) conversions,
       sum(end_state_probability) to_browse_prob
FROM
transition_matrix_detail2
where end_state != 'Null'
group by 
       uid, start_state, 
       total_state_transitions_initiated_from_start_state,
       total_transitions_per_user,
       start_state_probability

order by uid, start_state

"""
)
transition_details2.createOrReplaceTempView("transition_details2")
display(transition_details2)

# COMMAND ----------

transition_details2.columns

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select uid, start_state, 
# MAGIC sum(total_state_transitions_initiated_from_start_state) transitions,
# MAGIC sum(to_browse_prob) to_browse
# MAGIC from transition_details2
# MAGIC where start_state != 'Start'
# MAGIC group by uid, start_state
# MAGIC 
# MAGIC order by uid

# COMMAND ----------

user_prob_visit = spark.sql(
"""
SELECT uid, sum(to_browse) prob_to_browse
from 
(
select uid, start_state, 
sum(total_state_transitions_initiated_from_start_state) transitions,
sum(to_browse_prob) to_browse
from transition_details2
where start_state != 'Start'
group by uid, start_state
)
group by uid
"""
)
display(user_prob_visit)

# COMMAND ----------

user_prob_table = "user_prob_table"

(
  user_prob_visit
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(user_prob_table)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * 
# MAGIC from bronze_with_clicks_and_cost
# MAGIC where uid ='02a819c8fb4b4154b4caddab3bf9212c'

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
