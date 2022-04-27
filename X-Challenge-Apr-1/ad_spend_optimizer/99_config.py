# Databricks notebook source
#user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user') # user name associated with your account
#project_directory = '/home/{}/multi-touch-attribution'.format(user) # files will be written to this directory
#database_name = 'multi_touch_attribution' # tables will be stored in this database

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user') # user name associated with your account
folder_name = "ad-spend-optimizer"
project_directory = '/home/{}/x-challenge'.format(folder_name) # files will be written to this directory
database_name = 'ad_spend_optimizer' # tables will be stored in this database

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"project_directory":project_directory,"database_name":database_name}))

# COMMAND ----------

#display(dbutils.fs.ls("."))

# COMMAND ----------

#display(dbutils.fs.ls("/home/"))

# COMMAND ----------

#dbutils.fs.mkdirs("/home/ad-spend-optimizer/x-challenge/raw/")

# COMMAND ----------

#'/dbfs/home/ad-spend-optimizer/multi-touch-attribution2/raw/attribution_data.csv'
#dbutils.fs.mkdirs("/home/ad-spend-optimizer/multi-touch-attribution2/raw/")

# COMMAND ----------

#'dbfs:/home/ad-spend-optimizer/multi-touch-attribution2/raw'

# COMMAND ----------

#!ls /home

# COMMAND ----------

#!mkdir /home/ad-spend-optimizer

# COMMAND ----------

#!mkdir /home/ad-spend-optimizer/multi-touch-attribution2

# COMMAND ----------

#!ls /home/ad-spend-optimizer

# COMMAND ----------

!pwd

# COMMAND ----------

!ls /

# COMMAND ----------

!ls /dbfs

# COMMAND ----------


