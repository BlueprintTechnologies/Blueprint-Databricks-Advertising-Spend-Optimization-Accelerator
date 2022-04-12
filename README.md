# Blueprint Accelerator: Advertising Spend Optimizer

## Overview
The goal of this accelerator is to save weeks or months of development time for your data engineers and data scientists by providing a platform to optimize Return on Advertising Spend (ROAS) leveraging Spark, Machine Leaning, and Databricks.

This accelerator uses the python optimization engine Coin|Or [PuLP](https://coin-or.github.io/pulp/) library to provide a business friendly optimization layer. This provides configurable objective functions that are aligned to Operations Research best practices. No mysterious black boxes.

This accelerator uses a linearized utility function, enabling users to set target ROAS and marketing contribution efficiency, and market exploration with 3 simple parameters. Within this accelerator, we provide a Spark Boosted Tree implementation as an example.

## Special Thanks
This module builds upon an existing Databricks accelerator called “[Measure Ad Effectiveness with Multi-touch Attribution Accelerator](https://databricks.com/solutions/accelerators/multi-touch-attribution)”. The original Databricks accelerator is based on best practices from Databricks with leading global brands. Marketers and ad agencies are being held responsible to demonstrate ROI of their marketing dollars and optimize marketing channel spend to drive sales. The original accelerator focused solely on attribution

*The original version of this accelerator was created as an entry for Blueprint Technologies' X-Challenge. X-Challenges are designed to provide challenges that enable Blueprinters, through technical experimentation & exploration, to participate in a broad range of strategic initiatives.*
