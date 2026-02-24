# Databricks notebook source
# DBTITLE 1,get overview of KPI names
# MAGIC %sql
# MAGIC SELECT Plant,Department,Shop, Area, KpiName
# MAGIC FROM `westeurope_mo360dp_data_prd`.`shopfloor_iportal_bbac_kpi_cycletimes_emea`.`iportal_bbac_kpi_cycletimes`
# MAGIC WHERE Department IN ('mra1_ass', 'mra2_ass', 'mfa_ass', 'shn')
# MAGIC   AND Shop = 'MO'
# MAGIC GROUP BY Plant, Department, Shop, Area, KpiName

# COMMAND ----------

# DBTITLE 1,summarize KPI cycle times by plant and department
# MAGIC %sql
# MAGIC SELECT any_value(Timestamp) as Timestamp, Plant, Department, Shop, Area, KpiName
# MAGIC FROM `westeurope_mo360dp_data_prd`.`shopfloor_iportal_bbac_kpi_cycletimes_emea`.`iportal_bbac_kpi_cycletimes`
# MAGIC GROUP BY Plant, Department, Shop, Area, KpiName
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,assess data quality by department and shop null counts
# Check data quality by counting nulls and total records grouped by department and shop
from pyspark.sql import functions as F
df = spark.table("westeurope_mo360dp_data_prd.shopfloor_iportal_bbac_kpi_cycletimes_emea.iportal_bbac_kpi_cycletimes")
cols = [c for c in df.columns if c not in ["department", "shop"]]

data_quality_df = df.groupBy("department", "shop").agg(
    F.count("*").alias("total_records"),
    *[F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls") for c in cols]
)

display(data_quality_df)

# COMMAND ----------

# DBTITLE 1,query downtime data for specific departments in MO shop
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM `westeurope_mo360dp_data_prd`.`shopfloor_iportal_bbac_downtimes_emea`.`iportal_downtimes_bbac`
# MAGIC WHERE Department IN ('mra1_ass', 'mra2_ass', 'mfa_ass', 'shn')
# MAGIC   AND Shop = 'MO'
# MAGIC ORDER BY Timestamp DESC
# MAGIC

# COMMAND ----------

# DBTITLE 1,extract 2025 cycle times for selected departments in MO
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM `westeurope_mo360dp_data_prd`.`shopfloor_iportal_bbac_kpi_cycletimes_emea`.`iportal_bbac_kpi_cycletimes`
# MAGIC WHERE Department IN ('mra1_ass', 'mra2_ass', 'mfa_ass', 'shn')
# MAGIC   AND Shop = 'MO'
# MAGIC   AND Date >= date_add(current_date(), -365)
# MAGIC   AND Date <= current_date()
# MAGIC   AND dayofweek(Date) NOT IN (1, 7) -- 1=Sunday, 7=Saturday
# MAGIC ORDER BY Timestamp DESC

# COMMAND ----------

# DBTITLE 1,analyze average adjusted cycle times for selected depar ...
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   Plant, 
# MAGIC   Department, 
# MAGIC   Shop, 
# MAGIC   Area, 
# MAGIC   KpiName, 
# MAGIC   ProductionDay,
# MAGIC   round(avg(CycleTimeAdjusted),2) as avg_adjusted_cycletime
# MAGIC FROM `westeurope_mo360dp_data_prd`.`shopfloor_iportal_bbac_kpi_cycletimes_emea`.`iportal_bbac_kpi_cycletimes`
# MAGIC WHERE Department IN ('mra1_ass', 'mra2_ass', 'mfa_ass', 'shn')
# MAGIC   AND Shop = 'MO'
# MAGIC   AND year(Date) = year(current_timestamp())
# MAGIC GROUP BY Plant, Department, Shop, Area, KpiName, ProductionDay
# MAGIC ORDER BY ProductionDay ASC

# COMMAND ----------

# DBTITLE 1,Cell 7
df = spark.table("westeurope_mo360dp_data_prd.shopfloor_iportal_bbac_kpi_cycletimes_emea.iportal_bbac_kpi_cycletimes")
for col in df.columns:
    display(df.select(col).distinct().alias(f"distinct_{col}"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM `westeurope_mo360dp_data_prd`.`shopfloor_iportal_bbac_downtimes_emea`.`iportal_downtimes_bbac`
# MAGIC WHERE Department = 'eb5'
# MAGIC ORDER BY Timestamp DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM 'westeurope_mo360dp_data_prd.shopfloor_iportal_bbac_kpi_cycletimes_emea'.iportal_bbac_kpi_cycletimes
# MAGIC WHERE Department = 'eb5'
# MAGIC ORDER BY Timestamp DESC
