import dlt
from pyspark.sql import functions as F

# Read the scored transactions produced by the silver pipeline (separate pipeline -> spark.read.table)
SCORED = "workspace.silver.silver_transactions_scored"

# ---- Flagged (risky) transactions: feeds the AI assistant ----
@dlt.table(name="gold_flagged_transactions",
           comment="Risky transactions (Medium/High) with reasons.")
def gold_flagged_transactions():
    return (spark.read.table(SCORED)
            .filter("risk_level <> 'Low'")                     # keep only risky ones
            .select("transaction_id", "account_id", "merchant_id", "amount_usd",
                    "risk_level", "risk_score", "risk_reason", "transaction_date"))

# ---- Executive KPI summary: feeds the dashboard counters ----
@dlt.table(name="gold_fraud_kpis", comment="One-row executive fraud KPI summary.")
def gold_fraud_kpis():
    s = spark.read.table(SCORED)
    return s.agg(
        F.count("*").alias("total_transactions"),                                              # total txns
        F.sum((F.col("risk_level") != "Low").cast("int")).alias("flagged_transactions"),        # count flagged
        F.round(F.avg((F.col("risk_level") != "Low").cast("int")) * 100, 2).alias("flagged_pct"), # % flagged
        F.round(F.sum(F.when(F.col("risk_level") != "Low", F.col("amount_usd")).otherwise(0)), 2)  # $ at risk
            .alias("amount_at_risk_usd"))

# ---- Risk by merchant category: feeds a bar chart ----
@dlt.table(name="gold_risk_by_category", comment="Flagged rate per merchant category.")
def gold_risk_by_category():
    s = spark.read.table(SCORED)
    return (s.groupBy("category").agg(
                F.count("*").alias("txn_count"),
                F.sum((F.col("risk_level") != "Low").cast("int")).alias("flagged"),
                F.round(F.avg((F.col("risk_level") != "Low").cast("int")) * 100, 2).alias("flagged_pct"))
             .orderBy(F.desc("flagged_pct")))

# ---- Daily risk trend: feeds a line chart ----
@dlt.table(name="gold_daily_risk_trend", comment="Transactions and flagged count per day.")
def gold_daily_risk_trend():
    s = spark.read.table(SCORED)
    return (s.withColumn("day", F.to_date("transaction_date"))
             .groupBy("day").agg(
                F.count("*").alias("txn_count"),
                F.sum((F.col("risk_level") != "Low").cast("int")).alias("flagged"))
             .orderBy("day"))

# ---- Risk level breakdown: feeds a pie chart ----
@dlt.table(name="gold_risk_level_breakdown", comment="Transaction count by risk level.")
def gold_risk_level_breakdown():
    s = spark.read.table(SCORED)
    return s.groupBy("risk_level").agg(F.count("*").alias("txn_count"))