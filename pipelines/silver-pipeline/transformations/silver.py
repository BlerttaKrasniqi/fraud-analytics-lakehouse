import dlt
from pyspark.sql import functions as F

BRONZE = "workspace.bronze."
HIGH_AMOUNT = float(spark.conf.get("high_amount_threshold","10000"))

def bronze(t):
    return f"{BRONZE}bronze_{t}"

# warn rules

TXN_WARN = {
    "amount_positive": "amount IS NULL OR amount > 0",
    "date_not_future": "transaction_date IS NULL OR to_timestamp(transaction_date) <= current_timestamp()",
    "status_allowed":  "status IS NULL OR lower(status) IN ('completed','pending','failed')",
    "channel_allowed": "channel IS NULL OR lower(channel) IN ('online','pos','atm')",
}

@dlt.table(name="silver_transactions",comment="Cleansed & validated transactions",
           table_properties={"layer":"silver"})
@dlt.expect_all_or_drop({
    "txn_id_not_null":"transaction_id IS NOT NULL",
    "account_id_not_null":"account_id IS NOT NULL"
})

# keep but flag questionable rows
@dlt.expect_all(TXN_WARN)
def silver_transactions():
    df = spark.read.table(bronze("transactions"))
    for c,t in df.dtypes:
        if t=="string":
            df=df.withColumn(c,F.trim(F.col(c)))
    df = df.dropDuplicates(["transaction_id"])
    return df.withColumn("_loaded_at",F.current_timestamp())

# ---- Cleansed customers ----
@dlt.table(name="silver_customers", comment="Cleansed & validated customers.",
           table_properties={"layer": "silver"})
@dlt.expect_all_or_drop({"customer_id_not_null": "customer_id IS NOT NULL"})   # drop rows with no key
@dlt.expect_all({                                                              # warn-only quality checks
    "first_name_present": "first_name IS NOT NULL AND length(trim(first_name)) > 0",
    "email_valid": "email IS NULL OR email RLIKE '^[^@\\\\s]+@[^@\\\\s]+\\\\.[^@\\\\s]+$'",
})
def silver_customers():
    df = spark.read.table(bronze("customers"))          # read raw customers
    for c, t in df.dtypes:                              # trim all text columns
        if t == "string": df = df.withColumn(c, F.trim(F.col(c)))
    df = df.dropDuplicates(["customer_id"])            # one row per customer
    return df.withColumn("_loaded_at", F.current_timestamp())

# ---- Cleansed accounts ----
@dlt.table(name="silver_accounts", comment="Cleansed & validated accounts.",
           table_properties={"layer": "silver"})
@dlt.expect_all_or_drop({"account_id_not_null": "account_id IS NOT NULL"})
@dlt.expect_all({
    "account_status_allowed": "account_status IS NULL OR lower(account_status) IN ('active','frozen','closed')",
})
def silver_accounts():
    df = spark.read.table(bronze("accounts"))
    for c, t in df.dtypes:
        if t == "string": df = df.withColumn(c, F.trim(F.col(c)))
    df = df.dropDuplicates(["account_id"])
    return df.withColumn("_loaded_at", F.current_timestamp())

# ---- Cleansed merchants ----
@dlt.table(name="silver_merchants", comment="Cleansed & validated merchants.",
           table_properties={"layer": "silver"})
@dlt.expect_all_or_drop({"merchant_id_not_null": "merchant_id IS NOT NULL"})
@dlt.expect_all({
    "merchant_status_allowed": "merchant_status IS NULL OR lower(merchant_status) IN ('active','inactive')",
})
def silver_merchants():
    df = spark.read.table(bronze("merchants"))
    for c, t in df.dtypes:
        if t == "string": df = df.withColumn(c, F.trim(F.col(c)))
    df = df.dropDuplicates(["merchant_id"])
    return df.withColumn("_loaded_at", F.current_timestamp())

# ---- Risk-scored transactions ----
@dlt.table(name="silver_transactions_scored",
           comment="Transactions enriched with risk flags, score, level, and reason.",
           table_properties={"layer": "silver"})
def silver_transactions_scored():
    # base = the cleansed transactions from THIS pipeline (use dlt.read)
    t = dlt.read("silver_transactions")

    # reference data from bronze (outside this pipeline -> spark.read.table)
    a  = spark.read.table(bronze("accounts")).select("account_id", "home_country", "account_status")
    m  = spark.read.table(bronze("merchants")).select("merchant_id", "category", "merchant_status")
    fx = spark.read.table(bronze("fx_rates")).select("currency", "rate_to_usd")

    # join the reference data onto each transaction
    df = (t.join(a,  "account_id",  "left")
            .join(m,  "merchant_id", "left")
            .join(fx, "currency",    "left")
            # convert every amount to USD so amounts are comparable
            .withColumn("amount_usd",
                        F.round(F.col("amount") * F.coalesce(F.col("rate_to_usd"), F.lit(1.0)), 2)))

    # ----- RISK RULES (each produces a 0/1 flag) -----
    df = (df
        .withColumn("flag_high_amount",       (F.col("amount_usd") > F.lit(HIGH_AMOUNT)).cast("int"))                 # large payment
        .withColumn("flag_cross_border",      (F.col("country") != F.col("home_country")).cast("int"))               # txn country != account country
        .withColumn("flag_gambling",          (F.lower(F.col("category")) == "gambling").cast("int"))                # gambling merchant
        .withColumn("flag_inactive_merchant", (F.lower(F.col("merchant_status")) == "inactive").cast("int"))        # merchant not active
        .withColumn("flag_frozen_account",    (F.lower(F.col("account_status")) == "frozen").cast("int")))          # account frozen

    # risk_score = sum of the flags (0..5)
    df = df.withColumn("risk_score",
        F.col("flag_high_amount") + F.col("flag_cross_border") + F.col("flag_gambling") +
        F.col("flag_inactive_merchant") + F.col("flag_frozen_account"))

    # risk_level = bucket the score into Low / Medium / High
    df = df.withColumn("risk_level",
        F.when(F.col("risk_score") >= 3, "High")
         .when(F.col("risk_score") >= 1, "Medium")
         .otherwise("Low"))

    # risk_reason = human-readable list of which rules fired (for the chatbot)
    df = df.withColumn("risk_reason", F.concat_ws("; ",
        F.when(F.col("flag_high_amount") == 1,       F.lit("High amount")),
        F.when(F.col("flag_cross_border") == 1,      F.lit("Cross-border")),
        F.when(F.col("flag_gambling") == 1,          F.lit("Gambling merchant")),
        F.when(F.col("flag_inactive_merchant") == 1, F.lit("Inactive merchant")),
        F.when(F.col("flag_frozen_account") == 1,    F.lit("Frozen account"))))

    return df


