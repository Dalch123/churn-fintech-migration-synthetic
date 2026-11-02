#!/usr/bin/env python3
"""Sanity checks + baseline model on the synthetic churn panel."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

d = sys.argv[1] if len(sys.argv) > 1 else "data"
df = pd.read_csv(f"{d}/churn_panel.csv.gz")
cust = pd.read_csv(f"{d}/customers.csv.gz")
print("rows", len(df), "cols", df.shape[1], "customers", df.customer_id.nunique())
print("duplicate (customer, month):", df.duplicated(["customer_id", "month_idx"]).sum())
print("customer churn rate:", round(cust.churned.mean(), 4))
print("exit reasons:\n", cust.exit_interview_reason.value_counts(normalize=True).round(3).to_string())
print("label prevalence (labelled rows):")
for h in (1, 3, 6):
    c = f"churn_next_{h}m"; print(f"  {c}: {df[c].mean():.4f}  (NaN rows: {df[c].isna().sum():,})")
print("\nmissingness (top):\n", df.isna().mean().sort_values(ascending=False).head(10).round(3).to_string())
print("\nrows per month_idx min/max:", df.groupby("month_idx").size().min(), df.groupby("month_idx").size().max())

# --- baseline model, time-based split
lab = "churn_next_3m"
ids = ["customer_id", "month", "suggested_split", "churn_next_1m", "churn_next_3m", "churn_next_6m"]
cat = ["age_band", "segment", "location", "occupation_industry", "salary_status", "declared_primary_bank"]
for c in cat:
    df[c] = df[c].astype("category")
feats = [c for c in df.columns if c not in ids]
speed_rel = ["latency_median_ms", "latency_p95_ms", "time_to_credit_sec_avg", "pct_txn_slower_than_10s", "otp_delay_sec_avg",
             "failed_txn_count", "failure_rate", "reversal_count", "timeout_count", "retry_count", "debit_no_value_count",
             "otp_failure_count", "failed_logins", "app_crash_count", "session_abandon_rate", "outage_minutes_exposed",
             "n_outage_incidents_exposed", "failure_rate_roll3", "latency_p95_roll3", "outage_minutes_roll3",
             "own_bank_success_rate_pct", "own_bank_median_latency_ms", "speed_gap_ms", "complaints_count",
             "complaints_roll3", "unresolved_complaints", "support_contacts", "complaint_resolution_hours_avg"]
tr = df[df.suggested_split == "train"]; te = df[df.suggested_split == "test"]


def fit(cols, name):
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_leaf_nodes=31, categorical_features="from_dtype",
                                       random_state=0)
    m.fit(tr[cols], tr[lab].astype(int))
    p = m.predict_proba(te[cols])[:, 1]; y = te[lab].astype(int)
    print(f"{name:42s} AUC={roc_auc_score(y, p):.3f}  PR-AUC={average_precision_score(y, p):.3f}  (base rate {y.mean():.3f})")
    return m


print("\nBaseline model (train months 0-14, test months 18-20, target = churn in next 3 months)")
fit(feats, "all features")
fit([c for c in feats if c not in speed_rel], "without speed/reliability/experience")
fit([c for c in feats if c not in ["fintech_outflow_amount", "fintech_outflow_share", "fintech_outflow_count",
                                   "fintech_outflow_share_roll3", "fintech_outflow_share_chg3", "external_fintech_balance",
                                   "external_inflow", "third_party_wallet_spend_share", "merchant_settlement_share_other_psp"]],
    "without fintech-flow leading indicators")

# --- hypothesis check: churn rate by failure-rate quintile, controlling for volume band (the confounder)
x = df[(df.suggested_split != "unlabelled_tail")].copy()
x["vol_band"] = pd.qcut(x.txn_count_roll3, 3, labels=["low", "mid", "high"], duplicates="drop")
x["fail_q"] = pd.qcut(x.failure_rate_roll3, 5, labels=False, duplicates="drop")
print("\n3m churn rate by failure-rate quintile within volume band:")
print(x.pivot_table(index="fail_q", columns="vol_band", values=lab, aggfunc="mean", observed=True).round(4).to_string())
print("\nraw failed_txn_count vs volume corr:", round(df[["failed_txn_count", "txn_count"]].corr().iloc[0, 1], 3))
