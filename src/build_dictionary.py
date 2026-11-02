#!/usr/bin/env python3
"""Builds data_dictionary.csv: every panel column with family, meaning, and the real-world source to request."""
import sys
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/churn_dataset/data"
cols = pd.read_csv(f"{d}/churn_panel.csv.gz", nrows=5).columns.tolist()

SW, CORE, APP, CRM, BUR, MKT, SVY, OB = ("Payments switch / core banking txn logs", "Core banking system", "Mobile & web app telemetry (Firebase/Crashlytics/analytics)",
                                         "CRM / call-centre / complaints system", "Credit bureau", "Market benchmark (status pages, interbank switch reports)",
                                         "Customer survey platform", "Open banking / account aggregation (consented)")
D = {}
def add(fam, src, items):
    for k, v in items.items():
        D[k] = (fam, v, src)

add("Key / time", "Data warehouse", {"customer_id": "Pseudonymised customer key", "month_idx": "Month index 0..23 (0 = Oct 2024)", "month": "Calendar month YYYY-MM",
    "suggested_split": "Leakage-safe time split: train m0-14, valid m15-17, test m18-20; m21-23 have no complete 3m label"})
add("Customer context", "Core banking / KYC", {"age_band": "Age band", "segment": "Commercial segment", "location": "Primary city/region",
    "occupation_industry": "Occupation / industry (KYC)", "salary_status": "Salary status: domiciled / not domiciled / self-employed / etc.",
    "is_sme_or_corporate": "1 if SME or Corporate", "tenure_months": "Months since account opening at this month"})
add("Transaction", SW, {"txn_count": "Debit transactions in month", "txn_value_total": "Total debit value, NGN", "txn_value_avg": "Average transaction value, NGN",
    "txn_active_days": "Days with at least one transaction", "txn_freq_per_active_day": "Transactions per active day",
    "transfer_share": "Share of txns that are transfers", "pos_card_share": "Share that are POS/card", "bills_airtime_share": "Share that are bills/airtime/data",
    "cash_wd_share": "Share that are cash withdrawals"})
add("Speed", SW, {"latency_median_ms": "Median request-to-response latency, ms", "latency_p95_ms": "95th percentile latency, ms",
    "time_to_credit_sec_avg": "Average time for beneficiary credit, seconds", "pct_txn_slower_than_10s": "Share of txns taking more than 10 s",
    "otp_delay_sec_avg": "Average OTP delivery delay, seconds"})
add("Reliability", SW, {"failed_txn_count": "Failed transactions (VOLUME-CONFOUNDED: heavy users fail more often)", "failure_rate": "failed / total txns",
    "reversal_count": "Auto-reversed transactions", "timeout_count": "Timeouts", "retry_count": "Retries after a failure (frustration signal)",
    "debit_no_value_count": "Debit-without-value cases", "otp_failure_count": "OTP failures", "failed_logins": "Failed app logins",
    "app_crash_count": "App crashes", "session_abandon_rate": "Share of app sessions abandoned",
    "outage_minutes_exposed": "Outage minutes the customer was exposed to (channel-share and location weighted)",
    "n_outage_incidents_exposed": "Incidents touching the customer's channels and location"})
D["failed_txn_count"] = ("Reliability", D["failed_txn_count"][1], SW)
add("Digital engagement", APP, {"app_logins": "App logins", "app_sessions": "App sessions", "ussd_sessions": "USSD sessions", "web_logins": "Web logins",
    "avg_session_duration_sec": "Average app session length, s", "days_since_last_login": "Days since last app login at month end",
    "app_share": "Share of txns via app", "ussd_share": "Share via USSD", "card_share": "Share via card/POS", "web_share": "Share via web",
    "branch_atm_share": "Share via branch/ATM"})
add("Financial behaviour", CORE, {"avg_balance": "Average daily balance, NGN", "end_balance": "Month-end balance, NGN", "balance_change_pct": "End balance vs prior month (NaN in first month)",
    "inflow_total": "Total credits, NGN", "outflow_total": "Total debits, NGN", "turnover_ratio": "(inflow+outflow) / (2 x avg balance)",
    "salary_inflow_flag": "1 if a domiciled salary credit arrived", "salary_amount": "Salary credit amount, NGN", "large_outflow_flag": "1 if a sweep-out above 30% of average balance occurred",
    "fintech_outflow_amount": "Debits to identified fintech counterparties, NGN (counterparty bank/fintech codes in switch logs)",
    "fintech_outflow_share": "fintech_outflow_amount / outflow_total", "fintech_outflow_count": "Number of transfers to fintech counterparties"})
add("External / ecosystem", "Payment ecosystem & merchant/processor data", {"third_party_wallet_spend_share": "Share of spend via third-party wallets/QR",
    "merchant_settlement_share_other_psp": "SME only: share of merchant settlements routed to another processor (NaN otherwise)"})
add("External / ecosystem", OB, {"external_fintech_balance": "Balance at external fintech (consenting customers only, else NaN)",
    "external_inflow": "Inflow to external accounts (consenting customers only, else NaN)", "open_banking_consent": "1 if customer consented to data sharing"})
add("Relationship", CORE, {"n_products": "Number of products held", "has_savings": "Savings product", "has_current": "Current account", "has_debit_card": "Debit card",
    "has_credit_card": "Credit card", "has_loan": "Active loan", "has_overdraft": "Overdraft", "has_fx_account": "FX account", "has_investment": "Investment product"})
add("External / ecosystem", BUR, {"bureau_n_other_bank_accounts": "Accounts at other institutions (quarterly refresh)", "bureau_active_lenders": "Active lenders (quarterly refresh)",
    "bureau_score": "Bureau score 300-850 (NaN for thin files)"})
add("Experience", CRM, {"complaints_count": "Complaints logged", "support_contacts": "All support contacts", "complaint_resolution_hours_avg": "Mean resolution time, hours (NaN if no complaint)",
    "unresolved_complaints": "Complaints unresolved at month end"})
add("Experience", SVY, {"csat_score": "CSAT 1-5, sparse (~4% respond)", "nps_score": "NPS 0-10, sparse (~3% respond)",
    "declared_primary_bank": "Self-declared 'is this your primary bank' (~20% answered)"})
add("Cost", CORE, {"fee_total": "Fees paid, NGN", "fee_per_txn_avg": "Average fee per transaction, NGN"})
add("Market / benchmark", MKT, {"own_bank_success_rate_pct": "Own-bank success rate that month, %", "own_bank_median_latency_ms": "Own-bank median latency, ms",
    "fintech_success_rate_pct": "Benchmark fintech success rate, %", "fintech_median_latency_ms": "Benchmark fintech median latency, ms",
    "speed_gap_ms": "Own minus fintech median latency", "fintech_promo_intensity": "Fintech promo intensity 0-1 (campaign tracking)",
    "fee_change_flag": "1 in months the bank changed fees", "n_major_outages_in_month": "Severity 3-4 incidents in month",
    "peer_fintech_adoption_rate_location": "Share of customers in same location with a fintech account (peer influence)"})
for nme, base in [("txn_count_roll3", "txn_count"), ("failure_rate_roll3", "failure_rate"), ("latency_p95_roll3", "latency_p95_ms"), ("app_logins_roll3", "app_logins"),
                  ("avg_balance_roll3", "avg_balance"), ("fintech_outflow_share_roll3", "fintech_outflow_share"), ("outage_minutes_roll3", "outage_minutes_exposed"),
                  ("complaints_roll3", "complaints_count")]:
    D[nme] = ("Rolling feature", f"Mean of {base} over current + previous 2 months (backward-looking only)", "Derived")
for nme, base in [("txn_count_trend3", "txn_count"), ("app_logins_trend3", "app_logins"), ("avg_balance_trend3", "avg_balance")]:
    D[nme] = ("Rolling feature", f"{base} vs mean of previous 3 months, as a ratio minus 1", "Derived")
D["fintech_outflow_share_chg3"] = ("Rolling feature", "fintech_outflow_share minus its mean over previous 3 months", "Derived")
for h in (1, 3, 6):
    D[f"churn_next_{h}m"] = ("LABEL", f"1 if the customer migrated in months t+1..t+{h}; NaN if the window is not fully observable (right-censored)", "Derived label")

miss = [c for c in cols if c not in D]; extra = [k for k in D if k not in cols]
assert not miss and not extra, (miss, extra)
out = pd.DataFrame([(c, *D[c]) for c in cols], columns=["column", "family", "description", "real_world_source"])
out.to_csv(f"{d}/data_dictionary.csv", index=False)
print(len(out), "columns documented"); print(out.family.value_counts().to_string())
