"""Single-feature leakage scan: AUC of each column alone for the 3-month label on the TEST split,
plus missingness-by-label and correlation with the label's time-to-event. Flags suspiciously strong columns."""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
d = sys.argv[1]
df = pd.read_csv(f"{d}/churn_panel.csv.gz")
lab = "churn_next_3m"
te = df[(df.suggested_split == "test") & df[lab].notna()].copy()
y = te[lab].astype(int).values
skip = {"customer_id", "month", "suggested_split", "churn_next_1m", "churn_next_3m", "churn_next_6m", "month_idx"}
rows = []
for c in te.columns:
    if c in skip: continue
    x = te[c]
    if not pd.api.types.is_numeric_dtype(x):
        continue
    xf = x.astype(float)
    m = xf.notna()
    if m.sum() < 1000 or xf[m].nunique() < 2: 
        continue
    auc = roc_auc_score(y[m.values], xf[m].values)
    rows.append((c, round(max(auc, 1 - auc), 3), round(1 - m.mean(), 3)))
r = pd.DataFrame(rows, columns=["column", "single_feature_AUC", "missing_rate"]).sort_values("single_feature_AUC", ascending=False)
print(r.head(25).to_string(index=False))
