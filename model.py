"""Who buys in the next 15 days? Customer-level propensity, trained and tested on different time windows.

Train: behaviour 1-31 Oct  -> label: any purchase 1-15 Nov
Test:  behaviour 16 Oct-15 Nov -> label: any purchase 16-30 Nov   (includes the mid-November sale)
-> results/model.json
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
HERE = Path(__file__).parent
EVENTS = str(HERE / "data" / "2019-*.parquet")
NEG_KEEP = 0.25

NUMERIC = ["n_events", "n_views", "n_carts", "n_purchases", "n_sessions", "active_days", "recency_days", "tenure_days",
           "n_products", "n_categories", "avg_view_price", "max_cart_price", "spend", "events_7d", "carts_7d",
           "views_1d", "electronics_share", "cart_to_view"]
CATEGORICAL = ["top_category"]


def window(con, start: str, end: str, label_end: str) -> pd.DataFrame:
    return con.execute(f"""
      WITH w AS (SELECT * FROM e WHERE event_time >= TIMESTAMP '{start}' AND event_time < TIMESTAMP '{end}'),
      feat AS (
        SELECT user_id,
          count(*) AS n_events,
          count(*) FILTER (WHERE event_type = 'view') AS n_views,
          count(*) FILTER (WHERE event_type = 'cart') AS n_carts,
          count(*) FILTER (WHERE event_type = 'purchase') AS n_purchases,
          count(DISTINCT user_session) AS n_sessions,
          count(DISTINCT CAST(event_time AS DATE)) AS active_days,
          date_diff('minute', max(event_time), TIMESTAMP '{end}') / 1440.0 AS recency_days,
          date_diff('minute', min(event_time), TIMESTAMP '{end}') / 1440.0 AS tenure_days,
          count(DISTINCT product_id) AS n_products,
          count(DISTINCT category) AS n_categories,
          avg(price) FILTER (WHERE event_type = 'view') AS avg_view_price,
          max(price) FILTER (WHERE event_type = 'cart') AS max_cart_price,
          coalesce(sum(price) FILTER (WHERE event_type = 'purchase'), 0) AS spend,
          count(*) FILTER (WHERE event_time >= TIMESTAMP '{end}' - INTERVAL 7 DAY) AS events_7d,
          count(*) FILTER (WHERE event_type = 'cart' AND event_time >= TIMESTAMP '{end}' - INTERVAL 7 DAY) AS carts_7d,
          count(*) FILTER (WHERE event_type = 'view' AND event_time >= TIMESTAMP '{end}' - INTERVAL 1 DAY) AS views_1d,
          avg((category = 'electronics')::INT) AS electronics_share,
          mode(category) AS top_category
        FROM w GROUP BY user_id),
      lab AS (SELECT DISTINCT user_id FROM e WHERE event_type = 'purchase'
              AND event_time >= TIMESTAMP '{end}' AND event_time < TIMESTAMP '{label_end}')
      SELECT feat.*, feat.n_carts * 1.0 / greatest(feat.n_views, 1) AS cart_to_view,
             (lab.user_id IS NOT NULL)::INT AS y
      FROM feat LEFT JOIN lab USING (user_id)""").df()


def capture(y, score, share):
    k = int(share * len(y))
    top = np.argsort(-score, kind="stable")[:k]
    return {"share_of_users": share, "buyers_captured": round(float(y[top].sum() / y.sum()), 4),
            "precision": round(float(y[top].mean()), 4)}


def main():
    con = duckdb.connect()
    con.execute(f"SET memory_limit='4GB'; SET threads=4; SET temp_directory='{HERE / 'data' / 'tmp'}'")
    con.execute(f"CREATE VIEW e AS SELECT * FROM read_parquet('{EVENTS}')")
    train = window(con, "2019-10-01", "2019-11-01", "2019-11-16")
    test = window(con, "2019-10-16", "2019-11-16", "2019-12-01")
    print("train users", len(train), "buy rate", train.y.mean(), "| test users", len(test), "buy rate", test.y.mean())

    rng = np.random.default_rng(SEED)
    keep = (train["y"] == 1) | (rng.random(len(train)) < NEG_KEEP)
    tr = train[keep]
    weights = np.where(tr["y"] == 1, 1.0, 1 / NEG_KEEP)

    cols = NUMERIC + CATEGORICAL
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
             CATEGORICAL)])),
        ("m", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
            categorical_features=[False] * len(NUMERIC) + [True], early_stopping=True, validation_fraction=0.1,
            random_state=SEED))])
    model.fit(tr[cols], tr["y"], m__sample_weight=weights)

    y = test["y"].to_numpy()
    scores = {
        "Gradient boosting": model.predict_proba(test[cols])[:, 1],
        "Rule: carted in the last 7 days": test["carts_7d"].to_numpy() + test["events_7d"].to_numpy() * 1e-6,
        "Rule: bought before": test["n_purchases"].to_numpy() - test["recency_days"].to_numpy() * 1e-6,
        "Rule: most recently active": -test["recency_days"].to_numpy(),
    }
    comparison = []
    for name, s in scores.items():
        comparison.append({"method": name, "auc": round(float(roc_auc_score(y, s)), 4),
                           "pr_auc": round(float(average_precision_score(y, s)), 4),
                           "top": [capture(y, s, q) for q in (0.01, 0.05, 0.1, 0.2)]})
        print(comparison[-1]["method"], comparison[-1]["auc"], comparison[-1]["top"][1])

    p = scores["Gradient boosting"]
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[order])
    gains = [[g, round(float(cum[max(int(g * len(y) / 100), 1) - 1] / y.sum()), 4)] for g in range(1, 101)]
    rule = scores["Rule: carted in the last 7 days"]
    order_r = np.argsort(-rule, kind="stable")
    cum_r = np.cumsum(y[order_r])
    gains_rule = [[g, round(float(cum_r[max(int(g * len(y) / 100), 1) - 1] / y.sum()), 4)] for g in range(1, 101)]

    sample = test.sample(150000, random_state=SEED)
    imp = permutation_importance(model, sample[cols], sample["y"], scoring="roc_auc", n_repeats=3,
                                 random_state=SEED, n_jobs=2)
    importances = sorted([{"feature": f, "auc_drop": round(float(v), 4)} for f, v in zip(cols, imp.importances_mean)],
                         key=lambda d: -d["auc_drop"])[:10]

    out = {"train_users": int(len(train)), "train_buy_rate": round(float(train.y.mean()), 4),
           "test_users": int(len(test)), "test_buy_rate": round(float(y.mean()), 4), "test_buyers": int(y.sum()),
           "comparison": comparison, "gains": gains, "gains_rule": gains_rule, "importances": importances}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "model.json").write_text(json.dumps(out, indent=1))
    print("wrote results/model.json")


if __name__ == "__main__":
    main()
