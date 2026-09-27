"""Funnel, timing, pricing and retention analytics over all events with DuckDB -> results/analysis.json."""
import json
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
EVENTS = str(HERE / "data" / "2019-*.parquet")


def rows(con, sql):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [{c: (round(v, 6) if isinstance(v, float) else (str(v) if hasattr(v, "isoformat") else v))
             for c, v in zip(cols, r)} for r in cur.fetchall()]


def main():
    con = duckdb.connect()
    con.execute(f"SET memory_limit='4GB'; SET threads=4; SET temp_directory='{HERE / 'data' / 'tmp'}'")
    con.execute(f"CREATE VIEW e AS SELECT * FROM read_parquet('{EVENTS}')")
    out = {}

    out["overview"] = rows(con, """
      SELECT count(*) AS events, count(DISTINCT user_id) AS users, count(DISTINCT user_session) AS sessions,
             count(DISTINCT product_id) AS products, count(DISTINCT brand) AS brands,
             count(*) FILTER (WHERE event_type = 'view') AS views,
             count(*) FILTER (WHERE event_type = 'cart') AS carts,
             count(*) FILTER (WHERE event_type = 'purchase') AS purchases,
             sum(price) FILTER (WHERE event_type = 'purchase') AS revenue,
             min(event_time) AS first_event, max(event_time) AS last_event FROM e""")[0]
    print(out["overview"])

    out["daily"] = rows(con, """
      SELECT CAST(event_time AS DATE) AS day,
             count(*) FILTER (WHERE event_type = 'view') AS views,
             count(*) FILTER (WHERE event_type = 'purchase') AS purchases,
             sum(price) FILTER (WHERE event_type = 'purchase') AS revenue,
             count(DISTINCT user_id) AS active_users
      FROM e GROUP BY 1 ORDER BY 1""")

    # Session-level funnel: did the session view, cart, purchase at least once?
    con.execute("""
      CREATE TEMP TABLE s AS
      SELECT user_session, any_value(user_id) AS user_id,
             max(event_type = 'view')::INT AS viewed, max(event_type = 'cart')::INT AS carted,
             max(event_type = 'purchase')::INT AS purchased,
             arg_max(category, (event_type = 'purchase')::INT * 3 + (event_type = 'cart')::INT * 2 + 1) AS category
      FROM e WHERE user_session IS NOT NULL GROUP BY 1""")
    out["funnel"] = rows(con, """
      SELECT count(*) AS sessions, sum(viewed) AS with_view, sum(carted) AS with_cart, sum(purchased) AS with_purchase,
             sum(carted * (1 - purchased)) AS abandoned_carts,
             sum(carted * (1 - purchased)) * 1.0 / nullif(sum(carted), 0) AS cart_abandonment_rate
      FROM s""")[0]
    out["funnel_by_category"] = rows(con, """
      SELECT category, count(*) AS sessions, avg(carted) AS cart_rate, avg(purchased) AS purchase_rate,
             sum(purchased) * 1.0 / nullif(sum(carted), 0) AS cart_to_purchase
      FROM s GROUP BY 1 HAVING count(*) >= 100000 ORDER BY sessions DESC LIMIT 12""")

    out["hour_weekday"] = rows(con, """
      SELECT dayofweek(event_time) AS dow, hour(event_time) AS hour,
             count(*) FILTER (WHERE event_type = 'purchase') AS purchases,
             count(*) FILTER (WHERE event_type = 'purchase') * 1.0 / nullif(count(*) FILTER (WHERE event_type = 'view'), 0) AS purchases_per_view
      FROM e GROUP BY 1, 2 ORDER BY 1, 2""")

    out["top_categories_revenue"] = rows(con, """
      SELECT coalesce(category_code, 'uncategorised') AS category_code, sum(price) AS revenue, count(*) AS purchases
      FROM e WHERE event_type = 'purchase' GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")

    out["smartphone_brands"] = rows(con, """
      SELECT brand, count(*) FILTER (WHERE event_type = 'view') AS views,
             count(*) FILTER (WHERE event_type = 'purchase') AS purchases,
             sum(price) FILTER (WHERE event_type = 'purchase') AS revenue,
             count(*) FILTER (WHERE event_type = 'purchase') * 1.0 / nullif(count(*) FILTER (WHERE event_type = 'view'), 0) AS purchases_per_view,
             median(price) FILTER (WHERE event_type = 'purchase') AS median_price
      FROM e WHERE category_code = 'electronics.smartphone' AND brand IS NOT NULL
      GROUP BY 1 ORDER BY revenue DESC LIMIT 8""")

    out["smartphone_price_bands"] = rows(con, """
      SELECT least(floor(price / 100) * 100, 1500) AS price_band,
             count(*) FILTER (WHERE event_type = 'view') AS views,
             count(*) FILTER (WHERE event_type = 'purchase') AS purchases,
             count(*) FILTER (WHERE event_type = 'purchase') * 1.0 / nullif(count(*) FILTER (WHERE event_type = 'view'), 0) AS purchases_per_view
      FROM e WHERE category_code = 'electronics.smartphone' AND price > 0 GROUP BY 1 ORDER BY 1""")

    # Weekly cohorts by first-seen week; share of the cohort active in each later week.
    out["cohorts"] = rows(con, """
      WITH w AS (SELECT DISTINCT user_id, date_trunc('week', event_time) AS wk FROM e),
           f AS (SELECT user_id, min(wk) AS cohort FROM w GROUP BY 1),
           size AS (SELECT cohort, count(*) AS users FROM f GROUP BY 1)
      SELECT f.cohort, size.users, date_diff('week', f.cohort, w.wk) AS week_n, count(*) * 1.0 / size.users AS active_share
      FROM w JOIN f USING (user_id) JOIN size USING (cohort)
      WHERE f.cohort > (SELECT min(wk) FROM w)
      GROUP BY 1, 2, 3 ORDER BY 1, 3""")

    out["buyers"] = rows(con, """
      WITH b AS (SELECT user_id, count(DISTINCT CAST(event_time AS DATE)) AS buy_days, sum(price) AS spend
                 FROM e WHERE event_type = 'purchase' GROUP BY 1)
      SELECT count(*) AS buyers, avg((buy_days >= 2)::INT) AS repeat_share,
             sum(spend) FILTER (WHERE buy_days >= 2) / sum(spend) AS repeat_revenue_share,
             quantile_cont(spend, 0.5) AS median_spend FROM b""")[0]

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("funnel", "buyers")}, indent=1, default=str))


if __name__ == "__main__":
    main()
