# E-commerce clickstream: 110M events, from raw logs to a targeting model

Two months of behaviour from a large multi-category online store: **109,950,743 events** (views,
cart adds, purchases) from **5.3M users** in **23.0M sessions**, with **$505M** of purchases.
The questions are a product or growth team's: where does the funnel leak, what converts, do new
visitors come back, and who is about to buy?

**Data:** [eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
(REES46, Oct–Nov 2019) on Kaggle, 14.7 GB of CSV. DuckDB converts it to 2 GB of Parquet and runs
every query over all rows on an 8 GB laptop.

## Headline results

| | |
|---|---|
| Cart abandonment | **59.4%** of sessions with a cart add end without a purchase (1.38M sessions) |
| Repeat buyers | **30%** of the 697,470 buyers bought on 2+ days, and they bring **66%** of revenue |
| Smartphones | **66%** of revenue ($335M). Samsung sells more phones than Apple (305k vs 243k), Apple earns 2.7× the revenue |
| Return visits | 15–27% of each weekly cohort of new visitors is active again the following week (lowest for the post-sale cohort) |
| Targeting model | Top 10% of users by predicted propensity contain **41.7%** of next-fortnight buyers, vs 36.8% for the best simple rule; AUC **0.796** vs 0.761 |

![Daily revenue and users](figures/daily.png)

**Data-quality finding.** On 15 November the log has 5.7M views and zero purchases, then the
mid-November sale produces $23M and $58M on the 16th and 17th against a typical $7M day. That day
is kept visible rather than smoothed over, and it matters for reading the model's test window.

<p><img src="figures/funnel_by_category.png" width="49%"> <img src="figures/cohorts.png" width="49%"></p>

Electronics sessions add to cart most often (13.4%) and convert best (8.0%); apparel converts at 2.2%.
Once something is in the cart, 55–70% of carted sessions buy in every category.

![Purchases by weekday and hour](figures/hour_weekday.png)

Purchases concentrate between 04:00 and 17:00 UTC. The Saturday and Sunday peaks are partly the
16–17 November sale weekend.

<p><img src="figures/smartphone_price.png" width="60%"></p>

## Who buys in the next 15 days?

A customer-level model evaluated on a later time window than it was trained on:

| | Behaviour window | Label: bought in | Users | Buy rate |
|---|---|---|---|---|
| Train | 1–31 Oct | 1–15 Nov | 3,022,290 | 3.8% |
| Test | 16 Oct–15 Nov | 16–30 Nov | 3,393,482 | 6.0% |

| Method | AUC | Buyers in top 5% | Buyers in top 10% | Hit rate, top 10% |
|---|---|---|---|---|
| **Gradient boosting** | **0.796** | **26.7%** | **41.7%** | **25.1%** |
| Rule: bought before | 0.761 | 23.3% | 32.9% | 19.8% |
| Rule: carted in the last 7 days | 0.711 | 22.0% | 36.8% | 22.1% |
| Rule: most recently active | 0.717 | 19.0% | 33.4% | 20.1% |

<p><img src="figures/gains.png" width="49%"> <img src="figures/importance.png" width="49%"></p>

Past purchases, recency and breadth of products browsed carry most of the signal. The test fortnight
includes the sale, so buy rates are higher than in training; the ranking still holds.

## Method
1. `ingest.py`: DuckDB reads each monthly CSV as text, parses timestamps, splits the top-level category
   and writes ZSTD Parquet (42.4M October rows, 67.5M November rows).
2. `analyze.py`: session funnel, daily trend, weekday × hour, category and brand conversion, smartphone
   price bands, weekly cohorts and repeat buyers, all in SQL over the full data.
3. `model.py`: 19 behavioural features per user (activity counts, sessions, active days, recency,
   tenure, products and categories browsed, prices viewed and carted, spend, last-7-day and last-day
   activity, category mix and top category). Negatives are down-sampled to 25% with matching weights for training;
   the test set is scored in full.
4. `figures.py`: charts above.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh      # needs ~/.kaggle/kaggle.json; ~4.6 GB download, 14.7 GB unpacked
python ingest.py        # CSV -> Parquet (then the CSVs can be deleted)
python analyze.py       # results/analysis.json
python model.py         # results/model.json
python figures.py       # figures/*.png
```

*Timestamps are UTC. Prices are as logged by the store. Sessions follow the dataset's own `user_session` id.*
