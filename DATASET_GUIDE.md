# DATASET_GUIDE.md

> **Read this first.** This guide separates what was *verified from public sources* from what must be
> *verified on your machine*. The project was authored in a sandbox **without network access**, so the
> dataset files were never downloaded there. Every number that would normally appear here (rows, file
> sizes, date range, final column list) is deliberately left for you to generate with
> `python -m scripts.inspect_dataset` — nothing below is invented.

## 1. Selected primary dataset: FreshRetailNet-50K (provisional)

| Item | Status |
|---|---|
| Source | Hugging Face dataset repo `Dingdong-Inc/FreshRetailNet-50K` (the paper names this repo and the baseline code at `github.com/Dingdong-Inc/frn-50k-baseline`). A community mirror also exists under another namespace — prefer the official repo. **Verified from the paper/dataset card via web search.** |
| Paper to cite | *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail* (arXiv:2505.16319). **Verified.** |
| License | **CC-BY-4.0** per the dataset card → attribution required (cite the paper, link the dataset, state changes you made). **Verified from the dataset card; re-check the card when you download.** |
| Files | The card lists parquet splits `data/train.parquet` and `data/eval.parquet`. **Verified from the card; exact sizes: run the inspector.** |
| Key property | Hourly **stock-out annotations**, so zero/low sales caused by being out of stock can be separated from true zero demand (sales are *censored* demand). **Verified from the paper.** About one fifth of the data is organically occurring stock-outs (paper/card). |
| Row count, size, date range | **NOT VERIFIED here** → run `python -m scripts.inspect_dataset data/raw/freshretailnet_train.parquet` and paste the output below. |
| Final column list | **NOT VERIFIED here.** The adapter (`src/data/dataset_adapters.py`) expects `dt, store_id, product_id, sale_amount, first_category_id, discount, activity_flag, holiday_flag` and a 24-value `hours_stock_status` list (assumed `1` = out of stock). These come from the dataset documentation as understood by the author and **must be confirmed**; the adapter fails loudly with the list of available columns if a required one is missing. |

### Why it is suitable (and what to double-check)
* Perishable / fresh retail → matches the "food waste" framing.
* Stock-out flags → lets the pipeline treat censored demand explicitly (see §4).
* Store × product × day structure with promotion/discount/holiday indicators → supports the
  global-model + calendar + promotion feature design.
* **Check:** temporal coverage. If the history is only a few months, 28-day models and yearly seasonality are *not*
  supported; the app disables unsupported horizons automatically (`config.required_history_days`).

## 2. Verified download instructions

```bash
python -m scripts.download_dataset --split train     # tries the Hugging Face resolve URL, writes data/raw/freshretailnet_train.parquet
python -m scripts.inspect_dataset data/raw/freshretailnet_train.parquet
```
If the script fails (offline, proxy, URL change): open the dataset page on Hugging Face in a browser, download
`data/train.parquet`, save it as `data/raw/freshretailnet_train.parquet`, then run the inspector. Alternatively
use **Settings & Data Management → Upload** (CSV/Parquet with column mapping). Nothing is ever fabricated as a fallback.
`python -m scripts.make_synthetic` creates **clearly labelled synthetic smoke-test data** (red banner in the UI) — only
for trying the app, never for conclusions.

## 3. Inspection output (paste here after running the inspector)

```
<paste data/raw/inspection_report.md here: file size, row count, dtypes, date range, null counts>
```

## 4. Target, features, preprocessing

* **Target:** `sale_amount` → canonical `demand` (daily units per store × product). Observed sales, **not** latent demand.
* **Canonical schema:** `date, store_id, product_id, demand` (+ optional `category, price, discount, promo, holiday, stockout_hours`).
* **Stock-out handling:** `stockout_hours` = number of out-of-stock hours that day (derived from the hourly list).
  Days with ≥ `STOCKOUT_EXCLUDE_HOURS` (default 3, a *heuristic* in `config.py`) are excluded from model **training**, and
  evaluation is reported both on all rows and excluding such days.
* **Cleaning rules** (`src/data/preprocessing.py`): drop unparseable dates / missing ids / non-numeric demand (counted);
  negative demand (returns) clipped to 0 (or dropped); duplicate (date, store, product) rows aggregated (demand summed);
  missing calendar days inside a series' life filled with 0 demand and flagged `is_imputed` (assumes no record = no sale).
* **Storage:** processed table → `data/processed/demand_daily.parquet` (+ `dataset_meta.json`).
* **Memory/performance:** raw files are scanned lazily with Polars (column projection, date predicate pushdown, list-column
  reduction). Default `prepare_data` mode is a **seeded development sample** of series (`--max-series 2000`); `--full` processes everything.
  The sample size and seed are stored in the metadata and shown in the app.

## 5. Known limitations / missing information (by design of the dataset)
* **No inventory levels, expiry dates, spoilage or waste records** → inventory recommendations need user-supplied stock; waste is a **proxy risk score**, never measured waste.
* No price in currency units (only a discount indicator, per the schema above) → inventory *value* needs uploaded `unit_cost`.
* Sales are censored by stock-outs; promotions/holidays beyond history are unknown at forecast time (forecast assumes no promotion).
* Short histories restrict horizons and make yearly seasonality unobservable.

## 6. Fallback / alternative datasets (not evaluated in this build)
* **M5 Forecasting – Accuracy** (Walmart; sales + prices + calendar) and **Corporación Favorita Grocery Sales** are the alternatives named in the brief. Both are Kaggle competition datasets that normally require a Kaggle account and acceptance of competition rules; their licenses, sizes and schemas were **not checked** here. M5 is *wide* (one column per day) and Favorita is *long* with an `onpromotion` flag — each needs its own adapter (M5: melt + join calendar/prices). The adapter interface is `DatasetAdapter(column_map, prepare=...)`; the *Generic* adapter + column-mapping UI already works for any long-format CSV/Parquet.

## 7. Attribution template (CC-BY-4.0)
"Contains data from FreshRetailNet-50K (Dingdong-Inc), licensed under CC BY 4.0, processed/aggregated by FreshChain AI. Paper: arXiv:2505.16319."
