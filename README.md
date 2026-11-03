# 🥬 FreshChain AI
**Predictive Supply Chain & Food Waste Reduction Using Time-Series Forecasting**
Python · Polars · Pandas · LightGBM / XGBoost · scikit-learn · Streamlit · Plotly · SQLite

A retail decision-support app: real forecasting pipeline → leakage-safe evaluation → transparent inventory maths →
potential-waste-risk scoring → alerts, scenarios, exports and a grounded AI Copilot.

---
## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py          # then press "Load demo workspace" on the Home page (synthetic data + trained models, ~1 min)
```
Or from the terminal: `python -m scripts.make_synthetic --train`.  Tests: `pytest` (88 tests, all passing on Python 3.13 / pandas 3 / Streamlit 1.58).

**Real data (FreshRetailNet-50K)** - see `DATASET_GUIDE.md`:
```bash
python -m scripts.download_dataset --split train
python -m scripts.inspect_dataset data/raw/freshretailnet_train.parquet     # confirm the schema first
python -m scripts.prepare_data --raw data/raw/freshretailnet_train.parquet --max-series 2000
python -m scripts.train --horizons 7 14 28 --folds 3
```
**Docker:** `docker build -t freshchain . && docker run -p 8501:8501 freshchain`.

## AI Copilot (LLM tool-calling)
Connect a model in *Settings & Data -> Copilot / LLM* (Anthropic, or any OpenAI-compatible endpoint incl. local Ollama) or via `.env` (see `.env.example`).
* The model chooses among 11 validated tools (`src/chatbot/agent.py`); tool parameters are type/range checked, unknown tools are rejected, no code/SQL/file access exists.
* Every figure in the final answer must appear in the tool results. Otherwise the model is asked once to correct itself; if it still fails, the deterministic answer is shown with a notice.
* Conversation memory (last 12 messages), tables/charts/limitations attached to each answer, follow-up suggestions.
* With no key the Copilot runs in deterministic mode (rule-based router over the same tools).
* Verified against mock OpenAI-compatible and Anthropic endpoints (`tests/test_agent.py` + manual run); **not yet exercised against a live provider** - connect one and use "Apply & test connection".

## Status and caveats
Verified here: full test suite, end-to-end pipeline on synthetic data (LightGBM backend), every view rendered in a real Streamlit server, landing page and Copilot interaction in a browser.
Not verified: real FreshRetailNet-50K run (column names per `DATASET_GUIDE.md`), XGBoost/SHAP paths (optional packages not installed in the dev environment), live LLM provider, mobile layout. Demo numbers are **synthetic** and say nothing about real-world accuracy.

## Implemented vs optional vs future
| Implemented | Optional (needs setup) | Future / not implemented |
|---|---|---|
| Dataset adapter + generic column mapping, validation, cleaning, Polars/pandas loaders, seeded sampling | LightGBM (auto-used if installed, else scikit-learn HistGradientBoosting fallback — labelled in the UI) | M5 / Favorita adapters (interface ready) |
| Seasonal-naive baseline, global GBM, chronological split, rolling-origin backtest, empirical prediction intervals | XGBoost comparison (`--xgboost`) | Latent-demand (uncensoring) models; hierarchical reconciliation |
| (s,S) inventory engine, safety stock, MOQ/pack/shelf-life, stock-out risk, scenarios | SHAP (button on Model Performance; tree backends only) | Causal promotion/price-response model |
| Potential Waste Risk Score (proxy), rule-based alerts + acknowledgement, SQLite persistence | LLM wording via any OpenAI-compatible endpoint | Email/SMS delivery (**not implemented** — alerts are in-app/exported only) |
| 11-page Streamlit app, 16+ Plotly charts, CSV exports, PDF executive summary | | Cloud deployment, auth, multi-user roles |
| Copilot: router + 10 validated tools + deterministic mode + grounding guard | | Streaming / tool-calling LLM agents |

## Architecture (data flow)
```
raw file ──Polars lazy scan/project/sample──▶ DatasetAdapter ──▶ canonical frame ──validate──▶ clean_canonical
   (data/raw)                                                                                   │ (dedupe, returns, calendar fill)
                                                                                                ▼
                                                                       data/processed/demand_daily.parquet (+meta)
                                                                                                │
        build_feature_frame(H) ◀───────────────────────────────────────────────────────────────┘
        │  direct strategy: lags = H+{0,1,7}, rolling windows on demand shifted by H  → no look-ahead
        ▼
  chronological_split ─▶ GBMForecaster.fit (train, valid) ─▶ hold-out metrics vs seasonal-naive ─▶ rolling-origin backtest
        │                                                  └▶ empirical residual intervals (coverage checked on hold-out)
        ▼
  refit on all history (same n_estimators) ─▶ forecast_future ─▶ artifacts/*.parquet|joblib|run_summary.json
        ▼
  AppContext ─▶ services (policy table, waste table, alerts) ─▶ Streamlit pages  /  Copilot tools
```
`src/` layers: `data` (ingest/validate/clean) · `features` · `forecasting` · `inventory` · `waste` · `alerts` · `analytics` (metrics, aggregations, context, services, explanations) · `chatbot` · `reporting` · `storage` (SQLite) · `ui` (Streamlit glue + Plotly charts). Business logic never imports Streamlit (only `src/ui` and `views/` do), which is why it is unit-testable.

## Forecasting & leakage prevention (interview notes)
* **Direct multi-horizon strategy**: a model for horizon *H* uses only features computed from data at or before *t − H* (lags `H, H+1, H+7`; rolling mean/std of demand shifted by *H*). A forecast made at origin *T* for *T+1…T+H* therefore uses information available at *T*. `tests/test_features.py` proves this by corrupting all demand after a cutoff and asserting that earlier feature rows are unchanged.
* **Future inference uses the same code path**: future rows (NaN demand) are appended and run through `build_feature_frame`; lags ≥ H never touch NaN.
* **Known-future covariates** (promo, discount, price, holiday) are carried forward / set to 0 — the forecast therefore assumes *no promotion* unless a plan is supplied. Weather is intentionally not used (unknown in the future).
* **Splits are chronological**: train < validation (7 d) < test (last *H* days). No random splits anywhere (the sklearn fallback's internal early-stopping is disabled for this reason).
* **Censored demand**: days with ≥ 3 stock-out hours are excluded from *training* (heuristic, `config.STOCKOUT_EXCLUDE_HOURS`); metrics are shown with and without such days.
* **Supported horizons** = `2H + 49` days of history (`config.required_history_days`): 63 / 77 / 105 days for 7 / 14 / 28. Unsupported horizons are reported, not faked.
* **Intervals**: empirical quantiles of validation residuals (80% nominal); measured hold-out coverage is displayed. Not guaranteed coverage.
* **Metrics** (`src/analytics/metrics.py`): MAE, RMSE, **WAPE** = Σ|e| / Σy, **bias** = (Σŷ − Σy)/Σy (positive = over-forecast), **MASE** (vs in-sample seasonal-naive MAE). MAPE is deliberately absent (undefined at zero demand). Zero denominators → NaN, never inf.
* The reported hold-out metrics belong to the model trained up to the stated cutoff; the **deployed** forecast model is refit on all history with the same settings (documented in `run_summary.json`).

## Inventory maths (`src/inventory/optimization.py`)
With lead time *L*, review period *R*, protection interval *P = L + R*, forecast error σ_d (per-day, from hold-out residuals, else recent history), lead-time std σ_L:
```
μ_L  = Σ forecast over next L days                    μ_P = Σ forecast over next P days
SS   = z(service level) · √( P·σ_d² + d̄²·σ_L² )       ROP = μ_L + SS          S = μ_P + SS
IP   = on_hand + on_order − committed
order when IP ≤ ROP:  Q = constraints( max(0, S − IP) )   → round up to pack size, enforce MOQ, cap by shelf-life demand
stock-out risk = 1 − Φ( (IP − μ_L) / √(L·σ_d² + d̄²σ_L²) )     excess = max(0, IP − S)     days of cover = on_hand / d̄
```
Assumptions shown in the UI: normal errors, independent daily errors (likely understates risk), on-order stock arrives within L, forecast beyond the horizon repeats the last-7-day mean (flagged). **Sales history is not stock**: without uploaded `on_hand` the app shows policy parameters only and never an order quantity. Defaults for lead time etc. are *labelled assumptions* (`assumed_fields` column).

## Potential Waste Risk Score (proxy — not measured waste)
Weighted 0–100 score from: days of cover vs shelf life (or a 14-day reference), recent 7-day demand decline vs prior 21 days, demand variability (CV), optional perishability flag. Weights renormalise over available components; without stock data the score is labelled *demand-pattern proxy only*. High ≥ 60, Medium ≥ 30. The UI never calls this observed waste and makes **no waste-reduction claim**.

## Data dictionary (canonical table)
| column | meaning |
|---|---|
| `series_id` | `store_id|product_id` |
| `date` | calendar day (daily grain; gaps filled) |
| `demand` | observed units sold (censored by stock-outs) |
| `category` | product family (`unknown` if absent) |
| `price` / `discount` | unit price / discount factor, if the source has them |
| `promo`, `holiday` | 0/1 indicators (max over duplicates) |
| `stockout_hours` | out-of-stock hours that day (if available) |
| `is_imputed` | row was a missing day filled with 0 demand |

Artifacts (`artifacts/`): `model_h{H}.joblib` (trusted local pickles — don't load files from others), `forecast_h{H}.parquet`, `test_predictions_h{H}.parquet`, `backtest_folds_h{H}.parquet`, `importance_h{H}.parquet`, `run_summary.json`.

## Pages
Home (landing) · Executive Overview · Demand Forecasting · Inventory Optimization · Waste Risk Intelligence · Supply Chain Alerts · Store & Product Analytics · Scenario Simulator · AI Copilot · Model Performance · Data Explorer · Settings & Data. Global filters (date, family, store, product, horizon) live in the shared sidebar; unsupported horizons are marked "(no model)". Exports: forecasts, recommendations, waste risk, alerts, scenarios (CSV, formula-injection-safe) and a PDF executive summary.

## Tests
`pytest` (82 tests): validation & upload rejection, adapters, cleaning rules, lag/leakage, chronological splits, forecast shape/non-negativity, metric edge cases, inventory formulas & constraints, waste scoring, router/tool validation, LLM failure & grounding, CSV/PDF export, alerts, SQLite. Fixtures use `tests/synthetic.py` (**synthetic, test-only**). Results above are from the authoring sandbox; yours may differ — fix or report differences.

## Troubleshooting
* *"No dataset is loaded"* → run `prepare_data` or `make_synthetic`, or upload in Settings.
* *Horizon shows "(no model)"* → history too short (see supported horizons) or you haven't trained; run `scripts.train`.
* *`AdapterError: missing required source column`* → run `scripts.inspect_dataset`, then adjust `FRESHRETAILNET.column_map`.
* *Out of memory* → lower `--max-series`, restrict `--date-from/--date-to`, close other apps; avoid `--full` on laptops.
* *LightGBM import error* → the app falls back to scikit-learn (labelled "sklearn_hgb"); install `lightgbm` for the primary model. On macOS you may need `brew install libomp`.
* *Copilot says LLM unavailable* → expected without a key; check `.env`, network, rate limits. Answers still work.
* *Stale UI after changing files* → Streamlit "Rerun"/clear cache; caches are keyed by file modification times.

## Known limitations
Censored demand; no inventory/waste ground truth in the dataset; promotion effect not causal; normal-approximation inventory risk; short histories limit horizons/seasonality; CSS/layout and mobile behaviour untested; rule-based NLU; pins unverified (see top).

*Attribution:* if you use FreshRetailNet-50K (CC-BY-4.0), cite arXiv:2505.16319 and state your changes — see `DATASET_GUIDE.md`.
