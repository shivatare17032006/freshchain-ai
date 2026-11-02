# -------------------------------------------------
# run_all.ps1 – end‑to‑end execution (fast)
# -------------------------------------------------
# Activate the virtual environment (created as .venv2)
& "./.venv2/Scripts/Activate.ps1"

# -----------------------------------------------------------------
# 1️⃣ Download the official dataset (train split)
# -----------------------------------------------------------------
python -m scripts.download_dataset --split train

# -----------------------------------------------------------------
# 2️⃣ Inspect the parquet schema – write output for docs
# -----------------------------------------------------------------
python -m scripts.inspect_dataset data/raw/freshretailnet_train.parquet > inspection_output.txt
Write-Host "\n--- Inspection saved to inspection_output.txt ---`n"

# -----------------------------------------------------------------
# 3️⃣ Prepare a light‑weight processed version (max 2000 series)
# -----------------------------------------------------------------
python -m scripts.prepare_data --raw data/raw/freshretailnet_train.parquet --max-series 2000

# -----------------------------------------------------------------
# 4️⃣ Run the test suite – capture log for the final report
# -----------------------------------------------------------------
python -m pytest -q | Tee-Object -FilePath test_log.txt

# -----------------------------------------------------------------
# 5️⃣ Train models (LightGBM then XGBoost) – capture their printed metrics
# -----------------------------------------------------------------
python -m scripts.train --horizons 7 14 28 --folds 3 --backend lightgbm | Tee-Object -FilePath train_lgbm.txt
python -m scripts.train --horizons 7 14 28 --folds 3 --backend xgboost | Tee-Object -FilePath train_xgb.txt

# -----------------------------------------------------------------
# 6️⃣ Launch the Streamlit UI (will stay running – close when done)
# -----------------------------------------------------------------
streamlit run app.py
