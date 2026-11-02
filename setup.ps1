# -------------------------------------------------
# setup.ps1 – fast reproducible environment bootstrap
# -------------------------------------------------
# 1️⃣ Unzip the original archive if it is still present
if (Test-Path "./freshchain-ai.zip") {
    Write-Host "Unzipping freshchain-ai.zip ..."
    Expand-Archive -LiteralPath "./freshchain-ai.zip" -DestinationPath "." -Force
}

# 2️⃣ Create a clean Python 3.11 venv (named .venv2 to avoid lock issues)
$venvPath = ".\.venv2"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating Python 3.11 venv at $venvPath ..."
    py -3.11 -m venv $venvPath
}
# Activate the venv
& "$venvPath\Scripts\Activate.ps1"

# 3️⃣ Install all dependencies (allows version bumps)
Write-Host "\nInstalling requirements ..."
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ pip install failed – see output above."
    exit 1
}

# 4️⃣ Verify that optional heavy‑weight libs load correctly
Write-Host "\nVerifying optional libraries ..."
python -c 'import importlib; optional = ["lightgbm","xgboost","polars","pyarrow","shap","streamlit","plotly"]; missing = [m for m in optional if importlib.util.find_spec(m) is None]; print("Missing optional libs:", missing); print("All present?", not missing)'

Write-Host "`n✅ Setup finished. You can now run the full pipeline with .\\run_all.ps1"
