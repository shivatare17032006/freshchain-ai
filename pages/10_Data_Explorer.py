import pandas as pd
import streamlit as st

from src.data.validation import validate_canonical
from src.reporting.csv_export import to_csv_bytes
from src.ui import theme
from src.ui.state import history_for, init_page

ctx, flt = init_page("Data Explorer", "🔎", "Inspect the processed dataset behind every page")
h = history_for(ctx, flt)
theme.show_badges("observed")
m = ctx.meta
c = st.columns(4)
c[0].metric("Rows (filtered)", f"{len(h):,}")
c[1].metric("Series", f"{h['series_id'].nunique():,}")
c[2].metric("Dates", f"{h['date'].min().date() if len(h) else '-'} → {h['date'].max().date() if len(h) else '-'}")
c[3].metric("Dataset", str(m.get("dataset", "unknown"))[:28])
if m.get("sampling"):
    st.caption(f"Sampling: {m['sampling']}" + (f" · seed {m['seed']}" if m.get("seed") is not None else ""))
t1, t2, t3 = st.tabs(["Preview", "Data quality", "Summary statistics"])
with t1:
    theme.data_table(h.assign(date=h["date"].dt.date).head(5000), "explorer", 25)
    st.caption("Preview limited to the first 5,000 filtered rows; the download below exports up to 200,000 rows.")
    st.download_button("Download filtered data (CSV, ≤200k rows)", to_csv_bytes(h.head(200_000).assign(date=h["date"].head(200_000).dt.date)), "filtered_history.csv", "text/csv")
with t2:
    rep = validate_canonical(ctx.history.rename(columns={}))
    cl = m.get("cleaning", {})
    st.markdown("**Cleaning report (from preprocessing)**")
    st.json(cl) if cl else st.caption("No cleaning report stored.")
    st.markdown("**Missing values**")
    miss = ctx.history.isna().mean().rename("missing_fraction").reset_index().rename(columns={"index": "column"})
    st.dataframe(miss, hide_index=True, use_container_width=True)
    st.markdown("**Checks on the processed table**")
    st.write({"duplicate (date, store, product) rows": rep.stats.get("duplicate_keys"), "negative demand rows": rep.stats.get("negative_demand"), "imputed (missing-day) rows": int(ctx.history["is_imputed"].sum()) if "is_imputed" in ctx.history else "n/a"})
    for w in rep.warnings:
        st.warning(w)
    for w in m.get("validation_warnings", []):
        st.caption("At load time: " + w)
with t3:
    st.dataframe(h[[c for c in ("demand", "price", "discount", "promo", "holiday", "stockout_hours") if c in h.columns]].describe().T.round(3), use_container_width=True)
