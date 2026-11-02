"""PDF executive summary (reportlab; charts via matplotlib when available)."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd


def _chart_png(history_daily: pd.DataFrame | None, forecast_daily: pd.DataFrame | None) -> bytes | None:
    if history_daily is None or history_daily.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=140)
    ax.plot(history_daily["date"], history_daily["demand"], color="#0B1F33", lw=1.4, label="Observed demand")
    if forecast_daily is not None and len(forecast_daily):
        ax.plot(forecast_daily["date"], forecast_daily["forecast"], color="#14B8A6", lw=1.8, label="Model forecast")
    ax.set_ylabel("Units / day")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_executive_pdf(title: str, kpis: dict, notes: list[str], tables: dict[str, pd.DataFrame] | None = None,
                        history_daily: pd.DataFrame | None = None, forecast_daily: pd.DataFrame | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    el = [Paragraph(title, styles["Title"]),
          Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]), Spacer(1, 10)]
    if kpis:
        data = [["Metric", "Value"]] + [[str(k), str(v)] for k, v in kpis.items()]
        t = Table(data, colWidths=[9 * cm, 7 * cm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F33")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        el += [t, Spacer(1, 10)]
    png = _chart_png(history_daily, forecast_daily)
    if png:
        el += [Image(io.BytesIO(png), width=16 * cm, height=6.4 * cm), Spacer(1, 8)]
    for name, df in (tables or {}).items():
        el.append(Paragraph(name, styles["Heading3"]))
        if df is None or df.empty:
            el.append(Paragraph("No data.", styles["Normal"]))
            continue
        sub = df.head(12)
        data = [list(map(str, sub.columns))] + [[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in r] for r in sub.itertuples(index=False)]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14B8A6")), ("FONTSIZE", (0, 0), (-1, -1), 7),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
        el += [t, Spacer(1, 8)]
    for n in notes:
        el.append(Paragraph("• " + n, styles["Normal"]))
    doc.build(el)
    return buf.getvalue()
