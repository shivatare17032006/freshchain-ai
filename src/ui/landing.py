"""Landing-page building blocks (HTML/SVG strings). Values shown come from the loaded data - nothing is hard-coded."""
from __future__ import annotations

import html

import pandas as pd

from src.ui.theme import ACCENT, INK, MONO, MUTED, PAPER, RULE, SURFACE

CSS = f"""
<style>
.hm-hero h1 {{ font-size: clamp(2.2rem, 4.2vw, 3.6rem); line-height: 1.04; letter-spacing: -.035em; font-weight: 600; margin: .4rem 0 1.1rem; color:{INK}; padding:0; }}
.hm-hero h1 em {{ font-style: normal; color:{ACCENT}; }}
.hm-lede {{ font-size: 1.06rem; line-height: 1.55; color:#38424A; max-width: 34rem; margin-bottom: 1.3rem; }}
.hm-top {{ display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid {INK}; padding-bottom:.7rem; margin-bottom:2.2rem;
  font-family:{MONO}; font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; }}
.hm-top span:last-child {{ color:{MUTED}; }}
.hm-panel {{ background:{SURFACE}; border:1px solid {INK}; padding:0; }}
.hm-panel-h {{ display:flex; justify-content:space-between; padding:.55rem .9rem; border-bottom:1px solid {RULE}; font-family:{MONO}; font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.hm-panel-h b {{ color:{INK}; font-weight:500; }}
.hm-readouts {{ display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid {RULE}; }}
.hm-readouts div {{ padding:.65rem .9rem; border-right:1px solid {RULE}; }}
.hm-readouts div:last-child {{ border-right:0; }}
.hm-readouts small {{ display:block; font-family:{MONO}; font-size:.58rem; letter-spacing:.12em; text-transform:uppercase; color:{MUTED}; }}
.hm-readouts strong {{ font-family:{MONO}; font-size:1.2rem; font-weight:500; letter-spacing:-.02em; }}
.hm-strip {{ display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid {INK}; border-bottom:1px solid {INK}; margin:2.6rem 0 3rem; }}
.hm-strip div {{ padding:1rem 1.1rem; border-right:1px solid {RULE}; }}
.hm-strip div:last-child {{ border-right:0; }}
.hm-strip small {{ font-family:{MONO}; font-size:.62rem; letter-spacing:.13em; text-transform:uppercase; color:{MUTED}; }}
.hm-strip strong {{ display:block; font-family:{MONO}; font-size:1.9rem; font-weight:500; letter-spacing:-.03em; margin-top:.15rem; }}
.hm-sec {{ display:flex; align-items:baseline; gap:1rem; margin:3rem 0 1.1rem; }}
.hm-sec .n {{ font-family:{MONO}; font-size:.7rem; color:{ACCENT}; letter-spacing:.14em; }}
.hm-sec h2 {{ margin:0; padding:0; font-size:1.35rem; letter-spacing:-.02em; }}
.hm-sec p {{ margin:0 0 0 auto; color:{MUTED}; font-size:.86rem; max-width:30rem; text-align:right; }}
.hm-grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid {INK}; background:{SURFACE}; }}
.hm-grid4 > div {{ padding:1.1rem 1.15rem 1.3rem; border-right:1px solid {RULE}; }}
.hm-grid4 > div:last-child {{ border-right:0; }}
.hm-grid4 .k {{ font-family:{MONO}; font-size:.68rem; color:{ACCENT}; letter-spacing:.14em; }}
.hm-grid4 h4 {{ margin:.5rem 0 .4rem; font-size:1rem; font-weight:600; letter-spacing:-.01em; }}
.hm-grid4 p {{ margin:0; font-size:.86rem; line-height:1.5; color:#3F4A52; }}
.hm-two {{ display:grid; grid-template-columns:1fr 1fr; gap:0; border:1px solid {INK}; background:{SURFACE}; }}
.hm-two > div {{ padding:1.2rem 1.4rem; }}
.hm-two > div:first-child {{ border-right:1px solid {RULE}; }}
.hm-two h4 {{ margin:0 0 .6rem; font-family:{MONO}; font-size:.7rem; letter-spacing:.13em; text-transform:uppercase; }}
.hm-two li {{ font-size:.88rem; line-height:1.5; margin-bottom:.4rem; color:#2F3A42; }}
.hm-foot {{ margin-top:3rem; padding-top:.9rem; border-top:1px solid {INK}; font-family:{MONO}; font-size:.66rem; letter-spacing:.08em; color:{MUTED}; display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
@media (max-width: 900px) {{ .hm-strip, .hm-grid4 {{ grid-template-columns:1fr 1fr; }} .hm-two {{ grid-template-columns:1fr; }} .hm-two > div:first-child {{ border-right:0; border-bottom:1px solid {RULE}; }} }}
</style>
"""


def _scale(vals, lo, hi, a, b):
    span = (hi - lo) or 1.0
    return [a + (v - lo) / span * (b - a) for v in vals]


def forecast_svg(hist: pd.Series | None, fc: pd.DataFrame | None, w: int = 600, h: int = 250) -> str:
    """Observed daily total (ink) + model forecast with interval (orange) drawn from real data."""
    if hist is None or len(hist) < 2:
        return (f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg"><rect width="{w}" height="{h}" fill="{SURFACE}"/>'
                f'<text x="{w/2}" y="{h/2}" text-anchor="middle" font-family="{MONO}" font-size="12" fill="{MUTED}">NO DATA LOADED</text></svg>')
    hist = hist.tail(56)
    n_h = len(hist)
    n_f = 0 if fc is None else len(fc)
    allv = list(hist.values) + ([] if fc is None else list(fc["forecast"]) + list(fc.get("upper", fc["forecast"])) + list(fc.get("lower", fc["forecast"])))
    lo, hi = min(allv) * 0.92, max(allv) * 1.06
    px0, px1, py0, py1 = 14, w - 14, h - 26, 16
    total = n_h + n_f
    xs = [px0 + i / max(total - 1, 1) * (px1 - px0) for i in range(total)]
    yh = _scale(list(hist.values), lo, hi, py0, py1)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Observed demand and forecast">',
             f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>']
    for k in range(1, 5):
        y = py1 + (py0 - py1) * k / 5
        parts.append(f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E9E6DE"/>')
    parts.append(f'<line x1="{px0}" x2="{px1}" y1="{py0}" y2="{py0}" stroke="{INK}"/>')
    parts.append(f'<polyline fill="none" stroke="{INK}" stroke-width="1.5" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[:n_h], yh)) + '"/>')
    if n_f:
        fx = xs[n_h - 1:]
        yf = _scale([float(hist.values[-1])] + list(fc["forecast"]), lo, hi, py0, py1)
        if "upper" in fc.columns:
            yu = _scale([float(hist.values[-1])] + list(fc["upper"]), lo, hi, py0, py1)
            yl = _scale([float(hist.values[-1])] + list(fc["lower"]), lo, hi, py0, py1)
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(fx, yu)) + " " + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(reversed(fx), reversed(yl)))
            parts.append(f'<polygon points="{poly}" fill="{ACCENT}" opacity=".14"/>')
        parts.append(f'<polyline fill="none" stroke="{ACCENT}" stroke-width="2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(fx, yf)) + '"/>')
        x0 = xs[n_h - 1]
        parts.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{py1 - 6}" y2="{py0}" stroke="{MUTED}" stroke-dasharray="2 3"/>')
        parts.append(f'<text x="{px0}" y="{py1 - 4}" font-family="{MONO}" font-size="9" letter-spacing="1.2" fill="{MUTED}">OBSERVED · {n_h}D</text>')
        parts.append(f'<text x="{px1}" y="{py1 - 4}" text-anchor="end" font-family="{MONO}" font-size="9" letter-spacing="1.2" fill="{ACCENT}">FORECAST +{n_f}D ▸</text>')
    parts.append("</svg>")
    return "".join(parts)


def top_bar(left: str, right: str) -> str:
    return f'<div class="hm-top"><span>{html.escape(left)}</span><span>{html.escape(right)}</span></div>'


def panel(title: str, tag: str, svg: str, readouts: list[tuple[str, str]]) -> str:
    r = "".join(f"<div><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></div>" for k, v in readouts)
    return (f'<div class="hm-panel"><div class="hm-panel-h"><b>{html.escape(title)}</b><span>{html.escape(tag)}</span></div>{svg}'
            f'<div class="hm-readouts">{r}</div></div>')


def strip(items: list[tuple[str, str]]) -> str:
    return '<div class="hm-strip">' + "".join(f"<div><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></div>" for k, v in items) + "</div>"


def section(n: str, title: str, note: str = "") -> str:
    return f'<div class="hm-sec"><span class="n">{n}</span><h2>{html.escape(title)}</h2>' + (f"<p>{html.escape(note)}</p>" if note else "") + "</div>"


def grid4(cells: list[tuple[str, str, str]]) -> str:
    return '<div class="hm-grid4">' + "".join(f'<div><span class="k">{k}</span><h4>{html.escape(t)}</h4><p>{html.escape(d)}</p></div>' for k, t, d in cells) + "</div>"


def two_col(left_title: str, left: list[str], right_title: str, right: list[str]) -> str:
    li = lambda xs: "<ul style='padding-left:1.1rem;margin:0'>" + "".join(f"<li>{html.escape(x)}</li>" for x in xs) + "</ul>"
    return f'<div class="hm-two"><div><h4>{html.escape(left_title)}</h4>{li(left)}</div><div><h4>{html.escape(right_title)}</h4>{li(right)}</div></div>'
