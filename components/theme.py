"""Shared visual system for the QureAI research console.

The helpers in this module are intentionally presentation-only. They do not
read or mutate model/data state, which keeps the research workflows unchanged
while giving every view the same visual language.
"""

from contextlib import contextmanager

import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "navy": "#07111f",
    "navy_2": "#0b1a2b",
    "panel": "#0f2235",
    "panel_2": "#122b42",
    "line": "#1c4058",
    "text": "#e8f2f7",
    "muted": "#8ca7b7",
    "cyan": "#43d9df",
    "teal": "#29b7a8",
    "green": "#56d48b",
    "amber": "#f2bd62",
    "red": "#f17b83",
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="IBM Plex Mono, Consolas, monospace", color=COLORS["text"], size=12),
        colorway=[COLORS["cyan"], COLORS["teal"], "#8aa8ff", COLORS["amber"], COLORS["green"], COLORS["red"]],
        margin=dict(l=44, r=24, t=48, b=42),
        hoverlabel=dict(
            bgcolor=COLORS["panel_2"],
            bordercolor=COLORS["line"],
            font=dict(color=COLORS["text"]),
        ),
        xaxis=dict(
            gridcolor="#1a384d",
            zerolinecolor="#29556b",
            linecolor=COLORS["line"],
            tickfont=dict(color=COLORS["muted"], size=11),
            title_font=dict(color=COLORS["muted"], size=11),
        ),
        yaxis=dict(
            gridcolor="#1a384d",
            zerolinecolor="#29556b",
            linecolor=COLORS["line"],
            tickfont=dict(color=COLORS["muted"], size=11),
            title_font=dict(color=COLORS["muted"], size=11),
        ),
        legend=dict(font=dict(color=COLORS["muted"]), bgcolor="rgba(0,0,0,0)"),
    )
)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --q-navy: #07111f;
  --q-navy-2: #0b1a2b;
  --q-panel: #0f2235;
  --q-panel-2: #122b42;
  --q-line: #1c4058;
  --q-text: #e8f2f7;
  --q-muted: #8ca7b7;
  --q-cyan: #43d9df;
  --q-teal: #29b7a8;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--q-navy) !important;
  color: var(--q-text);
}
[data-testid="stAppViewContainer"] {
  background-image:
    linear-gradient(rgba(67, 217, 223, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(67, 217, 223, 0.035) 1px, transparent 1px);
  background-size: 32px 32px;
}
[data-testid="stHeader"] { background: rgba(7, 17, 31, 0.82) !important; }
[data-testid="stMainBlockContainer"] { max-width: 1480px; padding: 2.15rem 2.25rem 4rem; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #091827 0%, #07111f 100%) !important;
  border-right: 1px solid var(--q-line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--q-text); }

h1, h2, h3, h4 { font-family: "Space Grotesk", sans-serif !important; letter-spacing: -0.02em; }
h1 { font-size: clamp(2rem, 4vw, 3.15rem) !important; font-weight: 700 !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.15rem !important; }
p, li, label, [data-testid="stCaptionContainer"] { font-family: "Space Grotesk", sans-serif; }
code, pre, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-family: "IBM Plex Mono", Consolas, monospace !important;
}
[data-testid="stCaptionContainer"] { color: var(--q-muted) !important; }
[data-testid="stCaptionContainer"] { font-size: 0.78rem; min-height: 1.1rem; }

.q-brand {
  padding: 0.65rem 0.15rem 1.1rem;
  border-bottom: 1px solid var(--q-line);
  margin-bottom: 1rem;
}
.q-brand-mark { color: var(--q-cyan); font: 600 0.72rem "IBM Plex Mono", monospace; letter-spacing: 0.18em; }
.q-brand-name { color: var(--q-text); font: 700 1.65rem "Space Grotesk", sans-serif; margin: 0.2rem 0; }
.q-brand-meta { color: var(--q-muted); font: 400 0.72rem "IBM Plex Mono", monospace; }
.q-topbar {
  align-items: center; border-bottom: 1px solid var(--q-line); display: flex;
  justify-content: space-between; margin: -0.85rem 0 2rem; padding: 0 0 0.9rem;
}
.q-kicker { color: var(--q-cyan); font: 600 0.68rem "IBM Plex Mono", monospace; letter-spacing: 0.16em; text-transform: uppercase; }
.q-section { border-left: 2px solid var(--q-cyan); margin: 1.65rem 0 0.8rem; padding-left: 0.75rem; }
.q-section h2, .q-section h3 { margin: 0 !important; }
.q-section p { color: var(--q-muted); font: 0.76rem "IBM Plex Mono", monospace; margin: 0.25rem 0 0; }
.q-panel {
  background: rgba(15, 34, 53, 0.88); border: 1px solid var(--q-line);
  border-radius: 8px; box-shadow: 0 16px 36px rgba(0, 0, 0, 0.12);
  padding: 1.1rem 1.15rem; margin: 0.55rem 0 1rem;
}
.q-panel-title { color: var(--q-text); font: 600 0.76rem "IBM Plex Mono", monospace; letter-spacing: 0.08em; text-transform: uppercase; }
.q-hero {
  background:
    radial-gradient(circle at 88% 15%, rgba(67, 217, 223, 0.16), transparent 30%),
    linear-gradient(135deg, rgba(18, 43, 66, 0.98), rgba(9, 24, 39, 0.98));
  border: 1px solid rgba(67, 217, 223, 0.28); border-radius: 14px;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18); margin: 0 0 1.3rem;
  overflow: hidden; padding: 1.5rem 1.6rem; position: relative;
}
.q-hero::after {
  border: 1px solid rgba(67, 217, 223, 0.14); border-radius: 50%;
  content: ""; height: 240px; position: absolute; right: -80px; top: -100px; width: 240px;
}
.q-hero-kicker { color: var(--q-cyan); font: 600 0.68rem "IBM Plex Mono", monospace; letter-spacing: 0.18em; text-transform: uppercase; }
.q-hero-title { color: var(--q-text); font: 700 clamp(2.1rem, 4vw, 3.8rem) "Space Grotesk", sans-serif; letter-spacing: -0.055em; line-height: 0.98; margin: 0.45rem 0 0.75rem; max-width: 700px; }
.q-hero-copy { color: #b9ced8; font: 400 0.96rem/1.55 "Space Grotesk", sans-serif; max-width: 700px; }
.q-hero-chip { color: #b7f5f0; display: inline-block; font: 600 0.68rem "IBM Plex Mono", monospace; margin-top: 0.8rem; }
.q-divider { background: linear-gradient(90deg, var(--q-cyan), transparent); height: 1px; margin: 1.6rem 0; opacity: 0.5; }
.q-badge {
  border: 1px solid var(--q-line); border-radius: 999px; display: inline-flex;
  font: 600 0.68rem "IBM Plex Mono", monospace; letter-spacing: 0.04em; padding: 0.26rem 0.58rem;
}
.q-badge.ok { background: rgba(86, 212, 139, 0.12); border-color: rgba(86, 212, 139, 0.5); color: #80e6a8; }
.q-badge.warn { background: rgba(242, 189, 98, 0.12); border-color: rgba(242, 189, 98, 0.5); color: #ffd68b; }
.q-badge.error { background: rgba(241, 123, 131, 0.12); border-color: rgba(241, 123, 131, 0.5); color: #ffabb1; }
.q-badge.info { background: rgba(67, 217, 223, 0.12); border-color: rgba(67, 217, 223, 0.5); color: #8ceff1; }
.q-disclaimer {
  background: rgba(242, 189, 98, 0.08); border: 1px solid rgba(242, 189, 98, 0.42);
  border-left: 3px solid #f2bd62; color: #f8d99e; font: 0.76rem/1.45 "IBM Plex Mono", monospace;
  margin: 1.2rem 0; padding: 0.8rem 0.95rem;
}
.q-pipeline { color: var(--q-muted); font: 500 0.72rem "IBM Plex Mono", monospace; letter-spacing: 0.01em; }
.q-pipeline strong { color: var(--q-cyan); }

div[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(15, 34, 53, 0.78); border-color: var(--q-line) !important; border-radius: 3px !important;
}
[data-testid="stMetricLabel"] { color: var(--q-muted) !important; font-size: 0.7rem !important; letter-spacing: 0.04em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--q-cyan) !important; font-size: 1.4rem !important; }
button, [data-baseweb="select"] > div, input, textarea {
  border-radius: 6px !important; border-color: var(--q-line) !important;
}
button[kind="primary"] {
  background: linear-gradient(135deg, var(--q-teal), #43d9df) !important;
  box-shadow: 0 8px 20px rgba(41, 183, 168, 0.18); color: #041018 !important;
  font-weight: 700 !important;
}
button:hover { border-color: var(--q-cyan) !important; color: var(--q-cyan) !important; transform: translateY(-1px); }
[data-testid="stAlert"] { background: rgba(15, 34, 53, 0.92); border-radius: 3px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--q-line); border-radius: 3px; overflow: hidden; }
pre, [data-testid="stCodeBlock"] { background: #06101c !important; border: 1px solid var(--q-line); border-radius: 3px !important; }
hr { border-color: var(--q-line) !important; }
@media (max-width: 800px) {
  [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 3rem; }
  .q-topbar { align-items: flex-start; gap: 0.7rem; }
  .q-hero { padding: 1.2rem; }
}
@media (max-width: 480px) {
  [data-testid="stMainBlockContainer"] { padding: 1rem 0.7rem 2.5rem; }
  .q-topbar { flex-direction: column; align-items: flex-start; gap: 0.45rem; }
  .q-topbar > div:last-child { text-align: left !important; }
  .q-hero { padding: 0.95rem; }
  .q-hero-title { font-size: 2rem; }
  .q-hero-copy { font-size: 0.88rem; }
}
</style>
"""


def apply_theme():
    """Inject the visual system once per Streamlit run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand():
    st.markdown(
        """
        <div class="q-brand">
          <div class="q-brand-mark">QURE / RESEARCH CONSOLE</div>
          <div class="q-brand-name">QureAI</div>
          <div class="q-brand-meta">hybrid ML · biomedical intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_status(page_name):
    dataset = st.session_state.get("dataset")
    experiment = st.session_state.get("experiment_result")
    dataset_state = f"{len(dataset)} rows loaded" if dataset is not None else "dataset unavailable"
    experiment_state = "experiment artifact ready" if experiment else "no experiment artifact"
    session_state = "SESSION ACTIVE" if dataset is not None or experiment else "SESSION EMPTY"
    session_badge = "ok" if dataset is not None or experiment else "info"
    st.markdown(
        f"""
        <div class="q-topbar">
          <div><span class="q-kicker">active view</span><br><strong>{page_name}</strong></div>
          <div style="text-align:right"><span class="q-badge {session_badge}"><span aria-hidden="true">●</span> {session_state}</span><br>
          <span class="q-brand-meta">{dataset_state} · {experiment_state}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, description=None, icon=None):
    icon_text = f"{icon} " if icon else ""
    detail = f"<p>{description}</p>" if description else ""
    st.markdown(
        f'<div class="q-section"><h2>{icon_text}{title}</h2>{detail}</div>',
        unsafe_allow_html=True,
    )


# `hero()` is reserved for the landing page; interior views use section headers
# so dense workflows remain scannable and task-focused.
def hero(title, copy, kicker="HYBRID QUANTUM · BIOMEDICAL RESEARCH", chip=None):
    chip_markup = f'<div class="q-hero-chip"><span aria-hidden="true">✦</span> {chip}</div>' if chip else ""
    st.markdown(
        f"""
        <div class="q-hero">
          <div class="q-hero-kicker">{kicker}</div>
          <div class="q-hero-title">{title}</div>
          <div class="q-hero-copy">{copy}</div>
          {chip_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label, status="info"):
    """Render a compact status badge with an accessible text label."""
    safe_status = status if status in {"ok", "warn", "error", "info"} else "info"
    st.markdown(f'<span class="q-badge {safe_status}">{label}</span>', unsafe_allow_html=True)


@contextmanager
def panel(title=None):
    """Group a small view section in the shared bordered panel treatment."""
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="q-panel-title">{title}</div>', unsafe_allow_html=True)
        yield


def metric_strip(items):
    """Render metrics as a consistent, compact horizontal strip."""
    for start in range(0, len(items), 2):
        row = items[start : start + 2]
        cols = st.columns(len(row))
        for col, item in zip(cols, row):
            label, value = item[:2]
            help_text = item[2] if len(item) > 2 else None
            with col:
                st.metric(label, value, help=help_text, border=True)


def dataframe(frame, **kwargs):
    """Shared dataframe defaults; callers can still override any option."""
    options = {"width": "stretch", "hide_index": True}
    options.update(kwargs)
    st.dataframe(frame, **options)


def disclaimer(text):
    st.markdown(f'<div class="q-disclaimer">RESEARCH BOUNDARY · {text}</div>', unsafe_allow_html=True)


def style_figure(fig):
    """Apply the shared Plotly template without changing traces or values."""
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig
