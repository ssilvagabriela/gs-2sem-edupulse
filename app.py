"""Radar Docente — painel interativo com dados reais do projeto FIAP 2026.2.

Execute com: streamlit run app.py
Os CSVs são saídas do notebook RadarDocente.ipynb, sem retreinamento no painel.
"""
from __future__ import annotations

import html
import base64
import hashlib
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.alert_explanation import explain_prediction
from src.ai_summary import DEFAULT_MODEL, SUMMARY_VERSION, SummaryError, fallback_summary, generate_summary, indicator_evidence

ROOT = Path(__file__).resolve().parent
FILE_ESCOLAS = ROOT / "data" / "processed" / "radar_escolas.csv"
FILE_MUNICIPIOS = ROOT / "data" / "processed" / "radar_municipios_2026.csv"

NAVY = "#1b003f"
INK = "#191970"
MUTED = "#8190a4"
RED = "#ee5b62"
AMBER = "#edaa54"
TEAL = "#6495ed"
GREEN = "#3faf9b"
RISK_COLORS = {"alto": RED, "médio": AMBER, "baixo": GREEN}
RISK_ORDER = {"alto": 0, "médio": 1, "baixo": 2}

st.set_page_config(
    page_title="EduPulse | Inteligência que antecipa. Educação que evolui",
    page_icon="E",
    layout="wide",
)

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f6f4fb; }
[data-testid="stMainBlockContainer"] {padding-top: 3.5rem; padding-bottom: 3.5rem; max-width: 1800px;}
[data-testid="stHeader"] {background: #f6f4fb;}
.st-key-panel_filters {background:transparent; padding:24px 20px; border-radius:16px;}
/* Stretch only the filter column to the height of its neighboring page content. */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-panel_filters) {align-items:stretch;}
[data-testid="stColumn"]:has(.st-key-panel_filters) {background:linear-gradient(180deg,#191970,#1b003f); border-radius:16px; align-self:stretch; border:1px solid #4b0082;}
.st-key-panel_filters [data-testid="stWidgetLabel"] p,
.st-key-panel_filters label {color:#edf4ff !important;}
.st-key-panel_filters [data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-panel_filters [data-testid="stSelectbox"] [data-baseweb="select"] * {color:#191970 !important;}
.brand {display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; padding:0 0 8px;}
/* Display the supplied logo's visible area without its transparent margins. */
.brand-logo-frame {position:relative; width:260px; height:64px; overflow:hidden; flex-shrink:0;}
.brand-logo-frame > img.brand-logo {position:absolute; width:284px !important; max-width:none !important; height:213px !important; left:-15px; top:-77px; display:block;}
.brand-caption {font-size:12px; color:#696282; margin:0; padding-left:24px; border-left:1px solid #ded7ee; line-height:1.5;}
@media(max-width:640px){.brand{gap:8px;} .brand-caption{padding-left:0; border-left:0; width:100%;}}
.st-key-top_navigation {border-bottom:1px solid #ded7ee; padding-bottom:4px; margin-bottom:12px;}
.st-key-top_navigation [role="radiogroup"] {gap:4px 12px; flex-wrap:wrap;}
.st-key-top_navigation [data-baseweb="radio"] {margin:0; padding:10px 14px; border-bottom:3px solid transparent; border-radius:8px 8px 0 0;}
.st-key-top_navigation [data-baseweb="radio"] > div:first-child {position:absolute; opacity:0; width:1px; height:1px; overflow:hidden;}
.st-key-top_navigation [data-baseweb="radio"] p {font-size:14px; font-weight:600; color:#615a78;}
.st-key-top_navigation [data-baseweb="radio"]:hover {background:#efebf8;}
.st-key-top_navigation [data-baseweb="radio"]:has(input:checked) {border-bottom-color:#4b0082; background:#e6e6fa;}
.st-key-top_navigation [data-baseweb="radio"]:has(input:checked) p {color:#191970;}
.st-key-top_navigation [data-baseweb="radio"]:focus-within {outline:2px solid #6495ed; outline-offset:2px;}
.sidebar-kicker {text-transform: uppercase; letter-spacing:1.5px; font-size:10px; font-weight:800; color:#c8c3e9; margin: 8px 0 4px;}
.eyebrow {font-size:10px; font-weight:800; letter-spacing:1.6px; text-transform:uppercase; color:#4b0082; margin-bottom:10px;}
.headline {font-size:34px; font-weight:800; color:#1b003f; letter-spacing:-1.35px; margin:0; line-height:1.25;}
.subheadline {font-size:14px; color:#7d8a9a; margin-top:7px; margin-bottom:23px;}
.live-chip {display:inline-block; padding:8px 13px; border-radius:40px; color:#4b0082; background:#e6e6fa; border:1px solid #d6cdeb; font-size:10px; font-weight:800; letter-spacing:.8px; white-space:nowrap;}
.kpi-card {background:white; border:1px solid #e5dff0; border-radius:16px; padding:20px 20px 16px; box-shadow:0 5px 25px rgba(27,44,68,.035); min-height:128px; position:relative; overflow:hidden;}
.kpi-card:before {content:''; position:absolute; left:0; top:0; height:100%; width:4px; background:var(--kpi-accent);}
.kpi-top {font-size:12px; font-weight:700; color:#75849a; display:flex; align-items:center; justify-content:space-between;}
.kpi-value {font-size:29px; font-weight:800; color:#191970; margin-top:11px; letter-spacing:-.9px; line-height:1;}
.kpi-foot {font-size:11px; color:#687b91; margin-top:12px; line-height:1.5;}
.block-heading {font-size:17px; font-weight:800; color:#191970; margin:0 0 8px; line-height:1.4;}
.block-sub {font-size:12px; color:#6b7d92; margin-bottom:14px; line-height:1.6;}
.section-space {height:16px;}
[class*="st-key-visual_panel_"] {padding:22px; background:#ffffff; border-radius:14px;}
[class*="st-key-visual_panel_"] [data-testid="stMarkdownContainer"] p {line-height:1.7;}
[class*="st-key-visual_panel_"] [data-testid="stMarkdownContainer"] li {margin-bottom:8px;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color: #e7ecf4 !important; border-radius:16px !important; background: #fff; box-shadow:0 5px 24px rgba(29,42,60,.025);}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] > div {box-shadow:none;}
.stButton button[kind="primary"] {background-color:#4b0082; border-color:#4b0082; border-radius:10px;}
.stButton button, [data-testid="stDownloadButton"] button {border-radius:10px;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow:hidden;}
.status-pill {border-radius:30px; padding:7px 13px; display:inline-block; font-size:12px; font-weight:800; background:var(--pill-bg); color:var(--pill-color);}
.explain {border-radius:13px; padding:15px 17px; background:#f2f7fb; color:#615a78; font-size:12px; line-height:1.65; border-left:3px solid #61acb8;}
.quiet-note, .footnote {font-size:11px; color:#68768a; line-height:1.6; font-weight:400; margin:10px 0 0; padding:4px 0 6px;}
.footnote {padding:8px 0;}
.stTabs [data-baseweb="tab"] {font-weight: 700;}
@media(max-width:900px){.headline{font-size:27px;} .kpi-card{min-height:110px; padding:16px;} .kpi-value{font-size:24px;} [class*="st-key-visual_panel_"]{padding:16px;} .brand-caption{font-size:11px;}}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_float(n: float, digits: int = 2) -> str:
    if pd.isna(n):
        return "—"
    return f"{n:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_percent(n: float, digits: int = 1) -> str:
    return "—" if pd.isna(n) else f"{fmt_float(n, digits)}%"


def norm(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn").casefold()


@st.cache_data(show_spinner="Carregando dados do EduPulse...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FILE_ESCOLAS.exists() or not FILE_MUNICIPIOS.exists():
        raise FileNotFoundError("As bases radar_escolas.csv e radar_municipios_2026.csv devem estar na pasta data/processed/.")
    e = pd.read_csv(FILE_ESCOLAS, dtype={"co_escola": "string", "co_municipio": "string"}, low_memory=False)
    m = pd.read_csv(FILE_MUNICIPIOS, dtype={"co_municipio": "string"})
    e["ano"] = pd.to_numeric(e["ano"], errors="coerce").astype("Int64")
    e["classe_risco"] = e["classe_risco"].astype("string")
    e["rede"] = e["rede"].astype("string")
    return e, m


try:
    escolas, municipios = load_data()
except (OSError, ValueError) as exc:
    st.error(f"Não foi possível carregar as bases: {exc}")
    st.stop()

pred_all = escolas.loc[escolas["ano"].eq(2026)].copy()
if pred_all.empty:
    st.error("Não foram encontradas previsões para 2026.")
    st.stop()

logo_data = base64.b64encode((ROOT / "assets" / "edupulse-logo.png").read_bytes()).decode("ascii")
st.markdown(f"""<div class="brand"><div class="brand-logo-frame"><img class="brand-logo" src="data:image/png;base64,{logo_data}" alt="EduPulse"></div><div class="brand-caption">Inteligência que antecipa. Educação que evolui</div></div>""", unsafe_allow_html=True)
with st.container(key="top_navigation"):
    pagina = st.radio(
        "Navegação principal",
        ["Visão geral", "Explorar escolas", "Mapa de municípios", "Entenda os indicadores"],
        horizontal=True, label_visibility="collapsed", index=0, key="pagina",
    )

filters_column, page_content = st.columns([1, 4.5], gap="large")
with filters_column:
    with st.container(key="panel_filters"):
        st.markdown('<div class="sidebar-kicker">Filtros do painel</div>', unsafe_allow_html=True)
        mun_options = ["Todos os municípios"] + sorted(pred_all["municipio"].dropna().unique().tolist())
        filtro_mun = st.selectbox("Município", mun_options)
        redes = ["Todas as redes"] + sorted(pred_all["rede"].dropna().unique().tolist())
        filtro_rede = st.selectbox("Rede de ensino", redes)
        locs = ["Todas as localizações"] + sorted(pred_all["localizacao"].dropna().unique().tolist())
        filtro_loc = st.selectbox("Localização", locs)

sel = pred_all.copy()
if filtro_mun != "Todos os municípios":
    sel = sel.loc[sel["municipio"].eq(filtro_mun)]
if filtro_rede != "Todas as redes":
    sel = sel.loc[sel["rede"].eq(filtro_rede)]
if filtro_loc != "Todas as localizações":
    sel = sel.loc[sel["localizacao"].eq(filtro_loc)]

school_ids = set(sel["co_escola"].dropna())
historico = escolas.loc[escolas["co_escola"].isin(school_ids) & escolas["ano"].between(2022, 2025)].copy()


def header(kicker: str, title: str, subtitle: str):
    a, b = st.columns([4, 1], vertical_alignment="center")
    with a:
        st.markdown(
            f'<div class="eyebrow">{html.escape(kicker)}</div>'
            f'<div class="headline">{html.escape(title)}</div>'
            f'<div class="subheadline">{html.escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown('<div style="text-align:right"><span class="live-chip">PROJEÇÃO 2026</span></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, foot: str, accent: str):
    safe = [html.escape(str(t)) for t in (label, value, foot)]
    st.markdown(
        f'<div class="kpi-card" style="--kpi-accent:{accent};">'
        f'<div class="kpi-top"><span>{safe[0]}</span></div>'
        f'<div class="kpi-value">{safe[1]}</div><div class="kpi-foot">{safe[2]}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="block-heading">{html.escape(title)}</div><div class="block-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def note(text: str):
    """Render supporting notes with consistent, compact typography."""
    st.markdown(f'<div class="quiet-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def section_gap():
    """Keep a consistent separation between dashboard rows."""
    st.markdown('<div class="section-space" aria-hidden="true"></div>', unsafe_allow_html=True)


def style_plot(fig, height=310):
    fig.update_layout(
        height=height, margin=dict(l=20, r=24, t=20, b=36),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family="DM Sans, Segoe UI, sans-serif", color=INK, size=12),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def municipality_summary(p: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    if p.empty:
        return pd.DataFrame(columns=["co_municipio", "municipio", "escolas", "risco_alto", "risco_medio", "risco_baixo", "alertas", "pct_alertas", "ird_medio_2025", "latitude", "longitude"])
    g = p.groupby(["co_municipio", "municipio"], dropna=False).agg(
        escolas=("co_escola", "size"),
        risco_alto=("classe_risco", lambda s: s.eq("alto").sum()),
        risco_medio=("classe_risco", lambda s: s.eq("médio").sum()),
        risco_baixo=("classe_risco", lambda s: s.eq("baixo").sum()),
    ).reset_index()
    g["alertas"] = g["risco_alto"] + g["risco_medio"]
    g["pct_alertas"] = 100 * g["alertas"] / g["escolas"]
    h25 = h.loc[h["ano"].eq(2025)].groupby("co_municipio")["ird"].mean().rename("ird_medio_2025")
    g = g.merge(h25, on="co_municipio", how="left")
    coords = municipios[["co_municipio", "latitude", "longitude"]].drop_duplicates(subset=["co_municipio"])
    return g.merge(coords, on="co_municipio", how="left")


mun_sel = municipality_summary(sel, historico)


def show_map(df: pd.DataFrame, height: int = 410):
    data = df.dropna(subset=["latitude", "longitude"]).copy()
    if data.empty:
        st.info("Não há municípios com coordenadas nos filtros selecionados.")
        return
    data["raio"] = np.clip(1800 + np.sqrt(data["alertas"]) * 2100, 2200, 35000)
    data["cor"] = data["alertas"].apply(lambda a: [239, 91, 100, 187] if a > 0 else [125, 151, 172, 95])
    data["total_alertas"] = data["alertas"].astype(int)
    data["total_escolas"] = data["escolas"].astype(int)
    data["pct_formatado"] = data["pct_alertas"].map(lambda v: fmt_percent(v))
    view = pdk.ViewState(
        latitude=float(data["latitude"].mean()) if len(data) <= 2 else -22.8,
        longitude=float(data["longitude"].mean()) if len(data) <= 2 else -48.3,
        zoom=9.5 if len(data) == 1 else (7 if len(data) <= 4 else 6.2),
        pitch=0,
    )
    layer = pdk.Layer(
        "ScatterplotLayer", data=data, get_position="[longitude, latitude]",
        get_fill_color="cor", get_radius="raio", radius_min_pixels=3, radius_max_pixels=27,
        pickable=True, stroked=True, get_line_color=[255, 255, 255, 205], line_width_min_pixels=0.5,
    )
    deck = pdk.Deck(
        layers=[layer], initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>{municipio}</b><br/>Escolas: {total_escolas}<br/>Alertas (alto + médio): {total_alertas}<br/>Percentual: {pct_formatado}", "style": {"backgroundColor": NAVY, "color": "white", "fontSize": "12px", "borderRadius": "9px"}},
    )
    st.pydeck_chart(deck, use_container_width=True, height=height)


def risk_donut(p: pd.DataFrame):
    counts = p["classe_risco"].value_counts()
    labels = ["Alto", "Médio", "Baixo"]
    vals = [int(counts.get(s, 0)) for s in ["alto", "médio", "baixo"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=.73, sort=False,
        marker=dict(colors=[RED, AMBER, GREEN], line=dict(color="white", width=3)),
        textinfo="none", hovertemplate="%{label}: %{value:,} escolas (%{percent})<extra></extra>",
    ))
    fig.add_annotation(x=.5, y=.5, text=f"<b>{fmt_int(sum(vals))}</b><br><span style='font-size:11px;color:#8291a3'>escolas</span>", showarrow=False, font=dict(size=23, color=INK))
    style_plot(fig, 340)
    fig.update_layout(
        margin=dict(l=24, r=24, t=20, b=60), showlegend=True,
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.08, yanchor="top"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def historical_plot(df: pd.DataFrame, school: bool = False):
    if df.empty or df["ird"].notna().sum() == 0:
        st.info("Sem IRD observado para o recorte selecionado.")
        return
    d = df.groupby("ano", as_index=False).agg(ird=("ird", "mean"), escolas=("ird", "count"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["ano"], y=d["ird"], mode="lines+markers", name="IRD observado" if school else "IRD médio observado",
        fill="tozeroy" if not school else None,
        fillcolor="rgba(100,149,237,.12)", line=dict(color=TEAL, width=3.5),
        marker=dict(color="white", size=10, line=dict(color=TEAL, width=3)),
        customdata=np.array(d["escolas"]).reshape(-1, 1),
        hovertemplate="Ano: %{x}<br>IRD: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=2, line_dash="dash", line_color=RED, line_width=1.7,
                  annotation_text="Limiar de baixa regularidade (IRD < 2)", annotation_position="top left",
                  annotation_font_color=RED, annotation_font_size=10)
    fig.update_yaxes(range=[0, 5], dtick=1, gridcolor="#ecf0f5", zeroline=False, title=None)
    fig.update_xaxes(tickmode="array", tickvals=[2022, 2023, 2024, 2025],
                     range=[2021.85, 2025.15], gridcolor="#f3f6f9", title=None)
    style_plot(fig, 320 if school else 340)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def overview():
    header("Inteligência educacional · São Paulo", "Panorama da regularidade docente", "Indicadores históricos do INEP e alertas preditivos para apoiar decisões nas redes públicas.")
    if sel.empty:
        st.warning("Nenhuma escola encontrada com os filtros escolhidos. Altere os filtros na barra lateral.")
        return
    total = len(sel)
    alto = int(sel["classe_risco"].eq("alto").sum())
    medio = int(sel["classe_risco"].eq("médio").sum())
    alerta = alto + medio
    k = st.columns(4)
    with k[0]: kpi("Escolas monitoradas", fmt_int(total), "Com classificação para 2026", TEAL)
    with k[1]: kpi("Risco alto", fmt_int(alto), fmt_percent(100 * alto / total) + " das escolas", RED)
    with k[2]: kpi("Risco médio", fmt_int(medio), fmt_percent(100 * medio / total) + " das escolas", AMBER)
    with k[3]: kpi("Escolas em alerta", fmt_percent(100 * alerta / total), f"{fmt_int(alerta)} em risco alto ou médio", GREEN)

    section_gap()
    a, b = st.columns([1.5, 1], gap="medium")
    with a:
        with st.container(border=True, key="visual_panel_01"):
            section("Evolução histórica do IRD", "Média anual observada (2022–2025) das escolas presentes no recorte de 2026.")
            historical_plot(historico)
    with b:
        with st.container(border=True, key="visual_panel_02"):
            section("Classificação prevista para 2026", "Distribuição das escolas por faixa de atenção.")
            risk_donut(sel)

    section_gap()
    a, b = st.columns([1.5, 1], gap="medium")
    with a:
        with st.container(border=True, key="visual_panel_03"):
            section("Mapa de alertas por município", "As bolhas mostram onde se concentram escolas em risco alto ou médio.")
            show_map(mun_sel, height=378)
            note("Bolhas na sede municipal; o tamanho indica a quantidade de alertas.")
    with b:
        with st.container(border=True, key="visual_panel_04"):
            section("Municípios com mais alertas", "Quantidade de escolas com risco alto ou médio previsto.")
            top = mun_sel.nlargest(10, "alertas").sort_values("alertas")
            if not top.empty:
                fig = go.Figure()
                fig.add_bar(y=top["municipio"], x=top["risco_alto"], name="Alto", orientation="h", marker_color=RED)
                fig.add_bar(y=top["municipio"], x=top["risco_medio"], name="Médio", orientation="h", marker_color=AMBER)
                fig.update_layout(barmode="stack", xaxis_title="Nº de escolas", showlegend=True)
                fig.update_xaxes(gridcolor="#edf1f7", zeroline=False)
                fig.update_yaxes(showgrid=False, automargin=True)
                style_plot(fig, 389)
                fig.update_layout(margin=dict(l=5, r=10, t=30, b=9), legend=dict(orientation="h", x=0, y=1.09))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section_gap()
    with st.container(border=True, key="visual_panel_05"):
        section("Escolas que merecem atenção", "Priorização pelo risco e pela pontuação previstos para 2026.")
        show_school_table(sel, historico, n=25)
    note("O IRD mede a continuidade do vínculo docente. Alertas não confirmam falta de professores; a pontuação não é uma probabilidade.")


def school_table_data(p: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    obs25 = (h.loc[h["ano"].eq(2025), ["co_escola", "ird"]]
             .drop_duplicates("co_escola").rename(columns={"ird": "ird_2025"}))
    d = p[["co_escola", "nome", "municipio", "rede", "localizacao", "classe_risco", "prob_risco"]].merge(obs25, on="co_escola", how="left")
    d["ordem"] = d["classe_risco"].map(RISK_ORDER).fillna(9)
    return d.sort_values(["ordem", "prob_risco"], ascending=[True, False])


def show_school_table(p: pd.DataFrame, h: pd.DataFrame, n: int | None = None):
    d = school_table_data(p, h)
    if n is not None:
        d = d.head(n)
    d = d.rename(columns={"co_escola": "Código INEP", "nome": "Escola", "municipio": "Município", "rede": "Rede", "localizacao": "Localização", "classe_risco": "Risco 2026", "prob_risco": "Escore do modelo", "ird_2025": "IRD 2025"})
    d = d.drop(columns=["ordem"])
    st.dataframe(d, use_container_width=True, hide_index=True, height=min(575, 52 + max(1, len(d)) * 34), column_config={
        "Escore do modelo": st.column_config.ProgressColumn("Escore do modelo", min_value=0, max_value=1, format="%.3f", help="Escore da Random Forest; não é uma probabilidade calibrada."),
        "IRD 2025": st.column_config.NumberColumn("IRD 2025", format="%.2f"),
    })


def school_history_table(history: pd.DataFrame) -> pd.DataFrame:
    """Prepare readable labels and values without changing the source data."""
    labels = {
        "ano": "Ano",
        "ird": "Regularidade da equipe docente (0 a 5)",
        "afd_g1_fun": "Professores com formação adequada · Ensino Fundamental (%)",
        "afd_g1_med": "Professores com formação adequada · Ensino Médio (%)",
        "ied_alto_fun": "Professores com esforço de trabalho alto · Ensino Fundamental (%)",
        "ied_alto_med": "Professores com esforço de trabalho alto · Ensino Médio (%)",
        "atu_fun": "Média de alunos por turma · Ensino Fundamental",
        "atu_med": "Média de alunos por turma · Ensino Médio",
        "reprov_fun": "Alunos reprovados · Ensino Fundamental (%)",
        "reprov_med": "Alunos reprovados · Ensino Médio (%)",
        "aband_fun": "Alunos que abandonaram os estudos · Ensino Fundamental (%)",
        "aband_med": "Alunos que abandonaram os estudos · Ensino Médio (%)",
        "icg": "Nível de complexidade da gestão escolar (1 a 6)",
        "pct_baixa_mun": "Escolas com baixa regularidade no município (%)",
        "classe_risco": "Risco previsto para o ano seguinte",
        "prob_risco": "Pontuação de risco do modelo (0 a 1)",
        "tem_fundamental": "Oferece Ensino Fundamental?",
        "tem_medio": "Oferece Ensino Médio?",
        "nome": "Nome da escola",
        "municipio": "Município",
        "rede": "Rede de ensino",
        "localizacao": "Área urbana ou rural",
        "co_escola": "Código da escola no INEP",
        "co_municipio": "Código do município no IBGE",
        "tipo": "Tipo de registro",
    }
    data = history.sort_values("ano").copy()
    for col in ("tem_fundamental", "tem_medio"):
        data[col] = data[col].map({1: "Sim", 0: "Não"}).fillna("Não informado")
    data["tipo"] = data["tipo"].replace({
        "observado": "Dado observado",
        "observado+previsto": "Dado observado com previsão associada",
        "previsto": "Previsão",
    })
    data["classe_risco"] = data["classe_risco"].str.capitalize().fillna("Não disponível")
    numeric = ["ird", "pct_baixa_mun", "afd_g1_fun", "afd_g1_med",
               "ied_alto_fun", "ied_alto_med", "atu_fun", "atu_med",
               "reprov_fun", "reprov_med", "aband_fun", "aband_med", "icg", "prob_risco"]
    for col in numeric:
        data[col] = data[col].map(lambda value: "Não disponível" if pd.isna(value)
                                  else fmt_float(value, 3 if col == "prob_risco" else 1))
    return data[list(labels)].rename(columns=labels)


def show_followup_summary(school: pd.Series, history: pd.DataFrame):
    section("Entenda o alerta desta escola", "O que foi previsto, quais dados foram observados e o que verificar com a gestão.")
    history_columns = ["ano", "ird", "afd_g1_fun", "afd_g1_med", "ied_alto_fun",
                       "ied_alto_med", "atu_fun", "atu_med", "reprov_fun", "reprov_med",
                       "aband_fun", "aband_med", "icg"]
    context = {
        "versao_resumo": SUMMARY_VERSION,
        "escola": {field: str(school[field]) for field in ("nome", "municipio", "rede", "localizacao")},
        "filtros": {"municipio": filtro_mun, "rede": filtro_rede, "localizacao": filtro_loc},
        "recorte": {"escolas": len(sel), "risco_alto": int(sel["classe_risco"].eq("alto").sum()),
                    "risco_medio": int(sel["classe_risco"].eq("médio").sum())},
        "significado_dos_indicadores": {
            "afd_g1_fun": "Professores com formação adequada no Ensino Fundamental (%)",
            "afd_g1_med": "Professores com formação adequada no Ensino Médio (%)",
            "ied_alto_fun": "Professores com esforço de trabalho alto no Ensino Fundamental (%)",
            "ied_alto_med": "Professores com esforço de trabalho alto no Ensino Médio (%)",
            "atu_fun": "Média de alunos por turma no Ensino Fundamental",
            "atu_med": "Média de alunos por turma no Ensino Médio",
            "reprov_fun": "Alunos reprovados no Ensino Fundamental (%)",
            "reprov_med": "Alunos reprovados no Ensino Médio (%)",
            "aband_fun": "Alunos que abandonaram os estudos no Ensino Fundamental (%)",
            "aband_med": "Alunos que abandonaram os estudos no Ensino Médio (%)",
            "icg": "Nível de complexidade da gestão escolar (1 a 6)",
        },
        "historico": json.loads(history.sort_values("ano")[history_columns].to_json(orient="records")),
        "previsao_2026": {"classe_risco": str(school["classe_risco"]),
                          "pontuacao_nao_calibrada": None if pd.isna(school["prob_risco"]) else float(school["prob_risco"])},
    }
    context["indicadores_observados"] = indicator_evidence(context)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "").strip()
    try:
        api_key = api_key or str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        model = model or str(st.secrets.get("GEMINI_MODEL", "")).strip()
    except (FileNotFoundError, KeyError):
        pass
    model = model or DEFAULT_MODEL
    # Include the whole context and credential fingerprint so changed filters,
    # data, model or credentials never reuse an unrelated result.
    cache_key = hashlib.sha256(json.dumps(
        [context, model, hashlib.sha256(api_key.encode()).hexdigest()],
        ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()
    cache = st.session_state.setdefault("followup_summaries", {})
    if api_key and cache_key not in cache:
        with st.spinner("Preparando resumo da escola..."):
            try:
                cache[cache_key] = {"summary": generate_summary(context, api_key, model), "error": None}
            except SummaryError as exc:
                cache[cache_key] = {"summary": fallback_summary(context), "error": str(exc)}
        while len(cache) > 20:
            del cache[next(iter(cache))]
    result = cache.get(cache_key) if api_key else None
    summary = result["summary"] if result else fallback_summary(context)
    blocks = (
        ("1 · O que o modelo previu", [explain_prediction(context)],
         "A pontuação define a faixa; não é uma probabilidade nem uma explicação das causas. Risco baixo indica ausência de alerta prioritário, não ausência de problemas."),
        ("2 · O que foi observado", [summary["historico"], summary["indicadores"]],
         "Estes dados contextualizam a escola. O painel não calcula a contribuição de cada indicador para a previsão."),
        ("3 · O que o gestor pode fazer", [summary["proximo_passo"]],
         "Confirme os registros e a situação atual com a equipe antes de decidir uma ação."),
    )
    for column, (title, paragraphs, detail) in zip(st.columns(3, gap="medium"), blocks):
        with column:
            st.markdown(f"**{title}**")
            for paragraph in paragraphs:
                # Render model output as escaped text, never as executable HTML.
                st.markdown(f'<div style="color:{INK};font-size:14px;line-height:1.65;margin-bottom:12px">'
                            f'{html.escape(paragraph)}</div>', unsafe_allow_html=True)
            note(detail)
    if result and not result["error"]:
        source_note = "Contexto e sugestões com IA (Gemini) · Revise antes de usar. A classificação vem do modelo preditivo do projeto."
    else:
        source_note = "Resumo calculado dos dados, sem IA."
        if result:
            st.info(result["error"])
            if st.button("Tentar gerar com IA novamente", key="retry_followup_summary"):
                cache.pop(cache_key, None)
                st.rerun()
        else:
            note("IA não configurada. Defina GEMINI_API_KEY nos segredos do aplicativo.")
    note(f"{source_note} O alerta não confirma falta de professores nem identifica causas.")


def individual_school():
    header("Consulta detalhada", "Escolas monitoradas", "Pesquise uma escola, acompanhe seu IRD histórico e veja a classificação estimada para 2026.")
    if sel.empty:
        st.warning("Sem escolas para os filtros escolhidos.")
        return
    col_search, col_count = st.columns([3.1, 1], vertical_alignment="bottom")
    with col_search:
        busca = st.text_input("Buscar escola por nome ou código INEP", placeholder="Ex.: AYRES DE MOURA, 35000012...", key="search_school")
    with col_count:
        st.metric("Escolas no filtro", fmt_int(len(sel)))
    d = school_table_data(sel, historico)
    if busca.strip():
        q = norm(busca.strip())
        d = d.loc[d["nome"].map(norm).str.contains(q, regex=False) | d["co_escola"].str.contains(busca.strip(), regex=False, na=False)]
    if d.empty:
        st.info("Nenhuma escola encontrada. Tente outro nome ou código INEP.")
        return
    options = d["co_escola"].tolist()
    details = d.set_index("co_escola")
    chosen = st.selectbox(
        "Selecione uma escola", options[:500],
        format_func=lambda co: f"{details.at[co, 'nome']} · {details.at[co, 'municipio']} [{co}]",
        help="São exibidas até 500 correspondências. Digite parte do nome ou código para refinar.",
    )
    if len(options) > 500:
        note(f"Exibindo 500 de {fmt_int(len(options))} escolas. Refine a busca para encontrar outras.")
    school = details.loc[chosen]
    hs = historico.loc[historico["co_escola"].eq(chosen)].sort_values("ano")
    r = str(school["classe_risco"])
    current = hs.loc[hs["ano"].eq(2025)]
    ir25 = current["ird"].iloc[0] if not current.empty else np.nan
    name = html.escape(str(school["nome"]))
    st.markdown(f'<div style="font-size:22px;font-weight:800;color:{INK};margin-top:8px">{name}</div><div style="font-size:12px;color:{MUTED};margin-bottom:20px">Código INEP {html.escape(chosen)} · {html.escape(str(school["municipio"]))} · Rede {html.escape(str(school["rede"]))} · {html.escape(str(school["localizacao"]))}</div>', unsafe_allow_html=True)
    kk = st.columns(4)
    with kk[0]: kpi("Classificação 2026", r.capitalize(), "Risco preditivo · não confirmado", RISK_COLORS.get(r, TEAL))
    with kk[1]: kpi("IRD observado · 2025", fmt_float(ir25), "Baixa regularidade quando IRD < 2", TEAL)
    with kk[2]: kpi("Pontuação do modelo", fmt_float(school["prob_risco"], 3), "Escala de 0 a 1 · não é probabilidade", AMBER)
    with kk[3]: kpi("Histórico disponível", f"{hs['ird'].notna().sum()} anos", "Dados observados de 2022 a 2025", GREEN)

    section_gap()
    with st.container(border=True, key="visual_panel_07"):
        show_followup_summary(school, hs)

    section_gap()
    with st.container(border=True, key="visual_panel_06"):
        section("Histórico do IRD da escola", "Dados observados de 2022 a 2025. Para 2026, há apenas a classificação de risco.")
        historical_plot(hs, school=True)

    section_gap()
    with st.container(border=True, key="visual_panel_08"):
        section("Outros indicadores observados", "Dados da escola no ano de 2025. Fundamental e Médio são exibidos separadamente.")
        if current.empty:
            st.info("Indicadores de 2025 indisponíveis para esta escola.")
        else:
            rec = current.iloc[0]
            indicadores = [
                ("Professores com formação adequada · Ensino Fundamental (%)", "afd_g1_fun"),
                ("Professores com formação adequada · Ensino Médio (%)", "afd_g1_med"),
                ("Professores com esforço de trabalho alto · Ensino Fundamental (%)", "ied_alto_fun"),
                ("Professores com esforço de trabalho alto · Ensino Médio (%)", "ied_alto_med"),
                ("Média de alunos por turma · Ensino Fundamental", "atu_fun"),
                ("Média de alunos por turma · Ensino Médio", "atu_med"),
                ("Alunos reprovados · Ensino Fundamental (%)", "reprov_fun"),
                ("Alunos que abandonaram os estudos · Ensino Fundamental (%)", "aband_fun"),
                ("Nível de complexidade da gestão escolar (1 a 6)", "icg"),
            ]
            vals = [{"Indicador": title, "Valor em 2025": fmt_float(rec.get(col), 1)} for title, col in indicadores]
            st.dataframe(pd.DataFrame(vals), hide_index=True, use_container_width=True, height=350)

    section_gap()
    with st.expander("Ver todos os anos e indicadores disponíveis para esta escola"):
        note("Uma linha por ano. A identificação da escola aparece nas últimas colunas.")
        st.dataframe(school_history_table(hs), hide_index=True, use_container_width=True,
                     column_config={"Ano": st.column_config.NumberColumn("Ano", format="%d")})
        note("IRD: valores maiores indicam maior continuidade da equipe; abaixo de 2, baixa regularidade. Esforço de trabalho considera turmas, turnos e escolas de atuação.")
        note("Não disponível indica dado ausente, não zero. A previsão refere-se ao ano seguinte ao da linha; a pontuação não é uma probabilidade.")


def municipality_page():
    header("Território e priorização", "Mapa de municípios", "Explore a concentração de alertas para orientar o acompanhamento das redes escolares.")
    if mun_sel.empty:
        st.warning("Sem municípios para os filtros escolhidos.")
        return
    a,b,c = st.columns(3)
    with a: kpi("Municípios no recorte", fmt_int(len(mun_sel)), "Com escolas monitoradas em 2026", TEAL)
    with b: kpi("Municípios com alertas", fmt_int(int((mun_sel["alertas"] > 0).sum())), "Pelo menos uma escola em risco alto/médio", RED)
    with c: kpi("Escolas em alerta", fmt_int(mun_sel["alertas"].sum()), "Risco alto + risco médio", AMBER)
    section_gap()
    with st.container(border=True, key="visual_panel_09"):
        section("Distribuição territorial dos alertas", "Passe o cursor sobre uma bolha para consultar a quantidade de escolas em alerta.")
        show_map(mun_sel, height=450)
        note("O mapa mostra as sedes municipais, não os endereços das escolas.")
    section_gap()
    with st.container(border=True, key="visual_panel_10"):
        section("Ranking dos municípios", "As quantidades refletem o recorte selecionado na barra lateral.")
        ordem = st.selectbox("Ordenar por", ["Quantidade de escolas em alerta", "Percentual de escolas em alerta", "Escolas em risco alto"], index=0)
        sortcol = {"Quantidade de escolas em alerta":"alertas", "Percentual de escolas em alerta":"pct_alertas", "Escolas em risco alto":"risco_alto"}[ordem]
        ranking = mun_sel.sort_values([sortcol, "escolas"], ascending=[False, False]).copy()
        ranking = ranking.rename(columns={"municipio":"Município", "escolas":"Escolas", "risco_alto":"Risco alto", "risco_medio":"Risco médio", "risco_baixo":"Risco baixo", "alertas":"Total de alertas", "pct_alertas":"% em alerta", "ird_medio_2025":"IRD médio 2025"})
        view = ranking[["Município", "Escolas", "Risco alto", "Risco médio", "Risco baixo", "Total de alertas", "% em alerta", "IRD médio 2025"]]
        st.dataframe(view, use_container_width=True, hide_index=True, height=480, column_config={
            "% em alerta": st.column_config.NumberColumn("% em alerta", format="%.1f%%"),
            "IRD médio 2025": st.column_config.NumberColumn("IRD médio 2025", format="%.2f"),
        })
        st.download_button("Baixar ranking filtrado (CSV)", data=view.to_csv(index=False).encode("utf-8-sig"), file_name="radar_docente_municipios_2026.csv", mime="text/csv")


def indicators_page():
    header("Guia do painel", "Entenda os indicadores", "Saiba o que os números significam e como usá-los no acompanhamento das escolas.")
    with st.container(border=True, key="visual_panel_11"):
        section("O que é regularidade docente?", "É a continuidade do vínculo dos professores com a escola.")
        st.write("O **Índice de Regularidade do Corpo Docente (IRD)** vai de **0 a 5**. Quanto maior o valor, maior a regularidade do vínculo da equipe com a escola. Valores **abaixo de 2** são considerados de baixa regularidade neste projeto.")
        note("O IRD não mede diretamente a quantidade de professores nem a qualidade do ensino.")

    section_gap()
    with st.container(border=True, key="visual_panel_12"):
        section("O que significam as faixas de risco?", "Indicam a prioridade de acompanhamento para possível IRD abaixo de 2 em 2026.")
        a, b, c = st.columns(3)
        with a:
            kpi("Risco alto", "Priorizar", "Verificar a situação com a gestão escolar", RED)
        with b:
            kpi("Risco médio", "Acompanhar", "Revisar registros e observar mudanças", AMBER)
        with c:
            kpi("Risco baixo", "Monitorar", "Manter o acompanhamento da escola", GREEN)
        note("A classificação é uma previsão. Risco baixo não garante ausência de problemas; risco alto não confirma falta de professores.")
        st.write("A **pontuação do modelo**, de 0 a 1, define essas faixas. Uma pontuação de 0,70 não significa 70% de chance de baixa regularidade.")

    section_gap()
    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(border=True, key="visual_panel_13"):
            section("De onde vêm os dados?", "Indicadores públicos do INEP, organizados por escola.")
            st.write("O painel reúne escolas públicas das redes **estadual e municipal de São Paulo**. O histórico mostra dados observados de **2022 a 2025**; a classificação para **2026** é uma previsão baseada nos dados de 2025.")
            note("Não disponível indica ausência de informação, não valor zero. Os mapas mostram as sedes municipais.")
    with b:
        with st.container(border=True, key="visual_panel_14"):
            section("Como funciona o resumo por IA?", "Uma leitura dos dados da escola selecionada e do recorte atual.")
            st.write("O **Gemini** recebe os dados públicos da escola, o histórico, a previsão e o contexto dos filtros para resumir a situação e sugerir uma verificação. Ele não calcula a classificação de risco: essa classificação já vem do modelo preditivo do projeto.")
            note("O texto pode conter erros; revise antes de usar. Se a IA estiver indisponível, o painel identifica o resumo calculado diretamente dos dados.")

    section_gap()
    with st.container(border=True, key="visual_panel_15"):
        section("Como ler os outros indicadores?", "Ensino Fundamental e Ensino Médio são apresentados separadamente.")
        st.markdown("""
        - **Formação adequada:** percentual de professores com formação adequada à disciplina em que atuam.
        - **Esforço de trabalho alto:** percentual de professores nos níveis elevados de esforço, considerando fatores como turmas, turnos e escolas de atuação.
        - **Alunos por turma:** quantidade média de estudantes em cada turma.
        - **Reprovação e abandono:** percentuais de alunos reprovados ou que deixaram de frequentar a escola durante o ano letivo.
        - **Complexidade da gestão:** nível de 1 a 6 que considera características como porte, turnos e etapas oferecidas; não é uma nota de qualidade da gestão.
        """)
        note("Use os indicadores para orientar perguntas à gestão escolar. Eles não identificam, por si só, as causas de uma situação.")

    section_gap()
    with st.expander("Detalhes técnicos e metodologia", expanded=False):
        section("Dados e desenvolvimento do modelo")
        st.markdown("""
        1. **Fontes:** IRD, formação docente (AFD), esforço docente (IED), alunos por turma (ATU), complexidade de gestão (ICG) e taxas de rendimento do INEP.
        2. **Treino:** dados de 2022 e 2023, com a situação do ano seguinte como alvo.
        3. **Teste temporal:** ano-base 2024 para prever a situação observada em 2025.
        4. **Comparação:** referência simples (IRD < 2), Regressão Logística e Random Forest.
        5. **Projeção:** a Random Forest usa dados observados de 2025 para classificar o risco de 2026.
        """)
        section("Cortes das faixas de risco")
        st.write("Alto: pontuação ≥ 0,52. Médio: ≥ 0,10 e < 0,52. Baixo: < 0,10. Cortes definidos na validação cruzada, sem consultar os resultados do teste.")
        note("As pontuações não são probabilidades calibradas.")
        section("Desempenho na avaliação", "Teste com ano-base 2024 para prever 2025.")
        metrics = pd.DataFrame([
            ("Referência: IRD < 2", .66, .77, .71, 646, 194),
            ("Regressão Logística", .79, .65, .71, 778, 416),
            ("Random Forest", .77, .67, .72, 764, 369),
        ], columns=["Método", "Recall", "Precisão", "F1", "Verdadeiros positivos", "Falsos positivos"])
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        note("Recall: parcela dos casos reais identificada. Precisão: parcela dos alertas confirmada no teste. F1: equilíbrio entre recall e precisão. Valores arredondados do notebook RadarDocente.ipynb.")
        section("Uso responsável e limitações")
        st.markdown("""
        - Confirme os dados com a gestão antes de qualquer intervenção; os alertas não devem gerar decisões automáticas sobre escolas ou professores.
        - Monitore diferenças de desempenho entre redes, localização e perfis socioeconômicos. O INSE foi usado para auditoria, não como variável preditora.
        - Os dados são públicos e agregados por escola, sem identificação individual de professores.
        - A cobertura se limita às redes estadual e municipal de São Paulo.
        """)
    st.markdown('<div class="footnote">Projeto acadêmico · Global Solution FIAP 2026.2 · CitizenTech · ODS 4 (Educação de Qualidade) e ODS 10 (Redução das Desigualdades). Bases produzidas pelo notebook RadarDocente.ipynb.</div>', unsafe_allow_html=True)


with page_content:
    if pagina == "Visão geral":
        overview()
    elif pagina == "Explorar escolas":
        individual_school()
    elif pagina == "Mapa de municípios":
        municipality_page()
    else:
        indicators_page()

    note("Fonte: INEP · Escolas públicas de São Paulo · Histórico: 2022–2025 · Previsão: 2026.")
