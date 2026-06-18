"""
dashboard/app.py
======================================================================
Plataforma Predictiva Mundial 2026 — Dashboard Streamlit
Semana 4 — Lulu

Estructura:
  - Sidebar: selector de partido (equipos, torneo)
  - Tab 1: Probabilidades por mercado (con modelos o placeholders)
  - Tab 2: Injury Impact Score (NLP pipeline)
  - Tab 3: Forma reciente de cada selección
  - Tab 4: Comparativa de selecciones (Elo, FIFA, head-to-head)
  - Footer: advertencia legal

Correr: streamlit run dashboard/app.py
======================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DATA_RAW       = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
SRC_DIR        = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Mundial 2026 — Análisis Predictivo",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ESTILOS PERSONALIZADOS
# ============================================================
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d1b2a 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a5a;
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8888aa;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: #0f0f2a;
        border: 1px solid #2a2a5a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #8888aa;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-card .sublabel {
        color: #6666aa;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }

    /* Badges de resultado */
    .badge-fav  { background: #1a3a5c; color: #5ab0ff; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge-draw { background: #2a2a1a; color: #ffcc44; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .badge-dog  { background: #3a1a1a; color: #ff7777; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

    /* Advertencia legal */
    .legal-warning {
        background: #1a0a0a;
        border: 1px solid #5a2a2a;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 2rem;
        color: #aa7777;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    /* Injury score */
    .injury-negative { color: #ff6b6b; font-weight: 700; }
    .injury-neutral  { color: #aaaaaa; }
    .injury-positive { color: #6bffb8; font-weight: 700; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #0f0f2a;
        border-radius: 8px;
        color: #8888aa;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1a5a !important;
        color: #ffffff !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0a0a1a;
    }

    /* Ocultar footer de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS (cacheada)
# ============================================================
@st.cache_data(ttl=3600)
def cargar_dataset():
    path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df['_date'] = pd.to_datetime(df['_date'])
    return df

@st.cache_data(ttl=3600)
def cargar_jugadores_clave():
    path = os.path.join(SRC_DIR, 'jugadores_clave.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def cargar_injury_scores():
    path = os.path.join(DATA_PROCESSED, 'injury_impact_scores.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource
def cargar_modelos_dashboard():
    path = os.path.join(MODELS_DIR, 'modelos_dashboard.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def cargar_modelo_1x2():
    try:
        modelo = joblib.load(os.path.join(MODELS_DIR, '1x2_neutral_xgboost_calibrado.pkl'))
        imputer = joblib.load(os.path.join(MODELS_DIR, '1x2_neutral_imputer.pkl'))
        le = joblib.load(os.path.join(MODELS_DIR, '1x2_neutral_label_encoder.pkl'))
        return modelo, imputer, le
    except Exception:
        return None, None, None

@st.cache_resource
def cargar_modelo_goles(mercado):
    try:
        slug = 'over_under_2.5_goles' if mercado == 'ou25' else 'both_teams_to_score'
        modelo = joblib.load(os.path.join(MODELS_DIR, f'{slug}_xgboost_calibrado.pkl'))
        imputer = joblib.load(os.path.join(MODELS_DIR, f'{slug}_imputer.pkl'))
        return modelo, imputer
    except Exception:
        return None, None

# ============================================================
# FEATURES SEGURAS (igual que en feature_lists.py de Juanfe)
# ============================================================
FEATURES_SEGURAS = [
    'fav_dog_elo_diff', 'fav_elo', 'dog_elo',
    'fav_avg_overall', 'dog_avg_overall',
    'fav_max_overall', 'dog_max_overall',
    'fav_avg_attack', 'dog_avg_attack',
    'fav_avg_defense', 'dog_avg_defense',
    'fav_avg_pace', 'dog_avg_pace',
    'fav_avg_shooting', 'dog_avg_shooting',
    'fav_avg_passing', 'dog_avg_passing',
    'fav_form_scored', 'dog_form_scored',
    'fav_form_conceded', 'dog_form_conceded',
    'fav_form_win_rate', 'dog_form_win_rate',
    'tournament_weight', 'is_world_cup', 'is_world_cup_qualifier',
    'is_continental', 'is_neutral', 'mismo_confed',
    'fav_dias_descanso', 'dog_dias_descanso', 'descanso_diff',
]

FEATURES_NEUTRAL = [f for f in FEATURES_SEGURAS if f != 'is_neutral']

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def obtener_selecciones_wc2026(df):
    """Lista de selecciones únicas de los partidos más recientes (proxy WC2026)."""
    recientes = df[df['_date'] >= '2023-01-01']
    selecciones = sorted(set(
        recientes['fav_team'].dropna().tolist() +
        recientes['dog_team'].dropna().tolist()
    ))
    return selecciones

def obtener_ultimos_partidos(df, equipo, n=10):
    """Últimos N partidos de un equipo (como fav o dog)."""
    mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
    partidos = df[mask].sort_values('_date', ascending=False).head(n).copy()

    def resultado_para_equipo(row):
        if row['fav_team'] == equipo:
            gf, gc = row.get('fav_goals', np.nan), row.get('dog_goals', np.nan)
            rival = row['dog_team']
        else:
            gf, gc = row.get('dog_goals', np.nan), row.get('fav_goals', np.nan)
            rival = row['fav_team']
        if pd.isna(gf) or pd.isna(gc):
            resultado = '?'
        elif gf > gc:
            resultado = 'V'
        elif gf == gc:
            resultado = 'E'
        else:
            resultado = 'D'
        return pd.Series({'rival': rival, 'gf': gf, 'gc': gc, 'resultado': resultado})

    info = partidos.apply(resultado_para_equipo, axis=1)
    partidos = pd.concat([partidos[['_date', '_tournament']], info], axis=1)
    return partidos

def head_to_head(df, equipo1, equipo2, n=10):
    """Historial de enfrentamientos directos entre dos equipos."""
    mask = (
        ((df['fav_team'] == equipo1) & (df['dog_team'] == equipo2)) |
        ((df['fav_team'] == equipo2) & (df['dog_team'] == equipo1))
    )
    h2h = df[mask].sort_values('_date', ascending=False).head(n).copy()
    return h2h

def construir_fila_features(df, equipo1, equipo2, es_mundial=True):
    """
    Construye la fila de features para un partido equipo1 vs equipo2.
    Equipo1 = favorito si tiene mayor Elo, de lo contrario se intercambian.
    Usa los últimos valores conocidos de cada equipo.
    """
    def ultimos_valores(equipo, cols):
        mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
        sub = df[mask].sort_values('_date', ascending=False).head(1)
        if len(sub) == 0:
            return {c: np.nan for c in cols}
        row = sub.iloc[0]
        if row['fav_team'] == equipo:
            return {c: row.get(f'fav_{c.split("_", 1)[1]}' if c.startswith('fav_') else c, np.nan) for c in cols}
        return {c: row.get(f'dog_{c.split("_", 1)[1]}' if c.startswith('dog_') else c, np.nan) for c in cols}

    # Elo reciente de cada equipo
    def elo_reciente(equipo):
        mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
        sub = df[mask].sort_values('_date', ascending=False).head(1)
        if len(sub) == 0:
            return 1500.0
        row = sub.iloc[0]
        return row['fav_elo'] if row['fav_team'] == equipo else row['dog_elo']

    elo1 = elo_reciente(equipo1)
    elo2 = elo_reciente(equipo2)
    fav, dog = (equipo1, equipo2) if elo1 >= elo2 else (equipo2, equipo1)
    elo_fav, elo_dog = max(elo1, elo2), min(elo1, elo2)

    # Atributos FIFA y forma del favorito
    def attrs(equipo, prefix):
        mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
        sub = df[mask].sort_values('_date', ascending=False).head(1)
        if len(sub) == 0:
            return {}
        row = sub.iloc[0]
        side = 'fav' if row['fav_team'] == equipo else 'dog'
        return {
            f'{prefix}_avg_overall':  row.get(f'{side}_avg_overall', np.nan),
            f'{prefix}_max_overall':  row.get(f'{side}_max_overall', np.nan),
            f'{prefix}_avg_attack':   row.get(f'{side}_avg_attack', np.nan),
            f'{prefix}_avg_defense':  row.get(f'{side}_avg_defense', np.nan),
            f'{prefix}_avg_pace':     row.get(f'{side}_avg_pace', np.nan),
            f'{prefix}_avg_shooting': row.get(f'{side}_avg_shooting', np.nan),
            f'{prefix}_avg_passing':  row.get(f'{side}_avg_passing', np.nan),
            f'{prefix}_form_scored':   row.get(f'{side}_form_scored', np.nan),
            f'{prefix}_form_conceded': row.get(f'{side}_form_conceded', np.nan),
            f'{prefix}_form_win_rate': row.get(f'{side}_form_win_rate', np.nan),
            f'{prefix}_dias_descanso': row.get(f'{side}_dias_descanso', np.nan),
            f'{prefix}_elo': elo_fav if prefix == 'fav' else elo_dog,
        }

    fav_attrs = attrs(fav, 'fav')
    dog_attrs = attrs(dog, 'dog')

    fila = {
        'fav_dog_elo_diff':   elo_fav - elo_dog,
        'is_neutral':         1,
        'is_world_cup':       1 if es_mundial else 0,
        'is_world_cup_qualifier': 0,
        'is_continental':     0,
        'tournament_weight':  2.5 if es_mundial else 1.0,
        'mismo_confed':       0,
        'descanso_diff':      0,
        **fav_attrs,
        **dog_attrs,
    }

    return pd.DataFrame([fila]), fav, dog, elo_fav, elo_dog

def predecir_1x2(df, equipo1, equipo2):
    """Predice probabilidades 1X2 para el partido dado."""
    modelo, imputer, le = cargar_modelo_1x2()
    if modelo is None:
        return None, None, None

    fila, fav, dog, elo_fav, elo_dog = construir_fila_features(df, equipo1, equipo2)
    features = [f for f in FEATURES_NEUTRAL if f in fila.columns]
    X = fila[features].values
    X_imp = imputer.transform(X)
    proba = modelo.predict_proba(X_imp)[0]

    resultado = dict(zip(le.classes_, proba))
    return resultado, fav, dog

def predecir_goles(df, equipo1, equipo2):
    """Predice OU2.5 y BTTS."""
    resultados = {}
    for mercado in ['ou25', 'btts']:
        modelo, imputer = cargar_modelo_goles(mercado)
        if modelo is None:
            resultados[mercado] = None
            continue
        fila, _, _, _, _ = construir_fila_features(df, equipo1, equipo2)
        features = [f for f in FEATURES_SEGURAS if f in fila.columns]
        X = fila[features].values
        X_imp = imputer.transform(X)
        proba = modelo.predict_proba(X_imp)[0][1]
        resultados[mercado] = proba
    return resultados

# ============================================================
# HEADER PRINCIPAL
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>⚽ Plataforma Predictiva — Mundial 2026</h1>
    <p>Análisis basado en 43,816 partidos históricos · LSTM + XGBoost calibrado · NLP en tiempo real</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS
# ============================================================
df = cargar_dataset()
jugadores_clave = cargar_jugadores_clave()
injury_scores = cargar_injury_scores()
config_modelos = cargar_modelos_dashboard()

if df is None:
    st.error("No se encontró `data/processed/matches_features_v2.csv`. Corre los scripts de Semana 1-2 primero.")
    st.stop()

selecciones = obtener_selecciones_wc2026(df)

# ============================================================
# SIDEBAR — SELECTOR DE PARTIDO
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Configurar partido")

    col1, col2 = st.columns(2)
    with col1:
        equipo1 = st.selectbox("Equipo 1", selecciones,
                                index=selecciones.index("Argentina") if "Argentina" in selecciones else 0)
    with col2:
        opciones_eq2 = [e for e in selecciones if e != equipo1]
        default_eq2 = "France" if "France" in opciones_eq2 else opciones_eq2[0]
        equipo2 = st.selectbox("Equipo 2", opciones_eq2,
                                index=opciones_eq2.index(default_eq2))

    es_mundial = st.toggle("Partido del Mundial 2026", value=True)
    es_neutral = st.toggle("Sede neutral", value=True)

    st.divider()
    st.markdown("### 🏆 Cuotas 1xbet")
    st.caption("Ingresa las cuotas para detectar value bets")
    cuota_eq1  = st.number_input(f"Cuota {equipo1}", min_value=1.01, value=2.50, step=0.05)
    cuota_draw = st.number_input("Cuota Empate",      min_value=1.01, value=3.20, step=0.05)
    cuota_eq2  = st.number_input(f"Cuota {equipo2}", min_value=1.01, value=2.80, step=0.05)

    st.divider()
    st.caption(f"Dataset: {len(df):,} partidos · Hasta {df['_date'].max().strftime('%Y-%m')}")

# ============================================================
# CALCULAR PREDICCIONES
# ============================================================
proba_1x2, fav, dog = predecir_1x2(df, equipo1, equipo2)
proba_goles = predecir_goles(df, equipo1, equipo2)

# Elo reciente para mostrar
def elo_reciente_equipo(equipo):
    mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
    sub = df[mask].sort_values('_date', ascending=False).head(1)
    if len(sub) == 0:
        return 1500
    row = sub.iloc[0]
    return int(row['fav_elo'] if row['fav_team'] == equipo else row['dog_elo'])

elo1 = elo_reciente_equipo(equipo1)
elo2 = elo_reciente_equipo(equipo2)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predicciones",
    "🏥 Injury Impact",
    "📈 Forma reciente",
    "⚖️ Comparativa"
])

# ====================
# TAB 1: PREDICCIONES
# ====================
with tab1:
    st.markdown(f"### {equipo1} vs {equipo2}")
    st.caption(f"{'Mundial 2026 · Sede neutral' if es_mundial and es_neutral else 'Partido internacional'}")

    # Elo cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Elo {equipo1}</div>
            <div class="value">{elo1:,}</div>
            <div class="sublabel">Rating actual</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        diff = elo1 - elo2
        color = "#5ab0ff" if diff > 0 else ("#ff7777" if diff < 0 else "#aaaaaa")
        st.markdown(f"""<div class="metric-card">
            <div class="label">Diferencia Elo</div>
            <div class="value" style="color:{color}">{diff:+d}</div>
            <div class="sublabel">{'Favorito: ' + equipo1 if diff > 0 else 'Favorito: ' + equipo2 if diff < 0 else 'Igualados'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Elo {equipo2}</div>
            <div class="value">{elo2:,}</div>
            <div class="sublabel">Rating actual</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Probabilidades 1X2
    st.markdown("#### Mercado 1X2")
    if proba_1x2:
        # Identificar qué clase corresponde a cada equipo
        # fav_win = gana el favorito por Elo, dog_win = gana el no favorito
        fav_es_eq1 = (elo1 >= elo2)
        prob_eq1   = proba_1x2.get('fav_win', 0) if fav_es_eq1 else proba_1x2.get('dog_win', 0)
        prob_draw  = proba_1x2.get('draw', 0)
        prob_eq2   = proba_1x2.get('dog_win', 0) if fav_es_eq1 else proba_1x2.get('fav_win', 0)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(f"🏆 {equipo1}", f"{prob_eq1:.1%}")
            p_impl = 1 / cuota_eq1
            ev = prob_eq1 * cuota_eq1 - 1
            if ev > 0.05:
                st.success(f"✅ Value bet! EV = +{ev:.1%}")
            elif ev > 0:
                st.info(f"EV = +{ev:.1%}")
            else:
                st.caption(f"EV = {ev:.1%}")

        with col_b:
            st.metric("🤝 Empate", f"{prob_draw:.1%}")
            ev_draw = prob_draw * cuota_draw - 1
            if ev_draw > 0.05:
                st.success(f"✅ Value bet! EV = +{ev_draw:.1%}")
            elif ev_draw > 0:
                st.info(f"EV = +{ev_draw:.1%}")
            else:
                st.caption(f"EV = {ev_draw:.1%}")

        with col_c:
            st.metric(f"🏆 {equipo2}", f"{prob_eq2:.1%}")
            ev2 = prob_eq2 * cuota_eq2 - 1
            if ev2 > 0.05:
                st.success(f"✅ Value bet! EV = +{ev2:.1%}")
            elif ev2 > 0:
                st.info(f"EV = +{ev2:.1%}")
            else:
                st.caption(f"EV = {ev2:.1%}")

        # Gráfico de barras horizontales
        fig_1x2 = go.Figure(go.Bar(
            x=[prob_eq1, prob_draw, prob_eq2],
            y=[equipo1, "Empate", equipo2],
            orientation='h',
            marker_color=['#5ab0ff', '#ffcc44', '#ff7777'],
            text=[f"{v:.1%}" for v in [prob_eq1, prob_draw, prob_eq2]],
            textposition='auto',
        ))
        fig_1x2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cccccc', height=200,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_1x2, use_container_width=True)

        st.caption("⚠️ Modelo 1X2 con señal moderada (~54% accuracy). Las probabilidades son orientativas.")
    else:
        st.info("Modelo 1X2 no disponible. Verifica que `models/1x2_neutral_xgboost_calibrado.pkl` existe.")

    st.markdown("---")

    # Mercados de goles
    st.markdown("#### Mercados de goles")
    col_ou, col_btts = st.columns(2)

    with col_ou:
        st.markdown("**Over/Under 2.5 goles**")
        if proba_goles.get('ou25') is not None:
            p_over = proba_goles['ou25']
            p_under = 1 - p_over
            st.metric("Over 2.5", f"{p_over:.1%}")
            st.metric("Under 2.5", f"{p_under:.1%}")

            fig_ou = go.Figure(go.Pie(
                labels=["Over 2.5", "Under 2.5"],
                values=[p_over, p_under],
                hole=0.6,
                marker_colors=['#5ab0ff', '#2a2a5a'],
            ))
            fig_ou.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#cccccc', height=200, showlegend=True,
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(font=dict(color='#cccccc')),
            )
            st.plotly_chart(fig_ou, use_container_width=True)
        else:
            st.info("Modelo OU2.5 no disponible.")

    with col_btts:
        st.markdown("**Both Teams To Score**")
        if proba_goles.get('btts') is not None:
            p_btts = proba_goles['btts']
            p_no = 1 - p_btts
            st.metric("Ambos anotan", f"{p_btts:.1%}")
            st.metric("No ambos anotan", f"{p_no:.1%}")

            fig_btts = go.Figure(go.Pie(
                labels=["BTTS Sí", "BTTS No"],
                values=[p_btts, p_no],
                hole=0.6,
                marker_colors=['#6bffb8', '#1a3a2a'],
            ))
            fig_btts.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#cccccc', height=200, showlegend=True,
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(font=dict(color='#cccccc')),
            )
            st.plotly_chart(fig_btts, use_container_width=True)
        else:
            st.info("Modelo BTTS no disponible.")

    st.markdown("---")

    # Mercados de tarjetas
    st.markdown("#### Mercados de tarjetas (solo partidos de Mundial)")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Tarjetas Over/Under 3.5**")
        st.info("🔄 Modelo disponible · Integración en Semana 5")
        st.caption("AUC = 0.593 — el mercado con mejor señal del proyecto.")

    with col_t2:
        st.markdown("**Tarjeta roja**")
        st.warning("⚠️ Probabilidad base empírica (histórico WC 1970-2022)")
        st.metric("P(al menos 1 roja)", "10.8%")
        st.caption("El modelo predictivo no supera el baseline para este mercado con los datos disponibles.")

    # Value bet radar (placeholder visual)
    st.markdown("---")
    st.markdown("#### 🎯 Value Bet Radar")
    st.caption("Compara probabilidades del modelo vs probabilidad implícita en las cuotas")

    if proba_1x2:
        fav_es_eq1 = (elo1 >= elo2)
        prob_eq1   = proba_1x2.get('fav_win', 0) if fav_es_eq1 else proba_1x2.get('dog_win', 0)
        prob_draw  = proba_1x2.get('draw', 0)
        prob_eq2   = proba_1x2.get('dog_win', 0) if fav_es_eq1 else proba_1x2.get('fav_win', 0)

        impl_eq1  = 1 / cuota_eq1
        impl_draw = 1 / cuota_draw
        impl_eq2  = 1 / cuota_eq2

        categorias = [equipo1, "Empate", equipo2]
        modelo_vals = [prob_eq1, prob_draw, prob_eq2]
        impl_vals   = [impl_eq1, impl_draw, impl_eq2]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=modelo_vals + [modelo_vals[0]],
            theta=categorias + [categorias[0]],
            fill='toself', name='Modelo',
            line_color='#5ab0ff', fillcolor='rgba(90,176,255,0.2)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=impl_vals + [impl_vals[0]],
            theta=categorias + [categorias[0]],
            fill='toself', name='Cuota implícita',
            line_color='#ffcc44', fillcolor='rgba(255,204,68,0.15)',
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color='#555577'),
                angularaxis=dict(color='#aaaacc'),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cccccc',
            height=350,
            legend=dict(font=dict(color='#cccccc')),
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Cuando el modelo (azul) supera a la cuota implícita (amarillo), puede haber value. EV > 5% se considera value bet.")

# ====================
# TAB 2: INJURY IMPACT
# ====================
with tab2:
    st.markdown("### 🏥 Injury Impact Score")
    st.caption("Señales de bajas, dudas y recuperaciones detectadas en noticias RSS (Marca, ESPN)")

    if injury_scores.empty:
        st.warning("No hay datos de injury_impact_score. Corre `src/nlp_pipeline.py` para actualizar las noticias.")
    else:
        col_inj1, col_inj2 = st.columns(2)

        for i, equipo in enumerate([equipo1, equipo2]):
            col = col_inj1 if i == 0 else col_inj2
            with col:
                st.markdown(f"#### {equipo}")
                fila = injury_scores[injury_scores['seleccion'] == equipo]

                if len(fila) == 0:
                    st.info(f"Sin cobertura mediática reciente para {equipo}.")
                else:
                    score = fila.iloc[0]['injury_impact_score']
                    n_dudas = int(fila.iloc[0].get('duda', 0))
                    n_bajas = int(fila.iloc[0].get('lesion_baja', 0))
                    n_alta  = int(fila.iloc[0].get('recupero_alta', 0))

                    if score < -0.3:
                        clase = "injury-negative"
                        icono = "🔴"
                        texto = "Señales negativas"
                    elif score > 0.2:
                        clase = "injury-positive"
                        icono = "🟢"
                        texto = "Señales positivas"
                    else:
                        clase = "injury-neutral"
                        icono = "⚪"
                        texto = "Sin señales relevantes"

                    st.markdown(f"""
                    <div style="background:#0f0f2a; border:1px solid #2a2a5a; border-radius:10px; padding:1.2rem;">
                        <div style="font-size:2rem; font-weight:700;" class="{clase}">{icono} {score:+.2f}</div>
                        <div style="color:#8888aa; font-size:0.85rem; margin-top:0.3rem;">{texto}</div>
                        <div style="margin-top:1rem; display:flex; gap:1rem;">
                            <div>🔴 Bajas: <b>{n_bajas}</b></div>
                            <div>🟡 Dudas: <b>{n_dudas}</b></div>
                            <div>🟢 Altas: <b>{n_alta}</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Jugadores clave del diccionario
                st.markdown("**Jugadores clave en el diccionario:**")
                jugs = jugadores_clave.get(equipo, {})
                if jugs:
                    for nombre, info in list(jugs.items())[:5]:
                        importancia = info.get('importancia', 0)
                        titular = "✅" if info.get('titular_habitual') else "🔄"
                        nota = f" · {info['nota']}" if 'nota' in info else ""
                        bar = "█" * int(importancia * 10) + "░" * (10 - int(importancia * 10))
                        st.caption(f"{titular} **{nombre}** · {info.get('posicion','')} · {bar} {importancia:.2f}{nota}")
                else:
                    st.caption("Selección no cubierta en `jugadores_clave.json`.")

    st.markdown("---")
    if st.button("🔄 Actualizar noticias RSS"):
        with st.spinner("Scrapeando noticias..."):
            try:
                from nlp_pipeline import scrapear_noticias, detectar_selecciones, extraer_jugadores, calcular_injury_impact_score
                df_noticias = scrapear_noticias()
                df_noticias = detectar_selecciones(df_noticias)
                df_jug = extraer_jugadores(df_noticias)
                if len(df_jug) > 0:
                    df_jug, df_scores = calcular_injury_impact_score(df_jug)
                    df_scores.to_csv(os.path.join(DATA_PROCESSED, 'injury_impact_scores.csv'), index=False)
                    df_noticias.to_csv(os.path.join(DATA_RAW, 'noticias_scrapeadas.csv'), index=False)
                    st.success(f"✅ Actualizado: {len(df_noticias)} noticias, {len(df_jug)} menciones de jugadores.")
                    st.cache_data.clear()
                else:
                    st.warning("No se detectaron jugadores clave en las noticias actuales.")
            except Exception as e:
                st.error(f"Error al actualizar: {e}")

# ====================
# TAB 3: FORMA RECIENTE
# ====================
with tab3:
    st.markdown("### 📈 Forma reciente")

    col_f1, col_f2 = st.columns(2)

    for i, equipo in enumerate([equipo1, equipo2]):
        col = col_f1 if i == 0 else col_f2
        with col:
            st.markdown(f"#### {equipo} — últimos 10 partidos")
            partidos = obtener_ultimos_partidos(df, equipo, n=10)

            if len(partidos) == 0:
                st.info(f"Sin partidos recientes para {equipo}.")
            else:
                # Gráfico de goles por partido
                partidos_plot = partidos.dropna(subset=['gf', 'gc']).copy()
                partidos_plot['fecha'] = pd.to_datetime(partidos_plot['_date']).dt.strftime('%Y-%m-%d')

                fig_forma = go.Figure()
                fig_forma.add_trace(go.Bar(
                    x=partidos_plot['fecha'][::-1],
                    y=partidos_plot['gf'][::-1],
                    name='Goles a favor',
                    marker_color='#5ab0ff',
                ))
                fig_forma.add_trace(go.Bar(
                    x=partidos_plot['fecha'][::-1],
                    y=-partidos_plot['gc'][::-1],
                    name='Goles en contra',
                    marker_color='#ff7777',
                ))
                fig_forma.update_layout(
                    barmode='relative',
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#cccccc', height=250,
                    margin=dict(l=0, r=0, t=10, b=40),
                    legend=dict(font=dict(color='#cccccc'), orientation='h'),
                    xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=9)),
                    yaxis=dict(showgrid=True, gridcolor='#1a1a3a', zeroline=True, zerolinecolor='#555577'),
                )
                st.plotly_chart(fig_forma, use_container_width=True)

                # Tabla compacta de resultados
                tabla_display = partidos[['_date', 'rival', 'gf', 'gc', 'resultado', '_tournament']].copy()
                tabla_display['_date'] = pd.to_datetime(tabla_display['_date']).dt.strftime('%Y-%m-%d')
                tabla_display.columns = ['Fecha', 'Rival', 'GF', 'GC', 'R', 'Torneo']
                tabla_display['R'] = tabla_display['R'].map({'V': '🟢 V', 'E': '🟡 E', 'D': '🔴 D', '?': '⚪'})
                st.dataframe(
                    tabla_display.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

                # Métricas de forma
                partidos_validos = partidos.dropna(subset=['gf', 'gc'])
                if len(partidos_validos) > 0:
                    victorias = (partidos_validos['resultado'] == 'V').sum()
                    empates   = (partidos_validos['resultado'] == 'E').sum()
                    derrotas  = (partidos_validos['resultado'] == 'D').sum()
                    avg_gf = partidos_validos['gf'].mean()
                    avg_gc = partidos_validos['gc'].mean()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Victorias", f"{victorias}/{len(partidos_validos)}")
                    m2.metric("Empates",   empates)
                    m3.metric("Avg GF",    f"{avg_gf:.1f}")
                    m4.metric("Avg GC",    f"{avg_gc:.1f}")

# ====================
# TAB 4: COMPARATIVA
# ====================
with tab4:
    st.markdown("### ⚖️ Comparativa de selecciones")
    st.caption(f"{equipo1} vs {equipo2} — basado en el último partido disponible de cada selección")

    def attrs_recientes(equipo):
        mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
        sub = df[mask].sort_values('_date', ascending=False).head(1)
        if len(sub) == 0:
            return {}
        row = sub.iloc[0]
        side = 'fav' if row['fav_team'] == equipo else 'dog'
        return {
            'overall':  row.get(f'{side}_avg_overall', np.nan),
            'ataque':   row.get(f'{side}_avg_attack', np.nan),
            'defensa':  row.get(f'{side}_avg_defense', np.nan),
            'ritmo':    row.get(f'{side}_avg_pace', np.nan),
            'definicion': row.get(f'{side}_avg_shooting', np.nan),
            'pase':     row.get(f'{side}_avg_passing', np.nan),
        }

    attrs1 = attrs_recientes(equipo1)
    attrs2 = attrs_recientes(equipo2)

    categorias_fifa = list(attrs1.keys())
    vals1 = [attrs1.get(c, 0) or 0 for c in categorias_fifa]
    vals2 = [attrs2.get(c, 0) or 0 for c in categorias_fifa]

    if any(v > 0 for v in vals1 + vals2):
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatterpolar(
            r=vals1 + [vals1[0]],
            theta=categorias_fifa + [categorias_fifa[0]],
            fill='toself', name=equipo1,
            line_color='#5ab0ff', fillcolor='rgba(90,176,255,0.2)',
        ))
        fig_comp.add_trace(go.Scatterpolar(
            r=vals2 + [vals2[0]],
            theta=categorias_fifa + [categorias_fifa[0]],
            fill='toself', name=equipo2,
            line_color='#ff7777', fillcolor='rgba(255,119,119,0.2)',
        ))
        fig_comp.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[50, 90], color='#555577'),
                angularaxis=dict(color='#aaaacc'),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#cccccc', height=400,
            legend=dict(font=dict(color='#cccccc')),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Atributos FIFA no disponibles para estas selecciones.")

    # Tabla comparativa de atributos
    st.markdown("#### Tabla comparativa")
    tabla_comp = pd.DataFrame({
        'Atributo': ['Elo', 'Overall', 'Ataque', 'Defensa', 'Ritmo', 'Definición', 'Pase'],
        equipo1: [elo1] + [round(attrs1.get(c, np.nan) or np.nan, 1) for c in list(attrs1.keys())],
        equipo2: [elo2] + [round(attrs2.get(c, np.nan) or np.nan, 1) for c in list(attrs2.keys())],
    })
    st.dataframe(tabla_comp, use_container_width=True, hide_index=True)

    # Head to head
    st.markdown("#### ⚔️ Head-to-head histórico")
    h2h = head_to_head(df, equipo1, equipo2, n=10)
    if len(h2h) == 0:
        st.info(f"Sin enfrentamientos directos registrados entre {equipo1} y {equipo2}.")
    else:
        def resultado_h2h(row):
            gf1 = row['fav_goals'] if row['fav_team'] == equipo1 else row['dog_goals']
            gf2 = row['dog_goals'] if row['fav_team'] == equipo1 else row['fav_goals']
            if pd.isna(gf1) or pd.isna(gf2):
                return f"? - ?"
            return f"{int(gf1)} - {int(gf2)}"

        h2h_display = h2h[['_date', '_tournament']].copy()
        h2h_display['_date'] = pd.to_datetime(h2h['_date']).dt.strftime('%Y-%m-%d')
        h2h_display['resultado'] = h2h.apply(resultado_h2h, axis=1)
        h2h_display.columns = ['Fecha', 'Torneo', f'{equipo1} - {equipo2}']

        # Conteo de victorias
        vic1 = vic2 = emp = 0
        for _, row in h2h.iterrows():
            gf1 = row['fav_goals'] if row['fav_team'] == equipo1 else row['dog_goals']
            gf2 = row['dog_goals'] if row['fav_team'] == equipo1 else row['fav_goals']
            if pd.notna(gf1) and pd.notna(gf2):
                if gf1 > gf2:   vic1 += 1
                elif gf1 < gf2: vic2 += 1
                else:            emp += 1

        hc1, hc2, hc3 = st.columns(3)
        hc1.metric(f"Victorias {equipo1}", vic1)
        hc2.metric("Empates", emp)
        hc3.metric(f"Victorias {equipo2}", vic2)

        st.dataframe(h2h_display.reset_index(drop=True), use_container_width=True, hide_index=True)

# ============================================================
# FOOTER — ADVERTENCIA LEGAL
# ============================================================
st.markdown("""
<div class="legal-warning">
    <strong>⚠️ Advertencia de uso responsable</strong><br>
    Esta plataforma comunica <strong>probabilidades calibradas basadas en datos históricos</strong>,
    no certezas. Los modelos tienen una precisión aproximada del 50-60% para el mercado 1X2 y
    del 55-65% para los mercados de goles y tarjetas. Los resultados del fútbol tienen un componente
    de aleatoriedad inherente que ningún modelo puede eliminar.<br><br>
    El uso de esta herramienta para apuestas deportivas es <strong>responsabilidad exclusiva del usuario</strong>.
    Apueste solo lo que esté dispuesto a perder. Si el juego se convierte en un problema,
    contacte una línea de ayuda de juego responsable en su país.<br><br>
    <em>Proyecto académico — Materia de IA/ML · UNIVALLE · Cochabamba, Bolivia · 2026</em>
</div>
""", unsafe_allow_html=True)