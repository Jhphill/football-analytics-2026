"""
dashboard/app.py
======================================================================
Plataforma Predictiva Mundial 2026 — Dashboard Streamlit
Semana 4 — Lulu (integración final + value_bet + backtest)
- Corregido error DOM en Injury Impact
- Normalización de nombres de países
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
sys.path.insert(0, SRC_DIR)

try:
    from value_bet import analizar_partido
    print("✅ value_bet importado correctamente")
except ModuleNotFoundError:
    st.error(f"❌ No se encontró 'value_bet.py' en {SRC_DIR}.")
    st.stop()

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
# ESTILOS (solo CSS, sin HTML complejo)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
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
    .injury-negative { color: #ff6b6b; font-weight: 700; }
    .injury-neutral  { color: #aaaaaa; }
    .injury-positive { color: #6bffb8; font-weight: 700; }
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
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0a0a1a;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS
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
        data = json.load(f)
    # Normalizar nombres
    alias = {
        "Czech Republic": "Czechia",
        "Korea Republic": "South Korea",
        "Côte d'Ivoire": "Ivory Coast",
        "DR Congo": "Congo DR",
        "USA": "United States",
    }
    normalized = {}
    for key, value in data.items():
        normalized[key] = value
        for dataset_name, dict_name in alias.items():
            if dict_name == key:
                normalized[dataset_name] = value
    # Eliminar duplicado de Germany (si existe)
    # No es necesario, pero si hay dos, el primero prevalece
    return normalized

@st.cache_data(ttl=3600)
def cargar_injury_scores():
    path = os.path.join(DATA_PROCESSED, 'injury_impact_scores.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def cargar_backtest_resumen():
    path = os.path.join(MODELS_DIR, 'backtest_value_bet_resumen_comparativo.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def cargar_modelos_central():
    modelos = {}
    config_path = os.path.join(MODELS_DIR, 'modelos_dashboard.json')
    if not os.path.exists(config_path):
        st.error(f"❌ No se encontró {config_path}.")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    for mercado, cfg in config.items():
        if cfg.get('decision') == 'USAR_MODELO':
            try:
                modelo_pkl = os.path.join(MODELS_DIR, cfg['modelo_pkl'])
                imputer_pkl = os.path.join(MODELS_DIR, cfg['imputer_pkl'])
                modelos[mercado] = {
                    'model': joblib.load(modelo_pkl),
                    'imputer': joblib.load(imputer_pkl),
                    'decision': 'USAR_MODELO',
                }
                if 'label_encoder_pkl' in cfg:
                    le_pkl = os.path.join(MODELS_DIR, cfg['label_encoder_pkl'])
                    modelos[mercado]['le'] = joblib.load(le_pkl)
                print(f"✅ {mercado} cargado desde {cfg['modelo_pkl']}")
            except Exception as e:
                print(f"⚠️ Error cargando {mercado}: {e}")
                modelos[mercado] = {'decision': 'ERROR', 'error': str(e)}
        else:
            modelos[mercado] = {
                'decision': 'USAR_BASELINE_EMPIRICO',
                'prob_base': cfg.get('prob_base_empirica', 0.108),
                'advertencia': cfg.get('advertencia_dashboard', ''),
            }
            print(f"ℹ️ {mercado} usa baseline empírico ({modelos[mercado]['prob_base']:.1%})")
    return modelos

# ============================================================
# FEATURES SEGURAS
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
    recientes = df[df['_date'] >= '2023-01-01']
    selecciones = sorted(set(
        recientes['fav_team'].dropna().tolist() +
        recientes['dog_team'].dropna().tolist()
    ))
    return selecciones

def obtener_ultimos_partidos(df, equipo, n=10):
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
    mask = (
        ((df['fav_team'] == equipo1) & (df['dog_team'] == equipo2)) |
        ((df['fav_team'] == equipo2) & (df['dog_team'] == equipo1))
    )
    h2h = df[mask].sort_values('_date', ascending=False).head(n).copy()
    return h2h

def construir_fila_features(df, equipo1, equipo2, es_mundial=True):
    def ultimos_valores(equipo, cols):
        mask = (df['fav_team'] == equipo) | (df['dog_team'] == equipo)
        sub = df[mask].sort_values('_date', ascending=False).head(1)
        if len(sub) == 0:
            return {c: np.nan for c in cols}
        row = sub.iloc[0]
        if row['fav_team'] == equipo:
            return {c: row.get(f'fav_{c.split("_", 1)[1]}' if c.startswith('fav_') else c, np.nan) for c in cols}
        return {c: row.get(f'dog_{c.split("_", 1)[1]}' if c.startswith('dog_') else c, np.nan) for c in cols}
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

def predecir_1x2(df, equipo1, equipo2, es_mundial=True):
    modelos = cargar_modelos_central()
    if '1x2' not in modelos or modelos['1x2'].get('decision') != 'USAR_MODELO':
        return None, None, None
    cfg = modelos['1x2']
    fila, fav, dog, _, _ = construir_fila_features(df, equipo1, equipo2, es_mundial)
    features = [f for f in FEATURES_NEUTRAL if f in fila.columns]
    X = fila[features].values
    X_imp = cfg['imputer'].transform(X)
    proba = cfg['model'].predict_proba(X_imp)[0]
    resultado = dict(zip(cfg['le'].classes_, proba))
    return resultado, fav, dog

def predecir_goles(df, equipo1, equipo2, es_mundial=True):
    modelos = cargar_modelos_central()
    resultados = {}
    for mercado in ['ou25', 'btts']:
        if mercado not in modelos or modelos[mercado].get('decision') != 'USAR_MODELO':
            resultados[mercado] = None
            continue
        cfg = modelos[mercado]
        fila, _, _, _, _ = construir_fila_features(df, equipo1, equipo2, es_mundial)
        features = [f for f in FEATURES_SEGURAS if f in fila.columns]
        X = fila[features].values
        X_imp = cfg['imputer'].transform(X)
        proba = cfg['model'].predict_proba(X_imp)[0][1]
        resultados[mercado] = proba
    return resultados

def predecir_tarjetas(df, equipo1, equipo2, es_mundial=True):
    modelos = cargar_modelos_central()
    mercado = 'tarjetas_ou35'
    if mercado not in modelos or modelos[mercado].get('decision') != 'USAR_MODELO':
        return None
    cfg = modelos[mercado]
    fila, _, _, _, _ = construir_fila_features(df, equipo1, equipo2, es_mundial)
    features = [f for f in FEATURES_SEGURAS if f in fila.columns]
    X = fila[features].values
    X_imp = cfg['imputer'].transform(X)
    proba = cfg['model'].predict_proba(X_imp)[0][1]
    return proba

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>⚽ Plataforma Predictiva — Mundial 2026</h1>
    <p>Análisis basado en 43,816 partidos históricos · LSTM + XGBoost calibrado · NLP en tiempo real</p>
</div>
""", unsafe_allow_html=True)

df = cargar_dataset()
jugadores_clave = cargar_jugadores_clave()
injury_scores = cargar_injury_scores()
backtest_resumen = cargar_backtest_resumen()
modelos = cargar_modelos_central()

if df is None:
    st.error("No se encontró `data/processed/matches_features_v2.csv`.")
    st.stop()

selecciones = obtener_selecciones_wc2026(df)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Configurar partido")
    if 'equipo1' not in st.session_state:
        st.session_state.equipo1 = "Argentina" if "Argentina" in selecciones else selecciones[0]
    if 'equipo2' not in st.session_state:
        st.session_state.equipo2 = "France" if "France" in selecciones else (selecciones[1] if len(selecciones) > 1 else selecciones[0])

    col1, col2 = st.columns(2)
    with col1:
        equipo1 = st.selectbox(
            "Equipo 1",
            selecciones,
            index=selecciones.index(st.session_state.equipo1) if st.session_state.equipo1 in selecciones else 0,
            key="equipo1_selector"
        )
        if equipo1 != st.session_state.equipo1:
            st.session_state.equipo1 = equipo1
            if st.session_state.equipo2 == equipo1:
                otros = [e for e in selecciones if e != equipo1]
                st.session_state.equipo2 = otros[0] if otros else equipo1
    with col2:
        opciones_eq2 = [e for e in selecciones if e != equipo1]
        if st.session_state.equipo2 not in opciones_eq2:
            st.session_state.equipo2 = opciones_eq2[0] if opciones_eq2 else equipo1
        equipo2 = st.selectbox(
            "Equipo 2",
            opciones_eq2,
            index=opciones_eq2.index(st.session_state.equipo2) if st.session_state.equipo2 in opciones_eq2 else 0,
            key="equipo2_selector"
        )
        if equipo2 != st.session_state.equipo2:
            st.session_state.equipo2 = equipo2
    equipo1 = st.session_state.equipo1
    equipo2 = st.session_state.equipo2

    es_mundial = st.toggle("Partido del Mundial 2026", value=True)
    es_neutral = st.toggle("Sede neutral", value=True)

    st.divider()
    st.markdown("### 🏆 Cuotas 1xbet")
    st.caption("Ingresa las cuotas para detectar value bets")
    cuota_eq1  = st.number_input(f"Cuota {equipo1}", min_value=1.01, value=2.50, step=0.05)
    cuota_draw = st.number_input("Cuota Empate",      min_value=1.01, value=3.20, step=0.05)
    cuota_eq2  = st.number_input(f"Cuota {equipo2}", min_value=1.01, value=2.80, step=0.05)

    st.divider()
    st.markdown("### 🥅 Cuotas adicionales (para value bets)")
    cuota_over_25 = st.number_input("Cuota Over 2.5", min_value=1.01, value=1.85, step=0.05)
    cuota_btts_si = st.number_input("Cuota BTTS Sí", min_value=1.01, value=1.90, step=0.05)
    cuota_tarjetas_over_35 = st.number_input("Cuota Tarjetas Over 3.5", min_value=1.01, value=2.10, step=0.05)

    st.divider()
    st.caption(f"Dataset: {len(df):,} partidos · Hasta {df['_date'].max().strftime('%Y-%m')}")

# ============================================================
# PREDICCIONES
# ============================================================
proba_1x2, fav, dog = predecir_1x2(df, equipo1, equipo2, es_mundial)
proba_goles = predecir_goles(df, equipo1, equipo2, es_mundial)
proba_tarjetas = predecir_tarjetas(df, equipo1, equipo2, es_mundial)

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

# ==================== TAB 1: PREDICCIONES ====================
with tab1:
    st.markdown(f"### {equipo1} vs {equipo2}")
    st.caption(f"{'Mundial 2026 · Sede neutral' if es_mundial and es_neutral else 'Partido internacional'}")

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

    # 1X2
    st.markdown("#### Mercado 1X2")
    if proba_1x2:
        fav_es_eq1 = (elo1 >= elo2)
        prob_eq1   = proba_1x2.get('fav_win', 0) if fav_es_eq1 else proba_1x2.get('dog_win', 0)
        prob_draw  = proba_1x2.get('draw', 0)
        prob_eq2   = proba_1x2.get('dog_win', 0) if fav_es_eq1 else proba_1x2.get('fav_win', 0)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(f"🏆 {equipo1}", f"{prob_eq1:.1%}")
            analisis = analizar_partido(prob_eq1, cuota_eq1)
            if analisis['es_value_bet']:
                st.success(f"✅ Value bet! EV = +{analisis['expected_value']:.1%} · Kelly: {analisis['kelly']['fraccion_bankroll_recomendada']:.2%}%")
            elif analisis['expected_value'] is not None and analisis['expected_value'] > 0:
                st.info(f"EV = +{analisis['expected_value']:.1%}")
            else:
                st.caption(f"EV = {analisis['expected_value']:.1%}")

        with col_b:
            st.metric("🤝 Empate", f"{prob_draw:.1%}")
            analisis = analizar_partido(prob_draw, cuota_draw)
            if analisis['es_value_bet']:
                st.success(f"✅ Value bet! EV = +{analisis['expected_value']:.1%} · Kelly: {analisis['kelly']['fraccion_bankroll_recomendada']:.2%}%")
            elif analisis['expected_value'] is not None and analisis['expected_value'] > 0:
                st.info(f"EV = +{analisis['expected_value']:.1%}")
            else:
                st.caption(f"EV = {analisis['expected_value']:.1%}")

        with col_c:
            st.metric(f"🏆 {equipo2}", f"{prob_eq2:.1%}")
            analisis = analizar_partido(prob_eq2, cuota_eq2)
            if analisis['es_value_bet']:
                st.success(f"✅ Value bet! EV = +{analisis['expected_value']:.1%} · Kelly: {analisis['kelly']['fraccion_bankroll_recomendada']:.2%}%")
            elif analisis['expected_value'] is not None and analisis['expected_value'] > 0:
                st.info(f"EV = +{analisis['expected_value']:.1%}")
            else:
                st.caption(f"EV = {analisis['expected_value']:.1%}")

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
        st.info("Modelo 1X2 no disponible.")

    st.markdown("---")

    # GOLES
    st.markdown("#### Mercados de goles")
    col_ou, col_btts = st.columns(2)

    with col_ou:
        st.markdown("**Over/Under 2.5 goles**")
        if proba_goles.get('ou25') is not None:
            p_over = proba_goles['ou25']
            p_under = 1 - p_over
            st.metric("Over 2.5", f"{p_over:.1%}")
            st.metric("Under 2.5", f"{p_under:.1%}")

            analisis = analizar_partido(p_over, cuota_over_25)
            if analisis['es_value_bet']:
                st.success(f"✅ Value Bet! EV = +{analisis['expected_value']:.1%} · Kelly: {analisis['kelly']['fraccion_bankroll_recomendada']:.2%} del bankroll")
            elif analisis['expected_value'] is not None and analisis['expected_value'] > 0:
                st.info(f"ℹ️ EV = +{analisis['expected_value']:.1%} (no supera el umbral del 5%)")
            else:
                st.caption(f"EV = {analisis['expected_value']:.1%}")

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

            analisis = analizar_partido(p_btts, cuota_btts_si)
            if analisis['es_value_bet']:
                st.success(f"✅ Value Bet! EV = +{analisis['expected_value']:.1%} · Kelly: {analisis['kelly']['fraccion_bankroll_recomendada']:.2%}%")
            elif analisis['expected_value'] is not None and analisis['expected_value'] > 0:
                st.info(f"ℹ️ EV = +{analisis['expected_value']:.1%}")
            else:
                st.caption(f"EV = {analisis['expected_value']:.1%}")

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

    # TARJETAS
    st.markdown("#### Mercados de tarjetas")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Tarjetas Over/Under 3.5**")
        if proba_tarjetas is not None:
            p_over = proba_tarjetas
            p_under = 1 - p_over
            st.metric("Over 3.5", f"{p_over:.1%}")
            st.metric("Under 3.5", f"{p_under:.1%}")

            analisis = analizar_partido(p_over, cuota_tarjetas_over_35)
            if analisis['es_value_bet']:
                st.success(f"✅ Value Bet! EV = +{analisis['expected_value']:.1%} · Kelly: {analisis['kelly']['fraccion_bankroll_recomendada']:.2%}%")
            elif analisis['expected_value'] is not None and analisis['expected_value'] > 0:
                st.info(f"ℹ️ EV = +{analisis['expected_value']:.1%}")
            else:
                st.caption(f"EV = {analisis['expected_value']:.1%}")

            fig_tarjetas = go.Figure(go.Pie(
                labels=["Over 3.5", "Under 3.5"],
                values=[p_over, p_under],
                hole=0.6,
                marker_colors=['#ff8844', '#2a2a5a'],
            ))
            fig_tarjetas.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#cccccc', height=200, showlegend=True,
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(font=dict(color='#cccccc')),
            )
            st.plotly_chart(fig_tarjetas, use_container_width=True)
            st.caption("AUC = 0.593 — el mercado con mejor señal del proyecto.")
        else:
            st.info("Modelo de tarjetas no disponible.")

    with col_t2:
        st.markdown("**Tarjeta roja**")
        if 'tarjeta_roja' in modelos and modelos['tarjeta_roja']['decision'] == 'USAR_BASELINE_EMPIRICO':
            prob_roja = modelos['tarjeta_roja']['prob_base']
            st.warning(f"⚠️ Probabilidad base empírica (histórico WC 1970-2022): {prob_roja:.1%}")
            st.metric("P(al menos 1 roja)", f"{prob_roja:.1%}")
            st.caption("El modelo predictivo no supera el baseline para este mercado con los datos disponibles.")
        else:
            st.warning("⚠️ Probabilidad base empírica (histórico WC 1970-2022): 10.8%")
            st.metric("P(al menos 1 roja)", "10.8%")
            st.caption("El modelo predictivo no supera el baseline para este mercado con los datos disponibles.")

    st.markdown("---")

    # VALUE BET RADAR
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

    st.markdown("---")

    # BACKTEST
    st.markdown("#### 📈 Backtest histórico de Value Bet")
    if backtest_resumen:
        df_backtest = pd.DataFrame(backtest_resumen)
        df_backtest_display = df_backtest.rename(columns={
            'mercado': 'Mercado',
            'n_value_bets': 'Value Bets',
            'pct_value_bets': '% Partidos con Value',
            'tasa_acierto': 'Tasa de acierto (condicional)',
            'roi_oficial': 'ROI (sobre capital apostado)',
            'auc_modelo': 'AUC modelo',
            'auc_mercado_simulado': 'AUC mercado simulado'
        })
        st.dataframe(df_backtest_display, use_container_width=True, hide_index=True)
        st.caption("""
        **Nota:** La tasa de acierto es CONDICIONAL a que el detector encontró value (solo sobre ese subconjunto de partidos).  
        No es el accuracy general del modelo. El ROI se calcula sobre el capital total apostado, no sobre el bankroll de referencia.  
        ⚠️ Backtest con cuotas SIMULADAS (no reales de 1xbet) – ver advertencia metodológica en el informe.
        """)
    else:
        st.info("No se encontró el resumen del backtest. Ejecuta `src/16_value_bet_detector.py` para generarlo.")

# ==================== TAB 2: INJURY IMPACT (100% NATIVO) ====================
with tab2:
    st.markdown("### 🏥 Injury Impact Score")
    st.caption("Señales de bajas, dudas y recuperaciones detectadas en noticias RSS (Marca, ESPN)")

    if injury_scores.empty:
        st.warning("No hay datos. Corre `src/nlp_pipeline.py` para actualizar las noticias.")
    else:
        # ← SIN st.columns aquí, cada equipo en su propio bloque
        for equipo in [equipo1, equipo2]:
            st.markdown(f"---")
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
                    st.error(f"🔴 Score: **{score:+.2f}** — Señales negativas")
                elif score > 0.2:
                    st.success(f"🟢 Score: **{score:+.2f}** — Señales positivas")
                else:
                    st.info(f"⚪ Score: **{score:+.2f}** — Sin señales relevantes")

                st.markdown(
                    f"| 🔴 Bajas | 🟡 Dudas | 🟢 Altas |\n"
                    f"|:---:|:---:|:---:|\n"
                    f"| **{n_bajas}** | **{n_dudas}** | **{n_alta}** |"
                )

            st.markdown("**Jugadores clave en el diccionario:**")
            jugs = jugadores_clave.get(equipo, {})
            if jugs:
                jugadores_data = []
                for nombre, info in list(jugs.items())[:6]:
                    jugadores_data.append({
                        "Titular": "✅" if info.get('titular_habitual') else "🔄",
                        "Nombre": nombre,
                        "Pos": info.get('posicion', ''),
                        "Importancia": info.get('importancia', 0),
                        "Nota": info.get('nota', '')[:60] if info.get('nota') else ''
                    })
                st.dataframe(pd.DataFrame(jugadores_data),
                             use_container_width=True,
                             hide_index=True)
            else:
                st.caption("Selección no cubierta en `jugadores_clave.json`.")

    st.markdown("---")
    if st.button("🔄 Actualizar noticias RSS"):
        with st.spinner("Scrapeando noticias..."):
            try:
                from nlp_pipeline import (scrapear_noticias, detectar_selecciones,
                                           extraer_jugadores, calcular_injury_impact_score)
                df_noticias = scrapear_noticias()
                df_noticias = detectar_selecciones(df_noticias)
                df_jug = extraer_jugadores(df_noticias)
                if len(df_jug) > 0:
                    df_jug, df_scores = calcular_injury_impact_score(df_jug)
                    df_scores.to_csv(
                        os.path.join(DATA_PROCESSED, 'injury_impact_scores.csv'),
                        index=False)
                    df_noticias.to_csv(
                        os.path.join(DATA_RAW, 'noticias_scrapeadas.csv'),
                        index=False)
                    st.success(f"✅ {len(df_noticias)} noticias, {len(df_jug)} menciones.")
                    st.cache_data.clear()
                else:
                    st.warning("No se detectaron jugadores clave.")
            except Exception as e:
                st.error(f"Error: {e}")
# ==================== TAB 3: FORMA RECIENTE ====================
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
                tabla_display = partidos[['_date', 'rival', 'gf', 'gc', 'resultado', '_tournament']].copy()
                tabla_display['_date'] = pd.to_datetime(tabla_display['_date']).dt.strftime('%Y-%m-%d')
                tabla_display.columns = ['Fecha', 'Rival', 'GF', 'GC', 'R', 'Torneo']
                tabla_display['R'] = tabla_display['R'].map({'V': '🟢 V', 'E': '🟡 E', 'D': '🔴 D', '?': '⚪'})
                st.dataframe(tabla_display.reset_index(drop=True), use_container_width=True, hide_index=True, height=300)
                partidos_validos = partidos.dropna(subset=['gf', 'gc'])
                if len(partidos_validos) > 0:
                    victorias = (partidos_validos['resultado'] == 'V').sum()
                    empates   = (partidos_validos['resultado'] == 'E').sum()
                    avg_gf = partidos_validos['gf'].mean()
                    avg_gc = partidos_validos['gc'].mean()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Victorias", f"{victorias}/{len(partidos_validos)}")
                    m2.metric("Empates",   empates)
                    m3.metric("Avg GF",    f"{avg_gf:.1f}")
                    m4.metric("Avg GC",    f"{avg_gc:.1f}")

# ==================== TAB 4: COMPARATIVA ====================
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
        st.info("ℹ️ Atributos FIFA no disponibles para estas selecciones en el dataset procesado. Los valores aparecerán como 'None' en la tabla.")

    st.markdown("#### Tabla comparativa")
    tabla_comp = pd.DataFrame({
        'Atributo': ['Elo', 'Overall', 'Ataque', 'Defensa', 'Ritmo', 'Definición', 'Pase'],
        equipo1: [elo1] + [round(attrs1.get(c, np.nan) or np.nan, 1) for c in list(attrs1.keys())],
        equipo2: [elo2] + [round(attrs2.get(c, np.nan) or np.nan, 1) for c in list(attrs2.keys())],
    })
    st.dataframe(tabla_comp, use_container_width=True, hide_index=True)

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
# FOOTER
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