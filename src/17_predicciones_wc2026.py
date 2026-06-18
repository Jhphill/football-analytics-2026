"""
17_predicciones_wc2026.py
======================================================================
Aplica los modelos XGBoost calibrados + value_bet.py sobre los 72
partidos de fase de grupos del Mundial 2026 (wc2026_contexto_geografico.csv).

Dos modos de uso:
  1. SIN cuotas reales -> genera predicciones de probabilidad para todos
     los partidos y guarda en data/processed/wc2026_predicciones.csv
     (listo para que el dashboard lo consuma como base)

  2. CON cuotas reales (desde un CSV de cuotas o entrada manual) ->
     aplica el value bet detector y genera recomendaciones concretas

Sobre las features faltantes:
  wc2026_contexto_geografico.csv tiene Elo, confederación y distancia,
  pero NO tiene forma reciente ni atributos FIFA (son partidos futuros
  -> no hay historia pre-partido disponible en el dataset).
  Para esas columnas el imputer usa la MEDIANA del train set (imputación
  por defecto de sklearn). Esto se documenta explícitamente y significa
  que las predicciones usan el "jugador promedio histórico de WC" como
  proxy cuando no hay datos específicos disponibles.

  Features disponibles en el CSV (señal real):
    - fav_dog_elo_diff, fav_elo, dog_elo         ← señal principal
    - mismo_confed (derivada de confederaciones)  ← señal secundaria
    - is_world_cup=1, is_neutral=1, tournament_weight=2.5 ← contexto

  Features imputadas con mediana del train (proxy):
    - fav_avg_overall, dog_avg_overall, fav_avg_attack, etc.
    - fav_form_scored, dog_form_scored, etc.
    - fav_dias_descanso, dog_dias_descanso, descanso_diff
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(SRC_DIR)
from feature_lists import FEATURES_SEGURAS, TARGETS
from value_bet import analizar_partido

print("=" * 70)
print(" 17_predicciones_wc2026.py")
print(" Predicciones + Value Bet Detector — 72 partidos WC2026")
print("=" * 70)

# ============================================================
# 1. CARGAR EL CSV DE CONTEXTO WC2026
# ============================================================
wc_path = os.path.join(DATA_PROCESSED, 'wc2026_contexto_geografico.csv')
df_wc = pd.read_csv(wc_path)

# Limpiar espacios en los nombres de columna (el CSV tenía espacios)
df_wc.columns = df_wc.columns.str.strip()
df_wc['home_team'] = df_wc['home_team'].str.strip()
df_wc['away_team'] = df_wc['away_team'].str.strip()

print(f"\n✅ wc2026_contexto_geografico.csv cargado: {df_wc.shape}")

# Filtrar partidos con equipos no definidos aún (placeholders de playoffs)
df_definidos = df_wc[
    ~df_wc['home_team'].str.contains('Playoff|Interconf', na=False) &
    ~df_wc['away_team'].str.contains('Playoff|Interconf', na=False)
].copy()

df_placeholders = df_wc[
    df_wc['home_team'].str.contains('Playoff|Interconf', na=False) |
    df_wc['away_team'].str.contains('Playoff|Interconf', na=False)
].copy()

print(f"Partidos con equipos definidos: {len(df_definidos)}")
print(f"Partidos con placeholders (playoffs sin definir): {len(df_placeholders)}")
print("(los placeholders se procesarán una vez se definan los equipos)\n")

# ============================================================
# 2. CONSTRUIR VECTOR DE FEATURES PARA LOS MODELOS XGBOOST
# ============================================================
# Derivar mismo_confed desde las columnas del CSV
df_definidos['mismo_confed'] = (
    df_definidos['home_confed'].notna() &
    (df_definidos['home_confed'] == df_definidos['away_confed'])
).astype(int)

# Construir el DataFrame con las FEATURES_SEGURAS
# Las features disponibles se mapean desde las columnas del CSV
X_wc = pd.DataFrame(index=df_definidos.index)

# Features disponibles directamente
X_wc['fav_dog_elo_diff']        = df_definidos['fav_dog_elo_diff']
X_wc['fav_elo']                  = df_definidos['fav_elo']
X_wc['dog_elo']                  = df_definidos['dog_elo']
X_wc['mismo_confed']             = df_definidos['mismo_confed']
X_wc['is_world_cup']             = 1
X_wc['is_world_cup_qualifier']   = 0
X_wc['is_continental']           = 0
X_wc['is_neutral']               = 1
X_wc['tournament_weight']        = 2.5

# Features que NO están disponibles pre-partido -> NaN (imputer usará mediana)
features_imputar = [
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
    'fav_dias_descanso', 'dog_dias_descanso',
    'descanso_diff',
]
for col in features_imputar:
    X_wc[col] = np.nan

# Asegurar el orden exacto de FEATURES_SEGURAS
X_wc = X_wc[FEATURES_SEGURAS]

# --- Versión específica para el modelo 1X2 (sin is_neutral) ---
# El modelo 1x2_neutral_xgboost_calibrado.pkl se entrenó en
# 15_xai_shap.py con `features_1x2 = [f for f in FEATURES_SEGURAS if
# f != 'is_neutral']` -- ese modelo se entrena SOLO sobre partidos
# neutrales (is_neutral=1 para todos), así que la columna sería
# constante y se excluyó del entrenamiento. Pasarle las 32 columnas
# completas de FEATURES_SEGURAS (con is_neutral incluida) al
# imputer/modelo de 1x2 rompe con "Feature names unseen at fit time:
# is_neutral", porque sklearn valida que las columnas en transform()
# coincidan exactamente con las vistas en fit(). Fix: construir un
# X_wc_1x2 sin esa columna, respetando el mismo orden relativo que
# features_1x2 (FEATURES_SEGURAS filtrada, no reordenada a mano, para
# que coincida exactamente con cómo se entrenó).
features_1x2 = [f for f in FEATURES_SEGURAS if f != 'is_neutral']
X_wc_1x2 = X_wc[features_1x2]

print(f"Features disponibles (señal real): 9 de {len(FEATURES_SEGURAS)}")
print(f"Features imputadas con mediana del train: {len(features_imputar)}")
print("⚠️  Las predicciones son más confiables en fav_dog_elo_diff grande")
print("   (partidos muy asimétricos) donde el Elo domina la señal.\n")

# ============================================================
# 3. CARGAR MODELOS Y CONFIGURACIÓN
# ============================================================
with open(os.path.join(MODELS_DIR, 'modelos_dashboard.json'), 'r', encoding='utf-8') as f:
    config_modelos = json.load(f)

modelos = {}
for mercado, cfg in config_modelos.items():
    if cfg['decision'] == 'USAR_MODELO':
        try:
            modelos[mercado] = {
                'model': joblib.load(os.path.join(MODELS_DIR, cfg['modelo_pkl'])),
                'imputer': joblib.load(os.path.join(MODELS_DIR, cfg['imputer_pkl'])),
                'decision': 'USAR_MODELO',
            }
            if 'label_encoder_pkl' in cfg:
                modelos[mercado]['le'] = joblib.load(
                    os.path.join(MODELS_DIR, cfg['label_encoder_pkl'])
                )
            print(f"✅ {mercado}: {cfg['modelo_pkl']}")
        except FileNotFoundError as e:
            print(f"⚠️  {mercado}: no encontrado ({e})")
    else:
        modelos[mercado] = {
            'decision': 'USAR_BASELINE_EMPIRICO',
            'prob_base': cfg.get('prob_base_empirica', 0.108),
            'advertencia': cfg.get('advertencia_dashboard', ''),
        }
        print(f"⚠️  {mercado}: baseline empírico ({cfg.get('prob_base_empirica', 0.108):.1%})")

    
# ============================================================
# 4. GENERAR PREDICCIONES PARA TODOS LOS PARTIDOS DEFINIDOS
# ============================================================
print(f"\n{'='*70}")
print(f" PREDICCIONES — {len(df_definidos)} partidos definidos")
print(f"{'='*70}")

resultados = []

import unicodedata

def _normalizar_nombre_equipo(s):
    """Normaliza apóstrofes tipográficos, guiones bajos Y diacríticos
    (acentos, cedillas, diéresis) en nombres de equipo, para evitar
    mismatches silenciosos en merges posteriores (ej. con un CSV de
    cuotas tipeado a mano sin tildes: 'Cote d'Ivoire' vs 'Côte
    d'Ivoire' en el dataset, o 'Curacao' vs 'Curaçao').

    La normalización de diacríticos usa NFKD + filtrado de marcas
    combinantes (Mn) -> 'Côte' y 'Cote' colapsan al mismo string
    'Cote' después de este proceso, igual que 'Curaçao' y 'Curacao'
    colapsan a 'Curacao'. Esto es deliberadamente agresivo: prioriza
    que el merge encuentre el partido por sobre preservar la
    ortografía exacta del nombre (que de todas formas no se muestra
    al usuario final desde esta función -- se usa solo como CLAVE de
    matching, no como texto de display)."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.replace('\u2019', "'").replace('\u2018', "'")
    s = s.replace('_', ' ')
    # Quitar diacríticos: NFKD descompone 'ô' -> 'o' + marca combinante,
    # luego se filtran las marcas combinantes (categoría Unicode 'Mn').
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s


for idx, row in df_definidos.iterrows():
    partido = {
        'grupo': row['group'],
        'fav_team': _normalizar_nombre_equipo(row['fav_team']),
        'dog_team': _normalizar_nombre_equipo(row['dog_team']),
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'fav_dog_elo_diff': row.get('fav_dog_elo_diff', np.nan),
    }

    X_partido = X_wc.loc[[idx]]
    X_partido_1x2 = X_wc_1x2.loc[[idx]]
    fila = dict(partido)

    # --- 1X2 ---
    if '1x2' in modelos and modelos['1x2']['decision'] == 'USAR_MODELO':
        try:
            X_imp = modelos['1x2']['imputer'].transform(X_partido_1x2)
            probas = modelos['1x2']['model'].predict_proba(X_imp)[0]
            le = modelos['1x2']['le']
            prob_dict = dict(zip(le.classes_, probas))
            fila['p_fav_win']  = round(prob_dict.get('fav_win', np.nan), 4)
            fila['p_draw']     = round(prob_dict.get('draw', np.nan), 4)
            fila['p_dog_win']  = round(prob_dict.get('dog_win', np.nan), 4)
        except Exception as e:
            fila['p_fav_win'] = fila['p_draw'] = fila['p_dog_win'] = np.nan
            if idx == df_definidos.index[0]:
                print(f"\n⚠️  ERROR real en modelo 1X2 (mostrado solo para el "
                      f"primer partido, se repite en los demás): {type(e).__name__}: {e}")
                print(f"   Shape de X_partido_1x2: {X_partido_1x2.shape} | "
                      f"columnas: {list(X_partido_1x2.columns)}")

    # --- Over/Under 2.5 goles ---
    if 'over_under_25' in modelos and modelos['over_under_25']['decision'] == 'USAR_MODELO':
        try:
            X_imp = modelos['over_under_25']['imputer'].transform(X_partido)
            fila['p_over25'] = round(
                modelos['over_under_25']['model'].predict_proba(X_imp)[0][1], 4
            )
        except Exception as e:
            fila['p_over25'] = np.nan
            if idx == df_definidos.index[0]:
                print(f"⚠️  ERROR en modelo over_under_25 (primer partido): {type(e).__name__}: {e}")

    # --- BTTS ---
    if 'btts' in modelos and modelos['btts']['decision'] == 'USAR_MODELO':
        try:
            X_imp = modelos['btts']['imputer'].transform(X_partido)
            fila['p_btts'] = round(
                modelos['btts']['model'].predict_proba(X_imp)[0][1], 4
            )
        except Exception as e:
            fila['p_btts'] = np.nan
            if idx == df_definidos.index[0]:
                print(f"⚠️  ERROR en modelo btts (primer partido): {type(e).__name__}: {e}")

    # --- Tarjetas Over/Under 3.5 ---
    if 'tarjetas_ou35' in modelos and modelos['tarjetas_ou35']['decision'] == 'USAR_MODELO':
        try:
            X_imp = modelos['tarjetas_ou35']['imputer'].transform(X_partido)
            fila['p_tarjetas_over35'] = round(
                modelos['tarjetas_ou35']['model'].predict_proba(X_imp)[0][1], 4
            )
        except Exception as e:
            fila['p_tarjetas_over35'] = np.nan
            if idx == df_definidos.index[0]:
                print(f"⚠️  ERROR en modelo tarjetas_ou35 (primer partido): {type(e).__name__}: {e}")

    # --- Tarjeta Roja (baseline empírico) ---
    fila['p_tarjeta_roja'] = 0.108
    fila['tarjeta_roja_nota'] = 'baseline_empirico'

    resultados.append(fila)

df_pred = pd.DataFrame(resultados)

# ============================================================
# 5. MOSTRAR TABLA DE PREDICCIONES
# ============================================================
cols_mostrar = ['grupo', 'fav_team', 'dog_team',
                'p_fav_win', 'p_draw', 'p_dog_win',
                'p_over25', 'p_btts', 'p_tarjetas_over35']

cols_disponibles = [c for c in cols_mostrar if c in df_pred.columns]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)

print(df_pred[cols_disponibles].to_string(index=False))

# ============================================================
# 6. VALUE BET DETECTOR CON CUOTAS REALES
# ============================================================
# Este bloque se activa cuando tenés un CSV con cuotas reales de 1xbet.
# Estructura esperada del CSV de cuotas:
#
#   home_team, away_team, cuota_fav_win, cuota_draw, cuota_dog_win,
#   cuota_over25, cuota_btts, cuota_tarjetas_over35
#
# Si no tenés el CSV, el script igual genera las predicciones arriba.

cuotas_path = os.path.join(DATA_PROCESSED, 'wc2026_cuotas_1xbet.csv')

if os.path.exists(cuotas_path):
    print(f"\n{'='*70}")
    print(f" VALUE BET DETECTOR — cuotas reales de 1xbet")
    print(f"{'='*70}")

    df_cuotas = pd.read_csv(cuotas_path)
    df_cuotas.columns = df_cuotas.columns.str.strip()

    # --- Conversión automática: cuota americana -> decimal ---
    # Las casas de apuestas de EE.UU. (DraftKings, FanDuel, etc.) usan
    # formato "moneyline" (ej. -325, +481), distinto al formato decimal
    # que usa 1xbet y que espera value_bet.py. Se detectó en la práctica
    # que cuotas como "121" se ingresaron pensando en moneyline (+121 =
    # "apostando 100 ganás 121", cuota decimal real 2.21) pero el script
    # las tomaba literalmente como cuota decimal 121.00 -> probabilidad
    # implícita de ~0.8%, absurda para fútbol, y un "value bet" falso
    # con EV > 50 (matemáticamente imposible con una cuota real).
    #
    # Heurística de detección: una cuota decimal de fútbol real está
    # casi siempre en el rango (1.01, 100). Un valor >= 100 en valor
    # absoluto, o negativo, se interpreta como moneyline americana:
    #   positiva (ej. +481):  decimal = (americana / 100) + 1
    #   negativa (ej. -325):  decimal = (100 / |americana|) + 1
    # Si está en (1.01, 100) se asume que ya es decimal y se deja igual.
    def _convertir_cuota_si_es_americana(valor):
        if pd.isna(valor):
            return valor
        v = float(valor)
        if v < 0:
            return (100 / abs(v)) + 1
        elif v >= 100:
            return (v / 100) + 1
        else:
            return v

    columnas_cuota = ['cuota_fav_win', 'cuota_draw', 'cuota_dog_win',
                       'cuota_over25', 'cuota_btts', 'cuota_tarjetas_over35']
    cuotas_convertidas = []
    for col in columnas_cuota:
        if col in df_cuotas.columns:
            originales = df_cuotas[col].copy()
            df_cuotas[col] = df_cuotas[col].apply(_convertir_cuota_si_es_americana)
            # Avisar qué filas se convirtieron (valor original != decimal),
            # para que quede explícito en consola qué se interpretó como
            # americana -- evita sorpresas silenciosas en el otro sentido.
            cambiadas = (originales.notna()) & (originales != df_cuotas[col])
            if cambiadas.any():
                for i in df_cuotas[cambiadas].index:
                    cuotas_convertidas.append({
                        'fila': i, 'mercado': col,
                        'valor_original': originales.loc[i],
                        'cuota_decimal': round(df_cuotas.loc[i, col], 3),
                    })

    if cuotas_convertidas:
        print(f"\n⚠️  {len(cuotas_convertidas)} valor(es) de cuota detectados como "
              f"formato AMERICANO (moneyline) y convertidos a decimal:")
        print(pd.DataFrame(cuotas_convertidas).to_string(index=False))
        print("   Si alguno de estos en realidad SÍ era una cuota decimal real")
        print("   (ej. un evento con probabilidad <1%), revisar manualmente.")

    # Normalización defensiva del lado de las cuotas (df_pred ya viene
    # normalizado desde la construcción de resultados, arriba). Nombres
    # con apóstrofes tipográficos (') vs rectos (') o guiones bajos vs
    # espacios (Cape_Verde vs Cape Verde) pueden hacer que un partido
    # NO matchee SIN error visible -> se pierde silenciosamente.
    for col in ['fav_team', 'dog_team']:
        df_cuotas[col] = df_cuotas[col].apply(_normalizar_nombre_equipo)

    # --- Reordenar filas donde el usuario invirtió fav/dog a mano ---
    # El CSV de cuotas se llena manualmente mirando el fixture (ej.
    # "South Africa vs South Korea" en el orden del calendario), pero
    # quién es 'favorito' lo determina el Elo en el dataset (acá,
    # South Korea). Si el usuario puso el par invertido respecto al
    # dataset (fav_team=South Africa, dog_team=South Korea cuando el
    # dataset dice lo contrario), el merge exacto por columna pierde
    # el partido SIN avisar. Fix: para cada fila de df_cuotas que no
    # matchea en el orden tal cual, probar el par invertido contra
    # df_pred; si matchea invertido, swapear fav_team<->dog_team Y
    # las columnas de cuota fav_win<->dog_win correspondientes (el
    # resto -- draw, over25, btts, tarjetas -- no depende de fav/dog,
    # no se tocan).
    pares_pred = set(zip(df_pred['fav_team'], df_pred['dog_team']))
    filas_swapeadas = []
    for i in df_cuotas.index:
        par = (df_cuotas.loc[i, 'fav_team'], df_cuotas.loc[i, 'dog_team'])
        par_invertido = (par[1], par[0])
        if par not in pares_pred and par_invertido in pares_pred:
            df_cuotas.loc[i, 'fav_team'], df_cuotas.loc[i, 'dog_team'] = par_invertido
            if 'cuota_fav_win' in df_cuotas.columns and 'cuota_dog_win' in df_cuotas.columns:
                cuota_fav_orig = df_cuotas.loc[i, 'cuota_fav_win']
                df_cuotas.loc[i, 'cuota_fav_win'] = df_cuotas.loc[i, 'cuota_dog_win']
                df_cuotas.loc[i, 'cuota_dog_win'] = cuota_fav_orig
            filas_swapeadas.append(par)

    if filas_swapeadas:
        print(f"\n🔄 {len(filas_swapeadas)} partido(s) tenían fav_team/dog_team "
              f"invertido respecto al dataset (el favorito real es el otro "
              f"equipo según Elo) -- reordenados automáticamente:")
        for p in filas_swapeadas:
            print(f"   CSV decía fav={p[1]}, dog={p[0]}  ->  dataset dice fav={p[0]}, dog={p[1]}")

    # Merge por fav_team / dog_team (ya con los pares corregidos)
    df_merged = df_pred.merge(
        df_cuotas,
        on=['fav_team', 'dog_team'],
        how='inner'
    )
    print(f"\nPartidos con cuotas disponibles: {len(df_merged)}")

    # Diagnóstico: avisar si algún partido del CSV de cuotas TODAVÍA no
    # matcheó ni en orden normal ni invertido (en vez de fallar en
    # silencio) -- esto ya indica un problema real de nombres, no de
    # orden fav/dog.
    cuotas_no_matcheadas = df_cuotas.merge(
        df_pred[['fav_team', 'dog_team']], on=['fav_team', 'dog_team'],
        how='left', indicator=True
    ).query('_merge == "left_only"')
    if len(cuotas_no_matcheadas) > 0:
        print(f"\n⚠️  {len(cuotas_no_matcheadas)} fila(s) del CSV de cuotas NO matchearon "
              f"con ningún partido de las predicciones, ni en orden normal ni invertido "
              f"(revisar nombres de equipo -- puede ser un typo o un nombre distinto al dataset):")
        print(cuotas_no_matcheadas[['fav_team', 'dog_team']].to_string(index=False))

    value_bets_encontrados = []
    BANKROLL = 100.0

    for _, row in df_merged.iterrows():
        partido_str = f"{row['fav_team']} vs {row['dog_team']} (Grupo {row['grupo']})"

        # Revisar cada mercado
        pares_mercado = [
            ('1X2 fav_win',     row.get('p_fav_win'),         row.get('cuota_fav_win')),
            ('1X2 draw',        row.get('p_draw'),            row.get('cuota_draw')),
            ('1X2 dog_win',     row.get('p_dog_win'),         row.get('cuota_dog_win')),
            ('Over 2.5 goles',  row.get('p_over25'),          row.get('cuota_over25')),
            ('BTTS',            row.get('p_btts'),            row.get('cuota_btts')),
            ('Tarjetas O/U 3.5',row.get('p_tarjetas_over35'), row.get('cuota_tarjetas_over35')),
        ]

        for mercado_nombre, prob, cuota in pares_mercado:
            if pd.isna(prob) or pd.isna(cuota) or cuota <= 1.0:
                continue

            reporte = analizar_partido(
                prob_modelo=float(prob),
                cuota=float(cuota),
                margen_minimo=0.05,
                bankroll=BANKROLL
            )

            if reporte['es_value_bet']:
                # Nota de confianza: 1X2 es, según las métricas de
                # Semana 3 (modelos_dashboard.json), "el mercado con
                # señal más débil del proyecto (~50-55% accuracy)" --
                # las probabilidades se documentaron ahí mismo como
                # "orientativas". Además, en este script concreto solo
                # 9 de 32 features tienen señal real para los partidos
                # de WC2026 (el resto se imputa con la mediana del
                # train, ver sección 2 / aviso al inicio del output).
                # Un value bet de 1X2 detectado sobre un partido con
                # mayoría de features imputadas merece menos confianza
                # que uno de OU2.5/BTTS/Tarjetas, que vienen de modelos
                # con backtest validado (ROI positivo en los 3, ver
                # models/backtest_value_bet_*.json). Esto NO descarta
                # el value bet -- el mecanismo de detección es el
                # mismo y es válido -- pero se documenta la diferencia
                # de confianza explícitamente, en vez de mostrar ambos
                # casos con la misma aparente seguridad.
                if mercado_nombre.startswith('1X2'):
                    confianza = 'BAJA (mercado 1X2: señal más débil del proyecto, ~50-55% accuracy general)'
                elif len(features_imputar) >= 20:
                    confianza = 'MODERADA (mayoría de features imputadas por mediana del train, no historial real)'
                else:
                    confianza = 'MODERADA'

                value_bets_encontrados.append({
                    'partido': partido_str,
                    'grupo': row['grupo'],
                    'mercado': mercado_nombre,
                    'prob_modelo': f"{reporte['prob_modelo']:.1%}",
                    'prob_implicita': f"{reporte['prob_implicita']:.1%}",
                    'diferencia_pp': f"{reporte['diferencia_pp']:+.1f}pp",
                    'expected_value': f"{reporte['expected_value']:+.3f}",
                    'cuota': cuota,
                    'kelly_pct': f"{reporte['kelly']['fraccion_bankroll_recomendada']:.1%}",
                    'monto_sugerido': reporte['kelly']['monto_recomendado'],
                    'recomendacion': reporte['recomendacion'],
                    'confianza': confianza,
                })

    if value_bets_encontrados:
        df_vb = pd.DataFrame(value_bets_encontrados)
        print(f"\n🎯 VALUE BETS DETECTADOS: {len(df_vb)}\n")
        print(df_vb[['partido', 'mercado', 'prob_modelo', 'prob_implicita',
                      'diferencia_pp', 'expected_value', 'cuota',
                      'kelly_pct', 'monto_sugerido']].to_string(index=False))
        print(f"\n📋 Nota de confianza por value bet:")
        for _, r in df_vb.iterrows():
            print(f"   {r['partido']} | {r['mercado']}: {r['confianza']}")

        vb_path = os.path.join(DATA_PROCESSED, 'wc2026_value_bets.csv')
        df_vb.to_csv(vb_path, index=False)
        print(f"\n💾 {vb_path}")
    else:
        print("\n⚠️  No se detectaron value bets con las cuotas disponibles.")
        print("   (margen mínimo requerido: 5pp sobre la probabilidad implícita)")

else:
    print(f"\n{'='*70}")
    print(f" VALUE BET DETECTOR — en espera de cuotas reales")
    print(f"{'='*70}")
    print(f"\n⚠️  No se encontró: {cuotas_path}")
    print("""
Para activar el value bet detector, creá el archivo:
  data/processed/wc2026_cuotas_1xbet.csv

Con estas columnas (una fila por partido, en orientación fav/dog):
  fav_team, dog_team, cuota_fav_win, cuota_draw, cuota_dog_win,
  cuota_over25, cuota_btts, cuota_tarjetas_over35

Ejemplo de una fila:
  Argentina, Algeria, 1.25, 5.50, 12.00, 1.85, 1.70, 1.95

Las cuotas las tomás directamente de 1xbet antes del partido.
Con 0 disponibles, podés rellenar las que tenés y dejar NaN el resto
— el script ignora los mercados sin cuota.
""")

# ============================================================
# 7. GUARDAR PREDICCIONES BASE (sin cuotas)
# ============================================================
out_path = os.path.join(DATA_PROCESSED, 'wc2026_predicciones.csv')
df_pred.to_csv(out_path, index=False)

print(f"\n{'='*70}")
print(f" RESUMEN")
print(f"{'='*70}")
print(f"Partidos procesados: {len(df_pred)}")
print(f"Partidos con placeholder (pendientes de playoffs): {len(df_placeholders)}")
print(f"\n💾 Predicciones base guardadas: {out_path}")
print("""
PRÓXIMOS PASOS:
  1. Crear data/processed/wc2026_cuotas_1xbet.csv con las cuotas
     reales de 1xbet partido por partido (antes de cada jornada).
  2. Volver a correr este script -> el value bet detector se activa
     automáticamente cuando detecta el archivo de cuotas.
  3. El archivo wc2026_predicciones.csv ya puede ser consumido
     por el dashboard de Lulu como base de predicciones.
  4. Cuando se definan los equipos de los playoffs, reemplazar los
     placeholders en wc2026_contexto_geografico.csv y volver a correr.
""")