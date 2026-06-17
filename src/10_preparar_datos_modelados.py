"""
10_preparar_datos_modelado.py
======================================================================
Paso 0 de la Semana 3. Define:
  1. El set de FEATURES "seguras" (sin fuga de información) que se
     usarán en TODOS los modelos de la Semana 3.
  2. El split temporal train/test (NO aleatorio) para simular el
     escenario real de predecir partidos futuros con datos pasados.

Por qué split TEMPORAL y no aleatorio:
  - Un split aleatorio mezclaría partidos de 2024 en train con
    partidos de 1990 en test. Eso no representa el problema real:
    en producción, el modelo siempre predice partidos que todavía
    no pasaron, usando solo lo que pasó antes.
  - Partir por fecha es más exigente (y más honesto) que partir
    aleatorio, y es el estándar en series temporales / forecasting.

Fecha de corte: 2018-01-01
  - Deja ~5 años de partidos "recientes" como test (2018-2025),
    incluyendo Rusia 2018, Qatar 2022 y partidos camino a 2026.
  - Es un balance entre tener suficiente test reciente y no perder
    demasiado volumen de train.

Output:
  - data/processed/train_set.csv
  - data/processed/test_set.csv
  - src/feature_lists.py (listas de columnas reutilizables)
======================================================================
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
SRC_DIR = os.path.join(BASE_DIR, 'src')

FECHA_CORTE = '2018-01-01'

print("=" * 70)
print(" 10_preparar_datos_modelado.py - Features seguras + split temporal")
print("=" * 70)

# ============================================================
# 1. CARGAR
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df = pd.read_csv(in_path)
df['_date'] = pd.to_datetime(df['_date'])
print(f"\n✅ matches_features_v2.csv cargado: {df.shape}")

# ============================================================
# 2. DEFINIR FEATURES SEGURAS (sin fuga de información)
# ============================================================
# Estas son las columnas calculadas ANTES del partido. Ninguna usa
# información del resultado del partido en cuestión.
FEATURES_SEGURAS = [
    # Fortaleza relativa
    'fav_dog_elo_diff',
    'fav_elo', 'dog_elo',

    # Atributos FIFA de la plantilla
    'fav_avg_overall', 'dog_avg_overall',
    'fav_max_overall', 'dog_max_overall',
    'fav_avg_attack', 'dog_avg_attack',
    'fav_avg_defense', 'dog_avg_defense',
    'fav_avg_pace', 'dog_avg_pace',
    'fav_avg_shooting', 'dog_avg_shooting',
    'fav_avg_passing', 'dog_avg_passing',

    # Forma reciente (rolling, ya excluye el partido actual por shift(1))
    'fav_form_scored', 'dog_form_scored',
    'fav_form_conceded', 'dog_form_conceded',
    'fav_form_win_rate', 'dog_form_win_rate',

    # Contexto del torneo
    'tournament_weight',
    'is_world_cup', 'is_world_cup_qualifier', 'is_continental', 'is_neutral',
    'mismo_confed',

    # Descanso
    'fav_dias_descanso', 'dog_dias_descanso', 'descanso_diff',

    # Localía real (cuando aplica)
    # NOTA: 'fav_is_home' fue EXCLUIDA tras diagnóstico (ver notas
    # metodológicas). En el histórico correlaciona fuertemente con el
    # resultado incluso en partidos neutrales (sembrado/orden editorial
    # de la fuente), pero en future_match_probabilities_baseline.csv
    # (Mundial 2026) el "home_team" está determinado mayormente por ser
    # el país anfitrión (México/USA/Canadá), no por ser favorito. Usarla
    # habría inflado artificialmente la probabilidad de victoria de los
    # anfitriones sin fundamento real -> se remueve para evitar fuga
    # encubierta no transferible al escenario de predicción real.
]

# Columnas que NUNCA deben usarse como feature (son el resultado o lo derivan)
COLUMNAS_PROHIBIDAS = [
    'home_goals', 'away_goals', 'fav_goals', 'dog_goals', 'total_goals',
    'result', 'target_1x2_fav_dog', 'target_ou25', 'target_btts',
    'target_ou15', 'target_ou35_goals', 'target_cards_ou35', 'target_redcard',
    'home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards',
    'fav_yellow_cards', 'dog_yellow_cards', 'fav_red_cards', 'dog_red_cards',
    'total_cards',
]

# Verificación: ninguna feature segura debe estar en la lista prohibida
interseccion = set(FEATURES_SEGURAS) & set(COLUMNAS_PROHIBIDAS)
assert len(interseccion) == 0, f"¡FUGA DE INFORMACIÓN! Columnas en ambas listas: {interseccion}"
print(f"\n✅ Verificación de fuga de información: OK ({len(FEATURES_SEGURAS)} features, "
      f"0 coincidencias con columnas prohibidas)")

# Verificar que todas las features seguras existen en el dataset
faltantes = [c for c in FEATURES_SEGURAS if c not in df.columns]
if faltantes:
    print(f"⚠️ ADVERTENCIA: estas features no existen en el dataset: {faltantes}")
    FEATURES_SEGURAS = [c for c in FEATURES_SEGURAS if c not in faltantes]

print(f"\nFeatures finales a usar en los modelos ({len(FEATURES_SEGURAS)}):")
for f in FEATURES_SEGURAS:
    print(f"  - {f}")

# ============================================================
# 3. FILTRAR FILAS VÁLIDAS (con Elo, base mínima para cualquier modelo)
# ============================================================
tiene_elo = df['fav_elo'].notna() & df['dog_elo'].notna()
df_validos = df[tiene_elo].copy()
print(f"\nPartidos con Elo disponible (universo de modelado): {len(df_validos):,}/{len(df):,}")

# ============================================================
# 4. SPLIT TEMPORAL
# ============================================================
train = df_validos[df_validos['_date'] < FECHA_CORTE].copy()
test = df_validos[df_validos['_date'] >= FECHA_CORTE].copy()

print(f"\n--- Split temporal (corte: {FECHA_CORTE}) ---")
print(f"Train: {len(train):,} partidos ({train['_date'].min().date()} -> {train['_date'].max().date()})")
print(f"Test:  {len(test):,} partidos ({test['_date'].min().date()} -> {test['_date'].max().date()})")
print(f"Proporción train/test: {len(train)/len(df_validos)*100:.1f}% / {len(test)/len(df_validos)*100:.1f}%")

# Verificación de balance de clases en train vs test (1X2)
print(f"\nDistribución target_1x2_fav_dog en TRAIN:")
print(train['target_1x2_fav_dog'].value_counts(normalize=True).round(3))
print(f"\nDistribución target_1x2_fav_dog en TEST:")
print(test['target_1x2_fav_dog'].value_counts(normalize=True).round(3))

# Partidos de Mundial específicamente en cada split (importante para
# tarjetas/roja, que dependen de ese subconjunto)
print(f"\nPartidos de Mundial (is_world_cup=1) en train: {train['is_world_cup'].sum()}")
print(f"Partidos de Mundial (is_world_cup=1) en test: {test['is_world_cup'].sum()}")
print(f"Partidos de Mundial con tarjetas (target_cards_ou35 notna) en train: "
      f"{train['target_cards_ou35'].notna().sum()}")
print(f"Partidos de Mundial con tarjetas (target_cards_ou35 notna) en test: "
      f"{test['target_cards_ou35'].notna().sum()}")

# ============================================================
# 5. GUARDAR train_set.csv / test_set.csv
# ============================================================
train_path = os.path.join(DATA_PROCESSED, 'train_set.csv')
test_path = os.path.join(DATA_PROCESSED, 'test_set.csv')
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

# ============================================================
# 6. EXPORTAR feature_lists.py (módulo reutilizable)
# ============================================================
feature_lists_code = f'''"""
feature_lists.py
======================================================================
Listas de columnas reutilizables para los notebooks de modelado
(Semana 3). Generado por src/10_preparar_datos_modelado.py.

NO editar a mano salvo que sea para corregir un error -- si se agregan
features nuevas al dataset, regenerar este archivo corriendo de nuevo
10_preparar_datos_modelado.py.
======================================================================
"""

FECHA_CORTE = "{FECHA_CORTE}"

FEATURES_SEGURAS = {FEATURES_SEGURAS!r}

COLUMNAS_PROHIBIDAS = {COLUMNAS_PROHIBIDAS!r}

TARGETS = {{
    "1x2": "target_1x2_fav_dog",
    "over_under_25": "target_ou25",
    "btts": "target_btts",
    "over_under_15": "target_ou15",
    "over_under_35_goles": "target_ou35_goals",
    "tarjetas_ou35": "target_cards_ou35",   # solo ~751 partidos de Mundial
    "tarjeta_roja": "target_redcard",        # solo ~751 partidos de Mundial
}}
'''

feature_lists_path = os.path.join(SRC_DIR, 'feature_lists.py')
with open(feature_lists_path, 'w', encoding='utf-8') as f:
    f.write(feature_lists_code)

print("\n" + "=" * 70)
print(" RESUMEN")
print("=" * 70)
print(f"💾 {train_path} ({train.shape[0]:,} x {train.shape[1]})")
print(f"💾 {test_path} ({test.shape[0]:,} x {test.shape[1]})")
print(f"💾 {feature_lists_path} (FEATURES_SEGURAS, TARGETS)")

print("""
PRÓXIMO PASO:
  - 11_modelo_1x2.py -> baseline (Logistic Regression) + XGBoost
    calibrado para target_1x2_fav_dog
""")