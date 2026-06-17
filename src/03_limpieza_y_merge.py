"""
03_limpieza_y_merge.py
======================================================================
Construye el dataset limpio final de la Semana 1: matches_clean.csv

Qué hace:
  1. Carga results.csv, teams_match_features.csv, eloratings.csv
  2. Normaliza nombres de equipo (usando 02_normalizacion_nombres.py)
  3. Recalcula is_world_cup (bug confirmado en auditoría: estaba en 0)
  4. Identifica los partidos de FIFA World Cup faltantes en
     teams_match_features.csv y los recupera desde results.csv
  5. Para esos partidos recuperados:
       - Calcula home_elo / away_elo / elo_diff vía merge_asof con
         eloratings.csv (último rating conocido antes del partido)
       - Calcula is_neutral, is_world_cup, is_continental,
         home_goals, away_goals directamente desde results.csv
       - Deja en NaN las columnas que requieren feature engineering
         propio (home_avg_attack, home_form_scored, etc.) y las marca
         con la columna 'pending_feature_engineering = 1'
  6. Concatena ambos bloques, agrega total_goals y result (target base)
  7. Ordena, deduplica y guarda data/processed/matches_clean.csv

IMPORTANTE: las columnas marcadas con pending_feature_engineering = 1
se completan en 02_features.ipynb (Semana 2), donde se recalculan
home_form_* y los atributos FIFA para TODO el dataset de forma
uniforme (rolling windows + mapeo fifa_version), evitando mezclar
dos metodologías distintas.
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(DATA_PROCESSED, exist_ok=True)

# Importar normalización centralizada (script 2)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalizacion_nombres import normalizar_columna, validar_name_fixes

print("=" * 70)
print(" 03_limpieza_y_merge.py - Construcción de matches_clean.csv")
print("=" * 70)

validar_name_fixes()
print("✅ NAME_FIXES validado sin ciclos.")

# ============================================================
# 1. CARGAR DATASETS
# ============================================================
df_results = pd.read_csv(os.path.join(DATA_RAW, 'results.csv'))
df_features = pd.read_csv(os.path.join(DATA_RAW, 'teams_match_features.csv'))
df_elo = pd.read_csv(os.path.join(DATA_RAW, 'eloratings.csv'))

print(f"\nresults.csv:              {df_results.shape}")
print(f"teams_match_features.csv: {df_features.shape}")
print(f"eloratings.csv:           {df_elo.shape}")

# ============================================================
# 2. NORMALIZAR NOMBRES DE EQUIPO (antes de cualquier merge/key)
# ============================================================
df_results = normalizar_columna(df_results, 'home_team')
df_results = normalizar_columna(df_results, 'away_team')
df_features = normalizar_columna(df_features, '_home_team')
df_features = normalizar_columna(df_features, '_away_team')
df_elo = normalizar_columna(df_elo, 'team')

print("\n✅ Nombres normalizados en results, features y eloratings.")

# ============================================================
# 3. CONSTRUIR 'key' (necesaria para is_world_cup y para recuperar
#    los partidos faltantes)
# ============================================================
df_results['date'] = pd.to_datetime(df_results['date'])
df_features['_date'] = pd.to_datetime(df_features['_date'])

df_results['key'] = (
    df_results['date'].dt.strftime('%Y-%m-%d') + '_' +
    df_results['home_team'] + '_' + df_results['away_team']
)
df_features['key'] = (
    df_features['_date'].dt.strftime('%Y-%m-%d') + '_' +
    df_features['_home_team'] + '_' + df_features['_away_team']
)

# ============================================================
# 4. RECALCULAR is_world_cup / is_world_cup_qualifier
# ============================================================
# NO confiamos en _tournament de teams_match_features.csv: ahí el
# Mundial masculino y femenino comparten la misma etiqueta genérica
# "World Cup" (confirmado: con comparación exacta daba 1409, ~373 de
# más respecto a los 1036 de 'FIFA World Cup' en results.csv -> esos
# ~373 son partidos del Mundial FEMENINO coincidentemente etiquetados
# igual).
#
# En cambio, results.csv SÍ distingue 'FIFA World Cup' (masculino,
# 1036 partidos) de cualquier variante femenina. Usamos 'key' para
# marcar en df_features exactamente esos partidos -> consistente con
# el conteo de results.csv y libre de contaminación de género.
wc_keys = set(df_results.loc[df_results['tournament'] == 'FIFA World Cup', 'key'])
wcq_keys = set(df_results.loc[df_results['tournament'] == 'FIFA World Cup qualification', 'key'])

df_features['is_world_cup'] = df_features['key'].isin(wc_keys).astype(int)
df_features['is_world_cup_qualifier'] = df_features['key'].isin(wcq_keys).astype(int)

n_wc_features = df_features['is_world_cup'].sum()
n_wcq_features = df_features['is_world_cup_qualifier'].sum()
print(f"\n✅ is_world_cup recalculado vía cruce de 'key' con results.csv. Partidos: {n_wc_features}")
print(f"   (de los {len(wc_keys)} totales en results.csv, "
      f"{len(wc_keys) - n_wc_features} se recuperan en la sección 6)")
print(f"✅ is_world_cup_qualifier creado vía cruce de 'key'. Partidos: {n_wcq_features}")

# ============================================================
# 5. IDENTIFICAR PARTIDOS DE FIFA WORLD CUP FALTANTES
# ============================================================
es_wc = df_results['tournament'] == 'FIFA World Cup'
faltantes_wc = df_results[
    es_wc & (~df_results['key'].isin(df_features['key']))
].copy()

print(f"\nPartidos de FIFA World Cup en results.csv: {es_wc.sum()}")
print(f"Partidos de FIFA World Cup faltantes en features: {len(faltantes_wc)}")

# ============================================================
# 6. ENRIQUECER LOS PARTIDOS RECUPERADOS
# ============================================================
if len(faltantes_wc) > 0:

    # --- 6a. Elo vía merge_asof (último rating conocido antes del partido) ---
    df_elo['date'] = pd.to_datetime(df_elo['date'], format='mixed', dayfirst=False)

    n_nat = df_elo['date'].isna().sum()
    if n_nat > 0:
        print(f"⚠️ {n_nat} fechas en eloratings.csv no se pudieron parsear (quedaron NaT).")
        print("   Esas filas se ignorarán en el merge_asof.")
        df_elo = df_elo.dropna(subset=['date'])
    df_elo_sorted = df_elo.sort_values('date')

    faltantes_wc = faltantes_wc.sort_values('date')

    # Elo del equipo local
    faltantes_wc = pd.merge_asof(
        faltantes_wc,
        df_elo_sorted.rename(columns={'team': 'home_team', 'rating': 'home_elo'})[['date', 'home_team', 'home_elo']],
        on='date', by='home_team', direction='backward'
    )

    # Elo del equipo visitante
    faltantes_wc = pd.merge_asof(
        faltantes_wc,
        df_elo_sorted.rename(columns={'team': 'away_team', 'rating': 'away_elo'})[['date', 'away_team', 'away_elo']],
        on='date', by='away_team', direction='backward'
    )

    faltantes_wc['elo_diff'] = faltantes_wc['home_elo'] - faltantes_wc['away_elo']

    n_con_elo = faltantes_wc['home_elo'].notna().sum()
    print(f"\n✅ Elo recuperado vía merge_asof para {n_con_elo}/{len(faltantes_wc)} partidos.")
    print("   (los que queden en NaN son anteriores al inicio de eloratings.csv)")

    # --- 6b. Variables calculables directamente desde results.csv ---
    faltantes_wc['is_neutral'] = faltantes_wc['neutral'].astype(str).str.upper().eq('TRUE').astype(int)
    faltantes_wc['is_world_cup'] = 1
    faltantes_wc['is_world_cup_qualifier'] = 0
    faltantes_wc['is_continental'] = 0  # son Mundiales, no copas continentales
    faltantes_wc['home_goals'] = faltantes_wc['home_score']
    faltantes_wc['away_goals'] = faltantes_wc['away_score']

    # --- 6c. Renombrar para alinear con el esquema de df_features ---
    faltantes_wc = faltantes_wc.rename(columns={
        'date': '_date',
        'home_team': '_home_team',
        'away_team': '_away_team',
        'tournament': '_tournament',
    })

    # --- 6d. Columnas pendientes de feature engineering (Semana 2) ---
    cols_pending = [
        'home_avg_overall', 'home_max_overall', 'home_avg_attack', 'home_avg_defense',
        'home_avg_pace', 'home_avg_shooting', 'home_avg_passing',
        'away_avg_overall', 'away_max_overall', 'away_avg_attack', 'away_avg_defense',
        'away_avg_pace', 'away_avg_shooting', 'away_avg_passing',
        'overall_diff', 'attack_diff', 'defense_diff',
        'home_form_scored', 'home_form_conceded', 'home_form_win_rate',
        'away_form_scored', 'away_form_conceded', 'away_form_win_rate',
    ]
    for col in cols_pending:
        faltantes_wc[col] = np.nan

    faltantes_wc['pending_feature_engineering'] = 1
    df_features['pending_feature_engineering'] = 0

    # --- 6e. Quedarnos solo con las columnas del esquema final ---
    cols_finales = [c for c in df_features.columns if c != 'key'] + ['key']
    faltantes_wc = faltantes_wc[cols_finales]

    print(f"\n✅ {len(faltantes_wc)} partidos de Mundial enriquecidos y listos para concatenar.")
else:
    print("\n✅ No hay partidos de Mundial faltantes, nada que recuperar.")
    faltantes_wc = pd.DataFrame(columns=df_features.columns)

# ============================================================
# 7. CONCATENAR, AGREGAR TARGETS BASE Y GUARDAR
# ============================================================
df_clean = pd.concat([df_features, faltantes_wc], ignore_index=True)

# Eliminar posibles duplicados exactos por key (seguridad)
n_antes = len(df_clean)
df_clean = df_clean.drop_duplicates(subset='key', keep='first')
n_despues = len(df_clean)
if n_antes != n_despues:
    print(f"\n⚠️ Se eliminaron {n_antes - n_despues} duplicados por 'key'.")

# Targets / variables derivadas base
df_clean['total_goals'] = df_clean['home_goals'] + df_clean['away_goals']
df_clean['result'] = np.where(
    df_clean['home_goals'] > df_clean['away_goals'], 'home_win',
    np.where(df_clean['home_goals'] < df_clean['away_goals'], 'away_win', 'draw')
)

df_clean = df_clean.sort_values('_date').reset_index(drop=True)

out_path = os.path.join(DATA_PROCESSED, 'matches_clean.csv')
df_clean.to_csv(out_path, index=False)

# ============================================================
# 8. RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print(" RESUMEN FINAL")
print("=" * 70)
print(f"Shape final: {df_clean.shape}")
print(f"Rango de fechas: {df_clean['_date'].min().date()} -> {df_clean['_date'].max().date()}")
print(f"Partidos de Mundial - fase final (is_world_cup=1): {df_clean['is_world_cup'].sum()}")
print(f"Partidos de Mundial - clasificatorias (is_world_cup_qualifier=1): {df_clean['is_world_cup_qualifier'].sum()}")
print(f"Partidos con feature engineering pendiente: {df_clean['pending_feature_engineering'].sum()}")
print(f"Distribución de 'result':")
print(df_clean['result'].value_counts())
print(f"\n💾 Guardado en: {out_path}")

print("""
PRÓXIMO PASO (Semana 2 - 02_features.ipynb):
  - Recalcular home_form_* / away_form_* con rolling windows para
    TODO el dataset (incluye los partidos con pending_feature_engineering=1)
  - Mapear fifa_version -> año para completar avg_attack/defense/etc.
  - Construir variables de contexto Mundial: dias_descanso,
    distancia_sede_km, mismo_continente_sede, presion_situacional
""")