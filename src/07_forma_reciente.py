"""
07_forma_reciente.py
======================================================================
Recalcula home_form_scored/conceded/win_rate y away_form_* para TODO
el dataset (43,816 partidos) usando una metodología propia, uniforme
y explícitamente libre de data leakage.

Por qué recalcular TODO y no solo los 481 pendientes:
  - teams_match_features.csv ya traía home_form_*/away_form_* para
    ~43,335 filas, pero con una metodología que NO conocemos (ventana
    desconocida). Si solo calculamos los 481 faltantes con NUESTRA
    metodología, tendríamos dos métodos distintos mezclados en la
    misma columna -> inconsistente e indefendible en el informe.
  - Recalcular TODO con una sola metodología, documentada y
    parametrizable (N_FORMA), es más robusto y además resuelve los
    481 pendientes como efecto secundario.

Metodología:
  - Para cada selección, se ordenan TODOS sus partidos (cualquier
    torneo) por fecha.
  - form_scored / form_conceded = promedio de goles a favor/contra en
    los últimos N_FORMA partidos ANTERIORES (shift(1) antes de rolling,
    para que el partido actual NUNCA se use para calcular su propia
    feature -> sin leakage).
  - form_win_rate = proporción de victorias (empates y derrotas = 0)
    en esos mismos últimos N_FORMA partidos.
  - min_periods=1: si una selección tiene menos de N_FORMA partidos
    previos, se usa el promedio de los que existan. Si es su PRIMER
    partido en el dataset, queda en NaN (no hay historia -> correcto).

Después de recalcular home_form_*/away_form_*, se re-derivan
fav_form_*/dog_form_* (creadas en 06_favorito_no_favorito.py) usando
la misma máscara home_es_favorito (home_elo >= away_elo).

Output: sobreescribe data/processed/matches_features_v2.csv
  (mismas columnas, mismo shape, valores de forma actualizados)
======================================================================
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Ventana de forma reciente (documentar este valor en el informe)
N_FORMA = 10

print("=" * 70)
print(" 07_forma_reciente.py - Recálculo uniforme de forma reciente")
print(f" Ventana: últimos {N_FORMA} partidos (cualquier torneo), shift(1)")
print("=" * 70)

# ============================================================
# 1. CARGAR
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df = pd.read_csv(in_path)
df['_date'] = pd.to_datetime(df['_date'])
print(f"\n✅ matches_features_v2.csv cargado: {df.shape}")

n_nan_antes = df['home_form_scored'].isna().sum()
print(f"Filas con home_form_scored=NaN ANTES del recálculo: {n_nan_antes}")

# ============================================================
# 2. CONSTRUIR FORMATO LARGO (una fila por equipo por partido)
# ============================================================
df = df.reset_index().rename(columns={'index': '_row_id'})

home_long = df[['_row_id', '_date', '_home_team', 'home_goals', 'away_goals']].copy()
home_long.columns = ['_row_id', '_date', 'team', 'goals_for', 'goals_against']
home_long['side'] = 'home'

away_long = df[['_row_id', '_date', '_away_team', 'away_goals', 'home_goals']].copy()
away_long.columns = ['_row_id', '_date', 'team', 'goals_for', 'goals_against']
away_long['side'] = 'away'

long_df = pd.concat([home_long, away_long], ignore_index=True)

# Orden crítico: por equipo, por fecha, y por _row_id como desempate
# (para partidos del mismo equipo el mismo día, raro pero posible)
long_df = long_df.sort_values(['team', '_date', '_row_id']).reset_index(drop=True)

long_df['win'] = (long_df['goals_for'] > long_df['goals_against']).astype(float)

print(f"\nFormato largo construido: {long_df.shape} (2 filas por partido)")
print(f"Selecciones únicas: {long_df['team'].nunique()}")

# ============================================================
# 3. ROLLING WINDOW POR EQUIPO (shift(1) -> sin leakage)
# ============================================================
grp = long_df.groupby('team', group_keys=False)

long_df['form_scored'] = grp['goals_for'].apply(
    lambda s: s.shift(1).rolling(window=N_FORMA, min_periods=1).mean()
)
long_df['form_conceded'] = grp['goals_against'].apply(
    lambda s: s.shift(1).rolling(window=N_FORMA, min_periods=1).mean()
)
long_df['form_win_rate'] = grp['win'].apply(
    lambda s: s.shift(1).rolling(window=N_FORMA, min_periods=1).mean()
)

n_debut = long_df['form_scored'].isna().sum()
print(f"\nApariciones sin historia previa (debut, quedan en NaN): {n_debut}")

# ============================================================
# 4. PIVOTAR DE VUELTA A FORMATO ANCHO (home_*/away_*)
# ============================================================
home_form = (
    long_df[long_df['side'] == 'home']
    .set_index('_row_id')[['form_scored', 'form_conceded', 'form_win_rate']]
    .rename(columns={
        'form_scored': 'home_form_scored',
        'form_conceded': 'home_form_conceded',
        'form_win_rate': 'home_form_win_rate',
    })
)
away_form = (
    long_df[long_df['side'] == 'away']
    .set_index('_row_id')[['form_scored', 'form_conceded', 'form_win_rate']]
    .rename(columns={
        'form_scored': 'away_form_scored',
        'form_conceded': 'away_form_conceded',
        'form_win_rate': 'away_form_win_rate',
    })
)

cols_form_home = ['home_form_scored', 'home_form_conceded', 'home_form_win_rate']
cols_form_away = ['away_form_scored', 'away_form_conceded', 'away_form_win_rate']

df = df.set_index('_row_id')
df = df.drop(columns=cols_form_home + cols_form_away)
df = df.join(home_form).join(away_form)
df = df.reset_index(drop=True)

n_nan_despues = df['home_form_scored'].isna().sum()
print(f"\nFilas con home_form_scored=NaN DESPUÉS del recálculo: {n_nan_despues}")
print("(deberían ser solo debuts -> primer partido de una selección en el dataset)")

# ============================================================
# 5. RE-DERIVAR fav_form_* / dog_form_* (creadas en script 06)
# ============================================================
tiene_elo = df['home_elo'].notna() & df['away_elo'].notna()
home_es_favorito = df['home_elo'] >= df['away_elo']

pares_form = [
    ('home_form_scored', 'away_form_scored', 'fav_form_scored', 'dog_form_scored'),
    ('home_form_conceded', 'away_form_conceded', 'fav_form_conceded', 'dog_form_conceded'),
    ('home_form_win_rate', 'away_form_win_rate', 'fav_form_win_rate', 'dog_form_win_rate'),
]

for home_col, away_col, fav_col, dog_col in pares_form:
    df[fav_col] = np.where(home_es_favorito, df[home_col], df[away_col])
    df[dog_col] = np.where(home_es_favorito, df[away_col], df[home_col])
    df.loc[~tiene_elo, fav_col] = np.nan
    df.loc[~tiene_elo, dog_col] = np.nan

print("\n✅ fav_form_* / dog_form_* re-derivadas con los valores actualizados.")

# ============================================================
# 6. ACTUALIZAR pending_feature_engineering
# ============================================================
# Ahora 'pending_feature_engineering' refiere SOLO a los atributos FIFA
# (home_avg_attack, etc.), no a la forma reciente (ya resuelta para todos).
n_pending = df['pending_feature_engineering'].sum()
print(f"\nFilas con pending_feature_engineering=1 (solo atributos FIFA "
      f"pendientes, NO forma): {n_pending}")

# ============================================================
# 7. GUARDAR (sobreescribe matches_features_v2.csv)
# ============================================================
df.to_csv(in_path, index=False)

print("\n" + "=" * 70)
print(" RESUMEN")
print("=" * 70)
print(f"Shape final: {df.shape}")

# Verificación rápida: las filas que antes eran pending_feature_engineering=1
# (los 481 de Mundial recuperados) ya deberían tener home_form_scored no-NaN
pending_mask = df['pending_feature_engineering'] == 1
n_481_con_forma = pending_mask.sum() - df.loc[pending_mask, 'home_form_scored'].isna().sum()
print(f"\nDe los {pending_mask.sum()} partidos antes 'pending_feature_engineering=1', "
      f"{n_481_con_forma} ahora tienen home_form_scored calculado.")
print(f"(los que sigan en NaN serían debuts de selecciones en Mundial -> revisar)")

print(f"\n💾 Sobreescrito: {in_path}")

print("""
NOTA PARA EL INFORME:
  - N_FORMA = 10 (últimos 10 partidos, cualquier torneo, shift(1) sin leakage)
  - form_win_rate = proporción de VICTORIAS (empate y derrota cuentan 0)
  - Filas en NaN = debut de esa selección en el dataset (sin historia previa)

PRÓXIMO PASO (Juanfe, Semana 2):
  - Diccionario de confederaciones -> mismo_confed
  - tournament_weight
  - dias_descanso
""")