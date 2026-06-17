"""
06_favorito_no_favorito.py
======================================================================
Reemplaza la orientación "home/away" (sin sentido real en sede neutral)
por una orientación consistente "favorito / no favorito" basada en Elo.

Por qué:
  - En partidos is_neutral=1 (la gran mayoría de los partidos del
    Mundial), 'home_team'/'away_team' es solo el orden en que aparecían
    los equipos en la fuente original -> no representa localía real.
  - Para la LSTM, si la secuencia de "últimos partidos" de un equipo
    mezcla apariciones aleatorias como home/away sin lógica futbolística,
    el modelo puede aprender ruido en esa dimensión.

Qué hace:
  1. Carga matches_clean_tarjetas.csv (el output más reciente,
     incluye is_world_cup/qualifier corregidos + columnas de tarjetas)
  2. Para CADA partido (neutral o no), define:
       - fav_team / fav_elo / fav_goals / fav_*  = equipo con mayor Elo
       - dog_team / dog_elo / dog_goals / dog_*  = equipo con menor Elo
       - fav_is_home (0/1): si el favorito coincide con el "home" original
         (permite no perder la información de localía real cuando
         is_neutral=0)
  3. Si home_elo o away_elo es NaN, no se puede determinar favorito ->
     se deja fav_team/dog_team en NaN (afecta a partidos muy viejos o
     anteriores al inicio de eloratings.csv)
  4. Recalcula target_1x2 en términos fav/dog:
       - 'fav_win' / 'draw' / 'dog_win'
     (más útil para modelado que home_win/away_win en sede neutral)
  5. Guarda matches_features_v2.csv

IMPORTANTE PARA LULU:
  - Las columnas fav_*/dog_* son las recomendadas para construir las
    secuencias de la LSTM (en vez de home_*/away_*).
  - target_1x2_fav_dog reemplaza a 'result' (home_win/draw/away_win)
    como target del mercado 1X2 para el Mundial 2026.
  - Las columnas originales home_*/away_*/result NO se eliminan
    (se mantienen para trazabilidad), simplemente no se recomienda
    usarlas para Mundial 2026.
======================================================================
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

print("=" * 70)
print(" 06_favorito_no_favorito.py - Reorientación fav/dog vía Elo")
print("=" * 70)

# ============================================================
# 1. CARGAR EL DATASET MÁS RECIENTE
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_clean_tarjetas.csv')
df = pd.read_csv(in_path)
df['_date'] = pd.to_datetime(df['_date'])
print(f"\n✅ {os.path.basename(in_path)} cargado: {df.shape}")

# ============================================================
# 2. DETERMINAR FAVORITO (mayor Elo) Y NO FAVORITO (menor Elo)
# ============================================================
tiene_elo = df['home_elo'].notna() & df['away_elo'].notna()
home_es_favorito = df['home_elo'] >= df['away_elo']

print(f"\nPartidos con Elo disponible para ambos equipos: {tiene_elo.sum()}/{len(df)}")
print(f"Partidos sin Elo (fav/dog quedarán en NaN): {(~tiene_elo).sum()}")

# Mapeo de columnas: para cada columna "home_X"/"away_X" del dataset,
# construir "fav_X"/"dog_X" eligiendo según home_es_favorito
pares_home_away = [
    ('home_elo', 'away_elo'),
    ('home_goals', 'away_goals'),
    ('home_avg_overall', 'away_avg_overall'),
    ('home_max_overall', 'away_max_overall'),
    ('home_avg_attack', 'away_avg_attack'),
    ('home_avg_defense', 'away_avg_defense'),
    ('home_avg_pace', 'away_avg_pace'),
    ('home_avg_shooting', 'away_avg_shooting'),
    ('home_avg_passing', 'away_avg_passing'),
    ('home_form_scored', 'away_form_scored'),
    ('home_form_conceded', 'away_form_conceded'),
    ('home_form_win_rate', 'away_form_win_rate'),
    ('home_yellow_cards', 'away_yellow_cards'),
    ('home_red_cards', 'away_red_cards'),
    ('_home_team', '_away_team'),
]

for home_col, away_col in pares_home_away:
    if home_col not in df.columns or away_col not in df.columns:
        continue
    fav_col = home_col.replace('home_', 'fav_').replace('_home_', '_fav_')
    dog_col = away_col.replace('away_', 'dog_').replace('_away_', '_dog_')

    df[fav_col] = np.where(home_es_favorito, df[home_col], df[away_col])
    df[dog_col] = np.where(home_es_favorito, df[away_col], df[home_col])

    # Donde no hay Elo, no podemos saber quién es favorito -> NaN
    if df[fav_col].dtype != object:
        df.loc[~tiene_elo, fav_col] = np.nan
        df.loc[~tiene_elo, dog_col] = np.nan
    else:
        df.loc[~tiene_elo, fav_col] = None
        df.loc[~tiene_elo, dog_col] = None

# Nombres consistentes (el loop anterior puede dejar nombres raros para
# columnas con prefijo _ , las corregimos explícitamente)
rename_fix = {
    '_fav_team': 'fav_team', '_dog_team': 'dog_team',
}
# (el replace ya deja '_home_team' -> '_fav_team', lo normalizamos)
df = df.rename(columns={c: rename_fix.get(c, c) for c in df.columns if c in rename_fix})

# elo_diff en términos fav/dog (siempre >= 0 por construcción)
df['fav_dog_elo_diff'] = df['fav_elo'] - df['dog_elo']

# ============================================================
# 3. fav_is_home: ¿el favorito es el "home" original?
#    (preserva info de localía real para partidos is_neutral=0)
# ============================================================
df['fav_is_home'] = np.where(tiene_elo, home_es_favorito.astype('Int64'), pd.NA)

# ============================================================
# 4. TARGET 1X2 EN TÉRMINOS FAV/DOG
# ============================================================
df['target_1x2_fav_dog'] = np.where(
    ~tiene_elo, None,
    np.where(
        df['fav_goals'] > df['dog_goals'], 'fav_win',
        np.where(df['fav_goals'] < df['dog_goals'], 'dog_win', 'draw')
    )
)

# ============================================================
# 5. GUARDAR
# ============================================================
out_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df.to_csv(out_path, index=False)

# ============================================================
# RESUMEN
# ============================================================
print("\n" + "=" * 70)
print(" RESUMEN")
print("=" * 70)
print(f"Shape final: {df.shape}")
print(f"\nDistribución target_1x2_fav_dog (partidos con Elo, n={tiene_elo.sum()}):")
print(df.loc[tiene_elo, 'target_1x2_fav_dog'].value_counts())
print(f"\nDistribución fav_is_home (¿el favorito jugaba como 'home' original?):")
print(df.loc[tiene_elo, 'fav_is_home'].value_counts())
print(f"\n💾 Guardado en: {out_path}")

print("""
NOTA PARA LULU:
  - Usar fav_*/dog_* (no home_*/away_*) para construir las secuencias
    de la LSTM. Esto es válido tanto para partidos neutrales como no
    neutrales -> orientación consistente en TODO el dataset.
  - target_1x2_fav_dog reemplaza a 'result' como target del 1X2.
  - Filas con fav_team/dog_team en NaN (sin Elo disponible) deben
    excluirse del entrenamiento (son partidos muy antiguos).

PRÓXIMO PASO (Juanfe, Semana 2):
  - Rolling windows de forma reciente para los 481 pending_feature_engineering=1
  - Confederaciones, tournament_weight, dias_descanso
""")