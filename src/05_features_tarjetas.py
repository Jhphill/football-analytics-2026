"""
05_features_tarjetas.py
======================================================================
Integra tarjetas amarillas/rojas reales de Mundiales (1930-2022) desde
el Fjelstul World Cup Database (github.com/jfjelstul/worldcup,
CC-BY-SA 4.0) a matches_clean.csv.

Fuente: data/raw/fjelstul/matches.csv y bookings.csv
  (descargar de https://github.com/jfjelstul/worldcup/tree/master/data-csv)

Qué hace:
  1. Carga matches.csv y bookings.csv (solo torneos masculinos)
  2. Normaliza nombres de equipo con normalizacion_nombres.py
     (mismo diccionario usado en matches_clean.csv -> consistencia total)
  3. Agrega bookings.csv por (match_id, team) -> yellow_cards, red_cards
  4. Construye home_yellow_cards/red_cards y away_*  por partido
     - Partidos 1970+: sin bookings registrados = 0 tarjetas (correcto)
     - Partidos pre-1970: NaN (el sistema moderno de tarjetas no existía)
  5. Mergea contra matches_clean.csv por (_date, _home_team, _away_team),
     con fallback a orientación invertida (away/home swap)
  6. Crea target_cards_ou (over/under 3.5 tarjetas totales) y
     target_redcard (>=1 roja) SOLO para partidos de FIFA World Cup
  7. Guarda matches_clean_tarjetas.csv

IMPORTANTE: estas columnas solo tienen datos para partidos de FIFA World
Cup (is_world_cup=1, ~1036 filas de 43,816). Para el resto quedan NaN.
Esto se documenta explícitamente: el modelo de tarjetas se entrena SOLO
sobre población de Mundial (misma población a predecir en 2026).
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
FJELSTUL_DIR = os.path.join(DATA_RAW, 'fjelstul')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalizacion_nombres import normalizar_columna, validar_name_fixes

print("=" * 70)
print(" 05_features_tarjetas.py - Integración de tarjetas (Fjelstul WC DB)")
print("=" * 70)

validar_name_fixes()

# ============================================================
# 1. CARGAR FJELSTUL matches.csv y bookings.csv
# ============================================================
fj_matches_path = os.path.join(FJELSTUL_DIR, 'matches.csv')
fj_bookings_path = os.path.join(FJELSTUL_DIR, 'bookings.csv')

for p in [fj_matches_path, fj_bookings_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"No se encontró {p}\n"
            f"Descargar de: https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/{os.path.basename(p)}\n"
            f"y guardar en data/raw/fjelstul/"
        )

fj_matches = pd.read_csv(fj_matches_path)
fj_bookings = pd.read_csv(fj_bookings_path)

# Solo torneos masculinos
fj_matches = fj_matches[~fj_matches['tournament_name'].str.contains('Women', case=False, na=False)].copy()
print(f"\nPartidos masculinos de Mundial (Fjelstul): {len(fj_matches)}")
print(f"Bookings totales (Fjelstul, todos los torneos): {len(fj_bookings)}")

# ============================================================
# 2. NORMALIZAR NOMBRES (mismo diccionario que matches_clean.csv)
# ============================================================
fj_matches = normalizar_columna(fj_matches, 'home_team_name')
fj_matches = normalizar_columna(fj_matches, 'away_team_name')
fj_bookings = normalizar_columna(fj_bookings, 'team_name')

fj_matches['match_date'] = pd.to_datetime(fj_matches['match_date'])

print("✅ Nombres normalizados con NAME_FIXES (mismo módulo que Semana 1)")

# ============================================================
# 3. AGREGAR BOOKINGS POR (match_id, team)
# ============================================================
agg = (
    fj_bookings
    .groupby(['match_id', 'team_name'])[['yellow_card', 'red_card']]
    .sum()
    .reset_index()
)

# ============================================================
# 4. CONSTRUIR home_yellow_cards/red_cards y away_* POR PARTIDO
# ============================================================
# Merge para el equipo local
fj_matches = fj_matches.merge(
    agg.rename(columns={'team_name': 'home_team_name', 'yellow_card': 'home_yellow_cards', 'red_card': 'home_red_cards'}),
    on=['match_id', 'home_team_name'], how='left'
)
# Merge para el equipo visitante
fj_matches = fj_matches.merge(
    agg.rename(columns={'team_name': 'away_team_name', 'yellow_card': 'away_yellow_cards', 'red_card': 'away_red_cards'}),
    on=['match_id', 'away_team_name'], how='left'
)

cols_cards = ['home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards']

# Partidos desde el Mundial 1970 sin bookings = 0 tarjetas (dato real, no NaN)
mask_1970_plus = fj_matches['match_date'] >= '1970-01-01'
for c in cols_cards:
    fj_matches.loc[mask_1970_plus, c] = fj_matches.loc[mask_1970_plus, c].fillna(0)

n_pre1970 = (~mask_1970_plus).sum()
print(f"\nPartidos pre-1970 (sin sistema de tarjetas, quedan en NaN): {n_pre1970}")
print(f"Partidos 1970+ con tarjetas completas: {mask_1970_plus.sum()}")

# ============================================================
# 5. MERGEAR CON matches_clean.csv (SOLO partidos is_world_cup == 1)
# ============================================================
clean_path = os.path.join(DATA_PROCESSED, 'matches_clean.csv')
df_clean = pd.read_csv(clean_path)
df_clean['_date'] = pd.to_datetime(df_clean['_date'])

n_wc_clean = (df_clean['is_world_cup'] == 1).sum()
print(f"\nPartidos is_world_cup=1 en matches_clean.csv: {n_wc_clean}")

# Separar el universo: el merge (incluyendo la orientación invertida) solo
# se aplica sobre partidos de Mundial, para evitar que amistosos/clasificatorias
# con fecha+equipos coincidentes por azar reciban tarjetas de otro partido.
df_wc = df_clean[df_clean['is_world_cup'] == 1].copy()
df_other = df_clean[df_clean['is_world_cup'] != 1].copy()

fj_slim = fj_matches[[
    'match_date', 'home_team_name', 'away_team_name'
] + cols_cards].copy()

# Intento 1: orientación directa (home=home, away=away)
merge_directo = df_wc.merge(
    fj_slim,
    left_on=['_date', '_home_team', '_away_team'],
    right_on=['match_date', 'home_team_name', 'away_team_name'],
    how='left'
)

n_match_directo = merge_directo['home_yellow_cards'].notna().sum()
print(f"\nMatches con merge directo (home=home): {n_match_directo}/{len(fj_slim)}")

# Intento 2: orientación invertida (por si en alguna fuente quedó home/away
# intercambiado respecto a la otra) -> se aplica SOLO donde el directo no
# encontró match, y SOLO dentro de df_wc (partidos de Mundial)
sin_match = merge_directo['home_yellow_cards'].isna()
sin_match_positions = np.where(sin_match.values)[0]

fj_swapped = fj_slim.rename(columns={
    'home_team_name': 'away_team_name_tmp',
    'away_team_name': 'home_team_name',
    'home_yellow_cards': 'away_yellow_cards_sw',
    'home_red_cards': 'away_red_cards_sw',
    'away_yellow_cards': 'home_yellow_cards_sw',
    'away_red_cards': 'home_red_cards_sw',
})
fj_swapped = fj_swapped.rename(columns={'away_team_name_tmp': 'away_team_name'})

# df_wc.iloc[...] por POSICIÓN (no por índice/label) para que coincida con
# merge_directo, que siempre tiene índice 0..N-1 sin importar el de df_wc
merge_swapped = df_wc.iloc[sin_match_positions].reset_index(drop=True).merge(
    fj_swapped,
    left_on=['_date', '_home_team', '_away_team'],
    right_on=['match_date', 'home_team_name', 'away_team_name'],
    how='left'
)
n_match_swapped = merge_swapped['home_yellow_cards_sw'].notna().sum()
print(f"Matches adicionales con orientación invertida: {n_match_swapped}")

# Volcar los valores swapeados en las filas correspondientes de merge_directo,
# usando posiciones: la fila i de merge_swapped corresponde a la posición
# sin_match_positions[i] en merge_directo
for c, c_sw in [
    ('home_yellow_cards', 'home_yellow_cards_sw'),
    ('home_red_cards', 'home_red_cards_sw'),
    ('away_yellow_cards', 'away_yellow_cards_sw'),
    ('away_red_cards', 'away_red_cards_sw'),
]:
    values = merge_swapped[c_sw].values
    mask_found = ~pd.isna(values)
    target_positions = sin_match_positions[mask_found]
    col_idx = merge_directo.columns.get_loc(c)
    merge_directo.iloc[target_positions, col_idx] = values[mask_found]

# Restaurar los índices originales de df_clean (necesario para el concat
# + sort_index del final, que reintegra los partidos no-Mundial)
merge_directo.index = df_wc.index

# Limpiar columnas auxiliares del merge
merge_directo = merge_directo.drop(columns=['match_date', 'home_team_name', 'away_team_name'])

n_total_match = merge_directo['home_yellow_cards'].notna().sum()
print(f"\nTotal de filas de Mundial con tarjetas asignadas: {n_total_match}")
print(f"(de {len(fj_matches)} partidos masculinos de Mundial en Fjelstul)")
print(f"-> Cobertura: {n_total_match}/{n_wc_clean} "
      f"({100*n_total_match/n_wc_clean:.1f}%)")

# Reincorporar los partidos que NO son de Mundial (con columnas NaN) y
# restaurar el orden original
for c in cols_cards:
    df_other[c] = np.nan

merge_directo = pd.concat([merge_directo, df_other], ignore_index=False).sort_index()

# ============================================================
# 6. TARGETS DERIVADOS (solo donde hay datos de tarjetas)
# ============================================================
merge_directo['total_cards'] = (
    merge_directo['home_yellow_cards'] + merge_directo['away_yellow_cards'] +
    merge_directo['home_red_cards'] + merge_directo['away_red_cards']
)
merge_directo['target_cards_ou35'] = np.where(
    merge_directo['total_cards'].notna(),
    (merge_directo['total_cards'] > 3.5).astype('Int64'),
    pd.NA
)
merge_directo['target_redcard'] = np.where(
    merge_directo['total_cards'].notna(),
    ((merge_directo['home_red_cards'] + merge_directo['away_red_cards']) >= 1).astype('Int64'),
    pd.NA
)

# ============================================================
# 7. GUARDAR
# ============================================================
out_path = os.path.join(DATA_PROCESSED, 'matches_clean_tarjetas.csv')
merge_directo.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print(" RESUMEN")
print("=" * 70)
print(f"Shape final: {merge_directo.shape}")
print(f"Partidos con target_cards_ou35 definido: {merge_directo['target_cards_ou35'].notna().sum()}")
if merge_directo['target_cards_ou35'].notna().sum() > 0:
    print(f"Distribución target_cards_ou35:")
    print(merge_directo['target_cards_ou35'].value_counts(dropna=True))
    print(f"Distribución target_redcard:")
    print(merge_directo['target_redcard'].value_counts(dropna=True))
print(f"\n💾 Guardado en: {out_path}")