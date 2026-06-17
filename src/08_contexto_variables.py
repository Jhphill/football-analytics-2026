"""
08_contexto_variables.py
======================================================================
Agrega tres grupos de variables de contexto a matches_features_v2.csv:

  1. CONFEDERACIONES
     - home_confed / away_confed / fav_confed / dog_confed
     - mismo_confed (0/1): ambos equipos de la misma confederación
     Cubre las 48 selecciones del Mundial 2026 + selecciones históricas
     relevantes del dataset.

  2. TOURNAMENT_WEIGHT
     Peso numérico del torneo (importancia del contexto competitivo).
     Usa is_world_cup / is_world_cup_qualifier / is_continental / _tournament.
       - FIFA World Cup (fase final)      : 2.5
       - Copa continental (Euro, Copa Am.): 2.0
       - Clasificatoria mundialista       : 1.5
       - Clasificatoria continental       : 1.2
       - Amistoso oficial / otro torneo   : 1.0

  3. DÍAS DE DESCANSO
     - home_dias_descanso / away_dias_descanso
       Días transcurridos desde el partido anterior de cada selección
       (cualquier torneo). Primer partido de una selección -> NaN.
     - fav_dias_descanso / dog_dias_descanso (re-derivados)
     - descanso_diff = fav_dias_descanso - dog_dias_descanso

Output: sobreescribe data/processed/matches_features_v2.csv
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalizacion_nombres import normalizar_nombre

print("=" * 70)
print(" 08_contexto_variables.py - Confederaciones, peso torneo, descanso")
print("=" * 70)

# ============================================================
# CARGAR
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df = pd.read_csv(in_path)
df['_date'] = pd.to_datetime(df['_date'])
print(f"\n✅ matches_features_v2.csv cargado: {df.shape}")

# ============================================================
# 1. CONFEDERACIONES
# ============================================================
print("\n--- 1. Confederaciones ---")

CONFEDERACIONES = {
    # CONMEBOL (10)
    'Brazil': 'CONMEBOL',
    'Argentina': 'CONMEBOL',
    'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL',
    'Ecuador': 'CONMEBOL',
    'Paraguay': 'CONMEBOL',
    'Chile': 'CONMEBOL',
    'Peru': 'CONMEBOL',
    'Bolivia': 'CONMEBOL',
    'Venezuela': 'CONMEBOL',

    # CONCACAF — clasificadas/probables al Mundial 2026
    'USA': 'CONCACAF',
    'Mexico': 'CONCACAF',
    'Canada': 'CONCACAF',
    'Costa Rica': 'CONCACAF',
    'Panama': 'CONCACAF',
    'Honduras': 'CONCACAF',
    'Jamaica': 'CONCACAF',
    'Guatemala': 'CONCACAF',
    'El Salvador': 'CONCACAF',
    'Trinidad and Tobago': 'CONCACAF',
    'Haiti': 'CONCACAF',
    'Curacao': 'CONCACAF',
    'Cuba': 'CONCACAF',
    'Puerto Rico': 'CONCACAF',

    # UEFA — clasificadas/probables al Mundial 2026
    'France': 'UEFA',
    'Spain': 'UEFA',
    'England': 'UEFA',
    'Germany': 'UEFA',
    'Portugal': 'UEFA',
    'Netherlands': 'UEFA',
    'Belgium': 'UEFA',
    'Croatia': 'UEFA',
    'Italy': 'UEFA',
    'Switzerland': 'UEFA',
    'Denmark': 'UEFA',
    'Austria': 'UEFA',
    'Serbia': 'UEFA',
    'Poland': 'UEFA',
    'Ukraine': 'UEFA',
    'Turkey': 'UEFA',
    'Norway': 'UEFA',
    'Sweden': 'UEFA',
    'Scotland': 'UEFA',
    'Wales': 'UEFA',
    'Hungary': 'UEFA',
    'Czech Republic': 'UEFA',
    'Slovakia': 'UEFA',
    'Romania': 'UEFA',
    'Greece': 'UEFA',
    'Albania': 'UEFA',
    'Slovenia': 'UEFA',
    'North Macedonia': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA',
    'Iceland': 'UEFA',
    'Finland': 'UEFA',
    'Ireland': 'UEFA',
    'Northern Ireland': 'UEFA',
    'Russia': 'UEFA',
    'Georgia': 'UEFA',
    'Kosovo': 'UEFA',
    'Luxembourg': 'UEFA',
    'Estonia': 'UEFA',
    'Latvia': 'UEFA',
    'Lithuania': 'UEFA',
    'Moldova': 'UEFA',
    'Armenia': 'UEFA',
    'Azerbaijan': 'UEFA',
    'Belarus': 'UEFA',
    'Bulgaria': 'UEFA',
    'Montenegro': 'UEFA',
    'North Macedonia': 'UEFA',
    'Faroe Islands': 'UEFA',
    'Malta': 'UEFA',
    'Cyprus': 'UEFA',
    'Andorra': 'UEFA',
    'Liechtenstein': 'UEFA',
    'San Marino': 'UEFA',
    'Gibraltar': 'UEFA',

    # CAF — clasificadas/probables al Mundial 2026
    'Morocco': 'CAF',
    'Senegal': 'CAF',
    'Nigeria': 'CAF',
    'Egypt': 'CAF',
    'Tunisia': 'CAF',
    'Cameroon': 'CAF',
    'Ghana': 'CAF',
    'Ivory Coast': 'CAF',
    'Algeria': 'CAF',
    'Mali': 'CAF',
    'Burkina Faso': 'CAF',
    'Guinea': 'CAF',
    'Tanzania': 'CAF',
    'Zambia': 'CAF',
    'South Africa': 'CAF',
    'DR Congo': 'CAF',
    'Congo': 'CAF',
    'Kenya': 'CAF',
    'Uganda': 'CAF',
    'Cape Verde': 'CAF',
    'Benin': 'CAF',
    'Gabon': 'CAF',
    'Mozambique': 'CAF',
    'Angola': 'CAF',
    'Equatorial Guinea': 'CAF',
    'Mauritania': 'CAF',
    'Libya': 'CAF',
    'Ethiopia': 'CAF',
    'Zimbabwe': 'CAF',
    'Eswatini': 'CAF',

    # AFC — clasificadas/probables al Mundial 2026
    'Japan': 'AFC',
    'South Korea': 'AFC',
    'Australia': 'AFC',
    'Saudi Arabia': 'AFC',
    'Iran': 'AFC',
    'Qatar': 'AFC',
    'Jordan': 'AFC',
    'Iraq': 'AFC',
    'Uzbekistan': 'AFC',
    'China PR': 'AFC',
    'China': 'AFC',
    'Indonesia': 'AFC',
    'Thailand': 'AFC',
    'Vietnam': 'AFC',
    'UAE': 'AFC',
    'Oman': 'AFC',
    'Bahrain': 'AFC',
    'Kuwait': 'AFC',
    'Palestine': 'AFC',
    'India': 'AFC',
    'Philippines': 'AFC',
    'Tajikistan': 'AFC',
    'Kyrgyzstan': 'AFC',
    'Kazakhstan': 'AFC',
    'Hong Kong': 'AFC',

    # OFC
    'New Zealand': 'OFC',
    'Fiji': 'OFC',
    'Papua New Guinea': 'OFC',
    'Solomon Islands': 'OFC',
    'Vanuatu': 'OFC',
    'Tahiti': 'OFC',
    'New Caledonia': 'OFC',

    # Países disueltos (histórico)
    'GermanDR': 'UEFA',
    'Germany DR': 'UEFA',
    'Czechoslovakia': 'UEFA',
    'Yugoslavia': 'UEFA',
    'Soviet Union': 'UEFA',
    'German DR': 'UEFA',

    # Detectados en primera ejecución como faltantes en partidos de Mundial
    'Israel': 'UEFA',          # UEFA desde 1994
    'North Korea': 'AFC',
    'Togo': 'CAF',
    'United Arab Emirates': 'AFC',
}

df['home_confed'] = df['_home_team'].map(CONFEDERACIONES)
df['away_confed'] = df['_away_team'].map(CONFEDERACIONES)

# fav_confed / dog_confed
tiene_elo = df['home_elo'].notna() & df['away_elo'].notna()
home_es_fav = df['home_elo'] >= df['away_elo']

df['fav_confed'] = np.where(home_es_fav, df['home_confed'], df['away_confed'])
df['dog_confed'] = np.where(home_es_fav, df['away_confed'], df['home_confed'])

df['mismo_confed'] = (
    df['home_confed'].notna() &
    (df['home_confed'] == df['away_confed'])
).astype('Int64')

n_sin_confed = df['home_confed'].isna().sum() + df['away_confed'].isna().sum()
n_mundiales_sin = df.loc[df['is_world_cup'] == 1, 'home_confed'].isna().sum()

print(f"Equipos sin confederación asignada (total apariciones): {n_sin_confed}")
print(f"Partidos de Mundial sin confederación en home_team: {n_mundiales_sin}")
print(f"Distribución mismo_confed:")
print(df['mismo_confed'].value_counts(dropna=False))

# Listar los equipos sin confederación que aparecen en partidos de Mundial
if n_mundiales_sin > 0:
    sin_conf_wc = df.loc[
        (df['is_world_cup'] == 1) & df['home_confed'].isna(),
        '_home_team'
    ].unique()
    print(f"⚠️ Equipos de Mundial sin confederación: {sorted(sin_conf_wc)}")
    print("   Agrega estos al diccionario CONFEDERACIONES y vuelve a correr.")

# ============================================================
# 2. TOURNAMENT_WEIGHT
# ============================================================
print("\n--- 2. Tournament weight ---")

def calcular_tournament_weight(row):
    if row.get('is_world_cup', 0) == 1:
        return 2.5
    if row.get('is_world_cup_qualifier', 0) == 1:
        return 1.5
    if row.get('is_continental', 0) == 1:
        return 2.0
    # Clasificatorias continentales detectadas por el nombre del torneo
    t = str(row.get('_tournament', '')).lower()
    if any(kw in t for kw in [
        'qualification', 'qualifier', 'qualifying', 'eliminatoria'
    ]):
        return 1.2
    return 1.0

df['tournament_weight'] = df.apply(calcular_tournament_weight, axis=1)

print(f"Distribución tournament_weight:")
print(df['tournament_weight'].value_counts().sort_index())

# ============================================================
# 3. DÍAS DE DESCANSO
# ============================================================
print("\n--- 3. Días de descanso ---")

df = df.reset_index(drop=True)
df['_row_id'] = df.index

home_long = df[['_row_id', '_date', '_home_team']].rename(
    columns={'_home_team': 'team'}
)
home_long['side'] = 'home'

away_long = df[['_row_id', '_date', '_away_team']].rename(
    columns={'_away_team': 'team'}
)
away_long['side'] = 'away'

long_d = pd.concat([home_long, away_long], ignore_index=True)
long_d = long_d.sort_values(['team', '_date', '_row_id']).reset_index(drop=True)

# Días desde el partido anterior de ese equipo (shift(1) dentro de grupo)
long_d['dias_descanso'] = long_d.groupby('team')['_date'].diff().dt.days

# Pivotar de vuelta
home_descanso = (
    long_d[long_d['side'] == 'home']
    .set_index('_row_id')['dias_descanso']
    .rename('home_dias_descanso')
)
away_descanso = (
    long_d[long_d['side'] == 'away']
    .set_index('_row_id')['dias_descanso']
    .rename('away_dias_descanso')
)

df = df.set_index('_row_id')
df = df.join(home_descanso).join(away_descanso)
df = df.reset_index(drop=True)

# fav_dias_descanso / dog_dias_descanso
df['fav_dias_descanso'] = np.where(
    home_es_fav, df['home_dias_descanso'], df['away_dias_descanso']
)
df['dog_dias_descanso'] = np.where(
    home_es_fav, df['away_dias_descanso'], df['home_dias_descanso']
)
df['descanso_diff'] = df['fav_dias_descanso'] - df['dog_dias_descanso']

print(f"Estadísticas home_dias_descanso:")
print(df['home_dias_descanso'].describe().round(1))

# Detectar valores anómalos (>365 días -> selecciones con largos períodos inactivos,
# probablemente equipos pequeños o períodos de guerra/pandemia)
anomalos = (df['home_dias_descanso'] > 365).sum()
print(f"Partidos con >365 días desde el partido anterior: {anomalos}")
print("(normal para selecciones pequeñas o períodos sin actividad internacional)")

# ============================================================
# 4. GUARDAR
# ============================================================
df = df.drop(columns=['_row_id'], errors='ignore')
df.to_csv(in_path, index=False)

print("\n" + "=" * 70)
print(" RESUMEN FINAL")
print("=" * 70)
print(f"Shape final: {df.shape}")
nuevas_cols = ['home_confed', 'away_confed', 'fav_confed', 'dog_confed',
               'mismo_confed', 'tournament_weight',
               'home_dias_descanso', 'away_dias_descanso',
               'fav_dias_descanso', 'dog_dias_descanso', 'descanso_diff']
print(f"Columnas nuevas agregadas ({len(nuevas_cols)}): {nuevas_cols}")
print(f"\n💾 Sobreescrito: {in_path}")

print("""
PRÓXIMO PASO (Juanfe, Semana 2 — casi lista):
  - distancia_sede_km / mismo_continente_sede (solo para WC2026,
    sobre future_match_probabilities_baseline.csv)
  - presion_situacional (función + validación con Mundiales pasados)
  - Regenerar diccionario_variables.md (ya tiene ~91 columnas)
""")