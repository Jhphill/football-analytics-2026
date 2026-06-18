"""
18_pipeline_wc2026_live.py
======================================================================
Pipeline de actualización jornada por jornada — Mundial 2026.

Por qué existe este script:
  - Los modelos XGBoost fueron entrenados con datos históricos (1872-2025).
  - Una vez que el Mundial arranca, los partidos ya jugados deben
    usarse para recalcular la FORMA RECIENTE de cada selección antes
    de predecir su próximo partido.
  - Este script lee los resultados reales del torneo (que se van
    llenando manualmente después de cada jornada) y recalcula:
      * form_scored / form_conceded / form_win_rate (últimos N partidos
        incluyendo los ya jugados del Mundial)
      * presion_situacional (tabla del grupo actualizada)
      * dias_descanso (días desde el último partido jugado)
  - El output es un CSV listo para que 17_predicciones_wc2026.py
    lo use con features actualizadas en vez de las medianas del train.

Archivos que mantenés actualizados manualmente:
  - data/processed/wc2026_resultados_live.csv    ← llenás después de cada partido
  - data/processed/wc2026_cuotas_1xbet.csv       ← llenás antes de cada partido

Flujo por jornada:
  1. Terminó la jornada -> llenás wc2026_resultados_live.csv
  2. Corrés: python src/18_pipeline_wc2026_live.py
  3. Antes del próximo partido -> llenás wc2026_cuotas_1xbet.csv
  4. Corrés: python src/17_predicciones_wc2026.py
  5. El dashboard muestra predicciones actualizadas con la forma real

Estructura de wc2026_resultados_live.csv:
  fecha, home_team, away_team, home_goals, away_goals, grupo, jornada
  2026-06-11, Mexico, South Africa, 2, 1, A, 1
  2026-06-12, USA, Paraguay, 4, 1, D, 1
  ...
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(SRC_DIR)
from normalizacion_nombres import normalizar_columna

N_FORMA = 10  # misma ventana que en 07_forma_reciente.py

print("=" * 70)
print(" 18_pipeline_wc2026_live.py")
print(" Actualización de forma reciente y presión situacional")
print(f" Ventana de forma: últimos {N_FORMA} partidos (histórico + WC2026)")
print("=" * 70)

# ============================================================
# 1. VERIFICAR QUE EXISTE EL CSV DE RESULTADOS LIVE
# ============================================================
live_path = os.path.join(DATA_PROCESSED, 'wc2026_resultados_live.csv')

if not os.path.exists(live_path):
    # Crear el template vacío para que el usuario lo llene
    template = pd.DataFrame(columns=[
        'fecha', 'home_team', 'away_team',
        'home_goals', 'away_goals', 'grupo', 'jornada'
    ])
    template.to_csv(live_path, index=False)
    print(f"\n⚠️  No se encontró wc2026_resultados_live.csv")
    print(f"   Se creó el template vacío en: {live_path}")
    print("""
   INSTRUCCIONES:
   Abrí el archivo y llenalo con los resultados ya jugados. Ejemplo:
   fecha,home_team,away_team,home_goals,away_goals,grupo,jornada
   2026-06-11,Mexico,South Africa,2,1,A,1
   2026-06-12,United States,Paraguay,4,1,D,1
   
   Después de llenarlo, volvé a correr este script.
""")
    sys.exit(0)

df_live = pd.read_csv(live_path)
df_live['fecha'] = pd.to_datetime(df_live['fecha'])

if len(df_live) == 0:
    print("\n⚠️  wc2026_resultados_live.csv está vacío.")
    print("   Llená los resultados de los partidos ya jugados y volvé a correr.")
    sys.exit(0)

# Normalizar nombres
df_live = normalizar_columna(df_live, 'home_team')
df_live = normalizar_columna(df_live, 'away_team')

print(f"\n✅ Resultados cargados: {len(df_live)} partidos jugados")
print(f"   Desde: {df_live['fecha'].min().date()} hasta {df_live['fecha'].max().date()}")
print(f"   Jornadas cubiertas: {sorted(df_live['jornada'].unique())}")

# ============================================================
# 2. CARGAR HISTÓRICO (para calcular forma incluyendo partidos
#    anteriores al Mundial del equipo)
# ============================================================
historico_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df_hist = pd.read_csv(historico_path, usecols=[
    '_date', '_home_team', '_away_team', 'home_goals', 'away_goals'
])
df_hist['_date'] = pd.to_datetime(df_hist['_date'])
df_hist = df_hist.rename(columns={
    '_date': 'fecha', '_home_team': 'home_team', '_away_team': 'away_team'
})

print(f"\n✅ Histórico cargado: {len(df_hist):,} partidos (hasta pre-Mundial)")

# ============================================================
# 3. COMBINAR HISTÓRICO + RESULTADOS LIVE
# ============================================================
df_combinado = pd.concat([
    df_hist[['fecha', 'home_team', 'away_team', 'home_goals', 'away_goals']],
    df_live[['fecha', 'home_team', 'away_team', 'home_goals', 'away_goals']],
], ignore_index=True).sort_values('fecha').reset_index(drop=True)

print(f"\nDataset combinado: {len(df_combinado):,} partidos (histórico + WC2026)")

# ============================================================
# 4. CALCULAR FORMA RECIENTE POR EQUIPO (igual que 07_forma_reciente.py)
# ============================================================
df_combinado = df_combinado.reset_index().rename(columns={'index': '_row_id'})

home_long = df_combinado[['_row_id', 'fecha', 'home_team', 'home_goals', 'away_goals']].copy()
home_long.columns = ['_row_id', 'fecha', 'team', 'goals_for', 'goals_against']
home_long['side'] = 'home'

away_long = df_combinado[['_row_id', 'fecha', 'away_team', 'away_goals', 'home_goals']].copy()
away_long.columns = ['_row_id', 'fecha', 'team', 'goals_for', 'goals_against']
away_long['side'] = 'away'

long_df = pd.concat([home_long, away_long], ignore_index=True)
long_df = long_df.sort_values(['team', 'fecha', '_row_id']).reset_index(drop=True)
long_df['win'] = (long_df['goals_for'] > long_df['goals_against']).astype(float)

grp = long_df.groupby('team', group_keys=False)
long_df['form_scored']   = grp['goals_for'].apply(
    lambda s: s.shift(1).rolling(window=N_FORMA, min_periods=1).mean()
)
long_df['form_conceded'] = grp['goals_against'].apply(
    lambda s: s.shift(1).rolling(window=N_FORMA, min_periods=1).mean()
)
long_df['form_win_rate'] = grp['win'].apply(
    lambda s: s.shift(1).rolling(window=N_FORMA, min_periods=1).mean()
)

# Días de descanso desde el partido anterior
long_df['dias_descanso'] = long_df.groupby('team')['fecha'].diff().dt.days

print(f"\n✅ Forma reciente recalculada con N={N_FORMA} para {long_df['team'].nunique()} selecciones")

# ============================================================
# 5. EXTRAER LA FORMA MÁS RECIENTE POR EQUIPO
#    (el estado actual = últimas features calculadas para cada equipo)
# ============================================================
# Tomar el último registro de cada equipo (su forma ACTUAL)
forma_actual = (
    long_df.sort_values(['team', 'fecha', '_row_id'])
    .groupby('team')
    .last()
    .reset_index()
    [['team', 'form_scored', 'form_conceded', 'form_win_rate', 'dias_descanso', 'fecha']]
)
forma_actual.columns = [
    'team', 'form_scored', 'form_conceded', 'form_win_rate',
    'dias_descanso_desde_ultimo', 'ultimo_partido'
]

print(f"\n--- Forma actual de los equipos del Mundial ---")
print(forma_actual.to_string(index=False))

# ============================================================
# 6. CALCULAR TABLA DE GRUPOS Y PRESIÓN SITUACIONAL
# ============================================================
print(f"\n--- Tabla de grupos (partidos jugados) ---")

# Calcular puntos de cada equipo en el Mundial
filas_tabla = []
for _, row in df_live.iterrows():
    hg, ag = int(row['home_goals']), int(row['away_goals'])
    if hg > ag:
        ph, pa = 3, 0
    elif hg < ag:
        ph, pa = 0, 3
    else:
        ph, pa = 1, 1

    filas_tabla.append({
        'grupo': row['grupo'], 'team': row['home_team'],
        'pj': 1, 'puntos': ph, 'gf': hg, 'gc': ag, 'gd': hg - ag
    })
    filas_tabla.append({
        'grupo': row['grupo'], 'team': row['away_team'],
        'pj': 1, 'puntos': pa, 'gf': ag, 'gc': hg, 'gd': ag - hg
    })

df_tabla = (
    pd.DataFrame(filas_tabla)
    .groupby(['grupo', 'team'])
    .sum()
    .reset_index()
    .sort_values(['grupo', 'puntos', 'gd', 'gf'], ascending=[True, False, False, False])
)

# Partidos totales en fase de grupos por equipo (3 cada uno)
PARTIDOS_TOTALES_GRUPO = 3

def calcular_presion(row):
    """
    Determina el estado de presión situacional según puntos y partidos jugados.
    Misma lógica que presion_situacional.py.
    """
    pj = row['pj']
    pts = row['puntos']
    pr = PARTIDOS_TOTALES_GRUPO - pj  # partidos restantes

    if pj == 0:
        return 'normal'

    # Máximo de puntos alcanzables
    max_alcanzable = pts + pr * 3

    if pts >= 7:  # ya clasificado cómodamente (imposible en 3 partidos, pero por si acaso)
        return 'calificado_comodo'
    elif pts >= 4 and pj >= 2:
        return 'calificado_ajustado'
    elif max_alcanzable < 4 and pj >= 2:
        return 'eliminado'
    elif pts == 0 and pj == 2:
        return 'necesita_ganar_y_milagro'
    elif pts <= 1 and pj >= 2:
        return 'necesita_ganar'
    else:
        return 'normal'

df_tabla['presion'] = df_tabla.apply(calcular_presion, axis=1)

print(df_tabla[['grupo', 'team', 'pj', 'puntos', 'gd', 'presion']].to_string(index=False))

# ============================================================
# 7. CONSTRUIR EL DATASET DE FEATURES ACTUALIZADO PARA 17_predicciones
# ============================================================
# Cargar el CSV de contexto geográfico (base de los 72 partidos)
contexto_path = os.path.join(DATA_PROCESSED, 'wc2026_contexto_geografico.csv')
df_contexto = pd.read_csv(contexto_path)
df_contexto.columns = df_contexto.columns.str.strip()
df_contexto['home_team'] = df_contexto['home_team'].str.strip()
df_contexto['away_team'] = df_contexto['away_team'].str.strip()

# Filtrar solo partidos no jugados aún (sin resultado en live)
partidos_jugados_keys = set(
    df_live['home_team'].str.strip() + '_' + df_live['away_team'].str.strip()
)
df_contexto['_key'] = df_contexto['home_team'] + '_' + df_contexto['away_team']
df_pendientes = df_contexto[~df_contexto['_key'].isin(partidos_jugados_keys)].copy()
df_pendientes = df_pendientes[
    ~df_pendientes['home_team'].str.contains('Playoff|Interconf', na=False) &
    ~df_pendientes['away_team'].str.contains('Playoff|Interconf', na=False)
].copy()

print(f"\nPartidos pendientes (sin resultado aún): {len(df_pendientes)}")
print(f"Partidos ya jugados (excluidos): {len(df_live)}")

# Mergear forma reciente del home team
df_pendientes = df_pendientes.merge(
    forma_actual.rename(columns={
        'team': 'home_team',
        'form_scored': 'home_form_scored_live',
        'form_conceded': 'home_form_conceded_live',
        'form_win_rate': 'home_form_win_rate_live',
        'dias_descanso_desde_ultimo': 'home_dias_descanso_live',
    })[['home_team', 'home_form_scored_live', 'home_form_conceded_live',
        'home_form_win_rate_live', 'home_dias_descanso_live']],
    on='home_team', how='left'
)

# Mergear forma reciente del away team
df_pendientes = df_pendientes.merge(
    forma_actual.rename(columns={
        'team': 'away_team',
        'form_scored': 'away_form_scored_live',
        'form_conceded': 'away_form_conceded_live',
        'form_win_rate': 'away_form_win_rate_live',
        'dias_descanso_desde_ultimo': 'away_dias_descanso_live',
    })[['away_team', 'away_form_scored_live', 'away_form_conceded_live',
        'away_form_win_rate_live', 'away_dias_descanso_live']],
    on='away_team', how='left'
)

# Mergear presión situacional
df_pendientes = df_pendientes.merge(
    df_tabla[['team', 'presion', 'puntos', 'pj']].rename(columns={
        'team': 'home_team', 'presion': 'home_presion',
        'puntos': 'home_puntos', 'pj': 'home_pj'
    }),
    on='home_team', how='left'
)
df_pendientes = df_pendientes.merge(
    df_tabla[['team', 'presion', 'puntos', 'pj']].rename(columns={
        'team': 'away_team', 'presion': 'away_presion',
        'puntos': 'away_puntos', 'pj': 'away_pj'
    }),
    on='away_team', how='left'
)

# Rellenar equipos sin partidos jugados aún (jornada 1 de ese grupo)
df_pendientes['home_presion'] = df_pendientes['home_presion'].fillna('normal')
df_pendientes['away_presion'] = df_pendientes['away_presion'].fillna('normal')

# ============================================================
# 8. GUARDAR EL DATASET ACTUALIZADO
# ============================================================
out_path = os.path.join(DATA_PROCESSED, 'wc2026_features_live.csv')
df_pendientes = df_pendientes.drop(columns=['_key'], errors='ignore')
df_pendientes.to_csv(out_path, index=False)

# Guardar también la tabla de grupos y la forma actual
tabla_path = os.path.join(DATA_PROCESSED, 'wc2026_tabla_grupos.csv')
df_tabla.to_csv(tabla_path, index=False)

forma_path = os.path.join(DATA_PROCESSED, 'wc2026_forma_actual.csv')
forma_actual.to_csv(forma_path, index=False)

print(f"\n{'='*70}")
print(f" RESUMEN")
print(f"{'='*70}")
print(f"Partidos jugados procesados: {len(df_live)}")
print(f"Partidos pendientes con features actualizadas: {len(df_pendientes)}")
print(f"\n💾 {out_path}   ← features actualizadas (input para 17_predicciones_wc2026.py)")
print(f"💾 {tabla_path}  ← tabla de grupos con presión situacional")
print(f"💾 {forma_path}  ← forma reciente actual de cada selección")

print(f"""
PRÓXIMO PASO:
  1. Llenás las cuotas en: data/processed/wc2026_cuotas_1xbet.csv
  2. Corrés: python src/17_predicciones_wc2026.py
     (el script detecta wc2026_features_live.csv y usa la forma
      actualizada en vez de las medianas del train)

FLUJO COMPLETO POR JORNADA:
  Después de cada jornada:
    -> Agregás los resultados a wc2026_resultados_live.csv
    -> python src/18_pipeline_wc2026_live.py

  Antes del siguiente partido:
    -> Actualizás las cuotas en wc2026_cuotas_1xbet.csv
    -> python src/17_predicciones_wc2026.py
    -> El dashboard muestra predicciones con forma real del torneo
""")
