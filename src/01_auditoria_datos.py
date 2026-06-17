"""
01_auditoria_datos.py
======================================================================
Auditoría completa de datos - Plataforma Predictiva Mundial 2026

Este script NO modifica ni guarda archivos. Su único propósito es
diagnosticar el estado de los datasets crudos antes de construir el
pipeline de limpieza (03_limpieza_y_merge.py).

Chequeos incluidos:
  1. Carga y shapes de los datasets principales
  2. Diagnóstico de la columna is_world_cup
  3. Partidos faltantes en teams_match_features.csv (general)
  4. Partidos de FIFA World Cup faltantes (caso crítico)
  5. Nombres de equipo problemáticos (normalización)
  6. Verificación específica: Cabo Verde / Curaçao en player_aggregates
  7. Placeholders (playoffs) en future_match_probabilities_baseline.csv
  8. Verificación de data leakage en features de forma (rolling)
  9. Resumen final y checklist de acciones
======================================================================
"""

import pandas as pd
import os

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube de src/ a la raíz
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 70)
print(" AUDITORÍA COMPLETA DE DATOS - MUNDIAL 2026")
print("=" * 70)

# ============================================================
# 1. CARGAR DATASETS PRINCIPALES
# ============================================================
features_path = os.path.join(DATA_RAW, 'teams_match_features.csv')
results_path = os.path.join(DATA_RAW, 'results.csv')
future_path = os.path.join(DATA_RAW, 'future_match_probabilities_baseline.csv')
former_names_path = os.path.join(DATA_RAW, 'former_names.csv')
player_agg_path = os.path.join(DATA_RAW, 'player_aggregates.csv')
elo_path = os.path.join(DATA_RAW, 'eloratings.csv')

df_features = pd.read_csv(features_path)
df_results = pd.read_csv(results_path) if os.path.exists(results_path) else None
df_future = pd.read_csv(future_path) if os.path.exists(future_path) else None
df_former = pd.read_csv(former_names_path) if os.path.exists(former_names_path) else None
df_pa = pd.read_csv(player_agg_path) if os.path.exists(player_agg_path) else None
df_elo = pd.read_csv(elo_path) if os.path.exists(elo_path) else None

print(f"\n✅ teams_match_features.csv shape: {df_features.shape}")
if df_results is not None:
    print(f"✅ results.csv shape: {df_results.shape}")
if df_future is not None:
    print(f"✅ future_match_probabilities_baseline.csv shape: {df_future.shape}")
if df_former is not None:
    print(f"✅ former_names.csv shape: {df_former.shape}")
if df_pa is not None:
    print(f"✅ player_aggregates.csv shape: {df_pa.shape}")
if df_elo is not None:
    print(f"✅ eloratings.csv shape: {df_elo.shape}")

# ============================================================
# 2. DIAGNÓSTICO DE LA COLUMNA 'is_world_cup'
# ============================================================
print("\n" + "=" * 70)
print(" 1. DIAGNÓSTICO DE LA COLUMNA 'is_world_cup'")
print("=" * 70)

print(f"Valores únicos en is_world_cup: {df_features['is_world_cup'].unique()}")
print(f"Conteo: {df_features['is_world_cup'].value_counts().to_dict()}")

torneos = df_features['_tournament'].unique()
mundiales_like = [t for t in torneos if 'World' in str(t) or 'Cup' in str(t)]
print(f"\nTorneos con 'World' o 'Cup' en el nombre (primeros 10 de {len(mundiales_like)}):")
for t in mundiales_like[:10]:
    print(f"  - {t}")

is_wc_bug = df_features['is_world_cup'].nunique() == 1 and df_features['is_world_cup'].iloc[0] == 0
if is_wc_bug:
    print("\n⚠️ is_world_cup está completamente en 0 → se recalculará en 03_limpieza_y_merge.py")
    df_features['is_world_cup'] = df_features['_tournament'].str.contains(
        'World Cup', case=False, na=False
    ).astype(int)
    print(f"✅ Recalculado (solo para diagnóstico): {df_features['is_world_cup'].value_counts().to_dict()}")
else:
    print("\n✅ is_world_cup parece tener valores correctos.")

# ============================================================
# 3. PARTIDOS FALTANTES EN FEATURES (GENERAL)
# ============================================================
df_faltantes = None
if df_results is not None:
    print("\n" + "=" * 70)
    print(" 2. ANÁLISIS DE PARTIDOS FALTANTES EN FEATURES (GENERAL)")
    print("=" * 70)

    df_results['key'] = df_results['date'].astype(str) + '_' + df_results['home_team'] + '_' + df_results['away_team']
    df_features['key'] = df_features['_date'].astype(str) + '_' + df_features['_home_team'] + '_' + df_features['_away_team']

    faltantes = set(df_results['key']) - set(df_features['key'])
    print(f"Partidos en results.csv que NO están en features: {len(faltantes)}")

    df_faltantes = df_results[df_results['key'].isin(faltantes)].copy()

    print("\n📅 Distribución por año (últimos 20 años):")
    print(df_faltantes['date'].str[:4].value_counts().sort_index().tail(20))

    print("\n🏆 Torneos más frecuentes entre los faltantes:")
    print(df_faltantes['tournament'].value_counts().head(10))

    anios_faltantes = df_faltantes['date'].str[:4].astype(int)
    print(f"\n📊 Partidos faltantes posteriores a 2006: {(anios_faltantes > 2006).sum()}")
else:
    print("\n❌ No se encontró results.csv, omitiendo comparación general.")

# ============================================================
# 4. CASO CRÍTICO: PARTIDOS DE FIFA WORLD CUP FALTANTES
# ============================================================
faltantes_wc = None
if df_faltantes is not None:
    print("\n" + "=" * 70)
    print(" 3. CASO CRÍTICO: PARTIDOS DE 'FIFA World Cup' FALTANTES")
    print("=" * 70)

    total_wc = (df_results['tournament'] == 'FIFA World Cup').sum()
    faltantes_wc = df_faltantes[df_faltantes['tournament'] == 'FIFA World Cup'].copy()

    print(f"Partidos de Mundial en results.csv: {total_wc}")
    print(f"Partidos de Mundial faltantes en features: {len(faltantes_wc)}")
    print(f"Partidos de Mundial presentes en features: {total_wc - len(faltantes_wc)}")

    if len(faltantes_wc) > 0:
        print("\n🔍 Muestra de partidos de Mundial faltantes:")
        print(faltantes_wc[['date', 'home_team', 'away_team', 'tournament']].head(10).to_string(index=False))

        print("\n📅 Distribución por década de los Mundiales faltantes:")
        anios = faltantes_wc['date'].str[:4].astype(int)
        decadas = (anios // 10) * 10
        print(decadas.value_counts().sort_index())

# ============================================================
# 5. NOMBRES DE EQUIPO PROBLEMÁTICOS (NORMALIZACIÓN)
# ============================================================
equipos_problema = set()
if faltantes_wc is not None and len(faltantes_wc) > 0:
    print("\n" + "=" * 70)
    print(" 4. NOMBRES DE EQUIPO PROBLEMÁTICOS (no están en features)")
    print("=" * 70)

    equipos_faltantes = set(faltantes_wc['home_team']) | set(faltantes_wc['away_team'])
    equipos_en_features = set(df_features['_home_team']) | set(df_features['_away_team'])

    equipos_problema = equipos_faltantes - equipos_en_features
    print(f"Equipos con nombres que no calzan ({len(equipos_problema)}):")
    print(sorted(equipos_problema))

    # Cruce con former_names.csv para sugerir mapeos
    if df_former is not None:
        print("\n📋 Sugerencias desde former_names.csv:")
        for eq in sorted(equipos_problema):
            match_current = df_former[df_former['former'] == eq]
            match_former = df_former[df_former['current'] == eq]
            if len(match_current) > 0:
                print(f"  '{eq}' -> posible nombre actual: '{match_current['current'].values[0]}'")
            elif len(match_former) > 0:
                print(f"  '{eq}' -> tiene nombres anteriores: {match_former['former'].values.tolist()}")

# ============================================================
# 6. VERIFICACIÓN ESPECÍFICA: CABO VERDE / CURAÇAO
# ============================================================
print("\n" + "=" * 70)
print(" 5. VERIFICACIÓN: CABO VERDE Y CURAÇAO EN player_aggregates.csv")
print("=" * 70)
print("Estos dos debutan en el Mundial 2026 — su grafía debe coincidir")
print("exactamente con player_aggregates.csv para no perder atributos FIFA.\n")

if df_pa is not None:
    candidatos = {
        'Cabo Verde / Cape Verde': ['Cabo Verde', 'Cape Verde'],
        'Curaçao / Curacao': ['Curaçao', 'Curacao'],
        "Côte d'Ivoire / Ivory Coast": ["Côte d'Ivoire", 'Ivory Coast'],
        'Czechia / Czech Republic': ['Czechia', 'Czech Republic'],
        'Republic of Ireland / Ireland': ['Republic of Ireland', 'Ireland'],
    }

    for label, variantes in candidatos.items():
        print(f"  {label}:")
        for v in variantes:
            match = df_pa[df_pa['country'].str.contains(v, case=False, na=False, regex=False)]
            encontrados = match['country'].unique().tolist()
            status = "✅" if len(encontrados) > 0 else "❌"
            print(f"    {status} '{v}' -> {encontrados}")
        print()
else:
    print("❌ No se encontró player_aggregates.csv")

# ============================================================
# 7. PLACEHOLDERS EN future_match_probabilities_baseline.csv
# ============================================================
if df_future is not None:
    print("\n" + "=" * 70)
    print(" 6. PLACEHOLDERS (PLAYOFFS) EN DATASET FUTURO")
    print("=" * 70)

    placeholders = df_future[
        df_future['home_team'].str.contains('Playoff', na=False, case=False) |
        df_future['away_team'].str.contains('Playoff', na=False, case=False)
    ]
    print(f"Partidos con equipo no definido (Playoff): {len(placeholders)}")
    if len(placeholders) > 0:
        print(placeholders[['home_team', 'away_team', 'group']].to_string(index=False))
        print("\n⚠️ Estos se resuelven en marzo 2026 (repechajes). No bloqueante.")
    else:
        print("✅ No hay placeholders, todos los equipos están definidos.")
else:
    print("\n❌ No se encontró future_match_probabilities_baseline.csv")

# ============================================================
# 8. VERIFICACIÓN DE DATA LEAKAGE EN FEATURES DE FORMA
# ============================================================
print("\n" + "=" * 70)
print(" 7. VERIFICACIÓN DE DATA LEAKAGE EN FEATURES MÓVILES")
print("=" * 70)

# Inicializar variables que se usarán en el resumen
var_scored = None
var_winrate = None
leakage_check_ok = False  # bandera

equipo = 'Argentina'
df_equipo = df_features[
    (df_features['_home_team'] == equipo) | (df_features['_away_team'] == equipo)
].copy()
df_equipo = df_equipo.sort_values('_date')

cols_forma = ['home_form_scored', 'home_form_win_rate']
if all(c in df_equipo.columns for c in cols_forma):
    df_local = df_equipo[df_equipo['_home_team'] == equipo]
    if len(df_local) > 5:
        print(f"\n📊 Evolución de features de forma para {equipo} (como local, primeras 10 filas):")
        print(df_local[['_date', '_away_team'] + cols_forma].head(10).to_string(index=False))

        var_scored = df_local['home_form_scored'].var()
        var_winrate = df_local['home_form_win_rate'].var()

        if var_scored == 0 and var_winrate == 0:
            print("\n⚠️ ¡ALERTA! Columnas constantes -> posible data leakage (promedio global).")
            print("   Recalcular con rolling windows en 02_features.ipynb.")
        else:
            print("\n✅ Las features de forma varían partido a partido -> sin leakage aparente.")
            leakage_check_ok = True
    else:
        print(f"⚠️ Pocos partidos de {equipo} como local para evaluar.")
else:
    print("❌ No se encontraron columnas de forma esperadas.")
# ============================================================
# 9. RESUMEN FINAL Y CHECKLIST DE ACCIONES
# ============================================================
print("\n" + "=" * 70)
print(" RESUMEN Y CHECKLIST PARA 03_limpieza_y_merge.py")
print("=" * 70)

# Determinar estado del chequeo de data leakage
leakage_status = "PASÓ, dataset confiable como base."
if not leakage_check_ok:
    leakage_status = "FALLÓ (no se pudo verificar o constantes), recalcular con rolling windows"
elif var_scored is not None and var_scored == 0 and var_winrate == 0:
    leakage_status = "FALLÓ (columnas constantes), recalcular con rolling windows"

# Construir el resumen con manejo seguro de posibles variables no definidas
check_wc_bug = 'X' if is_wc_bug else ' '
check_wc_faltantes = 'X' if (faltantes_wc is not None and len(faltantes_wc) > 0) else ' '
num_faltantes = len(faltantes_wc) if faltantes_wc is not None else 0
check_equipos = 'X' if (equipos_problema and len(equipos_problema) > 0) else ' '
lista_equipos = sorted(equipos_problema) if equipos_problema else '(ninguno)'

resumen = f"""
[{check_wc_bug}] is_world_cup estaba en 0 -> recalcular con
    df['_tournament'].str.contains('World Cup', case=False, na=False)

[{check_wc_faltantes}] Recuperar partidos de FIFA World Cup faltantes
    ({num_faltantes} partidos) haciendo merge con player_aggregates.csv tras normalizar nombres.

[{check_equipos}] Aplicar NAME_FIXES (02_normalizacion_nombres.py) a:
    {lista_equipos}

[ ] Verificar manualmente el resultado de Cabo Verde/Curaçao (paso 5)
    y ajustar NAME_FIXES si la grafía no calza.

[X] Placeholders de playoffs detectados y documentados -> no bloqueante,
    se actualizan dinámicamente en marzo 2026.

[X] Chequeo de data leakage en features de forma -> {leakage_status}
"""

print(resumen)
print("✅ Auditoría completada.")