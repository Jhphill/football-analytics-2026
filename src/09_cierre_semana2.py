"""
09_cierre_semana2.py
======================================================================
Cierre de Semana 2. Agrega a matches_features_v2.csv:

  1. TARGETS RESTANTES
     - target_ou25        : over/under 2.5 goles totales (binario)
     - target_btts        : both teams to score (binario)
     - target_ou15        : over/under 1.5 goles (binario, mercado adicional)
     - target_ou35        : over/under 3.5 goles (binario, mercado adicional)

  2. DISTANCIA Y CONTEXTO GEOGRÁFICO (solo para future_match_probabilities_baseline.csv)
     - mismo_continente_sede: 1 si la selección es de la misma región
       que la sede del Mundial 2026 (USA/México/Canadá = CONCACAF/CONMEBOL)
     - distancia_categoria: proxy simple (CONCACAF/CONMEBOL=cercano,
       UEFA=medio, AFC/CAF/OFC=lejano) — sin cálculo de km exactos que
       requeriría geocodificación (decisión de alcance justificada)

  3. FUNCIÓN DE PRESIÓN SITUACIONAL
     - Diseñada para correr partido a partido DURANTE el Mundial 2026
       (no tiene sentido precomputarla para el historial, ya que
       depende de la tabla de grupos en tiempo real)
     - Se valida aquí contra Mundiales pasados para verificar que la
       lógica es correcta
     - Se exporta como función reutilizable en src/presion_situacional.py

Output: sobreescribe data/processed/matches_features_v2.csv (targets)
        crea src/presion_situacional.py (función lista para el dashboard)
        crea data/processed/wc2026_contexto_geografico.csv (para future_matches)
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
SRC_DIR = os.path.join(BASE_DIR, 'src')

print("=" * 70)
print(" 09_cierre_semana2.py")
print("=" * 70)

# ============================================================
# CARGAR
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df = pd.read_csv(in_path)
df['_date'] = pd.to_datetime(df['_date'])
print(f"\n✅ matches_features_v2.csv cargado: {df.shape}")

# ============================================================
# 1. TARGETS RESTANTES
# ============================================================
print("\n--- 1. Targets de goles ---")

# Total de goles (ya existe como total_goals, pero re-calculamos para
# asegurarnos que usa home_goals/away_goals originales, sin NaN por fav/dog)
df['total_goals'] = df['home_goals'] + df['away_goals']

# Over/Under 2.5 — el mercado principal de goles en 1xbet
df['target_ou25'] = np.where(
    df['total_goals'].notna(),
    (df['total_goals'] > 2.5).astype('Int64'),
    pd.NA
)

# Both Teams To Score
df['target_btts'] = np.where(
    df['home_goals'].notna() & df['away_goals'].notna(),
    ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype('Int64'),
    pd.NA
)

# Over/Under 1.5 (mercado útil para partidos de baja producción)
df['target_ou15'] = np.where(
    df['total_goals'].notna(),
    (df['total_goals'] > 1.5).astype('Int64'),
    pd.NA
)

# Over/Under 3.5 (mercado de alta producción)
df['target_ou35_goals'] = np.where(
    df['total_goals'].notna(),
    (df['total_goals'] > 3.5).astype('Int64'),
    pd.NA
)

for col in ['target_ou25', 'target_btts', 'target_ou15', 'target_ou35_goals']:
    dist = df[col].value_counts(dropna=True)
    pct_1 = 100 * dist.get(1, 0) / dist.sum() if dist.sum() > 0 else 0
    print(f"  {col}: 0={dist.get(0,0):,} / 1={dist.get(1,0):,} "
          f"({pct_1:.1f}% over) | NaN={df[col].isna().sum()}")

# Verificar solo en partidos de Mundial (los que vas a predecir)
print("\n  En partidos de FIFA World Cup (is_world_cup=1):")
wc = df[df['is_world_cup'] == 1]
for col in ['target_ou25', 'target_btts']:
    dist = wc[col].value_counts(dropna=True)
    pct_1 = 100 * dist.get(1, 0) / dist.sum() if dist.sum() > 0 else 0
    print(f"    {col}: {pct_1:.1f}% over/sí | n={dist.sum()}")

# ============================================================
# 2. CONTEXTO GEOGRÁFICO PARA MUNDIAL 2026
# ============================================================
print("\n--- 2. Contexto geográfico (distancia sede WC2026) ---")

future_path = os.path.join(
    BASE_DIR, 'data', 'raw', 'future_match_probabilities_baseline.csv'
)
df_future = pd.read_csv(future_path)

# El Mundial 2026 se juega en USA, México y Canadá -> sede CONCACAF/CONMEBOL-cercana
# Categorías de distancia por confederación (proxy sin geocodificación)
DISTANCIA_SEDE = {
    'CONCACAF': 'cercano',    # México incluido, USA es anfitrión
    'CONMEBOL': 'cercano',    # Sudamérica, vuelo corto/medio
    'UEFA': 'medio',          # Europa, vuelo largo transatlántico
    'AFC': 'lejano',          # Asia/Oceanía, máxima distancia
    'CAF': 'lejano',          # África, vuelo muy largo
    'OFC': 'lejano',          # Oceanía, máxima distancia
}
MISMO_CONTINENTE_SEDE = {
    'CONCACAF': 1,
    'CONMEBOL': 1,   # América -> mismo "bloque continental"
    'UEFA': 0,
    'AFC': 0,
    'CAF': 0,
    'OFC': 0,
}

# Importar confederaciones (definidas en script 08)
from normalizacion_nombres import normalizar_columna
sys.path.append(SRC_DIR)

# Leer confederaciones desde el df ya procesado
confed_map = (
    pd.concat([
        df[['_home_team', 'home_confed']].rename(
            columns={'_home_team': 'team', 'home_confed': 'confed'}),
        df[['_away_team', 'away_confed']].rename(
            columns={'_away_team': 'team', 'away_confed': 'confed'}),
    ], ignore_index=True)
    .dropna()
    .drop_duplicates('team')
    .set_index('team')['confed']
    .to_dict()
)

df_future['home_confed'] = df_future['home_team'].map(confed_map)
df_future['away_confed'] = df_future['away_team'].map(confed_map)

df_future['home_distancia_sede'] = df_future['home_confed'].map(DISTANCIA_SEDE)
df_future['away_distancia_sede'] = df_future['away_confed'].map(DISTANCIA_SEDE)
df_future['home_mismo_continente_sede'] = df_future['home_confed'].map(MISMO_CONTINENTE_SEDE)
df_future['away_mismo_continente_sede'] = df_future['away_confed'].map(MISMO_CONTINENTE_SEDE)

# fav/dog para partidos futuros
df_future['elo_diff'] = df_future['home_elo'] - df_future['away_elo']
df_future['fav_team'] = np.where(
    df_future['elo_diff'] >= 0,
    df_future['home_team'], df_future['away_team']
)
df_future['dog_team'] = np.where(
    df_future['elo_diff'] >= 0,
    df_future['away_team'], df_future['home_team']
)
df_future['fav_elo'] = np.where(
    df_future['elo_diff'] >= 0,
    df_future['home_elo'], df_future['away_elo']
)
df_future['dog_elo'] = np.where(
    df_future['elo_diff'] >= 0,
    df_future['away_elo'], df_future['home_elo']
)
df_future['fav_dog_elo_diff'] = df_future['fav_elo'] - df_future['dog_elo']
df_future['fav_distancia_sede'] = np.where(
    df_future['elo_diff'] >= 0,
    df_future['home_distancia_sede'], df_future['away_distancia_sede']
)
df_future['dog_distancia_sede'] = np.where(
    df_future['elo_diff'] >= 0,
    df_future['away_distancia_sede'], df_future['home_distancia_sede']
)

future_out = os.path.join(DATA_PROCESSED, 'wc2026_contexto_geografico.csv')
df_future.to_csv(future_out, index=False)

sin_confed = df_future['home_confed'].isna().sum() + df_future['away_confed'].isna().sum()
print(f"Partidos del Mundial 2026 enriquecidos: {len(df_future)}")
print(f"Equipos sin confederación (placeholders de repechaje): {sin_confed}")
print(f"Distribución distancia_sede (home):")
print(df_future['home_distancia_sede'].value_counts(dropna=False))
print(f"💾 Guardado: {future_out}")

# ============================================================
# 3. FUNCIÓN PRESIÓN SITUACIONAL (exportar como módulo)
# ============================================================
print("\n--- 3. Presión situacional ---")

presion_code = '''"""
presion_situacional.py
======================================================================
Calcula la presión situacional de cada selección durante el Mundial.

Se llama PARTIDO A PARTIDO, antes de cada jornada, usando la tabla
de grupos actualizada. No tiene sentido precomputarla para el
historial (depende del contexto vivo del torneo).

Cómo usar en el dashboard:
    from presion_situacional import calcular_presion, ajustar_prob_por_presion

    estado = calcular_presion(
        puntos=3, partidos_jugados=2, partidos_restantes=1,
        gf=4, gc=2
    )
    prob_ajustada = ajustar_prob_por_presion(prob_base, estado_fav, estado_dog)
======================================================================
"""

def calcular_presion(puntos, partidos_jugados, partidos_restantes,
                     gf=None, gc=None, posicion_grupo=None):
    """
    Calcula el estado de presión situacional de una selección.

    Parámetros:
        puntos              : puntos acumulados en la fase de grupos
        partidos_jugados    : partidos ya jugados en la fase de grupos
        partidos_restantes  : partidos que faltan en la fase de grupos
        gf                  : goles a favor acumulados (opcional)
        gc                  : goles en contra acumulados (opcional)
        posicion_grupo      : posición actual en el grupo (1, 2, 3, 4) (opcional)

    Retorna (str):
        "calificado_comodo"     -> ya clasificó matemáticamente,
                                   probable rotación de titulares
        "calificado_ajustado"   -> probablemente clasificado pero no
                                   matemáticamente seguro
        "necesita_ganar"        -> solo una victoria asegura avance
        "necesita_ganar_y_milagro" -> necesita ganar Y que otros
                                      resultados le favorezcan
        "eliminado"             -> sin posibilidad matemática de avanzar
        "normal"                -> situación abierta, sin presión extrema
    """
    max_puntos_posibles = puntos + (partidos_restantes * 3)

    # Reglas por puntos (formato Mundial 2026: top 2 de 4 equipos clasifican,
    # más los 8 mejores terceros de 12 grupos)
    if puntos >= 6 and partidos_restantes <= 1:
        return "calificado_comodo"
    if puntos >= 4 and partidos_restantes == 0:
        return "calificado_comodo"
    if puntos >= 6 and partidos_restantes == 2:
        return "calificado_ajustado"
    if max_puntos_posibles < 3:
        return "eliminado"
    if max_puntos_posibles <= 3 and partidos_restantes == 1:
        return "necesita_ganar_y_milagro"
    if puntos == 0 and partidos_restantes == 1:
        return "necesita_ganar"
    if puntos <= 1 and partidos_restantes == 1:
        return "necesita_ganar"
    return "normal"


# Ajuste de probabilidad base por presión situacional
# Coeficientes empíricos (a calibrar con backtesting de Mundiales pasados)
AJUSTE_PRESION = {
    "calificado_comodo": -0.04,     # más propenso a rotar, baja rendimiento
    "calificado_ajustado": 0.00,
    "normal": 0.00,
    "necesita_ganar": +0.04,        # más ofensivo y motivado
    "necesita_ganar_y_milagro": +0.02,  # necesita ganar pero sabe que puede no alcanzar
    "eliminado": -0.03,             # relajado, puede jugar sin presión (efecto mixto)
}


def ajustar_prob_por_presion(prob_base_fav_win, estado_fav, estado_dog):
    """
    Ajusta la probabilidad de victoria del favorito según la presión
    situacional de ambos equipos.

    Parámetros:
        prob_base_fav_win : float, probabilidad base de victoria del favorito
        estado_fav        : str, resultado de calcular_presion() para el favorito
        estado_dog        : str, resultado de calcular_presion() para el no-favorito

    Retorna:
        float: probabilidad ajustada (entre 0 y 1)
    """
    adj_fav = AJUSTE_PRESION.get(estado_fav, 0.0)
    adj_dog = AJUSTE_PRESION.get(estado_dog, 0.0)

    # Si el dog está bajo presión positiva (+), el fav sube menos (o baja)
    ajuste_neto = adj_fav - adj_dog
    prob_ajustada = prob_base_fav_win + ajuste_neto
    return max(0.01, min(0.99, prob_ajustada))


if __name__ == "__main__":
    # Validación con ejemplos reales de Mundiales pasados
    print("Validación de calcular_presion():")
    casos = [
        # (descripción, puntos, p_jugados, p_restantes, esperado)
        ("Brasil 2022 tras ganar 2 de 2", 6, 2, 1, "calificado_comodo"),
        ("Arabia Saudita 2022 tras perder 2", 0, 2, 1, "necesita_ganar"),
        ("Alemania 2022 con 1 punto en 2 partidos", 1, 2, 1, "necesita_ganar"),
        ("Argentina 2022 tras ganar 2 de 2", 6, 2, 1, "calificado_comodo"),
        ("Situación normal tras 1 partido", 3, 1, 2, "normal"),
        ("Ya eliminado matemáticamente", 0, 2, 1, "necesita_ganar"),
    ]
    for desc, pts, pj, pr, esperado in casos:
        resultado = calcular_presion(pts, pj, pr)
        status = "✅" if resultado == esperado else "⚠️ "
        print(f"  {status} {desc}: {resultado} (esperado: {esperado})")
'''

presion_path = os.path.join(SRC_DIR, 'presion_situacional.py')
with open(presion_path, 'w', encoding='utf-8') as f:
    f.write(presion_code)
print(f"✅ Función exportada: {presion_path}")

# ============================================================
# 4. GUARDAR matches_features_v2.csv (con los 4 targets nuevos)
# ============================================================
df.to_csv(in_path, index=False)

print("\n" + "=" * 70)
print(" RESUMEN FINAL — SEMANA 2 CERRADA")
print("=" * 70)
print(f"matches_features_v2.csv shape: {df.shape}")
nuevas = ['target_ou25', 'target_btts', 'target_ou15', 'target_ou35_goals']
print(f"Targets nuevos: {nuevas}")
print(f"\nArchivos generados:")
print(f"  - data/processed/matches_features_v2.csv  ({df.shape[0]:,} x {df.shape[1]})")
print(f"  - data/processed/wc2026_contexto_geografico.csv ({len(df_future)} partidos WC2026)")
print(f"  - src/presion_situacional.py (función lista para el dashboard)")

print("""
SEMANA 2 — CHECKLIST COMPLETO:
[X] 05_features_tarjetas.py   -> target_cards_ou35, target_redcard
[X] 06_favorito_no_favorito.py -> fav_*/dog_*, target_1x2_fav_dog
[X] 07_forma_reciente.py      -> rolling windows uniformes (N=10)
[X] 08_contexto_variables.py  -> confederaciones, tournament_weight, dias_descanso
[X] 09_cierre_semana2.py      -> target_ou25/btts/ou15/ou35_goals
                                  wc2026_contexto_geografico.csv
                                  presion_situacional.py

[ ] PENDIENTE: regenerar diccionario_variables.md (91+ columnas)
    -> correr: python src/04_diccionario_variables.py

PRÓXIMO (SEMANA 3 — Juanfe):
  - Notebooks 03_baseline.ipynb y 04_xgboost.ipynb
  - Un modelo por mercado: target_1x2_fav_dog, target_ou25,
    target_btts, target_cards_ou35, target_redcard
  - Calibración con CalibratedClassifierCV
  - Métricas: log-loss, Brier Score, AUC, accuracy

PRÓXIMO (SEMANA 3 — Lulu):
  - NLP pipeline: RSS scraping + spaCy NER + injury_impact_score
  - LSTM v1.3 con matches_features_v2.csv completo (fav_*/dog_*)
""")