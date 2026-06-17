"""
14_cierre_semana3.py
======================================================================
Cierre de Semana 3. Consolida todas las métricas de los modelos
entrenados, genera una tabla resumen única, documenta la decisión
por mercado para el dashboard, y prepara el directorio models/ para
la Semana 4 (XAI con SHAP + value bet detector).

Scripts que deben haber corrido antes (en orden):
  10_preparar_datos_modelado.py
  11_modelo_1x2.py
  11b_modelo_1x2_solo_neutrales.py
  12_modelo_goles.py
  13_modelo_tarjetas.py
  13b_modelo_tarjeta_roja_fix.py   ← fix de regularización + decisión

Output:
  - data/processed/semana3_metricas_resumen.csv   (tabla para el informe)
  - models/modelos_dashboard.json                  (qué modelo usa cada mercado)
  - Imprime checklist completo de semana 3
======================================================================
"""

import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("=" * 70)
print(" 14_cierre_semana3.py - Consolidación y cierre")
print("=" * 70)

# ============================================================
# 1. LEER TODOS LOS .json DE MÉTRICAS
# ============================================================
archivos_metricas = {
    '1X2 (todo el dataset)':        '1x2_metrics.json',
    '1X2 (solo neutrales)':         '1x2_neutral_metrics.json',
    'Over/Under 2.5 goles':         'over_under_2.5_goles_metrics.json',
    'Both Teams To Score':          'both_teams_to_score_metrics.json',
    'Tarjetas Over/Under 3.5':      'tarjetas_over_under_3.5_metrics.json',
    'Tarjeta Roja':                 'tarjeta_roja_metrics.json',
}

metricas_cargadas = {}
for nombre, archivo in archivos_metricas.items():
    path = os.path.join(MODELS_DIR, archivo)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            metricas_cargadas[nombre] = json.load(f)
        print(f"✅ {archivo}")
    else:
        print(f"❌ No encontrado: {archivo}  (¿corriste el script correspondiente?)")

# ============================================================
# 2. CONSTRUIR TABLA RESUMEN
# ============================================================
print("\n" + "=" * 70)
print(" TABLA RESUMEN — TODOS LOS MERCADOS")
print("=" * 70)

filas = []

def extraer_fila(nombre, m):
    """Extrae métricas clave de un dict de métricas, sin importar el formato."""
    fila = {'mercado': nombre}

    # n de partidos de entrenamiento
    fila['n_train'] = m.get('n_train') or m.get('n_train_neutral') or m.get('n_partidos_total', '?')

    # Modelo ingenuo
    fila['acc_ingenuo'] = m.get('modelo_ingenuo', {}).get('accuracy', np.nan)

    # XGBoost calibrado (buscamos la clave correcta según el script)
    xgb_key = next(
        (k for k in ['xgboost_calibrado', 'xgboost_cv'] if k in m), None
    )
    if xgb_key:
        xgb = m[xgb_key]
        fila['acc_xgb'] = xgb.get('accuracy', np.nan)
        fila['auc_xgb'] = xgb.get('auc', np.nan)
        fila['logloss_xgb'] = xgb.get('log_loss', np.nan)
        fila['brier_xgb'] = xgb.get('brier', np.nan)
    else:
        fila['acc_xgb'] = fila['auc_xgb'] = fila['logloss_xgb'] = fila['brier_xgb'] = np.nan

    # Métricas en partidos neutrales (si existen)
    neutral = m.get('xgboost_calibrado_solo_neutrales') or m.get('evaluacion_partidos_neutrales', {})
    fila['acc_neutrales'] = neutral.get('accuracy', np.nan) if neutral else np.nan
    fila['auc_neutrales'] = neutral.get('auc', np.nan) if neutral else np.nan
    fila['n_neutrales'] = neutral.get('n_partidos', np.nan) if neutral else np.nan

    # Mejora sobre ingenuo (puntos de accuracy)
    if not np.isnan(fila['acc_xgb']) and not np.isnan(fila['acc_ingenuo']):
        fila['mejora_pp'] = round((fila['acc_xgb'] - fila['acc_ingenuo']) * 100, 2)
    else:
        fila['mejora_pp'] = np.nan

    # Decision dashboard (solo en tarjeta roja)
    fila['decision_dashboard'] = m.get('decision_dashboard', 'USAR_MODELO')

    return fila

for nombre, m in metricas_cargadas.items():
    filas.append(extraer_fila(nombre, m))

df_resumen = pd.DataFrame(filas)

# Formatear para print
cols_print = ['mercado', 'n_train', 'acc_ingenuo', 'acc_xgb', 'auc_xgb',
              'logloss_xgb', 'mejora_pp', 'decision_dashboard']
df_print = df_resumen[cols_print].copy()
df_print['acc_ingenuo'] = df_print['acc_ingenuo'].map(lambda x: f"{x:.4f}" if pd.notna(x) else '-')
df_print['acc_xgb'] = df_print['acc_xgb'].map(lambda x: f"{x:.4f}" if pd.notna(x) else '-')
df_print['auc_xgb'] = df_print['auc_xgb'].map(lambda x: f"{x:.4f}" if pd.notna(x) else '-')
df_print['logloss_xgb'] = df_print['logloss_xgb'].map(lambda x: f"{x:.4f}" if pd.notna(x) else '-')
df_print['mejora_pp'] = df_print['mejora_pp'].map(lambda x: f"{x:+.2f}pp" if pd.notna(x) else '-')

pd.set_option('display.max_colwidth', 35)
pd.set_option('display.width', 200)
print(df_print.to_string(index=False))

# ============================================================
# 3. RANKING POR AUC (el indicador principal)
# ============================================================
print("\n" + "=" * 70)
print(" RANKING POR AUC (métrica principal, independiente del umbral)")
print("=" * 70)

df_rank = df_resumen[['mercado', 'auc_xgb', 'mejora_pp', 'decision_dashboard']].copy()
df_rank = df_rank.dropna(subset=['auc_xgb']).sort_values('auc_xgb', ascending=False)
df_rank['auc_xgb'] = df_rank['auc_xgb'].map(lambda x: f"{x:.4f}")
df_rank['prioridad_dashboard'] = range(1, len(df_rank) + 1)
print(df_rank.to_string(index=False))

# ============================================================
# 4. DECISIÓN POR MERCADO PARA EL DASHBOARD
# ============================================================
print("\n" + "=" * 70)
print(" DECISIÓN POR MERCADO PARA EL DASHBOARD (Semana 5)")
print("=" * 70)

# Determinamos el modelo a usar para cada mercado en producción
modelos_dashboard = {
    '1x2': {
        'modelo_pkl': '1x2_neutral_xgboost_calibrado.pkl',
        'imputer_pkl': '1x2_neutral_imputer.pkl',
        'label_encoder_pkl': '1x2_neutral_label_encoder.pkl',
        'decision': 'USAR_MODELO',
        'razon': (
            'Modelo entrenado solo en neutrales supera marginalmente al general '
            'en el subconjunto neutral, que es el escenario real del Mundial 2026. '
            'Verificar con 11b_metrics.json si la mejora es positiva.'
        ),
        'advertencia_dashboard': (
            '1X2 es el mercado con señal más débil (~50-55% accuracy). '
            'Las probabilidades son orientativas. Priorizar value bets '
            'donde la diferencia vs cuota sea > 5%.'
        ),
    },
    'over_under_25': {
        'modelo_pkl': 'over_under_2.5_goles_xgboost_calibrado.pkl',
        'imputer_pkl': 'over_under_2.5_goles_imputer.pkl',
        'decision': 'USAR_MODELO',
        'razon': 'AUC ~0.58, señal consistente, mejor que azar de forma estable.',
        'advertencia_dashboard': None,
    },
    'btts': {
        'modelo_pkl': 'both_teams_to_score_xgboost_calibrado.pkl',
        'imputer_pkl': 'both_teams_to_score_imputer.pkl',
        'decision': 'USAR_MODELO',
        'razon': (
            'AUC ~0.57, similar a OU2.5. '
            'Usar probabilidades calibradas directamente (no umbral 0.5) '
            'para el value bet detector — el umbral óptimo está alrededor de 0.42.'
        ),
        'advertencia_dashboard': (
            'Usar probabilidad cruda del modelo para comparar vs cuota implícita. '
            'El umbral 0.5 no es apropiado para este mercado (clase ligeramente desbalanceada).'
        ),
    },
    'tarjetas_ou35': {
        'modelo_pkl': 'tarjetas_over_under_3.5_xgboost_calibrado.pkl',
        'imputer_pkl': 'tarjetas_over_under_3.5_imputer.pkl',
        'decision': 'USAR_MODELO',
        'razon': (
            'AUC 0.593, el mejor de todos los mercados. '
            'Señal estable entre folds. Dataset pequeño (751 WC), '
            'pero suficiente para capturar patrones de estilo de juego.'
        ),
        'advertencia_dashboard': (
            'Solo disponible para partidos de FIFA World Cup. '
            'Dataset de entrenamiento: 751 partidos (1970-2022).'
        ),
    },
    'tarjeta_roja': {
        'modelo_pkl': 'tarjeta_roja_xgboost_calibrado.pkl',
        'imputer_pkl': 'tarjeta_roja_imputer.pkl',
        'decision': 'USAR_BASELINE_EMPIRICO',
        'prob_base_empirica': 0.108,
        'razon': (
            'AUC < 0.58 incluso con fix de regularización. '
            'Con solo 81 positivos en 751 partidos, el dataset es insuficiente '
            'para aprender el patrón con las features disponibles.'
        ),
        'advertencia_dashboard': (
            '⚠️ El modelo predictivo no mejora sobre el baseline empírico para '
            'este mercado. Se muestra la probabilidad histórica de tarjeta roja '
            'en Mundiales (≈10.8%) como referencia orientativa. '
            'Un modelo robusto requeriría historial disciplinario individual '
            'de jugadores y perfil del árbitro asignado.'
        ),
        'que_falta_para_mejorar': [
            'Historial de tarjetas rojas por jugador en últimos 20 partidos',
            'Perfil disciplinario del árbitro designado',
            'Rivalidad histórica entre los equipos (H2H agresividad)',
            'Fase del torneo (en semifinales/finales hay menos rojas que en grupos)',
        ],
    },
}

for mercado, config in modelos_dashboard.items():
    icono = "✅" if config['decision'] == 'USAR_MODELO' else "⚠️ "
    print(f"\n{icono} {mercado.upper()}")
    print(f"   Decisión: {config['decision']}")
    print(f"   Razón: {config['razon']}")
    if config.get('advertencia_dashboard'):
        print(f"   Advertencia en dashboard: {config['advertencia_dashboard']}")

# ============================================================
# 5. GUARDAR ARCHIVOS DE CIERRE
# ============================================================
resumen_path = os.path.join(DATA_PROCESSED, 'semana3_metricas_resumen.csv')
df_resumen.to_csv(resumen_path, index=False)

dashboard_config_path = os.path.join(MODELS_DIR, 'modelos_dashboard.json')
with open(dashboard_config_path, 'w', encoding='utf-8') as f:
    json.dump(modelos_dashboard, f, indent=2, ensure_ascii=False)

print(f"\n💾 {resumen_path}")
print(f"💾 {dashboard_config_path}")

# ============================================================
# 6. CHECKLIST SEMANA 3
# ============================================================
print("\n" + "=" * 70)
print(" CHECKLIST SEMANA 3 — JUANFE")
print("=" * 70)

checklist = """
[X] 10_preparar_datos_modelado.py
      Split temporal (corte 2018-01-01), FEATURES_SEGURAS definidas,
      feature_lists.py exportado. fav_is_home EXCLUIDA (fuga encubierta).

[X] 11_modelo_1x2.py
      LogReg baseline + XGBoost calibrado (todo el dataset).
      Evaluación específica en is_neutral=1 como proxy del WC2026.

[X] 11b_modelo_1x2_solo_neutrales.py
      Variante entrenando SOLO en neutrales. Comparar ambos en test
      neutral para elegir el que va al dashboard.

[X] 12_modelo_goles.py
      OU2.5 y BTTS. Identificado y documentado el problema del
      umbral=0.5 para clases desbalanceadas (BTTS colapsaba).
      Umbral óptimo Youden's J calculado y reportado.

[X] 13_modelo_tarjetas.py
      OU3.5 tarjetas (AUC 0.593, mejor del proyecto).
      Tarjeta roja: AUC 0.454 inicial (peor que azar por regularización excesiva).

[X] 13b_modelo_tarjeta_roja_fix.py
      Fix de regularización: 3 configuraciones probadas.
      DECISIÓN (Opción 2): mostrar baseline empírico (10.8%) en dashboard
      con advertencia explícita. Limitación documentada para el informe.

[X] 14_cierre_semana3.py (este script)
      Tabla resumen consolidada, decisión por mercado, modelos_dashboard.json.

[ ] PENDIENTE: actualizar diccionario_variables.md si se agregaron columnas.
    -> python src/04_diccionario_variables.py

MERCADOS LISTOS PARA EL DASHBOARD:
  ✅ 1X2              -> 1x2_neutral_xgboost_calibrado.pkl
  ✅ Over/Under 2.5   -> over_under_2.5_goles_xgboost_calibrado.pkl
  ✅ BTTS             -> both_teams_to_score_xgboost_calibrado.pkl
  ✅ Tarjetas OU3.5   -> tarjetas_over_under_3.5_xgboost_calibrado.pkl
  ⚠️  Tarjeta Roja    -> baseline empírico 10.8% (modelo no supera utilidad mínima)
"""
print(checklist)

# ============================================================
# 7. INSTRUCCIONES PARA SEMANA 4
# ============================================================
print("=" * 70)
print(" PRÓXIMOS PASOS — SEMANA 4")
print("=" * 70)

print("""
JUANFE (Semana 4):
  1. XAI CON SHAP
     - shap.TreeExplainer sobre cada XGBoost calibrado
     - Importancia global: beeswarm plot, bar plot
     - Explicación local: waterfall plot para un partido específico
       (ej: Argentina vs Francia, WC2022 Final)
     - Guardar los valores SHAP en data/processed/shap_values_*.npy
     - Script sugerido: src/15_xai_shap.py

  2. VALUE BET DETECTOR
     - Función: valor_apuesta(prob_modelo, cuota_1xbet)
         -> expected_value = prob_modelo * cuota_1xbet - 1
         -> value_bet = True si EV > 0.05 (umbral configurable)
     - Aplicar sobre los 104 partidos de future_match_probabilities_baseline.csv
     - Incluir Kelly Criterion para sizing de apuesta
     - Script sugerido: src/16_value_bet_detector.py

  3. BACKTESTING DEL VALUE BET
     - Simular el detector sobre el test set (2018-2025)
     - ¿Cuántos value bets detectados? ¿Qué % resultaron correctos?
     - ¿ROI simulado con Kelly?
     - Esto es el corazón del modelo de negocio -> dedicarle tiempo

LULU (Semana 4):
  - Dashboard Streamlit: estructura base + visualizaciones principales
  - Integrar modelos_dashboard.json para saber qué modelo cargar por mercado
  - Verificar que la advertencia de tarjeta roja aparece correctamente
  - Ver: dashboard/app.py

PARA AMBOS:
  - Reunión de integración: asegurarse de que los outputs de Juanfe
    (probabilidades del modelo + SHAP values) son consumibles por el
    dashboard de Lulu sin transformaciones manuales.
  - Actualizar CONTEXTO_PROYECTO_football_analytics_2026.md con el
    estado actual (Semana 3 cerrada, modelos listos).
""")
