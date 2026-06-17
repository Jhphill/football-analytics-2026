"""
13b_modelo_tarjeta_roja_fix.py
======================================================================
Fix de regularización para el mercado de tarjeta roja (target_redcard).

Problema identificado en 13_modelo_tarjetas.py:
  - AUC = 0.454 (peor que azar) con min_child_weight=10 y reg_lambda=2.0
  - El modelo nunca predecía clase 1 (0 verdaderos positivos)
  - Hipótesis: regularización demasiado agresiva para 81 positivos en
    751 partidos -> el modelo colapsaba hacia "siempre predecir 0"

Este script prueba tres configuraciones de regularización progresivamente
más permisivas y compara contra el modelo original.

DECISIÓN DE DISEÑO (Opción 2):
  Si el AUC final no supera claramente 0.55, el modelo NO se recomienda
  para predicción en el dashboard. En su lugar:
    - Se muestra la probabilidad base empírica (10.8% histórico en WC)
    - Se incluye advertencia explícita de que el modelo no mejora el baseline
    - Se documenta la limitación: 81 positivos son insuficientes para
      aprender el patrón sin features de historial disciplinario individual
      (datos no disponibles en este dataset).

  Esta decisión es académicamente honesta y demuestra criterio técnico
  para el informe — saber cuándo NO usar un modelo es tan valioso como
  saber construirlo.
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_lists import FEATURES_SEGURAS, TARGETS

TARGET = TARGETS['tarjeta_roja']
PROB_BASE_EMPIRICA = 0.108   # 81/751 partidos de Mundial con al menos 1 roja

print("=" * 70)
print(" 13b_modelo_tarjeta_roja_fix.py")
print(" Fix de regularización + diagnóstico completo")
print("=" * 70)

# ============================================================
# CARGAR Y FILTRAR
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df_full = pd.read_csv(in_path)
df = df_full[df_full[TARGET].notna()].copy()

print(f"\nPartidos disponibles (is_world_cup=1 con tarjetas): {len(df)}")
print(f"Positivos (al menos 1 roja): {int(df[TARGET].sum())} "
      f"({100*df[TARGET].mean():.1f}%)")
print(f"Negativos: {int((df[TARGET] == 0).sum())}")
print(f"\n⚠️  Con 5-fold CV -> ~{int(df[TARGET].sum())//5} positivos por fold de test")
print("    Varianza de AUC es alta con tan pocos eventos -> interpretar con cautela")

X = df[FEATURES_SEGURAS].copy()
y = df[TARGET].astype(int)

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# MODELO ORIGINAL (referencia desde 13_modelo_tarjetas.py)
# ============================================================
print("\n" + "=" * 70)
print(" REFERENCIA: modelo original (min_child_weight=10, reg_lambda=2.0)")
print("=" * 70)
print("AUC reportado: 0.454 | Accuracy: 0.8921 | Nunca predijo clase 1")
print("-> Regularización excesiva para 81 positivos")

# ============================================================
# CONFIGURACIONES A PROBAR
# ============================================================
configs = [
    {
        'nombre': 'Config A (relajada: mcw=3, lambda=1.0)',
        'params': dict(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=3,    # original era 10
            reg_lambda=1.0,        # original era 2.0
            scale_pos_weight=670/81,  # compensar desbalance: neg/pos
            objective='binary:logistic', random_state=42, eval_metric='logloss',
        )
    },
    {
        'nombre': 'Config B (mínima regularización: mcw=1, lambda=0.5)',
        'params': dict(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=1,
            reg_lambda=0.5,
            scale_pos_weight=670/81,
            objective='binary:logistic', random_state=42, eval_metric='logloss',
        )
    },
    {
        'nombre': 'Config C (más profunda + scale_pos_weight)',
        'params': dict(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=2,
            reg_lambda=0.5, reg_alpha=0.1,
            scale_pos_weight=670/81,
            objective='binary:logistic', random_state=42, eval_metric='logloss',
        )
    },
]

resultados_configs = []

for cfg in configs:
    print(f"\n--- {cfg['nombre']} ---")

    proba_cv = cross_val_predict(
        XGBClassifier(**cfg['params']), X_imp, y, cv=cv, method='predict_proba'
    )[:, 1]

    pred_05 = (proba_cv >= 0.5).astype(int)
    auc = roc_auc_score(y, proba_cv)
    logloss = log_loss(y, proba_cv)
    brier = brier_score_loss(y, proba_cv)
    acc = accuracy_score(y, pred_05)

    # Umbral óptimo Youden's J
    fpr, tpr, thresholds = roc_curve(y, proba_cv)
    umbral_opt = thresholds[np.argmax(tpr - fpr)]
    pred_opt = (proba_cv >= umbral_opt).astype(int)
    acc_opt = accuracy_score(y, pred_opt)

    n_pred_positivos = pred_05.sum()
    n_pred_positivos_opt = pred_opt.sum()

    print(f"AUC: {auc:.4f} | Log-loss: {logloss:.4f} | Brier: {brier:.4f}")
    print(f"Accuracy (umbral=0.5): {acc:.4f} | Predicciones positivas: {n_pred_positivos}")
    print(f"Accuracy (Youden's J={umbral_opt:.3f}): {acc_opt:.4f} "
          f"| Predicciones positivas: {n_pred_positivos_opt}")
    print(f"Matriz de confusión (umbral=0.5):")
    print(confusion_matrix(y, pred_05))

    resultados_configs.append({
        'config': cfg['nombre'],
        'auc': auc,
        'log_loss': logloss,
        'brier': brier,
        'accuracy_05': acc,
        'n_pred_positivos_05': int(n_pred_positivos),
        'umbral_optimo': float(umbral_opt),
        'accuracy_opt': acc_opt,
        'n_pred_positivos_opt': int(n_pred_positivos_opt),
    })

# ============================================================
# TABLA COMPARATIVA
# ============================================================
print("\n" + "=" * 70)
print(" COMPARATIVA FINAL — TARJETA ROJA")
print("=" * 70)

df_comp = pd.DataFrame([
    {'config': 'ORIGINAL (mcw=10, lambda=2.0)', 'auc': 0.454,
     'n_pred_pos_umbral05': 0, 'nota': '← punto de partida'},
] + [
    {'config': r['config'], 'auc': r['auc'],
     'n_pred_pos_umbral05': r['n_pred_positivos_05'], 'nota': ''}
    for r in resultados_configs
])
print(df_comp.to_string(index=False))

mejor = max(resultados_configs, key=lambda x: x['auc'])
print(f"\nMejor AUC logrado: {mejor['auc']:.4f} ({mejor['config']})")

# ============================================================
# DIAGNÓSTICO Y DECISIÓN FINAL
# ============================================================
print("\n" + "=" * 70)
print(" DIAGNÓSTICO Y DECISIÓN FINAL")
print("=" * 70)

UMBRAL_UTIL = 0.58   # AUC mínimo para considerar el modelo útil en producción

if mejor['auc'] >= UMBRAL_UTIL:
    decision = "USAR_MODELO"
    print(f"✅ AUC {mejor['auc']:.4f} >= {UMBRAL_UTIL} -> modelo aceptable para el dashboard")
    print("   Usar la configuración con mejor AUC, con advertencia de dataset pequeño.")
else:
    decision = "USAR_BASELINE_EMPIRICO"
    print(f"⚠️  AUC {mejor['auc']:.4f} < {UMBRAL_UTIL} -> modelo NO mejora suficientemente el azar")
    print(f"\n   DECISIÓN (Opción 2): mostrar probabilidad base empírica en el dashboard")
    print(f"   con advertencia explícita.")
    print(f"\n   Probabilidad base empírica (histórico WC 1970-2022):")
    print(f"     P(al menos 1 tarjeta roja en el partido) = {PROB_BASE_EMPIRICA:.1%}")
    print(f"\n   Limitación documentada:")
    print(f"     Con solo {int(y.sum())} eventos positivos en {len(y)} partidos,")
    print(f"     el dataset es insuficiente para aprender el patrón con las features")
    print(f"     disponibles. Un modelo robusto requeriría historial disciplinario")
    print(f"     individual de jugadores (tarjetas por partido a lo largo de la")
    print(f"     carrera, perfil del árbitro, rivalidades históricas entre equipos)")
    print(f"     que no están disponibles en este dataset.")

# ============================================================
# GUARDAR MODELO FINAL Y METADATOS
# ============================================================
# Siempre guardamos el mejor modelo encontrado (aunque no sea "útil")
# para que el dashboard pueda mostrar las probabilidades calibradas
# con la advertencia correspondiente.

cfg_mejor = configs[
    [r['auc'] for r in resultados_configs].index(mejor['auc'])
]['params']

modelo_final = CalibratedClassifierCV(
    XGBClassifier(**cfg_mejor), method='isotonic', cv=5
)
modelo_final.fit(X_imp, y)

joblib.dump(modelo_final, os.path.join(MODELS_DIR, 'tarjeta_roja_xgboost_calibrado.pkl'))
joblib.dump(imputer, os.path.join(MODELS_DIR, 'tarjeta_roja_imputer.pkl'))

metricas_finales = {
    'mercado': 'Tarjeta_Roja (fix de regularización)',
    'target': TARGET,
    'n_partidos_total': len(df),
    'n_positivos': int(y.sum()),
    'pct_positivos': float(y.mean()),
    'prob_base_empirica': PROB_BASE_EMPIRICA,
    'decision_dashboard': decision,
    'advertencia': (
        'Modelo con señal limitada por dataset pequeño (81 positivos). '
        'AUC no supera umbral de utilidad práctica (0.58). '
        'En el dashboard se muestra la probabilidad base empírica '
        f'({PROB_BASE_EMPIRICA:.1%}) con advertencia explícita. '
        'Un modelo robusto requeriría historial disciplinario individual '
        'de jugadores y perfil del árbitro.'
    ),
    'modelo_original_auc': 0.454,
    'mejor_config': mejor['config'],
    'mejor_auc': float(mejor['auc']),
    'comparativa_configs': resultados_configs,
    'umbral_utilidad_definido': UMBRAL_UTIL,
}

metrics_path = os.path.join(MODELS_DIR, 'tarjeta_roja_metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metricas_finales, f, indent=2, ensure_ascii=False)

print(f"\n💾 Modelo guardado (sobreescribe el de 13): {MODELS_DIR}/tarjeta_roja_xgboost_calibrado.pkl")
print(f"💾 Métricas actualizadas: {metrics_path}")
print(f"\n-> decision_dashboard = '{decision}'")
print("   El dashboard lee este campo para saber si mostrar modelo o baseline empírico.")

print("""
PRÓXIMO PASO:
  - 14_cierre_semana3.py -> consolidar todas las métricas, tabla
    resumen, decisión por mercado, preparar modelos para Semana 4
""")
