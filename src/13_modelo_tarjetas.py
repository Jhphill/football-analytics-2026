"""
13_modelo_tarjetas.py
======================================================================
Mercados de tarjetas: target_cards_ou35 (Over/Under 3.5 tarjetas) y
target_redcard (al menos 1 roja). SOLO disponibles para ~751 partidos
de FIFA World Cup (1970-2022) -- fuente: Fjelstul WC Database.

Por qué un script distinto al patrón de 11/12:
  - El universo es ~20x más chico (751 vs 43,690 partidos).
  - Un split temporal fijo (train/test por fecha) dejaría un test set
    de pocas decenas de partidos -> demasiado ruidoso para confiar en
    una sola medición.
  - En su lugar: validación cruzada estratificada (5-fold) sobre TODO
    el universo de 751 partidos, que es el estándar correcto para
    datasets chicos. El modelo final se reentrena sobre el 100% de
    los datos disponibles (no se "pierde" un test set permanente,
    ya que la CV ya da una estimación honesta de generalización).
  - Por el tamaño reducido, XGBoost se usa con regularización fuerte
    (profundidad baja, min_child_weight alto) para evitar overfitting.

NOTA IMPORTANTE: con tan pocos datos, cualquier métrica debe leerse
con más cautela que en los mercados anteriores. Esto se documenta
explícitamente en las métricas guardadas.
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_lists import FEATURES_SEGURAS, TARGETS

print("=" * 70)
print(" 13_modelo_tarjetas.py - target_cards_ou35 y target_redcard")
print(" (dataset reducido: ~751 partidos de Mundial, vía cross-validation)")
print("=" * 70)

# ============================================================
# CARGAR EL DATASET COMPLETO (no train/test separado -- se usa CV)
# ============================================================
in_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
df_full = pd.read_csv(in_path)
print(f"\n✅ matches_features_v2.csv cargado: {df_full.shape}")


def entrenar_con_cv(nombre_mercado, target_col, df_full, features, models_dir,
                     n_splits=5, min_child_weight=10, reg_lambda=2.0, max_depth=3):
    """
    Entrena y evalúa con validación cruzada estratificada (apropiado
    para datasets chicos). Usa cross_val_predict para obtener
    predicciones "out-of-fold" sobre TODO el dataset, que sirven como
    una estimación honesta de desempeño sin gastar un test set fijo.
    El modelo final se reentrena sobre el 100% de los datos.

    min_child_weight/reg_lambda/max_depth son configurables por
    mercado: para clases muy raras (ej. tarjeta roja, ~11% positivos),
    una regularización demasiado fuerte puede impedir que el modelo
    aprenda el patrón minoritario -> se reduce para esos casos.
    """
    print("\n" + "#" * 70)
    print(f" MERCADO: {nombre_mercado}  (target: {target_col})")
    print("#" * 70)

    df = df_full[df_full[target_col].notna()].copy()
    print(f"\nPartidos disponibles: {len(df)}")

    if len(df) < 100:
        print(f"⚠️ ADVERTENCIA: muy pocos datos ({len(df)}). Resultados poco confiables.")

    X = df[features].copy()
    y = df[target_col].astype(int)

    print(f"Distribución del target: {y.value_counts(normalize=True).round(3).to_dict()}")

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- Modelo ingenuo ---
    clase_mayoritaria = y.value_counts().idxmax()
    acc_ingenuo = accuracy_score(y, [clase_mayoritaria] * len(y))
    print(f"\n--- Modelo ingenuo (siempre {clase_mayoritaria}) ---")
    print(f"Accuracy: {acc_ingenuo:.4f}")

    # --- Baseline: Logistic Regression con CV ---
    print(f"\n--- BASELINE: Logistic Regression ({n_splits}-fold CV) ---")
    baseline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    proba_base_cv = cross_val_predict(
        baseline, X_imp, y, cv=cv, method='predict_proba'
    )[:, 1]
    pred_base_cv = (proba_base_cv >= 0.5).astype(int)

    acc_base = accuracy_score(y, pred_base_cv)
    logloss_base = log_loss(y, proba_base_cv)
    auc_base = roc_auc_score(y, proba_base_cv)
    print(f"Accuracy: {acc_base:.4f} | Log-loss: {logloss_base:.4f} | AUC: {auc_base:.4f}")

    # --- XGBoost regularizado, con CV ---
    print(f"\n--- AVANZADO: XGBoost regularizado ({n_splits}-fold CV) ---")
    xgb_params = dict(
        n_estimators=100,
        max_depth=3,            # conservador por el tamaño reducido
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,     # regularización fuerte
        reg_lambda=2.0,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
    )

    proba_xgb_cv = cross_val_predict(
        XGBClassifier(**xgb_params), X_imp, y, cv=cv, method='predict_proba'
    )[:, 1]
    pred_xgb_cv = (proba_xgb_cv >= 0.5).astype(int)

    acc_xgb = accuracy_score(y, pred_xgb_cv)
    logloss_xgb = log_loss(y, proba_xgb_cv)
    brier_xgb = brier_score_loss(y, proba_xgb_cv)
    auc_xgb = roc_auc_score(y, proba_xgb_cv)
    print(f"Accuracy: {acc_xgb:.4f} | Log-loss: {logloss_xgb:.4f} | "
          f"Brier: {brier_xgb:.4f} | AUC: {auc_xgb:.4f}")
    print(f"\nMatriz de confusión (out-of-fold, umbral=0.5):")
    print(confusion_matrix(y, pred_xgb_cv))
    print(f"\n{classification_report(y, pred_xgb_cv)}")

    # --- Umbral óptimo (Youden's J), igual que en goles ---
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, proba_xgb_cv)
    youden_j = tpr - fpr
    umbral_optimo = thresholds[np.argmax(youden_j)]
    pred_xgb_opt = (proba_xgb_cv >= umbral_optimo).astype(int)
    acc_xgb_opt = accuracy_score(y, pred_xgb_opt)
    print(f"\n--- Con umbral óptimo (Youden's J = {umbral_optimo:.3f}) ---")
    print(f"Accuracy: {acc_xgb_opt:.4f}")
    print(confusion_matrix(y, pred_xgb_opt))

    # --- Variabilidad entre folds (importante en datasets chicos) ---
    accs_por_fold = []
    for train_idx, test_idx in cv.split(X_imp, y):
        m = XGBClassifier(**xgb_params)
        m.fit(X_imp[train_idx], y.iloc[train_idx])
        p = m.predict(X_imp[test_idx])
        accs_por_fold.append(accuracy_score(y.iloc[test_idx], p))
    print(f"\nAccuracy por fold: {np.round(accs_por_fold, 4)}")
    print(f"Media: {np.mean(accs_por_fold):.4f} (+/- {np.std(accs_por_fold):.4f})")

    # --- Entrenar modelo FINAL sobre el 100% de los datos ---
    modelo_final = CalibratedClassifierCV(
        XGBClassifier(**xgb_params), method='isotonic', cv=5
    )
    modelo_final.fit(X_imp, y)

    # --- Importancia de variables (de un modelo simple sobre todo el dataset) ---
    modelo_importancia = XGBClassifier(**xgb_params)
    modelo_importancia.fit(X_imp, y)
    importancias = pd.Series(
        modelo_importancia.feature_importances_, index=features
    ).sort_values(ascending=False)
    print(f"\nTop 10 variables más importantes:")
    print(importancias.head(10))

    # --- Tabla comparativa ---
    print(f"\n--- TABLA COMPARATIVA: {nombre_mercado} (todo vía CV) ---")
    tabla = pd.DataFrame({
        'modelo': ['Ingenuo', 'Logistic Regression', 'XGBoost regularizado'],
        'accuracy': [acc_ingenuo, acc_base, acc_xgb],
        'log_loss': [np.nan, logloss_base, logloss_xgb],
        'auc': [np.nan, auc_base, auc_xgb],
    })
    print(tabla.to_string(index=False))

    # --- Guardar ---
    slug = nombre_mercado.lower().replace(' ', '_').replace('/', '')
    joblib.dump(modelo_final, os.path.join(models_dir, f'{slug}_xgboost_calibrado.pkl'))
    joblib.dump(imputer, os.path.join(models_dir, f'{slug}_imputer.pkl'))

    metricas = {
        'mercado': nombre_mercado,
        'target': target_col,
        'n_partidos_total': len(df),
        'metodologia': f'{n_splits}-fold StratifiedKFold cross-validation '
                        '(dataset reducido, sin split temporal fijo)',
        'advertencia': 'Dataset pequeño -- interpretar métricas con cautela. '
                        'Ver accuracy_por_fold para variabilidad.',
        'modelo_ingenuo': {'accuracy': float(acc_ingenuo)},
        'baseline_logreg_cv': {
            'accuracy': float(acc_base), 'log_loss': float(logloss_base), 'auc': float(auc_base)
        },
        'xgboost_cv': {
            'accuracy': float(acc_xgb), 'log_loss': float(logloss_xgb),
            'brier': float(brier_xgb), 'auc': float(auc_xgb),
            'accuracy_umbral_optimo': float(acc_xgb_opt),
            'umbral_optimo': float(umbral_optimo),
        },
        'accuracy_por_fold': [float(a) for a in accs_por_fold],
        'accuracy_fold_media': float(np.mean(accs_por_fold)),
        'accuracy_fold_std': float(np.std(accs_por_fold)),
        'top_10_features': importancias.head(10).to_dict(),
    }
    metrics_path = os.path.join(models_dir, f'{slug}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Modelo final (entrenado en 100% de los datos): {models_dir}/{slug}_xgboost_calibrado.pkl")
    print(f"💾 Métricas: {metrics_path}")

    return metricas


# ============================================================
# EJECUTAR PARA AMBOS TARGETS DE TARJETAS
# ============================================================
resultados = {}

resultados['cards_ou35'] = entrenar_con_cv(
    'Tarjetas_Over_Under_3.5', TARGETS['tarjetas_ou35'],
    df_full, FEATURES_SEGURAS, MODELS_DIR
)

resultados['redcard'] = entrenar_con_cv(
    'Tarjeta_Roja', TARGETS['tarjeta_roja'],
    df_full, FEATURES_SEGURAS, MODELS_DIR
)

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print(" RESUMEN FINAL - MERCADOS DE TARJETAS")
print("=" * 70)
for key, m in resultados.items():
    print(f"\n{m['mercado']} (n={m['n_partidos_total']}):")
    print(f"  Ingenuo:   {m['modelo_ingenuo']['accuracy']:.4f}")
    print(f"  XGBoost (CV): acc={m['xgboost_cv']['accuracy']:.4f} "
          f"| auc={m['xgboost_cv']['auc']:.4f} "
          f"| log_loss={m['xgboost_cv']['log_loss']:.4f}")
    print(f"  Estabilidad entre folds: {m['accuracy_fold_media']:.4f} "
          f"(+/- {m['accuracy_fold_std']:.4f})")

print("""
SEMANA 3 -- RESUMEN GLOBAL DE MERCADOS MODELADOS HASTA AHORA:
  1X2                  -> ver models/1x2_metrics.json
  Over/Under 2.5 goles  -> ver models/over_under_2.5_goles_metrics.json
  Both Teams To Score   -> ver models/both_teams_to_score_metrics.json
  Tarjetas Over/Under   -> ver models/tarjetas_over_under_3.5_metrics.json
  Tarjeta roja          -> ver models/tarjeta_roja_metrics.json

PRÓXIMO PASO:
  - Consolidar todas las métricas en una tabla única para el informe
  - Empezar XAI (SHAP) sobre los modelos finales (Semana 4)
""")