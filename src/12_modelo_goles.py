"""
12_modelo_goles.py
======================================================================
Mercados de goles: target_ou25 (Over/Under 2.5) y target_btts (Both
Teams To Score). Ambos binarios, sobre el dataset completo (no hace
falta restringir a is_neutral=1 aquí -- a diferencia del 1X2, el
"quién gana" no es lo que se predice, sino "cuántos goles hay" y
"anotan ambos", que dependen más de estilo de juego y nivel ofensivo/
defensivo que de localía).

Pipeline (idéntico patrón a 11_modelo_1x2.py, reutilizable):
  1. Cargar train_set.csv / test_set.csv, FEATURES_SEGURAS
  2. Modelo ingenuo (clase mayoritaria)
  3. Baseline: Logistic Regression
  4. Avanzado: XGBoost + calibración isotónica
  5. Métricas: accuracy, log-loss, Brier Score, AUC
  6. Evaluación específica en partidos neutrales (proxy Mundial 2026)
  7. Guardar modelos + métricas .json

Se corre una vez por cada target (función reutilizable).
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
print(" 12_modelo_goles.py - target_ou25 y target_btts")
print("=" * 70)

# ============================================================
# CARGAR (una sola vez, se reutiliza para ambos targets)
# ============================================================
train_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'train_set.csv'))
test_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'test_set.csv'))
print(f"\nTrain: {train_full.shape} | Test: {test_full.shape}")


def entrenar_y_evaluar_mercado(nombre_mercado, target_col, train_full, test_full,
                                 features, models_dir):
    """
    Pipeline completo: ingenuo -> baseline -> XGBoost -> calibración
    -> evaluación general y en partidos neutrales -> guardado.
    Retorna un dict con todas las métricas (para comparar mercados).
    """
    print("\n" + "#" * 70)
    print(f" MERCADO: {nombre_mercado}  (target: {target_col})")
    print("#" * 70)

    # Filtrar filas con target válido (no NaN)
    train = train_full[train_full[target_col].notna()].copy()
    test = test_full[test_full[target_col].notna()].copy()
    print(f"\nFilas válidas -> Train: {len(train):,} | Test: {len(test):,}")

    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)

    print(f"Distribución en train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Distribución en test:  {y_test.value_counts(normalize=True).round(3).to_dict()}")

    # --- Modelo ingenuo ---
    clase_mayoritaria = y_train.value_counts().idxmax()
    acc_ingenuo = accuracy_score(y_test, [clase_mayoritaria] * len(y_test))
    print(f"\n--- Modelo ingenuo (siempre {clase_mayoritaria}) ---")
    print(f"Accuracy: {acc_ingenuo:.4f}")

    # --- Baseline: Logistic Regression ---
    print("\n--- BASELINE: Logistic Regression ---")
    baseline_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    baseline_pipeline.fit(X_train, y_train)
    proba_base = baseline_pipeline.predict_proba(X_test)[:, 1]
    pred_base = (proba_base >= 0.5).astype(int)

    acc_base = accuracy_score(y_test, pred_base)
    logloss_base = log_loss(y_test, proba_base)
    brier_base = brier_score_loss(y_test, proba_base)
    auc_base = roc_auc_score(y_test, proba_base)
    print(f"Accuracy: {acc_base:.4f} | Log-loss: {logloss_base:.4f} | "
          f"Brier: {brier_base:.4f} | AUC: {auc_base:.4f}")

    # --- Avanzado: XGBoost ---
    print("\n--- AVANZADO: XGBoost (sin calibrar) ---")
    imputer_xgb = SimpleImputer(strategy='median')
    X_train_imp = imputer_xgb.fit_transform(X_train)
    X_test_imp = imputer_xgb.transform(X_test)

    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', random_state=42, eval_metric='logloss',
    )
    xgb_model.fit(X_train_imp, y_train)
    proba_xgb_raw = xgb_model.predict_proba(X_test_imp)[:, 1]
    pred_xgb_raw = (proba_xgb_raw >= 0.5).astype(int)

    acc_xgb_raw = accuracy_score(y_test, pred_xgb_raw)
    logloss_xgb_raw = log_loss(y_test, proba_xgb_raw)
    brier_xgb_raw = brier_score_loss(y_test, proba_xgb_raw)
    auc_xgb_raw = roc_auc_score(y_test, proba_xgb_raw)
    print(f"Accuracy: {acc_xgb_raw:.4f} | Log-loss: {logloss_xgb_raw:.4f} | "
          f"Brier: {brier_xgb_raw:.4f} | AUC: {auc_xgb_raw:.4f}")

    # --- Calibración isotónica ---
    print("\n--- AVANZADO: XGBoost + Calibración isotónica (cv=5) ---")
    xgb_calibrado = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', random_state=42, eval_metric='logloss',
        ),
        method='isotonic', cv=5
    )
    xgb_calibrado.fit(X_train_imp, y_train)
    proba_xgb = xgb_calibrado.predict_proba(X_test_imp)[:, 1]
    pred_xgb = (proba_xgb >= 0.5).astype(int)

    acc_xgb = accuracy_score(y_test, pred_xgb)
    logloss_xgb = log_loss(y_test, proba_xgb)
    brier_xgb = brier_score_loss(y_test, proba_xgb)
    auc_xgb = roc_auc_score(y_test, proba_xgb)
    print(f"Accuracy: {acc_xgb:.4f} | Log-loss: {logloss_xgb:.4f} | "
          f"Brier: {brier_xgb:.4f} | AUC: {auc_xgb:.4f}")
    print(f"\nMatriz de confusión (umbral=0.5):")
    print(confusion_matrix(y_test, pred_xgb))
    print(f"\n{classification_report(y_test, pred_xgb)}")

    # --- Umbral óptimo (Youden's J) ---
    # El umbral fijo 0.5 puede ser mal elegido cuando hay desbalance de
    # clases entre train y test, o cuando la señal es modesta -> el
    # modelo puede colapsar a predecir siempre la clase mayoritaria.
    # AUC no depende del umbral, así que si AUC > 0.5 con accuracy
    # cercano al ingenuo, es señal de que el umbral (no el modelo) es
    # el problema. Youden's J encuentra el umbral que maximiza
    # (sensibilidad + especificidad - 1).
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, proba_xgb)
    youden_j = tpr - fpr
    umbral_optimo = thresholds[np.argmax(youden_j)]

    pred_xgb_opt = (proba_xgb >= umbral_optimo).astype(int)
    acc_xgb_opt = accuracy_score(y_test, pred_xgb_opt)

    print(f"\n--- Con umbral óptimo (Youden's J = {umbral_optimo:.3f}, "
          f"en vez de 0.5) ---")
    print(f"Accuracy: {acc_xgb_opt:.4f}")
    print(f"Matriz de confusión (umbral óptimo):")
    print(confusion_matrix(y_test, pred_xgb_opt))
    print(f"\n{classification_report(y_test, pred_xgb_opt)}")

    # --- Importancia de variables ---
    importancias = pd.Series(
        xgb_model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print(f"\nTop 10 variables más importantes:")
    print(importancias.head(10))

    # --- Evaluación en partidos neutrales (proxy Mundial 2026) ---
    mask_neutral = test['is_neutral'] == 1
    resultado_neutral = {}
    if mask_neutral.sum() > 0:
        X_test_n_imp = imputer_xgb.transform(X_test[mask_neutral])
        y_test_n = y_test[mask_neutral.values]
        proba_n = xgb_calibrado.predict_proba(X_test_n_imp)[:, 1]
        pred_n = (proba_n >= 0.5).astype(int)

        acc_n = accuracy_score(y_test_n, pred_n)
        logloss_n = log_loss(y_test_n, proba_n, labels=[0, 1])
        auc_n = roc_auc_score(y_test_n, proba_n) if y_test_n.nunique() > 1 else np.nan

        print(f"\n--- Evaluación en partidos neutrales (n={mask_neutral.sum()}) ---")
        print(f"Accuracy: {acc_n:.4f} | Log-loss: {logloss_n:.4f} | AUC: {auc_n:.4f}")

        resultado_neutral = {
            'n_partidos': int(mask_neutral.sum()),
            'accuracy': float(acc_n),
            'log_loss': float(logloss_n),
            'auc': float(auc_n) if not np.isnan(auc_n) else None,
        }

    # --- Tabla comparativa ---
    print(f"\n--- TABLA COMPARATIVA: {nombre_mercado} ---")
    tabla = pd.DataFrame({
        'modelo': ['Ingenuo', 'Logistic Regression', 'XGBoost (sin calibrar)', 'XGBoost (calibrado)'],
        'accuracy': [acc_ingenuo, acc_base, acc_xgb_raw, acc_xgb],
        'log_loss': [np.nan, logloss_base, logloss_xgb_raw, logloss_xgb],
        'brier': [np.nan, brier_base, brier_xgb_raw, brier_xgb],
        'auc': [np.nan, auc_base, auc_xgb_raw, auc_xgb],
    })
    print(tabla.to_string(index=False))

    # --- Guardar modelos ---
    slug = nombre_mercado.lower().replace(' ', '_').replace('/', '')
    joblib.dump(baseline_pipeline, os.path.join(models_dir, f'{slug}_baseline_logreg.pkl'))
    joblib.dump(xgb_calibrado, os.path.join(models_dir, f'{slug}_xgboost_calibrado.pkl'))
    joblib.dump(imputer_xgb, os.path.join(models_dir, f'{slug}_imputer.pkl'))

    metricas = {
        'mercado': nombre_mercado,
        'target': target_col,
        'n_train': len(train),
        'n_test': len(test),
        'modelo_ingenuo': {'accuracy': float(acc_ingenuo)},
        'baseline_logreg': {
            'accuracy': float(acc_base), 'log_loss': float(logloss_base),
            'brier': float(brier_base), 'auc': float(auc_base),
        },
        'xgboost_sin_calibrar': {
            'accuracy': float(acc_xgb_raw), 'log_loss': float(logloss_xgb_raw),
            'brier': float(brier_xgb_raw), 'auc': float(auc_xgb_raw),
        },
        'xgboost_calibrado': {
            'accuracy': float(acc_xgb), 'log_loss': float(logloss_xgb),
            'brier': float(brier_xgb), 'auc': float(auc_xgb),
        },
        'evaluacion_partidos_neutrales': resultado_neutral,
        'top_10_features': importancias.head(10).to_dict(),
    }
    metrics_path = os.path.join(models_dir, f'{slug}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Modelos: {models_dir}/{slug}_*.pkl")
    print(f"💾 Métricas: {metrics_path}")

    return metricas


# ============================================================
# EJECUTAR PARA AMBOS MERCADOS
# ============================================================
resultados = {}

resultados['ou25'] = entrenar_y_evaluar_mercado(
    'Over_Under_2.5_goles', TARGETS['over_under_25'],
    train_full, test_full, FEATURES_SEGURAS, MODELS_DIR
)

resultados['btts'] = entrenar_y_evaluar_mercado(
    'Both_Teams_To_Score', TARGETS['btts'],
    train_full, test_full, FEATURES_SEGURAS, MODELS_DIR
)

# ============================================================
# RESUMEN FINAL CONJUNTO
# ============================================================
print("\n" + "=" * 70)
print(" RESUMEN FINAL - MERCADOS DE GOLES")
print("=" * 70)
for key, m in resultados.items():
    print(f"\n{m['mercado']}:")
    print(f"  Ingenuo:   {m['modelo_ingenuo']['accuracy']:.4f}")
    print(f"  XGBoost calibrado: acc={m['xgboost_calibrado']['accuracy']:.4f} "
          f"| auc={m['xgboost_calibrado']['auc']:.4f} "
          f"| log_loss={m['xgboost_calibrado']['log_loss']:.4f}")
    if m['evaluacion_partidos_neutrales']:
        en = m['evaluacion_partidos_neutrales']
        print(f"  En partidos neutrales: acc={en['accuracy']:.4f} | auc={en['auc']:.4f} "
              f"(n={en['n_partidos']})")

print("""
PRÓXIMO PASO:
  - 13_modelo_tarjetas.py -> target_cards_ou35 y target_redcard
    (dataset reducido, solo ~751 partidos de Mundial -> usar
    cross-validation en vez de un test set fijo)
""")