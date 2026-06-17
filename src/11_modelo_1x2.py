"""
11_modelo_1x2.py
======================================================================
Mercado 1X2 (target_1x2_fav_dog: fav_win / draw / dog_win).

Pipeline:
  1. Cargar train_set.csv / test_set.csv, usar FEATURES_SEGURAS de
     feature_lists.py
  2. Imputar NaN (mediana para numéricas) -- necesario para Logistic
     Regression; XGBoost podría manejar NaN nativo pero usamos la
     misma imputación en ambos para comparación justa
  3. BASELINE: Logistic Regression multiclase (one-vs-rest)
  4. AVANZADO: XGBoost multiclase + calibración (CalibratedClassifierCV,
     método isotonic, cv=5)
  5. Métrica de referencia ("modelo ingenuo"): predecir siempre la
     clase mayoritaria (fav_win) -> cualquier modelo debe superar esto
  6. Evaluación con accuracy, log-loss, matriz de confusión
  7. Guardar modelos en models/ y métricas en un .json para comparar
     con futuras versiones (ej. cuando Lulu integre NLP)
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_lists import FEATURES_SEGURAS, TARGETS

TARGET = TARGETS['1x2']

print("=" * 70)
print(f" 11_modelo_1x2.py - Mercado 1X2 (target: {TARGET})")
print("=" * 70)

# ============================================================
# 1. CARGAR
# ============================================================
train = pd.read_csv(os.path.join(DATA_PROCESSED, 'train_set.csv'))
test = pd.read_csv(os.path.join(DATA_PROCESSED, 'test_set.csv'))
print(f"\nTrain: {train.shape} | Test: {test.shape}")

X_train = train[FEATURES_SEGURAS].copy()
X_test = test[FEATURES_SEGURAS].copy()

# NOTA: 'fav_is_home' fue removida de FEATURES_SEGURAS (ver feature_lists.py
# y notas en 10_preparar_datos_modelado.py) tras detectar que, si bien
# parecía predictiva incluso en partidos neutrales, esa señal no es
# transferible al dataset de predicción real del Mundial 2026 (ahí el
# 'home_team' está determinado por ser el país anfitrión, no por ser
# favorito) -> habría sido una fuga de información encubierta y no
# generalizable.

FEATURES_SEGURAS_V2 = FEATURES_SEGURAS  # alias, ya no hay feature adicional

y_train_raw = train[TARGET]
y_test_raw = test[TARGET]

# ============================================================
# 2. CODIFICAR TARGET (texto -> número, necesario para XGBoost)
# ============================================================
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
print(f"\nClases codificadas: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ============================================================
# 3. MODELO INGENUO (referencia mínima a superar)
# ============================================================
clase_mayoritaria = pd.Series(y_train_raw).value_counts().idxmax()
y_pred_ingenuo = [clase_mayoritaria] * len(y_test_raw)
acc_ingenuo = accuracy_score(y_test_raw, y_pred_ingenuo)
print(f"\n--- Modelo ingenuo (predecir siempre '{clase_mayoritaria}') ---")
print(f"Accuracy: {acc_ingenuo:.4f}")
print("(cualquier modelo real debe superar esto claramente)")

# ============================================================
# 4. BASELINE: Logistic Regression
# ============================================================
print("\n--- BASELINE: Logistic Regression ---")

baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

baseline_pipeline.fit(X_train, y_train)

y_pred_base = baseline_pipeline.predict(X_test)
y_proba_base = baseline_pipeline.predict_proba(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
logloss_base = log_loss(y_test, y_proba_base)

print(f"Accuracy: {acc_base:.4f}")
print(f"Log-loss: {logloss_base:.4f}")
print(f"\nMatriz de confusión (filas=real, columnas=predicho, orden={list(le.classes_)}):")
print(confusion_matrix(y_test, y_pred_base))
print(f"\n{classification_report(y_test, y_pred_base, target_names=le.classes_)}")

# ============================================================
# 5. AVANZADO: XGBoost + Calibración
# ============================================================
print("\n--- AVANZADO: XGBoost (sin calibrar) ---")

# Imputación simple para XGBoost también (comparación justa con baseline,
# aunque XGBoost podría manejar NaN nativamente)
imputer_xgb = SimpleImputer(strategy='median')
X_train_imp = imputer_xgb.fit_transform(X_train)
X_test_imp = imputer_xgb.transform(X_test)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss',
)
xgb_model.fit(X_train_imp, y_train)

y_pred_xgb_raw = xgb_model.predict(X_test_imp)
y_proba_xgb_raw = xgb_model.predict_proba(X_test_imp)

acc_xgb_raw = accuracy_score(y_test, y_pred_xgb_raw)
logloss_xgb_raw = log_loss(y_test, y_proba_xgb_raw)
print(f"Accuracy (sin calibrar): {acc_xgb_raw:.4f}")
print(f"Log-loss (sin calibrar): {logloss_xgb_raw:.4f}")

print("\n--- AVANZADO: XGBoost + Calibración isotónica (cv=5) ---")

xgb_calibrado = CalibratedClassifierCV(
    XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=3,
        random_state=42, eval_metric='mlogloss',
    ),
    method='isotonic', cv=5
)
xgb_calibrado.fit(X_train_imp, y_train)

y_pred_xgb = xgb_calibrado.predict(X_test_imp)
y_proba_xgb = xgb_calibrado.predict_proba(X_test_imp)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
logloss_xgb = log_loss(y_test, y_proba_xgb)

print(f"Accuracy (calibrado): {acc_xgb:.4f}")
print(f"Log-loss (calibrado): {logloss_xgb:.4f}")
print(f"\nMatriz de confusión (XGBoost calibrado):")
print(confusion_matrix(y_test, y_pred_xgb))
print(f"\n{classification_report(y_test, y_pred_xgb, target_names=le.classes_)}")

# ============================================================
# 6. IMPORTANCIA DE VARIABLES (preview, XAI completo en Semana 4)
# ============================================================
print("\n--- Top 10 variables más importantes (XGBoost sin calibrar) ---")
importancias = pd.Series(
    xgb_model.feature_importances_, index=FEATURES_SEGURAS_V2
).sort_values(ascending=False)
print(importancias.head(10))

# ============================================================
# 6b. EVALUACIÓN ESPECÍFICA EN PARTIDOS NEUTRALES (lo más parecido
#     al escenario real del Mundial 2026)
# ============================================================
print("\n" + "=" * 70)
print(" EVALUACIÓN EN SUBCONJUNTO is_neutral=1 (proxy del Mundial 2026)")
print("=" * 70)

mask_neutral_test = test['is_neutral'] == 1
X_test_neutral = X_test[mask_neutral_test]
y_test_neutral = y_test[mask_neutral_test.values]

print(f"Partidos neutrales en test: {mask_neutral_test.sum()}/{len(test)}")

if mask_neutral_test.sum() > 0:
    X_test_neutral_imp = imputer_xgb.transform(X_test_neutral)
    y_pred_neutral = xgb_calibrado.predict(X_test_neutral_imp)
    y_proba_neutral = xgb_calibrado.predict_proba(X_test_neutral_imp)

    acc_neutral = accuracy_score(y_test_neutral, y_pred_neutral)
    logloss_neutral = log_loss(y_test_neutral, y_proba_neutral, labels=[0, 1, 2])

    print(f"Accuracy (XGBoost calibrado, solo neutrales): {acc_neutral:.4f}")
    print(f"Log-loss (XGBoost calibrado, solo neutrales): {logloss_neutral:.4f}")
    print(f"\nMatriz de confusión (solo partidos neutrales):")
    print(confusion_matrix(y_test_neutral, y_pred_neutral))

    clase_mayoritaria_neutral = pd.Series(
        y_test_raw[mask_neutral_test.values]
    ).value_counts().idxmax()
    acc_ingenuo_neutral = accuracy_score(
        y_test_raw[mask_neutral_test.values],
        [clase_mayoritaria_neutral] * mask_neutral_test.sum()
    )
    print(f"\nModelo ingenuo en neutrales (siempre '{clase_mayoritaria_neutral}'): "
          f"{acc_ingenuo_neutral:.4f}")
    print(f"Mejora del modelo sobre el ingenuo en neutrales: "
          f"{(acc_neutral - acc_ingenuo_neutral)*100:+.2f} puntos")
else:
    acc_neutral, logloss_neutral = np.nan, np.nan
    print("⚠️ No hay partidos neutrales en el test set.")

# ============================================================
# 7. TABLA COMPARATIVA FINAL
# ============================================================
print("\n" + "=" * 70)
print(" TABLA COMPARATIVA - MERCADO 1X2")
print("=" * 70)
tabla = pd.DataFrame({
    'modelo': ['Ingenuo (clase mayoritaria)', 'Logistic Regression (baseline)',
               'XGBoost (sin calibrar)', 'XGBoost (calibrado isotonic)'],
    'accuracy': [acc_ingenuo, acc_base, acc_xgb_raw, acc_xgb],
    'log_loss': [np.nan, logloss_base, logloss_xgb_raw, logloss_xgb],
})
print(tabla.to_string(index=False))

# ============================================================
# 8. GUARDAR MODELOS Y MÉTRICAS
# ============================================================
joblib.dump(baseline_pipeline, os.path.join(MODELS_DIR, '1x2_baseline_logreg.pkl'))
joblib.dump(xgb_calibrado, os.path.join(MODELS_DIR, '1x2_xgboost_calibrado.pkl'))
joblib.dump(imputer_xgb, os.path.join(MODELS_DIR, '1x2_imputer.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, '1x2_label_encoder.pkl'))

metricas = {
    'mercado': '1X2',
    'target': TARGET,
    'fecha_corte_train_test': '2018-01-01',
    'n_train': len(train),
    'n_test': len(test),
    'modelo_ingenuo': {'accuracy': float(acc_ingenuo)},
    'baseline_logreg': {'accuracy': float(acc_base), 'log_loss': float(logloss_base)},
    'xgboost_sin_calibrar': {'accuracy': float(acc_xgb_raw), 'log_loss': float(logloss_xgb_raw)},
    'xgboost_calibrado': {'accuracy': float(acc_xgb), 'log_loss': float(logloss_xgb)},
    'xgboost_calibrado_solo_neutrales': {
        'accuracy': float(acc_neutral) if not np.isnan(acc_neutral) else None,
        'log_loss': float(logloss_neutral) if not np.isnan(logloss_neutral) else None,
        'n_partidos': int(mask_neutral_test.sum()),
    },
    'top_10_features': importancias.head(10).to_dict(),
}
metrics_path = os.path.join(MODELS_DIR, '1x2_metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\n💾 Modelos guardados en: {MODELS_DIR}/")
print(f"💾 Métricas guardadas en: {metrics_path}")

print("""
PRÓXIMO PASO:
  - 12_modelo_goles.py -> target_ou25 y target_btts (mismo patrón)
""")