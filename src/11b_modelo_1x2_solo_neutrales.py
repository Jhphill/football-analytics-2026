"""
11b_modelo_1x2_solo_neutrales.py
======================================================================
Variante de 11_modelo_1x2.py: entrena el modelo SOLO con partidos
is_neutral=1 (en vez de entrenar con todo el histórico y evaluar
después en el subconjunto neutral).

Motivación:
  - El Mundial 2026 es ~100% sede neutral.
  - En 11_modelo_1x2.py, el modelo entrenado con TODO el histórico
    (mayoría no-neutral) solo superó al ingenuo por +0.6 puntos al
    evaluarlo en partidos neutrales -> probablemente está "gastando"
    capacidad en patrones de partidos no neutrales (donde sí hay
    ventaja de cancha real) que no aplican al caso de uso real.
  - Entrenar y testear exclusivamente sobre population neutral es
    conceptualmente más correcto para este proyecto, aunque el
    dataset de entrenamiento sea mucho más chico.

Limitación esperada: muchos menos datos de entrenamiento
(~8,000-9,000 partidos neutrales totales vs 43,690 en el dataset
completo) -> se usa un modelo XGBoost más simple (menos profundidad)
para reducir riesgo de overfitting, y se reporta validación cruzada
ademas del test temporal.

Output: models/1x2_neutral_xgboost_calibrado.pkl + métricas .json
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_lists import FEATURES_SEGURAS, TARGETS

TARGET = TARGETS['1x2']

print("=" * 70)
print(" 11b_modelo_1x2_solo_neutrales.py")
print(" Entrenamiento exclusivo sobre partidos is_neutral=1")
print("=" * 70)

# ============================================================
# 1. CARGAR Y FILTRAR SOLO is_neutral=1
# ============================================================
train_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'train_set.csv'))
test_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'test_set.csv'))

train = train_full[train_full['is_neutral'] == 1].copy()
test = test_full[test_full['is_neutral'] == 1].copy()

print(f"\nTrain neutral: {train.shape} (de {train_full.shape[0]:,} totales)")
print(f"Test neutral:  {test.shape} (de {test_full.shape[0]:,} totales)")

# is_neutral ya no aporta información (es constante =1) -> se excluye
features_neutral = [f for f in FEATURES_SEGURAS if f != 'is_neutral']
print(f"\nFeatures usadas ({len(features_neutral)}, excluida is_neutral por ser constante)")

X_train = train[features_neutral].copy()
X_test = test[features_neutral].copy()
y_train_raw = train[TARGET]
y_test_raw = test[TARGET]

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

print(f"\nDistribución target en train neutral:")
print(y_train_raw.value_counts(normalize=True).round(3))
print(f"\nDistribución target en test neutral:")
print(y_test_raw.value_counts(normalize=True).round(3))

# ============================================================
# 2. MODELO INGENUO (referencia)
# ============================================================
clase_mayoritaria = y_train_raw.value_counts().idxmax()
acc_ingenuo = accuracy_score(y_test_raw, [clase_mayoritaria] * len(y_test_raw))
print(f"\n--- Modelo ingenuo (siempre '{clase_mayoritaria}') ---")
print(f"Accuracy: {acc_ingenuo:.4f}")

# ============================================================
# 3. BASELINE: Logistic Regression
# ============================================================
print("\n--- BASELINE: Logistic Regression (solo neutrales) ---")

baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])
baseline_pipeline.fit(X_train, y_train)

y_pred_base = baseline_pipeline.predict(X_test)
y_proba_base = baseline_pipeline.predict_proba(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
logloss_base = log_loss(y_test, y_proba_base, labels=[0, 1, 2])

print(f"Accuracy: {acc_base:.4f}")
print(f"Log-loss: {logloss_base:.4f}")

# ============================================================
# 4. AVANZADO: XGBoost (modelo más simple, menos profundo, por
#    el tamaño reducido del dataset -> menor riesgo de overfitting)
# ============================================================
print("\n--- AVANZADO: XGBoost simplificado (solo neutrales) ---")

imputer_xgb = SimpleImputer(strategy='median')
X_train_imp = imputer_xgb.fit_transform(X_train)
X_test_imp = imputer_xgb.transform(X_test)

xgb_params = dict(
    n_estimators=120,
    max_depth=3,          # más conservador que el modelo general (era 4)
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,   # regularización adicional por menos datos
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss',
)

xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_train_imp, y_train)

y_pred_xgb_raw = xgb_model.predict(X_test_imp)
y_proba_xgb_raw = xgb_model.predict_proba(X_test_imp)
acc_xgb_raw = accuracy_score(y_test, y_pred_xgb_raw)
logloss_xgb_raw = log_loss(y_test, y_proba_xgb_raw, labels=[0, 1, 2])
print(f"Accuracy (sin calibrar): {acc_xgb_raw:.4f}")
print(f"Log-loss (sin calibrar): {logloss_xgb_raw:.4f}")

# Validación cruzada en TRAIN (5-fold) -> más robusto dado el tamaño chico
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    XGBClassifier(**xgb_params), X_train_imp, y_train, cv=cv, scoring='accuracy'
)
print(f"\nValidación cruzada (5-fold) en train neutral:")
print(f"  Accuracy por fold: {np.round(cv_scores, 4)}")
print(f"  Media: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n--- AVANZADO: XGBoost + Calibración isotónica (cv=5) ---")
xgb_calibrado = CalibratedClassifierCV(
    XGBClassifier(**xgb_params), method='isotonic', cv=5
)
xgb_calibrado.fit(X_train_imp, y_train)

y_pred_xgb = xgb_calibrado.predict(X_test_imp)
y_proba_xgb = xgb_calibrado.predict_proba(X_test_imp)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
logloss_xgb = log_loss(y_test, y_proba_xgb, labels=[0, 1, 2])

print(f"Accuracy (calibrado): {acc_xgb:.4f}")
print(f"Log-loss (calibrado): {logloss_xgb:.4f}")
print(f"\nMatriz de confusión (XGBoost calibrado, solo neutrales):")
print(confusion_matrix(y_test, y_pred_xgb))
print(f"\n{classification_report(y_test, y_pred_xgb, target_names=le.classes_)}")

# ============================================================
# 5. IMPORTANCIA DE VARIABLES
# ============================================================
print("\n--- Top 10 variables más importantes (modelo solo-neutrales) ---")
importancias = pd.Series(
    xgb_model.feature_importances_, index=features_neutral
).sort_values(ascending=False)
print(importancias.head(10))

# ============================================================
# 6. COMPARACIÓN FINAL: modelo general vs modelo solo-neutrales
#    (ambos evaluados en el MISMO subconjunto de test neutral)
# ============================================================
print("\n" + "=" * 70)
print(" COMPARACIÓN: modelo entrenado con TODO vs SOLO con neutrales")
print(" (evaluados en el mismo test set de partidos neutrales)")
print("=" * 70)

metrics_path_general = os.path.join(MODELS_DIR, '1x2_metrics.json')
if os.path.exists(metrics_path_general):
    with open(metrics_path_general, 'r', encoding='utf-8') as f:
        metrics_general = json.load(f)
    acc_general_neutral = metrics_general.get('xgboost_calibrado_solo_neutrales', {}).get('accuracy')
    logloss_general_neutral = metrics_general.get('xgboost_calibrado_solo_neutrales', {}).get('log_loss')
else:
    acc_general_neutral, logloss_general_neutral = None, None

tabla_comparativa = pd.DataFrame({
    'modelo': [
        'Ingenuo (clase mayoritaria)',
        'XGBoost entrenado con TODO (eval. en neutrales)',
        'XGBoost entrenado SOLO con neutrales',
    ],
    'accuracy': [acc_ingenuo, acc_general_neutral, acc_xgb],
    'log_loss': [np.nan, logloss_general_neutral, logloss_xgb],
})
print(tabla_comparativa.to_string(index=False))

mejora_vs_general = (acc_xgb - acc_general_neutral) if acc_general_neutral else None
if mejora_vs_general is not None:
    print(f"\nDiferencia (solo-neutrales vs entrenado-con-todo): {mejora_vs_general*100:+.2f} puntos de accuracy")

# ============================================================
# 7. GUARDAR
# ============================================================
joblib.dump(baseline_pipeline, os.path.join(MODELS_DIR, '1x2_neutral_baseline_logreg.pkl'))
joblib.dump(xgb_calibrado, os.path.join(MODELS_DIR, '1x2_neutral_xgboost_calibrado.pkl'))
joblib.dump(imputer_xgb, os.path.join(MODELS_DIR, '1x2_neutral_imputer.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, '1x2_neutral_label_encoder.pkl'))

metricas = {
    'mercado': '1X2 (solo partidos neutrales)',
    'target': TARGET,
    'n_train_neutral': len(train),
    'n_test_neutral': len(test),
    'modelo_ingenuo': {'accuracy': float(acc_ingenuo)},
    'baseline_logreg': {'accuracy': float(acc_base), 'log_loss': float(logloss_base)},
    'xgboost_sin_calibrar': {'accuracy': float(acc_xgb_raw), 'log_loss': float(logloss_xgb_raw)},
    'xgboost_calibrado': {'accuracy': float(acc_xgb), 'log_loss': float(logloss_xgb)},
    'cv_5fold_accuracy_mean': float(cv_scores.mean()),
    'cv_5fold_accuracy_std': float(cv_scores.std()),
    'comparacion_vs_modelo_general': {
        'acc_modelo_general_en_neutrales': acc_general_neutral,
        'acc_modelo_solo_neutrales': float(acc_xgb),
        'diferencia_puntos': float(mejora_vs_general) if mejora_vs_general is not None else None,
    },
    'top_10_features': importancias.head(10).to_dict(),
}
metrics_path = os.path.join(MODELS_DIR, '1x2_neutral_metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\n💾 Modelos guardados en: {MODELS_DIR}/")
print(f"💾 Métricas guardadas en: {metrics_path}")

print("""
DECISIÓN A TOMAR:
  - Si el modelo "solo-neutrales" mejora claramente sobre el general
    evaluado en neutrales -> usar este modelo para el dashboard del
    Mundial 2026 (1x2_neutral_xgboost_calibrado.pkl)
  - Si no mejora (dataset muy chico, ruido) -> documentar la
    limitación del mercado 1X2 puro y priorizar mercados con más
    señal (goles, tarjetas) para el dashboard

PRÓXIMO PASO:
  - 12_modelo_goles.py -> target_ou25 y target_btts
""")