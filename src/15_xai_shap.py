"""
15_xai_shap.py
======================================================================
Semana 4 — Explicabilidad (XAI) con SHAP para los modelos XGBoost
calibrados de la Semana 3.

Por qué SHAP sobre el modelo SIN calibrar (no el CalibratedClassifierCV):
  - shap.TreeExplainer necesita acceso directo a los árboles del
    modelo. CalibratedClassifierCV envuelve a XGBoost en una capa de
    calibración (regresión isotónica) que NO es un árbol -> SHAP no
    puede explicarla directamente.
  - Solución estándar: reentrenar un XGBoost simple (mismos
    hiperparámetros, mismos datos) SOLO para extraer los SHAP values.
    Las probabilidades que se muestran en el dashboard siguen siendo
    las del modelo calibrado; SHAP solo explica la lógica de decisión
    interna del árbol, que es la misma en ambos.

Genera, para cada mercado:
  1. SHAP values globales -> qué variables importan más en general
     (gráfico de barras + beeswarm)
  2. SHAP values locales -> por qué el modelo predijo X para un
     partido específico (waterfall plot)
  3. Guarda los SHAP values en .npy para reutilizar en el dashboard
     sin recalcular

Requiere: pip install shap
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')  # backend sin pantalla, solo guardar archivos
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'report', 'figuras_shap')
os.makedirs(REPORT_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_lists import FEATURES_SEGURAS, TARGETS

print("=" * 70)
print(" 15_xai_shap.py - Explicabilidad SHAP por mercado")
print("=" * 70)

train_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'train_set.csv'))
test_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'test_set.csv'))
df_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'matches_features_v2.csv'))


def calcular_shap_mercado(nombre_mercado, target_col, X_train, y_train, X_explicar,
                            features, xgb_params, is_multiclass=False, class_names=None):
    """
    Entrena un XGBoost simple (sin calibrar) sobre los mismos datos
    del modelo de producción, y calcula SHAP values para explicarlo.
    Retorna el explainer y los shap_values para uso posterior.
    """
    print("\n" + "#" * 70)
    print(f" XAI: {nombre_mercado}")
    print("#" * 70)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_explicar_imp = imputer.transform(X_explicar)

    model = XGBClassifier(**xgb_params)
    model.fit(X_train_imp, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_explicar_imp)

    print(f"Shape de SHAP values: {np.array(shap_values.values).shape}")

    # --- Gráfico 1: importancia global (bar plot) ---
    plt.figure()
    if is_multiclass:
        # Para multiclase, promediar |SHAP| sobre todas las clases
        shap.summary_plot(
            [shap_values.values[:, :, i] for i in range(shap_values.values.shape[2])],
            X_explicar, feature_names=features, class_names=class_names,
            plot_type='bar', show=False
        )
    else:
        shap.summary_plot(shap_values.values, X_explicar, feature_names=features,
                           plot_type='bar', show=False)
    plt.title(f'Importancia global SHAP — {nombre_mercado}')
    plt.tight_layout()
    bar_path = os.path.join(REPORT_DIR, f'{nombre_mercado}_shap_bar.png')
    plt.savefig(bar_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"💾 {bar_path}")

    # --- Gráfico 2: beeswarm (distribución de impacto por feature) ---
    plt.figure()
    if is_multiclass:
        # beeswarm solo tiene sentido por clase en multiclase -> usamos
        # la clase de mayor interés (normalmente la última, ej. fav_win)
        shap.summary_plot(
            shap_values.values[:, :, -1], X_explicar, feature_names=features, show=False
        )
    else:
        shap.summary_plot(shap_values.values, X_explicar, feature_names=features, show=False)
    plt.title(f'Distribución de impacto SHAP — {nombre_mercado}')
    plt.tight_layout()
    beeswarm_path = os.path.join(REPORT_DIR, f'{nombre_mercado}_shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"💾 {beeswarm_path}")

    # --- Guardar SHAP values en .npy para reutilizar sin recalcular ---
    np.save(os.path.join(DATA_PROCESSED, f'shap_values_{nombre_mercado}.npy'),
            shap_values.values)

    return explainer, shap_values, model, imputer


# ============================================================
# 1. SHAP PARA MERCADOS BINARIOS (mismo patrón, reutilizable)
# ============================================================
MERCADOS_BINARIOS = [
    {
        'nombre': 'over_under_25',
        'target': TARGETS['over_under_25'],
        'params': dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective='binary:logistic', random_state=42, eval_metric='logloss'),
    },
    {
        'nombre': 'btts',
        'target': TARGETS['btts'],
        'params': dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective='binary:logistic', random_state=42, eval_metric='logloss'),
    },
]

resultados_shap = {}

for cfg in MERCADOS_BINARIOS:
    target_col = cfg['target']
    train = train_full[train_full[target_col].notna()].copy()
    test = test_full[test_full[target_col].notna()].copy()

    X_train = train[FEATURES_SEGURAS]
    y_train = train[target_col].astype(int)
    # Para los gráficos globales, usamos una muestra del test (más rápido,
    # representativo). Para SHAP en datasets grandes, 500-1000 filas alcanzan.
    X_explicar = test[FEATURES_SEGURAS].sample(min(800, len(test)), random_state=42)

    explainer, shap_values, model, imputer = calcular_shap_mercado(
        cfg['nombre'], target_col, X_train, y_train, X_explicar,
        FEATURES_SEGURAS, cfg['params']
    )
    resultados_shap[cfg['nombre']] = {
        'explainer': explainer, 'shap_values': shap_values,
        'model': model, 'imputer': imputer, 'X_explicar': X_explicar,
    }

# ============================================================
# 2. SHAP PARA TARJETAS OU3.5 (dataset reducido, solo Mundial)
# ============================================================
target_col = TARGETS['tarjetas_ou35']
df_tarjetas = df_full[df_full[target_col].notna()].copy()
X_train_t = df_tarjetas[FEATURES_SEGURAS]
y_train_t = df_tarjetas[target_col].astype(int)
# Dataset chico -> explicar sobre el 100%, no hace falta muestrear
X_explicar_t = X_train_t.copy()

params_tarjetas = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                        reg_lambda=2.0, objective='binary:logistic',
                        random_state=42, eval_metric='logloss')

explainer_t, shap_values_t, model_t, imputer_t = calcular_shap_mercado(
    'tarjetas_ou35', target_col, X_train_t, y_train_t, X_explicar_t,
    FEATURES_SEGURAS, params_tarjetas
)
resultados_shap['tarjetas_ou35'] = {
    'explainer': explainer_t, 'shap_values': shap_values_t,
    'model': model_t, 'imputer': imputer_t, 'X_explicar': X_explicar_t,
}

# ============================================================
# 3. SHAP PARA 1X2 (multiclase) — usamos el modelo solo-neutrales
#    porque es el que va al dashboard según la decisión de Semana 3
# ============================================================
print("\n" + "#" * 70)
print(" XAI: 1x2 (multiclase, modelo solo-neutrales)")
print("#" * 70)

target_col = TARGETS['1x2']
train_n = train_full[train_full['is_neutral'] == 1].copy()
test_n = test_full[test_full['is_neutral'] == 1].copy()

features_1x2 = [f for f in FEATURES_SEGURAS if f != 'is_neutral']
X_train_1x2 = train_n[features_1x2]
le_1x2 = LabelEncoder()
y_train_1x2 = le_1x2.fit_transform(train_n[target_col])

imputer_1x2 = SimpleImputer(strategy='median')
X_train_1x2_imp = imputer_1x2.fit_transform(X_train_1x2)

params_1x2 = dict(n_estimators=120, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                   objective='multi:softprob', num_class=3,
                   random_state=42, eval_metric='mlogloss')

model_1x2 = XGBClassifier(**params_1x2)
model_1x2.fit(X_train_1x2_imp, y_train_1x2)

X_explicar_1x2 = test_n[features_1x2].sample(min(500, len(test_n)), random_state=42)
X_explicar_1x2_imp = imputer_1x2.transform(X_explicar_1x2)

explainer_1x2 = shap.TreeExplainer(model_1x2)
shap_values_1x2 = explainer_1x2(X_explicar_1x2_imp)

plt.figure()
shap.summary_plot(
    [shap_values_1x2.values[:, :, i] for i in range(3)],
    X_explicar_1x2, feature_names=features_1x2,
    class_names=list(le_1x2.classes_), plot_type='bar', show=False
)
plt.title('Importancia global SHAP — 1X2 (multiclase)')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, '1x2_shap_bar.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f"💾 {os.path.join(REPORT_DIR, '1x2_shap_bar.png')}")

np.save(os.path.join(DATA_PROCESSED, 'shap_values_1x2.npy'), shap_values_1x2.values)

resultados_shap['1x2'] = {
    'explainer': explainer_1x2, 'shap_values': shap_values_1x2,
    'model': model_1x2, 'imputer': imputer_1x2, 'X_explicar': X_explicar_1x2,
    'label_encoder': le_1x2, 'features': features_1x2,
}

# ============================================================
# 4. EXPLICACIÓN LOCAL (waterfall) — UN PARTIDO DE EJEMPLO
# ============================================================
print("\n" + "=" * 70)
print(" EXPLICACIÓN LOCAL — Ejemplo: partido con mayor probabilidad")
print(" de Over 2.5 en el set de explicación")
print("=" * 70)

ou25_data = resultados_shap['over_under_25']
probs_ou25 = ou25_data['model'].predict_proba(
    ou25_data['imputer'].transform(ou25_data['X_explicar'])
)[:, 1]
idx_ejemplo = np.argmax(probs_ou25)

print(f"Partido de ejemplo (índice {idx_ejemplo} en la muestra), "
      f"probabilidad over 2.5: {probs_ou25[idx_ejemplo]:.3f}")
print(f"Features de este partido:")
print(ou25_data['X_explicar'].iloc[idx_ejemplo])

plt.figure()
shap.plots.waterfall(ou25_data['shap_values'][idx_ejemplo], show=False)
plt.title('Explicación local — Over/Under 2.5 (ejemplo)')
plt.tight_layout()
waterfall_path = os.path.join(REPORT_DIR, 'ejemplo_local_ou25_waterfall.png')
plt.savefig(waterfall_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"💾 {waterfall_path}")

# ============================================================
# 5. RESUMEN: TOP FEATURES POR MERCADO (texto, para el informe)
# ============================================================
print("\n" + "=" * 70)
print(" RESUMEN: TOP 5 FEATURES POR MERCADO (|SHAP| promedio)")
print("=" * 70)

resumen_shap = {}
for nombre, data in resultados_shap.items():
    sv = data['shap_values'].values
    feats = data.get('features', FEATURES_SEGURAS)
    if sv.ndim == 3:  # multiclase: promediar sobre clases
        importancia = np.abs(sv).mean(axis=(0, 2))
    else:
        importancia = np.abs(sv).mean(axis=0)
    top5 = pd.Series(importancia, index=feats).sort_values(ascending=False).head(5)
    resumen_shap[nombre] = top5.to_dict()
    print(f"\n{nombre}:")
    print(top5)

resumen_path = os.path.join(MODELS_DIR, 'shap_resumen_por_mercado.json')
with open(resumen_path, 'w', encoding='utf-8') as f:
    json.dump(resumen_shap, f, indent=2, ensure_ascii=False)
print(f"\n💾 {resumen_path}")

print(f"\n💾 Gráficos guardados en: {REPORT_DIR}/")
print(f"💾 SHAP values (.npy) en: {DATA_PROCESSED}/shap_values_*.npy")
print("""
PRÓXIMO PASO:
  - 16_value_bet_detector.py -> función de detección de value bets
    + Kelly Criterion fraccionario
""")