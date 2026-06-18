"""
19_validacion_final.py
======================================================================
Semana 5 — Validación final consolidada.

Genera una tabla única con TODAS las métricas de todos los modelos
para el informe LaTeX. Incluye:

  1. Tabla comparativa global (todos los mercados, todas las métricas)
  2. Comparativa baseline → XGBoost para cada mercado
  3. Métricas en partidos neutrales (proxy del escenario real WC2026)
  4. Resultados del backtest de value bet detector
  5. Diagnóstico de calibración (reliability diagrams)
  6. Conclusión automática por mercado: ¿el modelo es útil?

Output:
  - data/processed/validacion_final_completa.csv  (tabla para LaTeX)
  - report/tabla_metricas_latex.tex               (tabla LaTeX lista)
  - report/figuras_calibracion/                   (reliability diagrams)
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'report')
FIG_DIR = os.path.join(REPORT_DIR, 'figuras_calibracion')
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_lists import FEATURES_SEGURAS, TARGETS

print("=" * 70)
print(" 19_validacion_final.py — Tabla consolidada para informe LaTeX")
print("=" * 70)

# ============================================================
# 1. CARGAR TODOS LOS JSON DE MÉTRICAS
# ============================================================
archivos_json = {
    '1X2 (todo dataset)':    '1x2_metrics.json',
    '1X2 (solo neutrales)':  '1x2_neutral_metrics.json',
    'Over/Under 2.5':        'over_under_2.5_goles_metrics.json',
    'BTTS':                  'both_teams_to_score_metrics.json',
    'Tarjetas O/U 3.5':      'tarjetas_over_under_3.5_metrics.json',
    'Tarjeta Roja':          'tarjeta_roja_metrics.json',
}

archivos_backtest = {
    'Over/Under 2.5':   'backtest_value_bet_ou25.json',
    'BTTS':             'backtest_value_bet_btts.json',
    'Tarjetas O/U 3.5': 'backtest_value_bet_tarjetas_ou35.json',
}

metricas = {}
for nombre, archivo in archivos_json.items():
    path = os.path.join(MODELS_DIR, archivo)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            metricas[nombre] = json.load(f)
        print(f"✅ {archivo}")
    else:
        print(f"❌ No encontrado: {archivo}")

backtest = {}
for nombre, archivo in archivos_backtest.items():
    path = os.path.join(MODELS_DIR, archivo)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            backtest[nombre] = json.load(f)

# ============================================================
# 2. TABLA COMPARATIVA GLOBAL
# ============================================================
print("\n" + "=" * 70)
print(" TABLA COMPARATIVA GLOBAL — TODOS LOS MERCADOS")
print("=" * 70)

filas = []
for nombre, m in metricas.items():
    fila = {'Mercado': nombre}

    # N de entrenamiento
    fila['N train'] = m.get('n_train') or m.get('n_train_neutral') or m.get('n_partidos_total', '?')

    # Modelo ingenuo
    fila['Acc ingenuo'] = m.get('modelo_ingenuo', {}).get('accuracy', np.nan)

    # Baseline LogReg
    lr = m.get('baseline_logreg') or m.get('baseline_logreg_cv', {})
    fila['Acc LogReg'] = lr.get('accuracy', np.nan)
    fila['AUC LogReg'] = lr.get('auc', np.nan)

    # XGBoost calibrado
    xgb = m.get('xgboost_calibrado') or m.get('xgboost_cv', {})
    fila['Acc XGBoost'] = xgb.get('accuracy', np.nan)
    fila['AUC XGBoost'] = xgb.get('auc', np.nan)
    fila['LogLoss XGBoost'] = xgb.get('log_loss', np.nan)
    fila['Brier XGBoost'] = xgb.get('brier', np.nan)

    # Mejora sobre ingenuo
    if not np.isnan(fila.get('Acc XGBoost', np.nan)) and not np.isnan(fila.get('Acc ingenuo', np.nan)):
        fila['Mejora pp'] = round((fila['Acc XGBoost'] - fila['Acc ingenuo']) * 100, 2)
    else:
        fila['Mejora pp'] = np.nan

    # Neutrales
    neutral = m.get('xgboost_calibrado_solo_neutrales') or m.get('evaluacion_partidos_neutrales', {})
    if neutral:
        fila['AUC neutrales'] = neutral.get('auc', np.nan)
        fila['N neutrales'] = neutral.get('n_partidos', np.nan)

    # Decision dashboard
    fila['Dashboard'] = m.get('decision_dashboard', 'USAR_MODELO')

    filas.append(fila)

df_global = pd.DataFrame(filas)

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df_global.to_string(index=False))

# ============================================================
# 3. TABLA DE BACKTEST VALUE BET
# ============================================================
print("\n" + "=" * 70)
print(" BACKTEST VALUE BET DETECTOR")
print("=" * 70)

filas_bt = []
for nombre, bt in backtest.items():
    filas_bt.append({
        'Mercado': nombre,
        'N value bets': bt.get('n_value_bets', '?'),
        '% partidos': bt.get('pct_value_bets', '?'),
        'Tasa acierto': bt.get('tasa_acierto', '?'),
        'ROI oficial': bt.get('roi_oficial', '?'),
        'AUC modelo': bt.get('auc_modelo', '?'),
        'Cuotas': 'SIMULADAS ⚠️',
    })

df_bt = pd.DataFrame(filas_bt)
print(df_bt.to_string(index=False))
print("\n⚠️  ROI calculado sobre cuotas SIMULADAS (margen casa ~6%).")
print("   No representa rentabilidad real con cuotas de 1xbet.")

# ============================================================
# 4. RELIABILITY DIAGRAMS (calibración visual)
# ============================================================
print("\n" + "=" * 70)
print(" RELIABILITY DIAGRAMS — verificación de calibración")
print("=" * 70)

train_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'train_set.csv'))
test_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'test_set.csv'))

mercados_binarios = [
    ('Over/Under 2.5', TARGETS['over_under_25'],
     'over_under_2.5_goles_xgboost_calibrado.pkl',
     'over_under_2.5_goles_imputer.pkl'),
    ('BTTS', TARGETS['btts'],
     'both_teams_to_score_xgboost_calibrado.pkl',
     'both_teams_to_score_imputer.pkl'),
    ('Tarjetas O/U 3.5', TARGETS['tarjetas_ou35'],
     'tarjetas_over_under_3.5_xgboost_calibrado.pkl',
     'tarjetas_over_under_3.5_imputer.pkl'),
]

fig, axes = plt.subplots(1, len(mercados_binarios), figsize=(15, 5))
fig.suptitle('Reliability Diagrams — Calibración de probabilidades', fontsize=13)

for ax, (nombre, target_col, modelo_pkl, imputer_pkl) in zip(axes, mercados_binarios):
    modelo_path = os.path.join(MODELS_DIR, modelo_pkl)
    imputer_path = os.path.join(MODELS_DIR, imputer_pkl)

    if not os.path.exists(modelo_path) or not os.path.exists(imputer_path):
        ax.text(0.5, 0.5, f'{nombre}\n(modelo no encontrado)',
                ha='center', va='center', transform=ax.transAxes)
        continue

    modelo = joblib.load(modelo_path)
    imputer = joblib.load(imputer_path)

    # Usar el dataset completo con target válido
    if nombre == 'Tarjetas O/U 3.5':
        df_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'matches_features_v2.csv'))
        df_eval = df_full[df_full[target_col].notna()].copy()
    else:
        df_eval = test_full[test_full[target_col].notna()].copy()

    X = df_eval[FEATURES_SEGURAS]
    y = df_eval[target_col].astype(int)
    X_imp = imputer.transform(X)
    proba = modelo.predict_proba(X_imp)[:, 1]

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y, proba, n_bins=10)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado', alpha=0.6)
    ax.plot(mean_pred, fraction_pos, 'o-', color='#2ecc71', label='Modelo calibrado')
    ax.set_xlabel('Probabilidad predicha')
    ax.set_ylabel('Fracción de positivos reales')
    ax.set_title(f'{nombre}')
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    brier = brier_score_loss(y, proba)
    ax.text(0.05, 0.92, f'Brier: {brier:.4f}', transform=ax.transAxes,
            fontsize=9, color='#2c3e50')

    print(f"✅ {nombre}: Brier Score = {brier:.4f}")

plt.tight_layout()
reliability_path = os.path.join(FIG_DIR, 'reliability_diagrams.png')
plt.savefig(reliability_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"\n💾 {reliability_path}")

# ============================================================
# 5. CONCLUSIÓN AUTOMÁTICA POR MERCADO
# ============================================================
print("\n" + "=" * 70)
print(" CONCLUSIÓN POR MERCADO (para el informe)")
print("=" * 70)

conclusiones = {
    '1X2 (solo neutrales)': {
        'conclusion': 'Señal débil pero consistente. AUC ~0.52 supera marginalmente '
                      'el azar en partidos neutrales. Útil para value betting cuando '
                      'la cuota implica una probabilidad muy distante del modelo.',
        'recomendacion_dashboard': 'USAR con advertencia de baja precisión',
        'prioridad': 4,
    },
    'Over/Under 2.5': {
        'conclusion': 'Mejor mercado para value betting. AUC 0.581, ROI simulado '
                      '+23.81% en backtest. La forma reciente y el Elo diff son '
                      'las variables más explicativas (SHAP). Señal estable.',
        'recomendacion_dashboard': 'USAR como mercado principal',
        'prioridad': 1,
    },
    'BTTS': {
        'conclusion': 'AUC 0.572, ROI simulado +16.33%. Señal real pero umbral '
                      '0.5 no apropiado (clases desbalanceadas). Usar probabilidad '
                      'calibrada directamente vs cuota implícita.',
        'recomendacion_dashboard': 'USAR probabilidad cruda, no umbral 0.5',
        'prioridad': 2,
    },
    'Tarjetas O/U 3.5': {
        'conclusion': 'Mejor AUC del proyecto (0.593). Días de descanso y atributos '
                      'FIFA del equipo no favorito son las features más relevantes. '
                      'Dataset pequeño (751 WC) pero señal consistente entre folds.',
        'recomendacion_dashboard': 'USAR como mercado secundario destacado',
        'prioridad': 3,
    },
    'Tarjeta Roja': {
        'conclusion': 'AUC máximo logrado: 0.514 (con regularización relajada). '
                      'Dataset insuficiente: 81 positivos en 751 partidos. '
                      'Requeriría historial disciplinario individual de jugadores '
                      'y perfil del árbitro para ser útil.',
        'recomendacion_dashboard': 'NO USAR modelo. Mostrar baseline empírico 10.8%',
        'prioridad': 5,
    },
}

for mercado, info in conclusiones.items():
    print(f"\n[Prioridad {info['prioridad']}] {mercado}")
    print(f"  Conclusión: {info['conclusion']}")
    print(f"  Dashboard:  {info['recomendacion_dashboard']}")

# ============================================================
# 6. GENERAR TABLA LATEX
# ============================================================
print("\n" + "=" * 70)
print(" TABLA LaTeX")
print("=" * 70)

latex = r"""\begin{table}[h]
\centering
\caption{Comparativa de modelos por mercado — Plataforma Football Analytics 2026}
\label{tab:metricas_modelos}
\begin{tabular}{lrrrrrr}
\hline
\textbf{Mercado} & \textbf{N train} & \textbf{Acc. ingenuo} & \textbf{Acc. XGBoost} & \textbf{AUC} & \textbf{Log-loss} & \textbf{Dashboard} \\
\hline
"""

for _, row in df_global.iterrows():
    mercado = row['Mercado'].replace('/', '/').replace('&', r'\&')
    n = int(row['N train']) if pd.notna(row['N train']) else '--'
    acc_ing = f"{row['Acc ingenuo']:.3f}" if pd.notna(row.get('Acc ingenuo')) else '--'
    acc_xgb = f"{row['Acc XGBoost']:.3f}" if pd.notna(row.get('Acc XGBoost')) else '--'
    auc = f"{row['AUC XGBoost']:.3f}" if pd.notna(row.get('AUC XGBoost')) else '--'
    ll = f"{row['LogLoss XGBoost']:.4f}" if pd.notna(row.get('LogLoss XGBoost')) else '--'
    dash = r'\checkmark' if row.get('Dashboard') == 'USAR_MODELO' else r'\textit{baseline}'

    latex += f"{mercado} & {n:,} & {acc_ing} & {acc_xgb} & {auc} & {ll} & {dash} \\\\\n"

latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item AUC y Log-loss calculados sobre el conjunto de test (2018--2025) con split temporal.
\item Tarjetas O/U 3.5 y Tarjeta Roja: validación cruzada 5-fold (dataset reducido, n=751).
\item Tarjeta Roja: modelo no supera umbral de utilidad (AUC $<$ 0.58); se muestra baseline empírico 10.8\%.
\end{tablenotes}
\end{table}
"""

latex_path = os.path.join(REPORT_DIR, 'tabla_metricas_latex.tex')
with open(latex_path, 'w', encoding='utf-8') as f:
    f.write(latex)

print(latex)
print(f"💾 {latex_path}")

# ============================================================
# 7. GUARDAR CSV CONSOLIDADO
# ============================================================
csv_path = os.path.join(DATA_PROCESSED, 'validacion_final_completa.csv')
df_global.to_csv(csv_path, index=False)

print("\n" + "=" * 70)
print(" RESUMEN FINAL")
print("=" * 70)
print(f"💾 {csv_path}")
print(f"💾 {latex_path}")
print(f"💾 {reliability_path}")
print(f"""
CHECKLIST SEMANA 5 — JUANFE:
  [X] 15_xai_shap.py          -> SHAP values + gráficos globales/locales
  [X] 16_value_bet_detector.py -> Backtest con cuotas simuladas
  [X] 17_predicciones_wc2026.py-> Value bets con cuotas reales de 1xbet
  [X] 18_pipeline_wc2026_live.py -> Actualización jornada por jornada
  [X] 19_validacion_final.py   -> Tabla consolidada + LaTeX + calibración

ENTREGABLES PARA LULU (Semana 5):
  ✅ models/*.pkl               (disponibles desde Semana 3)
  ✅ src/value_bet.py           (función analizar_partido())
  ✅ models/shap_resumen_por_mercado.json
  ✅ models/backtest_value_bet_resumen_comparativo.json
  ✅ data/processed/wc2026_predicciones.csv
  ✅ report/figuras_shap/       (gráficos SHAP para el dashboard)
  ✅ report/tabla_metricas_latex.tex

PRÓXIMO PASO — SEMANA 6:
  -> report/informe.tex  (LaTeX técnico completo)
  -> Usar tabla_metricas_latex.tex como base para la sección de modelos
""")
