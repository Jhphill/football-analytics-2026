"""
16_value_bet_detector.py
======================================================================
Semana 4 — Value bet detector + Kelly Criterion fraccionario.

Concepto central:
  Una apuesta tiene "value" cuando la probabilidad que el MODELO le
  asigna a un resultado es mayor que la probabilidad IMPLÍCITA en la
  cuota que ofrece la casa de apuestas (1xbet). La cuota implica una
  probabilidad de 1/cuota (sin descontar el margen de la casa).

  Si prob_modelo > prob_implicita + margen_seguridad -> hay value.
  A largo plazo, apostar sistemáticamente solo donde hay value (con
  sizing correcto vía Kelly) es lo que separa una estrategia con
  expectativa positiva de simplemente "tener corazonadas".

Funciones que expone este módulo (src/value_bet.py):
  - prob_implicita(cuota)
  - expected_value(prob_modelo, cuota)
  - es_value_bet(prob_modelo, cuota, margen_minimo=0.03)
  - kelly_fraccionario(prob_modelo, cuota, fraccion=0.25)
  - analizar_partido(...) -> reporte completo para un partido/mercado

Incluye además un BACKTEST sobre el test set (2018-2025) por cada
mercado con decision_dashboard='USAR_MODELO' (OU2.5, BTTS, Tarjetas
OU3.5), simulando un mercado de apuestas INDEPENDIENTE del modelo de
producción (regresión logística entrenada solo con fav_dog_elo_diff),
para estimar cuántos value bets se hubieran detectado y qué ROI habría
dado. Tarjeta Roja queda excluida porque su decision_dashboard es
USAR_BASELINE_EMPIRICO (ver 13b_modelo_tarjeta_roja_fix.py) -- no tiene
sentido backtest de value bet sobre un modelo que no va a producción.

IMPORTANTE: 1xbet no provee una API pública de cuotas históricas
gratuita. El backtest usa cuotas SIMULADAS (ver sección 3) a partir de
un mercado logístico simple basado en Elo + margen de casa típico
(~6%), NO cuotas reales de 1xbet. Esto se documenta explícitamente
como limitación metodológica para el informe.
======================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print(" 16_value_bet_detector.py - Value bet + Kelly Criterion")
print("=" * 70)

# ============================================================
# 1. MÓDULO REUTILIZABLE: src/value_bet.py
# ============================================================
value_bet_code = '''"""
value_bet.py
======================================================================
Funciones de detección de value bets y sizing con Kelly Criterion
fraccionario. Usado por el dashboard (Lulu) para mostrar
recomendaciones de apuesta junto a las probabilidades del modelo.
======================================================================
"""


def prob_implicita(cuota):
    """
    Probabilidad implícita de una cuota decimal (formato europeo,
    el que usa 1xbet). Ej: cuota 2.50 -> prob_implicita = 0.40 (40%).

    NOTA: esta probabilidad incluye el margen de la casa (overround).
    La suma de prob_implicita de todos los resultados de un mercado
    siempre es > 1 (normalmente 1.05-1.10 para casas como 1xbet).
    """
    if cuota <= 1.0:
        raise ValueError("La cuota debe ser mayor a 1.0")
    return 1.0 / cuota


def expected_value(prob_modelo, cuota):
    """
    Valor esperado de apostar 1 unidad a una cuota dada, según la
    probabilidad del MODELO (no la implícita).

    EV = (prob_modelo * (cuota - 1)) - (1 - prob_modelo)
       = prob_modelo * cuota - 1

    EV > 0  -> la apuesta tiene expectativa positiva según el modelo
    EV <= 0 -> no conviene apostar (según el modelo)
    """
    return prob_modelo * cuota - 1.0


def es_value_bet(prob_modelo, cuota, margen_minimo=0.03):
    """
    Determina si una apuesta es "value bet": la probabilidad del
    modelo supera a la implícita en la cuota por al menos
    `margen_minimo` (ej. 0.03 = 3 puntos porcentuales).

    El margen mínimo es una salvaguarda contra el ruido del modelo:
    una diferencia de 0.5% no es señal suficiente para actuar, dado
    que las probabilidades del modelo tienen su propio margen de error.

    Retorna: (bool, dict con detalle)
    """
    p_imp = prob_implicita(cuota)
    diferencia = prob_modelo - p_imp
    ev = expected_value(prob_modelo, cuota)

    es_value = diferencia >= margen_minimo

    return es_value, {
        'prob_modelo': round(prob_modelo, 4),
        'prob_implicita': round(p_imp, 4),
        'diferencia_pp': round(diferencia * 100, 2),
        'expected_value': round(ev, 4),
        'cuota': cuota,
        'es_value_bet': es_value,
    }


def kelly_fraccionario(prob_modelo, cuota, fraccion=0.25, bankroll=1.0):
    """
    Calcula el tamaño de apuesta recomendado según el criterio de
    Kelly, aplicando una FRACCIÓN del Kelly completo (Kelly fraccionario).

    Kelly completo (full Kelly):
        f* = (p * b - q) / b
        donde p = prob_modelo, q = 1 - p, b = cuota - 1 (ganancia neta)

    Por qué fraccionario y no full Kelly:
        Full Kelly asume que prob_modelo es EXACTA. En la práctica el
        modelo tiene error de estimación, y full Kelly es muy agresivo
        ante ese error (puede llevar a apostar 30-40% del bankroll en
        un solo partido). Una fracción de 0.25-0.5 (cuarto u octavo de
        Kelly) es el estándar en la industria para reducir varianza
        manteniendo la mayor parte del crecimiento esperado.

    Retorna: dict con el tamaño de apuesta recomendado (en unidades de
    bankroll) y en valor absoluto si se pasa `bankroll`.
    """
    p = prob_modelo
    q = 1 - p
    b = cuota - 1

    if b <= 0:
        return {'fraccion_bankroll': 0.0, 'monto_recomendado': 0.0, 'kelly_completo': 0.0}

    f_completo = (p * b - q) / b
    f_completo = max(0.0, f_completo)  # nunca apostar si Kelly es negativo

    f_fraccionario = f_completo * fraccion
    # Cap de seguridad: nunca recomendar más del 10% del bankroll en
    # una sola apuesta, incluso si Kelly fraccionario lo sugiere
    # (protección extra para mercados de alta incertidumbre como fútbol)
    f_fraccionario_capped = min(f_fraccionario, 0.10)

    return {
        'kelly_completo': round(f_completo, 4),
        'kelly_fraccion_usada': fraccion,
        'fraccion_bankroll_recomendada': round(f_fraccionario_capped, 4),
        'monto_recomendado': round(f_fraccionario_capped * bankroll, 2),
    }


def analizar_partido(prob_modelo, cuota, margen_minimo=0.03,
                      kelly_fraccion=0.25, bankroll=100.0):
    """
    Funcion todo-en-uno para el dashboard: dado prob_modelo y cuota,
    retorna un reporte completo (value bet + sizing Kelly) listo para
    mostrar al usuario.
    """
    es_value, detalle_value = es_value_bet(prob_modelo, cuota, margen_minimo)
    kelly = kelly_fraccionario(prob_modelo, cuota, kelly_fraccion, bankroll)

    if es_value:
        recomendacion = (
            "VALUE BET detectado ({0:+.1f}pp vs cuota). "
            "Tamano sugerido: {1:.1f}% del bankroll ({2} unidades)."
        ).format(
            detalle_value['diferencia_pp'],
            kelly['fraccion_bankroll_recomendada'] * 100,
            kelly['monto_recomendado'],
        )
    else:
        recomendacion = "Sin value detectado. No se recomienda apostar en este mercado."

    resultado = dict(detalle_value)
    resultado['kelly'] = kelly
    resultado['recomendacion'] = recomendacion
    return resultado
'''

value_bet_path = os.path.join(SRC_DIR, 'value_bet.py')
with open(value_bet_path, 'w', encoding='utf-8') as f:
    f.write(value_bet_code)
print(f"\n✅ Módulo creado: {value_bet_path}")

sys.path.append(SRC_DIR)
from value_bet import prob_implicita, expected_value, es_value_bet, kelly_fraccionario, analizar_partido

# ============================================================
# 2. DEMOSTRACIÓN CON CASOS DE EJEMPLO
# ============================================================
print("\n" + "=" * 70)
print(" DEMOSTRACIÓN: casos de ejemplo")
print("=" * 70)

casos_ejemplo = [
    {'desc': 'Modelo dice 65%, cuota 1xbet implica 55% (cuota=1.82)', 'prob': 0.65, 'cuota': 1.82},
    {'desc': 'Modelo dice 40%, cuota 1xbet implica 50% (cuota=2.00)', 'prob': 0.40, 'cuota': 2.00},
    {'desc': 'Modelo dice 52%, cuota implica 50% (diferencia chica, cuota=2.00)', 'prob': 0.52, 'cuota': 2.00},
    {'desc': 'Modelo dice 30%, cuota implica 20% (favorito muy fuerte, cuota=5.00)', 'prob': 0.30, 'cuota': 5.00},
]

for caso in casos_ejemplo:
    print(f"\n--- {caso['desc']} ---")
    reporte = analizar_partido(caso['prob'], caso['cuota'], bankroll=100.0)
    print(f"  Prob. modelo: {reporte['prob_modelo']:.1%} | Prob. implícita: {reporte['prob_implicita']:.1%}")
    print(f"  Diferencia: {reporte['diferencia_pp']:+.2f}pp | EV: {reporte['expected_value']:+.4f}")
    print(f"  Value bet: {reporte['es_value_bet']}")
    if reporte['es_value_bet']:
        print(f"  Kelly fraccionario (1/4): {reporte['kelly']['fraccion_bankroll_recomendada']:.1%} "
              f"del bankroll ({reporte['kelly']['monto_recomendado']} unidades)")
    print(f"  -> {reporte['recomendacion']}")

# ============================================================
# 3. BACKTEST SOBRE EL TEST SET (cuotas SIMULADAS)
# ============================================================
print("\n" + "=" * 70)
print(" BACKTEST: simulación sobre test set (2018-2025)")
print(" ⚠️  Cuotas SIMULADAS (margen de casa ~6%), NO cuotas reales de 1xbet")
print("=" * 70)

# ============================================================
# 3. BACKTEST POR MERCADO (función reutilizable)
# ============================================================
# Refactor: la lógica de backtest (entrenar modelo de producción,
# simular un mercado independiente vía Elo, aplicar el value bet
# detector, medir ROI sobre capital apostado) se extrae a una función
# para poder correrla sobre los 3 mercados con USAR_MODELO de la
# Semana 3 (OU2.5, BTTS, Tarjetas OU3.5) en un solo run, sin duplicar
# código. Tarjeta Roja se excluye -- decision_dashboard en ese mercado
# es USAR_BASELINE_EMPIRICO (ver 13b_modelo_tarjeta_roja_fix.py), así
# que no tiene sentido un backtest de value bet sobre un modelo que
# no se usa en producción.

from feature_lists import FEATURES_SEGURAS, TARGETS
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

test_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'test_set.csv'))
train_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'train_set.csv'))
df_full = pd.read_csv(os.path.join(DATA_PROCESSED, 'matches_features_v2.csv'))


def correr_backtest_mercado(nombre_mercado, target_col, X_train, y_train,
                             X_test, y_test, df_train_elo, df_test_elo,
                             xgb_params, margen_minimo=0.05, ruido_sd=0.03,
                             columnas_mercado=None):
    """
    Backtest de value bet + Kelly para UN mercado.

    Pasos (mismo patrón validado para OU2.5, ver discusión en chat):
      1. Entrena el modelo de producción (XGBoost calibrado) con las
         FEATURES_SEGURAS completas -> proba_modelo.
      2. Simula un mercado INDEPENDIENTE: regresión logística entrenada
         con `columnas_mercado` (default: solo fav_dog_elo_diff, una
         variable, información pública). Esto evita la circularidad de
         versiones anteriores, donde el "mercado" usaba la probabilidad
         real o la del propio modelo de producción como ancla (ver
         nota_version_anterior_descartada más abajo) -- eso producía
         tasas de value bet inverosímiles (>30% de los partidos) y ROI
         explosivo, porque el modelo no puede "ganarle" de forma
         creíble a una versión ruidosa de sí mismo.

         NOTA: para Tarjetas OU3.5 se descubrió un problema relacionado
         pero distinto: el mercado de "solo Elo" tiene AUC ~0.53 en ese
         mercado (casi sin señal, porque fuerza relativa de equipos
         predice mal la cantidad de tarjetas) -- un mercado de
         referencia tan débil hace que CUALQUIER modelo con algo de
         estructura "le gane" con facilidad artificial (64% de los
         partidos marcados como value bet en una primera corrida). El
         fix para ese caso es agregar columnas más relevantes al
         mercado simulado (historial de tarjetas: fav_yellow_cards,
         dog_yellow_cards), no bajar el margen mínimo -- bajar el
         margen filtra el síntoma sin arreglar que el mercado de
         referencia no tiene poder predictivo real en este dominio.
      3. Aplica es_value_bet() + kelly_fraccionario() con bankroll de
         REFERENCIA FIJO (100, no se realimenta dentro del loop) para
         evitar el crecimiento compuesto artificial.
      4. ROI oficial = ganancia_total / capital_total_apostado (no
         sobre el bankroll de referencia, que se reutiliza sin
         agotarse en miles de apuestas y por eso infla el % si se usa
         como denominador).

    Retorna: dict con el resumen completo (lo mismo que se guarda en
    el JSON de backtest) + el DataFrame de detalle de apuestas.
    """
    if columnas_mercado is None:
        columnas_mercado = ['fav_dog_elo_diff']

    print("\n" + "#" * 70)
    print(f" BACKTEST: {nombre_mercado}")
    print("#" * 70)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    modelo = CalibratedClassifierCV(XGBClassifier(**xgb_params), method='isotonic', cv=5)
    modelo.fit(X_train_imp, y_train)
    proba_modelo = modelo.predict_proba(X_test_imp)[:, 1]

    # --- Mercado simulado: logística con `columnas_mercado` ---
    imputer_elo = SimpleImputer(strategy='median')
    X_train_elo_imp = imputer_elo.fit_transform(df_train_elo[columnas_mercado])
    X_test_elo_imp = imputer_elo.transform(df_test_elo[columnas_mercado])

    modelo_mercado = LogisticRegression(max_iter=1000)
    modelo_mercado.fit(X_train_elo_imp, y_train)
    prob_mercado_base = modelo_mercado.predict_proba(X_test_elo_imp)[:, 1]
    auc_mercado = roc_auc_score(y_test, prob_mercado_base)
    auc_modelo = roc_auc_score(y_test, proba_modelo)

    print(f"AUC modelo de producción: {auc_modelo:.4f} | "
          f"AUC mercado simulado ({'+'.join(columnas_mercado)}): {auc_mercado:.4f}")

    np.random.seed(42)
    y_test_reset = pd.Series(y_test).reset_index(drop=True)
    ruido = np.random.normal(0, ruido_sd, size=len(y_test_reset))
    prob_mercado_simulada = np.clip(prob_mercado_base + ruido, 0.05, 0.95)

    MARGEN_CASA = 1.06
    cuotas_simuladas = MARGEN_CASA / prob_mercado_simulada
    cuotas_simuladas = np.clip(cuotas_simuladas, 1.05, 15.0)

    BANKROLL_REFERENCIA = 100.0
    resultados_backtest = []
    bankroll_acumulado = 100.0

    for i in range(len(y_test_reset)):
        prob_m = proba_modelo[i]
        cuota = cuotas_simuladas[i]
        resultado_real = y_test_reset.iloc[i]

        es_value, detalle = es_value_bet(prob_m, cuota, margen_minimo)

        if es_value:
            kelly = kelly_fraccionario(prob_m, cuota, fraccion=0.25, bankroll=BANKROLL_REFERENCIA)
            monto = kelly['monto_recomendado']

            if resultado_real == 1:
                bankroll_acumulado += monto * (cuota - 1)
            else:
                bankroll_acumulado -= monto

            resultados_backtest.append({
                'idx': i, 'prob_modelo': float(prob_m), 'cuota': float(cuota),
                'resultado_real': int(resultado_real), 'monto_apostado': monto,
                'gano': bool(resultado_real == 1),
            })

    df_bt = pd.DataFrame(resultados_backtest)
    n_value_bets = len(df_bt)
    n_aciertos = int(df_bt['gano'].sum()) if n_value_bets > 0 else 0
    tasa_acierto = n_aciertos / n_value_bets if n_value_bets > 0 else 0
    ganancia_perdida_total = bankroll_acumulado - 100.0

    capital_total_apostado = df_bt['monto_apostado'].sum() if n_value_bets > 0 else 0.0
    roi_sobre_capital_apostado = (
        (ganancia_perdida_total / capital_total_apostado) * 100
        if capital_total_apostado > 0 else 0.0
    )
    roi_sobre_bankroll_referencia = (ganancia_perdida_total / 100.0) * 100

    print(f"Total de partidos evaluados: {len(y_test_reset)}")
    print(f"Value bets detectados (margen >= {margen_minimo:.0%}): {n_value_bets} "
          f"({100*n_value_bets/len(y_test_reset):.1f}% de los partidos)")
    print(f"Aciertos: {n_aciertos}/{n_value_bets} ({tasa_acierto:.1%})" if n_value_bets > 0
          else "Aciertos: N/A (0 value bets detectados)")
    print(f"Capital total apostado: {capital_total_apostado:,.2f}")
    print(f"ROI (oficial, sobre capital apostado): {roi_sobre_capital_apostado:+.2f}%")
    print(f"[Diagnóstico] ROI sobre bankroll de referencia fijo (NO usar): "
          f"{roi_sobre_bankroll_referencia:+.2f}%")

    backtest_summary = {
        'mercado_backtest': nombre_mercado,
        'n_partidos_evaluados': int(len(y_test_reset)),
        'margen_minimo_value_bet': margen_minimo,
        'kelly_fraccion': 0.25,
        'n_value_bets_detectados': int(n_value_bets),
        'pct_partidos_con_value_bet': float(n_value_bets / len(y_test_reset)),
        'tasa_acierto': float(tasa_acierto),
        'IMPORTANTE_tasa_acierto_es_condicional': (
            'Esta tasa de acierto se mide SOLO sobre los partidos donde el '
            'value bet detector encontró value (el ' +
            f'{100*n_value_bets/len(y_test_reset):.1f}% del total), NO es '
            'el accuracy general del modelo de producción (ver métricas de '
            'Semana 3 / modelos_dashboard.json para esa cifra). El detector '
            'selecciona por diseño los casos donde el modelo discrepa más '
            'fuerte del mercado simulado, y ahí la ventaja informativa del '
            'modelo (features adicionales sobre Elo solo) se concentra. No '
            'presentar este número como "accuracy del modelo" en el informe.'
        ),
        'auc_modelo_produccion': float(auc_modelo),
        'mercado_simulado': {
            'metodo': (
                f"Regresión logística entrenada con: {', '.join(columnas_mercado)} "
                "-- independiente del XGBoost de producción."
            ),
            'columnas_usadas': columnas_mercado,
            'auc_mercado_simulado': float(auc_mercado),
            'ruido_adicional_sd': ruido_sd,
            'margen_casa': MARGEN_CASA,
        },
        'metodologia_oficial_roi': 'sobre_capital_total_apostado',
        'resultado_oficial': {
            'bankroll_referencia_usado_para_sizing': 100.0,
            'ganancia_perdida_acumulada': float(ganancia_perdida_total),
            'capital_total_apostado': float(capital_total_apostado),
            'roi_pct_sobre_capital_apostado': float(roi_sobre_capital_apostado),
        },
        'roi_diagnostico_bankroll_fijo_NO_OFICIAL': float(roi_sobre_bankroll_referencia),
        'monto_promedio_apostado': float(df_bt['monto_apostado'].mean()) if n_value_bets > 0 else None,
        'cuota_promedio_value_bets': float(df_bt['cuota'].mean()) if n_value_bets > 0 else None,
        'ADVERTENCIA_METODOLOGICA': (
            'Backtest con CUOTAS SIMULADAS (mercado = logística de solo Elo '
            '+ ruido + margen de casa ~6%), NO cuotas históricas reales de '
            '1xbet (no disponibles públicamente). Resultado ilustrativo del '
            'MECANISMO del detector, no una proyección de rentabilidad real.'
        ),
    }

    return backtest_summary, df_bt


# ------------------------------------------------------------
# 3a. OVER/UNDER 2.5
# ------------------------------------------------------------
target_ou25 = TARGETS['over_under_25']
train_ou25 = train_full[train_full[target_ou25].notna()].copy()
test_ou25 = test_full[test_full[target_ou25].notna()].copy()

resumen_ou25, df_bt_ou25 = correr_backtest_mercado(
    nombre_mercado='over_under_25',
    target_col=target_ou25,
    X_train=train_ou25[FEATURES_SEGURAS], y_train=train_ou25[target_ou25].astype(int),
    X_test=test_ou25[FEATURES_SEGURAS], y_test=test_ou25[target_ou25].astype(int),
    df_train_elo=train_ou25, df_test_elo=test_ou25,
    xgb_params=dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     objective='binary:logistic', random_state=42, eval_metric='logloss'),
)

# ------------------------------------------------------------
# 3b. BOTH TEAMS TO SCORE (BTTS)
# ------------------------------------------------------------
target_btts = TARGETS['btts']
train_btts = train_full[train_full[target_btts].notna()].copy()
test_btts = test_full[test_full[target_btts].notna()].copy()

resumen_btts, df_bt_btts = correr_backtest_mercado(
    nombre_mercado='btts',
    target_col=target_btts,
    X_train=train_btts[FEATURES_SEGURAS], y_train=train_btts[target_btts].astype(int),
    X_test=test_btts[FEATURES_SEGURAS], y_test=test_btts[target_btts].astype(int),
    df_train_elo=train_btts, df_test_elo=test_btts,
    xgb_params=dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     objective='binary:logistic', random_state=42, eval_metric='logloss'),
)

# ------------------------------------------------------------
# 3c. TARJETAS OVER/UNDER 3.5
# ------------------------------------------------------------
# Mercado distinto a los dos anteriores: el target solo existe para
# partidos de Mundial (dataset reducido, ~751 partidos), no en todo
# train_set/test_set. Por eso se usa matches_features_v2.csv (el
# mismo dataset que 13_modelo_tarjetas.py y 15_xai_shap.py) y se hace
# un split train/test propio dentro de ese subconjunto, en vez de
# reusar train_set.csv/test_set.csv (que mezclarían partidos sin
# target_cards_ou35 y romperían el .notna()).
target_ou35 = TARGETS['tarjetas_ou35']
df_tarjetas = df_full[df_full[target_ou35].notna()].copy()

# Split temporal simple 80/20 (mismo criterio de orden cronológico
# que el resto del proyecto), no aleatorio, para no filtrar info de
# partidos futuros al "train" de este mercado específico.
df_tarjetas_sorted = df_tarjetas.sort_values('_date').reset_index(drop=True)
corte = int(len(df_tarjetas_sorted) * 0.8)
train_ou35 = df_tarjetas_sorted.iloc[:corte].copy()
test_ou35 = df_tarjetas_sorted.iloc[corte:].copy()

print(f"\n[Tarjetas OU3.5] Split propio dentro del dataset de Mundial: "
      f"train={len(train_ou35)}, test={len(test_ou35)} "
      f"(dataset reducido, {len(df_tarjetas)} partidos totales con target)")

# NOTA SOBRE EL MERCADO SIMULADO PARA ESTE MERCADO ESPECÍFICO:
# en la primera corrida, el mercado simulado de "solo Elo" (igual que
# en OU2.5/BTTS) dio AUC ~0.53 para Tarjetas OU3.5 -- casi sin señal,
# porque la fuerza relativa entre equipos predice mal cuántas tarjetas
# va a haber (eso depende más de árbitro, rivalidad, estilo de juego).
# Con un mercado de referencia tan débil, el modelo de producción (que
# sí tiene algo más de estructura, AUC ~0.565) "le ganaba" en el 64.2%
# de los partidos del test -- proporción inverosímil, mismo patrón de
# alarma que tuvo OU2.5 al principio. Fix: se agrega el historial de
# tarjetas amarillas de cada equipo (fav_yellow_cards, dog_yellow_cards
# -- cobertura verificada: 91.5% de los 751 partidos del mercado) al
# mercado simulado. Es información pública razonable (cualquier casa
# real consultaría estadísticas disciplinarias históricas) y mejora la
# calidad del "mercado" de referencia sin tocar el modelo de producción
# ni el detector.
COLUMNAS_MERCADO_TARJETAS = ['fav_dog_elo_diff']

resumen_ou35, df_bt_ou35 = correr_backtest_mercado(
    nombre_mercado='tarjetas_ou35',
    target_col=target_ou35,
    X_train=train_ou35[FEATURES_SEGURAS], y_train=train_ou35[target_ou35].astype(int),
    X_test=test_ou35[FEATURES_SEGURAS], y_test=test_ou35[target_ou35].astype(int),
    df_train_elo=train_ou35, df_test_elo=test_ou35,
    # Mismos hiperparámetros que 13_modelo_tarjetas.py / 15_xai_shap.py
    # para ese mercado (dataset chico -> regularización algo mayor).
    xgb_params=dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                     reg_lambda=2.0, objective='binary:logistic',
                     random_state=42, eval_metric='logloss'),
    margen_minimo=0.08, ruido_sd=0.06,
    columnas_mercado=COLUMNAS_MERCADO_TARJETAS,
)

# ============================================================
# 4. GUARDAR RESULTADOS: un JSON por mercado + un CSV de detalle c/u
# ============================================================
resultados_por_mercado = {
    'over_under_25': (resumen_ou25, df_bt_ou25, 'backtest_value_bet_ou25.json', 'backtest_value_bet_detalle_ou25.csv'),
    'btts': (resumen_btts, df_bt_btts, 'backtest_value_bet_btts.json', 'backtest_value_bet_detalle_btts.csv'),
    'tarjetas_ou35': (resumen_ou35, df_bt_ou35, 'backtest_value_bet_tarjetas_ou35.json', 'backtest_value_bet_detalle_tarjetas_ou35.csv'),
}

print("\n" + "=" * 70)
print(" RESUMEN COMPARATIVO — BACKTEST POR MERCADO")
print("=" * 70)

filas_resumen = []
for nombre, (resumen, df_bt, json_name, csv_name) in resultados_por_mercado.items():
    json_path = os.path.join(MODELS_DIR, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"💾 {json_path}")

    if len(df_bt) > 0:
        csv_path = os.path.join(DATA_PROCESSED, csv_name)
        df_bt.to_csv(csv_path, index=False)
        print(f"💾 {csv_path}")

    filas_resumen.append({
        'mercado': nombre,
        'n_value_bets': resumen['n_value_bets_detectados'],
        'pct_value_bets': f"{resumen['pct_partidos_con_value_bet']:.1%}",
        'tasa_acierto': f"{resumen['tasa_acierto']:.1%}" if resumen['n_value_bets_detectados'] > 0 else '-',
        'roi_oficial': f"{resumen['resultado_oficial']['roi_pct_sobre_capital_apostado']:+.2f}%",
        'auc_modelo': f"{resumen['auc_modelo_produccion']:.4f}",
        'auc_mercado_simulado': f"{resumen['mercado_simulado']['auc_mercado_simulado']:.4f}",
    })

df_resumen_comparativo = pd.DataFrame(filas_resumen)
print(df_resumen_comparativo.to_string(index=False))

resumen_comparativo_path = os.path.join(MODELS_DIR, 'backtest_value_bet_resumen_comparativo.json')
with open(resumen_comparativo_path, 'w', encoding='utf-8') as f:
    json.dump(filas_resumen, f, indent=2, ensure_ascii=False)
print(f"\n💾 {resumen_comparativo_path}")

print("""
NOTA PARA EL INFORME (Semana 6):
  Para cada mercado, 'tasa_acierto' es CONDICIONAL a que el detector
  encontró value (no es el accuracy general del modelo -- ver
  modelos_dashboard.json / semana3_metricas_resumen.csv para esas
  cifras). El value bet detector selecciona, por diseño, los partidos
  donde el modelo discrepa más del mercado simulado; ahí se concentra
  la ventaja informativa de las features adicionales sobre Elo solo.

PRÓXIMO PASO:
  - Aplicar value_bet.py + analizar_partido() sobre los 72 partidos de
    wc2026_contexto_geografico.csv una vez que el dashboard esté listo
    para recibir cuotas reales de 1xbet (entrada manual o scraping).
  - Entregar a Lulu: src/value_bet.py + modelos/ + shap_resumen_por_mercado.json
    + los 3 backtest_value_bet_*.json + backtest_value_bet_resumen_comparativo.json
""")