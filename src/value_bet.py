"""
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
