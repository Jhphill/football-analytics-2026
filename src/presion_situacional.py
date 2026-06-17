"""
presion_situacional.py
======================================================================
Calcula la presión situacional de cada selección durante el Mundial.

Se llama PARTIDO A PARTIDO, antes de cada jornada, usando la tabla
de grupos actualizada. No tiene sentido precomputarla para el
historial (depende del contexto vivo del torneo).

Cómo usar en el dashboard:
    from presion_situacional import calcular_presion, ajustar_prob_por_presion

    estado = calcular_presion(
        puntos=3, partidos_jugados=2, partidos_restantes=1,
        gf=4, gc=2
    )
    prob_ajustada = ajustar_prob_por_presion(prob_base, estado_fav, estado_dog)
======================================================================
"""

def calcular_presion(puntos, partidos_jugados, partidos_restantes,
                     gf=None, gc=None, posicion_grupo=None):
    """
    Calcula el estado de presión situacional de una selección.

    Parámetros:
        puntos              : puntos acumulados en la fase de grupos
        partidos_jugados    : partidos ya jugados en la fase de grupos
        partidos_restantes  : partidos que faltan en la fase de grupos
        gf                  : goles a favor acumulados (opcional)
        gc                  : goles en contra acumulados (opcional)
        posicion_grupo      : posición actual en el grupo (1, 2, 3, 4) (opcional)

    Retorna (str):
        "calificado_comodo"     -> ya clasificó matemáticamente,
                                   probable rotación de titulares
        "calificado_ajustado"   -> probablemente clasificado pero no
                                   matemáticamente seguro
        "necesita_ganar"        -> solo una victoria asegura avance
        "necesita_ganar_y_milagro" -> necesita ganar Y que otros
                                      resultados le favorezcan
        "eliminado"             -> sin posibilidad matemática de avanzar
        "normal"                -> situación abierta, sin presión extrema
    """
    max_puntos_posibles = puntos + (partidos_restantes * 3)

    # Reglas por puntos (formato Mundial 2026: top 2 de 4 equipos clasifican,
    # más los 8 mejores terceros de 12 grupos)
    if puntos >= 6 and partidos_restantes <= 1:
        return "calificado_comodo"
    if puntos >= 4 and partidos_restantes == 0:
        return "calificado_comodo"
    if puntos >= 6 and partidos_restantes == 2:
        return "calificado_ajustado"
    if max_puntos_posibles < 3:
        return "eliminado"
    if max_puntos_posibles <= 3 and partidos_restantes == 1:
        return "necesita_ganar_y_milagro"
    if puntos == 0 and partidos_restantes == 1:
        return "necesita_ganar"
    if puntos <= 1 and partidos_restantes == 1:
        return "necesita_ganar"
    return "normal"


# Ajuste de probabilidad base por presión situacional
# Coeficientes empíricos (a calibrar con backtesting de Mundiales pasados)
AJUSTE_PRESION = {
    "calificado_comodo": -0.04,     # más propenso a rotar, baja rendimiento
    "calificado_ajustado": 0.00,
    "normal": 0.00,
    "necesita_ganar": +0.04,        # más ofensivo y motivado
    "necesita_ganar_y_milagro": +0.02,  # necesita ganar pero sabe que puede no alcanzar
    "eliminado": -0.03,             # relajado, puede jugar sin presión (efecto mixto)
}


def ajustar_prob_por_presion(prob_base_fav_win, estado_fav, estado_dog):
    """
    Ajusta la probabilidad de victoria del favorito según la presión
    situacional de ambos equipos.

    Parámetros:
        prob_base_fav_win : float, probabilidad base de victoria del favorito
        estado_fav        : str, resultado de calcular_presion() para el favorito
        estado_dog        : str, resultado de calcular_presion() para el no-favorito

    Retorna:
        float: probabilidad ajustada (entre 0 y 1)
    """
    adj_fav = AJUSTE_PRESION.get(estado_fav, 0.0)
    adj_dog = AJUSTE_PRESION.get(estado_dog, 0.0)

    # Si el dog está bajo presión positiva (+), el fav sube menos (o baja)
    ajuste_neto = adj_fav - adj_dog
    prob_ajustada = prob_base_fav_win + ajuste_neto
    return max(0.01, min(0.99, prob_ajustada))


if __name__ == "__main__":
    # Validación con ejemplos reales de Mundiales pasados
    print("Validación de calcular_presion():")
    casos = [
        # (descripción, puntos, p_jugados, p_restantes, esperado)
        ("Brasil 2022 tras ganar 2 de 2", 6, 2, 1, "calificado_comodo"),
        ("Arabia Saudita 2022 tras perder 2", 0, 2, 1, "necesita_ganar"),
        ("Alemania 2022 con 1 punto en 2 partidos", 1, 2, 1, "necesita_ganar"),
        ("Argentina 2022 tras ganar 2 de 2", 6, 2, 1, "calificado_comodo"),
        ("Situación normal tras 1 partido", 3, 1, 2, "normal"),
        ("Ya eliminado matemáticamente", 0, 2, 1, "necesita_ganar"),
    ]
    for desc, pts, pj, pr, esperado in casos:
        resultado = calcular_presion(pts, pj, pr)
        status = "✅" if resultado == esperado else "⚠️ "
        print(f"  {status} {desc}: {resultado} (esperado: {esperado})")
