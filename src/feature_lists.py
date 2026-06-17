"""
feature_lists.py
======================================================================
Listas de columnas reutilizables para los notebooks de modelado
(Semana 3). Generado por src/10_preparar_datos_modelado.py.

NO editar a mano salvo que sea para corregir un error -- si se agregan
features nuevas al dataset, regenerar este archivo corriendo de nuevo
10_preparar_datos_modelado.py.
======================================================================
"""

FECHA_CORTE = "2018-01-01"

FEATURES_SEGURAS = ['fav_dog_elo_diff', 'fav_elo', 'dog_elo', 'fav_avg_overall', 'dog_avg_overall', 'fav_max_overall', 'dog_max_overall', 'fav_avg_attack', 'dog_avg_attack', 'fav_avg_defense', 'dog_avg_defense', 'fav_avg_pace', 'dog_avg_pace', 'fav_avg_shooting', 'dog_avg_shooting', 'fav_avg_passing', 'dog_avg_passing', 'fav_form_scored', 'dog_form_scored', 'fav_form_conceded', 'dog_form_conceded', 'fav_form_win_rate', 'dog_form_win_rate', 'tournament_weight', 'is_world_cup', 'is_world_cup_qualifier', 'is_continental', 'is_neutral', 'mismo_confed', 'fav_dias_descanso', 'dog_dias_descanso', 'descanso_diff']

COLUMNAS_PROHIBIDAS = ['home_goals', 'away_goals', 'fav_goals', 'dog_goals', 'total_goals', 'result', 'target_1x2_fav_dog', 'target_ou25', 'target_btts', 'target_ou15', 'target_ou35_goals', 'target_cards_ou35', 'target_redcard', 'home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards', 'fav_yellow_cards', 'dog_yellow_cards', 'fav_red_cards', 'dog_red_cards', 'total_cards']

TARGETS = {
    "1x2": "target_1x2_fav_dog",
    "over_under_25": "target_ou25",
    "btts": "target_btts",
    "over_under_15": "target_ou15",
    "over_under_35_goles": "target_ou35_goals",
    "tarjetas_ou35": "target_cards_ou35",   # solo ~751 partidos de Mundial
    "tarjeta_roja": "target_redcard",        # solo ~751 partidos de Mundial
}
