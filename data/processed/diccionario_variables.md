# Diccionario de variables — `matches_features_v2.csv`

**Filas:** 43,816 | **Columnas:** 95

**Rango de fechas:** 1872-11-30 → 2026-06-27

**Generado automáticamente por:** `src/04_diccionario_variables.py`

**Fuente:** `data/processed/matches_features_v2.csv` (output final Semana 2)


## Resumen ejecutivo

- Total de partidos: **43,816**
- Partidos de Copa Mundial (fase final): **1,036**
- Partidos de clasificatoria mundialista: **7,325**
- Partidos en sede neutral: **10,922**
- Partidos con atributos FIFA pendientes: **481** (forma reciente ya calculada)
- Distribución `target_1x2_fav_dog`: {'fav_win': 23959, 'draw': 10291, 'dog_win': 9440}

## Identificación del partido

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `_date` | str | 0.0% | — | Fecha del partido (YYYY-MM-DD). |
| `_home_team` | str | 0.0% | — | Selección "local" (o primera mencionada si es sede neutral). Nombre normalizado. |
| `_away_team` | str | 0.0% | — | Selección "visitante" (o segunda mencionada si es sede neutral). Nombre normalizado. |
| `_tournament` | str | 0.0% | — | Nombre del torneo o competición (texto original de la fuente). |
| `key` | str | 0.0% | — | Clave compuesta única `_date_home_away`, usada internamente para deduplicar y hacer merges. |

## Resultado y targets base

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `home_goals` | float64 | 0.2% | min=0.00, max=21.00, media=1.84 | Goles anotados por la selección "local" al término del partido. |
| `away_goals` | float64 | 0.2% | min=0.00, max=15.00, media=1.00 | Goles anotados por la selección "visitante" al término del partido. |
| `total_goals` | float64 | 0.2% | min=0.00, max=21.00, media=2.84 | Suma de home_goals + away_goals. Insumo directo para targets Over/Under. |
| `result` | str | 0.0% | valores: ['draw', 'home_win', 'away_win'] | Resultado categórico: home_win / draw / away_win. Target base del mercado 1X2. |

## Contexto del torneo

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `is_neutral` | int64 | 0.0% | valores: [0, 1] | Indicador (0/1): 1 si el partido se jugó en sede neutral (sin ventaja de localía real). |
| `is_world_cup` | int64 | 0.0% | valores: [0, 1] | Indicador (0/1): 1 si _tournament es exactamente "FIFA World Cup" (fase final). |
| `is_world_cup_qualifier` | int64 | 0.0% | valores: [0, 1] | Indicador (0/1): 1 si _tournament es una clasificatoria mundialista ("FIFA World Cup qualification"). |
| `is_continental` | int64 | 0.0% | valores: [0, 1] | Indicador (0/1): 1 si el torneo es una copa continental (Euro, Copa América, AFCON, etc.). |
| `pending_feature_engineering` | int64 | 0.0% | valores: [0, 1] | Indicador (0/1): 1 en los 481 partidos de Mundial recuperados de results.csv. La forma reciente ya fue calculada para 477/481; los 4 restantes son debuts sin historia previa. |

## Fortaleza relativa (Elo)

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `home_elo` | float64 | 0.2% | min=952.19, max=2171.00, media=1600.32 | Rating Elo de la selección local en la fecha del partido (último valor conocido, vía merge_asof contra eloratings.csv). |
| `away_elo` | float64 | 0.1% | min=940.49, max=2171.00, media=1580.58 | Rating Elo de la selección visitante en la fecha del partido. |
| `elo_diff` | float64 | 0.3% | min=-997.22, max=972.10, media=19.73 | Diferencia de Elo: home_elo - away_elo. Variable clave de fortaleza relativa. |

## Atributos FIFA de la plantilla

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `home_avg_overall` | float64 | 1.1% | min=51.00, max=87.29, media=71.08 | Promedio del atributo "overall" (FIFA videojuego) de la plantilla local. |
| `home_max_overall` | float64 | 1.1% | min=51.00, max=94.00, media=76.49 | Máximo "overall" de la plantilla local (proxy de "jugador estrella"). |
| `home_avg_attack` | float64 | 1.1% | min=49.00, max=88.67, media=71.27 | Promedio del atributo de ataque de la plantilla local. |
| `home_avg_defense` | float64 | 1.1% | min=50.00, max=87.75, media=71.01 | Promedio del atributo de defensa de la plantilla local. |
| `home_avg_pace` | float64 | 2.7% | min=43.00, max=91.00, media=72.27 | Promedio del atributo de ritmo/velocidad de la plantilla local. |
| `home_avg_shooting` | float64 | 2.7% | min=21.00, max=80.00, media=59.69 | Promedio del atributo de definición/tiro de la plantilla local. |
| `home_avg_passing` | float64 | 2.7% | min=29.00, max=81.85, media=62.39 | Promedio del atributo de pase de la plantilla local. |
| `away_avg_overall` | float64 | 1.1% | min=51.00, max=87.29, media=70.49 | Promedio del atributo "overall" de la plantilla visitante. |
| `away_max_overall` | float64 | 1.1% | min=51.00, max=94.00, media=75.89 | Máximo "overall" de la plantilla visitante. |
| `away_avg_attack` | float64 | 1.1% | min=49.00, max=88.67, media=70.66 | Promedio del atributo de ataque de la plantilla visitante. |
| `away_avg_defense` | float64 | 1.1% | min=49.00, max=87.75, media=70.46 | Promedio del atributo de defensa de la plantilla visitante. |
| `away_avg_pace` | float64 | 2.9% | min=43.00, max=91.00, media=72.12 | Promedio del atributo de ritmo/velocidad de la plantilla visitante. |
| `away_avg_shooting` | float64 | 2.9% | min=21.00, max=80.00, media=59.31 | Promedio del atributo de definición/tiro de la plantilla visitante. |
| `away_avg_passing` | float64 | 2.9% | min=29.00, max=81.85, media=61.71 | Promedio del atributo de pase de la plantilla visitante. |
| `overall_diff` | float64 | 1.1% | min=-32.86, max=32.36, media=0.59 | Diferencia home_avg_overall - away_avg_overall. |
| `attack_diff` | float64 | 1.1% | min=-35.29, max=35.29, media=0.61 | Diferencia home_avg_attack - away_avg_attack. |
| `defense_diff` | float64 | 1.1% | min=-33.11, max=32.50, media=0.54 | Diferencia home_avg_defense - away_avg_defense. |

## Forma reciente (rolling)

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `home_form_scored` | float64 | 0.2% | min=0.00, max=11.00, media=1.45 | Goles a favor promedio (rolling N=10, shift(1) sin leakage) de la selección local. Calculado uniformemente en 07_forma_reciente.py. |
| `home_form_conceded` | float64 | 0.2% | min=0.00, max=11.00, media=1.39 | Goles en contra promedio (rolling N=10) de la selección local. |
| `home_form_win_rate` | float64 | 0.2% | min=0.00, max=1.00, media=0.39 | Tasa de victorias (rolling N=10) de la selección local. |
| `away_form_scored` | float64 | 0.2% | min=0.00, max=13.00, media=1.40 | Goles a favor promedio (rolling N=10) de la selección visitante. |
| `away_form_conceded` | float64 | 0.2% | min=0.00, max=16.00, media=1.47 | Goles en contra promedio (rolling N=10) de la selección visitante. |
| `away_form_win_rate` | float64 | 0.2% | min=0.00, max=1.00, media=0.37 | Tasa de victorias (rolling N=10) de la selección visitante. |

## Otras (sin clasificar)

| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |
|---|---|---|---|---|
| `home_yellow_cards` | float64 | 98.3% | valores: [1.0, 0.0, 2.0, 3.0, 6.0, 5.0, 4.0, 7.0, 8.0] | Tarjetas amarillas de la selección local. Solo partidos is_world_cup=1 desde 1970. Fuente: Fjelstul WC DB. |
| `home_red_cards` | float64 | 98.3% | valores: [0.0, 1.0] | Tarjetas rojas de la selección local. Misma cobertura. |
| `away_yellow_cards` | float64 | 98.3% | valores: [4.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 6.0, 8.0] | Tarjetas amarillas de la selección visitante. |
| `away_red_cards` | float64 | 98.3% | valores: [0.0, 1.0, 2.0] | Tarjetas rojas de la selección visitante. |
| `total_cards` | float64 | 98.3% | min=0.00, max=15.00, media=3.46 | Suma total de tarjetas (amarillas + rojas) del partido. |
| `target_cards_ou35` | float64 | 98.3% | valores: [1.0, 0.0] | Target Over/Under 3.5 tarjetas totales: 1=over, 0=under. 751 partidos de Mundial (1970-2022). |
| `target_redcard` | float64 | 98.3% | valores: [0.0, 1.0] | Target tarjeta roja: 1=al menos una roja, 0=sin rojas. 751 partidos de Mundial. |
| `fav_elo` | float64 | 0.3% | min=1029.81, max=2171.00, media=1661.49 | Rating Elo del favorito antes del partido. |
| `dog_elo` | float64 | 0.3% | min=940.49, max=2034.33, media=1518.76 | Rating Elo del no-favorito antes del partido. |
| `fav_goals` | float64 | 0.4% | min=0.00, max=21.00, media=1.88 | Goles del favorito en el partido. |
| `dog_goals` | float64 | 0.4% | min=0.00, max=16.00, media=0.96 | Goles del no-favorito en el partido. |
| `fav_avg_overall` | float64 | 1.1% | min=51.00, max=87.29, media=72.77 | Promedio overall FIFA del favorito. |
| `dog_avg_overall` | float64 | 1.1% | min=51.00, max=87.29, media=68.81 | Promedio overall FIFA del no-favorito. |
| `fav_max_overall` | float64 | 1.1% | min=51.00, max=94.00, media=78.37 | Máximo overall FIFA del favorito (proxy jugador estrella). |
| `dog_max_overall` | float64 | 1.1% | min=51.00, max=94.00, media=74.01 | Máximo overall FIFA del no-favorito. |
| `fav_avg_attack` | float64 | 1.1% | min=50.00, max=88.67, media=72.99 | Promedio atributo ataque del favorito. |
| `dog_avg_attack` | float64 | 1.1% | min=49.00, max=88.67, media=68.94 | Promedio atributo ataque del no-favorito. |
| `fav_avg_defense` | float64 | 1.1% | min=50.00, max=87.75, media=72.66 | Promedio atributo defensa del favorito. |
| `dog_avg_defense` | float64 | 1.1% | min=49.00, max=87.75, media=68.81 | Promedio atributo defensa del no-favorito. |
| `fav_avg_pace` | float64 | 2.0% | min=43.00, max=91.00, media=72.84 | Promedio atributo ritmo/velocidad del favorito. |
| `dog_avg_pace` | float64 | 3.6% | min=43.00, max=91.00, media=71.54 | Promedio atributo ritmo/velocidad del no-favorito. |
| `fav_avg_shooting` | float64 | 2.0% | min=21.00, max=80.00, media=61.34 | Promedio atributo definición del favorito. |
| `dog_avg_shooting` | float64 | 3.6% | min=21.00, max=79.08, media=57.63 | Promedio atributo definición del no-favorito. |
| `fav_avg_passing` | float64 | 2.0% | min=29.00, max=81.85, media=64.23 | Promedio atributo pase del favorito. |
| `dog_avg_passing` | float64 | 3.6% | min=29.00, max=81.85, media=59.83 | Promedio atributo pase del no-favorito. |
| `fav_form_scored` | float64 | 0.5% | min=0.00, max=9.00, media=1.62 | Goles a favor promedio (rolling N=10) del favorito. |
| `dog_form_scored` | float64 | 0.5% | min=0.00, max=13.00, media=1.23 | Goles a favor promedio (rolling N=10) del no-favorito. |
| `fav_form_conceded` | float64 | 0.5% | min=0.00, max=16.00, media=1.20 | Goles en contra promedio (rolling N=10) del favorito. |
| `dog_form_conceded` | float64 | 0.5% | min=0.00, max=15.00, media=1.67 | Goles en contra promedio (rolling N=10) del no-favorito. |
| `fav_form_win_rate` | float64 | 0.5% | min=0.00, max=1.00, media=0.46 | Tasa de victorias (rolling N=10) del favorito. |
| `dog_form_win_rate` | float64 | 0.5% | min=0.00, max=1.00, media=0.31 | Tasa de victorias (rolling N=10) del no-favorito. |
| `fav_yellow_cards` | float64 | 98.4% | valores: [4.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 6.0, 8.0] | Tarjetas amarillas del favorito (solo Mundial 1970+). |
| `dog_yellow_cards` | float64 | 98.4% | valores: [1.0, 0.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0, 8.0] | Tarjetas amarillas del no-favorito (solo Mundial 1970+). |
| `fav_red_cards` | float64 | 98.4% | valores: [0.0, 1.0] | Tarjetas rojas del favorito (solo Mundial 1970+). |
| `dog_red_cards` | float64 | 98.4% | valores: [0.0, 1.0, 2.0] | Tarjetas rojas del no-favorito (solo Mundial 1970+). |
| `fav_team` | str | 0.3% | — | Selección favorita (mayor Elo). Reemplaza home_team como referencia principal en sede neutral. |
| `dog_team` | str | 0.3% | — | Selección no-favorita (menor Elo). |
| `fav_dog_elo_diff` | float64 | 0.3% | min=0.00, max=997.22, media=142.73 | Diferencia fav_elo - dog_elo. Siempre >= 0. |
| `fav_is_home` | float64 | 0.3% | valores: [1.0, 0.0] | 1 si el favorito coincide con el equipo home original. Preserva info de localía real para partidos is_neutral=0. |
| `target_1x2_fav_dog` | str | 0.3% | valores: ['draw', 'fav_win', 'dog_win'] | Target principal del mercado 1X2: fav_win/draw/dog_win. Reemplaza result (home_win/away_win) que era arbitrario en sede neutral. |
| `home_confed` | str | 10.6% | valores: ['UEFA', 'CONMEBOL', 'CAF', 'AFC', 'CONCACAF', 'OFC'] | Confederación de la selección local (CONMEBOL/UEFA/CAF/AFC/CONCACAF/OFC). |
| `away_confed` | str | 13.0% | valores: ['UEFA', 'CONMEBOL', 'CONCACAF', 'CAF', 'AFC', 'OFC'] | Confederación de la selección visitante. |
| `fav_confed` | str | 7.1% | valores: ['UEFA', 'CONMEBOL', 'CAF', 'AFC', 'CONCACAF', 'OFC'] | Confederación del favorito. |
| `dog_confed` | str | 16.5% | valores: ['UEFA', 'CONMEBOL', 'CONCACAF', 'CAF', 'AFC', 'OFC'] | Confederación del no-favorito. |
| `mismo_confed` | int64 | 0.0% | valores: [1, 0] | 1 si ambas selecciones pertenecen a la misma confederación, 0 si son de confederaciones distintas. |
| `tournament_weight` | float64 | 0.0% | valores: [1.0, 2.5, 1.5, 1.2, 2.0] | Peso numérico del torneo: 2.5=Mundial (fase final), 2.0=Copa continental, 1.5=Clasificatoria mundialista, 1.2=Clasificatoria continental, 1.0=Amistoso/otro. |
| `home_dias_descanso` | float64 | 0.2% | min=0.00, max=20742.00, media=61.37 | Días desde el partido anterior de la selección local (cualquier torneo). NaN en primer partido histórico. |
| `away_dias_descanso` | float64 | 0.2% | min=0.00, max=20715.00, media=62.27 | Días desde el partido anterior de la selección visitante. |
| `fav_dias_descanso` | float64 | 0.2% | min=0.00, max=19871.00, media=54.77 | Días de descanso del favorito. |
| `dog_dias_descanso` | float64 | 0.2% | min=0.00, max=20742.00, media=68.88 | Días de descanso del no-favorito. |
| `descanso_diff` | float64 | 0.4% | min=-20700.00, max=15361.00, media=-13.77 | fav_dias_descanso - dog_dias_descanso. Positivo = favorito descansó más. |
| `target_ou25` | float64 | 0.2% | valores: [0.0, 1.0] | Target Over/Under 2.5 goles totales: 1=over, 0=under. Mercado principal de goles. 50.9% over en histórico. |
| `target_btts` | float64 | 0.2% | valores: [0.0, 1.0] | Target Both Teams To Score: 1=ambos anotaron, 0=al menos uno no anotó. 46.5% sí. |
| `target_ou15` | float64 | 0.2% | valores: [0.0, 1.0] | Target Over/Under 1.5 goles: 1=over. 73.7% over. |
| `target_ou35_goals` | float64 | 0.2% | valores: [0.0, 1.0] | Target Over/Under 3.5 goles: 1=over. 31.2% over. |

## Notas metodológicas


- **Normalización de nombres**: aplicada vía `src/normalizacion_nombres.py`
  (diccionario `NAME_FIXES`, 17 entradas validadas sin ciclos). Cobertura:
  equipos del Mundial 2026 con grafías inconsistentes entre fuentes
  (Curaçao, Cape Verde Islands→Cape Verde, Côte d'Ivoire→Ivory Coast,
  Czechia→Czech Republic, Republic of Ireland→Ireland), más países
  disueltos (Czechoslovakia, Yugoslavia, German DR, etc.). Reutilizado en
  TODOS los scripts del pipeline, incluida la integración de Fjelstul.
- **`is_world_cup` vs `is_world_cup_qualifier`**: calculados por cruce de
  `key` contra `results.csv`, NO por el string `_tournament` de
  `teams_match_features.csv` (que mezclaba Mundial masculino y femenino
  bajo la misma etiqueta genérica "World Cup" — bug detectado y corregido).
  Validado: 1,036 partidos = exactamente el conteo de `results.csv`.
- **`pending_feature_engineering = 1`**: marca los 481 partidos de Mundial
  recuperados que no estaban en `teams_match_features.csv` original. Tienen
  `home_elo`/`away_elo`/`elo_diff` y `home_form_*`/`away_form_*` (rolling)
  ya calculados; solo quedan pendientes los atributos FIFA por plantilla
  (`*_avg_attack`, etc.), que requieren mapeo `fifa_version`→año.
- **`fav_*`/`dog_*` (favorito/no favorito por Elo)**: en partidos de sede
  neutral, `home`/`away` no representa localía real. Se reorientó todo el
  dataset usando Elo (mayor Elo = favorito) para tener una referencia
  consistente, especialmente para las secuencias de la LSTM. `fav_is_home`
  preserva si el favorito coincide con el home original.
- **Tarjetas (Fjelstul World Cup Database, CC-BY-SA 4.0)**: cobertura
  751/1036 partidos de Mundial (72.5%). El resto son desajustes de fecha
  y replays en partidos anteriores a 1958, sin impacto relevante dado que
  el foco es el Mundial 2026.
- **`home_form_*`/`away_form_*`**: recalculadas para TODO el dataset con
  una sola metodología (ventana de 10 partidos, `shift(1)` antes del
  rolling para evitar data leakage), en vez de mezclar la metodología
  original de `teams_match_features.csv` con una nueva solo para los
  481 recuperados.
- **Corners**: fuera de alcance como mercado con modelo entrenado (no
  existe dataset histórico confiable de selecciones). Se maneja como
  variable operacional en vivo vía API-Football durante el Mundial 2026.
- **Nulos en `home_elo`/`away_elo`**: corresponden a partidos anteriores al
  inicio de `eloratings.csv` para ese equipo. XGBoost maneja NaN nativamente.
- **`result` (home_win/away_win/draw)**: se mantiene por trazabilidad pero
  NO se recomienda como target para Mundial 2026 (sede neutral). Usar
  `target_1x2_fav_dog` en su lugar.
