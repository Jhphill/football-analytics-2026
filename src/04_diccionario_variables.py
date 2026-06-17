"""
04_diccionario_variables.py
======================================================================
Genera data/processed/diccionario_variables.md a partir de
matches_clean.csv.

Por cada columna calcula automáticamente:
  - tipo de dato
  - % de nulos
  - ejemplo de valor (primera fila no nula)
  - para columnas numéricas: min/max/media
  - para columnas categóricas con pocos valores únicos: los valores

La DESCRIPCIÓN de cada variable se toma de un diccionario manual
(DESCRIPCIONES). Si una columna no está documentada ahí, se marca
como "(pendiente de documentar)" para que no se nos pase ninguna
columna silenciosamente.

Este script NO modifica matches_clean.csv, solo lo lee.
======================================================================
"""

import pandas as pd
import os

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

clean_path = os.path.join(DATA_PROCESSED, 'matches_features_v2.csv')
out_path = os.path.join(DATA_PROCESSED, 'diccionario_variables.md')

print("=" * 70)
print(" 04_diccionario_variables.py - Generación del diccionario")
print("=" * 70)

df = pd.read_csv(clean_path)
print(f"\n✅ matches_features_v2.csv cargado: {df.shape}")

# ============================================================
# DESCRIPCIONES MANUALES (criterio humano)
# ============================================================
DESCRIPCIONES = {
    # --- Identificación del partido ---
    '_date': 'Fecha del partido (YYYY-MM-DD).',
    '_home_team': 'Selección "local" (o primera mencionada si es sede neutral). Nombre normalizado.',
    '_away_team': 'Selección "visitante" (o segunda mencionada si es sede neutral). Nombre normalizado.',
    '_tournament': 'Nombre del torneo o competición (texto original de la fuente).',
    'key': 'Clave compuesta única `_date_home_away`, usada internamente para deduplicar y hacer merges.',

    # --- Resultado ---
    'home_goals': 'Goles anotados por la selección "local" al término del partido.',
    'away_goals': 'Goles anotados por la selección "visitante" al término del partido.',
    'total_goals': 'Suma de home_goals + away_goals. Insumo directo para targets Over/Under.',
    'result': 'Resultado categórico: home_win / draw / away_win. Target base del mercado 1X2.',

    # --- Contexto del torneo ---
    'is_neutral': 'Indicador (0/1): 1 si el partido se jugó en sede neutral (sin ventaja de localía real).',
    'is_world_cup': 'Indicador (0/1): 1 si _tournament es exactamente "FIFA World Cup" (fase final).',
    'is_world_cup_qualifier': 'Indicador (0/1): 1 si _tournament es una clasificatoria mundialista ("FIFA World Cup qualification").',
    'is_continental': 'Indicador (0/1): 1 si el torneo es una copa continental (Euro, Copa América, AFCON, etc.).',
    'pending_feature_engineering': 'Indicador (0/1): 1 en los 481 partidos de Mundial recuperados de results.csv. La forma reciente ya fue calculada para 477/481; los 4 restantes son debuts sin historia previa.',
    'tournament_weight': 'Peso numérico del torneo: 2.5=Mundial (fase final), 2.0=Copa continental, 1.5=Clasificatoria mundialista, 1.2=Clasificatoria continental, 1.0=Amistoso/otro.',

    # --- Fortaleza relativa (Elo) ---
    'home_elo': 'Rating Elo de la selección local en la fecha del partido (último valor conocido, vía merge_asof contra eloratings.csv).',
    'away_elo': 'Rating Elo de la selección visitante en la fecha del partido.',
    'elo_diff': 'Diferencia de Elo: home_elo - away_elo. Variable clave de fortaleza relativa.',

    # --- Atributos FIFA (videojuego) de la plantilla ---
    'home_avg_overall': 'Promedio del atributo "overall" (FIFA videojuego) de la plantilla local.',
    'home_max_overall': 'Máximo "overall" de la plantilla local (proxy de "jugador estrella").',
    'home_avg_attack': 'Promedio del atributo de ataque de la plantilla local.',
    'home_avg_defense': 'Promedio del atributo de defensa de la plantilla local.',
    'home_avg_pace': 'Promedio del atributo de ritmo/velocidad de la plantilla local.',
    'home_avg_shooting': 'Promedio del atributo de definición/tiro de la plantilla local.',
    'home_avg_passing': 'Promedio del atributo de pase de la plantilla local.',
    'away_avg_overall': 'Promedio del atributo "overall" de la plantilla visitante.',
    'away_max_overall': 'Máximo "overall" de la plantilla visitante.',
    'away_avg_attack': 'Promedio del atributo de ataque de la plantilla visitante.',
    'away_avg_defense': 'Promedio del atributo de defensa de la plantilla visitante.',
    'away_avg_pace': 'Promedio del atributo de ritmo/velocidad de la plantilla visitante.',
    'away_avg_shooting': 'Promedio del atributo de definición/tiro de la plantilla visitante.',
    'away_avg_passing': 'Promedio del atributo de pase de la plantilla visitante.',
    'overall_diff': 'Diferencia home_avg_overall - away_avg_overall.',
    'attack_diff': 'Diferencia home_avg_attack - away_avg_attack.',
    'defense_diff': 'Diferencia home_avg_defense - away_avg_defense.',

    # --- Forma reciente (rolling, calculada previo al partido) ---
    'home_form_scored': 'Goles a favor promedio (rolling N=10, shift(1) sin leakage) de la selección local. Calculado uniformemente en 07_forma_reciente.py.',
    'home_form_conceded': 'Goles en contra promedio (rolling N=10) de la selección local.',
    'home_form_win_rate': 'Tasa de victorias (rolling N=10) de la selección local.',
    'away_form_scored': 'Goles a favor promedio (rolling N=10) de la selección visitante.',
    'away_form_conceded': 'Goles en contra promedio (rolling N=10) de la selección visitante.',
    'away_form_win_rate': 'Tasa de victorias (rolling N=10) de la selección visitante.',

    # --- Tarjetas (Fjelstul World Cup Database, CC-BY-SA 4.0) ---
    'home_yellow_cards': 'Tarjetas amarillas de la selección local. Solo partidos is_world_cup=1 desde 1970. Fuente: Fjelstul WC DB.',
    'home_red_cards': 'Tarjetas rojas de la selección local. Misma cobertura.',
    'away_yellow_cards': 'Tarjetas amarillas de la selección visitante.',
    'away_red_cards': 'Tarjetas rojas de la selección visitante.',
    'total_cards': 'Suma total de tarjetas (amarillas + rojas) del partido.',
    'target_cards_ou35': 'Target Over/Under 3.5 tarjetas totales: 1=over, 0=under. 751 partidos de Mundial (1970-2022).',
    'target_redcard': 'Target tarjeta roja: 1=al menos una roja, 0=sin rojas. 751 partidos de Mundial.',

    # --- Orientación favorito/no favorito (06_favorito_no_favorito.py) ---
    'fav_team': 'Selección favorita (mayor Elo). Reemplaza home_team como referencia principal en sede neutral.',
    'dog_team': 'Selección no-favorita (menor Elo).',
    'fav_elo': 'Rating Elo del favorito antes del partido.',
    'dog_elo': 'Rating Elo del no-favorito antes del partido.',
    'fav_dog_elo_diff': 'Diferencia fav_elo - dog_elo. Siempre >= 0.',
    'fav_is_home': '1 si el favorito coincide con el equipo home original. Preserva info de localía real para partidos is_neutral=0.',
    'fav_goals': 'Goles del favorito en el partido.',
    'dog_goals': 'Goles del no-favorito en el partido.',
    'fav_avg_overall': 'Promedio overall FIFA del favorito.',
    'dog_avg_overall': 'Promedio overall FIFA del no-favorito.',
    'fav_max_overall': 'Máximo overall FIFA del favorito (proxy jugador estrella).',
    'dog_max_overall': 'Máximo overall FIFA del no-favorito.',
    'fav_avg_attack': 'Promedio atributo ataque del favorito.',
    'dog_avg_attack': 'Promedio atributo ataque del no-favorito.',
    'fav_avg_defense': 'Promedio atributo defensa del favorito.',
    'dog_avg_defense': 'Promedio atributo defensa del no-favorito.',
    'fav_avg_pace': 'Promedio atributo ritmo/velocidad del favorito.',
    'dog_avg_pace': 'Promedio atributo ritmo/velocidad del no-favorito.',
    'fav_avg_shooting': 'Promedio atributo definición del favorito.',
    'dog_avg_shooting': 'Promedio atributo definición del no-favorito.',
    'fav_avg_passing': 'Promedio atributo pase del favorito.',
    'dog_avg_passing': 'Promedio atributo pase del no-favorito.',
    'fav_form_scored': 'Goles a favor promedio (rolling N=10) del favorito.',
    'dog_form_scored': 'Goles a favor promedio (rolling N=10) del no-favorito.',
    'fav_form_conceded': 'Goles en contra promedio (rolling N=10) del favorito.',
    'dog_form_conceded': 'Goles en contra promedio (rolling N=10) del no-favorito.',
    'fav_form_win_rate': 'Tasa de victorias (rolling N=10) del favorito.',
    'dog_form_win_rate': 'Tasa de victorias (rolling N=10) del no-favorito.',
    'fav_yellow_cards': 'Tarjetas amarillas del favorito (solo Mundial 1970+).',
    'dog_yellow_cards': 'Tarjetas amarillas del no-favorito (solo Mundial 1970+).',
    'fav_red_cards': 'Tarjetas rojas del favorito (solo Mundial 1970+).',
    'dog_red_cards': 'Tarjetas rojas del no-favorito (solo Mundial 1970+).',
    'target_1x2_fav_dog': 'Target principal del mercado 1X2: fav_win/draw/dog_win. Reemplaza result (home_win/away_win) que era arbitrario en sede neutral.',

    # --- Confederaciones (08_contexto_variables.py) ---
    'home_confed': 'Confederación de la selección local (CONMEBOL/UEFA/CAF/AFC/CONCACAF/OFC).',
    'away_confed': 'Confederación de la selección visitante.',
    'fav_confed': 'Confederación del favorito.',
    'dog_confed': 'Confederación del no-favorito.',
    'mismo_confed': '1 si ambas selecciones pertenecen a la misma confederación, 0 si son de confederaciones distintas.',

    # --- Días de descanso (08_contexto_variables.py) ---
    'home_dias_descanso': 'Días desde el partido anterior de la selección local (cualquier torneo). NaN en primer partido histórico.',
    'away_dias_descanso': 'Días desde el partido anterior de la selección visitante.',
    'fav_dias_descanso': 'Días de descanso del favorito.',
    'dog_dias_descanso': 'Días de descanso del no-favorito.',
    'descanso_diff': 'fav_dias_descanso - dog_dias_descanso. Positivo = favorito descansó más.',

    # --- Targets de goles (09_cierre_semana2.py) ---
    'target_ou25': 'Target Over/Under 2.5 goles totales: 1=over, 0=under. Mercado principal de goles. 50.9% over en histórico.',
    'target_btts': 'Target Both Teams To Score: 1=ambos anotaron, 0=al menos uno no anotó. 46.5% sí.',
    'target_ou15': 'Target Over/Under 1.5 goles: 1=over. 73.7% over.',
    'target_ou35_goals': 'Target Over/Under 3.5 goles: 1=over. 31.2% over.',
}

# ============================================================
# GENERAR FILAS DE LA TABLA
# ============================================================
filas = []
columnas_sin_doc = []

for col in df.columns:
    serie = df[col]
    tipo = str(serie.dtype)
    pct_null = serie.isnull().mean() * 100

    # Ejemplo de valor (primera fila no nula)
    no_nulos = serie.dropna()
    if len(no_nulos) > 0:
        ejemplo = no_nulos.iloc[0]
        if isinstance(ejemplo, float):
            ejemplo = round(ejemplo, 3)
    else:
        ejemplo = '—'

    # Rango para columnas numéricas
    if pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 10:
        rango = f"min={serie.min():.2f}, max={serie.max():.2f}, media={serie.mean():.2f}"
    elif serie.nunique() <= 10:
        valores = serie.dropna().unique().tolist()
        rango = f"valores: {valores}"
    else:
        rango = '—'

    desc = DESCRIPCIONES.get(col)
    if desc is None:
        desc = '*(pendiente de documentar)*'
        columnas_sin_doc.append(col)

    filas.append({
        'Variable': f'`{col}`',
        'Tipo': tipo,
        '% Nulos': f'{pct_null:.1f}%',
        'Ejemplo / Rango': rango,
        'Descripción': desc,
    })

# ============================================================
# AGRUPAR POR SECCIÓN (para que el .md sea legible, no una tabla gigante)
# ============================================================
GRUPOS = {
    'Identificación del partido': ['_date', '_home_team', '_away_team', '_tournament', 'key'],
    'Resultado y targets base': ['home_goals', 'away_goals', 'total_goals', 'result'],
    'Contexto del torneo': [
        'is_neutral', 'is_world_cup', 'is_world_cup_qualifier',
        'is_continental', 'pending_feature_engineering'
    ],
    'Fortaleza relativa (Elo)': ['home_elo', 'away_elo', 'elo_diff'],
    'Atributos FIFA de la plantilla': [
        'home_avg_overall', 'home_max_overall', 'home_avg_attack', 'home_avg_defense',
        'home_avg_pace', 'home_avg_shooting', 'home_avg_passing',
        'away_avg_overall', 'away_max_overall', 'away_avg_attack', 'away_avg_defense',
        'away_avg_pace', 'away_avg_shooting', 'away_avg_passing',
        'overall_diff', 'attack_diff', 'defense_diff',
    ],
    'Forma reciente (rolling)': [
        'home_form_scored', 'home_form_conceded', 'home_form_win_rate',
        'away_form_scored', 'away_form_conceded', 'away_form_win_rate',
    ],
}

col_to_grupo = {}
for grupo, cols in GRUPOS.items():
    for c in cols:
        col_to_grupo[c] = grupo

# Columnas no clasificadas en ningún grupo (por si aparecen nuevas)
otras = [c for c in df.columns if c not in col_to_grupo]
if otras:
    GRUPOS['Otras (sin clasificar)'] = otras
    for c in otras:
        col_to_grupo[c] = 'Otras (sin clasificar)'

# ============================================================
# CONSTRUIR EL MARKDOWN
# ============================================================
fila_por_col = {f['Variable'].strip('`'): f for f in filas}

lineas = []
lineas.append("# Diccionario de variables — `matches_features_v2.csv`\n")
lineas.append(f"**Filas:** {df.shape[0]:,} | **Columnas:** {df.shape[1]}")
lineas.append(f"\n**Rango de fechas:** {df['_date'].min()} → {df['_date'].max()}")
lineas.append(f"\n**Generado automáticamente por:** `src/04_diccionario_variables.py`")
lineas.append(f"\n**Fuente:** `data/processed/matches_features_v2.csv` (output final Semana 2)\n")

lineas.append("\n## Resumen ejecutivo\n")
lineas.append(f"- Total de partidos: **{df.shape[0]:,}**")
lineas.append(f"- Partidos de Copa Mundial (fase final): **{df['is_world_cup'].sum():,}**")
lineas.append(f"- Partidos de clasificatoria mundialista: **{df['is_world_cup_qualifier'].sum():,}**")
lineas.append(f"- Partidos en sede neutral: **{df['is_neutral'].sum():,}**")
lineas.append(f"- Partidos con atributos FIFA pendientes: **{df['pending_feature_engineering'].sum():,}** (forma reciente ya calculada)")
lineas.append(f"- Distribución `target_1x2_fav_dog`: {df['target_1x2_fav_dog'].value_counts().to_dict()}")

for grupo, cols in GRUPOS.items():
    lineas.append(f"\n## {grupo}\n")
    lineas.append("| Variable | Tipo | % Nulos | Ejemplo / Rango | Descripción |")
    lineas.append("|---|---|---|---|---|")
    for c in cols:
        if c not in fila_por_col:
            continue
        f = fila_por_col[c]
        lineas.append(f"| {f['Variable']} | {f['Tipo']} | {f['% Nulos']} | {f['Ejemplo / Rango']} | {f['Descripción']} |")

# ============================================================
# NOTAS METODOLÓGICAS (cierre Semana 1 + 2)
# ============================================================
lineas.append("\n## Notas metodológicas\n")
lineas.append("""
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
""")

if columnas_sin_doc:
    lineas.append("\n## ⚠️ Columnas sin descripción manual\n")
    lineas.append("Las siguientes columnas no tienen entrada en `DESCRIPCIONES` "
                   "y deben documentarse:\n")
    for c in columnas_sin_doc:
        lineas.append(f"- `{c}`")

# ============================================================
# GUARDAR
# ============================================================
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lineas))

print(f"\n✅ Diccionario generado: {out_path}")
print(f"   Columnas documentadas: {df.shape[1] - len(columnas_sin_doc)}/{df.shape[1]}")
if columnas_sin_doc:
    print(f"   ⚠️ Columnas SIN documentar: {columnas_sin_doc}")
else:
    print("   ✅ Todas las columnas están documentadas.")

print("\n" + "=" * 70)
print(" SEMANA 1 + 2 — CHECKLIST DE CIERRE")
print("=" * 70)
print("""
[X] 01_auditoria_datos.py        -> diagnóstico completo
[X] normalizacion_nombres.py     -> NAME_FIXES validado, sin ciclos
[X] 03_limpieza_y_merge.py       -> matches_clean.csv (43,816 x 40)
[X] 05_features_tarjetas.py      -> target_cards_ou35, target_redcard
[X] 06_favorito_no_favorito.py   -> fav_*/dog_*, target_1x2_fav_dog
[X] 07_forma_reciente.py         -> rolling windows uniformes (N=10)
[X] 08_contexto_variables.py     -> confederaciones, tournament_weight, dias_descanso
[X] 09_cierre_semana2.py         -> target_ou25/btts/ou15/ou35_goals
[X] 04_diccionario_variables.py  -> diccionario_variables.md (95/95 columnas)

Dataset final: matches_features_v2.csv (43,816 x 95)

PRÓXIMO: SEMANA 3
  - 03_baseline.ipynb -> Logistic Regression por mercado
  - 04_xgboost.ipynb  -> XGBoost calibrado por mercado
  - Métricas: log-loss, Brier Score, AUC, accuracy
""")