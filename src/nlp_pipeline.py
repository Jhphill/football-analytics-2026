"""
nlp_pipeline.py
======================================================================
Pipeline NLP - Plataforma Predictiva Mundial 2026 (Semana 3, Lulu)

Módulos incluidos en este archivo (se van agregando por pasos):
  1. scrapear_noticias()     - descarga y normaliza entradas de los RSS
  2. detectar_selecciones()  - etiqueta cada noticia con la(s) selección(es)
                                que menciona, según SELECCIONES_KEYWORDS
  3. extraer_jugadores()     - NER con spaCy + cruce con jugadores_clave.json

Uso típico (paso 1-3):
    from nlp_pipeline import scrapear_noticias, detectar_selecciones, extraer_jugadores
    df_noticias = scrapear_noticias()
    df_noticias = detectar_selecciones(df_noticias)
    df_jugadores = extraer_jugadores(df_noticias)
    df_noticias.to_csv("data/raw/noticias_scrapeadas.csv", index=False)
    df_jugadores.to_csv("data/processed/jugadores_detectados.csv", index=False)
======================================================================
"""

import os
import json
import feedparser
import pandas as pd
import spacy
from datetime import datetime


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube de src/ a la raíz
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
FUENTES_RSS_PATH = os.path.join(DATA_RAW, 'fuentes_rss.json')
JUGADORES_CLAVE_PATH = os.path.join(BASE_DIR, 'src', 'jugadores_clave.json')


# ============================================================
# DICCIONARIO DE SELECCIONES: nombre canónico (matches_clean.csv)
# -> palabras clave de búsqueda en español/inglés
# ============================================================
SELECCIONES_KEYWORDS = {
    "Brazil":      ["Brasil", "Brazil"],
    "Argentina":   ["Argentina"],
    "France":      ["Francia", "France", "Les Bleus"],
    "Spain":       ["España", "Espana", "Spain", "La Roja"],
    "Netherlands": ["Países Bajos", "Paises Bajos", "Holanda", "Netherlands"],
    "England":     ["Inglaterra", "England", "Three Lions"],
    "Portugal":    ["Portugal"],
    "Germany":     ["Alemania", "Germany", "Die Mannschaft"],
    "USA":         ["Estados Unidos", "USA", "United States", "USMNT"],
    "Mexico":      ["México", "Mexico", "El Tri"],
    "Morocco":     ["Marruecos", "Morocco"],
}


# ============================================================
# PASO 1: SCRAPING DE RSS
# ============================================================
def cargar_fuentes_rss(ruta=FUENTES_RSS_PATH):
    """Carga el diccionario de fuentes RSS generado en la Semana 1."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró {ruta}. Corre primero src/test_rss.py (Semana 1)."
        )
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


def scrapear_noticias(fuentes=None, max_por_feed=None):
    """
    Descarga y parsea los feeds RSS definidos en fuentes_rss.json.

    Parámetros
    ----------
    fuentes : dict | None
        Diccionario {nombre: {"url": ..., ...}}. Si es None, se carga
        desde data/raw/fuentes_rss.json.
    max_por_feed : int | None
        Límite de entradas a tomar por feed (útil para pruebas rápidas).

    Retorna
    -------
    pd.DataFrame con columnas:
        fuente, titulo, resumen, fecha_publicacion, link, fecha_scrapeo
    """
    if fuentes is None:
        fuentes = cargar_fuentes_rss()

    filas = []
    fecha_scrapeo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for nombre_feed, info in fuentes.items():
        url = info["url"]
        f = feedparser.parse(url)

        entradas = f.entries
        if max_por_feed is not None:
            entradas = entradas[:max_por_feed]

        print(f"  {nombre_feed}: {len(entradas)} entradas (status={f.get('status', 'N/A')})")

        for e in entradas:
            filas.append({
                "fuente": nombre_feed,
                "titulo": e.get("title", ""),
                "resumen": e.get("summary", ""),
                "fecha_publicacion": e.get("published", ""),
                "link": e.get("link", ""),
                "fecha_scrapeo": fecha_scrapeo,
            })

    df = pd.DataFrame(filas)

    # Texto combinado (título + resumen) usado para detección de selecciones y NER
    df["texto_completo"] = (df["titulo"].fillna("") + ". " + df["resumen"].fillna("")).str.strip()

    # Deduplicar por (titulo, fuente) - algunos feeds repiten entradas entre llamadas
    n_antes = len(df)
    df = df.drop_duplicates(subset=["titulo", "fuente"]).reset_index(drop=True)
    if n_antes != len(df):
        print(f"  Duplicados eliminados: {n_antes - len(df)}")

    return df


# ============================================================
# PASO 2: DETECCIÓN DE SELECCIONES MENCIONADAS
# ============================================================
def detectar_selecciones(df, keywords=None, col_texto="texto_completo"):
    """
    Para cada noticia, detecta qué selección(es) del Mundial 2026 se
    mencionan en el texto (búsqueda simple de substrings, case-insensitive).

    Agrega la columna 'selecciones_detectadas' (lista de nombres canónicos,
    e.g. ['Spain', 'Morocco']). Las noticias sin ninguna selección detectada
    quedan con lista vacía [] (se filtran en un paso posterior si se desea).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna de texto (por defecto 'texto_completo').
    keywords : dict | None
        Diccionario {seleccion_canonica: [keywords...]}. Si es None, usa
        SELECCIONES_KEYWORDS.
    col_texto : str
        Nombre de la columna a analizar.

    Retorna
    -------
    pd.DataFrame con la columna nueva 'selecciones_detectadas'.
    """
    if keywords is None:
        keywords = SELECCIONES_KEYWORDS

    def _detectar(texto):
        if not isinstance(texto, str) or not texto:
            return []
        texto_lower = texto.lower()
        detectadas = []
        for seleccion, kws in keywords.items():
            if any(kw.lower() in texto_lower for kw in kws):
                detectadas.append(seleccion)
        return detectadas

    df = df.copy()
    df["selecciones_detectadas"] = df[col_texto].apply(_detectar)
    df["n_selecciones"] = df["selecciones_detectadas"].apply(len)

    return df


# ============================================================
# PASO 3: NER (spaCy) + CRUCE CON jugadores_clave.json
# ============================================================
_NLP_MODELS = {}  # cache simple para no recargar modelos por cada llamada


def _cargar_modelo_spacy(idioma):
    """Carga (y cachea) el modelo de spaCy correspondiente al idioma."""
    if idioma not in _NLP_MODELS:
        nombre_modelo = "es_core_news_md" if idioma == "es" else "en_core_web_md"
        _NLP_MODELS[idioma] = spacy.load(nombre_modelo)
    return _NLP_MODELS[idioma]


def cargar_jugadores_clave(ruta=JUGADORES_CLAVE_PATH):
    """Carga el diccionario de jugadores clave (Semana 1)."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró {ruta}. Completa primero src/jugadores_clave.json (Semana 1)."
        )
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


def _construir_indice_jugadores(jugadores_clave):
    """
    Construye un índice plano {nombre_jugador_lower: (seleccion, info)}
    para hacer el cruce NER -> jugador clave en O(1).
    También incluye el apellido solo (última palabra del nombre) como
    alias adicional, ya que las noticias suelen referirse solo por apellido
    (ej. "Yamal" en vez de "Lamine Yamal").
    """
    indice = {}
    for seleccion, jugadores in jugadores_clave.items():
        for nombre_completo, info in jugadores.items():
            indice[nombre_completo.lower()] = (seleccion, nombre_completo, info)
            apellido = nombre_completo.split()[-1].lower()
            # No sobrescribir si el apellido ya está mapeado a otro jugador
            # (evita colisiones, ej. dos jugadores con el mismo apellido)
            if apellido not in indice:
                indice[apellido] = (seleccion, nombre_completo, info)
    return indice


def extraer_jugadores(df, jugadores_clave=None, col_texto="texto_completo", col_idioma=None):
    """
    Aplica NER (spaCy, entidades PERSON/PER) a cada noticia y cruza las
    personas detectadas contra jugadores_clave.json.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de noticias (salida de scrapear_noticias / detectar_selecciones).
    jugadores_clave : dict | None
        Diccionario {seleccion: {nombre_jugador: {...}}}. Si es None, se
        carga desde src/jugadores_clave.json.
    col_texto : str
        Columna de texto a analizar.
    col_idioma : str | None
        Columna que indica el idioma ('es'/'en') por fila. Si es None,
        se infiere de forma simple: fuentes que empiezan con 'marca' -> 'es',
        el resto -> 'en'.

    Retorna
    -------
    pd.DataFrame en formato largo, una fila por (noticia, jugador detectado):
        idx_noticia, fuente, titulo, seleccion_jugador, jugador,
        importancia, titular_habitual, entidad_ner_original
    """
    if jugadores_clave is None:
        jugadores_clave = cargar_jugadores_clave()

    indice_jugadores = _construir_indice_jugadores(jugadores_clave)

    df = df.copy()
    if col_idioma is None:
        df["_idioma_tmp"] = df["fuente"].apply(lambda f: "es" if str(f).startswith("marca") else "en")
        col_idioma = "_idioma_tmp"

    filas = []
    for idx, row in df.iterrows():
        texto = row[col_texto]
        idioma = row[col_idioma]

        if not isinstance(texto, str) or not texto.strip():
            continue

        nlp = _cargar_modelo_spacy(idioma)
        doc = nlp(texto)

        # Etiqueta de persona: 'PER' en es_core_news_md, 'PERSON' en en_core_web_md
        personas = [ent.text for ent in doc.ents if ent.label_ in ("PER", "PERSON")]

        for persona in personas:
            persona_lower = persona.lower().strip()
            match = indice_jugadores.get(persona_lower)

            # Si no matchea el nombre completo, probar con la última palabra (apellido)
            if match is None:
                ultima_palabra = persona_lower.split()[-1] if persona_lower.split() else ""
                match = indice_jugadores.get(ultima_palabra)

            if match is not None:
                seleccion, nombre_canonico, info = match
                filas.append({
                    "idx_noticia": idx,
                    "fuente": row["fuente"],
                    "titulo": row["titulo"],
                    "texto_completo": row.get(col_texto, row["titulo"]),
                    "seleccion_jugador": seleccion,
                    "jugador": nombre_canonico,
                    "importancia": info.get("importancia"),
                    "titular_habitual": info.get("titular_habitual"),
                    "posicion": info.get("posicion"),
                    "entidad_ner_original": persona,
                })

    if "_idioma_tmp" in df.columns:
        df.drop(columns=["_idioma_tmp"], inplace=True)

    return pd.DataFrame(filas)


# ============================================================
# PASO 4: CLASIFICACIÓN DE NOTICIAS + injury_impact_score
# ============================================================

# Reglas por keyword (case-insensitive). Se evalúan en este orden:
# si el texto matchea varias categorías, gana la PRIMERA de la lista
# (irrelevante se chequea primero para filtrar ruido de fichajes/traspasos
# antes de que "deal"/"llegada" se confunda con otra cosa).
CLASIFICACION_KEYWORDS = {
    "irrelevante": [
        # fichajes / mercado de pases - no aportan señal de Mundial
        "fichaje", "traspaso", "ficha por", "llegada al", "llegada a",
        "transfer", "deal", "agree", "signs for", "se marcha al",
        "renovación", "renueva", "contrato con",
    ],
    "lesion_baja": [
        "lesión", "lesion", "lesionado", "lesionada", "se rompe",
        "rotura", "baja por lesión", "fuera del mundial", "descartado",
        "no podrá jugar", "se pierde el", "out for", "ruled out",
        "injury", "injured", "sidelined", "withdraws", "withdrawal",
        "season-ending", "tendrá que operarse",
    ],
    "recupero_alta": [
        "recuperado", "recuperada", "vuelve a", "regresa", "de vuelta",
        "alta médica", "back from injury", "available again",
        "returns to training", "ya entrena con el grupo", "se reincorpora",
    ],
    "duda": [
        "duda", "en duda", "podría perderse", "genera incertidumbre",
        "doubtful", "doubt", "on the bench", "en el banco", "banca",
        "suplente", "no es titular", "pierde la titularidad", "bench",
    ],
}

# Orden de evaluación (importante: irrelevante primero para filtrar ruido)
ORDEN_CATEGORIAS = ["irrelevante", "lesion_baja", "recupero_alta", "duda"]

# Peso de cada categoría sobre el injury_impact_score.
# Signo negativo = impacto negativo para la selección (jugador clave no
# disponible o en duda). Signo positivo = impacto positivo (recupera a
# un jugador clave). 'neutral'/'irrelevante' no aportan al score.
PESO_CATEGORIA = {
    "lesion_baja": -1.0,
    "duda": -0.5,
    "recupero_alta": +0.7,
    "neutral": 0.0,
    "irrelevante": 0.0,
}


def clasificar_noticia(texto):
    """
    Clasifica un texto de noticia en una de las categorías de
    CLASIFICACION_KEYWORDS (o 'neutral' si no matchea ninguna).

    La búsqueda es por substring, case-insensitive. Se evalúa en el
    orden de ORDEN_CATEGORIAS; la primera categoría que matchea gana.
    """
    if not isinstance(texto, str) or not texto.strip():
        return "neutral"

    texto_lower = texto.lower()

    for categoria in ORDEN_CATEGORIAS:
        keywords = CLASIFICACION_KEYWORDS[categoria]
        if any(kw.lower() in texto_lower for kw in keywords):
            return categoria

    return "neutral"


def calcular_injury_impact_score(df_jugadores, col_texto="texto_completo"):
    """
    Agrega la columna 'estado_noticia' (clasificación por keywords) y
    'impacto_individual' (importancia * peso_categoria) a df_jugadores,
    y calcula el injury_impact_score agregado por selección.

    impacto_individual = importancia_jugador * peso_categoria

    El injury_impact_score de una selección es la SUMA de los
    impacto_individual de todas sus menciones. Valores muy negativos
    indican selecciones con jugadores clave lesionados/en duda; valores
    positivos indican recuperaciones recientes de jugadores clave.

    Retorna
    -------
    df_jugadores_clasificado : pd.DataFrame
        df_jugadores con las columnas nuevas 'estado_noticia' e
        'impacto_individual'.
    df_scores : pd.DataFrame
        Una fila por selección con 'injury_impact_score' y el detalle
        de cuántas menciones tuvo en cada categoría.
    """
    df = df_jugadores.copy()

    df["estado_noticia"] = df[col_texto].apply(clasificar_noticia)
    df["impacto_individual"] = df.apply(
        lambda r: (r["importancia"] or 0) * PESO_CATEGORIA.get(r["estado_noticia"], 0.0),
        axis=1
    )

    # Agregación por selección
    score_total = df.groupby("seleccion_jugador")["impacto_individual"].sum()

    # Conteo de menciones por categoría y selección (para auditoría)
    conteo = (
        df.groupby(["seleccion_jugador", "estado_noticia"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ORDEN_CATEGORIAS + ["neutral"], fill_value=0)
    )

    df_scores = conteo.copy()
    df_scores["injury_impact_score"] = score_total
    df_scores["injury_impact_score"] = df_scores["injury_impact_score"].fillna(0.0)
    df_scores = df_scores.reset_index().rename(columns={"seleccion_jugador": "seleccion"})

    return df, df_scores


# ============================================================
# EJECUCIÓN DIRECTA (prueba rápida de todo el pipeline, pasos 1-4)
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" PASO 1: SCRAPING DE NOTICIAS")
    print("=" * 70)
    df_noticias = scrapear_noticias()
    print(f"\nTotal de noticias scrapeadas: {len(df_noticias)}")

    print("\n" + "=" * 70)
    print(" PASO 2: DETECCIÓN DE SELECCIONES MENCIONADAS")
    print("=" * 70)
    df_noticias = detectar_selecciones(df_noticias)

    con_seleccion = (df_noticias["n_selecciones"] > 0).sum()
    print(f"Noticias con al menos una selección detectada: {con_seleccion}/{len(df_noticias)}")

    print("\nDistribución de selecciones detectadas:")
    from collections import Counter
    todas = [s for lista in df_noticias["selecciones_detectadas"] for s in lista]
    print(Counter(todas))

    # Guardar resultado del paso 1-2
    out_path = os.path.join(DATA_RAW, "noticias_scrapeadas.csv")
    df_noticias.to_csv(out_path, index=False)
    print(f"\nGuardado en: {out_path}")

    print("\n" + "=" * 70)
    print(" PASO 3: NER (spaCy) + CRUCE CON jugadores_clave.json")
    print("=" * 70)
    df_jugadores = extraer_jugadores(df_noticias)
    print(f"Menciones de jugadores clave detectadas: {len(df_jugadores)}")

    if len(df_jugadores) > 0:
        print("\nMenciones por selección:")
        print(df_jugadores["seleccion_jugador"].value_counts())

        print("\n" + "=" * 70)
        print(" PASO 4: CLASIFICACIÓN + injury_impact_score")
        print("=" * 70)
        df_jugadores, df_scores = calcular_injury_impact_score(df_jugadores)

        print("\nClasificación de cada mención detectada:")
        print(df_jugadores[["jugador", "seleccion_jugador", "estado_noticia",
                             "impacto_individual", "titulo"]].to_string(index=False))

        print("\nDistribución de estados:")
        print(df_jugadores["estado_noticia"].value_counts())

        print("\ninjury_impact_score por selección:")
        print(df_scores.to_string(index=False))

        # Guardar outputs finales
        out_path_jug = os.path.join(BASE_DIR, "data", "processed", "jugadores_detectados.csv")
        os.makedirs(os.path.dirname(out_path_jug), exist_ok=True)
        df_jugadores.to_csv(out_path_jug, index=False)
        print(f"\nGuardado: {out_path_jug}")

        out_path_scores = os.path.join(BASE_DIR, "data", "processed", "injury_impact_scores.csv")
        df_scores.to_csv(out_path_scores, index=False)
        print(f"Guardado: {out_path_scores}")
    else:
        print("No se detectaron jugadores del diccionario en las noticias actuales.")