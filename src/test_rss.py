import feedparser
import json

# --- Feeds definitivos identificados en la exploración ---
FEEDS_DEFINITIVOS = {
    "espn_mundial_en": {
        "url": "https://www.espn.com/espn/rss/soccer/news",
        "idioma": "en",
        "notas": "Cobertura general de fútbol/Mundial en inglés, actualizado al día"
    },
    "marca_mundial_es": {
        "url": "https://www.marca.com/rss/futbol/mundial.xml",
        "idioma": "es",
        "notas": "Feed específico de Mundial 2026 en español, muy actualizado"
    },
    "marca_general": {
        "url": "https://www.marca.com/rss/futbol.xml",
        "idioma": "es",
        "notas": "Feed general de fútbol, útil como respaldo o para filtrar por selección"
    },
}

# --- Feeds descartados (referencia, no usar) ---
FEEDS_DESCARTADOS = {
    "espn_es": "https://www.espn.com.mx/espn/rss/futbol/news",       # status 503
    "as_general": "https://as.com/rss/futbol/portada.xml",            # contenido desactualizado/genérico
    "as_seleccion": "https://as.com/rss/futbol/seleccion/portada.xml" # status 404
}


def probar_feeds(feeds_dict):
    """Prueba cada feed y muestra status, cantidad de entradas y un ejemplo."""
    resultados = {}
    for nombre, info in feeds_dict.items():
        url = info["url"]
        f = feedparser.parse(url)
        status = f.get("status", "N/A")
        n_entradas = len(f.entries)

        print(f"\n--- {nombre} ---")
        print("URL:", url)
        print("Status HTTP:", status)
        print("Entradas:", n_entradas)

        if f.entries:
            ejemplo = f.entries[0]
            print("Ejemplo título:", ejemplo.title)
            print("Fecha:", ejemplo.get("published", "sin fecha"))
            print("Resumen:", ejemplo.get("summary", "")[:150])

        resultados[nombre] = {
            **info,
            "status": status,
            "n_entradas": n_entradas
        }
    return resultados


def guardar_fuentes(resultados, ruta="data/raw/fuentes_rss.json"):
    """Guarda la ficha de fuentes RSS para uso en notebooks futuros (NLP, semana 3)."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en {ruta}")


if __name__ == "__main__":
    resultados = probar_feeds(FEEDS_DEFINITIVOS)
    guardar_fuentes(resultados)