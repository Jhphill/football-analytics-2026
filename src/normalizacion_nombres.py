"""
02_normalizacion_nombres.py
======================================================================
Diccionario centralizado de normalización de nombres de selecciones.
Se importa desde los demás scripts del proyecto.

CONVENCIÓN: el nombre canónico es el que usa results.csv / 
teams_match_features.csv (_home_team / _away_team), ya que es el 
dataset "columna vertebral" del proyecto (49,477 partidos). 
player_aggregates.csv se normaliza HACIA esa convención.

Contiene:
  - NAME_FIXES: mapeo de variantes -> nombre canónico (UNIDIRECCIONAL)
  - validar_name_fixes(): detecta ciclos/conflictos en el diccionario
  - normalizar_nombre(): aplica el mapeo a un string
  - normalizar_columna(): aplica el mapeo a una columna de DataFrame
  - Bloque if __name__: verifica las grafías de los equipos problemáticos
    en player_aggregates.csv para ajustar el diccionario.
======================================================================
"""

import pandas as pd
import os

# ============================================================
# DICCIONARIO PRINCIPAL DE NORMALIZACIÓN
# Regla: cada entrada es VARIANTE -> CANÓNICO.
# El valor (canónico) NUNCA debe aparecer también como clave
# (eso crearía un ciclo). validar_name_fixes() lo chequea.
# ============================================================
NAME_FIXES = {
    # --- Equipos del Mundial 2026 con grafías inconsistentes ---
    # Canónico = como aparece en results.csv / teams_match_features.csv
    "Côte d'Ivoire": 'Ivory Coast',
    'Cape Verde Islands': 'Cape Verde', # player_aggregates usa "Islands"
    'Curaçao': 'Curacao',              # sin tilde
    'Czechia': 'Czech Republic',
    'Republic of Ireland': 'Ireland',

    # --- Selecciones con nombre histórico distinto en alguna fuente ---
    'Korea Republic': 'South Korea',
    'IR Iran': 'Iran',
    'United States': 'USA',
    'Türkiye': 'Turkey',

    # --- Países disueltos / renombrados (mapean a su sucesor cuando
    #     corresponde para cruzar con player_aggregates, que solo
    #     conoce países que existen desde ~2007) ---
    'German DR': 'GermanDR',           # Alemania del Este (sin espacio)
    'West Germany': 'Germany',
    'East Germany': 'Germany',
    'Soviet Union': 'Russia',
    'Zaire': 'DR Congo',
    'Congo DR': 'DR Congo',
    'Swaziland': 'Eswatini',
    'Macedonia': 'North Macedonia',

    # NOTA: 'Czechoslovakia' y 'Yugoslavia' NO se mapean a ningún
    # sucesor único (se dividieron en varios países). Se dejan tal cual
    # en results.csv; sus partidos quedarán sin atributos FIFA en
    # player_aggregates (NaN), lo cual es correcto y aceptable.
}

# ============================================================
# VALIDACIÓN DEL DICCIONARIO (evita ciclos/bidireccionalidad)
# ============================================================
def validar_name_fixes(name_fixes=NAME_FIXES):
    """
    Verifica que ningún valor (canónico) aparezca también como clave
    (variante). Si esto ocurre, hay un ciclo y la normalización no
    converge a un único nombre.
    """
    claves = set(name_fixes.keys())
    valores = set(name_fixes.values())

    conflictos = claves & valores
    if conflictos:
        raise ValueError(
            f"NAME_FIXES tiene mapeos en ciclo (clave == valor de otra "
            f"entrada): {conflictos}. Revisa y deja una sola dirección."
        )

    # Auto-mapeos inútiles (variante == canónico)
    inutiles = {k for k, v in name_fixes.items() if k == v}
    if inutiles:
        raise ValueError(
            f"NAME_FIXES tiene auto-mapeos sin efecto: {inutiles}. "
            f"Elimínalos."
        )

    return True


# ============================================================
# FUNCIONES DE NORMALIZACIÓN
# ============================================================
def normalizar_nombre(nombre):
    """Aplica el diccionario de normalización a un nombre de equipo."""
    if isinstance(nombre, str):
        return NAME_FIXES.get(nombre, nombre)
    return nombre


def normalizar_columna(df, columna):
    """
    Aplica normalización a una columna completa de un DataFrame.
    Modifica el DataFrame in-place y lo devuelve.
    """
    df[columna] = df[columna].apply(normalizar_nombre)
    return df


# ============================================================
# VERIFICACIÓN DE GRAFÍAS EN player_aggregates.csv
# (se ejecuta solo si corres este script directamente)
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print(" VALIDACIÓN DEL DICCIONARIO NAME_FIXES")
    print("=" * 70)
    validar_name_fixes()
    print(f"✅ Sin ciclos ni auto-mapeos inútiles. {len(NAME_FIXES)} entradas.")

    print("\n" + "=" * 70)
    print(" VERIFICACIÓN DE NOMBRES DE EQUIPOS EN player_aggregates.csv")
    print("=" * 70)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pa_path = os.path.join(BASE_DIR, 'data', 'raw', 'player_aggregates.csv')

    if not os.path.exists(pa_path):
        print(f"❌ No se encontró {pa_path}")
        print("   Asegúrate de tener el archivo en data/raw/")
        exit(1)

    df_pa = pd.read_csv(pa_path)
    print(f"✅ player_aggregates.csv cargado, shape: {df_pa.shape}")

    # Candidatos a verificar: tanto la variante como el canónico actual
    candidatos = [
        'Cape Verde', 'Cape Verde Islands',
        'Curacao', 'Curaçao',
        'Ivory Coast', "Côte d'Ivoire",
        'Czech Republic', 'Czechia',
        'Republic of Ireland', 'Ireland', 'Northern Ireland',
        'USA', 'United States',
        'GermanDR', 'German DR',
    ]

    print("\n🔍 Búsqueda EXACTA en columna 'country' (no substring):")
    valores_pa = set(df_pa['country'].unique())
    resultados = {}
    for c in candidatos:
        existe = c in valores_pa
        resultados[c] = existe
        if existe:
            print(f"  ✅ '{c:20s}' -> existe EXACTAMENTE en player_aggregates")
        else:
            print(f"  ❌ '{c:20s}' -> NO existe (coincidencia exacta)")

    # Chequeo dirigido: ¿el canónico actual de NAME_FIXES existe en player_aggregates?
    print("\n" + "=" * 70)
    print(" CHEQUEO: ¿el canónico elegido en NAME_FIXES existe en player_aggregates?")
    print("=" * 70)
    pares_a_revisar = {
        'Cape Verde Islands': 'Cape Verde',
        'Curaçao': 'Curacao',
        "Côte d'Ivoire": 'Ivory Coast',
        'Czechia': 'Czech Republic',
        'Republic of Ireland': 'Ireland',
    }
    for variante, canonico in pares_a_revisar.items():
        existe_canonico = resultados.get(canonico, False)
        existe_variante = resultados.get(variante, False)
        if existe_canonico:
            print(f"  ✅ '{variante}' -> '{canonico}' : OK, el canónico existe en player_aggregates")
        elif existe_variante:
            print(f"  ⚠️  '{variante}' -> '{canonico}' : el canónico NO existe, pero "
                  f"'{variante}' SÍ -> considera invertir el mapeo en NAME_FIXES")
        else:
            print(f"  ❌ Ninguna de las dos variantes existe exactamente en player_aggregates "
                  f"-> esos atributos quedarán en NaN (aceptable)")

    print("\n" + "=" * 70)
    print(" Ajusta NAME_FIXES según los resultados de arriba y vuelve a")
    print(" correr este script hasta que validar_name_fixes() pase y")
    print(" los pares críticos (Cape Verde, Curaçao) queden en ✅.")
    print("=" * 70)