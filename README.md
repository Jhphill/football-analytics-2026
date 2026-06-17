# ⚽ Football Analytics 2026
### Plataforma de Analítica Predictiva — FIFA World Cup 2026

> Proyecto final de la materia de Inteligencia Artificial y Machine Learning  
> Ingeniería en Ciencia de Datos e Inteligencia de Negocios — UNIVALLE, Cochabamba, Bolivia  
> Autores: **Juanfe** (modelos + pipeline de datos) · **Lulu** (LSTM + NLP + dashboard)

---

## ¿Qué es esto?

Sistema de predicción de mercados de apuestas deportivas para el Mundial 2026, construido sobre datos históricos de selecciones nacionales (1872–2025). Predice probabilidades calibradas para 5 mercados por partido y detecta **value bets** cuando el modelo supera la probabilidad implícita en las cuotas de 1xbet.

El sistema **no garantiza resultados** — comunica probabilidades calibradas con su incertidumbre. El valor real está en que la probabilidad del modelo sea mejor que la implícita en la cuota (value betting), no en la tasa de acierto bruta.

---

## Mercados modelados

| Mercado | Modelo | AUC | Estado |
|---|---|---|---|
| 1X2 (resultado) | XGBoost calibrado (solo neutrales) | ~0.52 | ✅ Producción |
| Over/Under 2.5 goles | XGBoost + isotonic calibration | 0.581 | ✅ Producción |
| Both Teams To Score | XGBoost + isotonic calibration | 0.572 | ✅ Producción |
| Tarjetas Over/Under 3.5 | XGBoost regularizado (CV) | **0.593** | ✅ Producción |
| Tarjeta Roja | — | 0.514 | ⚠️ Baseline empírico (10.8%) |

> **Nota sobre Tarjeta Roja:** con solo 81 eventos positivos en 751 partidos de Mundial, el dataset es insuficiente para superar el baseline empírico. Se muestra la probabilidad histórica (≈10.8%) con advertencia explícita. Un modelo robusto requeriría historial disciplinario individual de jugadores y perfil del árbitro asignado.

---

## Arquitectura del sistema

```
Datos históricos (1872–2025)
        ↓
Feature Engineering
  · Elo ratings (fav/dog)
  · Forma reciente rolling N=10
  · Atributos FIFA (plantilla)
  · Tarjetas reales WC (Fjelstul DB)
  · Confederaciones, días descanso
  · Tournament weight
        ↓
┌─────────────────┬──────────────────┐
│  Modelos XGBoost │  LSTM secuencial │
│  (uno/mercado)   │  (Lulu, Semana 2)│
└────────┬─────────┴────────┬─────────┘
         ↓                  ↓
   Calibración isotónica
         ↓
   Capa de ajuste dinámico
   · injury_impact_score (NLP/NER)
   · presión situacional (en vivo)
         ↓
   Value Bet Detector
   · EV = prob_modelo × cuota - 1
   · Kelly Criterion para sizing
         ↓
   Dashboard Streamlit
```

---

## Estructura del repositorio

```
football-analytics-2026/
├── data/
│   ├── raw/                        # Datos originales (no modificados)
│   │   └── fjelstul/               # Fjelstul World Cup Database (tarjetas)
│   └── processed/                  # Datasets limpios y features finales
│       ├── matches_clean.csv
│       ├── matches_features_v2.csv  # Dataset principal (43,816 × 95)
│       ├── train_set.csv
│       ├── test_set.csv
│       └── semana3_metricas_resumen.csv
├── models/                         # Modelos entrenados (.pkl) y métricas (.json)
│   ├── modelos_dashboard.json      # Config: qué modelo usa cada mercado
│   ├── 1x2_neutral_xgboost_calibrado.pkl
│   ├── over_under_2.5_goles_xgboost_calibrado.pkl
│   ├── both_teams_to_score_xgboost_calibrado.pkl
│   └── tarjetas_over_under_3.5_xgboost_calibrado.pkl
├── src/                            # Scripts de producción (numerados por orden)
│   ├── 01_auditoria_datos.py
│   ├── 03_limpieza_y_merge.py
│   ├── 04_diccionario_variables.py
│   ├── 05_features_tarjetas.py
│   ├── 06_favorito_no_favorito.py
│   ├── 07_forma_reciente.py
│   ├── 08_contexto_variables.py
│   ├── 09_cierre_semana2.py
│   ├── 10_preparar_datos_modelado.py
│   ├── 11_modelo_1x2.py
│   ├── 11b_modelo_1x2_solo_neutrales.py
│   ├── 12_modelo_goles.py
│   ├── 13_modelo_tarjetas.py
│   ├── 13b_modelo_tarjeta_roja_fix.py
│   ├── 14_cierre_semana3.py
│   ├── feature_lists.py            # Listas de features reutilizables
│   ├── normalizacion_nombres.py    # Diccionario NAME_FIXES
│   └── presion_situacional.py      # Función para dashboard (en vivo)
├── notebooks/
│   ├── 01_eda_limpieza.ipynb
│   ├── 02_features.ipynb
│   ├── 05_lstm.ipynb               # Lulu
│   └── 06_nlp.ipynb                # Lulu
├── dashboard/
│   └── app.py                      # Streamlit (Semana 5)
├── report/
│   └── informe.tex                 # LaTeX (Semana 6)
├── requirements.txt
└── README.md
```

---

## Cómo reproducir

### 1. Clonar y crear entorno

```bash
git clone https://github.com/Jhphill/football-analytics-2026.git
cd football-analytics-2026
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Descargar datos externos

Los datasets crudos no están en el repositorio por tamaño. Descargar y colocar en `data/raw/`:

| Dataset | Fuente |
|---|---|
| `results.csv` | [Kaggle: martj42/international-football-results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| `teams_match_features.csv` | Kaggle (mismo dataset) |
| `eloratings.csv` | [eloratings.net](https://www.eloratings.net) |
| `player_aggregates.csv` | Kaggle: FIFA ratings históricos |
| `future_match_probabilities_baseline.csv` | Generado internamente |
| `fjelstul/matches.csv` + `bookings.csv` | [GitHub: jfjelstul/worldcup](https://github.com/jfjelstul/worldcup/tree/master/data-csv) |

### 3. Correr el pipeline completo (Semanas 1–3)

```bash
# Semana 1 — Limpieza
python src/01_auditoria_datos.py
python src/03_limpieza_y_merge.py

# Semana 2 — Feature Engineering
python src/05_features_tarjetas.py
python src/06_favorito_no_favorito.py
python src/07_forma_reciente.py
python src/08_contexto_variables.py
python src/09_cierre_semana2.py

# Semana 3 — Modelado
python src/10_preparar_datos_modelado.py
python src/11_modelo_1x2.py
python src/11b_modelo_1x2_solo_neutrales.py
python src/12_modelo_goles.py
python src/13_modelo_tarjetas.py
python src/13b_modelo_tarjeta_roja_fix.py
python src/14_cierre_semana3.py
```

---

## Decisiones de diseño clave

**Orientación fav/dog en vez de home/away:** en el Mundial 2026 casi todos los partidos son en sede neutral (USA/México/Canadá), por lo que "local/visitante" no tiene significado real. Se reorienta cada partido según Elo: el equipo con mayor rating es `fav_*`, el otro es `dog_*`. Esto da una representación consistente para el modelo.

**Split temporal, no aleatorio:** el corte es 2018-01-01. Un split aleatorio mezclaría partidos de 2024 en train con partidos de 1990 en test, lo que no representa el problema real (siempre predecimos el futuro con datos del pasado).

**fav_is_home excluida de features:** aunque correlaciona con el resultado en el histórico (incluso en partidos neutrales, por sesgo editorial de la fuente), en el dataset de predicción real del WC2026 el "home_team" está determinado por ser el país anfitrión — usarla habría inflado artificialmente la probabilidad de los anfitriones.

**Tarjeta roja como baseline empírico:** con 81 positivos en 751 partidos, ninguna configuración de XGBoost supera AUC 0.52. Se muestra la probabilidad histórica (10.8%) con advertencia. Documentado en `tarjeta_roja_metrics.json`.

---

## Stack tecnológico

```
Python 3.11 · pandas · numpy · scikit-learn
xgboost · tensorflow/keras (LSTM)
shap · lime
streamlit · plotly
nltk · transformers · spaCy
SQLite · Git
```

---

## Estado del proyecto

| Semana | Juanfe | Lulu |
|---|---|---|
| 1 — Limpieza | ✅ Completo | ✅ Completo |
| 2 — Features | ✅ Completo | ✅ Completo |
| 3 — Modelos | ✅ Completo | 🔄 En curso (NLP + LSTM) |
| 4 — XAI + Value Bet | 🔜 Próximo | 🔜 Dashboard base |
| 5 — Validación | ⏳ | ⏳ |
| 6 — Informe LaTeX | ⏳ | ⏳ |

---

## Advertencia legal

Este sistema es un proyecto académico. Las probabilidades generadas son estimaciones estadísticas basadas en datos históricos y no constituyen asesoramiento financiero ni garantía de resultados. Las apuestas deportivas pueden generar pérdidas económicas. Jugar con responsabilidad.