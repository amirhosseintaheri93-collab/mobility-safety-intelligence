# Architecture and extension points

## Runtime architecture

```text
Browser
  |
  v
Streamlit dashboard (app/dashboard.py)
  |-- prepared event tables (data/*.CSV)
  |-- cached OSM street context (data/street_context_geo.json)
  |-- optional prepared hotspot HTML (data/hotspot_maps/)
  |-- LightGBM/SHAP artifacts (model_cache/)
  |-- scenario notes and references (llm/)
  `-- optional server-side OpenAI API
```

Expensive model training and publication KDE generation are deliberately separated from visitor sessions.

## Main extension points

### New SUMO study

Provide compatible event tables or add a schema adapter before `load_data()`. Keep the local coordinates metric and define the correct network offset and coordinate reference system.

### Different policy levers

Extend `FLEET_COMPOSITIONS` in `app/dashboard.py` and the policy-feature construction in `app/ml_modeling.py`.

### New spatial context

Generate a replacement `data/street_context_geo.json` with the required feature collections. The dashboard degrades gracefully if a context layer is absent.

### New model family

Add a model request and training implementation in `app/ml_modeling.py`, then extend the offline precomputation matrix. Keep group-aware validation and store provenance in the cache manifest.

### Future digital-twin connection

The present release is batch-oriented. A future data-source adapter could ingest scheduled detector, connected-vehicle, or operational feeds. That would create a digital-shadow or digital-twin pathway without changing the visual and analytical layers. No live synchronization is claimed in this version.

