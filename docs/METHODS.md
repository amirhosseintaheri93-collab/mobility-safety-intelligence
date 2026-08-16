# Methods and implementation

This document connects each analytical idea to the code that produces the corresponding output.

## 1. Input data and harmonisation

The application reads three semicolon-delimited event tables:

| Desired headway | File |
| --- | --- |
| 0.6 s | `data/ds_vt_ct_csv.CSV` |
| 0.8 s | `data/ds_vt_ct_0.8_csv.CSV` |
| 1.0 s | `data/ds_vt_ct_1.0_csv.CSV` |

Each row represents an extracted simulated vehicle-conflict event. During loading, numeric variables are normalised and every record receives a headway, integer scenario number, and stable scenario key. The dashboard performs this in `load_data()` in `app/dashboard.py`; the modelling equivalent is implemented in `prepare_model_data()` in `app/ml_modeling.py`.

## 2. Surrogate-safety summaries

The application treats `minTTC` as a surrogate-safety indicator. Users can vary a severe-event threshold between 0.5 and 1.0 seconds. Counts, shares, minimum and mean TTC, speed context, interaction types, fleet compositions, and headway comparisons are calculated from the selected event scope.

These are conditional summaries of extracted simulated conflicts. They are not crash-frequency estimates and should not be interpreted as real-world collision risk.

## 3. Coordinate transformation

SUMO event locations are stored in the network's local metric coordinate system. `local_xy_to_lat_lon()`:

1. applies the recorded SUMO network offset;
2. interprets the corrected coordinates in EPSG:25833;
3. transforms them to EPSG:4326 using `pyproj`; and
4. supplies latitude and longitude to the interactive maps.

Keeping the aggregation in metric coordinates prevents latitude/longitude distortion during distance-based binning and KDE.

## 4. Hotspot overview

`hotspot_table()` groups events into configurable metric spatial cells and calculates event counts, severe-event counts, minimum TTC, speed context, and representative coordinates. `hotspot_map_points()` converts the resulting cells for display. `hotspot_pydeck_chart()` renders the overview over a light CARTO/OpenStreetMap basemap and supports hotspot selection.

The overview is adaptive: scenario, desired headway, and TTC threshold filters are applied before aggregation.

## 5. Standardised kernel density estimation

`scripts/generate_kde_maps.py` provides the reproducible publication-oriented KDE workflow:

1. filter extracted events to the chosen TTC ceiling;
2. retain ego-event positions in SUMO metric coordinates;
3. evaluate a two-dimensional Gaussian kernel on a regular 25 m grid;
4. use a fixed 150 m bandwidth for every configuration;
5. convert the probability density to an estimated event count per standardised 150 m-radius neighbourhood;
6. keep common spatial bounds so configuration surfaces are comparable; and
7. rank peaks from the mean configuration surface while enforcing 750 m minimum separation.

The same bandwidth and neighbourhood definition are used across scenarios and headways. KDE values therefore describe smoothed absolute event concentration, not exposure-normalised risk.

## 6. Local 3D conflict landscape

`build_local_3d_conflict_bins()` selects events within a configurable radius of the chosen hotspot, aggregates them into metric cells, calculates severe-event counts and relative intensity, converts cell footprints to geographic polygons, and creates readable tooltip metadata.

`local_conflict_landscape_3d_chart()` renders the cells as extruded PyDeck polygons. Bar height represents relative conflict concentration within the selected view. The local hotspot marker, light street basemap, adjustable pitch, bearing, height, and transparency help users inspect the road geometry beneath the bars.

## 7. Whole-network 3D landscape

`build_whole_area_3d_bins()` adapts the spatial bin size to the full filtered study extent. The chart functions provide two complementary views:

- a cumulative Plotly surface for reading network-scale intensity; and
- an interactive PyDeck street-map landscape tied to the real network geography.

The adaptive binning limits rendering cost while retaining a consistent connection to the selected scenario, headway, and TTC threshold.

## 8. Street lens

`load_street_context()` reads a cached, lightweight OpenStreetMap-derived context file. The local and whole-network street-lens functions combine:

- simplified building footprints;
- neutral-grey road context;
- crossings and transit locations;
- recognisable traffic-signal icons;
- mapped water and green features; and
- translucent conflict cylinders.

`whole_network_street_lens_chart()` uses PyDeck `PolygonLayer`, `ColumnLayer`, and icon layers. The conflict columns remain separate from the contextual layers so users can reduce or hide the bars and inspect the mapped streets beneath them.

## 9. LightGBM model modes

`app/ml_modeling.py` defines three feature modes:

- **Microscopic:** event- and vehicle-level variables only.
- **Policy levers:** scenario-level AV penetration, fleet composition, and desired headway variables.
- **Combined:** microscopic and policy variables together.

It supports continuous minTTC regression and selected short-TTC classification. Models can be prepared for each headway or for the merged dataset.

Validation uses scenario-grouped folds so records from the same scenario pathway do not appear in both training and validation within a fold. This is more conservative than a random row split for highly related simulation events.

## 10. SHAP interpretation

Tree SHAP values are calculated on a bounded sample from each prepared LightGBM model. The application stores:

- mean absolute SHAP values for global importance;
- mean signed SHAP values;
- observation-level feature/SHAP pairs for dependence views; and
- grouped validation results.

SHAP explains how a fitted model uses the available predictors. It does not convert predictive associations into causal effects.

## 11. Offline model serving

`scripts/precompute_lightgbm_shap.py` trains all supported task, feature-mode, and headway combinations once. Each artifact is stored in `model_cache/`, and `manifest.json` records source hashes, library versions, requests, and output paths.

At runtime, the dashboard verifies cache freshness and loads the prepared artifact. Public visitors therefore receive immediate results without triggering training cost or server contention.

## 12. Grounded research assistant

The assistant has two response paths:

- prepared questions use deterministic, evidence-bounded answers; and
- custom questions retrieve relevant scenario notes, references, variable definitions, and calculated dataset context before optional OpenAI synthesis.

The API key is server-side. When it is unavailable, the dashboard returns a grounded local response. The public repository intentionally excludes the complete manuscript text.

## 13. Published literature benchmark

`llm/literature_benchmark.json` records the sensitivity-adjusted market-penetration and Safety Improvement Rate points reported by Taheri et al. (2026), together with the DOI, method summary, surrogate-measure distribution, licence, and explicit comparison boundaries.

The dashboard treats this material as a separate evidence layer:

- published SIR is a relative change from the baseline conflict count used in each included study;
- local dashboard counts and severe-conflict shares remain outputs of the Berlin SUMO study;
- the two layers may be compared conceptually where methods are consistent, but are never numerically pooled; and
- the displayed power equation is a transparent reconstruction from rounded published points, not a claim about an exact printed coefficient equation.

The Amir agent exposes a dedicated read-only literature-benchmark tool so it can quote exact published points with the DOI while preserving the separation from local simulation results.
