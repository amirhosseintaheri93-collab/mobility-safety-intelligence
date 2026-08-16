# Variable Definitions — AV Safety Policy Intelligence Platform

## Purpose

This document defines the variables, metrics, safety indicators, and explanatory features used in the AV Safety Policy Intelligence Platform.

The definitions are grounded in the validated SUMO microsimulation and post-processing pipeline used in the Phase I study.

The platform acts only as an explanation and policy-translation layer over these validated outputs.

---

# 1. Scenario Variables

## `scenario_id`

Unique identifier for a simulation configuration.

Example:

```text
S6_tau_0.8
```

---

## `scenario_name`

Human-readable scenario description.

Example:

```text
Mixed traffic — AV46 dominant
```

---

## `mpr`

Automated vehicle market penetration rate (%).

Definition:

Percentage of automated vehicles within the total simulated traffic demand.

Range in current study:

```text
0–100%
```

---

## `hdv_share`

Percentage of human-driven vehicles within total traffic demand.

Definition:

```text
HDV share = 100 − AV market penetration
```

---

## `av12_share`

Percentage of SAE Level 1–2 automated vehicles within total traffic demand.

Definition:

Vehicles representing lower-level driver assistance automation.

Examples may include:

* adaptive cruise control
* lane keeping assistance
* partially automated longitudinal control

---

## `av46_share`

Percentage of SAE Level 4–6 automated vehicles within total traffic demand.

Definition:

Vehicles representing highly automated or fully automated driving systems.

---

## `av46_within_av_fleet`

Percentage of AV46 vehicles within the automated vehicle fleet only.

Example:

```text
AV fleet = 60%
AV46 share = 20%

AV46 within AV fleet = 33.3%
```

---

## `tau`

Desired time headway parameter (seconds).

Definition:

Car-following time gap used within SUMO vehicle behavior models.

Tested values:

```text
0.6 s
0.8 s
1.0 s
```

Important interpretation:

Lower τ values may increase roadway throughput but can increase severe-conflict risk during mixed-traffic transition phases.

---

# 2. Exposure Variables

## `vkt`

Vehicle kilometers traveled.

Definition:

Total simulated travel distance normalized across the network.

Purpose:

Used for exposure normalization of surrogate safety indicators.

---

## `simulation_duration`

Total simulation runtime.

Unit:

```text
seconds
```

---

## `vehicle_count`

Total number of simulated vehicles.

---

# 3. Surrogate Safety Indicators

## `total_conflicts_per_million_vkt`

Total simulated traffic conflicts normalized by vehicle kilometers traveled.

Definition:

Conflict events identified using TTC-based surrogate safety analysis.

Normalization:

```text
conflicts / million VKT
```

---

## `severe_conflicts_per_million_vkt`

Severe traffic conflicts normalized by vehicle kilometers traveled.

Definition:

Subset of conflict events with TTC values below the severe-conflict threshold defined in the study methodology.

---

## `ttc`

Time-to-collision.

Definition:

Estimated remaining time before collision under current trajectories and speeds if no evasive action occurs.

Unit:

```text
seconds
```

Important note:

TTC is a surrogate safety indicator and does not represent an actual crash occurrence.

---

## `sir_total`

Safety Improvement Ratio.

Definition:

Relative safety improvement compared with the baseline scenario.

General interpretation:

```text
positive value → improvement
negative value → deterioration
```

---

# 4. Kinematic Conflict Metrics

## `mean_speed_at_conflict`

Average operating speed at conflict initiation.

Definition:

Mean vehicle speed at the moment a conflict event is identified.

Important terminology:

The platform uses the term “speed at conflict” rather than “impact speed,” since actual crashes are not modeled.

---

## `mean_speed_at_severe_conflict`

Average operating speed during severe conflict events.

---

## `mean_delta_v_total`

Average relative speed difference between interacting vehicles at conflict initiation.

Definition:

```text
ΔV = |speed_vehicle_A − speed_vehicle_B|
```

---

## `mean_delta_v_severe`

Average relative speed difference for severe conflict events only.

---

# 5. Interaction Variables

## `interaction_pair`

Vehicle interaction classification used in conflict analysis.

Examples:

```text
HDV–HDV
HDV–AV12
HDV–AV46
AV12–AV12
AV12–AV46
AV46–AV46
```

Important terminology:

The platform uses the term “interaction pairs” rather than “impact vehicle pairs.”

---

## `dominant_interaction_pair`

Interaction pair contributing the highest proportion of conflicts within a scenario.

---

# 6. Explainable AI Variables

## `dominant_shap_driver`

Feature with the highest SHAP contribution within the LightGBM model.

Examples:

```text
relative speed difference
traffic density
tau
AV penetration
```

---

## `shap_importance`

Mean absolute SHAP contribution of a feature.

Purpose:

Used to explain model sensitivity and variable influence.

Important note:

SHAP explanations describe model behavior within the tested simulation configurations and should not be interpreted as causal proof.

---

# 7. Spatial Variables

## `hotspot_cluster_id`

Identifier for a spatial hotspot cluster.

---

## `hotspot_summary`

Short textual interpretation of spatial conflict concentration.

Example:

```text
Residual conflicts remain concentrated near major signalized intersections.
```

---

## `conflict_latitude`

Latitude coordinate of conflict event location.

---

## `conflict_longitude`

Longitude coordinate of conflict event location.

---

# 8. Policy Interpretation Variables

## `policy_note`

Short grounded interpretation for planners and decision-makers.

Example:

```text
Increasing τ from 0.6 s to 0.8 s substantially reduces severe conflicts in the tested configurations.
```

---

## `limitation_note`

Scenario-specific methodological limitation.

Example:

```text
Results apply only to the tested network, demand assumptions, and behavioral configurations.
```

---

# 9. Scientific Interpretation Principles

The platform follows these principles:

1. SUMO simulation outputs are the scientific source of truth.
2. The LLM does not generate independent safety estimates.
3. The platform provides grounded post-hoc explanation and policy translation only.
4. Findings are limited to the tested simulation configurations.
5. Surrogate safety indicators should not be interpreted as observed crash outcomes.
6. Spatial hotspots represent simulated conflict concentration rather than observed crash records.

---

# 10. Recommended Citation Language

Recommended wording for generated explanations:

* “Based on the tested simulation configurations…”
* “Within the validated simulation environment…”
* “To the best of our knowledge…”
* “Based on the identified literature…”
* “The results suggest…”
* “The findings indicate…”

Avoid:

* “Proves”
* “Guarantees”
* “Eliminates crashes”
* “Zero hallucination”
* “Impact speed” (unless crashes are explicitly modeled)
