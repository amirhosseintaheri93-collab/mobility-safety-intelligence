from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except (ImportError, OSError):
    LGBMClassifier = None
    LGBMRegressor = None


FLEET_COMPOSITIONS = {
    1: {"av": 0, "hdv": 100, "av12": 0, "av46": 0},
    2: {"av": 20, "hdv": 80, "av12": 20, "av46": 0},
    3: {"av": 20, "hdv": 80, "av12": 10, "av46": 10},
    4: {"av": 40, "hdv": 60, "av12": 40, "av46": 0},
    5: {"av": 40, "hdv": 60, "av12": 30, "av46": 10},
    6: {"av": 60, "hdv": 40, "av12": 40, "av46": 20},
    7: {"av": 60, "hdv": 40, "av12": 30, "av46": 30},
    8: {"av": 80, "hdv": 20, "av12": 50, "av46": 30},
    9: {"av": 80, "hdv": 20, "av12": 30, "av46": 50},
    10: {"av": 100, "hdv": 0, "av12": 50, "av46": 50},
    11: {"av": 100, "hdv": 0, "av12": 70, "av46": 30},
    12: {"av": 100, "hdv": 0, "av12": 30, "av46": 70},
}

MODEL_MODE_DESCRIPTIONS = {
    "Microscopic": (
        "Available event-level variables only: relative speed, ego and foe speed, ego and foe vehicle "
        "class, conflict group, and event coordinates. Vehicle IDs and post-outcome timing fields are excluded."
    ),
    "Policy levers": (
        "Scenario-level study levers only: AV market penetration, AV46 share within the AV fleet, and "
        "AV desired headway. In a single-headway model, headway is fixed and is therefore omitted."
    ),
    "Combined": "The available microscopic variables and policy levers in one model.",
}

TASK_DESCRIPTIONS = {
    "Continuous minTTC": "LightGBM regression of minTTC within the already extracted 0-1.0 s event sample.",
    "Selected short-TTC classifier": (
        "LightGBM classification of minTTC < 0.5 s versus 0.5-1.0 s, conditional on an extracted event."
    ),
}

CACHE_SCHEMA_VERSION = "lightgbm-shap-cache-v1"
MODEL_TASKS = ["Continuous minTTC", "Selected short-TTC classifier"]
MODEL_MODES = ["Microscopic", "Policy levers", "Combined"]
HEADWAY_SCOPES = ["All headways", "0.6 s", "0.8 s", "1.0 s"]

FEATURE_LABELS = {
    "delta_speed_kmh": "Relative speed difference (km/h)",
    "ego_speed_kmh": "Ego speed (km/h)",
    "foe_speed_kmh": "Foe speed (km/h)",
    "ego_pos_x": "Event x coordinate (m)",
    "ego_pos_y": "Event y coordinate (m)",
    "av_market_penetration": "AV market penetration",
    "av46_within_av_share": "AV46 share within AV fleet",
    "av_desired_headway": "AV desired headway (s)",
}


@dataclass(frozen=True)
class ModelRequest:
    task: str
    model_mode: str
    headway_scope: str
    random_state: int = 42


def model_request_slug(request: ModelRequest) -> str:
    task_slug = {
        "Continuous minTTC": "regression",
        "Selected short-TTC classifier": "classifier",
    }[request.task]
    mode_slug = request.model_mode.lower().replace(" ", "-")
    headway_slug = request.headway_scope.lower().replace(" ", "-").replace(".", "p")
    return f"{task_slug}__{mode_slug}__{headway_slug}"


def model_artifact_path(cache_dir: Path, request: ModelRequest) -> Path:
    return Path(cache_dir) / f"{model_request_slug(request)}.joblib"


def save_precomputed_result(cache_dir: Path, result: dict[str, Any]) -> Path:
    request = result["request"]
    if not isinstance(request, ModelRequest):
        raise TypeError("The result does not contain a valid ModelRequest.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_artifact_path(cache_dir, request)
    joblib.dump(result, artifact_path, compress=("gzip", 3))
    return artifact_path


def _request_fields(value: Any) -> tuple[str, str, str, int] | None:
    """Return a stable request signature across equivalent import paths.

    Cached artifacts were originally created while ``ml_modeling`` was imported
    as a top-level module. The dashboard can import the same source as
    ``app.ml_modeling``. Dataclass equality treats those as different classes,
    even though their fields are identical, so compare the serialized fields
    rather than Python class identity.
    """
    if value is None:
        return None
    try:
        return (
            str(value.task),
            str(value.model_mode),
            str(value.headway_scope),
            int(getattr(value, "random_state", 42)),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def load_precomputed_result(cache_dir: Path, request: ModelRequest) -> dict[str, Any]:
    artifact_path = model_artifact_path(cache_dir, request)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Precomputed model artifact is missing: {artifact_path.name}")
    # Older cache files record ``ml_modeling.ModelRequest``. Make that import
    # name resolve to this module even when the dashboard imported it as
    # ``app.ml_modeling``.
    sys.modules.setdefault("ml_modeling", sys.modules[__name__])
    result = joblib.load(artifact_path)
    stored_request = result.get("request")
    if _request_fields(stored_request) != _request_fields(request):
        raise ValueError(f"Artifact request mismatch for {artifact_path.name}.")
    return result


def load_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_source_status(project_root: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        issues.append("cache schema version does not match the dashboard")
    expected_sources = manifest.get("sources", {})
    for relative_name, expected in expected_sources.items():
        source_path = Path(project_root) / relative_name
        if not source_path.exists():
            issues.append(f"source file is missing: {relative_name}")
            continue
        stat = source_path.stat()
        if stat.st_size != expected.get("size"):
            issues.append(f"source size changed: {relative_name}")
        if stat.st_mtime_ns != expected.get("mtime_ns"):
            issues.append(f"source timestamp changed: {relative_name}")
    return not issues, issues


def _add_policy_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    scenario_numbers = enriched["scenario_number"].astype(int)
    enriched["av_market_penetration"] = scenario_numbers.map(
        lambda value: FLEET_COMPOSITIONS[value]["av"] / 100
    )
    enriched["av46_within_av_share"] = scenario_numbers.map(
        lambda value: (
            FLEET_COMPOSITIONS[value]["av46"] / FLEET_COMPOSITIONS[value]["av"]
            if FLEET_COMPOSITIONS[value]["av"]
            else np.nan
        )
    )
    enriched["av_desired_headway"] = pd.to_numeric(enriched["tau"], errors="coerce")
    enriched.loc[enriched["av_market_penetration"].eq(0), "av_desired_headway"] = np.nan
    return enriched


def prepare_model_frame(
    conflicts: pd.DataFrame,
    request: ModelRequest,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    required = {
        "scenario_number",
        "tau",
        "minTTC",
        "delta_speed_kmh",
        "ego_speed_kmh",
        "foe_speed_kmh",
        "ego_vtype",
        "foe_vtype",
        "ego_conflict_type",
        "ego_pos_x",
        "ego_pos_y",
    }
    missing = sorted(required - set(conflicts.columns))
    if missing:
        raise ValueError(f"Modeling data is missing required columns: {', '.join(missing)}")

    working = conflicts.copy()
    working["tau"] = working["tau"].astype(str)
    rows_before_scope = len(working)
    repeated_baseline_rows_removed = 0
    if request.headway_scope == "All headways":
        repeated_baseline = working["scenario_number"].eq(1) & working["tau"].ne("0.6")
        repeated_baseline_rows_removed = int(repeated_baseline.sum())
        working = working.loc[~repeated_baseline].copy()
    else:
        requested_tau = request.headway_scope.replace(" s", "")
        working = working.loc[working["tau"].eq(requested_tau)].copy()

    if working.empty:
        raise ValueError("No rows are available for the selected headway scope.")

    working = _add_policy_features(working)
    microscopic_numeric = [
        "delta_speed_kmh",
        "ego_speed_kmh",
        "foe_speed_kmh",
        "ego_pos_x",
        "ego_pos_y",
    ]
    policy_numeric = ["av_market_penetration", "av46_within_av_share"]
    if request.headway_scope == "All headways":
        policy_numeric.append("av_desired_headway")

    working["ego_vehicle_class"] = working["ego_vtype"].map(
        {"DefaultVehicle": "HDV", "F2": "AV12", "F4": "AV46"}
    ).fillna("Other")
    working["foe_vehicle_class"] = working["foe_vtype"].map(
        {"DefaultVehicle": "HDV", "F2": "AV12", "F4": "AV46"}
    ).fillna("Other")
    working["conflict_group"] = np.where(
        pd.to_numeric(working["ego_conflict_type"], errors="coerce").eq(2),
        "Rear-end / following",
        "Other conflict types",
    )
    microscopic_categorical = ["ego_vehicle_class", "foe_vehicle_class", "conflict_group"]

    if request.model_mode == "Microscopic":
        numeric_features = microscopic_numeric
        categorical_features = microscopic_categorical
    elif request.model_mode == "Policy levers":
        numeric_features = policy_numeric
        categorical_features = []
    elif request.model_mode == "Combined":
        numeric_features = microscopic_numeric + policy_numeric
        categorical_features = microscopic_categorical
    else:
        raise ValueError(f"Unknown model mode: {request.model_mode}")

    model_frame = working[numeric_features].apply(pd.to_numeric, errors="coerce")
    if categorical_features:
        categorical_frame = pd.get_dummies(
            working[categorical_features].astype("string").fillna("Missing"),
            prefix={
                "ego_vehicle_class": "Ego vehicle",
                "foe_vehicle_class": "Foe vehicle",
                "conflict_group": "Conflict",
            },
            prefix_sep=" = ",
            dtype=float,
        )
        model_frame = pd.concat([model_frame, categorical_frame], axis=1)

    model_frame = model_frame.rename(columns=FEATURE_LABELS).astype(float)
    valid_rows = model_frame.notna().any(axis=1) & working["minTTC"].notna()
    model_frame = model_frame.loc[valid_rows].reset_index(drop=True)
    modeling_rows = working.loc[valid_rows].reset_index(drop=True)
    groups = modeling_rows["scenario_number"].astype(int)

    if request.task == "Continuous minTTC":
        target = pd.to_numeric(modeling_rows["minTTC"], errors="coerce").astype(float)
    elif request.task == "Selected short-TTC classifier":
        target = modeling_rows["minTTC"].lt(0.5).astype(int)
    else:
        raise ValueError(f"Unknown model task: {request.task}")

    metadata = {
        "rows_before_scope": rows_before_scope,
        "modeling_rows": len(model_frame),
        "repeated_baseline_rows_removed": repeated_baseline_rows_removed,
        "effective_configurations": int(
            modeling_rows[["scenario_number", "tau"]].drop_duplicates().shape[0]
        ),
        "scenario_groups": int(groups.nunique()),
        "feature_count": int(model_frame.shape[1]),
        "selected_short_ttc_events": int(modeling_rows["minTTC"].lt(0.5).sum()),
        "selected_short_ttc_share": float(modeling_rows["minTTC"].lt(0.5).mean()),
        "headway_fixed": request.headway_scope != "All headways",
        "feature_names": list(model_frame.columns),
    }
    return model_frame, target, groups, metadata


def _build_estimator(request: ModelRequest):
    if LGBMClassifier is None or LGBMRegressor is None:
        raise RuntimeError(
            "LightGBM training requires a working LightGBM/OpenMP installation. "
            "Precomputed dashboard results can still be loaded without it."
        )
    common = {
        "n_estimators": 320,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 120,
        "colsample_bytree": 0.85,
        "subsample": 0.85,
        "reg_lambda": 1.0,
        "random_state": request.random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if request.task == "Continuous minTTC":
        return LGBMRegressor(objective="regression_l1", **common)
    return LGBMClassifier(objective="binary", **common)


def _cross_validate(
    estimator,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    request: ModelRequest,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_splits = min(6, int(groups.nunique()))
    if request.task == "Selected short-TTC classifier":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=request.random_state,
        )
        splits = splitter.split(features, target, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(features, target, groups)

    out_of_fold = np.full(len(features), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        fold_model = clone(estimator)
        fold_model.fit(features.iloc[train_index], target.iloc[train_index])
        if request.task == "Continuous minTTC":
            predictions = fold_model.predict(features.iloc[test_index])
            predictions = np.clip(predictions, 0.0, 1.0)
            out_of_fold[test_index] = predictions
            fold_rows.append(
                {
                    "Fold": fold,
                    "MAE": mean_absolute_error(target.iloc[test_index], predictions),
                    "RMSE": mean_squared_error(target.iloc[test_index], predictions) ** 0.5,
                    "R2": r2_score(target.iloc[test_index], predictions),
                    "Test rows": len(test_index),
                }
            )
        else:
            probabilities = fold_model.predict_proba(features.iloc[test_index])[:, 1]
            out_of_fold[test_index] = probabilities
            binary_predictions = (probabilities >= 0.5).astype(int)
            fold_rows.append(
                {
                    "Fold": fold,
                    "ROC-AUC": roc_auc_score(target.iloc[test_index], probabilities),
                    "PR-AUC": average_precision_score(target.iloc[test_index], probabilities),
                    "Balanced accuracy": balanced_accuracy_score(
                        target.iloc[test_index], binary_predictions
                    ),
                    "Brier score": brier_score_loss(target.iloc[test_index], probabilities),
                    "Test rows": len(test_index),
                }
            )
    return pd.DataFrame(fold_rows), out_of_fold


def _summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in fold_metrics.columns if column not in {"Fold", "Test rows"}]
    return pd.DataFrame(
        {
            "Metric": numeric_columns,
            "Mean": [float(fold_metrics[column].mean()) for column in numeric_columns],
            "Std. dev.": [float(fold_metrics[column].std(ddof=1)) for column in numeric_columns],
        }
    )


def _compute_shap_outputs(
    model,
    features: pd.DataFrame,
    request: ModelRequest,
    max_rows: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import shap

    sample = features.sample(min(max_rows, len(features)), random_state=request.random_state)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]
    if shap_array.shape != sample.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_array.shape}; expected {sample.shape}."
        )

    importance = pd.DataFrame(
        {
            "Feature": sample.columns,
            "Mean absolute SHAP": np.abs(shap_array).mean(axis=0),
            "Mean signed SHAP": shap_array.mean(axis=0),
        }
    ).sort_values("Mean absolute SHAP", ascending=False, ignore_index=True)

    top_features = importance.head(10)["Feature"].tolist()
    long_rows: list[pd.DataFrame] = []
    for feature in top_features:
        feature_index = sample.columns.get_loc(feature)
        feature_frame = pd.DataFrame(
            {
                "Feature": feature,
                "Feature value": sample[feature].to_numpy(),
                "SHAP value": shap_array[:, feature_index],
            }
        )
        long_rows.append(feature_frame)
    shap_detail = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    return importance, shap_detail


def train_lightgbm_shap_model(
    conflicts: pd.DataFrame,
    request: ModelRequest,
) -> dict[str, Any]:
    features, target, groups, metadata = prepare_model_frame(conflicts, request)
    estimator = _build_estimator(request)
    fold_metrics, out_of_fold = _cross_validate(estimator, features, target, groups, request)
    estimator.fit(features, target)
    shap_importance, shap_detail = _compute_shap_outputs(estimator, features, request)

    prediction_sample = pd.DataFrame(
        {
            "Observed": target,
            "Predicted": out_of_fold,
            "Scenario": groups.map(lambda value: f"S{int(value)}"),
        }
    )
    if len(prediction_sample) > 6000:
        prediction_sample = prediction_sample.sample(6000, random_state=request.random_state)

    return {
        "request": request,
        "model": estimator,
        "metadata": metadata,
        "metric_summary": _summarize_metrics(fold_metrics),
        "fold_metrics": fold_metrics,
        "prediction_sample": prediction_sample.reset_index(drop=True),
        "shap_importance": shap_importance,
        "shap_detail": shap_detail,
    }
