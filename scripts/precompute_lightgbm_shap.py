from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import pandas as pd
import shap
import sklearn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    code_root = (args.code_root or project_root).resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code_root / "app"))

    from ml_modeling import (
        CACHE_SCHEMA_VERSION,
        HEADWAY_SCOPES,
        MODEL_MODES,
        MODEL_TASKS,
        ModelRequest,
        load_precomputed_result,
        model_artifact_path,
        model_request_slug,
        save_precomputed_result,
        train_lightgbm_shap_model,
    )

    dataset_paths = {
        "0.6": project_root / "data" / "ds_vt_ct_csv.CSV",
        "0.8": project_root / "data" / "ds_vt_ct_0.8_csv.CSV",
        "1.0": project_root / "data" / "ds_vt_ct_1.0_csv.CSV",
    }
    frames = []
    for tau, path in dataset_paths.items():
        frame = pd.read_csv(path, sep=";", low_memory=False)
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
        frame["scenario_number"] = frame["scenario"].round().astype(int)
        frame["tau"] = tau
        frames.append(frame)
    conflicts = pd.concat(frames, ignore_index=True)

    sources = {}
    for path in dataset_paths.values():
        stat = path.stat()
        relative_name = path.relative_to(project_root).as_posix()
        sources[relative_name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }

    manifest_path = cache_dir / "manifest.json"
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "combination_count": len(MODEL_TASKS) * len(MODEL_MODES) * len(HEADWAY_SCOPES),
        "sources": sources,
        "libraries": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__,
        },
        "artifacts": {},
    }

    requests = [
        ModelRequest(task, mode, headway)
        for task in MODEL_TASKS
        for mode in MODEL_MODES
        for headway in HEADWAY_SCOPES
    ]
    total = len(requests)
    for index, request in enumerate(requests, start=1):
        slug = model_request_slug(request)
        artifact_path = model_artifact_path(cache_dir, request)
        started = time.perf_counter()
        if artifact_path.exists() and not args.force:
            result = load_precomputed_result(cache_dir, request)
            action = "reused"
        else:
            result = train_lightgbm_shap_model(conflicts, request)
            artifact_path = save_precomputed_result(cache_dir, result)
            action = "trained"
        elapsed = time.perf_counter() - started
        manifest["artifacts"][slug] = {
            "file": artifact_path.name,
            "size": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "task": request.task,
            "model_mode": request.model_mode,
            "headway_scope": request.headway_scope,
            "modeling_rows": result["metadata"]["modeling_rows"],
            "feature_count": result["metadata"]["feature_count"],
            "effective_configurations": result["metadata"]["effective_configurations"],
            "runtime_seconds": round(elapsed, 3),
        }
        write_manifest(manifest_path, manifest)
        print(
            f"[{index:02d}/{total}] {action} {slug} in {elapsed:.1f}s "
            f"({artifact_path.stat().st_size / 1024 / 1024:.1f} MiB)",
            flush=True,
        )

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_artifact_count"] = len(manifest["artifacts"])
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "cache_dir": str(cache_dir),
        "artifacts": len(manifest["artifacts"]),
        "schema_version": CACHE_SCHEMA_VERSION,
    }), flush=True)


if __name__ == "__main__":
    main()
