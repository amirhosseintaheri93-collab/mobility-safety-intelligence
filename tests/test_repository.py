import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_headway_datasets_have_required_columns(self) -> None:
        required = {"scenario", "minTTC", "ego_pos_x", "ego_pos_y"}
        paths = [
            ROOT / "data" / "ds_vt_ct_csv.CSV",
            ROOT / "data" / "ds_vt_ct_0.8_csv.CSV",
            ROOT / "data" / "ds_vt_ct_1.0_csv.CSV",
        ]
        for path in paths:
            if not path.exists():
                continue
            columns = set(pd.read_csv(path, sep=";", nrows=2).columns)
            self.assertTrue(required.issubset(columns), path.name)

    def test_public_demo_covers_all_scenarios_and_headways(self) -> None:
        demo = pd.read_csv(ROOT / "data" / "demo_conflicts.csv")
        required = {
            "scenario",
            "scenario_number",
            "tau",
            "minTTC",
            "ego_pos_x",
            "ego_pos_y",
        }
        self.assertTrue(required.issubset(demo.columns))
        self.assertEqual(sorted(demo["scenario_number"].unique().tolist()), list(range(1, 13)))
        self.assertEqual(sorted(demo["tau"].astype(str).unique().tolist()), ["0.6", "0.8", "1.0"])

    def test_precomputed_model_manifest_is_present(self) -> None:
        manifest_path = ROOT / "model_cache" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "lightgbm-shap-cache-v1")
        self.assertTrue(manifest.get("artifacts"))

    def test_all_precomputed_model_requests_load_across_import_paths(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT))
        try:
            from app.ml_modeling import ModelRequest, load_precomputed_result

            manifest = json.loads(
                (ROOT / "model_cache" / "manifest.json").read_text(encoding="utf-8")
            )
            for artifact in manifest["artifacts"].values():
                request = ModelRequest(
                    artifact["task"],
                    artifact["model_mode"],
                    artifact["headway_scope"],
                )
                result = load_precomputed_result(ROOT / "model_cache", request)
                self.assertIn("metadata", result, artifact["file"])
        finally:
            sys.path.pop(0)

    def test_private_files_are_not_in_release(self) -> None:
        forbidden = [
            ROOT / ".streamlit" / "secrets.toml",
            ROOT / ".env",
            ROOT / "llm" / "revised_manuscript_2026.txt",
            ROOT / "llm" / "manuscript_evidence.json",
        ]
        self.assertFalse(any(path.exists() for path in forbidden))

    def test_literature_benchmark_is_attributed_and_bounded(self) -> None:
        benchmark = json.loads(
            (ROOT / "llm" / "literature_benchmark.json").read_text(encoding="utf-8")
        )
        points = benchmark["published_adjusted_points"]
        self.assertEqual(benchmark["doi"], "10.1186/s12544-026-00774-9")
        self.assertEqual(benchmark["license"], "Creative Commons Attribution 4.0 International")
        self.assertEqual(len(points), 9)
        self.assertEqual(points[0], {"mpr_percent": 10, "sir_percent": 2.8})
        self.assertEqual(points[-1], {"mpr_percent": 90, "sir_percent": 39.4})
        self.assertTrue(benchmark["comparison_boundaries"])


if __name__ == "__main__":
    unittest.main()
