import json
import unittest
from pathlib import Path
import wave

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

    def test_local_study_table_matches_reported_sir(self) -> None:
        rates = pd.read_csv(ROOT / "data" / "local_study_conflict_rates.csv")
        required = {
            "scenario_number",
            "av_percent",
            "total_conflicts_per_million_vkt",
            "severe_conflicts_per_million_vkt",
            "sir_total_percent",
        }
        self.assertTrue(required.issubset(rates.columns))
        self.assertEqual(rates["scenario_number"].tolist(), list(range(1, 13)))

        baseline = int(
            rates.loc[
                rates["scenario_number"].eq(1),
                "total_conflicts_per_million_vkt",
            ].iloc[0]
        )
        self.assertEqual(baseline, 95_385)
        expected_sir = [0.0, -0.3, -1.1, 0.5, 2.6, 7.8, 5.7, 20.4, 20.4, 44.4, 42.2, 44.6]
        self.assertEqual(rates["sir_total_percent"].tolist(), expected_sir)

        calculated = (
            (baseline - rates["total_conflicts_per_million_vkt"]) / baseline * 100
        ).round(1)
        pd.testing.assert_series_equal(
            calculated,
            rates["sir_total_percent"],
            check_names=False,
        )

    def test_entry_game_uses_bounded_surrogate_safety_language(self) -> None:
        dashboard = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
        for overclaim in (
            "how much safer this kind of network could become",
            "See the safety improvement",
            "safer than the baseline",
        ):
            self.assertNotIn(overclaim, dashboard)
        self.assertIn(
            "TTC-based surrogate evidence from prepared simulations; not observed crashes or a causal safety effect.",
            dashboard,
        )

    def test_original_audio_assets_are_valid_and_optional(self) -> None:
        audio_dir = ROOT / "assets" / "audio"
        expected = {
            "setar-inspired-acoustic.wav": 10.0,
            "setar-inspired-electronic.wav": 10.0,
            "ui-select.wav": 0.1,
            "ui-whoosh.wav": 0.3,
            "ui-reveal.wav": 0.5,
        }
        for name, minimum_seconds in expected.items():
            path = audio_dir / name
            self.assertTrue(path.exists(), name)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1, name)
                self.assertEqual(audio.getsampwidth(), 2, name)
                duration = audio.getnframes() / audio.getframerate()
                self.assertGreaterEqual(duration, minimum_seconds, name)

        dashboard = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn('"Off": None', dashboard)
        self.assertIn("Game clicks and reveals", dashboard)
        self.assertIn("Original audio made for this app", dashboard)


if __name__ == "__main__":
    unittest.main()
