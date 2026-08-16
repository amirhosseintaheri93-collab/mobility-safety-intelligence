"""Create the deterministic public demonstrator dataset from private source tables."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "0.6": ROOT / "data" / "ds_vt_ct_csv.CSV",
    "0.8": ROOT / "data" / "ds_vt_ct_0.8_csv.CSV",
    "1.0": ROOT / "data" / "ds_vt_ct_1.0_csv.CSV",
}
OUTPUT_PATH = ROOT / "data" / "demo_conflicts.csv"
ROWS_PER_SCENARIO_HEADWAY = 25
RANDOM_STATE = 2026


def main() -> None:
    samples: list[pd.DataFrame] = []
    for tau, path in SOURCE_FILES.items():
        frame = pd.read_csv(path, sep=";")
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")].copy()
        frame["scenario_number"] = pd.to_numeric(
            frame["scenario"], errors="coerce"
        ).round().astype(int)
        frame["tau"] = tau
        scenario_samples = []
        for scenario_number, group in frame.groupby("scenario_number"):
            scenario_samples.append(
                group.sample(
                    n=min(ROWS_PER_SCENARIO_HEADWAY, len(group)),
                    random_state=(
                        RANDOM_STATE + int(scenario_number) + int(float(tau) * 10)
                    ),
                )
            )
        sample = pd.concat(scenario_samples, ignore_index=True)
        samples.append(sample)

    demo = pd.concat(samples, ignore_index=True).sort_values(
        ["tau", "scenario_number", "conflict_time"],
        kind="stable",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Wrote {len(demo):,} rows covering "
        f"{demo[['tau', 'scenario_number']].drop_duplicates().shape[0]} configurations "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
