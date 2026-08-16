# Public demonstrator dataset

`demo_conflicts.csv` is a deterministic stratified sample of 25 simulated conflict-event records from each of the 12 scenarios at each of the three desired headways. It contains 900 rows across 36 configurations.

The sample is provided to document the schema, support interface inspection, and make the public repository reviewable without releasing the complete research event tables. It must not be used as a substitute for the complete dataset or represented as sufficient to reproduce the paper's numerical conclusions.

The demonstrator is licensed under CC BY 4.0. Use the attribution statement in `DATA_LICENSE.md` and cite the associated SSRN preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7025301.

The deterministic creation procedure is in `scripts/create_demo_dataset.py`.
