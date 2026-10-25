# Explainable AI Intrusion Detection System

This project aims to build an intrusion detection system using the CIC-IDS2017
network traffic dataset, an MLP classifier, and LIME and SHAP explanations. A
Streamlit dashboard is planned for a later module.

## Module 1: Project and Dataset Setup

### Create the Python environment

From the project root in PowerShell, create a virtual environment with Python
3.12 and install the Module 1 dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Download the dataset

The CIC-IDS2017 dataset is available from the [official CIC-IDS2017 dataset
page](https://www.unb.ca/cic/datasets/ids-2017.html). Download the dataset
manually, extract the CSV files, and place them in `data/raw/`. Dataset files
are not included in this project and are not downloaded automatically.

## Project directories

- `data/raw/`: manually downloaded CIC-IDS2017 files
- `models/`: generated model artifacts
- `outputs/`: generated reports and other outputs
- `src/`: project source code
- `tests/`: automated tests

### Inspect the dataset and run tests

After placing CSV files in `data/raw/`, inspect them from the project root:

```powershell
.\.venv\Scripts\python.exe src\inspect_dataset.py
```

Run the automated tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The inspection script also accepts `--data-dir` to inspect another directory
and `--chunk-size` to set the number of rows processed at a time. It does not
modify input CSV files.

## Module 2: Preprocessing

After installing the declared requirements and placing the raw CSVs in
`data/raw/`, create preprocessing artifacts with:

```powershell
.\.venv\Scripts\python.exe src\preprocess_dataset.py `
  --min-free-space-gib USER_CHOSEN_THRESHOLD
```

The preprocessing step validates and removes only the verified duplicate
`Fwd Header Length` column, preserves original labels and source-row metadata,
maps labels to binary targets, reports missing/infinite values, and stops on
nonnumeric features or unsafe group-aware splits. Exact feature groups are
kept together in deterministic 70/15/15 train/validation/test partitions
(seed 42). Median imputation and `StandardScaler` are fitted on training data
only. Numeric constant features are retained and listed in the split manifest.

Rows, canonical feature-group keys, group statistics, and partition assignments
are staged in a temporary SQLite database beneath the preprocessing output
directory. CSV ingestion, scaler fitting, and transformed partition writing use
bounded batches; the training median is computed from group-weighted training
values. The database is removed before artifacts are published.

Each successful run creates a new run directory beneath
`outputs/preprocessing/`, containing transformed CSV partitions, a reusable
pickled scikit-learn preprocessing pipeline, ordered feature names, target
mapping, and a split/quality manifest. Existing runs are not overwritten.
The custom imputer is defined in `src/preprocessing_components.py` so newly
saved pipeline pickles can be loaded by a separate process with the project
root on its Python import path. Existing pickles created when the preprocessing
script was executed directly may refer to `__main__.PrecomputedMedianImputer`;
those artifacts must be regenerated to obtain the portable module path.
The command accepts `--data-dir`, `--output-dir`, and `--chunk-size`. It
requires `--min-free-space-gib`, a user-selected minimum-free-space start
threshold checked on the output volume before staging begins. This is a
preflight gate, not a prediction or guarantee of the total space the run needs.
Observed pilot staging sizes and free-space changes can inform a later
full-dataset threshold; no expansion ratio is assumed.

For a controlled subset pilot, use `--pilot` with an explicit strict subset of
relative CSV paths and a separate output directory (default
`outputs/preprocessing-pilot/`). For example, replace the placeholder paths
and `USER_CHOSEN_THRESHOLD` with values selected for the machine:

```powershell
.\.venv\Scripts\python.exe src\preprocess_dataset.py `
  --pilot `
  --files "relative\first.csv" "relative\second.csv" `
  --chunk-size 10000 `
  --max-rows-per-file 50000 `
  --min-free-space-gib USER_CHOSEN_THRESHOLD
```

`--max-rows-per-file` limits ingestion to the first N data rows of each
selected CSV in source order; this is deterministic but may be biased. Selecting
files is preferable when broader coverage is needed. Duplicate-column
verification still scans each selected file before ingestion, even when a row
cap is set. Pilot mode rejects the normal `outputs/preprocessing/` destination
and selecting all CSV files. It does not copy or modify raw data.

Pilot runs log stage/file progress, cumulative ingested rows, elapsed time,
disk-space preflight, and error context. They also save `pilot_metrics.json`
with Windows process peak-memory counters and staging-size/free-space samples
at stage boundaries. Memory counters are process-level peaks; staging samples
can miss short-lived SQLite temporary or journal peaks. These are measurements
of the selected subset only, not guarantees or evidence of full-dataset
scalability. A subset can also fail the existing group/class split constraints.
Existing staging directories are never automatically removed; failure cleanup
is limited to the staging directory created by the current run.

Preprocessing does not train a classifier. Synthetic tests cover the
SQLite-backed implementation; full-dataset execution time, peak memory, and
disk requirements have not yet been validated.


Drive link for preprocessed data : https://drive.google.com/drive/u/2/home?hl=en_GB
https://drive.google.com/file/d/1vCZwBLXc05_SrW9dIXbUavHdWtguBzE4/view?usp=drive_link
https://drive.google.com/file/d/1xuyCF3eQoLJDz30YK8MY46rHl9l4qjBS/view?usp=drive_link
https://drive.google.com/file/d/1Rt6YGOV7EZCrKqhSCa-fnyEBLmA48wBn/view?usp=drive_link
https://drive.google.com/file/d/19m55Y4z5xlsikOhXyBSRwcNcA9BA8FpY/view?usp=drive_link
https://drive.google.com/file/d/1IGxb6J_6YQhfZmEGKv9nfyMPNgukyFDP/view?usp=drive_link
https://drive.google.com/file/d/1hDsl6WVvb0Vc2cNcd0PPfwoUAjiUY9Rm/view?usp=drive_link
https://drive.google.com/file/d/1szlN46srtNDsmqdTux1yl0uhXn46Q01g/view?usp=drive_link
https://drive.google.com/file/d/12DbMcxnAQTGJbYtZAIf4NArzRJLJ0AcZ/view?usp=drive_link
