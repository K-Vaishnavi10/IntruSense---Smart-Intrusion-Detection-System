import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd

import src.preprocess_dataset as preprocessing
from src.preprocess_dataset import (
    DataQualityReport,
    ExactFeatureGrouper,
    PreprocessingValidationError,
    encode_binary_label,
    iter_dataset_chunks,
    main,
    run_preprocessing,
    split_feature_groups,
    validate_data_quality,
    validate_header,
    validate_label_text,
    verify_duplicate_columns,
)


def test_known_duplicate_header_is_verified_and_second_copy_is_excluded(tmp_path):
    path = tmp_path / "traffic.csv"
    path.write_text(
        " Flow, Fwd Header Length, Fwd Header Length, Label\n"
        "1,40,40,BENIGN\n"
        "2,64,64,Attack\n",
        encoding="utf-8",
    )

    header = validate_header(
        (" Flow", " Fwd Header Length", " Fwd Header Length", " Label")
    )
    assert header.duplicate_column_pairs == ((1, 2),)

    verified = verify_duplicate_columns(path, header)

    assert verified.retained_columns == ("Flow", "Fwd Header Length", "Label")
    assert verified.feature_columns == ("Flow", "Fwd Header Length")
    assert verified.label_index == 3
    assert verified.row_count == 2


def test_duplicate_column_mismatch_is_rejected_without_dropping_the_row(tmp_path):
    path = tmp_path / "mismatch.csv"
    path.write_text(
        "Fwd Header Length,Fwd Header Length,Label\n"
        "40,41,Attack\n",
        encoding="utf-8",
    )
    header = validate_header(
        ("Fwd Header Length", "Fwd Header Length", "Label")
    )

    with pytest.raises(PreprocessingValidationError, match="record 2.*different values"):
        verify_duplicate_columns(path, header)


@pytest.mark.parametrize(
    "columns",
    [
        ("Flow", " Flow ", "Label"),
        ("Flow", "flow", "Label"),
        ("Fwd Header Length", " Fwd Header Length", " Fwd Header Length", "Label"),
        ("Flow", "Label", " label "),
        ("Flow", "   ", "Label"),
    ],
)
def test_unsafe_or_unexpected_header_collisions_are_rejected(columns):
    with pytest.raises(PreprocessingValidationError):
        validate_header(columns)


def test_header_requires_exactly_one_label_column():
    with pytest.raises(PreprocessingValidationError, match="exactly one 'Label'"):
        validate_header(("Flow", "Bytes"))


@pytest.mark.parametrize("label", ["BENIGN", " benign ", "BeNiGn"])
def test_benign_label_maps_to_zero(label):
    assert encode_binary_label(label) == 0


@pytest.mark.parametrize(
    "label",
    ["DDoS", "Web Attack � Brute Force", "Web Attack � Sql Injection", "Web Attack � XSS"],
)
def test_attack_label_is_preserved_as_attack_without_repair(label):
    assert validate_label_text(label) == label
    assert encode_binary_label(label) == 1


@pytest.mark.parametrize("label", [None, "", "   "])
def test_missing_or_blank_label_is_rejected(label):
    with pytest.raises(PreprocessingValidationError, match="missing or blank"):
        encode_binary_label(label)


def test_chunked_ingestion_preserves_order_labels_and_reports_quality(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "b.csv").write_text(
        "Feature,Fwd Header Length,Fwd Header Length,Label\n"
        "3,30,30,Attack\n",
        encoding="utf-8",
    )
    (raw / "a.csv").write_text(
        "Feature,Fwd Header Length,Fwd Header Length,Label\n"
        "1,10,10, benign \n"
        "2,20,20,DDoS\n"
        ",25,25,BENIGN\n",
        encoding="utf-8",
    )
    report = DataQualityReport()

    chunks = list(iter_dataset_chunks(raw, chunk_size=2, report=report))

    assert [chunk.source_file for chunk in chunks] == ["a.csv", "a.csv", "b.csv"]
    assert [chunk.source_row_numbers for chunk in chunks] == [(1, 2), (3,), (1,)]
    values = [value for chunk in chunks for value in chunk.features["Feature"]]
    assert values[:2] == [1, 2]
    assert pd.isna(values[2])
    assert values[3] == 3
    assert [value for chunk in chunks for value in chunk.target] == [0, 1, 0, 1]
    assert [value for chunk in chunks for value in chunk.label_original] == [
        " benign ", "DDoS", "BENIGN", "Attack"
    ]
    assert report.row_count == 4
    assert report.files == ["a.csv", "b.csv"]
    assert report.original_missing == {"Feature": 1}


def test_chunked_ingestion_counts_infinities_nonnumeric_and_constant_features(tmp_path):
    (tmp_path / "values.csv").write_text(
        "Constant,Value,Label\n"
        "7,inf,BENIGN\n"
        "7,-inf,Attack\n"
        "7,not-a-number,BENIGN\n",
        encoding="utf-8",
    )
    report = DataQualityReport()

    chunks = list(iter_dataset_chunks(tmp_path, chunk_size=2, report=report))

    assert report.positive_infinity == {"Value": 1}
    assert report.negative_infinity == {"Value": 1}
    assert report.nonnumeric_values == {"Value": 1}
    assert report.constant_columns == ("Constant",)
    assert chunks[0].features["Value"].isna().tolist() == [True, True]
    with pytest.raises(PreprocessingValidationError, match="nonnumeric"):
        validate_data_quality(report)


def test_exact_grouping_detects_conflicting_binary_labels_across_chunks():
    grouper = ExactFeatureGrouper(("bytes",))
    first = grouper.add(
        pd.DataFrame({"bytes": [10.0, 20.0]}),
        [0, 1],
        ["BENIGN", "DDoS"],
    )
    second = grouper.add(
        pd.DataFrame({"bytes": [10.0, 30.0]}),
        [1, 1],
        ["Attack", "PortScan"],
    )

    assert first.tolist() == [0, 1]
    assert second.tolist() == [0, 2]
    assert grouper.groups[0].row_count == 2
    assert grouper.conflicting_groups == {0: ("Attack", "BENIGN")}


def test_exact_grouping_checks_canonical_equality_after_digest_collision(monkeypatch):
    monkeypatch.setattr(preprocessing, "_feature_digest", lambda _: b"same digest")
    grouper = ExactFeatureGrouper(("bytes",))

    assignments = grouper.add(
        pd.DataFrame({"bytes": [1.0, 2.0, 1.0]}),
        [0, 1, 0],
        ["BENIGN", "Attack", "BENIGN"],
    )

    assert assignments.tolist() == [0, 1, 0]
    assert len(grouper.groups) == 2


def test_exact_grouping_preserves_high_precision_integer_distinctions_across_chunks():
    grouper = ExactFeatureGrouper(("bytes",))
    first = grouper.add(
        pd.DataFrame({"bytes": pd.Series([9007199254740992], dtype="int64")}),
        [0],
        ["BENIGN"],
    )
    second = grouper.add(
        pd.DataFrame(
            {"bytes": pd.Series([9007199254740993, 9007199254740992], dtype="int64")}
        ),
        [1, 0],
        ["Attack", "BENIGN"],
    )
    same_numeric_value_other_dtype = grouper.add(
        pd.DataFrame({"bytes": pd.Series([9007199254740992.0], dtype="float64")}),
        [0],
        ["BENIGN"],
    )

    assert first.tolist() == [0]
    assert second.tolist() == [1, 0]
    assert same_numeric_value_other_dtype.tolist() == [0]
    assert len(grouper.groups) == 2


def test_group_split_is_deterministic_and_keeps_groups_in_one_partition():
    grouper = ExactFeatureGrouper(("flow",))
    grouper.add(
        pd.DataFrame({"flow": list(map(float, range(60)))}),
        [index % 2 for index in range(60)],
        ["BENIGN" if index % 2 == 0 else "Attack" for index in range(60)],
    )

    split = split_feature_groups(grouper, seed=42)
    repeated = split_feature_groups(grouper, seed=42)

    assert split == repeated
    assert set(split.group_to_partition) == set(grouper.groups)
    assert sum(split.row_counts.values()) == 60
    assert set(split.row_counts) == {"train", "validation", "test"}
    assert all(count > 0 for counts in split.class_counts.values() for count in counts.values())
    flattened = [
        group_id
        for group_ids in split.partition_group_ids.values()
        for group_id in group_ids
    ]
    assert len(flattened) == len(set(flattened)) == len(grouper.groups)


def test_group_split_seeds_class_coverage_before_greedy_size_balancing():
    group_sizes_and_labels = [
        (12, [1] * 12),
        (9,  [0] * 9),
        (19, [1] * 19),
        (11, [0] * 11),
        (12, [0] * 12),
        (2,  [1, 1]),
    ]
    values: list[float] = []
    targets: list[int] = []
    labels: list[str] = []
    for group_id, (row_count, group_targets) in enumerate(group_sizes_and_labels):
        values.extend([float(group_id)] * row_count)
        targets.extend(group_targets)
        labels.extend("BENIGN" if target == 0 else "Attack" for target in group_targets)
    grouper = ExactFeatureGrouper(("flow",))
    grouper.add(pd.DataFrame({"flow": values}), targets, labels)

    split = split_feature_groups(grouper, seed=42)
    repeated = split_feature_groups(grouper, seed=42)

    assert split == repeated
    assert set(split.row_counts) == {"train", "validation", "test"}
    assert all(
        split.class_counts[partition][label] > 0
        for partition in split.class_counts
        for label in (0, 1)
    )
    assert sum(split.row_counts.values()) == len(values)
    assert len({
        split.group_to_partition[group_id] for group_id in grouper.groups
    }) == 3


def test_group_split_fails_when_group_isolation_cannot_cover_both_classes():
    grouper = ExactFeatureGrouper(("flow",))
    grouper.add(
        pd.DataFrame({"flow": [1.0, 2.0, 3.0, 4.0]}),
        [0, 0, 0, 1],
        ["BENIGN", "BENIGN", "BENIGN", "Attack"],
    )

    with pytest.raises(PreprocessingValidationError, match="fewer than three"):
        split_feature_groups(grouper)


def test_pipeline_fits_training_only_and_saves_validated_artifacts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    values = list(range(1, 31))
    values[3] = ""
    rows = [
        f"{value},9,{'BENIGN' if index % 2 == 0 else 'Attack'}"
        for index, value in enumerate(values)
    ]
    source = raw / "traffic.csv"
    source.write_text(
        "Feature,Constant,Label\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "preprocessing"

    run_directory = run_preprocessing(raw, output, chunk_size=7)

    assert sorted(path.name for path in run_directory.iterdir()) == [
        "feature_names.json",
        "preprocessing_pipeline.pkl",
        "split_manifest.json",
        "target_mapping.json",
        "test.csv",
        "train.csv",
        "validation.csv",
    ]
    feature_names = json.loads(
        (run_directory / "feature_names.json").read_text(encoding="utf-8")
    )
    assert feature_names == ["Feature", "Constant"]
    train = pd.read_csv(run_directory / "train.csv")
    with (run_directory / "preprocessing_pipeline.pkl").open("rb") as pipeline_file:
        pipeline = pickle.load(pipeline_file)
    training_source_values = [
        float(values[int(row_number) - 1])
        for row_number in train["source_row_number"]
        if values[int(row_number) - 1] != ""
    ]
    assert pipeline.named_steps["imputer"].statistics_[0] == pytest.approx(
        pd.Series(training_source_values).median()
    )
    assert pipeline.transform(pd.DataFrame({"Feature": [1.0], "Constant": [9.0]})).shape == (
        1,
        2,
    )
    with pytest.raises(PreprocessingValidationError, match="feature names or order"):
        pipeline.transform(pd.DataFrame({"Constant": [9.0], "Feature": [1.0]}))
    assert "label_original" in train.columns
    assert "source_file" in train.columns
    manifest = json.loads(
        (run_directory / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seed"] == 42
    assert sum(manifest["actual_rows"].values()) == len(values)
    assert all(
        set(class_counts) == {"0", "1"} and all(class_counts.values())
        for class_counts in manifest["binary_class_counts"].values()
    )
    assert manifest["quality"]["original_missing"] == {"Feature": 1}
    assert manifest["quality"]["constant_columns"] == ["Constant"]
    assert source.read_text(encoding="utf-8").startswith("Feature,Constant,Label\n")


def test_saved_pipeline_loads_in_separate_process_with_77_ordered_features(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    feature_names = [f"Feature_{index:02d}" for index in range(77)]
    source = raw / "traffic.csv"
    rows = []
    for row_index in range(60):
        values = [
            str(row_index * 100 + feature_index)
            for feature_index in range(len(feature_names))
        ]
        label = "BENIGN" if row_index % 2 == 0 else "Attack"
        rows.append(",".join([*values, label]))
    source.write_text(
        ",".join([*feature_names, "Label"]) + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    output_directory = tmp_path / "outputs" / "preprocessing"
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "src" / "preprocess_dataset.py"),
            "--data-dir",
            str(raw),
            "--output-dir",
            str(output_directory),
            "--chunk-size",
            "10",
            "--min-free-space-gib",
            "0",
        ],
        cwd=project_root,
        capture_output=True,
        check=True,
        text=True,
    )
    run_directories = list(output_directory.glob("run-*"))
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    pipeline_path = run_directory / "preprocessing_pipeline.pkl"
    feature_names_path = run_directory / "feature_names.json"

    subprocess_code = """
import json
import pickle
import sys
import numpy as np
import pandas as pd

pipeline_path, names_path, source_path = sys.argv[1:]
names = json.load(open(names_path, encoding="utf-8"))
with open(pipeline_path, "rb") as stream:
    pipeline = pickle.load(stream)
frame = pd.read_csv(source_path, nrows=8).loc[:, names]
transformed = pipeline.transform(frame)
print(json.dumps({
    "columns": list(frame.columns),
    "shape": list(transformed.shape),
    "finite": bool(np.isfinite(transformed).all()),
    "class_module": type(pipeline.named_steps["imputer"]).__module__,
}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            subprocess_code,
            str(pipeline_path),
            str(feature_names_path),
            str(source),
        ],
        cwd=project_root,
        capture_output=True,
        check=True,
        text=True,
    )
    loaded_result = json.loads(result.stdout)

    assert loaded_result["columns"] == feature_names
    assert loaded_result["shape"] == [8, 77]
    assert loaded_result["finite"] is True
    assert loaded_result["class_module"] == "src.preprocessing_components"


def test_pipeline_does_not_create_artifacts_when_numeric_validation_fails(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traffic.csv").write_text(
        "Feature,Label\n"
        "1,BENIGN\n"
        "bad,Attack\n"
        "3,BENIGN\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "preprocessing"
    report = DataQualityReport()
    list(iter_dataset_chunks(raw, report=report))
    assert report.nonnumeric_values == {"Feature": 1}

    with pytest.raises(PreprocessingValidationError, match="nonnumeric"):
        run_preprocessing(raw, output)
    assert not output.exists()


def test_sqlite_pipeline_groups_high_precision_duplicates_across_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    base = 9007199254740992
    first_rows = [
        f"{base + index},{'BENIGN' if index % 2 == 0 else 'Attack'}"
        for index in range(12)
    ]
    (raw / "a.csv").write_text(
        "Feature,Label\n" + "\n".join(first_rows) + "\n",
        encoding="utf-8",
    )
    (raw / "b.csv").write_text(
        f"Feature,Label\n{base},Attack\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "preprocessing"

    first_run = run_preprocessing(raw, output, chunk_size=2, seed=42)
    second_run = run_preprocessing(raw, output, chunk_size=3, seed=42)

    first_manifest = json.loads(
        (first_run / "split_manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (second_run / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["group_count"] == 12
    assert first_manifest["conflicting_group_count"] == 1
    assert first_manifest["conflicting_binary_label_groups"] == (
        second_manifest["conflicting_binary_label_groups"]
    )

    def source_partition_map(run_directory):
        mapping = {}
        for partition in ("train", "validation", "test"):
            data = pd.read_csv(run_directory / f"{partition}.csv")
            for row in data.itertuples(index=False):
                mapping[(row.source_file, row.source_row_number)] = partition
        return mapping

    first_partitions = source_partition_map(first_run)
    second_partitions = source_partition_map(second_run)
    assert first_partitions == second_partitions
    assert first_partitions[("a.csv", 1)] == first_partitions[("b.csv", 1)]
    combined = pd.concat(
        [pd.read_csv(first_run / f"{part}.csv") for part in ("train", "validation", "test")],
        ignore_index=True,
    )
    duplicate_labels = combined[
        combined["source_file"].isin(["a.csv", "b.csv"])
        & (combined["source_row_number"] == 1)
    ]["label_original"].tolist()
    assert duplicate_labels == ["BENIGN", "Attack"]


def test_sqlite_pipeline_consumes_each_chunk_before_requesting_the_next(
    tmp_path,
    monkeypatch,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traffic.csv").write_text(
        "Feature,Label\n"
        + "\n".join(
            f"{index},{'BENIGN' if index % 2 == 0 else 'Attack'}"
            for index in range(18)
        )
        + "\n",
        encoding="utf-8",
    )
    original_iter = preprocessing.iter_dataset_chunks
    original_store = preprocessing._store_chunk
    events: list[tuple[str, int]] = []

    def tracked_iter(*args, **kwargs):
        for chunk in original_iter(*args, **kwargs):
            events.append(("yield", len(chunk.features)))
            yield chunk
            assert events[-1][0] == "stored"

    def tracked_store(connection, chunk, feature_columns):
        assert len(chunk.features) <= 2
        original_store(connection, chunk, feature_columns)
        events.append(("stored", len(chunk.features)))

    monkeypatch.setattr(preprocessing, "iter_dataset_chunks", tracked_iter)
    monkeypatch.setattr(preprocessing, "_store_chunk", tracked_store)

    run_preprocessing(raw, tmp_path / "outputs" / "preprocessing", chunk_size=2)

    assert events
    assert all(kind in {"yield", "stored"} for kind, _ in events)
    assert sum(kind == "yield" for kind, _ in events) == sum(
        kind == "stored" for kind, _ in events
    )


def test_sqlite_pipeline_cleans_staging_database_on_late_split_failure(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traffic.csv").write_text(
        "Feature,Label\n1,BENIGN\n2,Attack\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "preprocessing"
    output.mkdir(parents=True)

    with pytest.raises(PreprocessingValidationError, match="every train"):
        run_preprocessing(raw, output, chunk_size=1)

    assert list(output.iterdir()) == []


def test_ingestion_can_select_files_and_cap_rows_without_changing_raw_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = raw / "selected.csv"
    untouched = raw / "untouched.csv"
    selected.write_text(
        "Feature,Label\n"
        + "\n".join(
            f"{index},{'BENIGN' if index % 2 == 0 else 'Attack'}"
            for index in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    untouched.write_text(
        "Feature,Label\n100,BENIGN\n",
        encoding="utf-8",
    )
    original_selected = selected.read_bytes()
    original_untouched = untouched.read_bytes()
    report = DataQualityReport()

    chunks = list(
        iter_dataset_chunks(
            raw,
            chunk_size=2,
            report=report,
            selected_files=["selected.csv"],
            max_rows_per_file=5,
        )
    )

    assert [chunk.source_file for chunk in chunks] == [
        "selected.csv",
        "selected.csv",
        "selected.csv",
    ]
    assert [row for chunk in chunks for row in chunk.source_row_numbers] == [
        1, 2, 3, 4, 5
    ]
    assert report.row_count == 5
    assert report.files == ["selected.csv"]
    assert selected.read_bytes() == original_selected
    assert untouched.read_bytes() == original_untouched


def test_pilot_outputs_measurements_in_separate_directory_and_logs_progress(
    tmp_path,
    caplog,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = raw / "selected.csv"
    selected.write_text(
        "Feature,Label\n"
        + "\n".join(
            f"{index},{'BENIGN' if index % 2 == 0 else 'Attack'}"
            for index in range(30)
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "not-selected.csv").write_text(
        "Feature,Label\n101,BENIGN\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "preprocessing-pilot"

    with caplog.at_level("INFO", logger=preprocessing.__name__):
        run_directory = run_preprocessing(
            raw,
            output,
            chunk_size=10,
            selected_files=["selected.csv"],
            max_rows_per_file=30,
            min_free_space_bytes=0,
            pilot_mode=True,
        )

    assert run_directory.parent == output
    assert (run_directory / "pilot_metrics.json").is_file()
    assert not any(path.name.startswith(".staging-") for path in output.iterdir())
    metrics = json.loads(
        (run_directory / "pilot_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["mode"] == "subset-pilot"
    assert metrics["files"] == ["selected.csv"]
    assert metrics["chunk_size"] == 10
    assert metrics["rows_processed"] == 30
    assert metrics["preflight_free_bytes"] > 0
    assert metrics["measurements"]
    assert all("staging_bytes" in item for item in metrics["measurements"])
    assert "stage=ingestion source_file=selected.csv" in caplog.text
    assert "rows_processed=30" in caplog.text
    assert "elapsed_seconds=" in caplog.text


def test_disk_preflight_refuses_threshold_before_creating_output(
    tmp_path,
    monkeypatch,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "outputs" / "preprocessing"
    monkeypatch.setattr(
        preprocessing.shutil,
        "disk_usage",
        lambda _: type("DiskUsage", (), {"free": 10})(),
    )

    with pytest.raises(PreprocessingValidationError, match="10 bytes available"):
        run_preprocessing(
            raw,
            output,
            min_free_space_bytes=11,
        )

    assert not output.exists()


def test_failure_logs_context_and_preserves_preexisting_staging_data(
    tmp_path,
    caplog,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "traffic.csv").write_text(
        "Feature,Label\n1,BENIGN\n2,Attack\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs" / "preprocessing"
    old_staging = output / ".staging-user-data"
    old_staging.mkdir(parents=True)
    keep = old_staging / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")

    with caplog.at_level("ERROR", logger=preprocessing.__name__):
        with pytest.raises(PreprocessingValidationError, match="every train"):
            run_preprocessing(raw, output, chunk_size=1)

    assert keep.read_text(encoding="utf-8") == "preserve"
    assert "Preprocessing failed stage=group-splitting" in caplog.text
    assert "rows_processed=2" in caplog.text


def test_cli_pilot_defaults_to_separate_output_and_uses_configured_limits(
    tmp_path,
    monkeypatch,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "selected.csv").write_text(
        "Feature,Label\n1,BENIGN\n",
        encoding="utf-8",
    )
    (raw / "other.csv").write_text(
        "Feature,Label\n2,Attack\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(data_directory, output_directory, **kwargs):
        captured["data_directory"] = data_directory
        captured["output_directory"] = output_directory
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(preprocessing, "run_preprocessing", fake_run)

    result = main(
        [
            "--data-dir", str(raw),
            "--pilot",
            "--files", "selected.csv",
            "--max-rows-per-file", "50000",
            "--chunk-size", "10000",
            "--min-free-space-gib", "2.5",
        ]
    )

    assert result == 0
    assert captured["output_directory"].name == "preprocessing-pilot"
    assert captured["selected_files"] == ["selected.csv"]
    assert captured["max_rows_per_file"] == 50000
    assert captured["chunk_size"] == 10000
    assert captured["min_free_space_bytes"] == math.ceil(2.5 * preprocessing.GIB)
    assert captured["pilot_mode"] is True


def _write_eight_candidate_files(directory, rows_per_file=40):
    directory.mkdir(parents=True, exist_ok=True)
    for file_index in range(8):
        rows = [
            f"{file_index * 100000 + row_index},"
            f"{'BENIGN' if row_index % 2 == 0 else 'Attack'}"
            for row_index in range(rows_per_file)
        ]
        (directory / f"day-{file_index}.csv").write_text(
            "Feature,Label\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )


def test_bounded_sampling_is_seeded_stratified_and_not_prefix_only(tmp_path):
    raw = tmp_path / "raw"
    _write_eight_candidate_files(raw, rows_per_file=40)

    first = preprocessing.sample_bounded_candidates(
        raw,
        sample_size=24,
        row_cap_per_file=40,
        seed=42,
    )
    repeated = preprocessing.sample_bounded_candidates(
        raw,
        sample_size=24,
        row_cap_per_file=40,
        seed=42,
    )
    changed_seed = preprocessing.sample_bounded_candidates(
        raw,
        sample_size=24,
        row_cap_per_file=40,
        seed=7,
    )

    assert first == repeated
    assert first.selected_row_numbers != changed_seed.selected_row_numbers
    assert sum(map(len, first.selected_row_numbers.values())) == 24
    assert all(
        len(first.selected_row_numbers[file_name]) < 40
        for file_name in first.selected_row_numbers
    )
    for file_index in range(8):
        file_name = f"day-{file_index}.csv"
        assert set(first.manifest["candidate_binary_class_counts"][file_name].values()) == {
            20
        }
        assert all(
            count >= 1
            for count in first.manifest["selected_binary_class_counts"][
                file_name
            ].values()
        )
    assert first.manifest["rows_beyond_cap_inspected"] is False
    assert "rare attacks occurring later" in first.manifest["limitation"]


def test_bounded_sampler_does_not_consume_rows_after_the_configured_cap(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for file_index in range(8):
        (raw / f"bounded-{file_index}.csv").write_text(
            "Feature,Fwd Header Length,Fwd Header Length,Label\n"
            "1,10,10,BENIGN\n"
            "2,20,20,Attack\n"
            "malformed,30,31,\n",
            encoding="utf-8",
        )

    sample = preprocessing.sample_bounded_candidates(
        raw,
        sample_size=16,
        row_cap_per_file=2,
        seed=42,
    )

    assert sum(map(len, sample.selected_row_numbers.values())) == 16
    assert set(sample.manifest["rows_inspected_per_file"].values()) == {2}
    assert sample.manifest["candidate_rows"] == 16


def test_bounded_sampler_fails_when_candidate_pool_is_too_small(tmp_path):
    raw = tmp_path / "raw"
    _write_eight_candidate_files(raw, rows_per_file=2)

    with pytest.raises(
        PreprocessingValidationError,
        match="Insufficient bounded candidate rows.*16 eligible rows, 17 required",
    ):
        preprocessing.sample_bounded_candidates(
            raw,
            sample_size=17,
            row_cap_per_file=2,
        )


def test_sample_mode_targets_the_eight_dataset_files_and_ignores_unrelated_csvs(
    tmp_path,
):
    raw = tmp_path / "raw"
    dataset = raw / "MachineLearningCVE"
    _write_eight_candidate_files(dataset, rows_per_file=4)
    (raw / "unrelated.csv").write_text(
        "Feature,Label\n100,BENIGN\n",
        encoding="utf-8",
    )

    sample = preprocessing.sample_bounded_candidates(
        raw,
        sample_size=16,
        row_cap_per_file=4,
    )

    assert len(sample.source_files) == 8
    assert all(name.startswith("MachineLearningCVE/") for name in sample.source_files)
    assert "unrelated.csv" not in sample.source_files


def test_sample_mode_rejects_an_incomplete_machinelearningcve_directory(tmp_path):
    raw = tmp_path / "raw"
    _write_eight_candidate_files(raw / "MachineLearningCVE", rows_per_file=4)
    (raw / "MachineLearningCVE" / "day-7.csv").unlink()

    with pytest.raises(PreprocessingValidationError, match="Expected eight.*found 7"):
        preprocessing.sample_bounded_candidates(
            raw,
            sample_size=16,
            row_cap_per_file=4,
        )


def _exact_split_connection(tmp_path, group_specs):
    connection = preprocessing._create_preprocessing_database(
        tmp_path / "exact-split.sqlite3"
    )
    for index, (row_count, benign_count, attack_count) in enumerate(group_specs):
        connection.execute(
            """
            INSERT INTO feature_groups(
                digest, canonical, row_count, benign_count, attack_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"digest-{index}".encode(),
                f"canonical-{index}".encode(),
                row_count,
                benign_count,
                attack_count,
            ),
        )
    connection.commit()
    return connection


def test_exact_group_split_succeeds_where_the_old_greedy_split_misses_targets(
    tmp_path,
):
    group_specs = [
        (4, 4, 0),
        (1, 1, 0),
        (1, 1, 0),
        (3, 0, 3),
        (1, 0, 1),
        (1, 0, 1),
    ]
    grouper = ExactFeatureGrouper(("flow",))
    values = [
        float(group_index)
        for group_index, (row_count, _, _) in enumerate(group_specs)
        for _ in range(row_count)
    ]
    targets = [
        0 if benign_count else 1
        for row_count, benign_count, _attack_count in group_specs
        for _ in range(row_count)
    ]
    labels = ["BENIGN" if target == 0 else "Attack" for target in targets]
    grouper.add(pd.DataFrame({"flow": values}), targets, labels)
    greedy = split_feature_groups(grouper, seed=42)
    connection = _exact_split_connection(tmp_path, group_specs)

    row_counts, class_counts, solver = preprocessing._split_sqlite_groups_exact(
        connection,
        seed=42,
        target_rows={"train": 7, "validation": 2, "test": 2},
        node_limit=10_000,
    )

    assert greedy.row_counts != {"train": 7, "validation": 2, "test": 2}
    assert row_counts == {"train": 7, "validation": 2, "test": 2}
    assert all(
        class_counts[partition][label] > 0
        for partition in class_counts
        for label in (0, 1)
    )
    assert solver["status"] == "FEASIBLE"
    assert solver["both_classes_in_every_partition"] is True
    assert connection.execute(
        "SELECT COUNT(*) FROM group_partitions"
    ).fetchone()[0] == len(group_specs)
    connection.close()


def test_exact_group_split_keeps_conflicting_groups_whole(tmp_path):
    group_specs = [
        (4, 4, 0),
        (3, 0, 3),
        (2, 1, 1),
        (1, 1, 0),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 1),
    ]
    connection = _exact_split_connection(tmp_path, group_specs)

    row_counts, class_counts, solver = preprocessing._split_sqlite_groups_exact(
        connection,
        seed=42,
        target_rows={"train": 9, "validation": 2, "test": 2},
        node_limit=10_000,
    )

    assert row_counts == {"train": 9, "validation": 2, "test": 2}
    assert all(
        class_counts[partition][label] > 0
        for partition in class_counts
        for label in (0, 1)
    )
    conflict_partition_count = connection.execute(
        """
        SELECT COUNT(DISTINCT p.partition)
        FROM group_partitions AS p
        JOIN feature_groups AS g ON g.group_id = p.group_id
        WHERE g.benign_count > 0 AND g.attack_count > 0
        """
    ).fetchone()[0]
    assert conflict_partition_count == 1
    assert solver["both_classes_in_every_partition"] is True
    connection.close()


def test_exact_group_split_uses_class_coverage_when_it_is_not_feasible(
    tmp_path,
):
    group_specs = [
        (2, 2, 0),
        (2, 0, 2),
        (2, 1, 1),
    ]
    connection = _exact_split_connection(tmp_path, group_specs)

    row_counts, class_counts, solver = preprocessing._split_sqlite_groups_exact(
        connection,
        seed=42,
        target_rows={"train": 2, "validation": 2, "test": 2},
        node_limit=10_000,
    )

    assert row_counts == {"train": 2, "validation": 2, "test": 2}
    assert sum(
        class_counts[partition][label] > 0
        for partition in class_counts
        for label in (0, 1)
    ) < 6
    assert solver["both_classes_in_every_partition"] is False
    connection.close()


def test_exact_group_split_reports_search_limit_as_inconclusive(tmp_path):
    group_specs = [
        (4, 4, 0),
        (1, 1, 0),
        (1, 1, 0),
        (3, 0, 3),
        (1, 0, 1),
        (1, 0, 1),
    ]
    connection = _exact_split_connection(tmp_path, group_specs)

    with pytest.raises(PreprocessingValidationError, match="SOLVER_LIMIT.*INCONCLUSIVE"):
        preprocessing._split_sqlite_groups_exact(
            connection,
            seed=42,
            target_rows={"train": 7, "validation": 2, "test": 2},
            node_limit=1,
        )

    assert connection.execute(
        "SELECT COUNT(*) FROM group_partitions"
    ).fetchone()[0] == 0
    connection.close()


def test_exact_group_split_reports_proven_infeasibility_and_group_sizes(tmp_path):
    group_specs = [
        (3, 3, 0),
        (3, 0, 3),
        (2, 1, 1),
    ]
    connection = _exact_split_connection(tmp_path, group_specs)

    with pytest.raises(
        PreprocessingValidationError,
        match="INFEASIBLE.*group_size_counts",
    ):
        preprocessing._split_sqlite_groups_exact(
            connection,
            seed=42,
            target_rows={"train": 4, "validation": 2, "test": 2},
            node_limit=10_000,
        )

    assert connection.execute(
        "SELECT COUNT(*) FROM group_partitions"
    ).fetchone()[0] == 0
    connection.close()


def test_sample_10k_cli_defaults_and_conflict_validation(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_eight_candidate_files(raw, rows_per_file=2)
    captured = {}

    def fake_run(data_directory, output_directory, **kwargs):
        captured["data_directory"] = data_directory
        captured["output_directory"] = output_directory
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(preprocessing, "run_preprocessing", fake_run)
    project_root = Path(preprocessing.__file__).resolve().parents[1]
    result = main(
        [
            "--data-dir", str(raw),
            "--sample-10k",
            "--seed", "123",
            "--max-candidate-rows-per-file", "77",
            "--solver-node-limit", "900",
            "--min-free-space-gib", "0",
        ]
    )

    assert result == 0
    assert captured["sample_10k"] is True
    assert captured["candidate_row_cap"] == 77
    assert captured["seed"] == 123
    assert captured["solver_node_limit"] == 900
    assert captured["output_directory"] == project_root / "outputs" / "preprocessing-10k"

    with pytest.raises(SystemExit):
        main(
            [
                "--data-dir", str(raw),
                "--sample-10k",
                "--files", "day-0.csv",
                "--min-free-space-gib", "0",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--data-dir", str(raw),
                "--sample-10k",
                "--max-rows-per-file", "25",
                "--min-free-space-gib", "0",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--data-dir", str(raw),
                "--sample-10k",
                "--output-dir",
                str(project_root / "outputs" / "preprocessing-pilot"),
                "--min-free-space-gib", "0",
            ]
        )

def test_sample_10k_publishes_exact_partitions_and_portable_artifacts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    original_files = {}
    for file_index in range(8):
        source = raw / f"day-{file_index}.csv"
        rows = [
            f"{file_index * 100000 + row_number},"
            f"{'BENIGN' if row_number % 2 == 0 else 'Attack'}"
            for row_number in range(1_251)
        ]
        source.write_text(
            "Flow,Label\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        original_files[source.name] = source.read_bytes()
    output = tmp_path / "outputs" / "preprocessing-10k"

    run_directory = run_preprocessing(
        raw,
        output,
        chunk_size=1_000,
        seed=42,
        min_free_space_bytes=0,
        sample_10k=True,
        candidate_row_cap=1_251,
        solver_node_limit=250_000,
    )

    assert run_directory.parent == output
    assert sorted(path.name for path in run_directory.iterdir()) == [
        "feature_names.json",
        "preprocessing_pipeline.pkl",
        "sampling_manifest.json",
        "split_manifest.json",
        "target_mapping.json",
        "test.csv",
        "train.csv",
        "validation.csv",
    ]
    assert {
        path.name: path.read_bytes()
        for path in raw.glob("*.csv")
    } == original_files

    sampling_manifest = json.loads(
        (run_directory / "sampling_manifest.json").read_text(encoding="utf-8")
    )
    assert sampling_manifest["sample_size"] == 10_000
    assert sampling_manifest["seed"] == 42
    assert sampling_manifest["row_cap_per_file"] == 1_251
    assert sampling_manifest["candidate_rows"] == 10_008
    assert set(sampling_manifest["rows_inspected_per_file"].values()) == {1_251}
    assert sampling_manifest["rows_beyond_cap_inspected"] is False

    feature_names = json.loads(
        (run_directory / "feature_names.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (run_directory / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert feature_names == ["Flow"]
    assert split_manifest["actual_rows"] == {
        "train": 7_000,
        "validation": 1_500,
        "test": 1_500,
    }
    assert split_manifest["exact_partition_targets"] == split_manifest["actual_rows"]
    assert split_manifest["group_solver"]["status"] == "FEASIBLE"
    assert split_manifest["group_solver"]["both_classes_in_every_partition"] is True
    assert sum(
        sum(counts.values())
        for counts in sampling_manifest["selected_binary_class_counts"].values()
    ) == 10_000
    original_labels_by_source = {}
    for source in raw.glob("*.csv"):
        source_data = pd.read_csv(source)
        for row_number, label in enumerate(source_data["Label"], start=1):
            original_labels_by_source[(source.name, row_number)] = label
    observed_labels = {}
    for partition, expected_rows in (
        ("train", 7_000),
        ("validation", 1_500),
        ("test", 1_500),
    ):
        data = pd.read_csv(run_directory / f"{partition}.csv")
        assert len(data) == expected_rows
        assert list(data.columns) == [
            "source_file",
            "source_row_number",
            "target",
            "label_original",
            *feature_names,
        ]
        assert set(data["target"].value_counts().to_dict()) == {0, 1}
        expected_class_counts = split_manifest["binary_class_counts"][partition]
        assert {
            str(label): int(count)
            for label, count in data["target"].value_counts().items()
        } == expected_class_counts
        for row in data.itertuples(index=False):
            observed_labels[(row.source_file, row.source_row_number)] = (
                row.label_original
            )
        assert data[feature_names].notna().all().all()
        assert data[feature_names].apply(
            lambda column: column.map(math.isfinite)
        ).all().all()
    assert observed_labels
    assert all(
        observed_labels[source_row] == original_labels_by_source[source_row]
        for source_row in observed_labels
    )

    project_root = Path(preprocessing.__file__).resolve().parents[1]
    subprocess_code = """
import json
import pickle
import sys
import numpy as np
import pandas as pd
pipeline_path, names_path, partition_path = sys.argv[1:]
with open(pipeline_path, "rb") as stream:
    pipeline = pickle.load(stream)
names = json.load(open(names_path, encoding="utf-8"))
frame = pd.read_csv(partition_path, nrows=4).loc[:, names]
transformed = pipeline.transform(frame)
print(json.dumps({
    "shape": list(transformed.shape),
    "finite": bool(np.isfinite(transformed).all()),
    "module": type(pipeline.named_steps["imputer"]).__module__,
}))
"""
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            subprocess_code,
            str(run_directory / "preprocessing_pipeline.pkl"),
            str(run_directory / "feature_names.json"),
            str(run_directory / "train.csv"),
        ],
        cwd=project_root,
        capture_output=True,
        check=True,
        text=True,
    )
    result = json.loads(loaded.stdout)
    assert result == {
        "shape": [4, 1],
        "finite": True,
        "module": "src.preprocessing_components",
    }


def test_sample_10k_exact_group_failure_cleans_only_its_staging_directory(
    tmp_path,
):
    raw = tmp_path / "raw"
    raw.mkdir()
    for file_index in range(8):
        rows = [
            f"1,{'BENIGN' if row_number % 2 == 0 else 'Attack'}"
            for row_number in range(1_251)
        ]
        (raw / f"day-{file_index}.csv").write_text(
            "Flow,Label\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "outputs" / "preprocessing-10k"
    preexisting_staging = output / ".staging-user-data"
    preexisting_staging.mkdir(parents=True)
    keep_file = preexisting_staging / "keep.txt"
    keep_file.write_text("preserve", encoding="utf-8")

    with pytest.raises(
        PreprocessingValidationError,
        match="INFEASIBLE.*exceed every partition capacity",
    ):
        run_preprocessing(
            raw,
            output,
            chunk_size=1_000,
            sample_10k=True,
            candidate_row_cap=1_251,
            solver_node_limit=1_000,
            min_free_space_bytes=0,
        )

    assert keep_file.read_text(encoding="utf-8") == "preserve"
    assert not list(output.glob("run-*"))
    assert [path.name for path in output.glob(".staging-*")] == [
        ".staging-user-data"
    ]