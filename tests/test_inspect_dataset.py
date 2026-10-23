import pandas as pd

from src.inspect_dataset import format_report, inspect_dataset, main


def _write_csv(path, rows, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_empty_directory_prints_download_instructions(tmp_path, capsys):
    exit_code = main(["--data-dir", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "No CSV files found" in output
    assert "https://www.unb.ca/cic/datasets/ids-2017.html" in output
    assert "data/raw/" in output


def test_one_csv_reports_rows_labels_and_percentages(tmp_path):
    _write_csv(
        tmp_path / "traffic.csv",
        {"Flow": [1, 2, 3, 4], "Label": ["BENIGN", "BENIGN", "Attack", "Attack"]},
    )

    report = inspect_dataset(tmp_path, chunk_size=2)
    file = report.files[0]
    output = format_report(report)

    assert file.row_count == 4
    assert file.column_count == 2
    assert file.label_counts == {"BENIGN": 2, "Attack": 2}
    assert "BENIGN: 2 (50.00%)" in output
    assert "Attack: 2 (50.00%)" in output


def test_multiple_csv_files_are_processed_recursively(tmp_path):
    nested = tmp_path / "day1"
    nested.mkdir()
    _write_csv(tmp_path / "first.csv", {"Label": ["BENIGN"]})
    _write_csv(nested / "second.CSV", {"Label": ["Attack", "BENIGN"]})

    report = inspect_dataset(tmp_path, chunk_size=1)

    assert len(report.files) == 2
    row_counts = {file.path.name: file.row_count for file in report.files}
    assert row_counts == {"first.csv": 1, "second.CSV": 2}
    assert "CSV files found: 2" in format_report(report)


def test_label_column_matching_ignores_case_and_surrounding_whitespace(tmp_path):
    _write_csv(
        tmp_path / "labels.csv",
        {" Flow ": [1, 2], " lAbEl ": ["BENIGN", "Attack"]},
    )

    file = inspect_dataset(tmp_path).files[0]

    assert file.label_column == " lAbEl "
    assert file.label_counts == {"BENIGN": 1, "Attack": 1}


def test_missing_and_infinite_numeric_values_are_reported(tmp_path):
    _write_csv(
        tmp_path / "values.csv",
        {"Feature": [1.0, float("inf"), float("-inf"), float("nan")],
         "Label": ["BENIGN", "Attack", "Attack", "BENIGN"]},
    )

    file = inspect_dataset(tmp_path).files[0]
    output = format_report(inspect_dataset(tmp_path))

    assert file.missing_values == {"Feature": 1}
    assert file.infinities == {"Feature": (1, 1)}
    assert "Feature" in output
    assert "+inf=1, -inf=1" in output


def test_blank_records_are_counted_instead_of_silently_skipped(tmp_path):
    (tmp_path / "blank_record.csv").write_text(
        "Feature,Label\n1,BENIGN\n\n2,Attack\n", encoding="utf-8"
    )

    file = inspect_dataset(tmp_path).files[0]

    assert file.row_count == 3
    assert file.missing_values == {"Feature": 1, "Label": 1}


def test_missing_label_column_is_reported(tmp_path):
    _write_csv(tmp_path / "no_label.csv", {"Flow": [1, 2]})

    file = inspect_dataset(tmp_path).files[0]

    assert file.label_column is None
    assert "Label column: not found" in format_report(inspect_dataset(tmp_path))


def test_inconsistent_schemas_are_reported(tmp_path):
    _write_csv(tmp_path / "first.csv", {"Flow": [1], "Label": ["BENIGN"]})
    _write_csv(tmp_path / "second.csv", {"Bytes": [2], "Label": ["Attack"]})

    output = format_report(inspect_dataset(tmp_path))

    assert "Inconsistent column schemas detected" in output
    assert "missing columns" in output
    assert "extra columns" in output


def test_duplicate_column_names_are_reported(tmp_path):
    (tmp_path / "duplicates.csv").write_text(
        "Feature,Feature,Label\n1,2,BENIGN\n", encoding="utf-8"
    )

    file = inspect_dataset(tmp_path).files[0]

    assert file.duplicate_columns == ("Feature", "Feature")
    assert "Duplicate column names" in format_report(inspect_dataset(tmp_path))


def test_malformed_csv_is_reported_without_stopping_other_files(tmp_path):
    (tmp_path / "malformed.csv").write_text(
        "Feature,Label\n1,BENIGN\n2,Attack,unexpected\n", encoding="utf-8"
    )
    _write_csv(tmp_path / "valid.csv", {"Label": ["BENIGN"]})

    report = inspect_dataset(tmp_path, chunk_size=1)
    malformed = next(file for file in report.files if file.path.name == "malformed.csv")
    valid = next(file for file in report.files if file.path.name == "valid.csv")

    assert malformed.error is not None
    assert "Could not read this CSV" in format_report(report)
    assert valid.row_count == 1