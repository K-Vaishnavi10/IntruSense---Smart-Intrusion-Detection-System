from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_CHUNK_SIZE = 100_000


@dataclass
class FileInspection:
    path: Path
    columns: tuple[str, ...] = ()
    row_count: int | None = None
    label_column: str | None = None
    label_counts: dict[str, int] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    infinities: dict[str, tuple[int, int]] = field(default_factory=dict)
    duplicate_columns: tuple[str, ...] = ()
    error: str | None = None

    @property
    def column_count(self) -> int:
        return len(self.columns)


@dataclass
class DatasetInspection:
    directory: Path
    files: list[FileInspection]


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        header = next(reader, None)
    if not header:
        raise ValueError("CSV is empty or does not contain a header row")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        next(reader)
        for record_number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(
                    f"record {record_number} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
    return tuple(header)


def inspect_csv_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> FileInspection:
    """Inspect one CSV incrementally; failed files return an error report."""
    result = FileInspection(path=path)
    try:
        columns = _read_header(path)
        result.columns = columns
        normalized_columns = [column.strip().casefold() for column in columns]
        duplicate_names = {
            name for name, count in Counter(normalized_columns).items() if count > 1
        }
        result.duplicate_columns = tuple(
            column for column, normalized in zip(columns, normalized_columns)
            if normalized in duplicate_names
        )

        label_indexes = [
            index for index, column in enumerate(columns)
            if column.strip().casefold() == "label"
        ]
        label_index = label_indexes[0] if label_indexes else None
        if label_index is not None:
            result.label_column = columns[label_index]

        row_count = 0
        label_counts: Counter[str] = Counter()
        missing_values: Counter[str] = Counter()
        infinities: dict[str, list[int]] = {}

        with pd.read_csv(
            path, chunksize=chunk_size, skip_blank_lines=False
        ) as chunks:
            for chunk in chunks:
                row_count += len(chunk)
                for index, column in enumerate(columns):
                    values = chunk.iloc[:, index]
                    missing_values[column] += int(values.isna().sum())
                    if pd.api.types.is_numeric_dtype(values.dtype):
                        positive = int((values == np.inf).sum())
                        negative = int((values == -np.inf).sum())
                        if positive or negative:
                            counts = infinities.setdefault(column, [0, 0])
                            counts[0] += positive
                            counts[1] += negative

                if label_index is not None:
                    counts = chunk.iloc[:, label_index].value_counts(
                        dropna=False, sort=False
                    )
                    for label, count in counts.items():
                        display_label = "<missing>" if pd.isna(label) else str(label)
                        label_counts[display_label] += int(count)

        result.row_count = row_count
        result.label_counts = dict(label_counts)
        result.missing_values = {
            column: count for column, count in missing_values.items() if count
        }
        result.infinities = {
            column: (counts[0], counts[1])
            for column, counts in infinities.items()
        }
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
    return result


def inspect_dataset(
    directory: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> DatasetInspection:
    """Recursively inspect CSV files in a directory without modifying them."""
    csv_files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".csv"
    ) if directory.is_dir() else []
    return DatasetInspection(
        directory=directory,
        files=[inspect_csv_file(path, chunk_size) for path in csv_files],
    )


def _schema_messages(files: list[FileInspection], directory: Path) -> list[str]:
    readable = [file for file in files if file.error is None]
    if len(readable) < 2:
        return []

    reference = readable[0]
    messages: list[str] = []
    for current in readable[1:]:
        if current.columns == reference.columns:
            continue
        missing = list((Counter(reference.columns) - Counter(current.columns)).elements())
        extra = list((Counter(current.columns) - Counter(reference.columns)).elements())
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if extra:
            details.append(f"extra columns: {extra}")
        if not details:
            details.append("column order differs")
        filename = current.path.relative_to(directory)
        messages.append(f"  {filename}: " + "; ".join(details))
    return messages


def format_report(report: DatasetInspection) -> str:
    lines = [f"Dataset directory: {report.directory}"]
    if not report.files:
        lines.extend([
            "No CSV files found.",
            "Download the CIC-IDS2017 CSV files manually from:",
            "https://www.unb.ca/cic/datasets/ids-2017.html",
            "Extract them into data/raw/ and run this script again.",
        ])
        return "\n".join(lines)

    lines.append(f"CSV files found: {len(report.files)}")
    for file in report.files:
        filename = file.path.relative_to(report.directory)
        lines.extend(["", f"File: {filename}"])
        if file.error:
            lines.append(f"  Could not read this CSV: {file.error}")
            continue

        lines.append(f"  Rows: {file.row_count}")
        lines.append(f"  Columns: {file.column_count}")
        if file.duplicate_columns:
            duplicates = ", ".join(repr(name) for name in file.duplicate_columns)
            lines.append(f"  Duplicate column names (case/whitespace-insensitive): {duplicates}")
        if file.label_column is None:
            lines.append("  Label column: not found")
        else:
            lines.append(f"  Label distribution ({file.label_column!r}):")
            for label, count in file.label_counts.items():
                percentage = (count / file.row_count * 100) if file.row_count else 0.0
                lines.append(f"    {label}: {count} ({percentage:.2f}%)")
        if file.missing_values:
            values = ", ".join(
                f"{column!r}: {count}" for column, count in file.missing_values.items()
            )
            lines.append(f"  Missing values: {values}")
        else:
            lines.append("  Missing values: none")
        if file.infinities:
            values = ", ".join(
                f"{column!r}: +inf={counts[0]}, -inf={counts[1]}"
                for column, counts in file.infinities.items()
            )
            lines.append(f"  Infinite numeric values: {values}")
        else:
            lines.append("  Infinite numeric values: none")

    schema_messages = _schema_messages(report.files, report.directory)
    readable_count = sum(file.error is None for file in report.files)
    if readable_count > 1:
        if schema_messages:
            lines.append("\nInconsistent column schemas detected:")
            lines.extend(schema_messages)
        else:
            lines.append("\nColumn schemas are consistent across readable CSV files.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect CSV files in data/raw/.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help="dataset directory (default: data/raw/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"rows per processing chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    arguments = parser.parse_args(argv)
    if arguments.chunk_size < 1:
        parser.error("--chunk-size must be greater than zero")

    report = inspect_dataset(arguments.data_dir, arguments.chunk_size)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())