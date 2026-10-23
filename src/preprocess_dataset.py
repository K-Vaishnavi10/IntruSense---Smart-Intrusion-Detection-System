from __future__ import annotations

import argparse
import csv
import ctypes
from decimal import Decimal, InvalidOperation
import heapq
import hashlib
from itertools import islice
import json
import logging
import math
import os
import pickle
import random
import sqlite3
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence
from ctypes import wintypes

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing_components import (
    PrecomputedMedianImputer,
    PreprocessingValidationError,
)


EXPECTED_DUPLICATE_COLUMN = "Fwd Header Length"
LABEL_COLUMN = "Label"
GIB = 1024**3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeaderValidation:
    source_columns: tuple[str, ...]
    label_index: int
    duplicate_column_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class VerifiedHeader:
    source_columns: tuple[str, ...]
    retained_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    label_index: int
    duplicate_column_pairs: tuple[tuple[int, int], ...]
    row_count: int


@dataclass
class DataQualityReport:
    """Counts collected while reading source rows in bounded-size chunks."""

    row_count: int = 0
    files: list[str] = field(default_factory=list)
    feature_columns: tuple[str, ...] = ()
    original_missing: dict[str, int] = field(default_factory=dict)
    positive_infinity: dict[str, int] = field(default_factory=dict)
    negative_infinity: dict[str, int] = field(default_factory=dict)
    nonnumeric_values: dict[str, int] = field(default_factory=dict)
    class_counts: dict[int, int] = field(default_factory=dict)
    schemas: list[dict[str, object]] = field(default_factory=list)
    _observed_values: dict[str, set[Decimal]] = field(default_factory=dict, repr=False)

    @property
    def constant_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for column, observed in self._observed_values.items()
            if len(observed) == 1
        )


@dataclass(frozen=True)
class IngestedChunk:
    """A source-ordered feature chunk with targets and provenance kept separate."""

    features: pd.DataFrame
    target: pd.Series
    label_original: pd.Series
    source_file: str
    source_row_numbers: tuple[int, ...]
    canonical_feature_rows: tuple[bytes, ...]


@dataclass(frozen=True)
class BoundedSample:
    selected_row_numbers: dict[str, frozenset[int]]
    source_files: tuple[str, ...]
    source_signatures: dict[str, tuple[int, int]]
    manifest: dict[str, object]


@dataclass
class FeatureGroup:
    group_id: int
    row_count: int = 0
    class_counts: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    original_labels: set[str] = field(default_factory=set)

    @property
    def conflicting_binary_labels(self) -> bool:
        return all(self.class_counts[label] for label in (0, 1))


@dataclass(frozen=True)
class GroupSplit:
    group_to_partition: dict[int, str]
    partition_group_ids: dict[str, tuple[int, ...]]
    row_counts: dict[str, int]
    class_counts: dict[str, dict[int, int]]


def _canonical_value_from_token(token: bytes) -> Decimal | None:
    if token == b"N":
        return None
    try:
        sign_text, exponent_text, digits_text = token.decode("ascii").split(":")
        return Decimal(
            (int(sign_text), tuple(int(digit) for digit in digits_text), int(exponent_text))
        )
    except (UnicodeDecodeError, ValueError, InvalidOperation) as error:
        raise PreprocessingValidationError(
            "Stored canonical feature value is invalid"
        ) from error


def _decimal_collation(left: str, right: str) -> int:
    left_value = Decimal(left)
    right_value = Decimal(right)
    return (left_value > right_value) - (left_value < right_value)


def _sqlite_tie_key(seed: int, group_id: int) -> bytes:
    return hashlib.sha256(f"{seed}:{group_id}".encode("ascii")).digest()


def _existing_volume_path(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise PreprocessingValidationError(
                f"Could not find an existing parent for output path: {path}"
            )
        candidate = parent
    return candidate


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _directory_size(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    )


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _process_memory_snapshot() -> dict[str, int] | None:
    """Read process peak memory counters using the Windows PSAPI."""
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
        "private_usage_bytes": int(counters.PrivateUsage),
    }


def _record_pilot_measurement(
    measurements: list[dict[str, object]],
    stage: str,
    staging_directory: Path,
    output_directory: Path,
    started_at: float,
    rows_processed: int,
) -> None:
    volume_path = _existing_volume_path(output_directory)
    measurement: dict[str, object] = {
        "stage": stage,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "rows_processed": rows_processed,
        "staging_bytes": _directory_size(staging_directory),
        "volume_free_bytes": shutil.disk_usage(volume_path).free,
    }
    memory = _process_memory_snapshot()
    if memory is not None:
        measurement["process_memory"] = memory
    measurements.append(measurement)
    logger.info(
        "Pilot measurement stage=%s rows=%d elapsed_seconds=%.3f "
        "staging_bytes=%d volume_free_bytes=%d process_memory=%s",
        stage,
        rows_processed,
        measurement["elapsed_seconds"],
        measurement["staging_bytes"],
        measurement["volume_free_bytes"],
        memory if memory is not None else "unavailable",
    )


def _create_preprocessing_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -16384")
    connection.create_collation("DECIMAL", _decimal_collation)
    connection.create_function(
        "split_tie_key",
        2,
        _sqlite_tie_key,
        deterministic=True,
    )
    connection.executescript(
        """
        CREATE TABLE feature_groups (
            group_id INTEGER PRIMARY KEY,
            digest BLOB NOT NULL,
            canonical BLOB NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            benign_count INTEGER NOT NULL DEFAULT 0,
            attack_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX feature_groups_digest ON feature_groups(digest);
        CREATE TABLE group_features (
            group_id INTEGER NOT NULL REFERENCES feature_groups(group_id),
            feature_index INTEGER NOT NULL,
            value_text TEXT,
            PRIMARY KEY (group_id, feature_index)
        );
        CREATE INDEX group_features_feature_decimal
            ON group_features(
                feature_index,
                value_text COLLATE DECIMAL,
                group_id
            );
        CREATE TABLE group_labels (
            group_id INTEGER NOT NULL REFERENCES feature_groups(group_id),
            original_label TEXT NOT NULL,
            PRIMARY KEY (group_id, original_label)
        );
        CREATE TABLE source_rows (
            row_id INTEGER PRIMARY KEY,
            group_id INTEGER NOT NULL REFERENCES feature_groups(group_id),
            target INTEGER NOT NULL CHECK (target IN (0, 1)),
            label_original TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_row_number INTEGER NOT NULL
        );
        CREATE INDEX source_rows_group ON source_rows(group_id);
        CREATE TABLE group_partitions (
            group_id INTEGER PRIMARY KEY REFERENCES feature_groups(group_id),
            partition TEXT NOT NULL CHECK (
                partition IN ('train', 'validation', 'test')
            )
        );
        CREATE INDEX group_partitions_partition
            ON group_partitions(partition, group_id);
        """
    )
    return connection


def _store_chunk(
    connection: sqlite3.Connection,
    chunk: IngestedChunk,
    feature_columns: Sequence[str],
) -> None:
    if len(chunk.canonical_feature_rows) != len(chunk.features):
        raise PreprocessingValidationError(
            "Canonical feature data does not match chunk row count"
        )
    for canonical, target, label, source_row in (
        zip(
            chunk.canonical_feature_rows,
            chunk.target.tolist(),
            chunk.label_original.tolist(),
            chunk.source_row_numbers,
        )
    ):
        digest = _feature_digest(canonical)
        group_row = connection.execute(
            """
            SELECT group_id FROM feature_groups
            WHERE digest = ? AND canonical = ?
            """,
            (digest, canonical),
        ).fetchone()
        if group_row is None:
            cursor = connection.execute(
                """
                INSERT INTO feature_groups(digest, canonical)
                VALUES (?, ?)
                """,
                (digest, canonical),
            )
            group_id = int(cursor.lastrowid)
            tokens = canonical.split(b";")
            if len(tokens) != len(feature_columns):
                raise PreprocessingValidationError(
                    "Canonical feature count differs from the validated schema"
                )
            connection.executemany(
                """
                INSERT INTO group_features(group_id, feature_index, value_text)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        group_id,
                        feature_index,
                        (
                            str(decimal_value)
                            if (decimal_value := _canonical_value_from_token(token))
                            is not None
                            else None
                        ),
                    )
                    for feature_index, token in enumerate(tokens)
                ],
            )
        else:
            group_id = int(group_row[0])

        connection.execute(
            """
            UPDATE feature_groups
            SET row_count = row_count + 1,
                benign_count = benign_count + ?,
                attack_count = attack_count + ?
            WHERE group_id = ?
            """,
            (int(target == 0), int(target == 1), group_id),
        )
        connection.execute(
            "INSERT OR IGNORE INTO group_labels(group_id, original_label) VALUES (?, ?)",
            (group_id, label),
        )
        connection.execute(
            """
            INSERT INTO source_rows(
                group_id, target, label_original, source_file, source_row_number
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (group_id, int(target), label, chunk.source_file, int(source_row)),
        )
    connection.commit()


def validate_header(columns: Sequence[str]) -> HeaderValidation:
    """Validate a source header and identify the one approved duplicate."""
    source_columns = tuple(columns)
    if not source_columns:
        raise PreprocessingValidationError("CSV header is empty")

    normalized = tuple(column.strip() for column in source_columns)
    empty_columns = [index for index, name in enumerate(normalized) if not name]
    if empty_columns:
        raise PreprocessingValidationError(
            f"CSV header contains empty column names at positions "
            f"{[index + 1 for index in empty_columns]}"
        )

    label_indexes = [
        index for index, name in enumerate(normalized)
        if name.casefold() == LABEL_COLUMN.casefold()
    ]
    if len(label_indexes) != 1:
        raise PreprocessingValidationError(
            f"Expected exactly one {LABEL_COLUMN!r} column; found {len(label_indexes)}"
        )

    indexes_by_name: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(normalized):
        indexes_by_name[name.casefold()].append(index)

    duplicate_pairs: list[tuple[int, int]] = []
    expected_duplicate = EXPECTED_DUPLICATE_COLUMN.casefold()
    for name, indexes in indexes_by_name.items():
        if len(indexes) == 1:
            continue
        if (
            name != expected_duplicate
            or len(indexes) != 2
            or any(normalized[index] != EXPECTED_DUPLICATE_COLUMN for index in indexes)
        ):
            columns_found = [source_columns[index] for index in indexes]
            raise PreprocessingValidationError(
                "Unexpected duplicate column names after trimming surrounding "
                f"whitespace: {columns_found!r}"
            )
        first, second = indexes
        duplicate_pairs.append((first, second))

    return HeaderValidation(
        source_columns=source_columns,
        label_index=label_indexes[0],
        duplicate_column_pairs=tuple(duplicate_pairs),
    )


def _validate_source_row(
    path: Path,
    header: HeaderValidation,
    record_number: int,
    row: Sequence[str],
) -> None:
    if len(row) != len(header.source_columns):
        raise PreprocessingValidationError(
            f"{path.name}: record {record_number} has {len(row)} fields; "
            f"expected {len(header.source_columns)}"
        )
    for first, second in header.duplicate_column_pairs:
        if row[first] != row[second]:
            raise PreprocessingValidationError(
                f"{path.name}: record {record_number} has different values "
                f"in duplicate {EXPECTED_DUPLICATE_COLUMN!r} columns "
                f"at positions {first + 1} and {second + 1}"
            )


def verify_duplicate_columns(
    path: Path,
    header: HeaderValidation,
    row_limit: int | None = None,
    row_callback: Callable[[Sequence[str], int], None] | None = None,
) -> VerifiedHeader:
    """Check row widths and duplicate equality, optionally within a row limit."""
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be greater than zero")
    row_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            actual_header = next(reader, None)
            if actual_header is None:
                raise PreprocessingValidationError("CSV is empty")
            if tuple(actual_header) != header.source_columns:
                raise PreprocessingValidationError(
                    f"{path.name}: CSV header differs from the validated header"
                )

            records = enumerate(reader, start=2)
            if row_limit is not None:
                records = islice(records, row_limit)
            for record_number, row in records:
                _validate_source_row(path, header, record_number, row)
                row_count += 1
                if row_callback is not None:
                    row_callback(row, row_count)
    except (csv.Error, UnicodeDecodeError) as error:
        raise PreprocessingValidationError(
            f"{path.name}: could not parse CSV: {error}"
        ) from error
    removed_indexes = {second for _, second in header.duplicate_column_pairs}
    normalized_columns = tuple(column.strip() for column in header.source_columns)
    retained_columns = tuple(
        name for index, name in enumerate(normalized_columns)
        if index not in removed_indexes
    )
    feature_columns = tuple(
        name for index, name in enumerate(normalized_columns)
        if index != header.label_index and index not in removed_indexes
    )
    return VerifiedHeader(
        source_columns=header.source_columns,
        retained_columns=retained_columns,
        feature_columns=feature_columns,
        label_index=header.label_index,
        duplicate_column_pairs=header.duplicate_column_pairs,
        row_count=row_count,
    )


def validate_label_text(original_label: str | None) -> str:
    """Validate a label and return its original text unchanged."""
    if original_label is None or not original_label.strip():
        raise PreprocessingValidationError(
            "Label is missing or blank; every row must have a nonempty label"
        )
    return original_label


def encode_binary_label(original_label: str | None) -> int:
    """Map BENIGN to 0 and any other nonempty original label to 1."""
    label = validate_label_text(original_label)
    if label.strip().casefold() == "benign":
        return 0
    return 1


def _allocate_stratified_quotas(
    stratum_counts: Mapping[tuple[str, int], int],
    sample_size: int,
) -> dict[tuple[str, int], int]:
    nonempty = sorted(
        (key, count)
        for key, count in stratum_counts.items()
        if count > 0
    )
    available = sum(count for _, count in nonempty)
    if available < sample_size:
        raise PreprocessingValidationError(
            "Insufficient bounded candidate rows for requested sample: "
            f"{available} eligible rows, {sample_size} required"
        )
    if len(nonempty) > sample_size:
        raise PreprocessingValidationError(
            "Sample size is too small to represent every nonempty "
            "source-file/binary-class stratum"
        )

    quotas = {key: 1 for key, _ in nonempty}
    remaining = sample_size - len(nonempty)
    while remaining:
        active = [
            (key, count - quotas[key])
            for key, count in nonempty
            if count > quotas[key]
        ]
        if not active:
            raise PreprocessingValidationError(
                "Could not allocate the requested sample across candidate strata"
            )
        total_weight = sum(count for _, count in active)
        allocated_this_round = 0
        remainders: list[tuple[int, tuple[str, int]]] = []
        for key, capacity in active:
            numerator = remaining * capacity
            allocation, remainder = divmod(numerator, total_weight)
            allocation = min(allocation, capacity)
            quotas[key] += allocation
            allocated_this_round += allocation
            remainders.append((remainder, key))
        remaining -= allocated_this_round
        if not remaining:
            break
        progressed = False
        for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if quotas[key] < stratum_counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed and remaining:
            continue
    return quotas


def sample_bounded_candidates(
    data_directory: Path,
    sample_size: int = 10_000,
    row_cap_per_file: int = 25_000,
    seed: int = 42,
    progress_callback: Callable[[str, str, int], None] | None = None,
) -> BoundedSample:
    """Select a deterministic stratified sample without reading past each cap."""
    if sample_size < 1:
        raise PreprocessingValidationError("sample_size must be greater than zero")
    if row_cap_per_file < 1:
        raise PreprocessingValidationError(
            "row_cap_per_file must be greater than zero"
        )
    dataset_directory = data_directory / "MachineLearningCVE"
    search_directory = (
        dataset_directory if dataset_directory.is_dir() else data_directory
    )
    files = sorted(
        path for path in search_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".csv"
    ) if data_directory.is_dir() else []
    if not files:
        raise PreprocessingValidationError(
            f"No CSV files found in dataset directory: {data_directory}"
        )
    if dataset_directory.is_dir() and len(files) != 8:
        raise PreprocessingValidationError(
            "Expected eight CIC-IDS2017 MachineLearningCVE CSV files, "
            f"found {len(files)} beneath {dataset_directory}"
        )

    headers: dict[str, HeaderValidation] = {}
    source_signatures: dict[str, tuple[int, int]] = {}
    rows_inspected: dict[str, int] = {}
    stratum_counts: dict[tuple[str, int], int] = defaultdict(int)
    candidate_label_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    expected_features: tuple[str, ...] | None = None

    for path in files:
        source_file = path.relative_to(data_directory).as_posix()
        if progress_callback is not None:
            progress_callback("candidate-inspection-start", source_file, 0)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source, strict=True)
                raw_header = next(reader, None)
        except (csv.Error, UnicodeDecodeError) as error:
            raise PreprocessingValidationError(
                f"{path.name}: could not parse CSV header: {error}"
            ) from error
        if raw_header is None:
            raise PreprocessingValidationError(f"{path.name}: CSV is empty")
        header = validate_header(raw_header)
        headers[source_file] = header
        source_stat = path.stat()
        source_signatures[source_file] = (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        )

        normalized_features = tuple(
            name.strip()
            for index, name in enumerate(header.source_columns)
            if index != header.label_index
            and index not in {second for _, second in header.duplicate_column_pairs}
        )
        if expected_features is None:
            expected_features = normalized_features
        elif normalized_features != expected_features:
            raise PreprocessingValidationError(
                f"{path.name}: feature columns or their order differ from "
                "the first CSV"
            )

        def count_candidate(row: Sequence[str], row_number: int) -> None:
            _ = row_number
            original_label = validate_label_text(row[header.label_index])
            target = encode_binary_label(original_label)
            stratum_counts[(source_file, target)] += 1
            candidate_label_counts[source_file][original_label] += 1

        verified = verify_duplicate_columns(
            path,
            header,
            row_limit=row_cap_per_file,
            row_callback=count_candidate,
        )
        rows_inspected[source_file] = verified.row_count
        if progress_callback is not None:
            progress_callback(
                "candidate-inspection-complete",
                source_file,
                verified.row_count,
            )

    quotas = _allocate_stratified_quotas(stratum_counts, sample_size)
    heaps: dict[tuple[str, int], list[tuple[int, int, int, str]]] = {
        stratum: [] for stratum in quotas
    }
    seen_stratum_counts: dict[tuple[str, int], int] = defaultdict(int)
    seen_label_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for path in files:
        source_file = path.relative_to(data_directory).as_posix()
        if progress_callback is not None:
            progress_callback("candidate-priority-start", source_file, 0)
        current_stat = path.stat()
        if source_signatures[source_file] != (
            current_stat.st_size,
            current_stat.st_mtime_ns,
        ):
            raise PreprocessingValidationError(
                f"{source_file}: file changed during bounded sampling"
            )
        header = headers[source_file]
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source, strict=True)
                actual_header = next(reader, None)
                if actual_header is None or tuple(actual_header) != header.source_columns:
                    raise PreprocessingValidationError(
                        f"{path.name}: CSV header changed during bounded sampling"
                    )
                for row_number, row in enumerate(
                    islice(reader, row_cap_per_file),
                    start=1,
                ):
                    _validate_source_row(path, header, row_number + 1, row)
                    original_label = validate_label_text(row[header.label_index])
                    target = encode_binary_label(original_label)
                    stratum = (source_file, target)
                    seen_stratum_counts[stratum] += 1
                    seen_label_counts[source_file][original_label] += 1
                    quota = quotas.get(stratum, 0)
                    if not quota:
                        continue
                    priority_bytes = hashlib.sha256(
                        f"{seed}\0{source_file}\0{row_number}".encode("utf-8")
                    ).digest()
                    priority = int.from_bytes(priority_bytes, "big")
                    candidate = (-priority, -row_number, row_number, original_label)
                    heap = heaps[stratum]
                    if len(heap) < quota:
                        heapq.heappush(heap, candidate)
                    elif candidate[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, candidate)
        except (csv.Error, UnicodeDecodeError) as error:
            raise PreprocessingValidationError(
                f"{path.name}: could not parse bounded sample: {error}"
            ) from error
        updated_stat = path.stat()
        if source_signatures[source_file] != (
            updated_stat.st_size,
            updated_stat.st_mtime_ns,
        ):
            raise PreprocessingValidationError(
                f"{source_file}: file changed during bounded sampling"
            )
        if progress_callback is not None:
            progress_callback(
                "candidate-priority-complete",
                source_file,
                rows_inspected[source_file],
            )

    if dict(seen_stratum_counts) != dict(stratum_counts):
        raise PreprocessingValidationError(
            "Candidate label counts changed between bounded sampling passes"
        )
    if {
        source_file: dict(counts)
        for source_file, counts in seen_label_counts.items()
    } != {
        source_file: dict(counts)
        for source_file, counts in candidate_label_counts.items()
    }:
        raise PreprocessingValidationError(
            "Original label counts changed between bounded sampling passes"
        )
    selected_rows: dict[str, set[int]] = defaultdict(set)
    selected_label_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    selected_binary_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"0": 0, "1": 0}
    )
    for stratum, quota in quotas.items():
        selected = heaps[stratum]
        if len(selected) != quota:
            raise PreprocessingValidationError(
                f"Could not select the allocated quota for {stratum[0]} "
                f"class {stratum[1]}: selected {len(selected)}, expected {quota}"
            )
        source_file, target = stratum
        for _, _, row_number, original_label in selected:
            selected_rows[source_file].add(row_number)
            selected_label_counts[source_file][original_label] += 1
            selected_binary_counts[source_file][str(target)] += 1

    selected_count = sum(len(rows) for rows in selected_rows.values())
    if selected_count != sample_size:
        raise PreprocessingValidationError(
            f"Bounded sampler selected {selected_count} rows; expected {sample_size}"
        )
    candidate_labels_json = {
        source_file: dict(sorted(counts.items()))
        for source_file, counts in sorted(candidate_label_counts.items())
    }
    selected_labels_json = {
        source_file: dict(sorted(counts.items()))
        for source_file, counts in sorted(selected_label_counts.items())
    }
    manifest: dict[str, object] = {
        "sample_size": sample_size,
        "seed": seed,
        "row_cap_per_file": row_cap_per_file,
        "row_cap_excludes_header": True,
        "source_files": [path.relative_to(data_directory).as_posix() for path in files],
        "source_file_metadata": {
            source_file: {
                "size_bytes": signature[0],
                "modified_time_ns": signature[1],
            }
            for source_file, signature in sorted(source_signatures.items())
        },
        "rows_inspected_per_file": rows_inspected,
        "candidate_rows": sum(rows_inspected.values()),
        "candidate_binary_class_counts": {
            source_file: {
                str(label): stratum_counts.get((source_file, label), 0)
                for label in (0, 1)
            }
            for source_file in rows_inspected
        },
        "candidate_label_counts": candidate_labels_json,
        "selection_quotas_per_file_and_class": {
            source_file: {
                str(label): quotas.get((source_file, label), 0)
                for label in (0, 1)
            }
            for source_file in rows_inspected
        },
        "selected_rows_per_file": {
            source_file: len(selected_rows.get(source_file, ()))
            for source_file in rows_inspected
        },
        "selected_binary_class_counts": dict(selected_binary_counts),
        "selected_label_counts": selected_labels_json,
        "selection_method": (
            "Stratify by relative source CSV and binary class; reserve one row "
            "for each nonempty stratum, apportion the remaining quota "
            "proportionally with largest remainders, then select rows with the "
            "lowest SHA-256 priority of seed, relative path, and 1-based source "
            "row number."
        ),
        "bounded_passes": {
            "candidate_counting": 1,
            "priority_selection": 1,
            "selected_row_ingestion": 1,
            "maximum_data_rows_per_file_per_pass": row_cap_per_file,
        },
        "rows_beyond_cap_inspected": False,
        "limitation": (
            "Rows after the configured per-file cap were not inspected; rare "
            "attacks occurring later in a CSV may be absent from this sample."
        ),
    }
    return BoundedSample(
        selected_row_numbers={
            source_file: frozenset(row_numbers)
            for source_file, row_numbers in selected_rows.items()
        },
        source_files=tuple(
            path.relative_to(data_directory).as_posix()
            for path in files
        ),
        source_signatures=source_signatures,
        manifest=manifest,
    )


def _is_missing_numeric_text(value: str) -> bool:
    return value.strip().casefold() in {"", "na", "n/a", "nan", "null", "none"}


def _canonical_decimal_token(value: Decimal | None) -> bytes:
    if value is None or value.is_nan():
        return b"N"
    if not value.is_finite():
        raise PreprocessingValidationError(
            "Infinite values must be converted to missing before grouping"
        )
    if value.is_zero():
        return b"0:0:0"
    parts = value.as_tuple()
    digits = list(parts.digits)
    exponent = parts.exponent
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return (
        f"{parts.sign}:{exponent}:{''.join(map(str, digits))}".encode("ascii")
    )


def _canonical_tokens(values: Sequence[str]) -> bytes:
    tokens: list[bytes] = []
    for value in values:
        stripped = value.strip()
        if _is_missing_numeric_text(stripped):
            tokens.append(_canonical_decimal_token(None))
            continue
        try:
            number = Decimal(stripped)
        except InvalidOperation:
            tokens.append(_canonical_decimal_token(None))
            continue
        if not number.is_finite():
            tokens.append(_canonical_decimal_token(None))
        else:
            tokens.append(_canonical_decimal_token(number))
    return b";".join(tokens)


def iter_dataset_chunks(
    data_directory: Path,
    chunk_size: int = 100_000,
    report: DataQualityReport | None = None,
    selected_files: Sequence[str] | None = None,
    max_rows_per_file: int | None = None,
    progress_callback: Callable[[str, str, int], None] | None = None,
    selected_row_numbers: Mapping[str, frozenset[int]] | None = None,
    bounded_verification: bool = False,
) -> Iterator[IngestedChunk]:
    """Yield validated, numeric chunks in sorted-file and original-row order.

    Numeric parse failures are counted and represented as missing in the yielded
    chunk; callers must inspect ``report.nonnumeric_values`` before fitting.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if max_rows_per_file is not None and max_rows_per_file < 1:
        raise ValueError("max_rows_per_file must be greater than zero")
    quality = report if report is not None else DataQualityReport()
    all_files = sorted(
        path for path in data_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".csv"
    ) if data_directory.is_dir() else []
    if not all_files:
        raise PreprocessingValidationError(
            f"No CSV files found in dataset directory: {data_directory}"
        )
    if selected_files is None:
        files = all_files
    else:
        if not selected_files:
            raise PreprocessingValidationError(
                "At least one CSV must be selected when --files is specified"
            )
        data_root = data_directory.resolve()
        discovered = {
            path.relative_to(data_directory).as_posix(): path
            for path in all_files
        }
        selected_paths: set[str] = set()
        for selection in selected_files:
            candidate = (data_root / selection).resolve()
            try:
                relative_path = candidate.relative_to(data_root).as_posix()
            except ValueError as error:
                raise PreprocessingValidationError(
                    f"Selected CSV must remain inside the data directory: "
                    f"{selection!r}"
                ) from error
            if relative_path not in discovered:
                raise PreprocessingValidationError(
                    f"Selected CSV does not exist beneath the data directory: "
                    f"{selection!r}"
                )
            if relative_path in selected_paths:
                raise PreprocessingValidationError(
                    f"Selected CSV was specified more than once: {selection!r}"
                )
            selected_paths.add(relative_path)
        files = [
            path for path in all_files
            if path.relative_to(data_directory).as_posix() in selected_paths
        ]

    expected_features: tuple[str, ...] | None = None
    for path in files:
        source_file = path.relative_to(data_directory).as_posix()
        if progress_callback is not None:
            progress_callback("source-file-start", source_file, quality.row_count)
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            raw_header = next(reader, None)
        if raw_header is None:
            raise PreprocessingValidationError(f"{path.name}: CSV is empty")
        validation = validate_header(raw_header)
        verified = verify_duplicate_columns(
            path,
            validation,
            row_limit=max_rows_per_file if bounded_verification else None,
        )
        if expected_features is None:
            expected_features = verified.feature_columns
            quality.feature_columns = expected_features
        elif verified.feature_columns != expected_features:
            raise PreprocessingValidationError(
                f"{path.name}: feature columns or their order differ from "
                "the first CSV"
            )

        quality.files.append(source_file)
        quality.schemas.append(
            {
                "source_file": source_file,
                "source_columns": list(raw_header),
                "feature_columns": list(verified.feature_columns),
                "label_index": validation.label_index,
                "duplicate_column_pairs": [
                    [first, second]
                    for first, second in validation.duplicate_column_pairs
                ],
            }
        )
        removed_indexes = {
            second for _, second in validation.duplicate_column_pairs
        }
        feature_indexes = [
            index for index in range(len(raw_header))
            if index != validation.label_index and index not in removed_indexes
        ]
        batch: list[list[str]] = []
        row_numbers: list[int] = []
        selected_seen: set[int] = set()
        selected_for_file = (
            selected_row_numbers.get(source_file)
            if selected_row_numbers is not None
            else None
        )
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source, strict=True)
                next(reader)
                records = enumerate(reader, start=2)
                if bounded_verification and max_rows_per_file is not None:
                    records = islice(records, max_rows_per_file)
                for record_number, row in records:
                    if (
                        not bounded_verification
                        and max_rows_per_file is not None
                        and record_number - 1 > max_rows_per_file
                    ):
                        break
                    if len(row) != len(raw_header):
                        raise PreprocessingValidationError(
                            f"{path.name}: record {record_number} has "
                            f"{len(row)} fields; expected {len(raw_header)}"
                        )
                    data_row_number = record_number - 1
                    if (
                        selected_for_file is not None
                        and data_row_number not in selected_for_file
                    ):
                        continue
                    batch.append(row)
                    row_numbers.append(data_row_number)
                    selected_seen.add(data_row_number)
                    if len(batch) == chunk_size:
                        yield _make_chunk(
                            batch, row_numbers, feature_indexes, validation.label_index,
                            verified.feature_columns, source_file, quality,
                        )
                        batch, row_numbers = [], []
                if batch:
                    yield _make_chunk(
                        batch, row_numbers, feature_indexes, validation.label_index,
                        verified.feature_columns, source_file, quality,
                    )
        except (csv.Error, UnicodeDecodeError) as error:
            raise PreprocessingValidationError(
                f"{path.name}: could not parse CSV: {error}"
            ) from error
        if selected_for_file is not None and selected_seen != set(selected_for_file):
            raise PreprocessingValidationError(
                f"{source_file}: selected source rows were not all found within "
                "the configured row cap"
            )


def _make_chunk(
    rows: list[list[str]],
    row_numbers: list[int],
    feature_indexes: list[int],
    label_index: int,
    feature_columns: tuple[str, ...],
    source_file: str,
    report: DataQualityReport,
) -> IngestedChunk:
    labels = [validate_label_text(row[label_index]) for row in rows]
    target = [encode_binary_label(label) for label in labels]
    features = pd.DataFrame(
        {
            name: pd.to_numeric(
                pd.Series([row[index].strip() for row in rows]),
                errors="coerce",
            )
            for name, index in zip(feature_columns, feature_indexes)
        }
    )
    for column, index in zip(feature_columns, feature_indexes):
        raw = [row[index].strip() for row in rows]
        numeric = features[column]
        missing_mask = np.array([_is_missing_numeric_text(value) for value in raw])
        nonnumeric_mask = numeric.isna().to_numpy() & ~missing_mask
        positive_mask = np.isposinf(numeric.to_numpy(dtype=float))
        negative_mask = np.isneginf(numeric.to_numpy(dtype=float))
        for field_name, mask in (
            ("original_missing", missing_mask),
            ("positive_infinity", positive_mask),
            ("negative_infinity", negative_mask),
            ("nonnumeric_values", nonnumeric_mask),
        ):
            count = int(mask.sum())
            if count:
                counts = getattr(report, field_name)
                counts[column] = counts.get(column, 0) + count
        finite = numeric.mask(positive_mask | negative_mask, np.nan)
        features[column] = finite
        observed = report._observed_values.setdefault(column, set())
        if len(observed) < 2:
            for value in raw:
                if _is_missing_numeric_text(value):
                    continue
                try:
                    decimal_value = Decimal(value)
                except InvalidOperation:
                    continue
                if decimal_value.is_finite():
                    observed.add(decimal_value)
                    if len(observed) == 2:
                        break

    for label in target:
        report.class_counts[label] = report.class_counts.get(label, 0) + 1
    report.row_count += len(rows)
    return IngestedChunk(
        features=features,
        target=pd.Series(target, name="target", dtype="int8"),
        label_original=pd.Series(labels, name="label_original", dtype="object"),
        source_file=source_file,
        source_row_numbers=tuple(row_numbers),
        canonical_feature_rows=tuple(
            _canonical_tokens(
                [row[index] for index in feature_indexes]
            )
            for row in rows
        ),
    )


def validate_data_quality(report: DataQualityReport) -> None:
    """Stop modeling when feature text cannot be interpreted numerically."""
    if report.nonnumeric_values:
        details = ", ".join(
            f"{column}: {count}" for column, count in sorted(report.nonnumeric_values.items())
        )
        raise PreprocessingValidationError(
            f"Unexpected nonnumeric feature values were found ({details})"
        )


def _canonical_feature_row(values: Sequence[object]) -> bytes:
    return _canonical_tokens([str(value) for value in values])


def _feature_digest(canonical: bytes) -> bytes:
    return hashlib.blake2b(canonical, digest_size=16).digest()


class ExactFeatureGrouper:
    """Incrementally group identical cleaned numeric feature rows.

    Only one canonical byte string per unique feature vector is retained. Digest
    matches are candidates only; canonical bytes are always compared exactly.
    """

    def __init__(self, feature_columns: Sequence[str]):
        self.feature_columns = tuple(feature_columns)
        self.groups: dict[int, FeatureGroup] = {}
        self._candidates: dict[bytes, list[tuple[bytes, int]]] = defaultdict(list)

    def add(
        self,
        features: pd.DataFrame,
        target: Sequence[int],
        label_original: Sequence[str],
    ) -> np.ndarray:
        if tuple(features.columns) != self.feature_columns:
            raise PreprocessingValidationError(
                "Feature columns or their order changed during grouping"
            )
        if len(features) != len(target) or len(features) != len(label_original):
            raise PreprocessingValidationError(
                "Features, binary targets, and original labels must have equal lengths"
            )
        assignments = np.empty(len(features), dtype=np.int64)
        for row_index, values in enumerate(features.itertuples(index=False, name=None)):
            canonical = _canonical_feature_row(values)
            digest = _feature_digest(canonical)
            group_id = next(
                (
                    candidate_id
                    for candidate_canonical, candidate_id in self._candidates[digest]
                    if candidate_canonical == canonical
                ),
                None,
            )
            if group_id is None:
                group_id = len(self.groups)
                self._candidates[digest].append((canonical, group_id))
                self.groups[group_id] = FeatureGroup(group_id)

            binary_label = int(target[row_index])
            if binary_label not in (0, 1):
                raise PreprocessingValidationError(
                    f"Binary target must be 0 or 1; found {binary_label}"
                )
            original_label = validate_label_text(str(label_original[row_index]))
            group = self.groups[group_id]
            group.row_count += 1
            group.class_counts[binary_label] += 1
            group.original_labels.add(original_label)
            assignments[row_index] = group_id
        return assignments

    @property
    def conflicting_groups(self) -> dict[int, tuple[str, ...]]:
        return {
            group_id: tuple(sorted(group.original_labels))
            for group_id, group in self.groups.items()
            if group.conflicting_binary_labels
        }


PARTITION_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def split_feature_groups(
    grouper: ExactFeatureGrouper,
    seed: int = 42,
) -> GroupSplit:
    """Assign each feature group wholly to one deterministic partition."""
    if not grouper.groups:
        raise PreprocessingValidationError("Cannot split an empty feature dataset")

    class_group_counts = {
        label: sum(group.class_counts[label] > 0 for group in grouper.groups.values())
        for label in (0, 1)
    }
    unavailable = [label for label, count in class_group_counts.items() if count < 3]
    if unavailable:
        raise PreprocessingValidationError(
            "Cannot place both binary classes in train, validation, and test while "
            "keeping feature groups isolated; fewer than three distinct groups "
            f"contain class(es) {unavailable}"
        )

    total_rows = sum(group.row_count for group in grouper.groups.values())
    total_classes = {
        label: sum(group.class_counts[label] for group in grouper.groups.values())
        for label in (0, 1)
    }
    names = tuple(PARTITION_FRACTIONS)
    desired = {
        name: {
            "rows": total_rows * fraction,
            0: total_classes[0] * fraction,
            1: total_classes[1] * fraction,
        }
        for name, fraction in PARTITION_FRACTIONS.items()
    }
    actual = {
        name: {"rows": 0, 0: 0, 1: 0}
        for name in names
    }

    rng = random.Random(seed)
    tie_breakers = {group_id: rng.random() for group_id in grouper.groups}
    ordered_groups = sorted(
        grouper.groups.values(),
        key=lambda group: (-group.row_count, tie_breakers[group.group_id]),
    )
    groups_by_label_set = {
        label_set: sorted(
            (
                group for group in grouper.groups.values()
                if tuple(
                    label for label in (0, 1) if group.class_counts[label] > 0
                ) == label_set
            ),
            key=lambda group: (group.row_count, tie_breakers[group.group_id]),
        )
        for label_set in ((0,), (1,), (0, 1))
    }
    both_class_groups = groups_by_label_set[(0, 1)]
    benign_only_groups = groups_by_label_set[(0,)]
    attack_only_groups = groups_by_label_set[(1,)]

    # Class coverage is feasible exactly when each partition can be seeded
    # with either a mixed-label group or one group of each single class.
    mixed_partitions = min(len(both_class_groups), len(names))
    remaining_partitions = len(names) - mixed_partitions
    if (
        len(benign_only_groups) < remaining_partitions
        or len(attack_only_groups) < remaining_partitions
    ):
        raise PreprocessingValidationError(
            "Cannot assign feature groups so every train, validation, and test "
            "partition contains both binary classes"
        )

    assignments: dict[int, str] = {}
    for partition, group in zip(names, both_class_groups[:mixed_partitions]):
        assignments[group.group_id] = partition
        actual[partition]["rows"] += group.row_count
        actual[partition][0] += group.class_counts[0]
        actual[partition][1] += group.class_counts[1]

    uncovered_partitions = names[mixed_partitions:]
    for partition, benign_group, attack_group in zip(
        uncovered_partitions,
        benign_only_groups[:remaining_partitions],
        attack_only_groups[:remaining_partitions],
    ):
        for group in (benign_group, attack_group):
            assignments[group.group_id] = partition
            actual[partition]["rows"] += group.row_count
            actual[partition][0] += group.class_counts[0]
            actual[partition][1] += group.class_counts[1]

    seeded_group_ids = set(assignments)
    for group in ordered_groups:
        if group.group_id in seeded_group_ids:
            continue
        scores: list[tuple[float, str]] = []
        for partition in names:
            score = 0.0
            for dimension, amount in (
                ("rows", group.row_count),
                (0, group.class_counts[0]),
                (1, group.class_counts[1]),
            ):
                target_amount = desired[partition][dimension]
                scale = max(target_amount, 1.0)
                before = (actual[partition][dimension] - target_amount) / scale
                after = (
                    actual[partition][dimension] + amount - target_amount
                ) / scale
                score += after * after - before * before
            scores.append((score, partition))
        _, selected = min(scores)
        assignments[group.group_id] = selected
        actual[selected]["rows"] += group.row_count
        actual[selected][0] += group.class_counts[0]
        actual[selected][1] += group.class_counts[1]

    missing_classes = {
        partition: [
            label for label in (0, 1) if actual[partition][label] == 0
        ]
        for partition in names
    }
    missing_classes = {
        partition: labels for partition, labels in missing_classes.items() if labels
    }
    if missing_classes:
        raise PreprocessingValidationError(
            "Group-aware split could not place both binary classes in every "
            f"partition without leakage: {missing_classes}"
        )

    partition_ids = {
        name: tuple(
            group_id for group_id, partition in assignments.items()
            if partition == name
        )
        for name in names
    }
    flattened_group_ids = [
        group_id for group_ids in partition_ids.values() for group_id in group_ids
    ]
    if (
        len(flattened_group_ids) != len(grouper.groups)
        or set(flattened_group_ids) != set(grouper.groups)
        or any(
            assignments[group_id] != partition
            for partition, group_ids in partition_ids.items()
            for group_id in group_ids
        )
    ):
        raise PreprocessingValidationError(
            "Feature-group separation validation failed"
        )
    row_counts = {
        name: actual[name]["rows"]
        for name in names
    }
    class_counts = {
        name: {label: actual[name][label] for label in (0, 1)}
        for name in names
    }
    if sum(row_counts.values()) != total_rows:
        raise PreprocessingValidationError(
            "Partition row counts do not match the ingested row count"
        )
    if len(assignments) != len(grouper.groups):
        raise PreprocessingValidationError(
            "Some feature groups were omitted from the partition assignment"
        )
    return GroupSplit(assignments, partition_ids, row_counts, class_counts)


def _quality_manifest(report: DataQualityReport) -> dict[str, object]:
    return {
        "row_count": report.row_count,
        "files": report.files,
        "schemas": report.schemas,
        "original_missing": report.original_missing,
        "positive_infinity": report.positive_infinity,
        "negative_infinity": report.negative_infinity,
        "nonnumeric_values": report.nonnumeric_values,
        "constant_columns": list(report.constant_columns),
        "binary_class_counts": report.class_counts,
    }


def _exact_group_search(
    groups: Sequence[tuple[int, int, int, int]],
    targets: tuple[int, int, int],
    singleton_counts: tuple[int, int],
    node_limit: int,
    require_both_classes: bool,
) -> tuple[list[int] | None, tuple[int, int, int] | None, int, bool]:
    """Search duplicate groups exactly; singleton groups are filled by class."""
    suffix_rows = [0] * (len(groups) + 1)
    suffix_benign_groups = [0] * (len(groups) + 1)
    suffix_attack_groups = [0] * (len(groups) + 1)
    for index in range(len(groups) - 1, -1, -1):
        _, row_count, benign_count, attack_count = groups[index]
        suffix_rows[index] = suffix_rows[index + 1] + row_count
        suffix_benign_groups[index] = (
            suffix_benign_groups[index + 1] + int(benign_count > 0)
        )
        suffix_attack_groups[index] = (
            suffix_attack_groups[index + 1] + int(attack_count > 0)
        )

    def state_key(
        index: int,
        remaining: tuple[int, int, int],
        assigned_benign: tuple[int, int, int],
    ) -> tuple[object, ...]:
        if targets[1] == targets[2]:
            validation_test = sorted(
                (
                    (remaining[1], assigned_benign[1]),
                    (remaining[2], assigned_benign[2]),
                )
            )
            return (
                index,
                remaining[0],
                assigned_benign[0],
                tuple(validation_test),
            )
        return (index, remaining, assigned_benign)

    def singleton_fill(
        remaining: tuple[int, int, int],
        assigned_benign: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
        assigned_attack = tuple(
            targets[index] - remaining[index] - assigned_benign[index]
            for index in range(3)
        )
        if any(value < 0 for value in assigned_attack):
            return None
        lower: list[int] = []
        upper: list[int] = []
        for index in range(3):
            min_benign = int(
                require_both_classes and assigned_benign[index] == 0
            )
            min_attack = int(
                require_both_classes and assigned_attack[index] == 0
            )
            lower.append(min_benign)
            upper.append(remaining[index] - min_attack)
            if upper[-1] < lower[-1]:
                return None
        if sum(remaining) != sum(singleton_counts):
            return None
        if not sum(lower) <= singleton_counts[0] <= sum(upper):
            return None
        benign_allocation = list(lower)
        left = singleton_counts[0] - sum(benign_allocation)
        for index in range(3):
            amount = min(left, upper[index] - benign_allocation[index])
            benign_allocation[index] += amount
            left -= amount
        if left:
            return None
        return tuple(benign_allocation)

    def class_coverage_still_possible(
        index: int,
        remaining: tuple[int, int, int],
        assigned_benign: tuple[int, int, int],
    ) -> bool:
        if not require_both_classes:
            return True
        assigned_attack = tuple(
            targets[position] - remaining[position] - assigned_benign[position]
            for position in range(3)
        )
        if any(value < 0 for value in assigned_attack):
            return False
        missing_benign = sum(value == 0 for value in assigned_benign)
        missing_attack = sum(value == 0 for value in assigned_attack)
        if any(
            remaining[position]
            < int(assigned_benign[position] == 0)
            + int(assigned_attack[position] == 0)
            for position in range(3)
        ):
            return False
        available_benign = singleton_counts[0] + sum(
            group[2] for group in groups[index:]
        )
        available_attack = singleton_counts[1] + sum(
            group[3] for group in groups[index:]
        )
        if available_benign < missing_benign or available_attack < missing_attack:
            return False
        if singleton_counts[0] == 0 and suffix_benign_groups[index] < missing_benign:
            return False
        if singleton_counts[1] == 0 and suffix_attack_groups[index] < missing_attack:
            return False
        return True

    assignments = [-1] * len(groups)
    dead_states: set[tuple[object, ...]] = set()
    initial_remaining = targets
    stack: list[list[object]] = [[0, initial_remaining, (0, 0, 0), 0, False]]
    explored = 0
    while stack:
        frame = stack[-1]
        index = int(frame[0])
        remaining = frame[1]
        assigned_benign = frame[2]
        next_partition = int(frame[3])
        if not isinstance(remaining, tuple) or not isinstance(assigned_benign, tuple):
            raise RuntimeError("Exact split search state is malformed")
        key = state_key(index, remaining, assigned_benign)
        if not bool(frame[4]):
            if explored >= node_limit:
                return None, None, explored, True
            explored += 1
            if key in dead_states or not class_coverage_still_possible(
                index,
                remaining,
                assigned_benign,
            ):
                dead_states.add(key)
                stack.pop()
                continue
            if index == len(groups):
                fill = singleton_fill(remaining, assigned_benign)
                if fill is not None:
                    return assignments, fill, explored, False
                dead_states.add(key)
                stack.pop()
                continue
            frame[4] = True
        if next_partition >= 3:
            dead_states.add(key)
            stack.pop()
            continue

        partition = next_partition
        frame[3] = partition + 1
        _, row_count, benign_count, _ = groups[index]
        if row_count > remaining[partition]:
            continue
        next_remaining = list(remaining)
        next_remaining[partition] -= row_count
        next_benign = list(assigned_benign)
        next_benign[partition] += benign_count
        new_remaining = tuple(next_remaining)
        new_benign = tuple(next_benign)
        if suffix_rows[index + 1] + sum(singleton_counts) != sum(new_remaining):
            continue
        child_key = state_key(index + 1, new_remaining, new_benign)
        if child_key in dead_states:
            continue
        assignments[index] = partition
        stack.append(
            [index + 1, new_remaining, new_benign, 0, False]
        )
    return None, None, explored, False


def _split_sqlite_groups_exact(
    connection: sqlite3.Connection,
    seed: int,
    target_rows: Mapping[str, int],
    node_limit: int,
) -> tuple[dict[str, int], dict[str, dict[int, int]], dict[str, object]]:
    """Assign complete groups to exact row capacities with a deterministic limit."""
    partition_names = ("train", "validation", "test")
    targets = tuple(int(target_rows[name]) for name in partition_names)
    if any(target < 1 for target in targets) or node_limit < 1:
        raise PreprocessingValidationError(
            "Exact split targets and search-node limit must be greater than zero"
        )
    total_rows = int(connection.execute(
        "SELECT COALESCE(SUM(row_count), 0) FROM feature_groups"
    ).fetchone()[0])
    if sum(targets) != total_rows:
        raise PreprocessingValidationError(
            "INFEASIBLE: requested exact partition capacities "
            f"{dict(zip(partition_names, targets))} total {sum(targets)} rows, "
            f"but the selected dataset contains {total_rows}"
        )

    class_group_counts = {
        label: int(connection.execute(
            "SELECT COUNT(*) FROM feature_groups WHERE "
            + ("benign_count > 0" if label == 0 else "attack_count > 0")
        ).fetchone()[0])
        for label in (0, 1)
    }
    histogram = {
        int(size): int(count)
        for size, count in connection.execute(
            """
            SELECT row_count, COUNT(*)
            FROM feature_groups
            GROUP BY row_count
            ORDER BY row_count
            """
        )
    }
    too_large = [
        size for size in histogram if size > max(targets)
    ]
    if too_large:
        raise PreprocessingValidationError(
            "INFEASIBLE: indivisible feature groups exceed every partition "
            f"capacity; blocking group sizes={too_large}, targets="
            f"{dict(zip(partition_names, targets))}, group_size_counts={histogram}"
        )

    singleton_counts = (
        int(connection.execute(
            "SELECT COALESCE(SUM(benign_count), 0) FROM feature_groups "
            "WHERE row_count = 1"
        ).fetchone()[0]),
        int(connection.execute(
            "SELECT COALESCE(SUM(attack_count), 0) FROM feature_groups "
            "WHERE row_count = 1"
        ).fetchone()[0]),
    )
    groups = [
        tuple(map(int, row))
        for row in connection.execute(
            """
            SELECT group_id, row_count, benign_count, attack_count
            FROM feature_groups
            WHERE row_count > 1
            ORDER BY row_count DESC, split_tie_key(?, group_id)
            """,
            (seed,),
        )
    ]
    if sum(group[1] for group in groups) + sum(singleton_counts) != total_rows:
        raise PreprocessingValidationError(
            "Feature-group statistics do not match the selected row count"
        )

    require_both_classes = all(count >= 3 for count in class_group_counts.values())
    assignments, singleton_benign, nodes, limited = _exact_group_search(
        groups,
        targets,
        singleton_counts,
        node_limit,
        require_both_classes,
    )
    if limited:
        raise PreprocessingValidationError(
            "SOLVER_LIMIT: exact group assignment reached the configured "
            f"search-node limit ({node_limit}); result is INCONCLUSIVE, not "
            f"proven infeasible. Explored {nodes} nodes; group_size_counts={histogram}"
        )
    if assignments is None or singleton_benign is None:
        if require_both_classes:
            assignments, singleton_benign, extra_nodes, limited = _exact_group_search(
                groups,
                targets,
                singleton_counts,
                node_limit - nodes,
                False,
            )
            nodes += extra_nodes
        if limited:
            raise PreprocessingValidationError(
                "SOLVER_LIMIT: exact group assignment reached the configured "
                f"search-node limit ({node_limit}); result is INCONCLUSIVE, not "
                f"proven infeasible. Explored {nodes} nodes; group_size_counts={histogram}"
            )
    if assignments is None or singleton_benign is None:
        raise PreprocessingValidationError(
            "INFEASIBLE: no whole-feature-group assignment satisfies exact "
            f"partition capacities {dict(zip(partition_names, targets))}; "
            f"group_size_counts={histogram}, class_group_counts={class_group_counts}"
        )

    connection.execute("DELETE FROM group_partitions")
    connection.executemany(
        "INSERT INTO group_partitions(group_id, partition) VALUES (?, ?)",
        [
            (group[0], partition_names[partition_index])
            for group, partition_index in zip(groups, assignments)
        ],
    )
    singleton_attack = tuple(
        targets[index]
        - sum(
            group[1]
            for group, partition_index in zip(groups, assignments)
            if partition_index == index
        )
        - singleton_benign[index]
        for index in range(3)
    )
    single_allocations = {
        0: singleton_benign,
        1: singleton_attack,
    }
    for label in (0, 1):
        condition = "benign_count = 1" if label == 0 else "attack_count = 1"
        cursor = connection.execute(
            f"""
            SELECT group_id FROM feature_groups
            WHERE row_count = 1 AND {condition}
            ORDER BY split_tie_key(?, group_id)
            """,
            (seed + label,),
        )
        allocation_index = 0
        remaining_for_partition = single_allocations[label][0]
        for (group_id,) in cursor:
            while (
                remaining_for_partition == 0
                and allocation_index < 2
            ):
                allocation_index += 1
                remaining_for_partition = single_allocations[label][allocation_index]
            if allocation_index >= 3 or remaining_for_partition <= 0:
                raise PreprocessingValidationError(
                    "Exact split singleton allocation does not match group counts"
                )
            connection.execute(
                "INSERT INTO group_partitions(group_id, partition) VALUES (?, ?)",
                (group_id, partition_names[allocation_index]),
            )
            remaining_for_partition -= 1
        if allocation_index < 3 and remaining_for_partition:
            raise PreprocessingValidationError(
                "Exact split did not assign every singleton group"
            )
    connection.commit()

    assignment_count = int(connection.execute(
        "SELECT COUNT(*) FROM group_partitions"
    ).fetchone()[0])
    group_count = int(connection.execute(
        "SELECT COUNT(*) FROM feature_groups"
    ).fetchone()[0])
    if assignment_count != group_count:
        raise PreprocessingValidationError(
            "Exact group split did not assign every feature group"
        )
    actual_rows = {name: 0 for name in partition_names}
    actual_classes = {
        name: {0: 0, 1: 0}
        for name in partition_names
    }
    for partition, rows, benign, attack in connection.execute(
        """
        SELECT p.partition, SUM(g.row_count), SUM(g.benign_count),
               SUM(g.attack_count)
        FROM group_partitions AS p
        JOIN feature_groups AS g ON g.group_id = p.group_id
        GROUP BY p.partition
        """
    ):
        actual_rows[str(partition)] = int(rows)
        actual_classes[str(partition)][0] = int(benign)
        actual_classes[str(partition)][1] = int(attack)
    if actual_rows != dict(zip(partition_names, targets)):
        raise PreprocessingValidationError(
            "Exact group split validation failed: "
            f"expected={dict(zip(partition_names, targets))}, actual={actual_rows}"
        )
    class_coverage = all(
        actual_classes[partition][label] > 0
        for partition in partition_names
        for label in (0, 1)
    )
    if require_both_classes and not class_coverage:
        raise PreprocessingValidationError(
            "Exact split search claimed class coverage but validation failed"
        )
    return (
        actual_rows,
        actual_classes,
        {
            "status": "FEASIBLE",
            "search_nodes": nodes,
            "search_node_limit": node_limit,
            "both_classes_in_every_partition": class_coverage,
            "group_size_counts": {
                str(size): count for size, count in histogram.items()
            },
        },
    )


def _split_sqlite_groups(
    connection: sqlite3.Connection,
    seed: int,
) -> tuple[dict[str, int], dict[str, dict[int, int]]]:
    partition_names = tuple(PARTITION_FRACTIONS)
    class_only_counts = {
        0: int(connection.execute(
            "SELECT COUNT(*) FROM feature_groups "
            "WHERE benign_count > 0 AND attack_count = 0"
        ).fetchone()[0]),
        1: int(connection.execute(
            "SELECT COUNT(*) FROM feature_groups "
            "WHERE attack_count > 0 AND benign_count = 0"
        ).fetchone()[0]),
    }
    mixed_count = int(connection.execute(
        "SELECT COUNT(*) FROM feature_groups "
        "WHERE benign_count > 0 AND attack_count > 0"
    ).fetchone()[0])
    mixed_partition_count = min(mixed_count, len(partition_names))
    remaining_partition_count = len(partition_names) - mixed_partition_count
    if any(count < remaining_partition_count for count in class_only_counts.values()):
        raise PreprocessingValidationError(
            "Cannot assign feature groups so every train, validation, and test "
            "partition contains both binary classes"
        )

    rows_total = int(connection.execute(
        "SELECT COALESCE(SUM(row_count), 0) FROM feature_groups"
    ).fetchone()[0])
    class_totals = {
        0: int(connection.execute(
            "SELECT COALESCE(SUM(benign_count), 0) FROM feature_groups"
        ).fetchone()[0]),
        1: int(connection.execute(
            "SELECT COALESCE(SUM(attack_count), 0) FROM feature_groups"
        ).fetchone()[0]),
    }
    desired = {
        partition: {
            "rows": rows_total * fraction,
            0: class_totals[0] * fraction,
            1: class_totals[1] * fraction,
        }
        for partition, fraction in PARTITION_FRACTIONS.items()
    }
    actual = {
        partition: {"rows": 0, 0: 0, 1: 0}
        for partition in partition_names
    }

    mixed_cursor = connection.execute(
        """
        SELECT group_id, row_count, benign_count, attack_count
        FROM feature_groups
        WHERE benign_count > 0 AND attack_count > 0
        ORDER BY row_count, split_tie_key(?, group_id)
        LIMIT ?
        """,
        (seed, mixed_partition_count),
    )
    for partition, (group_id, row_count, benign_count, attack_count) in zip(
        partition_names,
        mixed_cursor,
    ):
        connection.execute(
            "INSERT INTO group_partitions(group_id, partition) VALUES (?, ?)",
            (group_id, partition),
        )
        actual[partition]["rows"] += int(row_count)
        actual[partition][0] += int(benign_count)
        actual[partition][1] += int(attack_count)

    uncovered = partition_names[mixed_partition_count:]
    class_only_groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for label in (0, 1):
        condition = (
            "benign_count > 0 AND attack_count = 0"
            if label == 0
            else "attack_count > 0 AND benign_count = 0"
        )
        cursor = connection.execute(
            f"""
            SELECT group_id, row_count, benign_count, attack_count
            FROM feature_groups
            WHERE {condition}
            ORDER BY row_count, split_tie_key(?, group_id)
            LIMIT ?
            """,
            (seed, remaining_partition_count),
        )
        class_only_groups[label] = [
            tuple(map(int, row))
            for row in cursor
        ]
    for partition_index, partition in enumerate(uncovered):
        for label in (0, 1):
            group_id, row_count, benign_count, attack_count = (
                class_only_groups[label][partition_index]
            )
            connection.execute(
                "INSERT INTO group_partitions(group_id, partition) VALUES (?, ?)",
                (group_id, partition),
            )
            actual[partition]["rows"] += row_count
            actual[partition][0] += benign_count
            actual[partition][1] += attack_count
    connection.commit()

    connection.execute(
        """
        CREATE TEMP TABLE remaining_split_order (
            group_id INTEGER PRIMARY KEY
        )
        """
    )
    connection.execute(
        """
        INSERT INTO remaining_split_order(group_id)
        SELECT g.group_id
        FROM feature_groups AS g
        LEFT JOIN group_partitions AS p ON p.group_id = g.group_id
        WHERE p.group_id IS NULL
        ORDER BY g.row_count DESC, split_tie_key(?, g.group_id)
        """,
        (seed,),
    )
    cursor = connection.execute(
        """
        SELECT g.group_id, g.row_count, g.benign_count, g.attack_count
        FROM remaining_split_order AS o
        JOIN feature_groups AS g ON g.group_id = o.group_id
        ORDER BY g.row_count DESC, split_tie_key(?, g.group_id)
        """,
        (seed,),
    )
    for group_id, row_count, benign_count, attack_count in cursor:
        group_id = int(group_id)
        row_count = int(row_count)
        benign_count = int(benign_count)
        attack_count = int(attack_count)
        scores: list[tuple[float, str]] = []
        for partition in partition_names:
            score = 0.0
            for dimension, amount in (
                ("rows", row_count),
                (0, benign_count),
                (1, attack_count),
            ):
                target_amount = desired[partition][dimension]
                scale = max(target_amount, 1.0)
                before = (actual[partition][dimension] - target_amount) / scale
                after = (
                    actual[partition][dimension] + amount - target_amount
                ) / scale
                score += after * after - before * before
            scores.append((score, partition))
        _, selected = min(scores)
        connection.execute(
            "INSERT INTO group_partitions(group_id, partition) VALUES (?, ?)",
            (group_id, selected),
        )
        actual[selected]["rows"] += row_count
        actual[selected][0] += benign_count
        actual[selected][1] += attack_count
    connection.commit()
    connection.execute("DROP TABLE remaining_split_order")

    group_count = int(connection.execute(
        "SELECT COUNT(*) FROM feature_groups"
    ).fetchone()[0])
    assignment_count = int(connection.execute(
        "SELECT COUNT(*) FROM group_partitions"
    ).fetchone()[0])
    if group_count == 0 or assignment_count != group_count:
        raise PreprocessingValidationError(
            "Group-aware split did not assign every feature group"
        )
    missing_classes = {
        partition: [
            label for label in (0, 1) if actual[partition][label] == 0
        ]
        for partition in partition_names
    }
    missing_classes = {
        partition: labels for partition, labels in missing_classes.items() if labels
    }
    if missing_classes:
        raise PreprocessingValidationError(
            "Group-aware split could not place both binary classes in every "
            f"partition without leakage: {missing_classes}"
        )
    if sum(counts["rows"] for counts in actual.values()) != rows_total:
        raise PreprocessingValidationError(
            "Partition row counts do not match the ingested row count"
        )
    return (
        {partition: actual[partition]["rows"] for partition in partition_names},
        {
            partition: {
                label: actual[partition][label]
                for label in (0, 1)
            }
            for partition in partition_names
        },
    )


def _training_medians(
    connection: sqlite3.Connection,
    feature_columns: Sequence[str],
) -> tuple[float, ...]:
    medians: list[float] = []
    for feature_index, feature_name in enumerate(feature_columns):
        cursor = connection.execute(
            """
            SELECT f.value_text, g.row_count
            FROM group_features AS f
            JOIN feature_groups AS g ON g.group_id = f.group_id
            JOIN group_partitions AS p ON p.group_id = g.group_id
            WHERE p.partition = 'train'
              AND f.feature_index = ?
              AND f.value_text IS NOT NULL
            ORDER BY f.value_text COLLATE DECIMAL, g.group_id
            """,
            (feature_index,),
        )
        total = sum(int(count) for _, count in cursor)
        if total == 0:
            raise PreprocessingValidationError(
                "Median imputation is undefined for features with no training "
                f"observations: {[feature_name]}"
            )
        lower_index = (total - 1) // 2
        upper_index = total // 2
        lower_value: Decimal | None = None
        upper_value: Decimal | None = None
        cumulative = 0
        for value_text, count in connection.execute(
            """
            SELECT f.value_text, g.row_count
            FROM group_features AS f
            JOIN feature_groups AS g ON g.group_id = f.group_id
            JOIN group_partitions AS p ON p.group_id = g.group_id
            WHERE p.partition = 'train'
              AND f.feature_index = ?
              AND f.value_text IS NOT NULL
            ORDER BY f.value_text COLLATE DECIMAL, g.group_id
            """,
            (feature_index,),
        ):
            value = Decimal(value_text)
            next_cumulative = cumulative + int(count)
            if lower_value is None and cumulative <= lower_index < next_cumulative:
                lower_value = value
            if cumulative <= upper_index < next_cumulative:
                upper_value = value
                break
            cumulative = next_cumulative
        if lower_value is None or upper_value is None:
            raise PreprocessingValidationError(
                f"Could not compute the training median for {feature_name!r}"
            )
        medians.append(float((lower_value + upper_value) / 2))
    return tuple(medians)


def _fit_streaming_scaler(
    connection: sqlite3.Connection,
    feature_columns: Sequence[str],
    medians: Sequence[float],
    batch_size: int,
) -> StandardScaler:
    scaler = StandardScaler()
    columns_sql = ", ".join(
        f"MAX(CASE WHEN f.feature_index = {index} "
        f"THEN f.value_text END) AS feature_{index}"
        for index in range(len(feature_columns))
    )
    cursor = connection.execute(
        f"""
        SELECT g.group_id, g.row_count, {columns_sql}
        FROM feature_groups AS g
        JOIN group_partitions AS p ON p.group_id = g.group_id
        LEFT JOIN group_features AS f ON f.group_id = g.group_id
        WHERE p.partition = 'train'
        GROUP BY g.group_id
        ORDER BY g.group_id
        """
    )
    while rows := cursor.fetchmany(batch_size):
        matrix: list[list[float]] = []
        weights: list[int] = []
        for _, row_count, *values in rows:
            matrix.append(
                [
                    medians[index]
                    if value is None
                    else float(Decimal(value))
                    for index, value in enumerate(values)
                ]
            )
            weights.append(int(row_count))
        scaler.partial_fit(
            np.asarray(matrix, dtype=np.float64),
            sample_weight=np.asarray(weights, dtype=np.float64),
        )
    if not hasattr(scaler, "mean_"):
        raise PreprocessingValidationError(
            "Could not fit the scaler because the training partition is empty"
        )
    return scaler


def _write_transformed_partition(
    connection: sqlite3.Connection,
    pipeline: Pipeline,
    feature_columns: Sequence[str],
    partition: str,
    destination: Path,
    chunk_size: int,
) -> tuple[int, dict[int, int]]:
    cursor = connection.execute(
        """
        SELECT r.target, r.label_original, r.source_file, r.source_row_number,
               g.canonical
        FROM source_rows AS r
        JOIN group_partitions AS p ON p.group_id = r.group_id
        JOIN feature_groups AS g ON g.group_id = r.group_id
        WHERE p.partition = ?
        ORDER BY r.row_id
        """,
        (partition,),
    )
    write_header = True
    written_rows = 0
    written_classes = {0: 0, 1: 0}
    while rows := cursor.fetchmany(chunk_size):
        targets: list[int] = []
        labels: list[str] = []
        source_files: list[str] = []
        source_rows: list[int] = []
        feature_values: list[list[float]] = []
        for target, label, source_file, source_row, canonical in rows:
            tokens = canonical.split(b";")
            if len(tokens) != len(feature_columns):
                raise PreprocessingValidationError(
                    "Stored feature row does not match the output schema"
                )
            values = []
            for token in tokens:
                value = _canonical_value_from_token(token)
                values.append(np.nan if value is None else float(value))
            targets.append(int(target))
            labels.append(str(label))
            source_files.append(str(source_file))
            source_rows.append(int(source_row))
            feature_values.append(values)

        for target, label in zip(targets, labels):
            if encode_binary_label(label) != target:
                raise PreprocessingValidationError(
                    "Written original label does not match its binary target"
                )
            written_classes[target] += 1
        features = pd.DataFrame(feature_values, columns=feature_columns)
        transformed = pipeline.transform(features)
        if not np.isfinite(transformed).all():
            raise PreprocessingValidationError(
                f"Transformed {partition} features contain NaN or infinity"
            )
        output = pd.DataFrame(transformed, columns=feature_columns)
        output.insert(0, "label_original", labels)
        output.insert(0, "target", targets)
        output.insert(0, "source_row_number", source_rows)
        output.insert(0, "source_file", source_files)
        output.to_csv(
            destination,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
            encoding="utf-8",
        )
        write_header = False
        written_rows += len(output)
    if write_header:
        raise PreprocessingValidationError(
            f"The {partition} partition unexpectedly contains no rows"
        )
    return written_rows, written_classes


def run_preprocessing(
    data_directory: Path,
    output_directory: Path | None = None,
    chunk_size: int = 100_000,
    seed: int = 42,
    selected_files: Sequence[str] | None = None,
    max_rows_per_file: int | None = None,
    min_free_space_bytes: int = 0,
    pilot_mode: bool = False,
    sample_10k: bool = False,
    candidate_row_cap: int = 25_000,
    solver_node_limit: int = 250_000,
) -> Path:
    """Run disk-backed, bounded-batch preprocessing and atomically publish artifacts."""
    project_output = Path(__file__).resolve().parents[1] / "outputs"
    normal_output_directory = project_output / "preprocessing"
    pilot_output_directory = project_output / "preprocessing-pilot"
    sample_output_directory = project_output / "preprocessing-10k"
    if output_directory is None:
        output_directory = (
            pilot_output_directory
            if pilot_mode
            else sample_output_directory
            if sample_10k
            else normal_output_directory
        )
    if sample_10k and (pilot_mode or selected_files or max_rows_per_file is not None):
        raise PreprocessingValidationError(
            "10k sample mode cannot be combined with --pilot, --files, or "
            "--max-rows-per-file"
        )
    if pilot_mode and _is_within(output_directory, normal_output_directory):
        raise PreprocessingValidationError(
            "Pilot output must be separate from outputs/preprocessing/"
        )
    if sample_10k and (
        _paths_overlap(output_directory, normal_output_directory)
        or _paths_overlap(output_directory, pilot_output_directory)
    ):
        raise PreprocessingValidationError(
            "10k sample output must be isolated from the normal and pilot "
            "preprocessing directories"
        )
    if sample_10k and candidate_row_cap < 1:
        raise PreprocessingValidationError(
            "candidate_row_cap must be greater than zero"
        )
    if sample_10k and solver_node_limit < 1:
        raise PreprocessingValidationError(
            "solver_node_limit must be greater than zero"
        )
    if pilot_mode:
        if not selected_files:
            raise PreprocessingValidationError(
                "Pilot mode requires an explicit subset of CSV files"
            )
        available_files = [
            path for path in data_directory.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".csv"
        ] if data_directory.is_dir() else []
        if len(set(selected_files)) >= len(available_files):
            raise PreprocessingValidationError(
                "Pilot mode must select fewer CSV files than are present; "
                "the full dataset is not allowed in pilot mode"
            )

    output_existed = output_directory.exists()
    run_started = time.monotonic()
    current_stage = "preflight"
    current_file: str | None = None
    staging_directory: Path | None = None
    connection: sqlite3.Connection | None = None
    preflight_free_bytes: int | None = None
    selected_row_numbers: Mapping[str, frozenset[int]] | None = None
    sampling_source_files: tuple[str, ...] | None = None
    sampling_source_signatures: dict[str, tuple[int, int]] | None = None
    sampling_manifest: dict[str, object] | None = None
    destination = output_directory / (
        "run-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + os.urandom(4).hex()
    )
    report = DataQualityReport()
    measurements: list[dict[str, object]] = []

    def start_stage(stage: str) -> None:
        nonlocal current_stage, current_file
        current_stage = stage
        current_file = None
        logger.info(
            "Preprocessing stage=%s rows_processed=%d elapsed_seconds=%.3f",
            stage,
            report.row_count,
            time.monotonic() - run_started,
        )

    def record_measurement(stage: str) -> None:
        if pilot_mode and staging_directory is not None:
            _record_pilot_measurement(
                measurements,
                stage,
                staging_directory,
                output_directory,
                run_started,
                report.row_count,
            )

    def ingestion_progress(event: str, source_file: str, rows: int) -> None:
        nonlocal current_file
        current_file = source_file
        if event == "source-file-start":
            logger.info(
                "Preprocessing stage=ingestion source_file=%s "
                "rows_processed=%d elapsed_seconds=%.3f",
                source_file,
                rows,
                time.monotonic() - run_started,
            )

    def sampling_progress(event: str, source_file: str, rows: int) -> None:
        nonlocal current_file
        current_file = source_file
        logger.info(
            "Preprocessing stage=sampling source_file=%s event=%s "
            "candidate_rows_inspected=%d elapsed_seconds=%.3f",
            source_file,
            event,
            rows,
            time.monotonic() - run_started,
        )

    try:
        if chunk_size < 1:
            raise PreprocessingValidationError(
                "chunk_size must be greater than zero"
            )
        if max_rows_per_file is not None and max_rows_per_file < 1:
            raise PreprocessingValidationError(
                "max_rows_per_file must be greater than zero"
            )
        if sample_10k and candidate_row_cap < 1:
            raise PreprocessingValidationError(
                "candidate_row_cap must be greater than zero"
            )
        if sample_10k and solver_node_limit < 1:
            raise PreprocessingValidationError(
                "solver_node_limit must be greater than zero"
            )
        if min_free_space_bytes < 0:
            raise PreprocessingValidationError(
                "minimum free-space threshold must not be negative"
            )
        volume_path = _existing_volume_path(output_directory)
        available_bytes = shutil.disk_usage(volume_path).free
        preflight_free_bytes = available_bytes
        logger.info(
            "Disk preflight output=%s volume=%s available_bytes=%d "
            "required_threshold_bytes=%d",
            output_directory,
            volume_path,
            available_bytes,
            min_free_space_bytes,
        )
        if available_bytes < min_free_space_bytes:
            raise PreprocessingValidationError(
                "Insufficient free disk space for configured preflight "
                f"threshold: {available_bytes} bytes available, "
                f"{min_free_space_bytes} bytes required. This threshold is "
                "only a start gate, not a guarantee of sufficient space."
            )

        if sample_10k:
            start_stage("candidate-sampling")
            bounded_sample = sample_bounded_candidates(
                data_directory,
                sample_size=10_000,
                row_cap_per_file=candidate_row_cap,
                seed=seed,
                progress_callback=sampling_progress,
            )
            selected_row_numbers = bounded_sample.selected_row_numbers
            sampling_source_files = bounded_sample.source_files
            sampling_source_signatures = bounded_sample.source_signatures
            sampling_manifest = bounded_sample.manifest

        output_directory.mkdir(parents=True, exist_ok=True)
        staging_directory = Path(
            tempfile.mkdtemp(prefix=".staging-", dir=output_directory)
        )
        database_path = staging_directory / "preprocessing.sqlite3"
        start_stage("database-setup")
        connection = _create_preprocessing_database(database_path)
        record_measurement("database-setup")

        start_stage("ingestion")
        ingested_selected_labels: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        if sample_10k and sampling_source_signatures is not None:
            for source_file, signature in sampling_source_signatures.items():
                source_path = data_directory.joinpath(*source_file.split("/"))
                source_stat = source_path.stat()
                if signature != (source_stat.st_size, source_stat.st_mtime_ns):
                    raise PreprocessingValidationError(
                        f"{source_file}: file changed before selected-row ingestion"
                    )
        ingestion_files = (
            sampling_source_files
            if sample_10k
            else selected_files
        )
        for chunk in iter_dataset_chunks(
            data_directory,
            chunk_size,
            report,
            selected_files=ingestion_files,
            max_rows_per_file=(
                candidate_row_cap if sample_10k else max_rows_per_file
            ),
            progress_callback=ingestion_progress,
            selected_row_numbers=selected_row_numbers,
            bounded_verification=sample_10k,
        ):
            current_file = chunk.source_file
            _store_chunk(connection, chunk, report.feature_columns)
            if sample_10k:
                for label in chunk.label_original:
                    ingested_selected_labels[chunk.source_file][str(label)] += 1
            logger.info(
                "Preprocessing stage=ingestion source_file=%s rows_processed=%d "
                "chunk_rows=%d elapsed_seconds=%.3f",
                chunk.source_file,
                report.row_count,
                len(chunk.features),
                time.monotonic() - run_started,
            )
        if sample_10k:
            if report.row_count != 10_000:
                raise PreprocessingValidationError(
                    "Selected-row ingestion did not process exactly 10000 rows"
                )
            actual_selected_labels = {
                source_file: dict(sorted(counts.items()))
                for source_file, counts in sorted(ingested_selected_labels.items())
            }
            expected_selected_labels = (
                sampling_manifest.get("selected_label_counts")
                if sampling_manifest is not None
                else None
            )
            if actual_selected_labels != expected_selected_labels:
                raise PreprocessingValidationError(
                    "Selected original labels changed between sampling and ingestion"
                )
            if sampling_source_signatures is not None:
                for source_file, signature in sampling_source_signatures.items():
                    source_path = data_directory.joinpath(*source_file.split("/"))
                    source_stat = source_path.stat()
                    if signature != (
                        source_stat.st_size,
                        source_stat.st_mtime_ns,
                    ):
                        raise PreprocessingValidationError(
                            f"{source_file}: file changed during selected-row ingestion"
                        )
        current_file = None
        record_measurement("ingestion-complete")

        start_stage("quality-validation")
        validate_data_quality(report)
        if report.row_count == 0:
            raise PreprocessingValidationError(
                "No data rows were found in the CSV files"
            )
        record_measurement("quality-validation-complete")

        start_stage("group-splitting")
        split_solver: dict[str, object] | None = None
        if sample_10k:
            if report.row_count != 10_000:
                raise PreprocessingValidationError(
                    f"10k sample ingestion selected {report.row_count} rows; "
                    "exactly 10000 are required"
                )
            row_counts, class_counts, split_solver = _split_sqlite_groups_exact(
                connection,
                seed,
                {"train": 7_000, "validation": 1_500, "test": 1_500},
                solver_node_limit,
            )
        else:
            row_counts, class_counts = _split_sqlite_groups(connection, seed)
        record_measurement("group-splitting-complete")
        feature_columns = report.feature_columns

        start_stage("training-median-fit")
        medians = _training_medians(connection, feature_columns)
        record_measurement("training-median-fit-complete")

        start_stage("training-scaler-fit")
        scaler = _fit_streaming_scaler(
            connection,
            feature_columns,
            medians,
            chunk_size,
        )
        record_measurement("training-scaler-fit-complete")

        start_stage("pipeline-setup")
        imputer = PrecomputedMedianImputer(medians, feature_columns)
        imputer.fit(pd.DataFrame(columns=feature_columns))
        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("scaler", scaler),
            ]
        )

        written_counts: dict[str, int] = {}
        written_class_counts: dict[str, dict[int, int]] = {}
        for partition in PARTITION_FRACTIONS:
            start_stage(f"transform-{partition}")
            written_counts[partition], written_class_counts[partition] = (
                _write_transformed_partition(
                    connection,
                    pipeline,
                    feature_columns,
                    partition,
                    staging_directory / f"{partition}.csv",
                    chunk_size,
                )
            )
            record_measurement(f"transform-{partition}-complete")
        if written_counts != row_counts:
            raise PreprocessingValidationError(
                "Written partition row counts do not match the validated split"
            )
        if written_class_counts != class_counts:
            raise PreprocessingValidationError(
                "Written partition class counts do not match the validated split"
            )
        if sample_10k and row_counts != {
            "train": 7_000,
            "validation": 1_500,
            "test": 1_500,
        }:
            raise PreprocessingValidationError(
                "10k artifact validation failed exact partition row counts"
            )
        group_count = int(connection.execute(
            "SELECT COUNT(*) FROM feature_groups"
        ).fetchone()[0])
        conflicting_count = int(connection.execute(
            """
            SELECT COUNT(*) FROM feature_groups
            WHERE benign_count > 0 AND attack_count > 0
            """
        ).fetchone()[0])

        start_stage("artifact-metadata")
        with (staging_directory / "feature_names.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(list(feature_columns), output, indent=2, ensure_ascii=False)
            output.write("\n")
        with (staging_directory / "target_mapping.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(
                {
                    "BENIGN (trimmed, case-insensitive)": 0,
                    "all other nonblank original labels": 1,
                },
                output,
                indent=2,
            )
            output.write("\n")
        with (staging_directory / "split_manifest.json").open(
            "w", encoding="utf-8"
        ) as output:
            manifest_fields = {
                "seed": seed,
                "target_fractions": PARTITION_FRACTIONS,
                "actual_rows": row_counts,
                "actual_fractions": {
                    partition: row_counts[partition] / report.row_count
                    for partition in PARTITION_FRACTIONS
                },
                "binary_class_counts": class_counts,
                "group_count": group_count,
                "conflicting_group_count": conflicting_count,
            }
            if sample_10k:
                manifest_fields["exact_partition_targets"] = {
                    "train": 7_000,
                    "validation": 1_500,
                    "test": 1_500,
                }
                manifest_fields["group_solver"] = split_solver
            output.write("{\n")
            for key, value in manifest_fields.items():
                output.write(f"{json.dumps(key)}: ")
                json.dump(value, output, indent=2, ensure_ascii=False)
                output.write(",\n")
            output.write('"conflicting_binary_label_groups": {')
            conflict_cursor = connection.execute(
                """
                SELECT group_id
                FROM feature_groups
                WHERE benign_count > 0 AND attack_count > 0
                ORDER BY group_id
                """
            )
            first_conflict = True
            for (group_id,) in conflict_cursor:
                labels = [
                    label
                    for (label,) in connection.execute(
                        """
                        SELECT original_label FROM group_labels
                        WHERE group_id = ? ORDER BY original_label
                        """,
                        (group_id,),
                    )
                ]
                if not first_conflict:
                    output.write(",")
                output.write(
                    f"\n{json.dumps(str(group_id))}: "
                    f"{json.dumps(labels, ensure_ascii=False)}"
                )
                first_conflict = False
            if not first_conflict:
                output.write("\n")
            output.write("},\n")
            output.write('"quality": ')
            json.dump(_quality_manifest(report), output, indent=2, ensure_ascii=False)
            output.write("\n}\n")
        with (staging_directory / "preprocessing_pipeline.pkl").open("wb") as output:
            pickle.dump(pipeline, output)
        if sampling_manifest is not None:
            with (staging_directory / "sampling_manifest.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(sampling_manifest, output, indent=2, ensure_ascii=False)
                output.write("\n")

        record_measurement("artifacts-ready")
        start_stage("publish")
        connection.close()
        connection = None
        database_path.unlink()
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(str(database_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        record_measurement("publish-ready")
        if pilot_mode:
            metrics = {
                "mode": "subset-pilot",
                "files": list(report.files),
                "max_rows_per_file": max_rows_per_file,
                "chunk_size": chunk_size,
                "minimum_free_space_threshold_bytes": min_free_space_bytes,
                "preflight_free_bytes": preflight_free_bytes,
                "rows_processed": report.row_count,
                "measurements": measurements,
                "measurement_limitations": [
                    "Memory counters are process-lifetime peaks read at stage checkpoints.",
                    "Staging-byte samples are taken at stage boundaries and may miss "
                    "short-lived SQLite temporary or journal peaks.",
                    "The final staging-byte sample is taken before writing this "
                    "metrics file.",
                    "The configured free-space threshold is a start gate, not a "
                    "sufficiency guarantee.",
                ],
            }
            with (staging_directory / "pilot_metrics.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(metrics, output, indent=2, ensure_ascii=False)
                output.write("\n")
        os.replace(staging_directory, destination)
        staging_directory = None
        logger.info(
            "Preprocessing completed destination=%s rows_processed=%d "
            "elapsed_seconds=%.3f",
            destination,
            report.row_count,
            time.monotonic() - run_started,
        )
    except BaseException:
        logger.exception(
            "Preprocessing failed stage=%s source_file=%s rows_processed=%d "
            "elapsed_seconds=%.3f",
            current_stage,
            current_file,
            report.row_count,
            time.monotonic() - run_started,
        )
        if connection is not None:
            connection.close()
        if staging_directory is not None and staging_directory.exists():
            shutil.rmtree(staging_directory)
        if not output_existed:
            try:
                output_directory.rmdir()
            except OSError:
                pass
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Preprocess CIC-IDS2017 CSV files without training a model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="source CSV directory (default: data/raw/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory (default depends on the selected mode)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="rows read per ingestion chunk (default: 100000)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="relative CSV paths beneath --data-dir to process",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        help="optional deterministic source-order row cap for each selected CSV",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="write to a separate pilot directory and require a strict CSV subset",
    )
    parser.add_argument(
        "--sample-10k",
        action="store_true",
        help=(
            "sample 10000 rows from bounded prefixes of every discovered CSV "
            "and require exact 7000/1500/1500 group-aware partitions"
        ),
    )
    parser.add_argument(
        "--max-candidate-rows-per-file",
        type=int,
        default=25_000,
        help="maximum data rows inspected from each CSV in --sample-10k mode",
    )
    parser.add_argument(
        "--solver-node-limit",
        type=int,
        default=250_000,
        help="maximum exact group-assignment search nodes (default: 250000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="deterministic sampling and split seed (default: 42)",
    )
    parser.add_argument(
        "--min-free-space-gib",
        type=float,
        required=True,
        help="required free-space start threshold on the output volume (GiB)",
    )
    arguments = parser.parse_args(argv)
    if arguments.chunk_size < 1:
        parser.error("--chunk-size must be greater than zero")
    if arguments.max_rows_per_file is not None and arguments.max_rows_per_file < 1:
        parser.error("--max-rows-per-file must be greater than zero")
    if arguments.max_candidate_rows_per_file < 1:
        parser.error("--max-candidate-rows-per-file must be greater than zero")
    if arguments.solver_node_limit < 1:
        parser.error("--solver-node-limit must be greater than zero")
    if (
        not math.isfinite(arguments.min_free_space_gib)
        or arguments.min_free_space_gib < 0
    ):
        parser.error("--min-free-space-gib must be a finite nonnegative number")
    if arguments.pilot and not arguments.files:
        parser.error("--pilot requires an explicit --files subset")
    if arguments.sample_10k and arguments.pilot:
        parser.error("--sample-10k cannot be combined with --pilot")
    if arguments.sample_10k and arguments.files:
        parser.error("--sample-10k cannot be combined with --files")
    if arguments.sample_10k and arguments.max_rows_per_file is not None:
        parser.error(
            "--sample-10k cannot be combined with --max-rows-per-file; "
            "use --max-candidate-rows-per-file"
        )
    available_files = [
        path for path in arguments.data_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".csv"
    ] if arguments.data_dir.is_dir() else []
    if arguments.pilot and len(set(arguments.files)) >= len(available_files):
        parser.error("--pilot must select fewer CSV files than are present")

    normal_output = project_root / "outputs" / "preprocessing"
    pilot_output = project_root / "outputs" / "preprocessing-pilot"
    sample_output = project_root / "outputs" / "preprocessing-10k"
    if arguments.output_dir is None:
        output_directory = (
            pilot_output
            if arguments.pilot
            else sample_output
            if arguments.sample_10k
            else normal_output
        )
    else:
        output_directory = arguments.output_dir
    if arguments.pilot and _is_within(output_directory, normal_output):
        parser.error("--pilot output must be separate from outputs/preprocessing/")
    if arguments.sample_10k and (
        _paths_overlap(output_directory, normal_output)
        or _paths_overlap(output_directory, pilot_output)
    ):
        parser.error(
            "--sample-10k output must be isolated from normal and pilot outputs"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    destination = run_preprocessing(
        arguments.data_dir,
        output_directory,
        chunk_size=arguments.chunk_size,
        seed=arguments.seed,
        selected_files=arguments.files,
        max_rows_per_file=arguments.max_rows_per_file,
        min_free_space_bytes=math.ceil(arguments.min_free_space_gib * GIB),
        pilot_mode=arguments.pilot,
        sample_10k=arguments.sample_10k,
        candidate_row_cap=arguments.max_candidate_rows_per_file,
        solver_node_limit=arguments.solver_node_limit,
    )
    print(f"Preprocessing artifacts saved to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())