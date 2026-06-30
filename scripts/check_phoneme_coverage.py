#!/usr/bin/env python3
"""Check Allosaurus phone coverage against the IPA->phoneme mapping.

Purpose
-------
The acoustic pipeline maps the raw phones emitted by Allosaurus (for German,
``deu``) to a controlled set of analysis phonemes via ``config/ipa_phoneme_map.csv``
(see ``docs/acoustic-analysis-design.md`` sec. 3.2). Any phone that is *not* listed
in that map would silently fall through and be treated as ``NA``. This script flags
such **unmapped** phones so the mapping can be corrected before analysis.

It can read phones from either:

1. one or more ``*_phonemes.csv`` segmentation outputs (column ``phone``), and/or
2. a plain text/Allosaurus ``--timestamp`` output (3 columns: ``time duration phone``),
   passed via ``--phones-file``.

Usage
-----
    # Check the segmentation CSVs produced by the pipeline
    python scripts/check_phoneme_coverage.py path/to/*_phonemes.csv

    # Check a raw Allosaurus timestamped dump
    python scripts/check_phoneme_coverage.py --phones-file allosaurus_out.txt

    # Use a non-default map location
    python scripts/check_phoneme_coverage.py --map config/ipa_phoneme_map.csv inputs/*.csv

Exit codes
----------
    0 : every observed phone is present in the map (full coverage)
    1 : one or more observed phones are missing from the map
    2 : usage / IO error (e.g. map not found, no inputs)

This script has no third-party dependencies (standard library only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_MAP = Path("config/ipa_phoneme_map.csv")


def load_mapped_phones(map_path: Path) -> set[str]:
    """Return the set of IPA phones declared in the mapping CSV.

    The mapping file may contain comment lines starting with ``#`` (which are
    skipped) followed by a header row that includes an ``ipa`` column.
    """
    if not map_path.is_file():
        raise FileNotFoundError(f"mapping file not found: {map_path}")

    mapped: set[str] = set()
    with map_path.open(encoding="utf-8", newline="") as fh:
        # Skip leading comment lines so csv.DictReader sees the real header.
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        if reader.fieldnames is None or "ipa" not in reader.fieldnames:
            raise ValueError(
                f"{map_path} must have a header row containing an 'ipa' column "
                f"(found: {reader.fieldnames})"
            )
        for row in reader:
            ipa = (row.get("ipa") or "").strip()
            if ipa:
                mapped.add(ipa)
    return mapped


def _phones_from_csv(path: Path) -> Iterable[str]:
    """Yield phones from a ``*_phonemes.csv`` file (column ``phone``)."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        if reader.fieldnames is None or "phone" not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain a 'phone' column (found: {reader.fieldnames})"
            )
        for row in reader:
            phone = (row.get("phone") or "").strip()
            if phone:
                yield phone


def _phones_from_timestamp_dump(path: Path) -> Iterable[str]:
    """Yield phones from a raw Allosaurus ``--timestamp`` dump.

    Each non-empty, non-comment line is ``time duration phone``. Lines that start
    with ``#`` (Allosaurus per-file headers when recognizing a directory) are
    skipped.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect "time duration phone"; the phone is the last field.
            if len(parts) >= 3:
                yield parts[-1]


def collect_observed_phones(
    csv_inputs: list[Path], phones_file: Path | None
) -> dict[str, int]:
    """Return a {phone: count} mapping across all provided inputs."""
    counts: dict[str, int] = {}

    def _tally(phones: Iterable[str]) -> None:
        for phone in phones:
            counts[phone] = counts.get(phone, 0) + 1

    for path in csv_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")
        _tally(_phones_from_csv(path))

    if phones_file is not None:
        if not phones_file.is_file():
            raise FileNotFoundError(f"--phones-file not found: {phones_file}")
        _tally(_phones_from_timestamp_dump(phones_file))

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flag Allosaurus phones that are missing from the IPA->phoneme map "
            "(config/ipa_phoneme_map.csv)."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="One or more *_phonemes.csv segmentation files (column 'phone').",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP,
        help=f"Path to the IPA->phoneme mapping CSV (default: {DEFAULT_MAP}).",
    )
    parser.add_argument(
        "--phones-file",
        type=Path,
        default=None,
        help="A raw Allosaurus --timestamp dump (lines: 'time duration phone').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.inputs and args.phones_file is None:
        sys.stderr.write(
            "error: provide at least one *_phonemes.csv input or --phones-file\n"
        )
        return 2

    try:
        mapped = load_mapped_phones(args.map)
        observed = collect_observed_phones(list(args.inputs), args.phones_file)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if not observed:
        sys.stderr.write("error: no phones found in the provided input(s)\n")
        return 2

    missing = {p: c for p, c in observed.items() if p not in mapped}

    n_observed = len(observed)
    n_missing = len(missing)
    n_covered = n_observed - n_missing

    print(f"Map:            {args.map}  ({len(mapped)} phones declared)")
    print(f"Observed:       {n_observed} distinct phone(s)")
    print(f"Covered by map: {n_covered}")
    print(f"Missing:        {n_missing}")

    if missing:
        print("\nUnmapped phones (would silently become NA) — add these to the map:")
        # Sort by frequency (desc), then symbol, so the most common gaps come first.
        for phone, count in sorted(missing.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {phone!r}\tcount={count}")
        return 1

    print("\nOK: every observed phone is present in the map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
