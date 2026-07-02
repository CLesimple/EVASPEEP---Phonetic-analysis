#!/usr/bin/env python3
"""Batch runner for the EVASPEEP pipeline.

Runs ``evaspeep_pipeline.analyze_file`` over every ``.wav`` in an input
directory, writing per-file CSVs to a SEPARATE output directory (audio is never
mixed with derived data), then pools the per-file vowel / fricative / VSA tables
into three combined CSVs for cross-file analysis.

Example (matches the VP01 layout):
    python scripts/analyze_dir.py \
        --audio-dir audio/VP01 \
        --gender-table participant_gender.csv \
        --out output/VP01 \
        --map config/ipa_phoneme_map.csv

Pooled outputs are written to the PARENT of --out (e.g. ``output/``) so they sit
above the per-participant folders:
    output/VP01_vowels_pooled.csv
    output/VP01_fricatives_pooled.csv
    output/VP01_vsa_pooled.csv
    output/VP01_corner_token_counts.csv   (QA: flags corners with < MIN_CORNER_TOKENS)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the pipeline module that lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaspeep_pipeline as ev  # noqa: E402

# Corners with fewer than this many tokens are flagged as low-confidence (§4.4.1
# requires >= 3 to compute tVSA at all; 5 is a stricter "trust it" threshold).
MIN_CORNER_TOKENS = 5
CORNERS = ("i", "a", "u")


def find_wavs(audio_dir: Path) -> list[Path]:
    """Return sorted ``.wav`` files in ``audio_dir`` (non-recursive)."""
    wavs = sorted(p for p in audio_dir.glob("*.wav") if p.is_file())
    return wavs


def analyze_dir(
    audio_dir: str | Path,
    gender_table: str | Path,
    out_dir: str | Path,
    phoneme_map_path: str | Path = ev.DEFAULT_MAP_PATH,
) -> dict[str, Path]:
    """Process every wav in ``audio_dir`` and pool the results.

    Per-file CSVs go in ``out_dir``; pooled CSVs go in ``out_dir.parent``.
    Returns a dict of the pooled output paths. Individual file failures are
    reported and skipped so one bad file doesn't abort the whole batch.
    """
    import pandas as pd  # local import so pure-import of the module stays light

    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir = out_dir.parent
    pooled_dir.mkdir(parents=True, exist_ok=True)
    tag = out_dir.name  # e.g. "VP01" -> used to name pooled files

    wavs = find_wavs(audio_dir)
    if not wavs:
        raise FileNotFoundError(f"no .wav files found in {audio_dir}")

    print(f"Found {len(wavs)} wav file(s) in {audio_dir}")

    vowel_frames, fric_frames, vsa_frames = [], [], []
    n_ok, n_fail = 0, 0

    for wav in wavs:
        try:
            paths = ev.analyze_file(
                wav_path=wav,
                gender_table=gender_table,
                out_dir=out_dir,
                phoneme_map_path=phoneme_map_path,
            )
        except Exception as exc:  # skip & report, keep batch going
            n_fail += 1
            print(f"  [SKIP] {wav.name}: {type(exc).__name__}: {exc}")
            continue

        n_ok += 1
        print(f"  [ OK ] {wav.name}")

        # Collect per-file CSVs, tagging each row with the source file.
        for key, frames in (("vowels", vowel_frames),
                            ("fricatives", fric_frames),
                            ("vsa", vsa_frames)):
            fp = paths[key]
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if df.empty:
                continue
            df.insert(0, "source_file", wav.name)
            frames.append(df)

    def _concat_write(frames, name):
        path = pooled_dir / f"{tag}_{name}_pooled.csv"
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        else:
            path.write_text("", encoding="utf-8")
        return path

    pooled = {
        "vowels": _concat_write(vowel_frames, "vowels"),
        "fricatives": _concat_write(fric_frames, "fricatives"),
        "vsa": _concat_write(vsa_frames, "vsa"),
    }

    # --- QA: corner token counts across the whole participant ---------------
    counts_path = pooled_dir / f"{tag}_corner_token_counts.csv"
    if vowel_frames:
        allv = pd.concat(vowel_frames, ignore_index=True)
        rows = []
        for c in CORNERS:
            n = int((allv["phoneme"] == c).sum())
            rows.append({
                "corner": c,
                "n_tokens": n,
                "meets_min": n >= MIN_CORNER_TOKENS,
                "min_required": MIN_CORNER_TOKENS,
            })
        counts = pd.DataFrame(rows)
        counts.to_csv(counts_path, index=False)
        pooled["corner_token_counts"] = counts_path

        print("\nCorner token counts (pooled):")
        for r in rows:
            flag = "" if r["meets_min"] else f"  <-- LOW (< {MIN_CORNER_TOKENS})"
            print(f"  {r['corner']}: {r['n_tokens']}{flag}")
    else:
        counts_path.write_text("", encoding="utf-8")
        pooled["corner_token_counts"] = counts_path

    print(f"\nProcessed {n_ok} file(s) OK, {n_fail} skipped.")
    print("Pooled outputs:")
    for k, v in pooled.items():
        print(f"  {k:20s} {v}")
    return pooled


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-run EVASPEEP over a directory of wavs and pool results."
    )
    p.add_argument("--audio-dir", required=True, type=Path,
                   help="Directory of input .wav files (e.g. audio/VP01).")
    p.add_argument("--gender-table", required=True, type=Path,
                   help="CSV with VP_ID,Gender columns.")
    p.add_argument("--out", required=True, type=Path,
                   help="Per-file output dir (e.g. output/VP01). "
                        "Pooled CSVs go to its parent.")
    p.add_argument("--map", type=Path, default=ev.DEFAULT_MAP_PATH,
                   help=f"IPA->phoneme map CSV (default: {ev.DEFAULT_MAP_PATH}).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        analyze_dir(args.audio_dir, args.gender_table, args.out, args.map)
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())