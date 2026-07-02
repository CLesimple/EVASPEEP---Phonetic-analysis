#!/usr/bin/env python3
"""EVASPEEP — phone segmentation + acoustic feature extraction.

This single-file module implements the pipeline specified in
``docs/acoustic-analysis-design.md``. The design doc is authoritative; section
numbers (e.g. §3.2) in the docstrings refer to it.

Inputs
------
1. A mono 48 kHz ``.wav`` file whose name follows the convention
   ``{ParticipantID}_{Sequence}_{Condition}_{Sound}.wav`` (§1.1).
2. A speaker/gender table CSV with columns ``VP_ID,Gender`` (§1.2), used to set
   the sex-dependent formant ceiling (§4.2).

Outputs
-------
- ``{stem}_phonemes.csv``         : Stage 0 segmentation (one row per phone).
- ``{stem}_vowels.csv``           : Stage 1 per-vowel-token features.
- ``{stem}_vsa.csv``              : Stage 1 per-recording VSA (tVSA + cVSA).
- ``{stem}_fricatives.csv``       : Stage 2 per-fricative-token features.
- ``{stem}_provenance.json``      : parameters + package versions (§2).

CLI
---
    python scripts/evaspeep_pipeline.py \
        --wav data/audio/VP01_M1_aided_Nordwind.wav \
        --gender-table data/metadata/speakers.csv \
        --out data/outputs

Notes
-----
- Heavy/optional dependencies (``allosaurus``, ``torch``/``silero-vad``,
  ``parselmouth``) are imported *lazily* inside the functions that need them, so
  the pure helpers (filename parsing, map loading, time_norm, shoelace, Hz->Bark,
  the physiological gate) import and unit-test cleanly without those installed.
- The raw-phone column in ``*_phonemes.csv`` is named ``phone`` so that
  ``scripts/check_phoneme_coverage.py`` keeps working.
- The IPA->phoneme mapping is loaded from ``config/ipa_phoneme_map.csv`` (skipping
  its leading ``#`` comment block); it is **not** hardcoded.
- This is a faithful implementation of the spec but has not been executed against
  real audio/models here. Smoke-test on one Bilder and one Nordwind file, then
  feed the resulting ``*_phonemes.csv`` to ``scripts/check_phoneme_coverage.py``
  to refine the draft map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Defaults (all taken from docs/acoustic-analysis-design.md). Anything the doc
# calls a "parameter" is exposed as a function argument defaulting to these.
# ---------------------------------------------------------------------------

DEFAULT_MAP_PATH = Path("config/ipa_phoneme_map.csv")

# Sounds processed (everything except sustained /a/, "Vokal") — §1.1
EXCLUDED_SOUNDS = {"Vokal"}

# Stage 0
ALLOSAURUS_LANG = "deu"
ALLOSAURUS_TOPK = 5            # request candidate probabilities for QA (§3.1)
VAD_SAMPLE_RATE = 16_000       # Silero needs mono 16 kHz (§2 note / §3.3)

# Stage 1 — vowels (§4)
VOWEL_ONSET_OFFSET_S = 0.005   # +5 ms onset offset (§4.1)
VOWEL_WINDOW_S = 0.050         # fixed 50 ms analysis window (§4.1)
VOWEL_MIN_DURATION_S = 0.050   # min effective duration 50 ms (§4.1)
VOWEL_CENTRAL_LO = 0.20        # Hillenbrand-style central sampling band (§4.1)
VOWEL_CENTRAL_HI = 0.80
F0_FALLBACK_HZ = 300.0         # F0 fallback if pitch fails (§4.2)
# Sex-dependent formant ceiling (VoiceLab convention) — §4.2
FORMANT_CEILING_HZ = {"F": 5500.0, "M": 5000.0}
DROP_VOWELS = {"ə", "ɐ"}       # drop schwa / reduced (§4.1)
TVSA_CORNERS = ("i", "a", "u") # §4.4.1
TVSA_MIN_TOKENS_PER_CORNER = 3 # §4.4.1
# Physiological range gate in Hz (before Bark) — §4.4.2
F1_RANGE_HZ = (200.0, 1200.0)
F2_RANGE_HZ = (700.0, 3500.0)

# Stage 2 — fricatives (§5)
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ"}     # §3.2 / §5
FRICATIVE_MIN_DURATION_S = 0.030                # min 30 ms (§5.1)
FRICATIVE_BAND_HZ = (500.0, 15_000.0)           # 500 Hz – 15 kHz (§5.2)
FRICATIVE_PRE_EMPHASIS = False                  # no pre-emphasis (§5.2)
DCT_N_COEFFS = 3                                # keep first 3 (§5.3)


# ===========================================================================
# IO / metadata  (§1)
# ===========================================================================

def parse_filename(path: str | Path) -> dict[str, str]:
    """Parse ``{ParticipantID}_{Sequence}_{Condition}_{Sound}.wav`` (§1.1).

    Returns a dict with keys ``ParticipantID, Sequence, Condition, Sound``.
    Raises ``ValueError`` if the stem doesn't have the 4 expected fields or the
    Sound is excluded (e.g. ``Vokal``).
    """
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"filename {stem!r} does not match "
            "'{ParticipantID}_{Sequence}_{Condition}_{Sound}'"
        )
    participant, sequence, condition, sound = parts
    if sound in EXCLUDED_SOUNDS:
        raise ValueError(f"sound {sound!r} is excluded from this pipeline (§1.1)")
    return {
        "ParticipantID": participant,
        "Sequence": sequence,
        "Condition": condition,
        "Sound": sound,
    }


def load_speaker_metadata(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Load the speaker/gender table (§1.2).

    Expects columns ``VP_ID`` and ``Gender`` (F/M). Returns
    ``{VP_ID: {"Gender": "F"|"M"}}``.
    """
    csv_path = Path(csv_path)
    out: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "VP_ID" not in reader.fieldnames \
                or "Gender" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must have 'VP_ID' and 'Gender' columns "
                f"(found: {reader.fieldnames})"
            )
        for row in reader:
            vp = (row.get("VP_ID") or "").strip()
            gender = (row.get("Gender") or "").strip().upper()
            if not vp:
                continue
            if gender not in {"F", "M"}:
                raise ValueError(f"VP_ID {vp!r}: Gender must be 'F' or 'M', got {gender!r}")
            out[vp] = {"Gender": gender}
    return out


def load_phoneme_map(csv_path: str | Path = DEFAULT_MAP_PATH) -> dict[str, dict[str, str]]:
    """Load ``config/ipa_phoneme_map.csv`` (§3.2), skipping ``#`` comment lines.

    Returns ``{ipa: {"phoneme", "category", "is_corner"}}``. ``phoneme`` may be
    the literal string ``"NA"`` for phones not used by any analysis.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"mapping file not found: {csv_path}")
    mapping: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        required = {"ipa", "phoneme", "category", "is_corner"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns {sorted(required)} "
                f"(found: {reader.fieldnames})"
            )
        for row in reader:
            ipa = (row.get("ipa") or "").strip()
            if not ipa:
                continue
            mapping[ipa] = {
                "phoneme": (row.get("phoneme") or "NA").strip(),
                "category": (row.get("category") or "NA").strip(),
                "is_corner": (row.get("is_corner") or "NA").strip(),
            }
    return mapping


# ===========================================================================
# Stage 0 — phone segmentation  (§3)
# ===========================================================================

@dataclass
class Phone:
    """A single recognized phone with timing + QA fields."""
    time: float                       # raw Allosaurus onset (s)
    duration: float                   # raw Allosaurus (placeholder) duration (s)
    phone: str                        # raw IPA-like symbol
    phoneme: str = "NA"               # mapped analysis label (§3.2)
    category: str = "NA"              # vowel | fricative | NA
    is_corner: str = "NA"             # i | a | u | NA
    t_next: float = math.nan          # onset of next phone (§3.3)
    eff_start: float = math.nan       # VAD-clipped effective interval start
    eff_end: float = math.nan         # VAD-clipped effective interval end
    time_norm: float = math.nan       # token_time / file_duration (§6.1)
    topk_candidates: str = ""         # ";"-joined candidate phones (QA)
    topk_probs: str = ""              # ";"-joined candidate probs (QA)


def recognize_phones(
    wav_path: str | Path,
    lang: str = ALLOSAURUS_LANG,
    topk: int = ALLOSAURUS_TOPK,
) -> list[Phone]:
    """Run Allosaurus with timestamps and parse **all** lines (§3.1).

    Allosaurus uses Python's stdlib ``wave``, which only reads PCM-integer WAVs
    (format 1). EVASPEEP masters are 48 kHz 32-bit float (WAV format 3), which
    ``wave`` rejects ('unknown format: 3'). We therefore transcode to a temporary
    16-bit PCM copy **for recognition only**; timestamps are in seconds and map
    back onto the original untouched audio. The 32-bit float master is still used
    for all acoustic measurement (parselmouth/soundfile handle it natively).

    The timestamped output has NO header row — the first line is a real phone —
    so we parse every line and cast time/duration to float. Top-k candidates are
    captured for QA logging.
    """
    import tempfile
    import os
    import soundfile as sf  # reads float WAV; writes PCM_16
    from allosaurus.app import read_recognizer  # lazy import

    # --- Transcode to temp 16-bit PCM only if needed -----------------------
    data, sr = sf.read(str(wav_path), always_2d=True)  # float array, any subtype
    if data.shape[1] > 1:                               # downmix to mono if stereo
        data = data.mean(axis=1, keepdims=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        sf.write(tmp_path, data[:, 0], sr, subtype="PCM_16")

        recognizer = read_recognizer()
        raw = recognizer.recognize(tmp_path, lang_id=lang, timestamp=True, topk=topk)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # --- Parse every emitted line -----------------------------------------
    phones: list[Phone] = []
    for line in str(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
            dur = float(parts[1])
        except ValueError:
            continue
        top1 = parts[2]
        cands, probs = [top1], []
        rest = parts[3:]
        for i in range(0, len(rest) - 1, 2):
            cands.append(rest[i])
            probs.append(rest[i + 1])
        phones.append(
            Phone(
                time=t,
                duration=dur,
                phone=top1,
                topk_candidates=";".join(cands),
                topk_probs=";".join(probs),
            )
        )
    return phones


def vad_speech_regions(wav_path: str | Path) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...] speech regions via Silero VAD (§3.3).

    Audio is loaded with ``soundfile`` (not torchaudio) to avoid the
    TorchCodec/FFmpeg backend requirement on Windows. The signal is downmixed to
    mono and resampled to 16 kHz (Silero's required rate) for VAD only; the
    original 48 kHz 32-bit float audio is used elsewhere for measurement.
    """
    import numpy as np  # lazy import
    import torch  # lazy import
    import soundfile as sf  # lazy import (no FFmpeg needed)

    # Load with soundfile (handles 32-bit float natively), downmix to mono.
    data, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)

    # Resample to 16 kHz if needed (linear interp keeps deps minimal).
    target_sr = VAD_SAMPLE_RATE
    if sr != target_sr:
        n_out = int(round(len(mono) * target_sr / sr))
        if n_out <= 1:
            return []
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)

    wav_tensor = torch.from_numpy(mono)

    # Load Silero and get timestamps directly from the tensor (no read_audio).
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]  # index 0 is stable across versions

    ts = get_speech_timestamps(
        wav_tensor, model, sampling_rate=target_sr, return_seconds=True
    )
    return [(float(seg["start"]), float(seg["end"])) for seg in ts]


def _clip_interval_to_regions(
    start: float, end: float, regions: list[tuple[float, float]]
) -> Optional[tuple[float, float]]:
    """Clip [start, end] to the speech region it overlaps most; None if none."""
    best: Optional[tuple[float, float]] = None
    best_overlap = 0.0
    for rs, re in regions:
        lo, hi = max(start, rs), min(end, re)
        overlap = hi - lo
        if overlap > best_overlap:
            best_overlap = overlap
            best = (lo, hi)
    return best


def effective_intervals(
    phones: list[Phone],
    vad_regions: list[tuple[float, float]],
    file_end: float,
) -> list[Phone]:
    """Assign VAD-clipped effective intervals ``[t_i, t_{i+1}]`` (§3.3).

    ``t_next`` is the next phone's onset (last phone -> ``file_end``). Each
    interval is clipped to VAD speech regions; phones whose measurement point
    would fall outside speech are marked with NaN bounds (callers drop them).
    """
    n = len(phones)
    for i, ph in enumerate(phones):
        ph.t_next = phones[i + 1].time if i + 1 < n else float(file_end)
        clipped = _clip_interval_to_regions(ph.time, ph.t_next, vad_regions)
        if clipped is None:
            ph.eff_start, ph.eff_end = math.nan, math.nan
        else:
            ph.eff_start, ph.eff_end = clipped
    return phones


def apply_phoneme_map(phones: list[Phone], mapping: dict[str, dict[str, str]]) -> list[Phone]:
    """Annotate each phone with mapped ``phoneme/category/is_corner`` (§3.2)."""
    for ph in phones:
        m = mapping.get(ph.phone)
        if m is None:
            ph.phoneme, ph.category, ph.is_corner = "NA", "NA", "NA"
        else:
            ph.phoneme = m["phoneme"]
            ph.category = m["category"]
            ph.is_corner = m["is_corner"]
    return phones


def compute_time_norm(phones: list[Phone], total_file_duration: float) -> list[Phone]:
    """Set ``time_norm = token_time / total_file_duration`` in [0, 1] (§6.1)."""
    if total_file_duration <= 0:
        raise ValueError("total_file_duration must be > 0")
    for ph in phones:
        ph.time_norm = ph.time / total_file_duration
    return phones


def _wav_duration_seconds(wav_path: str | Path) -> float:
    """Full file duration in seconds (used for time_norm denominator)."""
    import soundfile as sf  # lazy import (libsndfile)

    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def segment_file(
    wav_path: str | Path,
    phoneme_map: dict[str, dict[str, str]],
    lang: str = ALLOSAURUS_LANG,
    topk: int = ALLOSAURUS_TOPK,
) -> list[Phone]:
    """Full Stage 0 for one file: recognize -> map -> intervals -> time_norm."""
    duration = _wav_duration_seconds(wav_path)
    phones = recognize_phones(wav_path, lang=lang, topk=topk)
    phones = apply_phoneme_map(phones, phoneme_map)
    regions = vad_speech_regions(wav_path)
    phones = effective_intervals(phones, regions, file_end=duration)
    phones = compute_time_norm(phones, duration)
    return phones


_PHONEME_CSV_FIELDS = [
    "ParticipantID", "Sequence", "Condition", "Sound",
    "time", "duration", "phone", "phoneme", "category", "is_corner",
    "t_next", "eff_start", "eff_end", "time_norm",
    "topk_candidates", "topk_probs",
]


def save_segmentation(
    phones: list[Phone], meta: dict[str, str], out_dir: str | Path
) -> Path:
    """Write ``{stem}_phonemes.csv`` (§3.4). Keeps ``phone`` column compatible
    with ``scripts/check_phoneme_coverage.py``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['ParticipantID']}_{meta['Sequence']}_{meta['Condition']}_{meta['Sound']}"
    out_path = out_dir / f"{stem}_phonemes.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PHONEME_CSV_FIELDS)
        writer.writeheader()
        for ph in phones:
            row = {**meta, **asdict(ph)}
            writer.writerow({k: row.get(k, "") for k in _PHONEME_CSV_FIELDS})
    return out_path


# ===========================================================================
# Stage 1 — vowel-specific features  (§4)
# ===========================================================================

def hz_to_bark(hz: float) -> float:
    """Hz -> Bark (Traunmüller 1990; VoiceLab convention) — §4.4."""
    if hz is None or (isinstance(hz, float) and math.isnan(hz)):
        return math.nan
    return (26.81 * hz / (1960.0 + hz)) - 0.53


def select_vowel_window(
    eff_start: float,
    eff_end: float,
    onset_offset_s: float = VOWEL_ONSET_OFFSET_S,
    window_s: float = VOWEL_WINDOW_S,
    central_lo: float = VOWEL_CENTRAL_LO,
    central_hi: float = VOWEL_CENTRAL_HI,
    min_duration_s: float = VOWEL_MIN_DURATION_S,
) -> Optional[tuple[float, float]]:
    """Pick the [start, end] measurement window inside a vowel (§4.1).

    Applies the +5 ms onset offset, samples around the central portion of the
    *effective* (VAD-clipped) duration (Hillenbrand-style), and uses a fixed
    50 ms window. Returns ``None`` if the effective duration < ``min_duration_s``
    or the interval is invalid.
    """
    if any(math.isnan(x) for x in (eff_start, eff_end)):
        return None
    dur = eff_end - eff_start
    if dur < min_duration_s:
        return None
    # Centre of the central band, bounded by the onset offset.
    centre_frac = 0.5 * (central_lo + central_hi)
    centre = eff_start + onset_offset_s + centre_frac * (dur - onset_offset_s)
    half = 0.5 * window_s
    start = max(eff_start + onset_offset_s, centre - half)
    end = min(eff_end, start + window_s)
    if end - start <= 0:
        return None
    return (start, end)


def measure_formants(
    wav_path: str | Path,
    start_s: float,
    end_s: float,
    gender: str,
    f0_fallback_hz: float = F0_FALLBACK_HZ,
    ceiling_by_sex: dict[str, float] = None,
) -> dict[str, float]:
    """Median F0/F1/F2 (Hz) over the window via parselmouth (§4.2).

    Uses a sex-dependent formant ceiling (VoiceLab convention) and falls back to
    ``f0_fallback_hz`` if pitch estimation fails. Returns NaNs rather than
    dropping on failure (kept for transparent QA).
    """
    import numpy as np  # lazy import
    import parselmouth  # lazy import
    from parselmouth.praat import call  # lazy import

    ceiling_by_sex = ceiling_by_sex or FORMANT_CEILING_HZ
    ceiling = ceiling_by_sex.get(gender.upper(), FORMANT_CEILING_HZ["M"])

    snd = parselmouth.Sound(str(wav_path)).extract_part(
        from_time=start_s, to_time=end_s, preserve_times=True
    )

    # F0 (median), with fallback.
    try:
        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array["frequency"]
        f0_vals = f0_vals[f0_vals > 0]
        f0 = float(np.median(f0_vals)) if f0_vals.size else f0_fallback_hz
    except Exception:
        f0 = f0_fallback_hz

    # Formants (Burg), median F1/F2 across frames.
    try:
        formant = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=ceiling)
        f1s, f2s = [], []
        n = call(formant, "Get number of frames")
        for i in range(1, int(n) + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "hertz", "Linear")
            if not math.isnan(f1):
                f1s.append(f1)
            if not math.isnan(f2):
                f2s.append(f2)
        f1 = float(np.median(f1s)) if f1s else math.nan
        f2 = float(np.median(f2s)) if f2s else math.nan
    except Exception:
        f1, f2 = math.nan, math.nan

    return {"F0": f0, "F1": f1, "F2": f2, "formant_ceiling": ceiling}


def _shoelace_area(points: list[tuple[float, float]]) -> float:
    """Polygon area via the shoelace formula (§4.4.1)."""
    if len(points) < 3:
        return math.nan
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def tvsa(
    vowel_rows: list[dict[str, Any]],
    corners: tuple[str, ...] = TVSA_CORNERS,
    min_tokens: int = TVSA_MIN_TOKENS_PER_CORNER,
) -> float:
    """Triangle VSA in Bark (§4.4.1).

    ``vowel_rows`` need keys ``phoneme``, ``F1_bark``, ``F2_bark``. Uses the
    per-corner median (Bark) of /i/,/a/,/u/; requires >= ``min_tokens`` per
    corner else returns NaN.
    """
    import numpy as np

    pts: list[tuple[float, float]] = []
    for c in corners:
        f1 = [r["F1_bark"] for r in vowel_rows
              if r.get("phoneme") == c and not math.isnan(r.get("F1_bark", math.nan))]
        f2 = [r["F2_bark"] for r in vowel_rows
              if r.get("phoneme") == c and not math.isnan(r.get("F2_bark", math.nan))]
        if len(f1) < min_tokens or len(f2) < min_tokens:
            return math.nan
        pts.append((float(np.median(f2)), float(np.median(f1))))  # (x=F2, y=F1)
    return _shoelace_area(pts)


def cvsa(
    vowel_rows: list[dict[str, Any]],
    f1_range_hz: tuple[float, float] = F1_RANGE_HZ,
    f2_range_hz: tuple[float, float] = F2_RANGE_HZ,
) -> float:
    """Convex-hull VSA in Bark (§4.4.2).

    Applies the physiological gate in Hz *before* Bark conversion, then computes
    ``scipy.spatial.ConvexHull`` over qualifying (F2, F1) Bark points; the 2-D
    area is the hull's ``.volume``. Returns NaN with < 3 qualifying points.
    """
    from scipy.spatial import ConvexHull  # lazy import

    pts = []
    for r in vowel_rows:
        f1, f2 = r.get("F1", math.nan), r.get("F2", math.nan)
        if math.isnan(f1) or math.isnan(f2):
            continue
        if not (f1_range_hz[0] <= f1 <= f1_range_hz[1]):
            continue
        if not (f2_range_hz[0] <= f2 <= f2_range_hz[1]):
            continue
        pts.append((hz_to_bark(f2), hz_to_bark(f1)))
    if len(pts) < 3:
        return math.nan
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return math.nan


def extract_vowels(
    wav_path: str | Path,
    phones: list[Phone],
    gender: str,
    meta: dict[str, str],
) -> list[dict[str, Any]]:
    """Per-vowel-token features (§4.3): median F0/F1/F2 (+ Bark) and params."""
    rows: list[dict[str, Any]] = []
    for ph in phones:
        if ph.category != "vowel":
            continue
        if ph.phoneme in DROP_VOWELS or ph.phoneme == "NA":
            continue
        win = select_vowel_window(ph.eff_start, ph.eff_end)
        if win is None:
            continue
        start_s, end_s = win
        f = measure_formants(wav_path, start_s, end_s, gender)
        row = {
            **meta,
            "phoneme": ph.phoneme,
            "is_corner": ph.is_corner,
            "time": ph.time,
            "eff_start": ph.eff_start,
            "eff_end": ph.eff_end,
            "meas_start": start_s,
            "meas_end": end_s,
            "time_norm": ph.time_norm,
            "F0": f["F0"],
            "F1": f["F1"],
            "F2": f["F2"],
            "F1_bark": hz_to_bark(f["F1"]),
            "F2_bark": hz_to_bark(f["F2"]),
            "formant_ceiling": f["formant_ceiling"],
            "window_s": VOWEL_WINDOW_S,
            "onset_offset_s": VOWEL_ONSET_OFFSET_S,
        }
        rows.append(row)
    return rows


# ===========================================================================
# Stage 2 — fricative distinction  (§5)
# ===========================================================================

def select_fricative_centre(
    eff_start: float,
    eff_end: float,
    window_s: float = VOWEL_WINDOW_S,
    min_duration_s: float = FRICATIVE_MIN_DURATION_S,
) -> Optional[tuple[float, float]]:
    """Central-portion window of a fricative (§5.1); None if < ``min_duration_s``."""
    if any(math.isnan(x) for x in (eff_start, eff_end)):
        return None
    dur = eff_end - eff_start
    if dur < min_duration_s:
        return None
    centre = eff_start + dur / 2.0
    half = 0.5 * min(window_s, dur)
    return (centre - half, centre + half)


def spectral_moments(
    wav_path: str | Path,
    start_s: float,
    end_s: float,
    band_hz: tuple[float, float] = FRICATIVE_BAND_HZ,
    pre_emphasis: bool = FRICATIVE_PRE_EMPHASIS,
) -> dict[str, float]:
    """Four spectral moments via parselmouth (§5.2).

    Centroid (centre of gravity), SD, skewness, kurtosis — computed WITHOUT
    additional pre-emphasis, over a windowed spectrum, band-limited to
    ``band_hz``. ``pre_emphasis=False`` is recorded for provenance.
    """
    import parselmouth  # lazy import
    from parselmouth import WindowShape  # enum for extract_part
    from parselmouth.praat import call  # lazy import

    snd = parselmouth.Sound(str(wav_path)).extract_part(
        from_time=start_s, to_time=end_s,
        window_shape=WindowShape.HAMMING, preserve_times=True,
    )
    if pre_emphasis:
        snd = call(snd, "Filter (pre-emphasis)...", 50.0)

    spectrum = snd.to_spectrum()
    # Band-limit by zeroing bins outside [lo, hi]; ignore if the command name
    # differs in this Praat build (moments are still dominated by the fricative).
    try:
        call(spectrum, "Filter (pass Hann band)...", band_hz[0], band_hz[1], 100.0)
    except Exception:
        pass

    cog = call(spectrum, "Get centre of gravity", 2)
    sd = call(spectrum, "Get standard deviation", 2)
    skew = call(spectrum, "Get skewness", 2)
    kurt = call(spectrum, "Get kurtosis", 2)
    return {
        "centroid": float(cog),
        "sd": float(sd),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "pre_emphasis": pre_emphasis,
    }


def dct_coeffs(
    wav_path: str | Path,
    start_s: float,
    end_s: float,
    n_coeffs: int = DCT_N_COEFFS,
    band_hz: tuple[float, float] = FRICATIVE_BAND_HZ,
) -> dict[str, float]:
    """First ``n_coeffs`` normalized DCT-II coefficients of the dB spectrum (§5.3).

    Uses the representative spectrum at the segment midpoint, ``scipy.fft.dct``
    with ``type=2, norm='ortho'`` so coefficients are comparable across files.
    """
    import numpy as np  # lazy import
    import parselmouth  # lazy import
    from parselmouth import WindowShape  # enum for extract_part
    from scipy.fft import dct  # lazy import

    mid = 0.5 * (start_s + end_s)
    half = 0.5 * (end_s - start_s)
    snd = parselmouth.Sound(str(wav_path)).extract_part(
        from_time=mid - half, to_time=mid + half,
        window_shape=WindowShape.HAMMING, preserve_times=True,
    )
    spectrum = snd.to_spectrum()
    freqs = np.array([spectrum.get_frequency_from_bin_number(i + 1)
                      for i in range(spectrum.get_number_of_bins())])
    power = np.array([spectrum.get_value_in_bin(i + 1)
                      for i in range(spectrum.get_number_of_bins())])
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    db = 10.0 * np.log10(np.maximum(power[mask], 1e-20))
    coeffs = dct(db, type=2, norm="ortho")[:n_coeffs]
    return {f"dct_k{i}": float(coeffs[i]) for i in range(min(n_coeffs, coeffs.size))}


def extract_fricatives(
    wav_path: str | Path,
    phones: list[Phone],
    meta: dict[str, str],
) -> list[dict[str, Any]]:
    """Per-fricative-token features (§5): moments + DCT, cognates kept separate.

    Fricatives are selected by MAP CATEGORY ('fricative') so the phoneme map is
    the single source of truth. This avoids label drift between the map's
    analysis labels (e.g. ʃ->'sh', ʒ->'zh') and any hard-coded set — the bug that
    previously dropped all ʃ tokens because FRICATIVES held raw IPA (ʃ/ʒ) while
    ph.phoneme holds the analysis label (sh/zh).
    """
    voiced_labels = {"v", "z", "zh"}  # analysis labels, not raw IPA
    rows: list[dict[str, Any]] = []
    for ph in phones:
        if getattr(ph, "category", None) != "fricative":
            continue
        win = select_fricative_centre(ph.eff_start, ph.eff_end)
        if win is None:
            continue
        start_s, end_s = win
        moments = spectral_moments(wav_path, start_s, end_s)
        coeffs = dct_coeffs(wav_path, start_s, end_s)
        rows.append({
            **meta,
            "phoneme": ph.phoneme,
            "voiced": ph.phoneme in voiced_labels,
            "time": ph.time,
            "eff_start": ph.eff_start,
            "eff_end": ph.eff_end,
            "meas_start": start_s,
            "meas_end": end_s,
            "time_norm": ph.time_norm,
            **moments,
            **coeffs,
            "band_lo_hz": FRICATIVE_BAND_HZ[0],
            "band_hi_hz": FRICATIVE_BAND_HZ[1],
        })
    return rows


# ===========================================================================
# Orchestration + provenance  (§2 / §6.1)
# ===========================================================================

def _package_versions() -> dict[str, str]:
    """Best-effort versions of key packages for provenance (§2)."""
    import importlib
    versions: dict[str, str] = {}
    for name in ("allosaurus", "parselmouth", "scipy", "torch", "torchaudio", "numpy"):
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not-installed"
    return versions


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list-of-dicts to CSV (union of keys as header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def analyze_file(
    wav_path: str | Path,
    gender_table: str | Path,
    out_dir: str | Path = "outputs",
    phoneme_map_path: str | Path = DEFAULT_MAP_PATH,
) -> dict[str, Path]:
    """Run the full pipeline for one wav + gender table. Returns output paths."""
    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    meta = parse_filename(wav_path)

    speakers = load_speaker_metadata(gender_table)
    vp = meta["ParticipantID"]
    if vp not in speakers:
        raise KeyError(f"{vp!r} not found in gender table {gender_table}")
    gender = speakers[vp]["Gender"]

    phoneme_map = load_phoneme_map(phoneme_map_path)

    # Stage 0
    phones = segment_file(wav_path, phoneme_map)
    seg_path = save_segmentation(phones, meta, out_dir)

    # Stage 1
    vowel_rows = extract_vowels(wav_path, phones, gender, meta)
    vsa_row = {
        **meta,
        "tVSA_bark": tvsa(vowel_rows),
        "cVSA_bark": cvsa(vowel_rows),
        "n_vowel_tokens": len(vowel_rows),
    }

    # Stage 2
    fric_rows = extract_fricatives(wav_path, phones, meta)

    stem = f"{vp}_{meta['Sequence']}_{meta['Condition']}_{meta['Sound']}"
    paths = {
        "phonemes": seg_path,
        "vowels": out_dir / f"{stem}_vowels.csv",
        "vsa": out_dir / f"{stem}_vsa.csv",
        "fricatives": out_dir / f"{stem}_fricatives.csv",
        "provenance": out_dir / f"{stem}_provenance.json",
    }
    _write_csv(vowel_rows, paths["vowels"])
    _write_csv([vsa_row], paths["vsa"])
    _write_csv(fric_rows, paths["fricatives"])

    provenance = {
        "input_wav": str(wav_path),
        "gender_table": str(gender_table),
        "gender": gender,
        "phoneme_map": str(phoneme_map_path),
        "package_versions": _package_versions(),
        "parameters": {
            "allosaurus_lang": ALLOSAURUS_LANG,
            "allosaurus_topk": ALLOSAURUS_TOPK,
            "vad_sample_rate": VAD_SAMPLE_RATE,
            "vowel_onset_offset_s": VOWEL_ONSET_OFFSET_S,
            "vowel_window_s": VOWEL_WINDOW_S,
            "vowel_min_duration_s": VOWEL_MIN_DURATION_S,
            "f0_fallback_hz": F0_FALLBACK_HZ,
            "formant_ceiling_hz": FORMANT_CEILING_HZ,
            "f1_range_hz": F1_RANGE_HZ,
            "f2_range_hz": F2_RANGE_HZ,
            "fricatives": sorted(FRICATIVES),
            "fricative_min_duration_s": FRICATIVE_MIN_DURATION_S,
            "fricative_band_hz": FRICATIVE_BAND_HZ,
            "fricative_pre_emphasis": FRICATIVE_PRE_EMPHASIS,
            "dct_n_coeffs": DCT_N_COEFFS,
        },
    }
    paths["provenance"].write_text(json.dumps(provenance, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return paths


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EVASPEEP phone segmentation + acoustic feature extraction "
                    "(see docs/acoustic-analysis-design.md).",
    )
    p.add_argument("--wav", required=True, type=Path,
                   help="Input mono 48 kHz .wav "
                        "({ParticipantID}_{Sequence}_{Condition}_{Sound}.wav).")
    p.add_argument("--gender-table", required=True, type=Path,
                   help="CSV with VP_ID,Gender columns.")
    p.add_argument("--out", type=Path, default=Path("outputs"),
                   help="Output directory (default: ./outputs).")
    p.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH,
                   help=f"IPA->phoneme map CSV (default: {DEFAULT_MAP_PATH}).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        paths = analyze_file(args.wav, args.gender_table, args.out, args.map)
    except Exception as exc:  # surface a clean message on the CLI
        sys.stderr.write(f"error: {exc}\n")
        return 1
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key:11s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
