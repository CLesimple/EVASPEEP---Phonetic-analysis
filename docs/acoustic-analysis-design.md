# EVASPEEP — Acoustic Feature Analysis Design Document

**Status:** Design (reviewed, pre-implementation)
**Scope:** Phonemic segmentation of speech recordings followed by two independent acoustic analyses — (1) vowel-specific features and vowel space area, and (2) fricative distinction via spectral moments and DCT.
**Goal of the study:** Quantify the effect of hearing-aid amplification (`aided` vs `unaided`, treated as a fixed effect) on speech acoustics in adult talkers.

> This document captures the agreed design decisions. It deliberately does **not** contain the implementation functions — it is the specification that the implementation must follow.

---

## 1. Inputs

### 1.1 Audio files

- **Format:** mono `.wav`, **48 kHz**, **32-bit**, variable duration.
- **Recording conditions:** quiet, positive SNR.
- **Talkers:** adults only (HI and NH).

**Naming convention**

```
{ParticipantID}_{Sequence}_{Condition}_{Sound}.wav
```

| Field | Values |
|---|---|
| `ParticipantID` | `VP01`, `VP02`, … |
| `Sequence` | `M1`, `M2` |
| `Condition` | `aided`, `unaided` |
| `Sound` | `Bilder` (isolated words), `Nordwind` (read passage), `Diapix` (picture description), `Routine` (routine description), `Vokal` (sustained /a/) |

**Processed sounds:** all sounds **except `Vokal`** (sustained /a/ is out of scope for this pipeline).

**Sequence semantics:** `M1` and `M2` are a **test–retest repetition of the entire task set** (a reliability factor), **not** a within-session time axis. Session/task order is **randomized and counterbalanced across subjects**, which protects the amplification contrast from systematic order confounds. There are **no whole-recording wall-clock timestamps**.

All speech is **German**.

### 1.2 Speaker metadata file

A CSV with one row per participant is provided and consumed by the pipeline to set sex-dependent analysis parameters:

| Column | Values |
|---|---|
| `VP_ID` | `VP01`, `VP02`, … (must match `ParticipantID` in filenames) |
| `Gender` | `F` or `M` |

---

## 2. Reproducibility

Pin the following versions (record them in `environment.yml` / `requirements.txt` and echo them into every output for provenance):

| Package | Version |
|---|---|
| `allosaurus` | `1.0.2` |
| `praat-parselmouth` | `0.4.7` |
| `scipy` | `1.15.3` |
| `torch` | `2.12.0` |
| `torchaudio` | `2.11.0` |
| `torchcodec` | `0.14.0` |
| `silero-vad` | recent **5.x/6.x** release (compatible with the torch line above); pin exact version + model hash at implementation time |

> **Silero VAD note:** Silero is loaded as a separate package/model (via `torch.hub` or the `silero-vad` pip package), not pinned directly by torch. The given `torch 2.12.0` / `torchaudio 2.11.0` line is recent, so use a current Silero VAD (5.x/6.x); older Silero releases predate this torch line and may mismatch. `torchcodec 0.14.0` only matters if audio is decoded through torchcodec. Silero needs a **mono 16 kHz** tensor, so the VAD step resamples to 16 kHz independently of the 48 kHz kept for acoustic measurement.

Additional reproducibility rules:

- Persist **all parameters** used for every measurement (offsets, window length, formant ceiling, pitch floor/ceiling, frequency band, pre-emphasis flag, DCT settings) as columns/sidecar so any row can be reproduced.
- Record the **Allosaurus model version** used.
- Save QA plots (see §7).
- Pin the **statistics environment** used for §6 modeling: **R with `lme4` + `lmerTest`** (record R and package versions).

---

## 3. Stage 0 — Phonemic segmentation (Allosaurus)

### 3.1 Recognition call

- Use `allosaurus.app.read_recognizer()` with language **`deu`** and `timestamp=True`.
- Also request **candidate probabilities** (`topk` > 1) for quality assurance / confidence logging. Keep the top-1 phone as the working label and log the candidate set + probabilities.
- **Parsing fix:** the timestamped output has **no header row**. The first emitted line is a real phone. Parse **all** lines (`rows`, not `rows[1:]`) and cast `time` and `duration` to `float` immediately.

Output of recognition is **phones** (narrow IPA-like), *not* phonemes and *not* category labels.

### 3.2 IPA → analysis-phoneme mapping (new column)

Allosaurus emits phones; the analyses need a controlled set of **phonemes**. Add a dedicated mapping column. **Only the phonemes required by the analyses are mapped; everything else becomes `NA`.**

Mapping rules:

1. **Long vowels** (`ː`, e.g. `iː`) are **assimilated to their short counterpart** (`iː → i`).
2. **Diphthongs** → `NA`.
3. **Fricatives:** keep **only** `f v s z ʃ ʒ`. All other fricatives — notably `ç x h ʁ` — → `NA` (we restrict to the highest-frequency / sibilant + labiodental fricatives).
4. **`/a/ merge:** `ɑ` and `a` are merged to a single `a` category (relevant for tVSA corner vowels).
5. Everything not needed by either analysis → `NA`.

> The mapping table is the single largest source of downstream error and **must be reviewed and version-controlled** as an explicit asset: see `config/ipa_phoneme_map.csv` (currently a **draft scaffold** — verify against real Allosaurus `deu` output before production use).

### 3.3 Effective segment duration (Allosaurus + next-onset + VAD)

Allosaurus emits a **constant placeholder duration**, so a true per-phone duration/midpoint is not directly available. Therefore:

- Define each phone's effective interval as **`[t_i, t_{i+1}]`**, where `t_{i+1}` is the **start timestamp of the next phone** (for the last phone, use end-of-file or end-of-last-speech).
- **Silence guard (Silero VAD):** because a phone may be followed by a speech **pause**, the naive interval — and any midpoint within it — can fall into silence. Run **Silero VAD** to obtain speech regions and **clip every effective interval to voiced/speech regions only**. Measurement points (vowel sampling point, fricative central portion) **must lie inside a VAD speech segment**; otherwise the token is dropped.

### 3.4 Saved segmentation output

Save per file as CSV:

```
{ParticipantID}_{Sequence}_{Condition}_{Sound}_phonemes.csv
```

Suggested columns: `time`, `duration` (raw Allosaurus), `phone` (raw IPA), `phoneme` (mapped or `NA`), `t_next`, `eff_start`, `eff_end` (after VAD clipping), `time_norm` (token position 0–1, see §6.1), `topk_candidates`, `topk_probs`, plus filename metadata (`ParticipantID, Sequence, Condition, Sound`).

---

## 4. Stage 1 — Vowel-specific features

### 4.1 Segment selection

- **Onset-approximation mitigation:** apply a **+5 ms offset** to the (approximate) vowel onset to skip the transition into the steady state.
- **Window length:** **fixed 50 ms** analysis window. Rationale: long enough for stable LPC formant estimation (Praat internal 25 ms window) while remaining quasi-stationary. This is a **parameter** in the implementation.
- **Relative-position sampling (Hillenbrand-style):** rather than trusting a single absolute onset, position the sampling within the vowel using its **effective duration** (from §3.3, i.e. `[t_i, t_{i+1}]` clipped by VAD) — e.g. measure around the central portion (central third / 20–80%), bounded by the +5 ms offset and the 50 ms window. (Ref: Hillenbrand, Getty, Clark & Wheeler, 1995, *JASA* 97(5):3099–3111.)
- **Minimum duration:** **50 ms**. Vowels with effective duration < 50 ms are dropped.
- **Schwa:** **drop `ə`** (and reduced `ɐ`) from the vowel-space computation.

### 4.2 F0, F1, F2 (praat-parselmouth, VoiceLab parameters)

- **Sex-dependent formant ceiling** read from the speaker metadata `Gender` column, following VoiceLab's pitch/sex-dependent convention (`MeasureFormantNode` / `VoicelabNode.max_formant`): set per-speaker ceiling appropriate for F vs M adults (do **not** use a single global 5500 Hz for everyone).
- **F0 fallback:** if pitch estimation fails (needed to derive the ceiling), fall back to **300 Hz**.
- **Pitch floor/ceiling:** keep **VoiceLab defaults** for adults.
- **Aggregation:** take the **median** value across frames in the window for **F0, F1, and F2** (robust to spurious frames).

**Output addition:** add the **median F0, F1, F2** for each vowel phoneme to the output CSV.

### 4.3 Data frame assembly

Combine, per vowel token:

- Filename metadata (`ParticipantID, Sequence, Condition, Sound`),
- the vowel (mapped phoneme),
- the time stamp / effective interval and actual measurement point,
- median **F0**, **F1**, **F2** (Hz),
- the parameters used (formant ceiling, pitch floor/ceiling, window length, offset).

Keep raw values even when `NaN` (do not silently drop) for transparent QA.

### 4.4 Vowel Space Area (VSA)

**Both** measures are computed, **per recording** (i.e. per `{ParticipantID}_{Sequence}_{Condition}_{Sound}` file — VSA must be computed precisely for each recording).

**Units:** convert F1/F2 to **Bark** before area computation (enables within- and between-speaker comparison). Use VoiceLab's `hz_to_bark` convention.

#### 4.4.1 Triangle VSA (tVSA)
- Corner vowels **/i/, /a/, /u/**.
- **Long vowels assimilated to their short counterpart**; **`ɑ` and `a` merged**.
- Representative point per corner = **median across tokens** (Bark) for that recording.
- Area via the **shoelace formula**.
- **Minimum tokens per corner vowel: 3** (chosen because recordings differ in length). If a corner has < 3 tokens, tVSA for that recording is undefined (`NA`).

#### 4.4.2 Convex-hull VSA (cVSA)
- Preferred for robustness to vowel **misclassification**.
- Compute `scipy.spatial.ConvexHull` over all qualifying vowel tokens' (F1, F2) in **Bark**; the 2-D area is the hull's `.volume`.
- **Outlier handling — physiological range gate (Hz, before Bark conversion):**
  - **F1:** ~200–**1200** Hz
  - **F2:** ~700–**3500** Hz
  - Tokens outside these ranges are rejected. Optionally add a per-speaker×vowel robust trim (e.g. ±2–2.5 SD) so a single bad frame cannot inflate the hull.
  - Ranges follow Hillenbrand et al. (1995) adult vowel data, extended at the top of F1/F2 to safely accommodate open `/a/` and front `/i/` (esp. female talkers).

---

## 5. Stage 2 — Fricative distinction

Restricted to **`f v s z ʃ ʒ`** (from §3.2). `ç x h ʁ` are `NA`.

### 5.1 Segment selection

- Use the **central portion** of the fricative (exclude onset/offset transitions and any preceding stop burst), using the effective interval `[t_i, t_{i+1}]` (VAD-clipped, §3.3) to locate the centre.
- **Minimum duration: 30 ms** (fricative spectral analysis tolerates shorter windows than formant analysis).
- **Keep voiced and voiceless cognates separated** (`s/z`, `f/v`, `ʃ/ʒ`).

### 5.2 Method 1 — Spectral moments

Compute the four spectral moments — **mean (centroid / centre of gravity), variance (→ SD), skewness, kurtosis** — to distinguish e.g. `/s/` vs `/ʃ/`.

- **Pre-emphasis:** **none** (compute moments on the spectrum **without** additional pre-emphasis). Rationale: pre-emphasis tilts the spectrum and shifts centroid/skewness; the dominant convention (Forrest et al., 1988; Jongman, Wayland & Wong, 2000) computes moments without it. VoiceLab has no dedicated fricative-moments node to inherit from. Store a `pre_emphasis=False` flag for reproducibility.
- **Frequency band:** **500 Hz – 15 kHz** (source is 48 kHz, so high-frequency sibilant energy is preserved).
- **Windowing:** **Hamming** window, **averaged over frames** within the central portion.
- **Implementation:** use the **Praat / parselmouth** spectrum + moment calls (`To Spectrum`, then `Get centre of gravity`, `Get standard deviation`, `Get skewness`, `Get kurtosis`) for consistency with the rest of the stack.

### 5.3 Method 2 — Discrete Cosine Transform (DCT)

Following the EMU-SDMS recipe (Ch. 21, *Discrete Cosine Transform*):

- Compute the DCT-II on the **log / dB spectrum** of the **representative spectrum at the segment midpoint**.
- Keep the **first 3 coefficients** (`k0 ∝ mean level`, `k1 ∝ slope/tilt`, `k2 ∝ curvature`); `k1`/`k2` are strong `s`–`ʃ` discriminators.
- **Normalize** the coefficients (so they are comparable **across files** of differing length/Nyquist).
- Use `scipy.fft.dct(..., type=2, norm='ortho')`.

---

## 6. Cross-cutting decisions

| Topic | Decision |
|---|---|
| **Effect of interest** | Amplification (`aided` vs `unaided`) as a **fixed effect**. |
| **Recording quality** | Quiet, positive SNR; no explicit denoising assumed. |
| **Speaker normalization** | **Bark is the single normalization scheme** for all formant-based outcomes (tVSA, cVSA). The study is a **within-speaker** comparison (aided vs unaided per talker), so idiosyncratic between-speaker variation is allowed and acceptable; Bark is sufficient. **No pooled cross-speaker formant modeling / Lobanov normalization** is performed. |
| **Hearing-aid confound** | Aided processing (compression, noise reduction, feedback cancellation) can alter the spectrum (esp. high frequencies) and thus affect sibilant moments/DCT and possibly formants. Treat as an interpretive caveat, not just signal. |
| **Sampling rate caveat** | Allosaurus resamples internally to 16 kHz for *recognition*; all **acoustic measurements** are taken from the **original 48 kHz** audio, preserving the 500 Hz–15 kHz band for fricatives. |

### 6.1 Within-recording drift (time effect)

**Scope:** the only time analysis of interest is **within-recording drift** (e.g. warm-up / fatigue *inside* a single recording). Whole-session timing is **out of scope** (no session timestamps; `M1`/`M2` is test–retest, not a time axis — see §1.1).

**Predictor — `time_norm`:** for each token, `time_norm = token_timestamp / total_file_duration`, giving a position in **[0, 1] within the recording**.
- The denominator is the **full file duration** (start→end of file), chosen so a token's position can be mapped straight back onto the file. (Consequence: any leading/trailing silence is included in the scale — an accepted trade-off.)
- `time_norm` is a **token-level** predictor; it is most meaningful for **connected speech** (`Nordwind`, `Diapix`, `Routine`). For `Bilder` (isolated words) it is weaker and treated as secondary.
- Store `time_norm` in the per-token output (§3.4 / §4.3).

**Modeling (R / lme4 + lmerTest):**

- **Primary — linear drift** on a token-level feature (e.g. `F1_bark`, `F2_bark`, F0, spectral centroid, DCT `k1`/`k2`):

  ```r
  library(lme4); library(lmerTest)

  # Fit with ML (REML = FALSE) so nested models can be compared with anova()
  m_lin <- lmer(feature ~ Condition * time_norm + Sound +
                          (1 + time_norm | VP_ID) + (1 | VP_ID:Sequence),
                data = df, REML = FALSE)
  summary(m_lin)
  ```

  - `Condition * time_norm` tests both the amplification effect and whether drift differs by condition.
  - `Sound` is a covariate (materials differ systematically).
  - Random intercept + random `time_norm` slope per `VP_ID`; `Sequence` (M1/M2 test–retest) as a grouping factor.

- **Quadratic — only if a time effect appears.** If `time_norm` (or its interaction) is non-negligible, fit a 2nd-order polynomial and **test it against the linear model**; keep the quadratic **only if it significantly improves fit**:

  ```r
  m_quad <- lmer(feature ~ Condition * poly(time_norm, 2) + Sound +
                           (1 + time_norm | VP_ID) + (1 | VP_ID:Sequence),
                 data = df, REML = FALSE)
  anova(m_lin, m_quad)   # likelihood-ratio test; compare AIC/BIC
  ```

- **Recording-level outcomes (cVSA / tVSA) cannot use `time_norm`** — they are a single value per file. Model them **without** the drift term, e.g.:

  ```r
  m_vsa <- lmer(cVSA_bark ~ Condition + Sound +
                            (1 | VP_ID) + (1 | VP_ID:Sequence),
                data = df_vsa, REML = TRUE)
  ```

**Notes / caveats:**
- Within-file drift (`time_norm`) is **not** session-level fatigue; interpret accordingly.
- Several features are tested → control multiple comparisons (e.g. FDR) or pre-specify a primary outcome.
- Watch token attrition across the file (fewer tokens late in long recordings → noisier estimates); report token counts.
- Pin the R + `lme4`/`lmerTest` versions (§2).

---

## 7. Quality-assurance plots

Generate and save check plots:

- **Vowels:** F1–F2 scatter (Bark) per recording with the **tVSA triangle** and **cVSA convex hull** overlaid; mark rejected outliers.
- **Fricatives:** spectrum per token with the **centroid** marked; optionally the first DCT coefficients.
- **Segmentation:** spectrogram with Allosaurus phone boundaries and VAD speech regions overlaid for spot-checking.
- **Within-recording drift:** token-level feature vs `time_norm` (0–1) per recording, aided/unaided distinguished, to visually inspect drift and any `Condition × time_norm` interaction.

---

## 8. Segmentation validation (optional / sanity check)

Allosaurus timestamps are **approximate** (CTC, peaky emissions). Two more accurate aligners are available for validation. Neither replaces Allosaurus for spontaneous speech (Diapix/Routine), because they require a transcript; but for **Bilder** (known words) and **Nordwind** (canonical reading passage) they are gold-standard references.

### 8.1 WebMAUS (BAS Web Services) — REST API, scriptable

WebMAUS is a **forced aligner**: it needs **audio + transcript** and places known phones in time far more accurately than CTC. It exposes a REST API (BAS Web Services), so it can be driven from the command line / Python:

- Pipelines such as **`runPipeline`** (G2P → MAUS) accept the audio file, the orthographic/phonemic transcript, and `LANGUAGE=deu-DE`, and return a time-aligned segmentation (e.g. TextGrid).
- Typical usage: an HTTP `POST` (multipart form) with `curl` or Python `requests`, uploading `SIGNAL` (wav) and `TEXT` (transcript) plus parameters; parse the returned aligned file.
- Use it on a **subset** of files to quantify boundary agreement vs Allosaurus (e.g. mean absolute boundary error, vowel-midpoint placement).

### 8.2 Montreal Forced Aligner (MFA) — local CLI

- **MFA is a forced aligner, not a phone recognizer** — it is **not** a drop-in replacement for Allosaurus on unknown/spontaneous speech. It aligns a **known transcript** to audio.
- Installable via conda; **German acoustic + G2P models** are available.
- Recommended as the **best long-term aligner** where transcripts exist (Bilder, Nordwind), and as a second validation reference alongside WebMAUS.

**Plan:** run the sanity check (Allosaurus vs WebMAUS, and exploratory MFA) on a handful of Bilder/Nordwind files; report boundary agreement before relying on Allosaurus boundaries for the full corpus.

---

## 9. Open implementation notes / assets to version-control

- `config/ipa_phoneme_map.csv` — the IPA→phoneme mapping (§3.2). **Draft scaffold**; verify against real Allosaurus `deu` output.
- Speaker metadata CSV (`VP_ID`, `Gender`).
- Parameter defaults file (offsets, window, bands, DCT settings, pitch/formant settings).
- `requirements.txt` — pinned Python dependencies (§2).
- All outputs carry full parameter provenance and package versions (§2).

---

## 10. Key references

- Hillenbrand, J., Getty, L. A., Clark, M. J., & Wheeler, K. (1995). Acoustic characteristics of American English vowels. *JASA*, 97(5), 3099–3111. *(vowel sampling, F1/F2 ranges)*
- Chung et al. (2017). Vowel space area measures. PMC5724721. *(tVSA / cVSA)*
- Lobanov, B. M. (1971). Classification of Russian vowels spoken by different speakers. *JASA*, 49(2B), 606–608. *(speaker normalization — noted as not used; Bark only)*
- Adank, P., Smits, R., & van Hout, R. (2004). A comparison of vowel normalization procedures. *JASA*, 116(5), 3099–3107. *(normalization rationale)*
- Forrest, K., Weismer, G., Milenkovic, P., & Dougall, R. N. (1988). Statistical analysis of word-initial voiceless obstruents. *JASA*. *(spectral moments)*
- Jongman, A., Wayland, R., & Wong, S. (2000). Acoustic characteristics of English fricatives. *JASA*. *(spectral moments)*
- EMU-SDMS Manual, Ch. 21 — Discrete Cosine Transform. *(fricative DCT)*
- VoiceLab — Automated Reproducible Acoustical Analysis (Voice-Lab/VoiceLab). *(parselmouth pitch/formant parameters)*
- Allosaurus (xinjli/allosaurus). *(universal phone recognizer; approximate CTC timestamps)*
