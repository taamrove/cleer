# CLAUDE.md — guidance for agents working in this repo

This file orients an agent (e.g. Claude Code running locally on macOS) so it can
continue the work. Read it before making changes.

## What this project is

**Cleer** is a from-scratch reimplementation of an AI "de-feedback" tool for live
vocals: suppress microphone **feedback (howl)**, reduce **noise**, and reduce
**reverb**, in real time. Three parts:

| Folder | Purpose | Runs where |
|--------|---------|------------|
| `reference/` | Validated Python reference of the 3 DSP stages | any Python |
| `ml/` | Neural denoise mask: train, validate, export updatable CoreML | any Python |
| `macos/` | Native app (Swift + Accelerate + CoreAudio + CoreML + SwiftUI) | Apple-silicon Mac, Xcode |

The user's real use case: **live mix** (audio interface in → out) and **testing**
a recorded vocal via **BlackHole → Mac speakers**. See `docs/usage.md`.

## IMPORTANT: history / how this was built

The first author worked in a **Linux container with no Swift/Xcode**, so:

- **Validated (actually run):** everything in `reference/` and `ml/` — including
  the measured results (howl −60.9 dB; neural denoise +5.45 dB SNR vs classical
  +3.75 dB; CoreML export `updatable=True`).
- **NOT compile-tested:** all of `macos/` (Swift). The DSP/ML Swift is a faithful
  line-by-line port of the validated Python, but the **CoreAudio I/O** and the
  **CoreML on-device training** layers were written blind.

So on a Mac, your job is usually: **build it, fix any compile errors, and verify
the two risky layers** (below). Prefer small fixes that keep the Swift aligned
with the Python reference.

## Build & run (macOS)

```bash
cd macos
brew install xcodegen        # one-time
xcodegen generate            # creates Cleer.xcodeproj from project.yml
open Cleer.xcodeproj         # set a Signing Team, then Run (Cmd-R)
# tests:
xcodebuild test -project Cleer.xcodeproj -scheme Cleer -destination 'platform=macOS'
```

- App is sandboxed with the audio-input entitlement (`Resources/Cleer.entitlements`)
  and declares `NSMicrophoneUsageDescription` (`Resources/Info.plist`). Signing a
  Team is required for the mic + for `MLUpdateTask` writing to Application Support.
- The trained model `Resources/CleerDenoiser.mlmodel` is committed; Xcode compiles
  it to `.mlmodelc` automatically. `CoreML computeUnits = .all` (uses the ANE).

## Verify-first list (the parts written without a compiler)

1. **CoreAudio routing** — `macos/Cleer/Audio/ProcessingInstance.swift`.
   Two `AVAudioEngine`s: a capture engine (input device, input tap → ring buffer)
   and a playback engine (output device, `AVAudioSourceNode` ← ring buffer), to
   support different in/out interfaces. Confirm:
   - the capture engine actually delivers audio with a **tap only** (no output
     connection); if not, connect `inputNode` to a zero-volume mixer to pump it.
   - device selection via `kAudioOutputUnitProperty_CurrentDevice` works on both
     `inputNode.audioUnit` and `outputNode.audioUnit`.
   - the ring-buffer drift correction keeps latency bounded across two clocks.
2. **On-device training** — `macos/Cleer/Audio/ModelTrainer.swift`.
   Uses `MLUpdateTask` against the bundled model. The model's training inputs are
   **`features` (64)** and **`gains_true` (32)** (MSE loss) — see
   `ml/export_coreml.py` output. Confirm the `MLUpdateProgressHandlers` API and
   `ctx.model.write(to:)` paths compile on the target OS.

## Architecture map (Python ↔ Swift must stay in sync)

| Concept | Python | Swift |
|--------|--------|-------|
| STFT/ISTFT | `reference/defeedback/stft.py` | `macos/Cleer/DSP/STFT.swift` (vDSP) |
| Feedback suppressor | `feedback_suppressor.py` | `FeedbackSuppressor.swift` (+ `SpectrumAnalyzer.swift`, `Biquad.swift`) |
| Classical denoise | `noise_reducer.py` | `NoiseReducer.swift` |
| Dereverb | `dereverb.py` | `Dereverb.swift` |
| Pipeline | `pipeline.py` | `VoicePipeline.swift` |
| Mel bands + features | `ml/bands.py`, `ml/dataset.py:FeatureExtractor` | `MelBands.swift` (incl. `BandFeatureExtractor`) |
| Neural mask | `ml/model.py` + CoreML export | `NeuralNoiseReducer.swift` |

**Rule:** if you change DSP/feature math in Swift, change the Python to match (or
vice-versa), and re-run the Python validators so behaviour stays proven.

## Validate the algorithms (any machine, fast)

```bash
cd reference && pip install -r requirements.txt
python3 tools/make_test_signal.py && python3 tools/evaluate.py   # howl/noise/reverb dB

cd ../ml && pip install -r requirements.txt
python3 train.py && python3 evaluate_ml.py        # neural vs classical SNR
python3 export_coreml.py                           # -> out/CleerDenoiser.mlmodel
cp out/CleerDenoiser.mlmodel ../macos/Cleer/Resources/   # ship retrained model
```

## Key facts / defaults

- `sr = 48000`, `win = 1024`, `hop = 256` (≈21 ms STFT latency; feedback path ≈0).
  Latency knobs in `docs/latency.md`.
- Denoise has two interchangeable backends behind one `mask()` interface:
  classical (`NoiseReducer`) and neural (`NeuralNoiseReducer`); per-instance
  **Neural** toggle picks between them.
- "Clean speech" in `ml/dataset.py` is **synthesised**. For a stronger model,
  swap in a real speech corpus — only `dataset.py` changes.
- Compute rationale (CPU/Accelerate now; ANE via CoreML for the model): `docs/compute.md`.

## Conventions

- Keep commits focused; end commit messages with the session URL line if present.
- Don't open a PR unless the user asks.
- Develop on the branch the user specifies; default work branch so far:
  `claude/de-feedback-internals-vgaZe`.
- Match the surrounding code style; keep the Python reference runnable (it's the
  source of truth for the DSP math).
