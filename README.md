# Cleer — an open de-feedback for live vocals

A from-scratch reimplementation of the idea behind [Alpha Labs'
De-Feedback](https://www.alphalabsaudio.com/defeedback/): take a live vocal
signal and, in real time, **kill microphone feedback (howl/ring), remove
background noise, and reduce room reverb** — keeping the voice.

This repo has two parts:

| Folder        | What it is | Runs where |
|---------------|------------|------------|
| `reference/`  | A **runnable, validated Python reference** of the three DSP stages. This is where we figured out (and proved) the internals. | Any machine with Python |
| `ml/`         | The **neural denoise mask**: train it, validate it (it beats the classical denoiser), and export an **updatable CoreML model** the app can fine-tune on device. | Any machine with Python |
| `macos/`      | A **native macOS app** (Swift + Accelerate + CoreAudio + CoreML + SwiftUI) where you add **instances**, each assigned to an **audio device + channel**, running the same algorithms on live audio — including the neural mask and **on-device "train in the app"** personalisation. | Apple-silicon Mac, built in Xcode |

## How de-feedback actually works (the reverse-engineered internals)

Three stages, all low-latency:

1. **Feedback (howl) suppression** — *the* defining feature. Feedback is a
   closed acoustic loop that runs away at a room/mic resonance; in the spectrum
   it's a **narrow, tonal, persistent, rising peak**, which is exactly what
   tells it apart from voice (broadband, time-varying). A detector FFTs the
   signal and flags peaks that are prominent above the spectral floor, in the
   feedback band, and *persist* across frames; an actuator drops an **adaptive
   biquad notch** on each one. Detection is frequency-domain (accurate);
   filtering is time-domain (zero added latency). Same principle as dbx AFS /
   Behringer FBQ. → `feedback_suppressor.py`, `FeedbackSuppressor.swift`

2. **Noise reduction** — a **spectral gain mask** in [0,1] multiplied onto the
   STFT: keep voice bins, attenuate noise bins. Two interchangeable
   implementations behind one interface:
   - *classical* (minimum-statistics noise floor + Wiener gain) — no weights,
     → `noise_reducer.py`, `NoiseReducer.swift`;
   - *neural* (a tiny MLP predicting 32 mel-band gains, trained in `ml/`,
     shipped as CoreML, **fine-tunable on device**) → `ml/`,
     `NeuralNoiseReducer.swift`. It measurably beats the classical version
     (+5.45 dB vs +3.75 dB SNR — see `ml/README.md`).

3. **Dereverb** — model the late-reverb tail as a delayed, decayed copy of
   recent band power and subtract it with another spectral gain. → `dereverb.py`,
   `Dereverb.swift`

Stages 2 and 3 share **one STFT/ISTFT round-trip**; their masks multiply.

### Proof it works

```bash
cd reference
pip install -r requirements.txt
python3 tools/make_test_signal.py   # voice + 3147 Hz howl + noise + reverb
python3 tools/evaluate.py           # processes it and measures each stage
```

Measured on the synthetic torture-test signal:

```
[1] FEEDBACK   howl @ 3147 Hz attenuated by 60.9 dB (notch placed at 3158 Hz)
[2] NOISE+TAIL silent-gap residual reduced by 33.4 dB
[3] DEREVERB   post-word reverb tail shortened by 8.0 dB
```

## The macOS app

```bash
cd macos
brew install xcodegen      # one-time
xcodegen generate
open Cleer.xcodeproj       # build & run on your Mac
```

You get a window where you **Add instance**, pick an **input device + channel**
and **output device** per instance, toggle Feedback / Denoise / **Neural** /
Dereverb live (plus a master **Process / Bypass** switch to A/B the whole
chain), and watch level meters + the active-notch count. Each instance runs its
own `VoicePipeline`.

**One interface in, a different interface out:** yes. Each instance uses two
audio engines — a capture engine pinned to the input interface and a playback
engine pinned to the output interface — so they can be entirely different
devices. A ring buffer bridges their independent hardware clocks with drift
correction, so latency stays bounded. For *sample-accurate* sync across two
interfaces, you can alternatively make an **Aggregate Device** in Audio MIDI
Setup and select it for both; the UI flags when in/out are on different
interfaces.

**Setup recipes** (live mix; or testing a recorded vocal via BlackHole → Mac
speakers): see [`docs/usage.md`](docs/usage.md).

**Train in the app:** click **Personalise**, sample a few seconds of your room
noise (stay silent), and Cleer fine-tunes the CoreML denoiser to your
environment on-device via `MLUpdateTask` — the personalised model is saved to
Application Support and applied to every instance. Nothing leaves your Mac.

> The DSP core (`macos/Cleer/DSP/`) is a faithful port of the validated Python
> reference. The CoreAudio I/O layer (`macos/Cleer/Audio/`) is the one part that
> was written without a Mac to compile against (this was built in a Linux
> container) — expect minor on-device tuning there. `macos/Tests/DSPTests.swift`
> re-checks the DSP on-device.

## Is it using the GPU? (short answer: no, and that's deliberate)

See [`docs/compute.md`](docs/compute.md). TL;DR: the DSP runs on the **CPU via
Accelerate/vDSP** because GPU dispatch latency is poison for tiny real-time
audio buffers. The GPU / Apple Neural Engine only becomes relevant if we swap
the classical denoiser for a trained neural mask — at which point it runs via
**CoreML** on the ANE. The code is structured for exactly that upgrade.

## Latency

Feedback-only mode is ~0 ms algorithmic (time-domain notches). The full chain
adds one STFT window (~21 ms at the current settings); the neural net itself is
< 1 ms and never the bottleneck. Full breakdown and the knobs to cut it:
[`docs/latency.md`](docs/latency.md).

## Status / roadmap

- [x] Validated Python reference for all three stages
- [x] Swift DSP port (Accelerate)
- [x] SwiftUI instance/device/channel manager + CoreAudio engine
- [x] Neural denoise mask (trained, validated, exported to updatable CoreML)
- [x] On-device "train in the app" personalisation (MLUpdateTask)
- [ ] On-device compile + latency tuning of the CoreAudio layer
- [ ] Low-latency time-domain filterbank denoise (sub-10 ms full chain)
- [ ] Per-instance presets, persistence of layouts
```
