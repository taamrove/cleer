# Latency budget

Where the delay actually comes from, with the numbers for the current config
(`sr = 48 kHz`, `win = 1024`, `hop = 256`).

## Per-stage

| Stage | Added latency | Why |
|-------|---------------|-----|
| **Feedback suppression** | **~0 ms** | Detection is FFT-based but the *filtering* is time-domain biquads applied sample-by-sample. No buffering delay — this is the whole point of doing notches in the time domain. |
| **Noise reduction** (classical or neural) | **= one STFT window** | Spectral masking needs a full analysis window before it can produce output: `1024 / 48000 = 21.3 ms`. |
| **Dereverb** | shares the same STFT | No *extra* latency — it rides on the same transform as denoise. |
| **Neural inference itself** | **< 1 ms** | The model is ~5k params (64→48→32). On the ANE/CPU that's tens of µs per frame — far inside the `5.3 ms` per-hop budget. The network is *not* the bottleneck. |
| **CoreAudio I/O** | 1 buffer in + 1 out | e.g. 128–256 samples each = `2.7–5.3 ms` per direction, set by the device/driver. |

## Totals

- **Feedback only** (denoise/dereverb off): essentially just the I/O buffers,
  **~5–11 ms** round trip. This is the "zero-latency feedback killer" mode.
- **Full chain** (denoise + dereverb on): **~21 ms** algorithmic + I/O,
  so **~27–32 ms** end to end.

## The knobs to cut the full-chain number

The STFT window dominates, so latency trades against frequency resolution:

| `win` / `hop` | STFT latency | Trade-off |
|---------------|--------------|-----------|
| 1024 / 256 (now) | 21.3 ms | best mask/notch frequency resolution |
| 512 / 128 | 10.7 ms | good balance |
| 256 / 64  | 5.3 ms  | low latency; coarser bands, gentler denoise |

Other levers:
- Lower the CoreAudio buffer (128 or 64 samples) on capable hardware.
- The **neural model never needs to grow** to hit low latency — it's already
  negligible; keep `win` small and it stays real-time.
- For *true* sub-10 ms full-chain, replace the STFT denoise with a low-latency
  **time-domain filterbank** (subband gains applied via short FIRs) — same mask
  idea, no window delay. That's the next step if you need it.

## What I have NOT measured on hardware

These are computed from the signal-processing structure, which is exact for the
algorithmic latency. The **I/O buffer numbers and the neural inference time are
estimates** until profiled on your Mac — the model size makes the inference
estimate safe, but profile `os_signpost` around `model.prediction` to confirm.
