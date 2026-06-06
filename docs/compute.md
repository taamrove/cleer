# Does Cleer use the GPU?

**Right now: no. It runs on the CPU, using Apple's Accelerate (vDSP) framework.
That is the correct choice for this workload, and here's why.**

## Why not the GPU for the DSP

Live feedback suppression has to process audio in tiny buffers (we use a
256-sample hop ≈ 5 ms at 48 kHz). The work per buffer is small: one FFT, a few
hundred biquad samples, some vector multiplies.

The GPU is a throughput machine, not a latency machine. Each dispatch to the GPU
costs you:

- command-buffer encode + submit overhead,
- a round-trip across the CPU↔GPU memory boundary (even on unified-memory Apple
  silicon there's synchronization cost),
- scheduling latency you don't control on the audio render thread.

That overhead is **larger than the entire compute budget of one audio buffer**,
and the audio render callback is real-time: miss the deadline and you get a click
or dropout. So pushing per-buffer DSP to the GPU would *add* latency and
*reduce* reliability — the opposite of what a "zero-latency" feedback tool wants.

On Apple silicon the CPU path is also genuinely fast here:

- **vDSP** uses the NEON SIMD units for the FFT and vector math.
- The biquad notches are a recursive (IIR) filter — inherently sequential, so
  they don't parallelize onto a GPU well anyway.
- Everything stays on one thread, in cache, with deterministic timing.

This matches how real low-latency audio plugins are built: tight, SIMD CPU code,
not GPU compute.

## Where the GPU / Neural Engine *does* belong

The one stage that could become a real neural network is the **noise-reduction
mask** (and optionally dereverb). Today we compute that mask classically
(minimum-statistics + Wiener gain). If we replace it with a trained model
(RNNoise-style GRU, or a small U-Net), then:

- the model runs through **CoreML**, which automatically targets the **Apple
  Neural Engine (ANE)** first, then GPU, then CPU;
- the ANE is purpose-built for exactly this — low-power, low-latency small-tensor
  inference every few ms — and is the *right* accelerator, more so than the GPU.

The code is already shaped for this: `NoiseReducer.mask(_:)` takes a magnitude
spectrum and returns a per-bin gain. Swapping its body for a CoreML
`MLModel.prediction` call is the entire integration — nothing else in the
pipeline changes.

## Summary

| Workload                         | Best engine on Apple silicon | Used today |
|----------------------------------|------------------------------|------------|
| FFT / STFT                       | CPU (vDSP / NEON)            | ✅ |
| Biquad notch filters (IIR)       | CPU (sequential)            | ✅ |
| Spectral gain masks (classical)  | CPU (vDSP)                  | ✅ |
| Neural denoise mask (future)     | **ANE** via CoreML (GPU fallback) | ⬜ planned |

So: **CPU/Accelerate now by design; ANE (not really the GPU) is the accelerator
we'd reach for when we add a learned model.**
