# Cleer ML — neural denoise mask

A tiny neural net that predicts a per-band gain mask for the denoise stage,
plus everything to **train it, validate it, export it to CoreML, and fine-tune
it on device** inside the app.

## Why this design

- The net predicts gains for **32 mel bands**, not 513 FFT bins → it's a ~5k-param
  MLP that runs in well under a millisecond and trains in seconds.
- It's exported as a **CoreML updatable model**, so the macOS app can fine-tune
  it on the user's own room noise with `MLUpdateTask` (the "train in the app"
  feature) — no cloud, no labelled data.
- The classical Wiener denoiser stays in the app as a fallback / A-B reference.

## Pipeline

```bash
cd ml
pip install -r requirements.txt

python3 train.py          # train base model -> out/mask_mlp.npz
python3 evaluate_ml.py     # compare neural vs classical denoise (output SNR)
python3 export_coreml.py   # -> out/CleerDenoiser.mlmodel (updatable)
cp out/CleerDenoiser.mlmodel ../macos/Cleer/Resources/   # ship it in the app
```

## Validated results (run here)

```
base training:   val MSE 0.018   (gain=1 baseline 0.749)

neural vs classical, held-out 5 dB mixture (output SNR, higher better):
  noisy input        :  5.00 dB
  classical (Wiener) :  8.75 dB   (+3.75 dB)
  neural (MLP mask)  : 10.45 dB   (+5.45 dB)   <- better

export: CleerDenoiser.mlmodel  updatable=True
  inputs  features(64)  ->  outputs gains(32)
  training inputs: features, gains_true   (MSE loss, SGD)
```

## Files

| File | Role |
|------|------|
| `bands.py` | Mel filterbank (513 bins ↔ 32 bands). Mirrored by `MelBands.swift`. |
| `stft_np.py` | Frame-based STFT magnitude for offline training. |
| `dataset.py` | Synthesises clean speech + noise → features + Ideal-Ratio-Mask targets. Reused for on-device personalisation (swap in captured noise). |
| `model.py` | Pure-numpy MLP + Adam (forward/backprop/save/load). |
| `train.py` | Trains the base model. |
| `evaluate_ml.py` | Neural-vs-classical SNR comparison. |
| `export_coreml.py` | Emits the updatable `.mlmodel`. |

## Production notes

- "Clean speech" here is **synthesised**. For a real product, train on a speech
  corpus (e.g. read speech) mixed with a noise corpus — only `dataset.py`
  changes; the model, export, and Swift side stay the same.
- The Swift on-device trainer (`macos/Cleer/Audio/ModelTrainer.swift`) computes
  features/targets identically and feeds them to `MLUpdateTask`.
