"""Train the base mask predictor and save weights to ml/out/mask_mlp.npz."""

from __future__ import annotations

import os
import numpy as np

from dataset import make_examples
from model import MaskMLP

N_BANDS = 32
HIDDEN = 48


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    print("generating training data ...")
    X, Y, bands = make_examples(n_bands=N_BANDS, n_utts=160,
                                rng=np.random.default_rng(0))
    print(f"  {X.shape[0]} frames, {X.shape[1]} features -> {Y.shape[1]} band gains")

    model = MaskMLP(n_in=2 * N_BANDS, n_hidden=HIDDEN, n_out=N_BANDS, seed=1)
    print("training ...")
    hist = model.train(X, Y, epochs=60, batch=256, lr=2e-3)
    for ep in (0, 9, 29, 59):
        tr, va = hist[ep]
        print(f"  epoch {ep+1:3d}  train MSE {tr:.4f}  val MSE {va:.4f}")

    # Baseline: always-pass (gain=1) and IRM-mean, to contextualise the MSE.
    base = float(np.mean((1.0 - Y) ** 2))
    print(f"  (gain=1 baseline MSE {base:.4f})")

    path = os.path.join(out_dir, "mask_mlp.npz")
    model.save(path)
    print("saved", os.path.abspath(path))


if __name__ == "__main__":
    main()
