"""Tiny MLP mask predictor (pure numpy, Adam).

Architecture:  in(2*B) -> Dense(H) -> ReLU -> Dense(B) -> Sigmoid  (= band gains)

It is deliberately small (~a few thousand weights) for three reasons:
  * it runs in microseconds per frame on the ANE/CPU,
  * it trains fast, and
  * it maps cleanly onto a CoreML *updatable* model so the app can fine-tune it
    on device with MLUpdateTask.

We train in numpy here so the reference is dependency-light and fully runnable;
`export_coreml.py` copies these exact weights into the .mlmodel.
"""

from __future__ import annotations

import numpy as np


class MaskMLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He init for ReLU layer, Xavier for the sigmoid layer.
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        self._t = 0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        y = self._sigmoid(z2)
        return y, (X, z1, a1, z2, y)

    def predict(self, X):
        return self.forward(X)[0]

    def train(self, X, Y, epochs=40, batch=256, lr=2e-3, val=0.1, seed=0):
        rng = np.random.default_rng(seed)
        n = len(X)
        idx = rng.permutation(n)
        n_val = int(n * val)
        vi, ti = idx[:n_val], idx[n_val:]
        Xv, Yv = X[vi], Y[vi]
        history = []
        for ep in range(epochs):
            perm = rng.permutation(len(ti))
            for s in range(0, len(perm), batch):
                bi = ti[perm[s:s + batch]]
                self._step(X[bi], Y[bi], lr)
            tr = self._mse(X[ti], Y[ti])
            va = self._mse(Xv, Yv)
            history.append((tr, va))
        return history

    def _mse(self, X, Y):
        return float(np.mean((self.predict(X) - Y) ** 2))

    def _step(self, X, Y, lr):
        y, (X_, z1, a1, z2, _) = self.forward(X)
        m = len(X)
        # MSE loss gradient through sigmoid.
        dz2 = (y - Y) * y * (1 - y) * (2.0 / m)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_.T @ dz1
        db1 = dz1.sum(0)
        self._adam({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, lr)

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(d["W1"].shape[0], d["W1"].shape[1], d["W2"].shape[1])
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        m._init_adam()
        return m
