"""Export the trained MLP to a CoreML *updatable* model.

The model is marked updatable so the macOS app can fine-tune it on device with
MLUpdateTask (the "train in the app" feature). It exposes:

  input   features : MultiArray(64)   -- [log band energy(32), band SNR(32)]
  output  gains    : MultiArray(32)   -- per-band gain in [0,1]
  loss    MSE between predicted gains and the supplied target gains
  optim   SGD, a few epochs (overridable from Swift at update time)

Runs on Linux/macOS (only writes the protobuf; no on-device runtime needed).
Output: ml/out/CleerDenoiser.mlmodel  -> copy into the Xcode project.
"""

from __future__ import annotations

import os
import numpy as np
import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.neural_network.update_optimizer_utils import SgdParams

N_IN, HIDDEN, N_OUT = 64, 48, 32


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    d = np.load(os.path.join(out_dir, "mask_mlp.npz"))
    W1, b1, W2, b2 = d["W1"], d["b1"], d["W2"], d["b2"]

    input_features = [("features", datatypes.Array(N_IN))]
    output_features = [("gains", datatypes.Array(N_OUT))]
    builder = NeuralNetworkBuilder(
        input_features, output_features,
        disable_rank5_shape_mapping=True)

    # inner_product expects W as (out_channels, in_channels) -> transpose ours.
    builder.add_inner_product(
        name="fc1", W=W1.T.copy(), b=b1.copy(),
        input_channels=N_IN, output_channels=HIDDEN, has_bias=True,
        input_name="features", output_name="fc1_out")
    builder.add_activation(
        name="relu1", non_linearity="RELU",
        input_name="fc1_out", output_name="relu1_out")
    builder.add_inner_product(
        name="fc2", W=W2.T.copy(), b=b2.copy(),
        input_channels=HIDDEN, output_channels=N_OUT, has_bias=True,
        input_name="relu1_out", output_name="fc2_out")
    builder.add_activation(
        name="sigmoid", non_linearity="SIGMOID",
        input_name="fc2_out", output_name="gains")

    # Make it trainable on device.
    builder.make_updatable(["fc1", "fc2"])
    builder.set_mean_squared_error_loss(
        name="lossLayer", input_feature=("gains", datatypes.Array(N_OUT)))
    builder.set_sgd_optimizer(SgdParams(lr=0.02, batch=16))
    builder.set_epochs(5)

    spec = builder.spec
    spec.description.metadata.shortDescription = (
        "Cleer per-band denoise mask predictor (updatable for on-device "
        "personalisation).")
    model = ct.models.MLModel(spec)
    path = os.path.join(out_dir, "CleerDenoiser.mlmodel")
    model.save(path)

    # Report what training inputs the app must provide.
    print("saved", os.path.abspath(path))
    print("inputs :", [i.name for i in spec.description.input])
    print("outputs:", [o.name for o in spec.description.output])
    print("training inputs:", [i.name for i in spec.description.trainingInput])
    print("updatable:", spec.isUpdatable)


if __name__ == "__main__":
    main()
