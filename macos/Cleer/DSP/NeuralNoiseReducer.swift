import Foundation
import CoreML

/// Neural denoiser: predicts a per-band gain mask with the CoreML model, then
/// interpolates to per-bin gains. Drop-in alternative to `NoiseReducer` — same
/// `mask(_:)` signature — so `VoicePipeline` can switch between classical and
/// neural at runtime.
///
/// The model is `CleerDenoiser.mlmodel` (trained + exported by `ml/`). It is
/// *updatable*, so `ModelTrainer` can fine-tune it on device and we reload the
/// personalised copy here via `reload(from:)`.
final class NeuralNoiseReducer {
    var enabled = true
    private let bands: MelBands
    private let fe: BandFeatureExtractor
    private var model: MLModel?
    private let nBands: Int

    private let inputArray: MLMultiArray   // reused to avoid per-frame alloc

    init?(bins: Int, sampleRate: Double, nBands: Int = 32) {
        self.bands = MelBands(nBins: bins, sr: sampleRate, nBands: nBands)
        self.fe = BandFeatureExtractor(bands: bands)
        self.nBands = nBands
        guard let arr = try? MLMultiArray(shape: [NSNumber(value: 2 * nBands)], dataType: .double) else {
            return nil
        }
        self.inputArray = arr
        guard let url = NeuralNoiseReducer.bundledModelURL() else { return nil }
        self.model = try? MLModel(contentsOf: url)
        if model == nil { return nil }
    }

    /// Compiled model shipped in the app bundle (Xcode compiles .mlmodel -> .mlmodelc).
    static func bundledModelURL() -> URL? {
        Bundle.main.url(forResource: "CleerDenoiser", withExtension: "mlmodelc")
    }

    /// Swap in a personalised model produced by on-device training.
    func reload(from url: URL) {
        if let m = try? MLModel(contentsOf: url) { self.model = m }
    }

    func mask(_ mag: [Double]) -> [Double] {
        let bins = mag.count
        guard enabled, let model = model else { return [Double](repeating: 1, count: bins) }

        var power = [Double](repeating: 0, count: bins)
        for k in 0..<bins { power[k] = mag[k] * mag[k] }
        let feat = fe.features(power: power)
        for i in 0..<feat.count { inputArray[i] = NSNumber(value: feat[i]) }

        guard
            let provider = try? MLDictionaryFeatureProvider(dictionary: ["features": inputArray]),
            let out = try? model.prediction(from: provider),
            let gains = out.featureValue(for: "gains")?.multiArrayValue
        else {
            return [Double](repeating: 1, count: bins)
        }
        var bandGains = [Double](repeating: 0, count: nBands)
        for b in 0..<nBands { bandGains[b] = gains[b].doubleValue }
        return bands.gainsToBins(bandGains)
    }
}
