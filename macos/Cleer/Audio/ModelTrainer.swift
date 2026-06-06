import Foundation
import CoreML
import Combine

/// On-device training ("train in the app"). The user captures a few seconds of
/// their room/mic noise; we mix it with synthetic clean speech at a range of
/// SNRs to build exact Ideal-Ratio-Mask targets, then fine-tune the updatable
/// CoreML model with `MLUpdateTask`. The personalised model is saved to
/// Application Support and hot-swapped into the running pipelines.
///
/// This adapts the denoiser to the user's actual noise (HVAC hum, fan, room
/// tone) — the part that benefits most from personalisation — without needing
/// any cloud training or labelled data.
final class ModelTrainer: ObservableObject {
    @Published var isTraining = false
    @Published var progress: Double = 0
    @Published var status = "Idle"
    @Published var hasPersonalizedModel = false

    let sr: Double
    let win: Int
    let hop: Int
    let nBands: Int
    private let bands: MelBands

    init(sr: Double = 48000, win: Int = 1024, hop: Int = 256, nBands: Int = 32) {
        self.sr = sr; self.win = win; self.hop = hop; self.nBands = nBands
        self.bands = MelBands(nBins: win / 2 + 1, sr: sr, nBands: nBands)
        self.hasPersonalizedModel = FileManager.default.fileExists(atPath: Self.personalizedURL().path)
    }

    static func personalizedURL() -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Cleer", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("CleerDenoiser.mlmodelc")
    }

    /// Fine-tune on captured noise. `noise` is mono float samples of room tone.
    func personalize(noise: [Float], epochs: Int = 8,
                     onComplete: @escaping (URL?) -> Void) {
        guard let modelURL = NeuralNoiseReducer.bundledModelURL() else {
            status = "Bundled model missing"; onComplete(nil); return
        }
        isTraining = true; progress = 0; status = "Building training data…"

        DispatchQueue.global(qos: .userInitiated).async {
            let batch = self.buildBatch(noise: noise.map { Double($0) })
            let config = MLModelConfiguration()
            config.computeUnits = .all   // ANE / GPU / CPU as available

            let handlers = MLUpdateProgressHandlers(
                forEvents: [.trainingBegin, .epochEnd],
                progressHandler: { ctx in
                    DispatchQueue.main.async {
                        if ctx.event == .epochEnd {
                            let metrics = ctx.metrics[.lossValue] as? Double ?? 0
                            self.progress = min(1, Double(ctx.metrics.count) / Double(epochs))
                            self.status = String(format: "Training… loss %.4f", metrics)
                        }
                    }
                },
                completionHandler: { ctx in
                    let outURL = Self.personalizedURL()
                    try? FileManager.default.removeItem(at: outURL)
                    do {
                        try ctx.model.write(to: outURL)
                        DispatchQueue.main.async {
                            self.isTraining = false; self.progress = 1
                            self.status = "Personalised model ready"
                            self.hasPersonalizedModel = true
                            onComplete(outURL)
                        }
                    } catch {
                        DispatchQueue.main.async {
                            self.isTraining = false
                            self.status = "Save failed: \(error.localizedDescription)"
                            onComplete(nil)
                        }
                    }
                })

            do {
                let task = try MLUpdateTask(forModelAt: modelURL,
                                            trainingData: batch,
                                            configuration: config,
                                            progressHandlers: handlers)
                task.resume()
            } catch {
                DispatchQueue.main.async {
                    self.isTraining = false
                    self.status = "Train init failed: \(error.localizedDescription)"
                    onComplete(nil)
                }
            }
        }
    }

    // MARK: - Training-data synthesis

    private func buildBatch(noise: [Double]) -> MLArrayBatchProvider {
        var providers: [MLFeatureProvider] = []
        var rng = SystemRandomNumberGenerator()
        let snrs: [Double] = [-3, 0, 3, 6, 10, 15]

        for snr in snrs {
            let clean = SyntheticVoice.utterance(sr: sr, seconds: 2.0, rng: &rng)
            let nz = tileNoise(noise, count: clean.count)
            let cStd = std(clean), nStd = std(nz) + 1e-9
            let g = pow(10.0, -snr / 20.0) * cStd / nStd
            var scaledNoise = nz; for i in 0..<scaledNoise.count { scaledNoise[i] *= g }
            var noisy = clean; for i in 0..<noisy.count { noisy[i] += scaledNoise[i] }

            let cMag = frameMags(clean)
            let nMag = frameMags(scaledNoise)
            let xMag = frameMags(noisy)
            let fe = BandFeatureExtractor(bands: bands)
            let frames = min(cMag.count, min(nMag.count, xMag.count))
            for t in 0..<frames {
                var px = [Double](repeating: 0, count: xMag[t].count)
                for k in 0..<px.count { px[k] = xMag[t][k] * xMag[t][k] }
                let feat = fe.features(power: px)

                let ec = bands.bandEnergy(power2(cMag[t]))
                let en = bands.bandEnergy(power2(nMag[t]))
                var target = [Double](repeating: 0, count: nBands)
                for b in 0..<nBands { target[b] = ec[b] / (ec[b] + en[b] + 1e-9) }

                if let fp = makeProvider(features: feat, target: target) {
                    providers.append(fp)
                }
            }
        }
        DispatchQueue.main.async { self.status = "Training on \(providers.count) frames…" }
        return MLArrayBatchProvider(array: providers)
    }

    private func makeProvider(features: [Double], target: [Double]) -> MLFeatureProvider? {
        guard
            let f = try? MLMultiArray(shape: [NSNumber(value: features.count)], dataType: .double),
            let y = try? MLMultiArray(shape: [NSNumber(value: target.count)], dataType: .double)
        else { return nil }
        for i in 0..<features.count { f[i] = NSNumber(value: features[i]) }
        for i in 0..<target.count { y[i] = NSNumber(value: target[i]) }
        return try? MLDictionaryFeatureProvider(dictionary: ["features": f, "gains_true": y])
    }

    // MARK: - helpers
    private func frameMags(_ x: [Double]) -> [[Double]] {
        let analyzer = SpectrumAnalyzer(win: win, hop: hop)
        var out: [[Double]] = []
        var idx = 0
        while idx + hop <= x.count {
            out.append(analyzer.push(Array(x[idx..<idx + hop])))
            idx += hop
        }
        return out
    }
    private func power2(_ mag: [Double]) -> [Double] { mag.map { $0 * $0 } }
    private func std(_ x: [Double]) -> Double {
        let m = x.reduce(0, +) / Double(x.count)
        let v = x.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(x.count)
        return v.squareRoot()
    }
    private func tileNoise(_ n: [Double], count: Int) -> [Double] {
        guard !n.isEmpty else { return [Double](repeating: 0, count: count) }
        var out = [Double](repeating: 0, count: count)
        for i in 0..<count { out[i] = n[i % n.count] }
        return out
    }
}

/// Minimal clean-speech synthesiser for building personalisation targets —
/// harmonic voiced bursts + gaps. Swift port of `dataset.clean_utterance`.
enum SyntheticVoice {
    static func utterance<R: RandomNumberGenerator>(sr: Double, seconds: Double, rng: inout R) -> [Double] {
        let n = Int(sr * seconds)
        var out = [Double](repeating: 0, count: n)
        var pos = 0
        while pos < n {
            let segS = Double.random(in: 0.12...0.30, using: &rng)
            let L = Int(segS * sr)
            let f0 = Double.random(in: 90...220, using: &rng)
            var phase = 0.0
            for i in 0..<L where pos + i < n {
                let f = f0 * (1 + 0.03 * sin(2 * .pi * 4 * Double(i) / sr))
                phase += 2 * .pi * f / sr
                var s = 0.0
                for h in 1...7 { s += (1.0 / Double(h)) * sin(Double(h) * phase) }
                let env = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(L))  // Hann
                out[pos + i] += s * env * 0.5
            }
            pos += L + Int(Double.random(in: 0.03...0.18, using: &rng) * sr)
        }
        return out
    }
}
