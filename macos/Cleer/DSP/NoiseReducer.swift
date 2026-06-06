import Foundation

/// Spectral-gain noise reduction — Swift port of `noise_reducer.py`.
/// Tracks the noise floor with minimum statistics and returns a Wiener gain
/// mask per bin. `mask(_:)` has the same shape a CoreML gain-predictor would
/// have, so a trained model can replace this with no other changes.
final class NoiseReducer {
    struct Params {
        var overSubtraction = 1.5
        var gainFloorDB = -18.0
        var noiseSmooth = 0.9
        var minWindowFrames = 60
        var ddAlpha = 0.96
    }

    var params: Params
    var enabled = true
    private let bins: Int

    private var pSmooth: [Double]
    private var noisePSD: [Double]
    private var minBuf: [[Double]] = []
    private var prevGain: [Double]
    private var prevPower: [Double]
    private var initialised = false

    init(bins: Int, params: Params = Params()) {
        self.bins = bins
        self.params = params
        self.pSmooth = [Double](repeating: 0, count: bins)
        self.noisePSD = [Double](repeating: 1e-6, count: bins)
        self.prevGain = [Double](repeating: 1, count: bins)
        self.prevPower = [Double](repeating: 0, count: bins)
    }

    private func updateNoise(_ power: [Double]) {
        let a = params.noiseSmooth
        if !initialised {
            pSmooth = power; noisePSD = power; initialised = true
        } else {
            for k in 0..<bins { pSmooth[k] = a * pSmooth[k] + (1 - a) * power[k] }
        }
        minBuf.append(pSmooth)
        if minBuf.count > params.minWindowFrames { minBuf.removeFirst() }
        for k in 0..<bins {
            var m = Double.greatestFiniteMagnitude
            for frame in minBuf { m = min(m, frame[k]) }
            noisePSD[k] = 1.5 * m + 1e-12
        }
    }

    func mask(_ mag: [Double]) -> [Double] {
        guard enabled else { return [Double](repeating: 1, count: bins) }
        var power = [Double](repeating: 0, count: bins)
        for k in 0..<bins { power[k] = mag[k] * mag[k] }
        updateNoise(power)

        let floorGain = pow(10.0, params.gainFloorDB / 20.0)
        var gain = [Double](repeating: 0, count: bins)
        for k in 0..<bins {
            let noise = params.overSubtraction * noisePSD[k] + 1e-12
            let gamma = power[k] / noise
            var xi = params.ddAlpha * prevGain[k] * prevGain[k] * prevPower[k] / noise
                + (1 - params.ddAlpha) * max(gamma - 1.0, 0.0)
            xi = max(xi, 1e-6)
            var g = xi / (1.0 + xi)
            g = max(g, floorGain)
            gain[k] = g
        }
        prevGain = gain
        prevPower = power
        return gain
    }
}
