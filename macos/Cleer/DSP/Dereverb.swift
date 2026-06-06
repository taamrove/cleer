import Foundation

/// Late-reverberation suppression by spectral gain — Swift port of
/// `dereverb.py`. Models the reverb tail as a delayed, decayed copy of recent
/// band power and subtracts it.
final class Dereverb {
    struct Params {
        var rt60 = 0.6
        var delayMs = 40.0
        var strength = 1.0
        var gainFloorDB = -15.0
        var smooth = 0.6
    }

    var params: Params
    var enabled = true
    private let bins: Int
    private let delay: Int
    private let decay: Double
    private var powerHist: [[Double]] = []
    private var tail: [Double]

    init(bins: Int, sampleRate: Double, hop: Int, params: Params = Params()) {
        self.bins = bins
        self.params = params
        let frameS = Double(hop) / sampleRate
        self.delay = max(1, Int((params.delayMs / 1000.0 / frameS).rounded()))
        let decayPerFrame = pow(10.0, -3.0 * frameS / params.rt60)
        self.decay = pow(decayPerFrame, Double(self.delay))
        self.tail = [Double](repeating: 0, count: bins)
    }

    func mask(_ mag: [Double]) -> [Double] {
        guard enabled else { return [Double](repeating: 1, count: bins) }
        var power = [Double](repeating: 0, count: bins)
        for k in 0..<bins { power[k] = mag[k] * mag[k] }
        powerHist.append(power)
        if powerHist.count > delay + 1 { powerHist.removeFirst() }

        if powerHist.count > delay {
            let delayed = powerHist[0]
            for k in 0..<bins {
                let est = decay * delayed[k]
                tail[k] = params.smooth * tail[k] + (1 - params.smooth) * est
            }
        }

        let floorGain = pow(10.0, params.gainFloorDB / 20.0)
        var gain = [Double](repeating: 0, count: bins)
        for k in 0..<bins {
            let late = params.strength * tail[k]
            var g = max(power[k] - late, 0.0) / (power[k] + 1e-12)
            g = max(g, floorGain)
            gain[k] = g
        }
        return gain
    }
}
