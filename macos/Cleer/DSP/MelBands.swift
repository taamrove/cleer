import Foundation

/// Mel band filterbank — exact Swift mirror of `ml/bands.py`. Compresses the
/// 513-bin spectrum to `nBands` perceptual bands for the neural mask, and
/// spreads the predicted band gains back onto bins.
final class MelBands {
    let nBins: Int
    let nBands: Int
    let sr: Double
    private let bandMatrix: [[Double]]   // nBands x nBins  (analysis)
    private let interp: [[Double]]       // nBins x nBands  (synthesis)

    init(nBins: Int, sr: Double, nBands: Int = 32) {
        self.nBins = nBins; self.nBands = nBands; self.sr = sr

        func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

        var binFreqs = [Double](repeating: 0, count: nBins)
        for k in 0..<nBins { binFreqs[k] = Double(k) * (sr / 2) / Double(nBins - 1) }

        let mLo = hzToMel(0), mHi = hzToMel(sr / 2)
        var edges = [Double](repeating: 0, count: nBands + 2)
        for i in 0..<(nBands + 2) {
            edges[i] = melToHz(mLo + (mHi - mLo) * Double(i) / Double(nBands + 1))
        }
        let centers = Array(edges[1..<(nBands + 1)])

        // Analysis filterbank (triangular, row-normalised).
        var W = [[Double]](repeating: [Double](repeating: 0, count: nBins), count: nBands)
        for b in 0..<nBands {
            let lo = edges[b], ctr = edges[b + 1], hi = edges[b + 2]
            var rowSum = 0.0
            for k in 0..<nBins {
                let f = binFreqs[k]
                var w = 0.0
                if f >= lo && f <= ctr && ctr > lo { w = (f - lo) / (ctr - lo) }
                else if f > ctr && f <= hi && hi > ctr { w = (hi - f) / (hi - ctr) }
                W[b][k] = w; rowSum += w
            }
            if rowSum > 0 { for k in 0..<nBins { W[b][k] /= rowSum } }
        }
        self.bandMatrix = W

        // Synthesis: linear interp across nearest band centres (rows sum to 1).
        var B = [[Double]](repeating: [Double](repeating: 0, count: nBands), count: nBins)
        for k in 0..<nBins {
            let f = binFreqs[k]
            if f <= centers[0] { B[k][0] = 1.0 }
            else if f >= centers[nBands - 1] { B[k][nBands - 1] = 1.0 }
            else {
                var j = 0
                while j < nBands - 1 && centers[j + 1] < f { j += 1 }
                let f0 = centers[j], f1 = centers[j + 1]
                let w = (f - f0) / (f1 - f0)
                B[k][j] = 1 - w; B[k][j + 1] = w
            }
        }
        self.interp = B
    }

    func bandEnergy(_ power: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: nBands)
        for b in 0..<nBands {
            var s = 0.0
            let row = bandMatrix[b]
            for k in 0..<nBins { s += row[k] * power[k] }
            out[b] = s
        }
        return out
    }

    func gainsToBins(_ bandGains: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: nBins)
        for k in 0..<nBins {
            var s = 0.0
            let row = interp[k]
            for b in 0..<nBands { s += row[b] * bandGains[b] }
            out[k] = s
        }
        return out
    }
}

/// Running per-band noise-floor tracker producing the 64-dim feature vector
/// [log band energy (32), band SNR (32)] — identical to `dataset.FeatureExtractor`.
final class BandFeatureExtractor {
    private let bands: MelBands
    private let minWindow: Int
    private var buf: [[Double]] = []

    init(bands: MelBands, minWindow: Int = 60) {
        self.bands = bands; self.minWindow = minWindow
    }

    func reset() { buf.removeAll(keepingCapacity: true) }

    func features(power: [Double]) -> [Double] {
        var e = bands.bandEnergy(power)
        for i in 0..<e.count { e[i] += 1e-9 }
        buf.append(e)
        if buf.count > minWindow { buf.removeFirst() }
        var floor = e
        for frame in buf { for i in 0..<floor.count { floor[i] = min(floor[i], frame[i]) } }
        var feat = [Double](repeating: 0, count: 2 * bands.nBands)
        for i in 0..<bands.nBands {
            let le = log(e[i])
            feat[i] = le
            feat[bands.nBands + i] = le - log(floor[i] + 1e-9)
        }
        return feat
    }
}
