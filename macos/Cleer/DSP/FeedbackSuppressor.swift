import Foundation

/// Adaptive feedback (howl) suppression — the Swift port of the validated
/// `feedback_suppressor.py`. Detection is frequency-domain (peaks that are
/// tonal, prominent, in-band and persistent); the actuator is a pool of
/// time-domain biquad notches whose depth tracks each howl.
///
/// See the Python reference for the full commentary on the algorithm.
final class FeedbackSuppressor {
    struct Params {
        var fLo = 150.0, fHi = 9000.0
        var prominenceDB = 12.0
        var holdDB = 6.0
        var persistFrames = 6.0
        var maxDepthDB = 36.0
        var q = 18.0
        var attackDB = 12.0
        var releaseDB = 1.5
        var maxNotches = 16
        var floorMedian = 41   // bins for the spectral-floor median filter
    }

    private final class Notch {
        let biquad: Biquad
        var freq: Double
        var depthDB = 0.0
        var idle = 0
        init(biquad: Biquad, freq: Double) { self.biquad = biquad; self.freq = freq }
    }

    let sampleRate: Double
    let win: Int
    let hop: Int
    var params: Params
    var enabled = true

    private let analyzer: SpectrumAnalyzer
    private let bins: Int
    private var persist: [Double]
    private var notches: [Notch] = []
    private(set) var detectedFrequencies: [Double] = []

    init(sampleRate: Double, win: Int, hop: Int, params: Params = Params()) {
        self.sampleRate = sampleRate; self.win = win; self.hop = hop
        self.params = params
        self.analyzer = SpectrumAnalyzer(win: win, hop: hop)
        self.bins = win / 2 + 1
        self.persist = [Double](repeating: 0, count: bins)
    }

    private func binToFreq(_ k: Int) -> Double { Double(k) * sampleRate / Double(win) }
    private func freqToBin(_ f: Double) -> Int { Int((f * Double(win) / sampleRate).rounded()) }

    /// Process one hop-sized block in place (time domain) and update notches.
    func process(_ block: inout [Double]) {
        let mags = analyzer.push(block)
        if enabled { updateNotches(mags) }
        guard enabled, !notches.isEmpty else { return }
        for i in 0..<block.count {
            var s = block[i]
            for n in notches { s = n.biquad.process(s) }
            block[i] = s
        }
    }

    private func updateNotches(_ mags: [Double]) {
        // Power and spectral floor (median over frequency).
        var power = [Double](repeating: 0, count: bins)
        for k in 0..<bins { power[k] = mags[k] * mags[k] + 1e-12 }
        let floor = medianFilter(power, radius: params.floorMedian / 2)

        var promDB = [Double](repeating: 0, count: bins)
        for k in 0..<bins { promDB[k] = 10.0 * log10(power[k] / (floor[k] + 1e-12)) }

        // Candidate peaks: in-band tonal local maxima above the floor.
        var candidate = [Bool](repeating: false, count: bins)
        for k in 1..<(bins - 1) {
            let f = binToFreq(k)
            if f >= params.fLo && f <= params.fHi
                && power[k] > power[k - 1] && power[k] > power[k + 1]
                && promDB[k] > params.prominenceDB {
                candidate[k] = true
            }
        }

        // Persistence: tonal feedback survives, transients decay.
        for k in 0..<bins { persist[k] = candidate[k] ? persist[k] + 1.0 : persist[k] * 0.5 }

        // Allocate notches for confirmed howls.
        for k in 0..<bins where persist[k] >= params.persistFrames {
            let already = notches.contains { abs(freqToBin($0.freq) - k) <= 1 }
            if already || notches.count >= params.maxNotches { continue }
            let freq = interpFreq(power, k)
            let n = Notch(biquad: Biquad(sampleRate: sampleRate), freq: freq)
            notches.append(n)
            detectedFrequencies.append(freq)
            if detectedFrequencies.count > 64 { detectedFrequencies.removeFirst() }
        }

        // Drive depths; release & reap dead notches.
        var survivors: [Notch] = []
        for n in notches {
            let b = freqToBin(n.freq)
            let hot = b >= 0 && b < bins && promDB[b] > params.holdDB
            let target = hot ? -params.maxDepthDB : 0.0
            if n.depthDB > target { n.depthDB = max(target, n.depthDB - params.attackDB) }
            else { n.depthDB = min(target, n.depthDB + params.releaseDB) }
            n.biquad.setPeaking(freq: n.freq, q: params.q, gainDB: n.depthDB)
            n.idle = n.depthDB >= -1e-6 ? n.idle + 1 : 0
            if n.idle < 20 { survivors.append(n) }
        }
        notches = survivors
    }

    private func interpFreq(_ power: [Double], _ k: Int) -> Double {
        guard k > 0, k < bins - 1 else { return binToFreq(k) }
        let a = log(power[k - 1] + 1e-12), b = log(power[k] + 1e-12), c = log(power[k + 1] + 1e-12)
        let denom = a - 2 * b + c
        var delta = denom != 0 ? 0.5 * (a - c) / denom : 0.0
        delta = min(0.5, max(-0.5, delta))
        return (Double(k) + delta) * sampleRate / Double(win)
    }

    var activeNotchCount: Int { notches.filter { $0.depthDB < -1.0 }.count }
}

/// Simple sliding median over frequency — estimates the broadband spectral
/// floor so that narrow tonal peaks stand out from it.
func medianFilter(_ x: [Double], radius: Int) -> [Double] {
    let n = x.count
    var out = [Double](repeating: 0, count: n)
    var windowBuf = [Double]()
    windowBuf.reserveCapacity(2 * radius + 1)
    for i in 0..<n {
        windowBuf.removeAll(keepingCapacity: true)
        let lo = max(0, i - radius), hi = min(n - 1, i + radius)
        for j in lo...hi { windowBuf.append(x[j]) }
        windowBuf.sort()
        out[i] = windowBuf[windowBuf.count / 2]
    }
    return out
}
