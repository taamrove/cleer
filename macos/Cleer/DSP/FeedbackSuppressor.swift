import Foundation

/// Adaptive feedback (howl) suppression.
///
/// Detection is frequency-domain; actuation is a pool of time-domain biquad
/// notches. The original Cleer implementation flagged any persistent tonal
/// peak as feedback — which on a recorded vocal placed hundreds of notches on
/// the singer's harmonics (sustained sung notes look exactly like persistent
/// tonal peaks). The key distinguishing feature of real feedback is that it
/// **grows exponentially** between frames until something limits it; a held
/// vocal note has roughly constant or decaying power. We gate notch placement
/// on per-bin growth in addition to prominence + persistence, so vocal
/// harmonics are left alone.
///
/// Defaults are tuned conservative — better to miss a borderline howl for one
/// frame than to gouge harmonics off the vocal.
final class FeedbackSuppressor {
    struct Params {
        var fLo = 300.0, fHi = 9000.0     // raised lo: most vocal fundamentals sit below 300
        var prominenceDB = 18.0           // must be 18 dB above local floor
        var holdDB = 9.0
        var persistFrames = 12.0          // ~64 ms at hop=256 / 48k — well past most syllable lengths
        var growthDBperFrame = 1.5        // bin must rise this fast across the persist window
        var maxDepthDB = 24.0             // shallower than original (was 36) — gentler on the program
        var q = 28.0                      // narrower notch — was 18
        var attackDB = 18.0
        var releaseDB = 2.5
        var maxNotches = 6                // was 16; you don't actually need that many simultaneous howls
        var floorWidthBins = 31           // half-width of the broadband floor estimator
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

    // Per-bin running state. Persist accumulates while bin is tonal+rising;
    // prevPowerDB feeds the growth check.
    private var persist: [Double]
    private var prevPowerDB: [Double]

    // Scratch buffers — pre-sized, reused, never reallocated.
    private var powerDB: [Double]
    private var floorDB: [Double]

    private var notches: [Notch] = []
    private(set) var detectedFrequencies: [Double] = []

    init(sampleRate: Double, win: Int, hop: Int, params: Params = Params()) {
        self.sampleRate = sampleRate; self.win = win; self.hop = hop
        self.params = params
        self.analyzer = SpectrumAnalyzer(win: win, hop: hop)
        self.bins = win / 2 + 1
        self.persist = [Double](repeating: 0, count: bins)
        self.prevPowerDB = [Double](repeating: -120, count: bins)
        self.powerDB = [Double](repeating: 0, count: bins)
        self.floorDB = [Double](repeating: 0, count: bins)
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
        // 1. Power in dB.
        for k in 0..<bins {
            let p = mags[k] * mags[k] + 1e-12
            powerDB[k] = 10.0 * log10(p)
        }

        // 2. Broadband floor via cheap sliding mean (NOT median — sorting a
        // 41-wide window per bin was the previous hot path). Average tracks the
        // broadband level fine for this purpose; tonal peaks still poke above.
        slidingMeanInPlace(powerDB, into: &floorDB, radius: params.floorWidthBins / 2)

        // 3. Per-bin assessment. Three independent conditions, all must hold:
        //    (a) in-band, (b) tonal+prominent above the local floor,
        //    (c) GROWING vs. previous frame (the feedback signature).
        let nyq = bins - 1
        for k in 1..<nyq {
            let f = binToFreq(k)
            let inBand = f >= params.fLo && f <= params.fHi
            let isPeak = powerDB[k] > powerDB[k - 1] && powerDB[k] > powerDB[k + 1]
            let prominent = (powerDB[k] - floorDB[k]) > params.prominenceDB
            let growing = (powerDB[k] - prevPowerDB[k]) > params.growthDBperFrame

            let candidate = inBand && isPeak && prominent && growing
            if candidate {
                persist[k] += 1.0
            } else {
                // Decay quickly so a single quiet frame breaks persistence.
                persist[k] *= 0.5
            }

            // Allocate a notch only on sustained-AND-still-growing confirmation.
            if persist[k] >= params.persistFrames {
                let already = notches.contains { abs(freqToBin($0.freq) - k) <= 1 }
                if !already && notches.count < params.maxNotches {
                    let freq = interpFreq(mags, k)
                    let n = Notch(biquad: Biquad(sampleRate: sampleRate), freq: freq)
                    notches.append(n)
                    detectedFrequencies.append(freq)
                    if detectedFrequencies.count > 64 { detectedFrequencies.removeFirst() }
                }
            }
        }
        // Update prev frame.
        for k in 0..<bins { prevPowerDB[k] = powerDB[k] }

        // 4. Drive depths; release & reap dead notches.
        var survivors: [Notch] = []
        for n in notches {
            let b = freqToBin(n.freq)
            let hot = b >= 0 && b < bins && (powerDB[b] - floorDB[b]) > params.holdDB
            let target = hot ? -params.maxDepthDB : 0.0
            if n.depthDB > target { n.depthDB = max(target, n.depthDB - params.attackDB) }
            else { n.depthDB = min(target, n.depthDB + params.releaseDB) }
            n.biquad.setPeaking(freq: n.freq, q: params.q, gainDB: n.depthDB)
            n.idle = n.depthDB >= -1e-6 ? n.idle + 1 : 0
            if n.idle < 20 { survivors.append(n) }
        }
        notches = survivors
    }

    private func interpFreq(_ mags: [Double], _ k: Int) -> Double {
        guard k > 0, k < bins - 1 else { return binToFreq(k) }
        let a = log(mags[k - 1] * mags[k - 1] + 1e-12)
        let b = log(mags[k] * mags[k] + 1e-12)
        let c = log(mags[k + 1] * mags[k + 1] + 1e-12)
        let denom = a - 2 * b + c
        var delta = denom != 0 ? 0.5 * (a - c) / denom : 0.0
        delta = min(0.5, max(-0.5, delta))
        return (Double(k) + delta) * sampleRate / Double(win)
    }

    var activeNotchCount: Int { notches.filter { $0.depthDB < -1.0 }.count }
}

/// Sliding mean over `x` into `out`. O(n) — uses a running sum, no per-bin
/// inner loop or sort. Replaces the old median-filter floor estimator that
/// sorted a 41-wide window 513 times per hop (~80 k ops per hop just for floor).
private func slidingMeanInPlace(_ x: [Double], into out: inout [Double], radius: Int) {
    let n = x.count
    if n == 0 { return }
    let r = max(1, radius)
    // Prefix sum trick avoids an inner loop. Allocates one array of size n+1
    // per call — bounded by `bins` (513) and only happens once per hop.
    var sum = [Double](repeating: 0, count: n + 1)
    for i in 0..<n { sum[i + 1] = sum[i] + x[i] }
    for i in 0..<n {
        let lo = max(0, i - r)
        let hi = min(n - 1, i + r)
        let w = Double(hi - lo + 1)
        out[i] = (sum[hi + 1] - sum[lo]) / w
    }
}
