import XCTest
@testable import Cleer

/// Mirrors the Python `evaluate.py` checks: feed the DSP a tone-plus-noise
/// signal and assert each stage measurably attenuates its target. These run on
/// an actual Mac (`xcodebuild test`) and are the on-device counterpart to the
/// validated Python reference.
final class DSPTests: XCTestCase {
    let sr = 48000.0

    func testSTFTReconstructsSignal() {
        let stft = STFT(win: 1024, hop: 256)
        let hop = 256
        var input = [Double]()
        for i in 0..<(hop * 200) { input.append(sin(2 * .pi * 440 * Double(i) / sr)) }

        var output = [Double]()
        var idx = 0
        while idx + hop <= input.count {
            let block = Array(input[idx..<idx + hop])
            let out = stft.process(block) { mags in [Double](repeating: 1, count: mags.count) }
            output.append(contentsOf: out)
            idx += hop
        }
        // After the one-window latency, output should track the input closely.
        let lat = 1024
        var err = 0.0, energy = 0.0
        for i in lat..<(output.count - hop) {
            let d = output[i] - input[i - lat]
            err += d * d; energy += input[i - lat] * input[i - lat]
        }
        XCTAssertLessThan(err / energy, 0.05, "STFT round-trip should reconstruct within 5%")
    }

    func testFeedbackSuppressorKillsTone() {
        let fb = FeedbackSuppressor(sampleRate: sr, win: 1024, hop: 256)
        let hop = 256
        let f = 3147.0
        var detected = false
        var lateEnergyDirty = 0.0, lateEnergyClean = 0.0
        let nBlocks = 600
        for b in 0..<nBlocks {
            var block = [Double](repeating: 0, count: hop)
            for i in 0..<hop {
                let t = Double(b * hop + i) / sr
                block[i] = 0.6 * sin(2 * .pi * f * t)
            }
            let before = block.reduce(0) { $0 + $1 * $1 }
            fb.process(&block)
            let after = block.reduce(0) { $0 + $1 * $1 }
            if b > 400 { lateEnergyDirty += before; lateEnergyClean += after }
        }
        detected = fb.detectedFrequencies.contains { abs($0 - f) < 60 }
        XCTAssertTrue(detected, "should detect the howl near \(f) Hz")
        let reductionDB = 10 * log10(lateEnergyClean / lateEnergyDirty)
        XCTAssertLessThan(reductionDB, -20, "howl should be attenuated >20 dB, got \(reductionDB)")
    }

    func testNoiseReducerAttenuatesNoiseFloor() {
        let bins = 1024 / 2 + 1
        let nr = NoiseReducer(bins: bins)
        // Flat noise spectrum -> low SNR -> gains should pull well below unity.
        let mag = [Double](repeating: 0.01, count: bins)
        var lastMean = 1.0
        for _ in 0..<100 { lastMean = nr.mask(mag).reduce(0, +) / Double(bins) }
        XCTAssertLessThan(lastMean, 0.5, "steady noise should be attenuated, mean gain \(lastMean)")
    }
}
