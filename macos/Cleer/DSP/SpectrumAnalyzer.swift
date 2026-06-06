import Foundation
import Accelerate

/// Analysis-only windowed magnitude spectrum, fed `hop` samples at a time.
/// Used by the feedback detector, which needs the spectrum of the *input*
/// (before the notches) to decide where to place them.
final class SpectrumAnalyzer {
    let win: Int
    let hop: Int
    let bins: Int
    private let log2n: vDSP_Length
    private let fftSetup: FFTSetupD
    private var window: [Double]
    private var ring: [Double]
    private var realp: [Double]
    private var imagp: [Double]

    init(win: Int, hop: Int) {
        self.win = win; self.hop = hop; self.bins = win / 2 + 1
        self.log2n = vDSP_Length(log2(Double(win)))
        self.fftSetup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2))!
        var w = [Double](repeating: 0, count: win)
        for i in 0..<win { w[i] = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(win)) }
        self.window = w
        self.ring = [Double](repeating: 0, count: win)
        self.realp = [Double](repeating: 0, count: win / 2)
        self.imagp = [Double](repeating: 0, count: win / 2)
    }

    deinit { vDSP_destroy_fftsetupD(fftSetup) }

    /// Slide in `hop` samples and return |X| (power-of-two windowed magnitude).
    func push(_ input: [Double]) -> [Double] {
        for i in 0..<(win - hop) { ring[i] = ring[i + hop] }
        for i in 0..<hop { ring[win - hop + i] = input[i] }

        var windowed = [Double](repeating: 0, count: win)
        vDSP_vmulD(ring, 1, window, 1, &windowed, 1, vDSP_Length(win))

        var mags = [Double](repeating: 0, count: bins)
        realp.withUnsafeMutableBufferPointer { rp in
            imagp.withUnsafeMutableBufferPointer { ip in
                var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                windowed.withUnsafeBufferPointer { wb in
                    wb.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: win / 2) {
                        vDSP_ctozD($0, 2, &split, 1, vDSP_Length(win / 2))
                    }
                }
                vDSP_fft_zripD(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                mags[0] = abs(rp[0]) * 0.5
                mags[win / 2] = abs(ip[0]) * 0.5
                for k in 1..<(win / 2) {
                    mags[k] = 0.5 * (rp[k] * rp[k] + ip[k] * ip[k]).squareRoot()
                }
            }
        }
        return mags
    }
}
