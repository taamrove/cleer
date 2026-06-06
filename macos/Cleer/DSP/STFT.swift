import Foundation
import Accelerate

/// Streaming STFT/ISTFT built on Accelerate's vDSP real FFT — the Apple-silicon
/// optimised equivalent of the numpy `STFT` class in the Python reference.
///
/// It is fed `hop` samples at a time. Each push windows the latest `win`
/// samples, forward-FFTs them, hands the magnitude/complex spectrum to a gain
/// callback, inverse-FFTs, and overlap-adds into an output ring from which the
/// caller pops `hop` processed samples. Latency is one window.
final class STFT {
    let win: Int
    let hop: Int
    private let log2n: vDSP_Length
    private let fftSetup: FFTSetupD
    let bins: Int

    private var window: [Double]
    private var inRing: [Double]      // last `win` input samples
    private var olaBuffer: [Double]   // overlap-add accumulator (length win)
    private var realp: [Double]
    private var imagp: [Double]
    private let cola: Double

    init(win: Int = 1024, hop: Int = 256) {
        precondition(win % hop == 0, "win must be a multiple of hop")
        self.win = win
        self.hop = hop
        self.bins = win / 2 + 1
        self.log2n = vDSP_Length(log2(Double(win)))
        self.fftSetup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2))!

        // Periodic Hann.
        var w = [Double](repeating: 0, count: win)
        for i in 0..<win { w[i] = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(win)) }
        self.window = w

        // COLA constant for analysis*synthesis windowing.
        var norm = 0.0
        var k = 0
        while k < win { norm += w[k] * w[k]; k += hop }
        self.cola = norm

        self.inRing = [Double](repeating: 0, count: win)
        self.olaBuffer = [Double](repeating: 0, count: win)
        self.realp = [Double](repeating: 0, count: win / 2)
        self.imagp = [Double](repeating: 0, count: win / 2)
    }

    deinit { vDSP_destroy_fftsetupD(fftSetup) }

    /// Push `hop` input samples, get `hop` processed samples back.
    /// `applyGain(mag, bin) -> gain` returns a per-bin gain in [0, 1].
    func process(_ input: [Double], applyGain: ([Double]) -> [Double]) -> [Double] {
        precondition(input.count == hop)
        // Slide ring buffer.
        for i in 0..<(win - hop) { inRing[i] = inRing[i + hop] }
        for i in 0..<hop { inRing[win - hop + i] = input[i] }

        // Windowed forward FFT (packed real).
        var windowed = [Double](repeating: 0, count: win)
        vDSP_vmulD(inRing, 1, window, 1, &windowed, 1, vDSP_Length(win))

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

                // Unpack to magnitudes (vDSP packs Nyquist into imagp[0]).
                let nyq = ip[0]
                mags[0] = abs(rp[0]) * 0.5
                mags[win / 2] = abs(nyq) * 0.5
                for k in 1..<(win / 2) {
                    mags[k] = 0.5 * (rp[k] * rp[k] + ip[k] * ip[k]).squareRoot()
                }

                // Caller computes per-bin gain from magnitudes.
                let gain = applyGain(mags)

                // Apply gain to the complex spectrum.
                rp[0] *= gain[0]
                ip[0] *= gain[win / 2]
                for k in 1..<(win / 2) { rp[k] *= gain[k]; ip[k] *= gain[k] }

                vDSP_fft_zripD(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Inverse))
                var out = [Double](repeating: 0, count: win)
                out.withUnsafeMutableBufferPointer { ob in
                    ob.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: win / 2) {
                        vDSP_ztocD(&split, 1, $0, 2, vDSP_Length(win / 2))
                    }
                }
                // vDSP scaling: ifft needs /(2*win); then synthesis window and COLA.
                var scale = 1.0 / (2.0 * Double(win))
                vDSP_vsmulD(out, 1, &scale, &out, 1, vDSP_Length(win))
                vDSP_vmulD(out, 1, window, 1, &out, 1, vDSP_Length(win))

                // Overlap-add.
                for i in 0..<win { olaBuffer[i] += out[i] }
            }
        }

        // Pop `hop` finished samples and shift the OLA buffer.
        var result = [Double](repeating: 0, count: hop)
        let inv = 1.0 / cola
        for i in 0..<hop { result[i] = olaBuffer[i] * inv }
        for i in 0..<(win - hop) { olaBuffer[i] = olaBuffer[i + hop] }
        for i in (win - hop)..<win { olaBuffer[i] = 0 }
        return result
    }
}
