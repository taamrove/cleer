import Foundation

/// A second-order IIR (biquad) filter, Direct Form I, with RBJ "cookbook"
/// peaking-EQ coefficients. A peaking filter with a large *negative* gain is
/// exactly the adaptive notch the feedback suppressor drops onto a howl.
///
/// Processing is sample-by-sample so the filter state stays continuous across
/// audio buffers — essential for a notch whose depth changes every few ms.
final class Biquad {
    private var b0 = 1.0, b1 = 0.0, b2 = 0.0
    private var a1 = 0.0, a2 = 0.0
    private var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0

    let sampleRate: Double
    init(sampleRate: Double) { self.sampleRate = sampleRate }

    /// Configure as a peaking EQ. `gainDB < 0` => notch of that depth.
    func setPeaking(freq: Double, q: Double, gainDB: Double) {
        let A = pow(10.0, gainDB / 40.0)
        let w0 = 2.0 * Double.pi * freq / sampleRate
        let cosw = cos(w0)
        let alpha = sin(w0) / (2.0 * q)
        let a0 = 1.0 + alpha / A
        b0 = (1.0 + alpha * A) / a0
        b1 = (-2.0 * cosw) / a0
        b2 = (1.0 - alpha * A) / a0
        a1 = (-2.0 * cosw) / a0
        a2 = (1.0 - alpha / A) / a0
    }

    @inline(__always)
    func process(_ x: Double) -> Double {
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1; x1 = x
        y2 = y1; y1 = y
        return y
    }

    func reset() { x1 = 0; x2 = 0; y1 = 0; y2 = 0 }
}
