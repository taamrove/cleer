#pragma once
#include <cmath>

namespace cleer
{

// Second-order IIR (biquad), Direct Form I, with RBJ "cookbook" peaking-EQ
// coefficients. A peaking filter with a large *negative* gain is exactly the
// adaptive notch the feedback suppressor drops onto a howl. Port of the
// validated Swift/Python implementation (macos/Cleer/DSP/Biquad.swift,
// reference/defeedback/feedback_suppressor.py).
class Biquad
{
public:
    explicit Biquad (double sampleRateIn) : sampleRate (sampleRateIn) {}

    // Configure as a peaking EQ. gainDB < 0 => notch of that depth.
    void setPeaking (double freq, double q, double gainDB)
    {
        const double A = std::pow (10.0, gainDB / 40.0);
        const double w0 = 2.0 * M_PI * freq / sampleRate;
        const double cosw = std::cos (w0);
        const double alpha = std::sin (w0) / (2.0 * q);
        const double a0 = 1.0 + alpha / A;
        b0 = (1.0 + alpha * A) / a0;
        b1 = (-2.0 * cosw) / a0;
        b2 = (1.0 - alpha * A) / a0;
        a1 = (-2.0 * cosw) / a0;
        a2 = (1.0 - alpha / A) / a0;
    }

    inline double process (double x)
    {
        const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;
        return y;
    }

    void reset() { x1 = x2 = y1 = y2 = 0.0; }

private:
    double sampleRate;
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
};

} // namespace cleer
