#pragma once
#include <functional>
#include <vector>
#include "Fft.h"

namespace cleer
{

// Streaming overlap-add STFT: feed one hop of samples, get one hop back,
// with a per-bin magnitude gain applied in between. Hann window with
// hop = win/4 satisfies COLA, so with unity gains the round trip is exact
// up to float precision. C++ twin of reference/defeedback/stft.py.
//
// Inherent latency: win - hop samples (the last frame that can still touch
// an output sample must arrive before that sample is emitted), plus the
// caller's hop accumulation.
class OlaStft
{
public:
    OlaStft (int winIn = 1024, int hopIn = 256)
        : win (winIn), hop (hopIn), bins (winIn / 2 + 1), fft (winIn)
    {
        window.resize ((size_t) win);
        for (int i = 0; i < win; ++i)
            window[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * M_PI * (double) i / (double) win);

        // COLA normalisation: sum of the squared window sampled at every
        // overlap offset (periodic with period hop; column 0 suffices).
        cola = 0.0;
        for (int k = 0; k < win; k += hop)
            cola += window[(size_t) k] * window[(size_t) k];

        reset();
    }

    void reset()
    {
        inRing.assign ((size_t) win, 0.0);
        accum.assign ((size_t) win, 0.0);
        re.assign ((size_t) win, 0.0);
        im.assign ((size_t) win, 0.0);
        mags.assign ((size_t) bins, 0.0);
        gains.assign ((size_t) bins, 1.0);
    }

    // Process exactly `hop` samples in place. maskFn(mags, gains, bins)
    // fills `gains` from the magnitude spectrum.
    template <typename MaskFn>
    void processHop (double* block, MaskFn&& maskFn)
    {
        const size_t w = (size_t) win, h = (size_t) hop;

        for (size_t i = 0; i < w - h; ++i)
            inRing[i] = inRing[i + h];
        for (size_t i = 0; i < h; ++i)
            inRing[w - h + i] = block[i];

        for (size_t i = 0; i < w; ++i)
        {
            re[i] = inRing[i] * window[i];
            im[i] = 0.0;
        }
        fft.forward (re, im);

        for (int k = 0; k < bins; ++k)
            mags[(size_t) k] = std::sqrt (re[(size_t) k] * re[(size_t) k]
                                        + im[(size_t) k] * im[(size_t) k]);
        maskFn (mags.data(), gains.data(), bins);

        // Apply gains with hermitian symmetry (real signal).
        re[0] *= gains[0];
        im[0] *= gains[0];
        re[(size_t) win / 2] *= gains[(size_t) bins - 1];
        im[(size_t) win / 2] *= gains[(size_t) bins - 1];
        for (int k = 1; k < win / 2; ++k)
        {
            const double g = gains[(size_t) k];
            re[(size_t) k] *= g;
            im[(size_t) k] *= g;
            re[(size_t) (win - k)] *= g;
            im[(size_t) (win - k)] *= g;
        }

        // Inverse FFT via conjugate trick: ifft(X) = conj(fft(conj(X))) / N.
        for (size_t i = 0; i < w; ++i)
            im[i] = -im[i];
        fft.forward (re, im);
        const double invN = 1.0 / (double) win;

        // Overlap-add the windowed synthesis frame.
        for (size_t i = 0; i < w; ++i)
            accum[i] += (re[i] * invN) * window[i];

        // Emit the first hop (fully summed by now) and slide.
        for (size_t i = 0; i < h; ++i)
            block[i] = accum[i] / cola;
        for (size_t i = 0; i < w - h; ++i)
            accum[i] = accum[i + h];
        for (size_t i = w - h; i < w; ++i)
            accum[i] = 0.0;
    }

    const int win, hop, bins;

private:
    Fft fft;
    double cola = 1.5;
    std::vector<double> window, inRing, accum, re, im, mags, gains;
};

} // namespace cleer
