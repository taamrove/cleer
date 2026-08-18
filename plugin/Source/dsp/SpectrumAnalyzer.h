#pragma once
#include <vector>
#include "Fft.h"

namespace cleer
{

// Analysis-only windowed magnitude spectrum, fed `hop` samples at a time.
// Used by the feedback detector, which needs the spectrum of the *input*
// (before the notches) to decide where to place them. Port of
// macos/Cleer/DSP/SpectrumAnalyzer.swift; the 0.5x vDSP packing factor there
// cancels against vDSP's 2x scaling, so plain DFT magnitudes match it.
class SpectrumAnalyzer
{
public:
    SpectrumAnalyzer (int winIn, int hopIn)
        : win (winIn), hop (hopIn), bins (winIn / 2 + 1), fft (winIn)
    {
        window.resize ((size_t) win);
        for (int i = 0; i < win; ++i)
            window[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * M_PI * (double) i / (double) win);

        ring.assign ((size_t) win, 0.0);
        re.assign ((size_t) win, 0.0);
        im.assign ((size_t) win, 0.0);
        mags.assign ((size_t) bins, 0.0);
    }

    // Slide in `hop` samples and return |X| of the windowed frame.
    const std::vector<double>& push (const double* input)
    {
        const size_t w = (size_t) win, h = (size_t) hop;
        for (size_t i = 0; i < w - h; ++i)
            ring[i] = ring[i + h];
        for (size_t i = 0; i < h; ++i)
            ring[w - h + i] = input[i];

        for (size_t i = 0; i < w; ++i)
        {
            re[i] = ring[i] * window[i];
            im[i] = 0.0;
        }
        fft.forward (re, im);

        for (int k = 0; k < bins; ++k)
            mags[(size_t) k] = std::sqrt (re[(size_t) k] * re[(size_t) k]
                                        + im[(size_t) k] * im[(size_t) k]);
        return mags;
    }

    const int win, hop, bins;

private:
    Fft fft;
    std::vector<double> window, ring, re, im, mags;
};

} // namespace cleer
