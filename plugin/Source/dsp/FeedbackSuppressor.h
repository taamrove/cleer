#pragma once
#include <memory>
#include <vector>
#include "Biquad.h"
#include "SpectrumAnalyzer.h"

namespace cleer
{

// Adaptive feedback (howl) suppression. Direct port of the validated Swift
// implementation (macos/Cleer/DSP/FeedbackSuppressor.swift).
//
// Detection is frequency-domain; actuation is a pool of time-domain biquad
// notches, so the audio path adds zero latency. A bin becomes a candidate
// only when it is (a) in-band, (b) a tonal peak prominent above the local
// broadband floor, and (c) GROWING frame-over-frame — the exponential-growth
// signature that separates real feedback from a held vocal note. Only
// sustained-and-still-growing bins get a notch.
//
// One difference from the Swift original: this version streams. The host
// hands it arbitrary block sizes; input samples accumulate into an internal
// hop buffer and the notch set updates each time a hop completes, so notch
// changes take effect one hop (~5 ms) later than in the Swift code, which
// analysed a hop before filtering it. For a detector that requires 12
// consecutive frames of growth this is immaterial, and it keeps added
// latency at zero for any host buffer size.
class FeedbackSuppressor
{
public:
    struct Params
    {
        double fLo = 300.0, fHi = 9000.0;  // most vocal fundamentals sit below 300
        double prominenceDB = 18.0;        // must be this far above the local floor
        double holdDB = 9.0;
        double persistFrames = 12.0;       // ~64 ms at hop=256 / 48k
        double growthDBperFrame = 1.5;     // bin must rise this fast while persisting
        double maxDepthDB = 24.0;
        double q = 28.0;
        double attackDB = 18.0;            // dB of extra depth per frame when hot
        double releaseDB = 2.5;            // dB of recovery per frame when cold
        int    maxNotches = 6;
        int    floorWidthBins = 31;        // width of the broadband floor estimator
    };

    FeedbackSuppressor (double sampleRateIn, int winIn = 1024, int hopIn = 256)
        : sampleRate (sampleRateIn), win (winIn), hop (hopIn),
          analyzer (winIn, hopIn), bins (winIn / 2 + 1)
    {
        persist.assign ((size_t) bins, 0.0);
        prevPowerDB.assign ((size_t) bins, -120.0);
        powerDB.assign ((size_t) bins, -120.0);
        floorDB.assign ((size_t) bins, 0.0);
        prefix.assign ((size_t) bins + 1, 0.0);
        hopBuf.assign ((size_t) hop, 0.0);
        notches.reserve (32);
    }

    Params params;
    bool enabled = true;

    // Process an arbitrary-length block in place.
    void processBlock (double* x, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            hopBuf[(size_t) hopFill] = x[i];   // analyse the input, pre-notch
            if (++hopFill == hop)
            {
                hopFill = 0;
                const auto& mags = analyzer.push (hopBuf.data());
                // Power spectrum updates even when disabled so a UI fed from
                // lastPowerDB() stays live while the suppressor is bypassed.
                for (int k = 0; k < bins; ++k)
                {
                    const double p = mags[(size_t) k] * mags[(size_t) k] + 1e-12;
                    powerDB[(size_t) k] = 10.0 * std::log10 (p);
                }
                if (enabled)
                    updateNotches (mags);
            }

            if (enabled && ! notches.empty())
            {
                double s = x[i];
                for (auto& nt : notches)
                    s = nt.biquad.process (s);
                x[i] = s;
            }
        }
    }

    // UI snapshot helpers — called from the audio thread only; the caller
    // copies the results into whatever thread-safe storage its UI reads.
    const std::vector<double>& lastPowerDB() const { return powerDB; }

    int snapshotNotches (float* freqs, float* depths, int maxN) const
    {
        int n = 0;
        for (const auto& nt : notches)
        {
            if (n >= maxN)
                break;
            freqs[n] = (float) nt.freq;
            depths[n] = (float) nt.depthDB;
            ++n;
        }
        return n;
    }

    int activeNotchCount() const
    {
        int c = 0;
        for (const auto& n : notches)
            if (n.depthDB < -1.0)
                ++c;
        return c;
    }

    void reset()
    {
        notches.clear();
        std::fill (persist.begin(), persist.end(), 0.0);
        std::fill (prevPowerDB.begin(), prevPowerDB.end(), -120.0);
        hopFill = 0;
        std::fill (hopBuf.begin(), hopBuf.end(), 0.0);
    }

private:
    struct Notch
    {
        Biquad biquad;
        double freq;
        double depthDB = 0.0;
        int idle = 0;
        Notch (double sr, double f) : biquad (sr), freq (f) {}
    };

    double binToFreq (int k) const { return (double) k * sampleRate / (double) win; }
    int freqToBin (double f) const { return (int) std::lround (f * (double) win / sampleRate); }

    void updateNotches (const std::vector<double>& mags)
    {
        // 1. powerDB was just refreshed by processBlock at the hop boundary.
        // 2. Broadband floor via O(n) sliding mean (prefix sums).
        slidingMean (powerDB, floorDB, params.floorWidthBins / 2);

        // 3. Per-bin assessment: in-band AND tonal+prominent AND growing.
        const int nyq = bins - 1;
        for (int k = 1; k < nyq; ++k)
        {
            const double f = binToFreq (k);
            const bool inBand = f >= params.fLo && f <= params.fHi;
            const bool isPeak = powerDB[(size_t) k] > powerDB[(size_t) k - 1]
                             && powerDB[(size_t) k] > powerDB[(size_t) k + 1];
            const bool prominent = (powerDB[(size_t) k] - floorDB[(size_t) k]) > params.prominenceDB;
            const bool growing = (powerDB[(size_t) k] - prevPowerDB[(size_t) k]) > params.growthDBperFrame;

            if (inBand && isPeak && prominent && growing)
                persist[(size_t) k] += 1.0;
            else
                persist[(size_t) k] *= 0.5;   // one quiet frame breaks persistence

            if (persist[(size_t) k] >= params.persistFrames)
            {
                bool already = false;
                for (const auto& n : notches)
                    if (std::abs (freqToBin (n.freq) - k) <= 1) { already = true; break; }

                if (! already && (int) notches.size() < params.maxNotches)
                    notches.emplace_back (sampleRate, interpFreq (mags, k));
            }
        }

        for (int k = 0; k < bins; ++k)
            prevPowerDB[(size_t) k] = powerDB[(size_t) k];

        // 4. Drive depths; release and reap dead notches.
        size_t out = 0;
        for (size_t i = 0; i < notches.size(); ++i)
        {
            Notch& n = notches[i];
            const int b = freqToBin (n.freq);
            const bool hot = b >= 0 && b < bins
                          && (powerDB[(size_t) b] - floorDB[(size_t) b]) > params.holdDB;
            const double target = hot ? -params.maxDepthDB : 0.0;
            if (n.depthDB > target)
                n.depthDB = std::max (target, n.depthDB - params.attackDB);
            else
                n.depthDB = std::min (target, n.depthDB + params.releaseDB);
            n.biquad.setPeaking (n.freq, params.q, n.depthDB);
            n.idle = n.depthDB >= -1e-6 ? n.idle + 1 : 0;
            if (n.idle < 20)
            {
                if (out != i)
                    notches[out] = std::move (n);
                ++out;
            }
        }
        notches.erase (notches.begin() + (long) out, notches.end());
    }

    double interpFreq (const std::vector<double>& mags, int k) const
    {
        if (k <= 0 || k >= bins - 1)
            return binToFreq (k);
        const double a = std::log (mags[(size_t) k - 1] * mags[(size_t) k - 1] + 1e-12);
        const double b = std::log (mags[(size_t) k] * mags[(size_t) k] + 1e-12);
        const double c = std::log (mags[(size_t) k + 1] * mags[(size_t) k + 1] + 1e-12);
        const double denom = a - 2.0 * b + c;
        double delta = denom != 0.0 ? 0.5 * (a - c) / denom : 0.0;
        delta = std::min (0.5, std::max (-0.5, delta));
        return ((double) k + delta) * sampleRate / (double) win;
    }

    void slidingMean (const std::vector<double>& xin, std::vector<double>& out, int radius)
    {
        const int n = (int) xin.size();
        if (n == 0)
            return;
        const int r = std::max (1, radius);
        for (int i = 0; i < n; ++i)
            prefix[(size_t) i + 1] = prefix[(size_t) i] + xin[(size_t) i];
        for (int i = 0; i < n; ++i)
        {
            const int lo = std::max (0, i - r);
            const int hi = std::min (n - 1, i + r);
            out[(size_t) i] = (prefix[(size_t) hi + 1] - prefix[(size_t) lo]) / (double) (hi - lo + 1);
        }
    }

    const double sampleRate;
    const int win, hop;
    SpectrumAnalyzer analyzer;
    const int bins;

    std::vector<double> persist, prevPowerDB, powerDB, floorDB, prefix;
    std::vector<double> hopBuf;
    int hopFill = 0;
    std::vector<Notch> notches;
};

} // namespace cleer
