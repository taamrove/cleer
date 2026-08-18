#pragma once
#include <memory>
#include <vector>
#include "Expander.h"
#include "FeedbackSuppressor.h"
#include "NoiseSuppressor.h"

namespace cleer
{

// The plugin's full chain: per-channel feedback suppression -> neural noise
// suppression (RNNoise) -> linked downward expander. Feedback and expander
// are time-domain and latency-free; the noise stage adds ~10 ms of frame
// latency when enabled (latencySamples() reports it, 0 when disabled).
class Engine
{
public:
    void prepare (double sampleRate, int numChannels, int maxBlockSize)
    {
        sr = sampleRate;
        suppressors.clear();
        for (int c = 0; c < numChannels; ++c)
            suppressors.push_back (std::make_unique<FeedbackSuppressor> (sampleRate));
        noise = std::make_unique<NoiseSuppressor>();
        noise->prepare (sampleRate, numChannels);
        expander = std::make_unique<Expander> (sampleRate);

        scratch.assign ((size_t) numChannels, {});
        for (auto& v : scratch)
            v.assign ((size_t) maxBlockSize, 0.0);
        scratchPtrs.assign ((size_t) numChannels, nullptr);
    }

    bool feedbackEnabled = true;
    bool noiseEnabled = true;
    bool gateEnabled = true;

    void setNoiseMix (double m)
    {
        if (noise)
            noise->mix = m;
    }

    // Host samples of delay added by the noise stage; 0 when it is off.
    int latencySamples() const
    {
        return (noiseEnabled && noise) ? noise->latencySamples() : 0;
    }

    float voiceProbability() const { return noise ? noise->voiceProbability() : 0.0f; }

    void setFeedbackParams (const FeedbackSuppressor::Params& p)
    {
        for (auto& s : suppressors)
            s->params = p;
    }

    void setExpanderParams (const Expander::Params& p)
    {
        if (expander)
            expander->setParams (p);
    }

    void processBlock (float* const* channels, int numCh, int n)
    {
        if (suppressors.empty() || n <= 0)
            return;
        numCh = std::min (numCh, (int) suppressors.size());
        if ((size_t) n > scratch[0].size())
            for (auto& v : scratch)
                v.resize ((size_t) n);   // hosts can exceed their declared max; not RT-clean but safe

        for (int c = 0; c < numCh; ++c)
        {
            double* d = scratch[(size_t) c].data();
            for (int i = 0; i < n; ++i)
                d[i] = (double) channels[c][i];
            scratchPtrs[(size_t) c] = d;

            suppressors[(size_t) c]->enabled = feedbackEnabled;
            suppressors[(size_t) c]->processBlock (d, n);
        }

        if (noise)
        {
            noise->enabled = noiseEnabled;
            if (noiseEnabled)
                noise->processBlock (scratchPtrs.data(), numCh, n);
        }

        if (expander)
        {
            expander->enabled = gateEnabled;
            expander->processBlock (scratchPtrs.data(), numCh, n);
        }

        for (int c = 0; c < numCh; ++c)
        {
            const double* d = scratchPtrs[(size_t) c];
            for (int i = 0; i < n; ++i)
                channels[c][i] = (float) d[i];
        }
    }

    // UI snapshots (audio thread; channel 0 drives the display).
    const std::vector<double>* uiSpectrumDB() const
    {
        return suppressors.empty() ? nullptr : &suppressors[0]->lastPowerDB();
    }

    int uiNotches (float* freqs, float* depths, int maxN) const
    {
        return suppressors.empty() ? 0 : suppressors[0]->snapshotNotches (freqs, depths, maxN);
    }

    int activeNotchCount() const
    {
        int c = 0;
        for (const auto& s : suppressors)
            c = std::max (c, s->activeNotchCount());
        return c;
    }

    double gainReductionDB() const
    {
        return expander ? expander->gainReductionDB() : 0.0;
    }

    void reset()
    {
        for (auto& s : suppressors)
            s->reset();
        if (noise)
            noise->reset();
        if (expander)
            expander->reset();
    }

private:
    double sr = 48000.0;
    std::vector<std::unique_ptr<FeedbackSuppressor>> suppressors;
    std::unique_ptr<NoiseSuppressor> noise;
    std::unique_ptr<Expander> expander;
    std::vector<std::vector<double>> scratch;
    std::vector<double*> scratchPtrs;
};

} // namespace cleer
