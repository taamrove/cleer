#pragma once
#include <memory>
#include <vector>
#include "Expander.h"
#include "FeedbackSuppressor.h"

namespace cleer
{

// The plugin's full chain: per-channel feedback suppression -> linked
// downward expander. All time-domain, zero added latency; the feedback
// detector's STFT is analysis-only. Equivalent of the app's VoicePipeline
// (macos/Cleer/DSP/VoicePipeline.swift) for N channels and float buffers.
class Engine
{
public:
    void prepare (double sampleRate, int numChannels, int maxBlockSize)
    {
        sr = sampleRate;
        suppressors.clear();
        for (int c = 0; c < numChannels; ++c)
            suppressors.push_back (std::make_unique<FeedbackSuppressor> (sampleRate));
        expander = std::make_unique<Expander> (sampleRate);

        scratch.assign ((size_t) numChannels, {});
        for (auto& v : scratch)
            v.assign ((size_t) maxBlockSize, 0.0);
        scratchPtrs.assign ((size_t) numChannels, nullptr);
    }

    bool feedbackEnabled = true;
    bool gateEnabled = true;

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
        if (expander)
            expander->reset();
    }

private:
    double sr = 48000.0;
    std::vector<std::unique_ptr<FeedbackSuppressor>> suppressors;
    std::unique_ptr<Expander> expander;
    std::vector<std::vector<double>> scratch;
    std::vector<double*> scratchPtrs;
};

} // namespace cleer
