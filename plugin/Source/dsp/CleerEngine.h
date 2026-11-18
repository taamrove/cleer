#pragma once
#include <memory>
#include <vector>
#include "Dereverb.h"
#include "Expander.h"
#include "FeedbackSuppressor.h"
#include "NoiseSuppressor.h"

namespace cleer
{

// The plugin's full chain: per-channel feedback suppression -> neural noise
// suppression (RNNoise) -> late-reverb suppression -> voice-adaptive
// downward expander. Feedback and expander are time-domain and
// latency-free; the noise stage adds ~10 ms and the dereverb ~21 ms when
// enabled (latencySamples() reports the sum of the active stages).
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
        dereverb = std::make_unique<DereverbStage>();
        dereverb->prepare (sampleRate, numChannels);
        expander = std::make_unique<Expander> (sampleRate);
        outAnalyzer = std::make_unique<SpectrumAnalyzer> (1024, 256);
        outHopBuf.assign (256, 0.0);
        outHopFill = 0;
        outPowerDB.assign (513, -120.0);

        scratch.assign ((size_t) numChannels, {});
        for (auto& v : scratch)
            v.assign ((size_t) maxBlockSize, 0.0);
        scratchPtrs.assign ((size_t) numChannels, nullptr);
    }

    bool feedbackEnabled = true;
    bool noiseEnabled = true;
    bool dereverbEnabled = false;
    bool gateEnabled = true;
    bool gateSmart = true;

    void setNoiseLive (bool v)
    {
        if (noise)
            noise->setLive (v);
    }

    bool noiseLive() const { return noise && noise->live; }

    void setNoiseOfflineExact (bool v)
    {
        if (noise)
            noise->offlineExact = v;
    }

    void setNoiseMix (double m)
    {
        if (noise)
            noise->mix = m;
    }

    void setDereverbParams (const DereverbStage::Params& p)
    {
        if (dereverb)
            dereverb->setParams (p);
    }

    // Host samples of delay added by the enabled lookahead stages.
    int latencySamples() const
    {
        int l = 0;
        if (noiseEnabled && noise)
            l += noise->latencySamples();
        if (dereverbEnabled && dereverb)
            l += dereverb->latencySamples();
        return l;
    }

    float voiceProbability() const { return noise ? noise->voiceProbability() : 0.0f; }
    long noiseUnderruns() const { return noise ? noise->underruns : 0; }

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
            // Full processing when the stage is on; VAD-only analysis when
            // it is off but the smart gate still wants the network's ears.
            const bool wantVad = gateEnabled && gateSmart;
            noise->enabled = noiseEnabled || wantVad;
            noise->outputEnabled = noiseEnabled;
            if (noise->enabled)
                noise->processBlock (scratchPtrs.data(), noiseEnabled ? numCh : 1, n);
        }

        if (dereverb)
        {
            dereverb->enabled = dereverbEnabled;
            if (dereverbEnabled)
                dereverb->processBlock (scratchPtrs.data(), numCh, n);
        }

        if (expander)
        {
            expander->enabled = gateEnabled;
            expander->smart = gateSmart && noise != nullptr;
            if (noise)
                expander->setVoiceProbability (noise->voiceProbability());
            expander->processBlock (scratchPtrs.data(), numCh, n);
        }

        // Output spectrum (channel 0, post-chain) for the UI overlay.
        if (outAnalyzer)
            for (int i = 0; i < n; ++i)
            {
                outHopBuf[(size_t) outHopFill] = scratchPtrs[0][i];
                if (++outHopFill == 256)
                {
                    outHopFill = 0;
                    const auto& mags = outAnalyzer->push (outHopBuf.data());
                    for (size_t k = 0; k < outPowerDB.size() && k < mags.size(); ++k)
                        outPowerDB[k] = 10.0 * std::log10 (mags[k] * mags[k] + 1e-12);
                }
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

    const std::vector<double>* uiSpectrumOutDB() const
    {
        return outAnalyzer ? &outPowerDB : nullptr;
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
        if (dereverb)
            dereverb->reset();
        if (expander)
            expander->reset();
        std::fill (outPowerDB.begin(), outPowerDB.end(), -120.0);
        outHopFill = 0;
    }

private:
    double sr = 48000.0;
    std::vector<std::unique_ptr<FeedbackSuppressor>> suppressors;
    std::unique_ptr<NoiseSuppressor> noise;
    std::unique_ptr<DereverbStage> dereverb;
    std::unique_ptr<Expander> expander;
    std::unique_ptr<SpectrumAnalyzer> outAnalyzer;
    std::vector<double> outHopBuf, outPowerDB;
    int outHopFill = 0;
    std::vector<std::vector<double>> scratch;
    std::vector<double*> scratchPtrs;
};

} // namespace cleer
