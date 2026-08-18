#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include "OlaStft.h"

namespace cleer
{

// Late-reverberation suppression by spectral gain — direct port of the
// validated reference/defeedback/dereverb.py (Lebart-style late-tail
// spectral subtraction). The tail at the current frame is modelled as a
// delayed, decayed copy of the recent per-bin power; it is subtracted with
// a floored Wiener-like gain.
//
// Runs on an internal OLA STFT (1024/256 at the host rate), so enabling it
// costs win-hop + hop = 1024 samples of latency (~21 ms at 48 kHz). Like
// the noise stage the exact figure is measured empirically at prepare().
class DereverbStage
{
public:
    struct Params
    {
        double rt60 = 0.6;          // assumed room reverb time, seconds
        double strength = 0.6;      // 0 = off, 1 = full late-tail subtraction
        double delayMs = 40.0;      // early (kept) / late (removed) boundary
        double gainFloorDB = -15.0; // per-bin attenuation cap against artefacts
        double smooth = 0.6;        // tail-estimate power smoothing
    };

    void prepare (double sampleRateIn, int numChannels, bool measureLatencyNow = true)
    {
        sampleRate = sampleRateIn;
        channels.clear();
        channels.resize ((size_t) numChannels);
        recompute();
        reset();
        if (measureLatencyNow)
            measureLatency();
    }

    void setParams (const Params& p)
    {
        params = p;
        recompute();
    }

    const Params& getParams() const { return params; }
    bool enabled = false;
    int latencySamples() const { return latency; }

    void processBlock (double* const* ch, int numCh, int n)
    {
        if (! enabled || channels.empty())
            return;
        numCh = numCh < (int) channels.size() ? numCh : (int) channels.size();
        for (int c = 0; c < numCh; ++c)
        {
            auto& st = channels[(size_t) c];
            double* x = ch[c];
            for (int i = 0; i < n; ++i)
            {
                st.hopBuf[(size_t) st.fill] = x[i];
                if (++st.fill == kHop)
                {
                    st.fill = 0;
                    st.stft.processHop (st.hopBuf.data(),
                        [&st, this] (const double* mags, double* gains, int bins)
                        { computeMask (st, mags, gains, bins); });
                    for (int k = 0; k < kHop; ++k)
                        fifoPush (st, st.hopBuf[(size_t) k]);
                }
                x[i] = fifoPop (st);
            }
        }
    }

    void reset()
    {
        const int bins = kWin / 2 + 1;
        for (auto& st : channels)
        {
            st.stft.reset();
            st.hopBuf.assign ((size_t) kHop, 0.0);
            st.fill = 0;
            st.powerHist.assign ((size_t) (delayFrames + 1) * (size_t) bins, 0.0);
            st.histCount = 0;
            st.histHead = 0;
            st.tail.assign ((size_t) bins, 0.0);
            st.outFifo.assign ((size_t) (4 * kHop), 0.0);
            st.outHead = 0;
            st.outCount = 0;
        }
    }

private:
    static constexpr int kWin = 1024, kHop = 256;

    struct Channel
    {
        OlaStft stft { kWin, kHop };
        std::vector<double> hopBuf;
        int fill = 0;
        std::vector<double> powerHist;   // ring of (delayFrames+1) power frames
        int histCount = 0, histHead = 0;
        std::vector<double> tail;
        std::vector<double> outFifo;
        int outHead = 0, outCount = 0;
    };

    void recompute()
    {
        const double frameS = kHop / sampleRate;
        delayFrames = (int) std::lround ((params.delayMs / 1000.0) / frameS);
        if (delayFrames < 1)
            delayFrames = 1;
        const double decayPerFrame = std::pow (10.0, -3.0 * frameS / params.rt60);
        decay = std::pow (decayPerFrame, (double) delayFrames);
        gainFloor = std::pow (10.0, params.gainFloorDB / 20.0);

        const int bins = kWin / 2 + 1;
        for (auto& st : channels)
            if (st.powerHist.size() != (size_t) (delayFrames + 1) * (size_t) bins)
            {
                st.powerHist.assign ((size_t) (delayFrames + 1) * (size_t) bins, 0.0);
                st.histCount = 0;
                st.histHead = 0;
            }
    }

    void computeMask (Channel& st, const double* mags, double* gains, int bins)
    {
        // Push current power frame into the ring.
        double* cur = st.powerHist.data() + (size_t) st.histHead * (size_t) bins;
        for (int k = 0; k < bins; ++k)
            cur[k] = mags[k] * mags[k];
        st.histHead = (st.histHead + 1) % (delayFrames + 1);
        if (st.histCount < delayFrames + 1)
            ++st.histCount;

        if (st.histCount > delayFrames)
        {
            // Oldest frame in the ring is `delayFrames` back.
            const double* delayed = st.powerHist.data() + (size_t) st.histHead * (size_t) bins;
            for (int k = 0; k < bins; ++k)
                st.tail[(size_t) k] = params.smooth * st.tail[(size_t) k]
                                    + (1.0 - params.smooth) * (decay * delayed[k]);
        }

        for (int k = 0; k < bins; ++k)
        {
            const double power = mags[k] * mags[k];
            const double late = params.strength * st.tail[(size_t) k];
            double g = power - late;
            g = g > 0.0 ? g / (power + 1e-12) : 0.0;
            gains[k] = g > gainFloor ? g : gainFloor;
        }
    }

    void fifoPush (Channel& st, double v)
    {
        const size_t cap = st.outFifo.size();
        st.outFifo[(size_t) ((st.outHead + st.outCount) % (int) cap)] = v;
        if (st.outCount < (int) cap)
            ++st.outCount;
    }

    double fifoPop (Channel& st)
    {
        if (st.outCount == 0)
            return 0.0;
        const double v = st.outFifo[(size_t) st.outHead];
        st.outHead = (st.outHead + 1) % (int) st.outFifo.size();
        --st.outCount;
        return v;
    }

    void measureLatency()
    {
        DereverbStage probe;
        probe.params = params;
        probe.prepare (sampleRate, 1, false);
        probe.enabled = true;

        const int impulseAt = 6 * kWin + 63;
        const int cap = impulseAt + 8 * kWin;
        std::vector<double> block (256, 0.0);
        int fed = 0, bestIdx = -1;
        double best = 0.0;
        double* p = block.data();
        while (fed < cap)
        {
            std::memset (block.data(), 0, block.size() * sizeof (double));
            if (impulseAt >= fed && impulseAt < fed + (int) block.size())
                block[(size_t) (impulseAt - fed)] = 1.0;
            probe.processBlock (&p, 1, (int) block.size());
            for (int i = 0; i < (int) block.size(); ++i)
                if (std::abs (block[(size_t) i]) > best)
                {
                    best = std::abs (block[(size_t) i]);
                    bestIdx = fed + i;
                }
            fed += (int) block.size();
            if (bestIdx > impulseAt && fed > bestIdx + 2 * kWin)
                break;
        }
        latency = bestIdx > impulseAt ? bestIdx - impulseAt : 0;
    }

    double sampleRate = 48000.0;
    Params params;
    int delayFrames = 8;
    double decay = 0.5;
    double gainFloor = 0.18;
    int latency = 0;
    std::vector<Channel> channels;
};

} // namespace cleer
