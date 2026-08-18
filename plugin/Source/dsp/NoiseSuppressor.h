#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include <rnnoise.h>

namespace cleer
{

// Neural noise suppression around RNNoise (xiph.org) — the same GRU +
// band-gain architecture family the commercial De-Feedback plugin embeds.
//
// RNNoise is fixed at 48 kHz / 480-sample (10 ms) frames, so this wrapper
// owns the plumbing: a Catmull-Rom resampler pair for hosts not running at
// 48 kHz (at exactly 48 kHz the interpolator degenerates to a pass-through
// copy), frame accumulation, and a wet/dry `mix` blended per-frame against
// the *same* input frame, so the blend is phase-aligned by construction.
//
// The price of frame accumulation is latency (~10 ms + a few resampler
// samples). It is deterministic per sample rate; latencySamples() measures
// it empirically at prepare() by running an impulse through a scratch
// instance, so the reported value is exact rather than derived.
class NoiseSuppressor
{
public:
    static constexpr int kModelRate = 48000;
    static constexpr int kPrefill = 8;   // anti-underrun cushion, in host samples

    ~NoiseSuppressor() { releaseStates(); }

    void prepare (double sampleRateIn, int numChannels, bool measureLatencyNow = true)
    {
        sampleRate = sampleRateIn;
        frameSize = rnnoise_get_frame_size();   // 480

        releaseStates();
        channels.resize ((size_t) numChannels);
        for (auto& c : channels)
        {
            c.state = rnnoise_create (nullptr);
            c.frameOrig.assign ((size_t) frameSize, 0.0);
            c.frameScaled.assign ((size_t) frameSize, 0.0f);
            c.wet.assign ((size_t) frameSize, 0.0f);
        }
        reset();

        if (measureLatencyNow)
            measureLatency();
    }

    bool enabled = true;
    double mix = 1.0;          // 0 = dry (still delayed), 1 = fully denoised

    int latencySamples() const { return latency; }

    // Voice-activity probability from the last processed frame (channel 0).
    float voiceProbability() const { return lastVad; }

    void processBlock (double* const* ch, int numCh, int n)
    {
        if (! enabled || channels.empty())
            return;
        numCh = numCh < (int) channels.size() ? numCh : (int) channels.size();
        for (int c = 0; c < numCh; ++c)
            processChannel (channels[(size_t) c], ch[c], n, c == 0);
    }

    void reset()
    {
        for (auto& c : channels)
        {
            c.up = CubicStream (sampleRate / (double) kModelRate);
            c.down = CubicStream ((double) kModelRate / sampleRate);
            c.fill = 0;
            c.outFifo.assign (fifoCapacity(), 0.0);
            c.outHead = 0;
            // Small zero cushion: burst emission from the frame processor
            // briefly lags the per-sample pops by a couple of samples after
            // the first frame; without the cushion those pops underrun and
            // inject mid-stream zeros, making the delay time-varying.
            c.outCount = kPrefill;
            if (c.state != nullptr)
                rnnoise_init (c.state, nullptr);
        }
        lastVad = 0.0f;
    }

private:
    // Output-driven Catmull-Rom resampler: push source samples, read all
    // ready destination samples. At step == 1 it reproduces its input
    // exactly (one-sample delay), so 48 kHz hosts pay no quality cost.
    struct CubicStream
    {
        explicit CubicStream (double stepIn = 1.0) : step (stepIn) {}

        void push (double s)
        {
            win[0] = win[1]; win[1] = win[2]; win[2] = win[3]; win[3] = s;
            ++base;
        }

        bool ready() const { return (long) std::floor (pos) + 2 <= base; }

        double read()
        {
            const double t = pos - std::floor (pos);
            const double s0 = win[0], s1 = win[1], s2 = win[2], s3 = win[3];
            const double out = 0.5 * (2.0 * s1
                                      + (-s0 + s2) * t
                                      + (2.0 * s0 - 5.0 * s1 + 4.0 * s2 - s3) * t * t
                                      + (-s0 + 3.0 * s1 - 3.0 * s2 + s3) * t * t * t);
            pos += step;
            return out;
        }

        double win[4] = { 0, 0, 0, 0 };
        long base = -1;
        double pos = 0.0;
        double step = 1.0;
    };

    struct Channel
    {
        DenoiseState* state = nullptr;
        CubicStream up, down;
        std::vector<double> frameOrig;
        std::vector<float> frameScaled;
        std::vector<float> wet;
        int fill = 0;
        std::vector<double> outFifo;   // host-rate ring, pre-sized
        int outHead = 0, outCount = 0;
    };

    size_t fifoCapacity() const
    {
        // Worst case: a whole model frame lands at once, in host samples,
        // plus generous slack. Sized once; never grows on the audio thread.
        return (size_t) (4.0 * frameSize * sampleRate / (double) kModelRate) + 1024;
    }

    void fifoPush (Channel& c, double v)
    {
        const size_t cap = c.outFifo.size();
        c.outFifo[(size_t) ((c.outHead + c.outCount) % (int) cap)] = v;
        if (c.outCount < (int) cap)
            ++c.outCount;   // overflow drops oldest implicitly; sized to never happen
    }

    double fifoPop (Channel& c)
    {
        if (c.outCount == 0)
            return 0.0;     // warm-up: zeros account for the reported latency
        const double v = c.outFifo[(size_t) c.outHead];
        c.outHead = (c.outHead + 1) % (int) c.outFifo.size();
        --c.outCount;
        return v;
    }

    void processChannel (Channel& c, double* x, int n, bool trackVad)
    {
        for (int i = 0; i < n; ++i)
        {
            c.up.push (x[i]);
            while (c.up.ready())
            {
                const double s48 = c.up.read();
                c.frameOrig[(size_t) c.fill] = s48;
                c.frameScaled[(size_t) c.fill] = (float) (s48 * 32768.0);
                if (++c.fill == frameSize)
                {
                    c.fill = 0;
                    const float vad = rnnoise_process_frame (c.state, c.wet.data(), c.frameScaled.data());
                    if (trackVad)
                        lastVad = vad;
                    for (int k = 0; k < frameSize; ++k)
                    {
                        const double wet = (double) c.wet[(size_t) k] / 32768.0;
                        const double blended = mix * wet + (1.0 - mix) * c.frameOrig[(size_t) k];
                        c.down.push (blended);
                        while (c.down.ready())
                            fifoPush (c, c.down.read());
                    }
                }
            }
            x[i] = fifoPop (c);
        }
    }

    void measureLatency()
    {
        // Feed an impulse through a scratch instance configured identically
        // (mix 0 keeps the path linear) and find where it comes out. The
        // impulse goes in well past the first few frames so the measured
        // value is the steady-state delay, not the warm-up one.
        NoiseSuppressor probe;
        probe.prepare (sampleRate, 1, false);
        probe.mix = 0.0;

        const int impulseAt = (int) (4.0 * frameSize * sampleRate / (double) kModelRate) + 63;
        const int cap = impulseAt + 8 * frameSize;
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
            {
                const double a = std::abs (block[(size_t) i]);
                if (a > best)
                {
                    best = a;
                    bestIdx = fed + i;
                }
            }
            fed += (int) block.size();
            if (bestIdx > impulseAt && fed > bestIdx + frameSize)
                break;   // impulse found and settled
        }
        latency = bestIdx > impulseAt ? bestIdx - impulseAt : 0;
    }

    void releaseStates()
    {
        for (auto& c : channels)
            if (c.state != nullptr)
                rnnoise_destroy (c.state);
        channels.clear();
    }

    double sampleRate = 48000.0;
    int frameSize = 480;
    int latency = 0;
    float lastVad = 0.0f;
    std::vector<Channel> channels;
};

} // namespace cleer
