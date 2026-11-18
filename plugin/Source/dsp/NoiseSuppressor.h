#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <thread>
#include <rnnoise.h>
#include "Fft.h"

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
// Latency: one frame of accumulation plus the model's own two frames of
// conv lookahead (the 2024 RNNoise peeks ahead for quality), ~30 ms total
// at 48 kHz. The dry blend branch is delayed to match, so AMOUNT never
// comb-filters. latencySamples() measures the exact figure empirically at
// prepare() by running an impulse through a scratch instance.
//
// LIVE mode (live = true): the audio no longer goes *through* the model.
// RNNoise still runs on the worker, but only as analysis: each frame its
// 481-bin gain curve (patched out of the vendored denoise.c) is turned into
// a minimum-phase FIR (cepstral method, kLive.taps taps) that is convolved,
// in the time domain, with the undelayed input -> zero latency. The gains
// are ~40 ms stale, so they are smoothed (fast open, slower close), capped
// by AMOUNT, and a zero-latency onset detector bypasses the filter for the
// first ~45 ms of a word that starts out of silence. The audio thread never
// waits on the worker in this mode (a busy worker just skips a frame).
class NoiseSuppressor
{
public:
    static constexpr int kModelRate = 48000;
    static constexpr int kPrefill = 32;  // anti-underrun cushion, in host samples
    static constexpr int kLiveTaps = 256;     // min-phase FIR length
    static constexpr int kRing = 96, kRingStep = 64;   // 128 ms of level history at 48k
    static constexpr int kLiveFft = 512;      // cepstral design size (>= 2x taps)

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
            c.origPrev1.assign ((size_t) frameSize, 0.0);
            c.origPrev2.assign ((size_t) frameSize, 0.0);
            c.origPrev3.assign ((size_t) frameSize, 0.0);
            c.worker = std::make_unique<Worker> (c.state);
            liveFft = std::make_unique<Fft> (kLiveFft);
            c.frameScaled.assign ((size_t) frameSize, 0.0f);
            c.wet.assign ((size_t) frameSize, 0.0f);
        }
        reset();

        if (measureLatencyNow)
            measureLatency();
    }

    bool enabled = true;
    double mix = 1.0;          // 0 = dry (still delayed), 1 = fully denoised

    // Analysis-only mode: run the network for its VAD output but leave the
    // audio untouched (no output substitution, no FIFO, no latency). Used
    // by the smart gate when the noise-removal stage itself is bypassed.
    bool outputEnabled = true;

    // Live mode: zero-latency min-phase filter steered by the model.
    // Switching resets the stage (call from the audio thread or while idle).
    bool live = false;
    void setLive (bool v)
    {
        if (v != live)
        {
            live = v;
            reset();
        }
    }

    // Offline renders / tests only: wait for the worker so faster-than-real-time
    // processing sees every model frame (what a real-time run does). Never
    // set this in a plug-in's audio callback.
    bool offlineExact = false;

    int latencySamples() const { return live ? 0 : latency; }

    // Diagnostic: pops that found the FIFO empty. A bounded number at
    // startup is the reported latency; growth during steady state is a bug.
    long underruns = 0;

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
            std::fill (c.origPrev1.begin(), c.origPrev1.end(), 0.0);
            std::fill (c.origPrev2.begin(), c.origPrev2.end(), 0.0);
            std::fill (c.origPrev3.begin(), c.origPrev3.end(), 0.0);
            c.outFifo.assign (fifoCapacity(), 0.0);
            c.outHead = 0;
            // Small zero cushion: burst emission from the frame processor
            // briefly lags the per-sample pops by a couple of samples after
            // the first frame; without the cushion those pops underrun and
            // inject mid-stream zeros, making the delay time-varying.
            c.outCount = kPrefill;
            c.resetLive (sampleRate);
            if (c.worker)
            {
                c.worker->waitIdleOrDone();
                c.worker->st.store (0);
            }
            if (c.state != nullptr)
                rnnoise_init (c.state, nullptr);
        }
        lastVad = 0.0f;
    }

private:
    // Output-driven Catmull-Rom resampler: push source samples, read all
    // ready destination samples. At step == 1 it reproduces its input
    // exactly (one-sample delay), so 48 kHz hosts pay no quality cost.
    //
    // The phase is COMPUTED (reads * step), never accumulated. `pos += step`
    // rounds once per sample, and at 44.1 kHz (step not binary-representable)
    // hours of running drifted the phase tens of samples, walking the output
    // FIFO occupancy into permanent underrun — heard as a plugin that was
    // fine all soundcheck and glitched every frame after sitting idle for a
    // couple of hours. One multiply per read has bounded error forever.
    struct CubicStream
    {
        explicit CubicStream (double stepIn = 1.0) : step (stepIn) {}

        void push (double s)
        {
            win[0] = win[1]; win[1] = win[2]; win[2] = win[3]; win[3] = s;
            ++base;
        }

        double pos() const { return (double) reads * step; }
        bool ready() const { return (long) std::floor (pos()) + 2 <= base; }

        double read()
        {
            const double p = pos();
            const double t = p - std::floor (p);
            const double s0 = win[0], s1 = win[1], s2 = win[2], s3 = win[3];
            const double out = 0.5 * (2.0 * s1
                                      + (-s0 + s2) * t
                                      + (2.0 * s0 - 5.0 * s1 + 4.0 * s2 - s3) * t * t
                                      + (-s0 + 3.0 * s1 - 3.0 * s2 + s3) * t * t * t);
            ++reads;
            return out;
        }

        double win[4] = { 0, 0, 0, 0 };
        long base = -1;
        long reads = 0;
        double step = 1.0;
    };

    // One RNNoise frame costs ~1 ms (big 2024 model) — a whole 64-sample
    // callback's budget at 48 kHz. Running it inline made every ~7th
    // callback overrun (constant "digital" glitching at small buffers,
    // clean at >=128). The audio thread only hands frames to this worker
    // and picks the result up one frame later (+10 ms latency). Results are
    // collected in strict frame order, so output is deterministic and
    // independent of thread timing.
    struct Worker
    {
        Worker (DenoiseState* s) : state (s), th ([this] { run(); }) {}
        ~Worker() { stop.store (true); th.join(); }

        void run()
        {
            while (! stop.load())
            {
                if (st.load (std::memory_order_acquire) == 1)
                {
                    vad = rnnoise_process_frame (state, out, in);
                    rnnoise_get_gains (state, gf);
                    st.store (2, std::memory_order_release);
                }
                else
                    std::this_thread::sleep_for (std::chrono::microseconds (200));
            }
        }

        void waitIdleOrDone() const
        {
            while (st.load (std::memory_order_acquire) == 1)
                std::this_thread::yield();
        }

        DenoiseState* state;
        float in[480], out[480], gf[481];
        float vad = 0.0f;
        std::atomic<int> st { 0 };   // 0 idle, 1 pending, 2 result ready
        std::atomic<bool> stop { false };
        std::thread th;
    };

    struct Channel
    {
        DenoiseState* state = nullptr;
        std::unique_ptr<Worker> worker;
        std::vector<double> origPrev3;
        CubicStream up, down;
        std::vector<double> frameOrig, origPrev1, origPrev2;
        std::vector<float> frameScaled;
        std::vector<float> wet;
        int fill = 0;
        std::vector<double> outFifo;   // host-rate ring, pre-sized
        int outHead = 0, outCount = 0;

        // ---- Live mode state ----
        std::vector<double> hist, hPrev, hCur, gs;   // input history (2x), taps, smoothed gains
        int hw = 0, ramp = 0, rampLen = 480;
        double eFast = 0, eMed = 0, onset = 0, target = 0, lp = 0, lpA = 0.026;
        int hold = 0, cool = 0, tick = 0, ringPos = 0;
        double ring[kRing];   // eMed sampled every kRingStep samples

        void resetLive (double sr)
        {
            hist.assign (2 * kLiveTaps, 0.0);
            hPrev.assign (kLiveTaps, 0.0);
            hCur.assign (kLiveTaps, 0.0);
            hPrev[0] = hCur[0] = 1.0;           // identity filter until the model speaks
            gs.assign (481, 1.0);
            hw = 0;
            rampLen = std::max (1, (int) std::lround (480.0 * sr / kModelRate));
            ramp = rampLen;
            eFast = eMed = onset = lp = 0.0;
            lpA = 1.0 - std::exp (-2.0 * M_PI * 250.0 / sr);
            hold = cool = tick = ringPos = 0;
            for (auto& r : ring) r = 1.0;
        }
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
        {
            ++underruns;    // steady-state underruns are a bug; warm-up zeros are counted too but bounded
            return 0.0;
        }
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
                    // Collect the previous frame's result (normally long
                    // done), then hand this frame to the worker.
                    Worker& w = *c.worker;
                    if (live)
                    {
                        if (offlineExact)
                            w.waitIdleOrDone();
                        // Never wait (real time): take a finished result if there is one,
                        // hand over the new frame only if the worker is idle.
                        if (w.st.load (std::memory_order_acquire) == 2)
                        {
                            if (trackVad)
                                lastVad = w.vad;
                            updateLiveFilter (c, w.gf);
                            w.st.store (0, std::memory_order_release);
                        }
                        if (w.st.load (std::memory_order_acquire) == 0)
                        {
                            std::memcpy (w.in, c.frameScaled.data(), sizeof (float) * (size_t) frameSize);
                            w.st.store (1, std::memory_order_release);
                        }
                        continue;
                    }
                    const bool haveWet = w.st.load (std::memory_order_acquire) != 0;
                    if (haveWet)
                    {
                        w.waitIdleOrDone();
                        std::memcpy (c.wet.data(), w.out, sizeof (float) * (size_t) frameSize);
                        if (trackVad)
                            lastVad = w.vad;
                    }
                    std::memcpy (w.in, c.frameScaled.data(), sizeof (float) * (size_t) frameSize);
                    w.st.store (1, std::memory_order_release);
                    if (haveWet && outputEnabled)
                        for (int k = 0; k < frameSize; ++k)
                        {
                            // The 2024 RNNoise model has two frames of conv
                            // lookahead (output lags input by 2*frameSize,
                            // measured), plus our one frame of worker
                            // hand-off: blend against the input from three
                            // frames ago so wet and dry stay sample-aligned
                            // (misaligned sums comb-filter at partial AMOUNT).
                            const double wet = (double) c.wet[(size_t) k] / 32768.0;
                            const double blended = mix * wet + (1.0 - mix) * c.origPrev3[(size_t) k];
                            c.down.push (blended);
                            while (c.down.ready())
                                fifoPush (c, c.down.read());
                        }
                    std::swap (c.origPrev3, c.origPrev2);
                    std::swap (c.origPrev2, c.origPrev1);
                    std::swap (c.origPrev1, c.frameOrig);
                }
            }
            if (outputEnabled)
                x[i] = live ? liveSample (c, x[i]) : fifoPop (c);
        }
    }


    // ---- Live mode ------------------------------------------------------
    // Tuned by test (see testLive*): gains open instantly, close at 0.5/frame;
    // graded onset bypass opens at +14..26 dB over the recent minimum, holds 45 ms.
    void updateLiveFilter (Channel& c, const float* gf)
    {
        for (int k = 0; k < 481; ++k)
        {
            const double g = (1.0 - mix) + mix * (double) gf[k];   // AMOUNT = max attenuation
            c.gs[(size_t) k] += (g >= c.gs[(size_t) k] ? 1.0 : 0.5) * (g - c.gs[(size_t) k]);
        }
        c.hPrev = c.hCur;
        designMinPhase (c.gs, c.hCur);
        c.ramp = 0;
    }

    // Gain curve (481 bins, 50 Hz apart) -> minimum-phase FIR via the
    // homomorphic (cepstral) method: log|H| -> cepstrum -> fold causal -> exp.
    void designMinPhase (const std::vector<double>& g, std::vector<double>& h)
    {
        const int N = kLiveFft;
        auto& re = dRe; auto& im = dIm;
        re.assign ((size_t) N, 0.0); im.assign ((size_t) N, 0.0);
        for (int k = 0; k <= N / 2; ++k)
        {
            const double pos = std::min (480.0, (double) k * sampleRate / N / 50.0);
            const int i0 = (int) pos;
            const double fr = pos - i0;
            const double v = g[(size_t) i0] * (1.0 - fr) + g[(size_t) std::min (480, i0 + 1)] * fr;
            re[(size_t) k] = std::log (std::max (v, 1e-4));
            if (k > 0 && k < N / 2)
                re[(size_t) (N - k)] = re[(size_t) k];
        }
        liveFft->forward (re, im);                    // real-even in -> cepstrum * N
        for (int n = 0; n < N; ++n)
        {
            const double cn = re[(size_t) n] / N;
            re[(size_t) n] = n == 0 || n == N / 2 ? cn : (n < N / 2 ? 2.0 * cn : 0.0);
            im[(size_t) n] = 0.0;
        }
        liveFft->forward (re, im);                    // min-phase log spectrum
        for (int k = 0; k < N; ++k)
        {
            const double m = std::exp (re[(size_t) k]), ph = im[(size_t) k];
            re[(size_t) k] = m * std::cos (ph);
            im[(size_t) k] = -m * std::sin (ph);      // conjugate for the inverse
        }
        liveFft->forward (re, im);
        for (int n = 0; n < kLiveTaps; ++n)
        {
            double w = 1.0;
            const int tail = 64;
            if (n >= kLiveTaps - tail)
                w = 0.5 * (1.0 + std::cos (M_PI * (n - (kLiveTaps - tail)) / tail));
            h[(size_t) n] = re[(size_t) n] / N * w;
        }
    }

    double liveSample (Channel& c, double x)
    {
        const int L = kLiveTaps;
        c.hist[(size_t) c.hw] = x;
        c.hist[(size_t) (c.hw + L)] = x;
        const double* xp = c.hist.data() + c.hw + L;   // xp[-k] = x[n-k]
        const double* a = c.hPrev.data();
        const double* b = c.hCur.data();
        double ya = 0.0, yb = 0.0;
        for (int k = 0; k < L; ++k)
        {
            ya += a[k] * xp[-k];
            yb += b[k] * xp[-k];
        }
        c.hw = (c.hw + 1) % L;
        const double t = c.ramp >= c.rampLen ? 1.0 : (double) c.ramp++ / c.rampLen;
        double y = ya + t * (yb - ya);

        // Onset protection: the gains are ~40 ms stale, so when the input
        // leaps above the tracked noise floor, bypass the filter until the
        // model has caught up.
        c.lp += c.lpA * (x - c.lp);                 // high-pass: rumble/hum would
        const double hp = x - c.lp, p = hp * hp;   // make the 2 ms envelope swing wildly
        c.eFast += (p - c.eFast) / (0.002 * sampleRate);
        c.eMed += (p - c.eMed) / (0.010 * sampleRate);
        if (++c.tick >= kRingStep)
        {
            c.tick = 0;
            c.ring[c.ringPos] = c.eMed;
            c.ringPos = (c.ringPos + 1) % kRing;
        }
        // A jump is "well above the quietest recent level" (not above a long-term
        // floor, which real vocals' breaths/room tails keep tripping). The
        // bypass is graded: noise bursts (+14 dB) barely open it, a word
        // starting out of silence (+26 dB and up) opens it fully. Tuned on the
        // carina vocal: 14/26 keeps speech within 0.1 dB of Quality mode and gaps
        // at -16 dB (Quality -19); lower trips on noise bursts, higher dulls onsets.
        double ref = c.ring[0];
        for (int k = 1; k < kRing; ++k) ref = std::min (ref, c.ring[k]);
        const double ex = 10.0 * std::log10 (c.eFast / std::max (ref, 1e-10) + 1e-12);
        const double grade = std::min (1.0, std::max (0.0, (ex - 14.0) / 12.0));
        if (c.cool > 0) --c.cool;
        else if (grade > 0.0)
        {
            c.hold = (int) (0.045 * sampleRate);
            c.cool = c.hold + (int) (0.075 * sampleRate);
            c.target = 0.0;
        }
        if (c.hold > 0)
        {
            --c.hold;
            c.target = std::max (c.target, grade);
            c.onset += std::min (1.0 / (0.001 * sampleRate), std::max (0.0, c.target - c.onset));
        }
        else c.onset = std::max (0.0, c.onset - 1.0 / (0.015 * sampleRate));
        return y + c.onset * (x - y);
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
            c.worker.reset();   // join before the state goes away
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
    std::unique_ptr<Fft> liveFft;
    std::vector<double> dRe, dIm;
};

} // namespace cleer
