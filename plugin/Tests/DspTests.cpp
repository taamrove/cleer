// Offline DSP harness for the Cleer plugin core. Builds without JUCE.
// Synthesises signals with known properties and checks the chain does what
// the validated Python reference proved:
//   1. a growing tonal howl gets notched down,
//   2. a steady harmonic "sung note" is left alone (growth gate),
//   3. the expander pulls inter-phrase noise down toward the floor without
//      touching the phrase itself.
// Exits non-zero on failure; `ctest` runs it.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>
#include "../Source/dsp/CleerEngine.h"
#include "../Source/dsp/Dereverb.h"
#include "../Source/dsp/NoiseSuppressor.h"

namespace
{
constexpr double kSampleRate = 48000.0;
int failures = 0;

void check (bool ok, const char* what)
{
    std::printf ("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (! ok)
        ++failures;
}

// RMS of a slice, in dBFS.
double rmsDB (const std::vector<float>& x, size_t from, size_t to)
{
    double acc = 0.0;
    for (size_t i = from; i < to; ++i)
        acc += (double) x[i] * (double) x[i];
    const double rms = std::sqrt (acc / (double) (to - from));
    return 20.0 * std::log10 (rms + 1e-12);
}

// Goertzel power of one frequency over a slice, in dB (relative units).
double toneDB (const std::vector<float>& x, size_t from, size_t to, double freq)
{
    const double w = 2.0 * M_PI * freq / kSampleRate;
    const double coeff = 2.0 * std::cos (w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (size_t i = from; i < to; ++i)
    {
        s0 = (double) x[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    const double p = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    return 10.0 * std::log10 (p / (double) (to - from) + 1e-18);
}

// Run a mono signal through an Engine in 512-sample host-style blocks.
void run (cleer::Engine& e, std::vector<float>& x)
{
    const int block = 512;
    for (size_t i = 0; i < x.size(); i += (size_t) block)
    {
        float* p = x.data() + i;
        const int n = (int) std::min ((size_t) block, x.size() - i);
        e.processBlock (&p, 1, n);
    }
}

std::vector<float> whiteNoise (size_t total, double levelDB, unsigned seed)
{
    std::vector<float> x (total);
    unsigned rng = seed;
    const double amp = std::pow (10.0, levelDB / 20.0);
    for (size_t i = 0; i < total; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        x[i] = (float) (((double) (rng >> 8) / 8388608.0 - 1.0) * amp);
    }
    return x;
}

void testHowlSuppressed()
{
    // A howl: 1240 Hz sine growing 2.5 dB per hop (5.33 ms) from -70 dBFS,
    // clamped at -6 dBFS, over quiet broadband noise. The growth phase is
    // long enough (~25 frames above prominence while growing) to trip the
    // 12-frame persistence gate.
    const double f = 1240.0;
    const size_t total = (size_t) (4.0 * kSampleRate);
    std::vector<float> x (total);
    unsigned rng = 1;
    const double hop = 256.0;
    for (size_t i = 0; i < total; ++i)
    {
        const double frames = (double) i / hop;
        const double ampDB = std::min (-6.0, -70.0 + 2.5 * frames);
        const double amp = std::pow (10.0, ampDB / 20.0);
        rng = rng * 1664525u + 1013904223u;
        const double noise = ((double) (rng >> 8) / 8388608.0 - 1.0) * std::pow (10.0, -60.0 / 20.0);
        x[(size_t) i] = (float) (amp * std::sin (2.0 * M_PI * f * (double) i / kSampleRate) + noise);
    }

    const size_t tail0 = total - (size_t) kSampleRate, tail1 = total;
    const double before = toneDB (x, tail0, tail1, f);

    cleer::Engine e;
    e.prepare (kSampleRate, 1, 512);
    e.gateEnabled = false;
    e.noiseEnabled = false;   // isolate the suppressor
    run (e, x);

    const double after = toneDB (x, tail0, tail1, f);
    const double reduction = before - after;
    std::printf ("       howl at %.0f Hz: %.1f dB reduction, %d active notch(es)\n",
                 f, reduction, e.activeNotchCount());
    check (e.activeNotchCount() >= 1, "howl: at least one active notch");
    check (reduction > 15.0, "howl: tone reduced by > 15 dB in steady state");
}

void testSteadyVocalUntouched()
{
    // A held "sung note": constant-amplitude 220 Hz fundamental + harmonics
    // (several inside the 300-9000 Hz detection band). Constant power means
    // the growth gate must keep every notch away.
    const size_t total = (size_t) (3.0 * kSampleRate);
    std::vector<float> x (total);
    for (size_t i = 0; i < total; ++i)
    {
        const double t = (double) i / kSampleRate;
        double s = 0.0;
        for (int h = 1; h <= 8; ++h)
            s += (1.0 / (double) h) * std::sin (2.0 * M_PI * 220.0 * (double) h * t);
        x[i] = (float) (0.1 * s);
    }

    const double before = rmsDB (x, total / 2, total);

    cleer::Engine e;
    e.prepare (kSampleRate, 1, 512);
    e.gateEnabled = false;
    e.noiseEnabled = false;   // RNNoise rightly eats synthetic tones; test the detector alone
    run (e, x);

    const double after = rmsDB (x, total / 2, total);
    std::printf ("       steady note: %d notches, level delta %.2f dB\n",
                 e.activeNotchCount(), after - before);
    check (e.activeNotchCount() == 0, "steady note: no notches placed");
    check (std::abs (after - before) < 0.1, "steady note: level unchanged");
}

void testExpander()
{
    // 1 s of loud 400 Hz phrase at -12 dBFS, then 2 s of -60 dBFS noise.
    // The phrase must pass untouched; the noise must end up pulled down by
    // roughly the 24 dB floor.
    const size_t loud = (size_t) kSampleRate;
    const size_t total = (size_t) (3.0 * kSampleRate);
    std::vector<float> x (total);
    unsigned rng = 7;
    for (size_t i = 0; i < total; ++i)
    {
        if (i < loud)
            x[i] = (float) (0.25 * std::sin (2.0 * M_PI * 400.0 * (double) i / kSampleRate));
        else
        {
            rng = rng * 1664525u + 1013904223u;
            x[i] = (float) (((double) (rng >> 8) / 8388608.0 - 1.0) * std::pow (10.0, -60.0 / 20.0));
        }
    }

    const double loudBefore = rmsDB (x, loud / 2, loud);
    const double quietBefore = rmsDB (x, total - (size_t) kSampleRate, total);

    cleer::Engine e;
    e.prepare (kSampleRate, 1, 512);
    e.feedbackEnabled = false;
    e.noiseEnabled = false;
    run (e, x);

    const double loudAfter = rmsDB (x, loud / 2, loud);
    const double quietAfter = rmsDB (x, total - (size_t) kSampleRate, total);
    std::printf ("       expander: phrase %.2f dB, noise %.1f dB (floor -24)\n",
                 loudAfter - loudBefore, quietAfter - quietBefore);
    check (std::abs (loudAfter - loudBefore) < 0.5, "expander: phrase untouched");
    check (quietAfter - quietBefore < -18.0, "expander: tail noise pulled down > 18 dB");
    check (quietAfter - quietBefore > -30.0, "expander: attenuation clamped near the floor");
}

void testNoiseSuppressor()
{
    // Pure broadband noise at -35 dBFS: RNNoise should hear no voice and
    // strip most of it. 4 s gives the noise-floor tracker time to settle.
    const size_t total = (size_t) (4.0 * kSampleRate);
    auto x = whiteNoise (total, -35.0, 3);
    const double before = rmsDB (x, total - (size_t) kSampleRate, total);

    cleer::NoiseSuppressor ns;
    ns.prepare (kSampleRate, 1);
    std::vector<double> d (x.begin(), x.end());
    for (size_t i = 0; i < d.size(); i += 512)
    {
        double* p = d.data() + i;
        ns.processBlock (&p, 1, (int) std::min ((size_t) 512, d.size() - i));
    }
    std::vector<float> y (d.begin(), d.end());

    const double after = rmsDB (y, total - (size_t) kSampleRate, total);
    std::printf ("       rnnoise: noise %.1f -> %.1f dB (%.1f dB removed), latency %d\n",
                 before, after, before - after, ns.latencySamples());
    check (before - after > 10.0, "rnnoise: broadband noise cut by > 10 dB");
    // 1 frame accumulation + 2 frames model lookahead + 1 frame worker hand-off.
    check (ns.latencySamples() > 1850 && ns.latencySamples() < 2050,
           "rnnoise: latency ~4 frames at 48k");
}

void testNoiseLatencyAlignment()
{
    // With mix = 0 the stage is a pure delay: a sine must come out intact,
    // shifted by exactly latencySamples().
    cleer::NoiseSuppressor ns;
    ns.prepare (kSampleRate, 1);
    ns.mix = 0.0;
    const int lat = ns.latencySamples();

    const size_t total = (size_t) kSampleRate;
    std::vector<double> in (total), out (total);
    for (size_t i = 0; i < total; ++i)
        in[i] = 0.5 * std::sin (2.0 * M_PI * 440.0 * (double) i / kSampleRate);
    out = in;
    double* p = out.data();
    for (size_t i = 0; i < total; i += 512)
    {
        p = out.data() + i;
        ns.processBlock (&p, 1, (int) std::min ((size_t) 512, total - i));
    }

    double maxErr = 0.0;
    for (size_t i = (size_t) kSampleRate / 2; i < total; ++i)
        maxErr = std::max (maxErr, std::abs (out[i] - in[i - (size_t) lat]));
    std::printf ("       alignment: latency %d, max dry-path error %.2e\n", lat, maxErr);
    check (maxErr < 1e-3, "rnnoise: mix=0 path is a clean delay of latencySamples()");
}

void testNoiseAmountBoundsAttenuation()
{
    // A bare sine is not speech, so RNNoise removes it entirely — which
    // makes it the perfect probe for the AMOUNT blend: at mix = 0.5 the
    // output must contain at least the dry half, i.e. the tone can lose at
    // most 6 dB no matter what the model decides.
    const size_t total = (size_t) (3.0 * kSampleRate);
    std::vector<double> d (total);
    for (size_t i = 0; i < total; ++i)
        d[i] = 0.15 * std::sin (2.0 * M_PI * 330.0 * (double) i / kSampleRate);
    std::vector<float> fx (d.begin(), d.end());
    const double before = toneDB (fx, total / 2, total, 330.0);

    cleer::NoiseSuppressor ns;
    ns.prepare (kSampleRate, 1);
    ns.mix = 0.5;
    for (size_t i = 0; i < total; i += 512)
    {
        double* p = d.data() + i;
        ns.processBlock (&p, 1, (int) std::min ((size_t) 512, total - i));
    }
    std::vector<float> fy (d.begin(), d.end());
    const double after = toneDB (fy, total / 2, total, 330.0);
    std::printf ("       amount: mix=0.5 keeps tone %.1f -> %.1f dB (>= -6.1)\n", before, after);
    check (after > before - 6.1, "rnnoise: mix bounds attenuation at 6 dB for mix=0.5");
}

void testNoiseResampledRate()
{
    // 44.1 kHz host: the resampler pair must still deliver noise reduction
    // and keep a voice-band tone, and latency must scale with the rate.
    const double sr = 44100.0;
    cleer::NoiseSuppressor ns;
    ns.prepare (sr, 1);
    const int lat = ns.latencySamples();

    const size_t total = (size_t) (4.0 * sr);
    std::vector<double> d (total);
    unsigned rng = 23;
    for (size_t i = 0; i < total; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        d[i] = ((double) (rng >> 8) / 8388608.0 - 1.0) * std::pow (10.0, -35.0 / 20.0);
    }
    std::vector<float> fx (d.begin(), d.end());
    const double before = 20.0 * std::log10 (1e-12 + [&] {
        double acc = 0.0;
        for (size_t i = total - (size_t) sr; i < total; ++i) acc += d[i] * d[i];
        return std::sqrt (acc / sr);
    }());

    double* p = d.data();
    for (size_t i = 0; i < total; i += 441)
    {
        p = d.data() + i;
        ns.processBlock (&p, 1, (int) std::min ((size_t) 441, total - i));
    }
    double acc = 0.0;
    for (size_t i = total - (size_t) sr; i < total; ++i) acc += d[i] * d[i];
    const double after = 20.0 * std::log10 (1e-12 + std::sqrt (acc / sr));

    std::printf ("       44.1k: noise %.1f -> %.1f dB, latency %d (~%.1f ms)\n",
                 before, after, lat, 1000.0 * lat / sr);
    check (before - after > 10.0, "rnnoise 44.1k: noise cut by > 10 dB");
    check (lat > 1650 && lat < 1850, "rnnoise 44.1k: latency ~4 frames scaled");
}

void testWetDryAligned()
{
    // Regression for the AMOUNT "flange": the model's 2-frame lookahead
    // means wet lags the raw input; the blend must delay dry to match.
    // Cross-correlate a mix=1 run against a mix=0 run of a speech-like
    // probe — the best lag must be zero.
    const double sr = kSampleRate;
    const size_t n = (size_t) (3.0 * sr);
    std::vector<double> voice (n);
    for (size_t i = 0; i < n; ++i)
    {
        const double t = (double) i / sr;
        const double f0 = 220.0 * (1.0 + 0.02 * std::sin (2.0 * M_PI * 5.5 * t));
        const double am = 0.55 + 0.45 * std::sin (2.0 * M_PI * 3.0 * t);
        double s = 0.0;
        for (int h = 1; h <= 8; ++h)
            s += (1.0 / h) * std::sin (2.0 * M_PI * f0 * h * t);
        voice[i] = 0.15 * am * s;
    }

    auto process = [&] (double mixVal) {
        cleer::NoiseSuppressor ns;
        ns.prepare (sr, 1);
        ns.mix = mixVal;
        std::vector<double> d = voice;
        for (size_t i = 0; i < n; i += 512)
        {
            double* p = d.data() + i;
            ns.processBlock (&p, 1, (int) std::min ((size_t) 512, n - i));
        }
        return d;
    };
    const auto wet = process (1.0);
    const auto dry = process (0.0);

    const size_t s0 = (size_t) sr, s1 = (size_t) (2.5 * sr);
    double bestC = -1e18;
    int bestLag = 0;
    for (int lag = -1200; lag <= 1200; ++lag)
    {
        double c = 0.0;
        for (size_t i = s0; i < s1; i += 2)
            c += wet[i] * dry[(size_t) ((long) i + lag)];
        if (c > bestC)
        {
            bestC = c;
            bestLag = lag;
        }
    }
    std::printf ("       wet/dry alignment: best lag %d samples\n", bestLag);
    check (std::abs (bestLag) <= 8, "rnnoise: wet and dry branches sample-aligned (no flange)");
}

void testDereverb()
{
    // Speech-burst-like tone bursts convolved with a synthetic room
    // (exponentially decaying noise IR, RT60 ~0.5 s). The dereverb stage
    // must pull the inter-burst reverb tail down without wrecking the
    // bursts themselves.
    const double rt60 = 0.5;
    const size_t irLen = (size_t) (0.4 * kSampleRate);
    std::vector<double> ir (irLen);
    unsigned rng = 5;
    for (size_t i = 0; i < irLen; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        const double t = (double) i / kSampleRate;
        ir[i] = ((double) (rng >> 8) / 8388608.0 - 1.0) * std::pow (10.0, -3.0 * t / rt60) * 0.05;
    }
    ir[0] = 1.0;   // direct path

    // Dry: 0.4 s bursts of a 300 Hz + harmonics tone, 0.6 s gaps, 6 s total.
    const size_t total = (size_t) (6.0 * kSampleRate);
    std::vector<double> dry (total, 0.0);
    for (size_t i = 0; i < total; ++i)
    {
        const double t = (double) i / kSampleRate;
        const double phase = std::fmod (t, 1.0);
        if (phase < 0.4)
        {
            const double env = std::min (1.0, std::min (phase / 0.02, (0.4 - phase) / 0.02));
            dry[i] = env * 0.2 * (std::sin (2.0 * M_PI * 300.0 * t)
                                + 0.5 * std::sin (2.0 * M_PI * 600.0 * t));
        }
    }
    // Convolve (sparse-ish: only every 4th IR tap to keep the test fast).
    std::vector<double> wet (total, 0.0);
    for (size_t i = 0; i < total; ++i)
        if (dry[i] != 0.0)
            for (size_t k = 0; k < irLen && i + k < total; k += 4)
                wet[i + k] += dry[i] * ir[k] * 4.0 * 0.25;

    // Tail windows: 150-400 ms after each burst ends (bursts end at 0.4+n).
    auto tailRms = [&] (const std::vector<double>& x) {
        double acc = 0.0;
        size_t cnt = 0;
        for (int b = 1; b < 5; ++b)
        {
            const size_t a = (size_t) ((b + 0.55) * kSampleRate);
            const size_t e = (size_t) ((b + 0.80) * kSampleRate);
            for (size_t i = a; i < e; ++i, ++cnt)
                acc += x[i] * x[i];
        }
        return 20.0 * std::log10 (std::sqrt (acc / (double) cnt) + 1e-12);
    };

    const double tailBefore = tailRms (wet);

    cleer::DereverbStage dv;
    cleer::DereverbStage::Params dp;
    dp.rt60 = rt60;
    dp.strength = 1.0;
    dv.prepare (kSampleRate, 1);
    dv.setParams (dp);
    dv.enabled = true;
    std::vector<double> proc = wet;
    for (size_t i = 0; i < total; i += 512)
    {
        double* p = proc.data() + i;
        dv.processBlock (&p, 1, (int) std::min ((size_t) 512, total - i));
    }
    // Latency-compensate before windowed measurement.
    const int lat = dv.latencySamples();
    proc.erase (proc.begin(), proc.begin() + lat);
    proc.resize (total, 0.0);

    const double tailAfter = tailRms (proc);
    std::printf ("       dereverb: tail %.1f -> %.1f dB (%.1f dB removed), latency %d\n",
                 tailBefore, tailAfter, tailBefore - tailAfter, lat);
    check (tailBefore - tailAfter > 3.0, "dereverb: reverb tail cut by > 3 dB");
    check (lat > 450 && lat < 600, "dereverb: latency ~512 samples");
}

void testDereverbBypassClean()
{
    // strength = 0 must be a bit-clean delay (COLA identity through the
    // OLA STFT round trip).
    cleer::DereverbStage dv;
    cleer::DereverbStage::Params dp;
    dp.strength = 0.0;
    dp.gainFloorDB = -60.0;   // floor must not bite with zero subtraction
    dv.prepare (kSampleRate, 1);
    dv.setParams (dp);
    dv.enabled = true;
    const int lat = dv.latencySamples();

    const size_t total = (size_t) kSampleRate;
    std::vector<double> in (total), out (total);
    for (size_t i = 0; i < total; ++i)
        in[i] = 0.4 * std::sin (2.0 * M_PI * 523.0 * (double) i / kSampleRate);
    out = in;
    for (size_t i = 0; i < total; i += 512)
    {
        double* p = out.data() + i;
        dv.processBlock (&p, 1, (int) std::min ((size_t) 512, total - i));
    }
    double maxErr = 0.0;
    for (size_t i = total / 2; i < total; ++i)
        maxErr = std::max (maxErr, std::abs (out[i] - in[i - (size_t) lat]));
    std::printf ("       dereverb bypass: latency %d, max error %.2e\n", lat, maxErr);
    check (maxErr < 1e-6, "dereverb: strength=0 is a clean COLA delay");
}

void testSmartGate()
{
    // The mechanism, isolated from RNNoise: with the envelope sitting
    // between the two VAD-shifted thresholds, the gate must be OPEN when
    // the VAD says voice and CLOSED when it says none. Signal at -40 dB,
    // static threshold -42 dB: plain mode keeps it open; smart mode with
    // vad=0 shifts the threshold to -30 dB and closes it.
    auto runCase = [] (bool smart, double vad) {
        cleer::Expander ex (kSampleRate);
        cleer::Expander::Params p;
        p.thresholdDB = -42.0;
        p.holdMs = 0.0;
        ex.setParams (p);
        ex.smart = smart;
        ex.setVoiceProbability (vad);
        const size_t total = (size_t) kSampleRate;
        std::vector<double> d (total);
        for (size_t i = 0; i < total; ++i)
            d[i] = 0.01 * std::sin (2.0 * M_PI * 500.0 * (double) i / kSampleRate);   // -40 dB
        double* ptr = d.data();
        ex.processBlock (&ptr, 1, (int) total);
        double acc = 0.0;
        for (size_t i = total / 2; i < total; ++i)
            acc += d[i] * d[i];
        return 20.0 * std::log10 (std::sqrt (acc / (total / 2.0)) + 1e-12);
    };

    const double plain = runCase (false, 0.0);
    const double voiceless = runCase (true, 0.0);
    const double voiced = runCase (true, 1.0);
    std::printf ("       smart gate: plain %.1f, vad=0 %.1f, vad=1 %.1f dB\n",
                 plain, voiceless, voiced);
    check (plain > -44.0, "smart gate: plain mode leaves -40 dB signal open");
    check (voiceless < plain - 10.0, "smart gate: vad=0 closes on the same signal");
    check (voiced > -44.0, "smart gate: vad=1 keeps it open");
}

// Regression: RNNoise inference (~1 ms/frame) ran inline in the callback, so
// at 48k/64-sample buffers (1.33 ms budget) every ~7th callback overran and
// the output glitched constantly; >=128 was fine. No callback may now do the
// inference itself, whatever the block size.
void testNoSlowCallbackAtSmallBlocks()
{
    const int bs = 64;
    cleer::Engine e;
    e.prepare (48000.0, 1, bs);
    e.gateEnabled = false;
    std::vector<float> blk ((size_t) bs);
    unsigned seed = 1;
    double worstMs = 0.0;
    int slow = 0;   // old inline code: one per frame (~1 in 7.5 callbacks)
    auto next = std::chrono::steady_clock::now();
    for (int b = 0; b < 1000; ++b)
    {
        next += std::chrono::microseconds (1333);   // real-time pacing
        std::this_thread::sleep_until (next);
        for (auto& v : blk) { seed = seed * 1664525u + 1013904223u; v = 0.05f * ((seed >> 8) / 8388608.0f - 1.0f); }
        float* p = blk.data();
        const auto t0 = std::chrono::steady_clock::now();
        e.processBlock (&p, 1, bs);
        const double ms = std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - t0).count();
        if (b >= 50) { worstMs = std::max (worstMs, ms); slow += ms > 0.4; }   // skip warm-up
    }
    std::printf ("       64-sample callbacks: worst %.3f ms, %d slow (budget 1.333 ms)\n", worstMs, slow);
    check (slow <= 10, "rnnoise: no callback runs a model frame inline at 64-sample blocks");
}

// ---- Live mode -----------------------------------------------------------

// Speech-like test signal: 120 Hz glottal pulses (slight vibrato) through
// three formant resonators, gated into 0.4 s "syllables" every 0.9 s,
// over stationary white noise. RNNoise treats this as voice (a bare sine
// it would annihilate).
struct SpeechNoise
{
    std::vector<float> mix, noiseOnly, speechOnly;
    std::vector<std::pair<size_t, size_t>> speechSpans, noiseSpans;
};

SpeechNoise makeSpeechNoise (double seconds, double noiseDB, unsigned seed)
{
    const size_t total = (size_t) (seconds * kSampleRate);
    SpeechNoise s;
    s.noiseOnly = whiteNoise (total, noiseDB, seed);
    s.speechOnly.assign (total, 0.0f);
    const double fc[3] = { 700, 1200, 2600 }, bw[3] = { 90, 110, 160 }, amp[3] = { 1.0, 0.6, 0.3 };
    double y1[3] = {}, y2[3] = {}, ph = 0.0;
    for (size_t i = 0; i < total; ++i)
    {
        const double t = (double) i / kSampleRate;
        ph += (120.0 + 4.0 * std::sin (2.0 * M_PI * 5.0 * t)) / kSampleRate;
        double src = 0.0;
        if (ph >= 1.0) { ph -= 1.0; src = 1.0; }
        double o = 0.0;
        for (int f = 0; f < 3; ++f)
        {
            const double r = std::exp (-M_PI * bw[f] / kSampleRate);
            const double a1 = -2.0 * r * std::cos (2.0 * M_PI * fc[f] / kSampleRate);
            const double y = src - a1 * y1[f] - r * r * y2[f];
            y2[f] = y1[f]; y1[f] = y;
            o += amp[f] * y * (1.0 - r);
        }
        const double cyc = std::fmod (t, 0.9);
        const double env = cyc < 0.4 ? std::min (1.0, std::min (cyc, 0.4 - cyc) / 0.02) : 0.0;
        s.speechOnly[i] = (float) (o * env * 6.0);
    }
    // normalise speech to -22 dBFS RMS over the voiced spans
    double acc = 0.0; size_t cnt = 0;
    for (size_t i = 0; i < total; ++i)
        if (std::fmod ((double) i / kSampleRate, 0.9) < 0.4) { acc += (double) s.speechOnly[i] * s.speechOnly[i]; ++cnt; }
    const double g = std::pow (10.0, -22.0 / 20.0) / std::sqrt (acc / (double) cnt);
    for (auto& v : s.speechOnly) v = (float) (v * g);
    s.mix.resize (total);
    for (size_t i = 0; i < total; ++i)
        s.mix[i] = s.speechOnly[i] + s.noiseOnly[i];
    for (double t0 = 0.0; (t0 + 0.9) * kSampleRate < (double) total; t0 += 0.9)
    {
        s.speechSpans.push_back ({ (size_t) ((t0 + 0.1) * kSampleRate), (size_t) ((t0 + 0.35) * kSampleRate) });
        s.noiseSpans.push_back ({ (size_t) ((t0 + 0.6) * kSampleRate), (size_t) ((t0 + 0.88) * kSampleRate) });
    }
    return s;
}

double spanRms (const std::vector<float>& x, const std::vector<std::pair<size_t, size_t>>& sp, size_t skip)
{
    double acc = 0.0, n = 0.0;
    for (size_t k = skip; k < sp.size(); ++k)
        for (size_t i = sp[k].first; i < sp[k].second; ++i) { acc += (double) x[i] * x[i]; n += 1.0; }
    return 20.0 * std::log10 (std::sqrt (acc / n) + 1e-12);
}

void testLiveLatency()
{
    // Impulse through the live stage comes out where it went in.
    cleer::NoiseSuppressor ns;
    ns.prepare (kSampleRate, 1);
    ns.setLive (true);
    const size_t total = (size_t) (2.0 * kSampleRate), at = (size_t) (1.5 * kSampleRate) + 17;
    std::vector<double> d (total, 0.0);
    d[at] = 0.5;
    for (size_t i = 0; i < total; i += 64)
    {
        double* p = d.data() + i;
        ns.processBlock (&p, 1, 64);
    }
    size_t best = 0;
    for (size_t i = 0; i < total; ++i)
        if (std::abs (d[i]) > std::abs (d[best])) best = i;
    std::printf ("       live: reported latency %d, impulse peak at +%d samples\n",
                 ns.latencySamples(), (int) best - (int) at);
    check (ns.latencySamples() == 0, "live: reported latency is 0");
    check ((int) best - (int) at >= 0 && (int) best - (int) at <= 1, "live: impulse aligned with input");
}

void testLiveSuppresses()
{
    auto s = makeSpeechNoise (8.0, -38.0, 11);
    cleer::Engine e;
    e.prepare (kSampleRate, 1, 64);
    e.gateEnabled = false;
    e.feedbackEnabled = false;
    e.setNoiseLive (true);
    e.setNoiseOfflineExact (true);
    e.setNoiseMix (0.9);
    auto y = s.mix;
    for (size_t i = 0; i < y.size(); i += 64)
    {
        float* p = y.data() + i;
        e.processBlock (&p, 1, (int) std::min ((size_t) 64, y.size() - i));
    }
    const double nIn = spanRms (s.mix, s.noiseSpans, 2), nOut = spanRms (y, s.noiseSpans, 2);
    const double sIn = spanRms (s.mix, s.speechSpans, 2), sOut = spanRms (y, s.speechSpans, 2);
    std::printf ("       live: noise %.1f -> %.1f dB (%.1f removed), speech %.1f -> %.1f dB (%+.1f)\n",
                 nIn, nOut, nIn - nOut, sIn, sOut, sOut - sIn);
    {   // reference: Quality mode on the same material (latency-compensated)
        cleer::Engine q;
        q.prepare (kSampleRate, 1, 512);
        q.gateEnabled = false; q.feedbackEnabled = false;
        q.setNoiseMix (0.9);
        auto z = s.mix;
        z.resize (z.size() + 4096, 0.0f);
        run (q, z);
        z.erase (z.begin(), z.begin() + q.latencySamples());
        z.resize (s.mix.size());
        std::printf ("       (quality reference: noise %.1f dB removed, speech %+.1f dB)\n",
                     nIn - spanRms (z, s.noiseSpans, 2), spanRms (z, s.speechSpans, 2) - sIn);
    }
    check (nIn - nOut > 12.0, "live: noise-only stretches cut by > 12 dB");
    check (std::abs (sOut - sIn) < 2.0, "live: speech level within 2 dB");
}

void testLiveOnset()
{
    // The first 30 ms of each word must not be dulled while the model's
    // gains are still those of the preceding silence.
    auto s = makeSpeechNoise (8.0, -38.0, 11);
    cleer::Engine e;
    e.prepare (kSampleRate, 1, 64);
    e.gateEnabled = false; e.feedbackEnabled = false;
    e.setNoiseLive (true);
    e.setNoiseOfflineExact (true);
    e.setNoiseMix (0.9);
    auto y = s.mix;
    for (size_t i = 0; i < y.size(); i += 64)
    {
        float* p = y.data() + i;
        e.processBlock (&p, 1, (int) std::min ((size_t) 64, y.size() - i));
    }
    std::vector<std::pair<size_t, size_t>> on;
    for (double t0 = 1.8; t0 + 0.9 < 7.5; t0 += 0.9)
        on.push_back ({ (size_t) (t0 * kSampleRate), (size_t) ((t0 + 0.12) * kSampleRate) });
    const double a = spanRms (s.mix, on, 0), b = spanRms (y, on, 0);
    std::printf ("       live onset (first 120 ms of words): %.1f -> %.1f dB (%+.1f)\n", a, b, b - a);
    check (b - a > -0.9, "live: onset bypass keeps first 120 ms of words within 0.9 dB (no bypass: -1.2)");
}

void testLiveNoSlowCallback()
{
    const int bs = 64;
    cleer::Engine e;
    e.prepare (48000.0, 1, bs);
    e.gateEnabled = false;
    e.setNoiseLive (true);
    std::vector<float> blk ((size_t) bs);
    unsigned seed = 1;
    double worstMs = 0.0, sumMs = 0.0;
    int slow = 0;
    auto next = std::chrono::steady_clock::now();
    for (int b = 0; b < 1500; ++b)
    {
        next += std::chrono::microseconds (1333);
        std::this_thread::sleep_until (next);
        for (auto& v : blk) { seed = seed * 1664525u + 1013904223u; v = 0.05f * ((seed >> 8) / 8388608.0f - 1.0f); }
        float* p = blk.data();
        const auto t0 = std::chrono::steady_clock::now();
        e.processBlock (&p, 1, bs);
        const double ms = std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - t0).count();
        if (b >= 50) { worstMs = std::max (worstMs, ms); sumMs += ms; slow += ms > 0.4; }
    }
    std::printf ("       live 64-sample callbacks: mean %.3f ms, worst %.3f ms, %d > 0.4 ms (budget 1.333)\n",
                 sumMs / 1450.0, worstMs, slow);
    // Mean = the DSP cost; the over-0.4 ms count is dominated by OS scheduling noise
    // (the Quality test shows the same on a busy machine), so keep it loose.
    check (sumMs / 1450.0 < 0.2 && slow <= 30, "live: callbacks stay under budget at 64-sample blocks (mean < 0.2 ms)");
}

} // namespace

int main()
{
    testHowlSuppressed();
    testSteadyVocalUntouched();
    testExpander();
    testNoiseSuppressor();
    testNoiseLatencyAlignment();
    testNoiseAmountBoundsAttenuation();
    testNoiseResampledRate();
    testWetDryAligned();
    testDereverb();
    testDereverbBypassClean();
    testSmartGate();
    testNoSlowCallbackAtSmallBlocks();
    testLiveLatency();
    testLiveSuppresses();
    testLiveOnset();
    testLiveNoSlowCallback();
    std::printf (failures == 0 ? "\nAll tests passed.\n" : "\n%d test(s) FAILED.\n", failures);
    return failures == 0 ? 0 : 1;
}
