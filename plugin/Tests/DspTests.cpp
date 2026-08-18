// Offline DSP harness for the Cleer plugin core. Builds without JUCE.
// Synthesises signals with known properties and checks the chain does what
// the validated Python reference proved:
//   1. a growing tonal howl gets notched down,
//   2. a steady harmonic "sung note" is left alone (growth gate),
//   3. the expander pulls inter-phrase noise down toward the floor without
//      touching the phrase itself.
// Exits non-zero on failure; `ctest` runs it.

#include <cmath>
#include <cstdio>
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
    check (ns.latencySamples() > 400 && ns.latencySamples() < 600,
           "rnnoise: latency ~1 frame at 48k");
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
    check (lat > 380 && lat < 560, "rnnoise 44.1k: latency ~1 frame scaled");
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
    check (lat > 900 && lat < 1200, "dereverb: latency ~1024 samples");
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
    testDereverb();
    testDereverbBypassClean();
    testSmartGate();
    std::printf (failures == 0 ? "\nAll tests passed.\n" : "\n%d test(s) FAILED.\n", failures);
    return failures == 0 ? 0 : 1;
}
