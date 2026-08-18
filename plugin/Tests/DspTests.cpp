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
    run (e, x);

    const double loudAfter = rmsDB (x, loud / 2, loud);
    const double quietAfter = rmsDB (x, total - (size_t) kSampleRate, total);
    std::printf ("       expander: phrase %.2f dB, noise %.1f dB (floor -24)\n",
                 loudAfter - loudBefore, quietAfter - quietBefore);
    check (std::abs (loudAfter - loudBefore) < 0.5, "expander: phrase untouched");
    check (quietAfter - quietBefore < -18.0, "expander: tail noise pulled down > 18 dB");
    check (quietAfter - quietBefore > -30.0, "expander: attenuation clamped near the floor");
}

} // namespace

int main()
{
    testHowlSuppressed();
    testSteadyVocalUntouched();
    testExpander();
    std::printf (failures == 0 ? "\nAll tests passed.\n" : "\n%d test(s) FAILED.\n", failures);
    return failures == 0 ? 0 : 1;
}
