// Soak test: hours of low-level mic hiss + hum through the full chain at a
// chosen sample rate, with a voice-like burst injected periodically. Prints
// per-burst health metrics so "fine at soundcheck, wrong two hours later"
// becomes a number instead of an anecdote.
//
//   cleer_soak [hours] [sampleRate]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../Source/dsp/CleerEngine.h"

int main (int argc, char** argv)
{
    const double hours = argc > 1 ? atof (argv[1]) : 3.0;
    const double sr = argc > 2 ? atof (argv[2]) : 48000.0;

    cleer::Engine e;
    e.prepare (sr, 1, 512);

    const long total = (long) (hours * 3600.0 * sr);
    const long burstEvery = (long) (600.0 * sr);      // burst every 10 sim-minutes
    const long burstLen = (long) (3.0 * sr);

    std::vector<float> block (512);
    unsigned rng = 42;
    long pos = 0, lastUnderruns = 0;
    double burstPhase = 0.0;

    std::printf ("soak: %.1f h at %.0f Hz, burst every 10 min\n", hours, sr);
    std::printf ("%8s %10s %9s %8s %8s %9s\n", "t(min)", "burstRMS", "maxJump", "notches", "gateGR", "underruns");

    while (pos < total)
    {
        const long inBurst = pos % burstEvery;
        const bool burst = inBurst < burstLen;

        double maxJump = 0.0, rmsAcc = 0.0;
        float prev = 0.0f;

        for (int i = 0; i < 512; ++i)
        {
            // mic at idle: hiss ~ -60 dBFS + 50 Hz hum harmonics ~ -70 dBFS
            rng = rng * 1664525u + 1013904223u;
            const double t = (double) (pos + i) / sr;
            double s = ((double) (rng >> 8) / 8388608.0 - 1.0) * 1.0e-3
                     + 3.0e-4 * std::sin (2.0 * M_PI * 50.0 * t)
                     + 1.5e-4 * std::sin (2.0 * M_PI * 150.0 * t);
            if (burst)
            {
                // voice-ish: vibrato harmonic complex with syllabic AM
                const double f0 = 220.0 * (1.0 + 0.02 * std::sin (2.0 * M_PI * 5.5 * t));
                const double am = 0.55 + 0.45 * std::sin (2.0 * M_PI * 3.0 * t);
                burstPhase += 2.0 * M_PI * f0 / sr;
                double v = 0.0;
                for (int h = 1; h <= 6; ++h)
                    v += std::sin (burstPhase * h) / (double) h;
                s += 0.15 * am * v;
            }
            block[(size_t) i] = (float) s;
        }

        float* p = block.data();
        e.processBlock (&p, 1, 512);

        if (burst)
            for (int i = 0; i < 512; ++i)
            {
                maxJump = std::max (maxJump, (double) std::abs (block[(size_t) i] - prev));
                prev = block[(size_t) i];
                rmsAcc += (double) block[(size_t) i] * block[(size_t) i];
            }

        // report once per burst, near its end
        if (burst && inBurst + 512 >= burstLen)
        {
            const long u = e.noiseUnderruns();
            std::printf ("%8.1f %10.4f %9.4f %8d %8.1f %9ld\n",
                         pos / sr / 60.0,
                         std::sqrt (rmsAcc / 512.0),
                         maxJump,
                         e.activeNotchCount(),
                         e.gainReductionDB(),
                         u - lastUnderruns);
            std::fflush (stdout);
            lastUnderruns = u;
        }
        pos += 512;
    }
    std::printf ("done. total underruns %ld\n", e.noiseUnderruns());
    return 0;
}
