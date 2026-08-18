#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>

namespace cleer
{

// Downward expander for live vocals: keeps the signal full above threshold,
// pulls it down (NOT all the way) below threshold so bleed/room noise between
// phrases is reduced without chopping phrase tails. Port of
// macos/Cleer/DSP/Expander.swift, extended to N channels: the envelope
// follows the loudest channel and one common gain is applied to all, so a
// stereo image doesn't wobble.
//
//   gain_dB = (ratio - 1) * (envelope_dB - threshold_dB)   when env < threshold
//   gain_dB = 0                                            otherwise
// clamped at floorDB so very quiet bleed isn't fully silenced (a hard mute
// reads as unnatural; a -20..-30 dB pull-down sounds clean).
class Expander
{
public:
    struct Params
    {
        double thresholdDB = -42.0;
        double ratio = 3.0;          // 1.0 = off; 3.0 = a 10 dB dip becomes 30 dB
        double floorDB = -24.0;      // max attenuation — don't fully mute
        double attackMs = 2.0;       // gate opening
        double releaseMs = 150.0;    // gate closing
        double holdMs = 40.0;        // minimum open time after last trigger
        double envAttackMs = 1.0;    // peak follower attack
        double envReleaseMs = 30.0;  // peak follower release

        // Change detection for cheap coefficient recompution — exact bit
        // equality is the right test here, epsilon would just delay updates.
        bool sameAs (const Params& o) const
        {
            auto eq = [] (double a, double b) { return std::memcmp (&a, &b, sizeof a) == 0; };
            return eq (thresholdDB, o.thresholdDB) && eq (ratio, o.ratio)
                && eq (floorDB, o.floorDB) && eq (attackMs, o.attackMs)
                && eq (releaseMs, o.releaseMs) && eq (holdMs, o.holdMs)
                && eq (envAttackMs, o.envAttackMs) && eq (envReleaseMs, o.envReleaseMs);
        }
    };

    explicit Expander (double sampleRateIn) : sampleRate (sampleRateIn)
    {
        recompute();
    }

    void setParams (const Params& p)
    {
        if (! p.sameAs (params))
        {
            params = p;
            recompute();
        }
    }

    const Params& getParams() const { return params; }
    bool enabled = true;

    // Voice-adaptive ("smart") mode: the neural VAD from the noise stage
    // slides the effective threshold by up to +/-kVoiceSwingDB. When the
    // model hears voice the gate is 12 dB more permissive (keeps breaths
    // and phrase tails); when it hears none the gate is 12 dB stricter, so
    // bleed louder than the static threshold still gets pulled down. This
    // is what makes the gate track *the vocal* rather than a level.
    bool smart = false;
    static constexpr double kVoiceSwingDB = 12.0;

    void setVoiceProbability (double p)
    {
        vadTarget = p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
    }

    // Process numCh channel pointers of n samples in place with a shared gain.
    void processBlock (double* const* ch, int numCh, int n)
    {
        if (! enabled || numCh <= 0)
            return;

        const double rm1 = params.ratio - 1.0;
        double thrLin = std::pow (10.0, params.thresholdDB / 20.0);
        double thrDB = params.thresholdDB;

        for (int i = 0; i < n; ++i)
        {
            if (smart)
            {
                // ~30 ms one-pole toward the frame-rate VAD value.
                vadSmoothed += vadCoeff * (vadTarget - vadSmoothed);
                const double d = params.thresholdDB - kVoiceSwingDB * (2.0 * vadSmoothed - 1.0);
                if (std::abs (d - thrDB) > 0.01)
                {
                    thrDB = d;
                    thrLin = std::pow (10.0, d / 20.0);
                }
            }

            double absX = 0.0;
            for (int c = 0; c < numCh; ++c)
                absX = std::max (absX, std::abs (ch[c][i]));

            // Peak follower: fast up, slow decay.
            if (absX > env)
                env = envAttackCoeff * env + (1.0 - envAttackCoeff) * absX;
            else
                env = envReleaseCoeff * env + (1.0 - envReleaseCoeff) * absX;

            double targetGain = 1.0;
            if (env < thrLin)
            {
                const double envDB = 20.0 * std::log10 (env + 1e-12);
                const double gainDB = rm1 * (envDB - thrDB);
                targetGain = std::pow (10.0, gainDB / 20.0);
                if (targetGain < floorLinear)
                    targetGain = floorLinear;
            }

            // Hold the gate open for holdMs after the envelope last crossed.
            if (env >= thrLin)
            {
                holdSamples = holdLengthSamples;
                targetGain = 1.0;
            }
            else if (holdSamples > 0)
            {
                --holdSamples;
                targetGain = 1.0;
            }

            // Asymmetric smoothing — fast open, slow close.
            const double coeff = targetGain > gain ? gainAttackCoeff : gainReleaseCoeff;
            gain = coeff * gain + (1.0 - coeff) * targetGain;

            for (int c = 0; c < numCh; ++c)
                ch[c][i] *= gain;
        }
    }

    // Current gain reduction in dB (negative; 0 means fully open).
    double gainReductionDB() const { return 20.0 * std::log10 (std::max (gain, 1e-6)); }

    void reset()
    {
        env = 0.0;
        gain = 1.0;
        holdSamples = 0;
        vadSmoothed = 0.0;
        vadTarget = 0.0;
    }

private:
    void recompute()
    {
        envAttackCoeff = onePole (params.envAttackMs);
        envReleaseCoeff = onePole (params.envReleaseMs);
        gainAttackCoeff = onePole (params.attackMs);
        gainReleaseCoeff = onePole (params.releaseMs);
        holdLengthSamples = (int) (params.holdMs * 0.001 * sampleRate);
        floorLinear = std::pow (10.0, params.floorDB / 20.0);
        vadCoeff = 1.0 - onePole (30.0);
    }

    double onePole (double ms) const
    {
        const double t = std::max (0.0001, ms * 0.001);
        return std::exp (-1.0 / (t * sampleRate));
    }

    double sampleRate;
    Params params;

    double env = 0.0;
    double gain = 1.0;
    int holdSamples = 0;

    double envAttackCoeff = 0.0, envReleaseCoeff = 0.0;
    double gainAttackCoeff = 0.0, gainReleaseCoeff = 0.0;
    int holdLengthSamples = 0;
    double floorLinear = 0.0;
    double vadTarget = 0.0, vadSmoothed = 0.0, vadCoeff = 0.001;
};

} // namespace cleer
