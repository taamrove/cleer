import Foundation

/// Downward expander for live vocals: keeps the signal full above threshold,
/// pulls it down (NOT all the way) below threshold so bleed/room noise between
/// phrases is reduced without chopping phrase tails.
///
/// Time-domain, real-time-safe, zero allocations per process call. Standard
/// expander math:
///   gain_dB = (ratio - 1) * (envelope_dB - threshold_dB)   when env < threshold
///   gain_dB = 0                                            otherwise
/// then clamped at `floorDB` so very quiet bleed isn't fully silenced
/// (a hard mute reads as unnatural; a -20…-30 dB pull-down sounds clean).
///
/// Envelope: peak follower with fast attack (1 ms) and medium release (30 ms),
/// so the gate opens on transients but doesn't chatter on inter-syllable dips.
/// Gain smoothing uses separate `attack` (open) and `release` (close) one-pole
/// coefficients, plus a `hold` time so brief syllable gaps don't close the gate.
final class Expander {
    struct Params {
        var thresholdDB: Double = -42.0
        var ratio: Double = 3.0           // 1.0 = off; 3.0 = a 10 dB dip becomes 30 dB
        var floorDB: Double = -24.0       // max attenuation — don't fully mute
        var attackMs: Double = 2.0        // gate opening
        var releaseMs: Double = 150.0     // gate closing
        var holdMs: Double = 40.0         // minimum open time after last trigger
        var envAttackMs: Double = 1.0     // peak follower attack
        var envReleaseMs: Double = 30.0   // peak follower release
    }

    var params: Params { didSet { recomputeCoefficients() } }
    var enabled = true

    private let sampleRate: Double
    private var env: Double = 0
    private var gain: Double = 1.0
    private var holdSamples: Int = 0

    private var envAttackCoeff: Double = 0
    private var envReleaseCoeff: Double = 0
    private var gainAttackCoeff: Double = 0
    private var gainReleaseCoeff: Double = 0
    private var holdLengthSamples: Int = 0
    private var floorLinear: Double = 0

    init(sampleRate: Double, params: Params = Params()) {
        self.sampleRate = sampleRate
        self.params = params
        recomputeCoefficients()
        self.gain = 1.0
    }

    private func recomputeCoefficients() {
        envAttackCoeff = onePoleCoeff(ms: params.envAttackMs)
        envReleaseCoeff = onePoleCoeff(ms: params.envReleaseMs)
        gainAttackCoeff = onePoleCoeff(ms: params.attackMs)
        gainReleaseCoeff = onePoleCoeff(ms: params.releaseMs)
        holdLengthSamples = Int(params.holdMs * 0.001 * sampleRate)
        floorLinear = pow(10.0, params.floorDB / 20.0)
    }

    private func onePoleCoeff(ms: Double) -> Double {
        let t = max(0.0001, ms * 0.001)
        return exp(-1.0 / (t * sampleRate))
    }

    /// Process a hop in place.
    func process(_ block: inout [Double]) {
        guard enabled else { return }
        let thrLin = pow(10.0, params.thresholdDB / 20.0)
        let rm1 = params.ratio - 1.0

        for i in 0..<block.count {
            let absX = abs(block[i])
            // Peak follower: fast follow up on attack, slow decay on release.
            if absX > env {
                env = envAttackCoeff * env + (1 - envAttackCoeff) * absX
            } else {
                env = envReleaseCoeff * env + (1 - envReleaseCoeff) * absX
            }

            // Target gain from expander curve (only attenuates below threshold).
            var targetGain: Double = 1.0
            if env < thrLin {
                // gain_dB = (ratio - 1) * (env_dB - threshold_dB)
                let envDB = 20.0 * log10(env + 1e-12)
                let gainDB = rm1 * (envDB - params.thresholdDB)
                targetGain = pow(10.0, gainDB / 20.0)
                if targetGain < floorLinear { targetGain = floorLinear }
            }

            // Hold: once the gate is open (gain ~ 1), keep it open for holdMs
            // after the envelope last crossed threshold.
            if env >= thrLin {
                holdSamples = holdLengthSamples
                targetGain = 1.0
            } else if holdSamples > 0 {
                holdSamples -= 1
                targetGain = 1.0
            }

            // Asymmetric smoothing on gain — fast open, slow close.
            let coeff = (targetGain > gain) ? gainAttackCoeff : gainReleaseCoeff
            gain = coeff * gain + (1 - coeff) * targetGain

            block[i] *= gain
        }
    }

    /// Current gain reduction in dB (negative number; 0 means fully open).
    var gainReductionDB: Double { 20.0 * log10(max(gain, 1e-6)) }
}
