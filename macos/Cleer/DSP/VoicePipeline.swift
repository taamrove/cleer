import Foundation

/// One instance's full DSP chain: feedback (time domain) -> STFT -> noise mask
/// x dereverb mask -> ISTFT. Swift port of `pipeline.py`.
///
/// Audio is processed `hop` samples at a time so it slots straight into a
/// CoreAudio render callback. The host feeds whatever buffer size CoreAudio
/// gives it through `BlockChunker`, which re-chunks to the DSP hop size.
final class VoicePipeline {
    let sampleRate: Double
    let win: Int
    let hop: Int

    let feedback: FeedbackSuppressor
    let noise: NoiseReducer
    let dereverb: Dereverb
    private let stft: STFT

    // Live toggles (bound to the UI).
    var feedbackEnabled = true
    var denoiseEnabled = true
    var dereverbEnabled = true

    init(sampleRate: Double = 48000, win: Int = 1024, hop: Int = 256) {
        self.sampleRate = sampleRate; self.win = win; self.hop = hop
        let bins = win / 2 + 1
        self.feedback = FeedbackSuppressor(sampleRate: sampleRate, win: win, hop: hop)
        self.noise = NoiseReducer(bins: bins)
        self.dereverb = Dereverb(bins: bins, sampleRate: sampleRate, hop: hop)
        self.stft = STFT(win: win, hop: hop)
    }

    /// Process exactly `hop` samples in place.
    func processHop(_ block: inout [Double]) {
        precondition(block.count == hop)
        feedback.enabled = feedbackEnabled
        if feedbackEnabled { feedback.process(&block) }

        guard denoiseEnabled || dereverbEnabled else { return }
        block = stft.process(block) { mags in
            var gain = [Double](repeating: 1.0, count: mags.count)
            if self.denoiseEnabled {
                let g = self.noise.mask(mags)
                for k in 0..<gain.count { gain[k] *= g[k] }
            }
            if self.dereverbEnabled {
                let g = self.dereverb.mask(mags)
                for k in 0..<gain.count { gain[k] *= g[k] }
            }
            return gain
        }
    }
}

/// Re-chunks arbitrary CoreAudio buffer sizes into fixed `hop`-sized blocks for
/// the pipeline, buffering the remainder between callbacks.
final class BlockChunker {
    private let hop: Int
    private var inBuf: [Double] = []
    private var outBuf: [Double] = []
    private let pipeline: VoicePipeline

    init(pipeline: VoicePipeline) {
        self.pipeline = pipeline
        self.hop = pipeline.hop
    }

    /// Feed `samples`, get the same number of processed samples back (delayed
    /// by up to one hop + the STFT latency).
    func process(_ samples: [Float]) -> [Float] {
        for s in samples { inBuf.append(Double(s)) }
        while inBuf.count >= hop {
            var block = Array(inBuf.prefix(hop))
            inBuf.removeFirst(hop)
            pipeline.processHop(&block)
            outBuf.append(contentsOf: block)
        }
        let n = samples.count
        if outBuf.count < n {
            // Prime with silence until the pipeline has produced enough.
            return [Float](repeating: 0, count: n)
        }
        let out = outBuf.prefix(n).map { Float($0) }
        outBuf.removeFirst(n)
        return Array(out)
    }
}
