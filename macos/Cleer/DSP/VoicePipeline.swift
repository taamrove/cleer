import Foundation

/// Live vocal cleanup chain: feedback suppression -> expander/gate.
/// Both stages are time-domain and operate per hop. No STFT round-trip in the
/// audio path (feedback detection uses an internal STFT for analysis only;
/// actuation is biquads). Aim: minimum CPU, minimum latency, no spectral
/// artefacts, no harmonic damage from over-aggressive denoise/dereverb.
final class VoicePipeline {
    let sampleRate: Double
    let win: Int
    let hop: Int

    let feedback: FeedbackSuppressor
    let expander: Expander

    // Live toggles (bound to the UI).
    var feedbackEnabled = true
    var gateEnabled = true

    init(sampleRate: Double = 48000, win: Int = 1024, hop: Int = 256) {
        self.sampleRate = sampleRate; self.win = win; self.hop = hop
        self.feedback = FeedbackSuppressor(sampleRate: sampleRate, win: win, hop: hop)
        self.expander = Expander(sampleRate: sampleRate)
    }

    /// Process exactly `hop` samples in place.
    func processHop(_ block: inout [Double]) {
        precondition(block.count == hop)
        feedback.enabled = feedbackEnabled
        if feedbackEnabled { feedback.process(&block) }
        expander.enabled = gateEnabled
        if gateEnabled { expander.process(&block) }
    }
}

/// Re-chunks arbitrary CoreAudio buffer sizes into fixed `hop`-sized blocks for
/// the pipeline, buffering the remainder between callbacks.
///
/// Allocation-free on the audio thread: input and output buffers are
/// pre-sized; the output side is a ring with read/write indices so there's
/// never an O(n) `removeFirst` shift. The input side keeps a running
/// `inCount`, with a bounded shift only when we drain (at most one hop's worth
/// of remainder, never the whole buffer).
final class BlockChunker {
    private let hop: Int
    private let pipeline: VoicePipeline

    // Input accumulator: valid samples occupy inBuf[0..<inCount].
    private var inBuf: [Double]
    private var inCount = 0

    // Output ring: outBuf[outRead..<outWrite] (mod capacity) is valid.
    private var outBuf: [Double]
    private let outCap: Int
    private var outWrite = 0
    private var outRead = 0

    // Scratch reused every hop (no per-call alloc).
    private var hopScratch: [Double]

    init(pipeline: VoicePipeline) {
        self.pipeline = pipeline
        self.hop = pipeline.hop
        // Generous headroom — we never grow at runtime.
        self.inBuf = [Double](repeating: 0, count: hop * 32)
        self.outCap = hop * 32
        self.outBuf = [Double](repeating: 0, count: outCap)
        self.hopScratch = [Double](repeating: 0, count: hop)
    }

    /// Feed `samples`, get the same number of processed samples back (delayed
    /// by up to one hop + the STFT latency).
    func process(_ samples: [Float]) -> [Float] {
        let n = samples.count

        // Append to inBuf (bounded; if we'd overrun the pre-sized buffer, drop
        // the oldest accumulated samples — should never happen in normal ops
        // because we drain to hops below, but this prevents OOB writes if a
        // hostile callback hands us a huge buffer).
        if inCount + n > inBuf.count {
            let drop = inCount + n - inBuf.count
            for i in 0..<(inCount - drop) { inBuf[i] = inBuf[i + drop] }
            inCount -= drop
        }
        for i in 0..<n { inBuf[inCount + i] = Double(samples[i]) }
        inCount += n

        // Drain hop-sized blocks.
        var consumed = 0
        while inCount - consumed >= hop {
            for i in 0..<hop { hopScratch[i] = inBuf[consumed + i] }
            pipeline.processHop(&hopScratch)
            for i in 0..<hop {
                outBuf[(outWrite + i) % outCap] = hopScratch[i]
            }
            outWrite += hop
            consumed += hop
        }
        // Shift the (bounded) remainder back to inBuf[0]. < hop samples => O(hop).
        if consumed > 0 {
            let remaining = inCount - consumed
            for i in 0..<remaining { inBuf[i] = inBuf[consumed + i] }
            inCount = remaining
        }

        // Drain n samples from the output ring (zero-fill on prime).
        var out = [Float](repeating: 0, count: n)
        let avail = outWrite - outRead
        let take = min(n, avail)
        for i in 0..<take {
            out[i] = Float(outBuf[(outRead + i) % outCap])
        }
        outRead += take
        return out
    }
}
