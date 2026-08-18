import Foundation
import AVFoundation
import CoreAudio
import Combine

/// One running "instance": capture from a chosen input device + channel, run
/// the VoicePipeline, and play to a chosen output device — which may be a
/// *different* audio interface from the input.
///
/// Routing strategy (public-API, compiles cleanly):
///   * TWO AVAudioEngines so input and output can live on different interfaces
///     with independent hardware clocks: a capture engine pinned to the input
///     device, and a playback engine pinned to the output device.
///   * The capture engine's input tap pulls the selected channel, runs it
///     through the DSP `BlockChunker`, and writes to a ring buffer.
///   * The playback engine's AVAudioSourceNode drains the ring buffer.
///   * The ring buffer bridges the two clock domains; `popRing` does simple
///     drift correction so latency stays bounded if the two interfaces' clocks
///     run at slightly different rates. For glitch-free cross-interface use,
///     macOS users can also make an Aggregate Device (CoreAudio then does
///     sample-accurate drift correction for us) — see `crossDevice`.
///
/// NOTE: this is the layer I could not compile-test on Linux. The DSP it drives
/// is validated by the Python reference; expect a little on-device tuning here.
/// For lowest latency the tap can later be replaced by a V3 AUAudioUnit render
/// block — see README.
final class ProcessingInstance: ObservableObject, Identifiable {
    let id = UUID()

    @Published var name: String
    @Published var inputDevice: AudioDevice? { didSet { if isRunning { restart() } } }
    @Published var inputChannel: Int = 0 { didSet { if isRunning { restart() } } }
    @Published var outputDevice: AudioDevice? { didSet { if isRunning { restart() } } }
    @Published var isRunning = false
    @Published var inputLevel: Float = 0      // for the meter
    @Published var outputLevel: Float = 0
    @Published var activeNotches: Int = 0

    /// Master bypass: when on, audio passes through untouched (raw monitor) so
    /// you can A/B the processing.
    @Published var bypassed = false

    @Published var feedbackEnabled = true { didSet { pipeline.feedbackEnabled = feedbackEnabled } }
    @Published var gateEnabled = true { didSet { pipeline.gateEnabled = gateEnabled } }
    @Published var gateThresholdDB: Double = -42.0 { didSet { pipeline.expander.params.thresholdDB = gateThresholdDB } }
    @Published var gateRatio: Double = 3.0 { didSet { pipeline.expander.params.ratio = gateRatio } }

    /// True when input and output are on different interfaces (cross-clock).
    var crossDevice: Bool {
        guard let i = inputDevice, let o = outputDevice else { return false }
        return i.id != o.id
    }

    private let pipeline: VoicePipeline
    private let chunker: BlockChunker
    private var captureEngine: AVAudioEngine?
    private var playbackEngine: AVAudioEngine?
    private var sourceNode: AVAudioSourceNode?

    // Ring buffer bridging the capture and playback clock domains.
    private var ring = [Float](repeating: 0, count: 1 << 16)
    private var writeIdx = 0
    private var readIdx = 0
    // os_unfair_lock: bounded-priority, real-time-safe (NSLock is not — it can
    // cause priority inversion on the audio render thread → underruns/stutter).
    private var ringLock = os_unfair_lock_s()
    // Target fill (~85 ms) the drift corrector tries to hold. Bigger than the
    // strict minimum so a one-off scheduling spike (a frame the DSP took a
    // little longer on, a Bluetooth radio re-poll, etc.) can't underrun.
    private let targetFill = 4096
    // Whether to actively drift-correct. Only true when input and output are on
    // DIFFERENT interfaces (independent hardware clocks). Same interface = one
    // clock, no drift; correcting in that case just chases sub-sample jitter
    // and produces audible pulsing at the correction cadence.
    private var driftCorrect = false
    // Don't let the playback engine drain the ring until it has been filled to
    // `targetFill` at least once. Without this, on start playback pulls before
    // capture has produced anything → zero-fill underrun → stutter cascade.
    private var primed = false

    // Meter update throttle. Dispatching to main on every audio callback floods
    // the main queue; we only need ~30 Hz for a UI meter.
    private var meterSampleCounter = 0
    private let meterSampleInterval = 1600   // ~30 Hz at 48 kHz

    init(name: String, sampleRate: Double = 48000) {
        self.name = name
        self.pipeline = VoicePipeline(sampleRate: sampleRate)
        self.chunker = BlockChunker(pipeline: pipeline)
    }

    func start() {
        guard let input = inputDevice else { return }
        resetRing()
        // Only drift-correct when in/out are on different interfaces. Same
        // device = same hardware clock, no drift to correct — and trying to
        // correct sub-sample scheduling jitter creates audible pulsing.
        driftCorrect = crossDevice

        // --- Capture engine (input interface) ---------------------------------
        let capture = AVAudioEngine()
        self.captureEngine = capture
        setDevice(capture.inputNode.audioUnit, deviceID: input.id)

        let inputFormat = capture.inputNode.inputFormat(forBus: 0)
        let channels = Int(inputFormat.channelCount)
        let ch = min(inputChannel, max(0, channels - 1))

        // 2048 frames (~43 ms @ 48k) — observed CPU was hitting 149% (DSP
        // running over the wall-clock budget) and CoreAudio was logging
        // "skipping cycle due to overload". Doubling the per-block budget +
        // killing per-callback allocations in BlockChunker pulls us back under.
        capture.inputNode.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buf, _ in
            guard let self = self, let data = buf.floatChannelData else { return }
            let n = Int(buf.frameLength)
            var mono = [Float](repeating: 0, count: n)
            let src = data[ch]
            var peak: Float = 0
            for i in 0..<n { mono[i] = src[i]; peak = max(peak, abs(src[i])) }
            let out = self.bypassed ? mono : self.chunker.process(mono)
            self.pushRing(out)
            // Throttle the UI meter — flooding the main queue at audio rate
            // serialises against the rest of the UI and can starve the render.
            self.meterSampleCounter += n
            if self.meterSampleCounter >= self.meterSampleInterval {
                self.meterSampleCounter = 0
                let notches = self.bypassed ? 0 : self.pipeline.feedback.activeNotchCount
                DispatchQueue.main.async {
                    self.inputLevel = peak
                    self.activeNotches = notches
                }
            }
        }

        // --- Playback engine (output interface) -------------------------------
        let playback = AVAudioEngine()
        self.playbackEngine = playback
        if let output = outputDevice { setDevice(playback.outputNode.audioUnit, deviceID: output.id) }
        let outFormat = playback.outputNode.inputFormat(forBus: 0)

        // The source node will produce samples at the CAPTURE sample rate — we
        // don't resample in the chunker, we just buffer. If we lie to the
        // engine and declare the source as outFormat (output device rate), it
        // plays our 48 k samples at the device's e.g. 44.1 k rate → audible
        // pitch shift. Declaring inputFormat.sampleRate lets mainMixerNode
        // insert an SRC between us and the output device automatically.
        let sourceFormat = AVAudioFormat(
            standardFormatWithSampleRate: inputFormat.sampleRate,
            channels: outFormat.channelCount) ?? outFormat
        NSLog("Cleer: capture sr=\(inputFormat.sampleRate) ch=\(inputFormat.channelCount), "
            + "output sr=\(outFormat.sampleRate) ch=\(outFormat.channelCount), "
            + "source declared sr=\(sourceFormat.sampleRate)")

        var outMeterCounter = 0
        let source = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let n = Int(frameCount)
            let samples = self.popRing(n)
            var peak: Float = 0
            for buffer in abl {
                let ptr = buffer.mData!.assumingMemoryBound(to: Float.self)
                for i in 0..<n { ptr[i] = samples[i]; peak = max(peak, abs(samples[i])) }
            }
            // Same throttle as input meter — ~30 Hz UI updates, not audio rate.
            outMeterCounter += n
            if outMeterCounter >= self.meterSampleInterval {
                outMeterCounter = 0
                DispatchQueue.main.async { self.outputLevel = peak }
            }
            return noErr
        }
        self.sourceNode = source
        playback.attach(source)
        playback.connect(source, to: playback.mainMixerNode, format: sourceFormat)

        do {
            try playback.start()
            try capture.start()
            isRunning = true
        } catch {
            NSLog("Cleer: engine start failed: \(error)")
            stop()
        }
    }

    func stop() {
        captureEngine?.inputNode.removeTap(onBus: 0)
        captureEngine?.stop()
        playbackEngine?.stop()
        captureEngine = nil
        playbackEngine = nil
        sourceNode = nil
        isRunning = false
    }

    private func restart() { stop(); start() }

    // MARK: - Ring buffer (bridges the two clock domains)
    private func resetRing() {
        os_unfair_lock_lock(&ringLock)
        writeIdx = 0; readIdx = 0; primed = false
        os_unfair_lock_unlock(&ringLock)
    }

    private func pushRing(_ samples: [Float]) {
        os_unfair_lock_lock(&ringLock); defer { os_unfair_lock_unlock(&ringLock) }
        for s in samples {
            ring[writeIdx % ring.count] = s
            writeIdx += 1
        }
        let available = writeIdx - readIdx
        // Soft drift correction — ONLY when crossing clock domains.
        if driftCorrect && available > targetFill + 64 { readIdx += 1 }
        // Hard cap: if something is wildly off (device swap, paused output),
        // still don't let latency grow without bound. Always on.
        if available > targetFill * 4 { readIdx = writeIdx - targetFill }
        // Gate playback until the ring has reached targetFill at least once.
        if !primed && available >= targetFill { primed = true }
    }

    private func popRing(_ n: Int) -> [Float] {
        os_unfair_lock_lock(&ringLock); defer { os_unfair_lock_unlock(&ringLock) }
        var out = [Float](repeating: 0, count: n)
        // Pre-prime: emit silence until we have a healthy buffer to drain from.
        guard primed else { return out }
        let available = writeIdx - readIdx
        let take = min(n, max(0, available))
        for i in 0..<take {
            out[i] = ring[readIdx % ring.count]
            readIdx += 1
        }
        // Soft drift correction on the underflow side — ONLY cross-device.
        if driftCorrect && take == n && (writeIdx - readIdx) < targetFill - 64 && readIdx > 0 {
            readIdx -= 1
        }
        // Underflow tail stays zero — a brief gap is far less audible than
        // re-priming and emitting 85 ms of silence on every underrun.
        return out
    }

    // MARK: - Device routing
    private func setDevice(_ audioUnit: AudioUnit?, deviceID: AudioDeviceID) {
        guard let au = audioUnit else { return }
        var dev = deviceID
        AudioUnitSetProperty(au,
                             kAudioOutputUnitProperty_CurrentDevice,
                             kAudioUnitScope_Global, 0,
                             &dev, UInt32(MemoryLayout<AudioDeviceID>.size))
    }
}
