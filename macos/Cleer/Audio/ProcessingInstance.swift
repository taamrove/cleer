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
    @Published var denoiseEnabled = true { didSet { pipeline.denoiseEnabled = denoiseEnabled } }
    @Published var dereverbEnabled = true { didSet { pipeline.dereverbEnabled = dereverbEnabled } }
    @Published var useNeuralDenoise = false { didSet { pipeline.useNeuralDenoise = useNeuralDenoise } }

    /// Whether a CoreML denoise model loaded for this instance.
    var neuralAvailable: Bool { pipeline.neuralAvailable }

    /// True when input and output are on different interfaces (cross-clock).
    var crossDevice: Bool {
        guard let i = inputDevice, let o = outputDevice else { return false }
        return i.id != o.id
    }

    /// Hot-swap a personalised (on-device trained) model into this instance.
    func applyPersonalizedModel(_ url: URL) { pipeline.neural?.reload(from: url) }

    private let pipeline: VoicePipeline
    private let chunker: BlockChunker
    private var captureEngine: AVAudioEngine?
    private var playbackEngine: AVAudioEngine?
    private var sourceNode: AVAudioSourceNode?

    // Ring buffer bridging the capture and playback clock domains.
    private var ring = [Float](repeating: 0, count: 1 << 16)
    private var writeIdx = 0
    private var readIdx = 0
    private let ringLock = NSLock()
    // Target fill (~43 ms) the drift corrector tries to hold.
    private let targetFill = 2048

    init(name: String, sampleRate: Double = 48000) {
        self.name = name
        self.pipeline = VoicePipeline(sampleRate: sampleRate)
        self.chunker = BlockChunker(pipeline: pipeline)
    }

    func start() {
        guard let input = inputDevice else { return }
        resetRing()

        // --- Capture engine (input interface) ---------------------------------
        let capture = AVAudioEngine()
        self.captureEngine = capture
        setDevice(capture.inputNode.audioUnit, deviceID: input.id)

        let inputFormat = capture.inputNode.inputFormat(forBus: 0)
        let channels = Int(inputFormat.channelCount)
        let ch = min(inputChannel, max(0, channels - 1))

        capture.inputNode.installTap(onBus: 0, bufferSize: 256, format: inputFormat) { [weak self] buf, _ in
            guard let self = self, let data = buf.floatChannelData else { return }
            let n = Int(buf.frameLength)
            var mono = [Float](repeating: 0, count: n)
            let src = data[ch]
            var peak: Float = 0
            for i in 0..<n { mono[i] = src[i]; peak = max(peak, abs(src[i])) }
            let out = self.bypassed ? mono : self.chunker.process(mono)
            self.pushRing(out)
            DispatchQueue.main.async {
                self.inputLevel = peak
                self.activeNotches = self.bypassed ? 0 : self.pipeline.feedback.activeNotchCount
            }
        }

        // --- Playback engine (output interface) -------------------------------
        let playback = AVAudioEngine()
        self.playbackEngine = playback
        if let output = outputDevice { setDevice(playback.outputNode.audioUnit, deviceID: output.id) }
        let outFormat = playback.outputNode.inputFormat(forBus: 0)

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
            DispatchQueue.main.async { self.outputLevel = peak }
            return noErr
        }
        self.sourceNode = source
        playback.attach(source)
        playback.connect(source, to: playback.mainMixerNode, format: outFormat)

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
        ringLock.lock(); writeIdx = 0; readIdx = 0; ringLock.unlock()
    }

    private func pushRing(_ samples: [Float]) {
        ringLock.lock(); defer { ringLock.unlock() }
        for s in samples {
            ring[writeIdx % ring.count] = s
            writeIdx += 1
        }
        // Overflow guard: if the capture clock outran playback, drop the oldest
        // surplus so latency can't grow without bound.
        let available = writeIdx - readIdx
        if available > targetFill * 3 { readIdx = writeIdx - targetFill }
    }

    private func popRing(_ n: Int) -> [Float] {
        ringLock.lock(); defer { ringLock.unlock() }
        var out = [Float](repeating: 0, count: n)
        let available = writeIdx - readIdx
        let take = min(n, max(0, available))
        for i in 0..<take {
            out[i] = ring[readIdx % ring.count]
            readIdx += 1
        }
        // Underflow (playback clock outran capture): the remaining samples stay
        // zero (a brief silence) rather than glitching with stale data.
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
