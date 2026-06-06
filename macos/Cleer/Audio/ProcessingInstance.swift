import Foundation
import AVFoundation
import CoreAudio
import Combine

/// One running "instance": capture from a chosen input device + channel, run
/// the VoicePipeline, and play to a chosen output device.
///
/// Routing strategy (public-API, compiles cleanly):
///   * a dedicated AVAudioEngine per instance, with its input/output HAL units
///     pointed at the selected devices,
///   * an input tap that pulls the selected channel, runs it through the DSP
///     `BlockChunker`, and pushes processed samples into a ring buffer,
///   * an AVAudioSourceNode that drains the ring buffer to the output.
///
/// NOTE: this is the layer I could not compile-test on Linux. The DSP it drives
/// is validated by the Python reference; expect to do a little on-device tuning
/// here (tap buffer size, clock-drift handling). For lowest latency the tap can
/// later be replaced by a V3 AUAudioUnit render block — see README.
final class ProcessingInstance: ObservableObject, Identifiable {
    let id = UUID()

    @Published var name: String
    @Published var inputDevice: AudioDevice? { didSet { if isRunning { restart() } } }
    @Published var inputChannel: Int = 0
    @Published var outputDevice: AudioDevice? { didSet { if isRunning { restart() } } }
    @Published var isRunning = false
    @Published var inputLevel: Float = 0      // for the meter
    @Published var outputLevel: Float = 0
    @Published var activeNotches: Int = 0

    @Published var feedbackEnabled = true { didSet { pipeline.feedbackEnabled = feedbackEnabled } }
    @Published var denoiseEnabled = true { didSet { pipeline.denoiseEnabled = denoiseEnabled } }
    @Published var dereverbEnabled = true { didSet { pipeline.dereverbEnabled = dereverbEnabled } }
    @Published var useNeuralDenoise = false { didSet { pipeline.useNeuralDenoise = useNeuralDenoise } }

    /// Whether a CoreML denoise model loaded for this instance.
    var neuralAvailable: Bool { pipeline.neuralAvailable }

    /// Hot-swap a personalised (on-device trained) model into this instance.
    func applyPersonalizedModel(_ url: URL) { pipeline.neural?.reload(from: url) }

    private let pipeline: VoicePipeline
    private let chunker: BlockChunker
    private var engine: AVAudioEngine?
    private var sourceNode: AVAudioSourceNode?

    // Ring buffer of processed samples awaiting the output node.
    private var ring = [Float](repeating: 0, count: 1 << 16)
    private var writeIdx = 0
    private var readIdx = 0
    private let ringLock = NSLock()

    init(name: String, sampleRate: Double = 48000) {
        self.name = name
        self.pipeline = VoicePipeline(sampleRate: sampleRate)
        self.chunker = BlockChunker(pipeline: pipeline)
    }

    func start() {
        guard let input = inputDevice else { return }
        let engine = AVAudioEngine()
        self.engine = engine

        setDevice(engine.inputNode.audioUnit, deviceID: input.id)
        if let output = outputDevice { setDevice(engine.outputNode.audioUnit, deviceID: output.id) }

        let inputFormat = engine.inputNode.inputFormat(forBus: 0)
        let channels = Int(inputFormat.channelCount)
        let ch = min(inputChannel, max(0, channels - 1))

        // Capture -> DSP -> ring buffer.
        engine.inputNode.installTap(onBus: 0, bufferSize: 256, format: inputFormat) { [weak self] buf, _ in
            guard let self = self, let data = buf.floatChannelData else { return }
            let n = Int(buf.frameLength)
            var mono = [Float](repeating: 0, count: n)
            let src = data[ch]
            var peak: Float = 0
            for i in 0..<n { mono[i] = src[i]; peak = max(peak, abs(src[i])) }
            let processed = self.chunker.process(mono)
            self.pushRing(processed)
            DispatchQueue.main.async {
                self.inputLevel = peak
                self.activeNotches = self.pipeline.feedback.activeNotchCount
            }
        }

        // Ring buffer -> output.
        let outFormat = engine.outputNode.inputFormat(forBus: 0)
        let source = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let n = Int(frameCount)
            let samples = self.popRing(n)
            for buffer in abl {
                let ptr = buffer.mData!.assumingMemoryBound(to: Float.self)
                for i in 0..<n { ptr[i] = samples[i] }
            }
            return noErr
        }
        self.sourceNode = source
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: outFormat)

        do {
            try engine.start()
            isRunning = true
        } catch {
            NSLog("Cleer: engine start failed: \(error)")
            self.engine = nil
        }
    }

    func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        sourceNode = nil
        isRunning = false
    }

    private func restart() { stop(); start() }

    // MARK: - Ring buffer
    private func pushRing(_ samples: [Float]) {
        ringLock.lock(); defer { ringLock.unlock() }
        for s in samples {
            ring[writeIdx % ring.count] = s
            writeIdx += 1
        }
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
