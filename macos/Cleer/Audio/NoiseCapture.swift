import Foundation
import AVFoundation
import CoreAudio

/// Records a few seconds of mono audio from a chosen device — used to grab the
/// user's room/mic noise for on-device personalisation. Have the user stay
/// silent while this runs so the captured audio is pure noise.
final class NoiseCapture {
    private var engine: AVAudioEngine?

    func capture(device: AudioDevice, channel: Int, seconds: Double,
                 completion: @escaping ([Float]) -> Void) {
        let engine = AVAudioEngine()
        self.engine = engine
        if let au = engine.inputNode.audioUnit {
            var dev = device.id
            AudioUnitSetProperty(au, kAudioOutputUnitProperty_CurrentDevice,
                                 kAudioUnitScope_Global, 0, &dev,
                                 UInt32(MemoryLayout<AudioDeviceID>.size))
        }
        let format = engine.inputNode.inputFormat(forBus: 0)
        let ch = min(channel, max(0, Int(format.channelCount) - 1))
        let target = Int(format.sampleRate * seconds)
        var collected = [Float]()
        collected.reserveCapacity(target)

        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buf, _ in
            guard let data = buf.floatChannelData else { return }
            let n = Int(buf.frameLength)
            let src = data[ch]
            for i in 0..<n where collected.count < target { collected.append(src[i]) }
            if collected.count >= target {
                self?.stop()
                DispatchQueue.main.async { completion(collected) }
            }
        }
        do { try engine.start() } catch { completion([]) }
    }

    func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
    }
}
