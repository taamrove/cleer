import Foundation
import CoreAudio

/// A CoreAudio device with its channel counts.
struct AudioDevice: Identifiable, Hashable {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let inputChannels: Int
    let outputChannels: Int
}

/// Enumerates CoreAudio HAL devices so the UI can offer per-instance device and
/// channel pickers. Read-only; the actual routing happens in ProcessingInstance.
enum AudioDeviceManager {
    static func allDevices() -> [AudioDevice] {
        var size = UInt32(0)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size)
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)

        return ids.compactMap { id in
            let name = stringProperty(id, kAudioObjectPropertyName) ?? "Device \(id)"
            let uid = stringProperty(id, kAudioDevicePropertyDeviceUID) ?? "\(id)"
            let ins = channelCount(id, scope: kAudioObjectPropertyScopeInput)
            let outs = channelCount(id, scope: kAudioObjectPropertyScopeOutput)
            guard ins > 0 || outs > 0 else { return nil }
            return AudioDevice(id: id, uid: uid, name: name, inputChannels: ins, outputChannels: outs)
        }
    }

    static func inputDevices() -> [AudioDevice] { allDevices().filter { $0.inputChannels > 0 } }
    static func outputDevices() -> [AudioDevice] { allDevices().filter { $0.outputChannels > 0 } }

    private static func channelCount(_ id: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: scope, mElement: kAudioObjectPropertyElementMain)
        var size = UInt32(0)
        guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let bufList = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
        defer { bufList.deallocate() }
        guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, bufList) == noErr else { return 0 }
        let abl = bufList.assumingMemoryBound(to: AudioBufferList.self)
        let buffers = UnsafeMutableAudioBufferListPointer(abl)
        return buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    private static func stringProperty(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var size = UInt32(MemoryLayout<CFString?>.size)
        var cf: CFString? = nil
        let status = withUnsafeMutablePointer(to: &cf) {
            AudioObjectGetPropertyData(id, &addr, 0, nil, &size, $0)
        }
        guard status == noErr, let s = cf else { return nil }
        return s as String
    }
}
