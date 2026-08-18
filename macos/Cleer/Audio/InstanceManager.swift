import Foundation
import Combine

/// Owns the list of processing instances the user has created. The "+" button
/// in the UI calls `addInstance`; each instance independently binds to its own
/// device + channel and runs its own pipeline.
final class InstanceManager: ObservableObject {
    @Published var instances: [ProcessingInstance] = []
    @Published var inputDevices: [AudioDevice] = []
    @Published var outputDevices: [AudioDevice] = []

    init() {
        refreshDevices()
    }

    func refreshDevices() {
        inputDevices = AudioDeviceManager.inputDevices()
        outputDevices = AudioDeviceManager.outputDevices()
    }

    @discardableResult
    func addInstance() -> ProcessingInstance {
        let inst = ProcessingInstance(name: "Instance \(instances.count + 1)")
        inst.inputDevice = inputDevices.first
        inst.outputDevice = outputDevices.first
        instances.append(inst)
        return inst
    }

    /// The simple case: mic in -> Mac speakers out, processed, started. Picks the
    /// default input and the built-in speakers and hits go.
    @discardableResult
    func quickStartMicToSpeakers() -> ProcessingInstance {
        let inst = ProcessingInstance(name: "Live vocal")
        inst.inputDevice = inputDevices.first
        inst.outputDevice = outputDevices.first { $0.name.localizedCaseInsensitiveContains("speaker")
            || $0.name.localizedCaseInsensitiveContains("built-in") } ?? outputDevices.first
        instances.append(inst)
        inst.start()
        return inst
    }

    func remove(_ instance: ProcessingInstance) {
        instance.stop()
        instances.removeAll { $0.id == instance.id }
    }

    func startAll() { instances.forEach { $0.start() } }
    func stopAll() { instances.forEach { $0.stop() } }
}
