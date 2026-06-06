import Foundation
import Combine

/// Owns the list of processing instances the user has created. The "+" button
/// in the UI calls `addInstance`; each instance independently binds to its own
/// device + channel and runs its own pipeline.
final class InstanceManager: ObservableObject {
    @Published var instances: [ProcessingInstance] = []
    @Published var inputDevices: [AudioDevice] = []
    @Published var outputDevices: [AudioDevice] = []

    init() { refreshDevices() }

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

    func remove(_ instance: ProcessingInstance) {
        instance.stop()
        instances.removeAll { $0.id == instance.id }
    }

    func startAll() { instances.forEach { $0.start() } }
    func stopAll() { instances.forEach { $0.stop() } }
}
