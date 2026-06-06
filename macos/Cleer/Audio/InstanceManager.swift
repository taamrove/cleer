import Foundation
import Combine

/// Owns the list of processing instances the user has created. The "+" button
/// in the UI calls `addInstance`; each instance independently binds to its own
/// device + channel and runs its own pipeline.
final class InstanceManager: ObservableObject {
    @Published var instances: [ProcessingInstance] = []
    @Published var inputDevices: [AudioDevice] = []
    @Published var outputDevices: [AudioDevice] = []

    let trainer = ModelTrainer()
    private let capture = NoiseCapture()

    init() {
        refreshDevices()
        // If a personalised model already exists from a previous session, it is
        // applied to instances as they start (see ProcessingInstance).
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

    func remove(_ instance: ProcessingInstance) {
        instance.stop()
        instances.removeAll { $0.id == instance.id }
    }

    func startAll() { instances.forEach { $0.start() } }
    func stopAll() { instances.forEach { $0.stop() } }

    /// Capture room noise from a device, fine-tune the model on it, then apply
    /// the personalised model to every instance and switch them to neural mode.
    func captureAndTrain(device: AudioDevice, channel: Int, seconds: Double = 4.0) {
        trainer.status = "Capturing \(Int(seconds))s of room noise — stay silent…"
        capture.capture(device: device, channel: channel, seconds: seconds) { [weak self] noise in
            guard let self = self, !noise.isEmpty else {
                self?.trainer.status = "Capture failed"; return
            }
            self.trainer.personalize(noise: noise) { url in
                guard let url = url else { return }
                for inst in self.instances {
                    inst.applyPersonalizedModel(url)
                    inst.useNeuralDenoise = true
                }
            }
        }
    }

    /// Apply the most recent personalised model (if any) to all instances.
    func applyPersonalizedModelIfAvailable() {
        let url = ModelTrainer.personalizedURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        instances.forEach { $0.applyPersonalizedModel(url) }
    }
}
