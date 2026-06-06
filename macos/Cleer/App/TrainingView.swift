import SwiftUI

/// "Train in the app" panel. Pick the device to sample, capture a few seconds of
/// room noise (stay silent), then fine-tune the CoreML denoiser on it. The
/// personalised model is applied to all instances when done.
struct TrainingView: View {
    @EnvironmentObject var manager: InstanceManager
    @ObservedObject var trainer: ModelTrainer
    @Environment(\.dismiss) private var dismiss

    @State private var device: AudioDevice?
    @State private var channel = 0
    @State private var seconds = 4.0

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Personalise denoiser").font(.title2.bold())
            Text("Capture a few seconds of your room/mic noise (HVAC, fans, hum) "
                 + "while staying silent. Cleer fine-tunes the neural mask to it on "
                 + "this device — nothing leaves your Mac.")
                .foregroundStyle(.secondary)

            HStack {
                Picker("Sample from", selection: $device) {
                    Text("—").tag(AudioDevice?.none)
                    ForEach(manager.inputDevices) { d in Text(d.name).tag(AudioDevice?.some(d)) }
                }
                Picker("Channel", selection: $channel) {
                    ForEach(0..<max(1, device?.inputChannels ?? 1), id: \.self) { c in
                        Text("Ch \(c + 1)").tag(c)
                    }
                }.frame(width: 110)
            }

            HStack {
                Text("Duration")
                Slider(value: $seconds, in: 2...10, step: 1)
                Text("\(Int(seconds))s").monospacedDigit()
            }

            if trainer.isTraining {
                ProgressView(value: trainer.progress)
            }
            Text(trainer.status).font(.callout).foregroundStyle(.secondary)

            HStack {
                if trainer.hasPersonalizedModel {
                    Label("Personalised model active", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green).font(.caption)
                }
                Spacer()
                Button("Close") { dismiss() }
                Button {
                    if let d = device { manager.captureAndTrain(device: d, channel: channel, seconds: seconds) }
                } label: {
                    Label("Capture & train", systemImage: "waveform.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .disabled(device == nil || trainer.isTraining)
            }
        }
        .padding(20)
        .frame(width: 480)
        .onAppear { device = manager.inputDevices.first }
    }
}
