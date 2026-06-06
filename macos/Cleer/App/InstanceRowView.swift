import SwiftUI

/// One card per instance: device + channel assignment, the three stage toggles,
/// run control and live meters.
struct InstanceRowView: View {
    @EnvironmentObject var manager: InstanceManager
    @ObservedObject var instance: ProcessingInstance

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            routing
            stages
            meters
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.gray.opacity(0.2)))
    }

    private var header: some View {
        HStack {
            TextField("Name", text: $instance.name).textFieldStyle(.plain).font(.headline)
            Spacer()
            Circle().fill(instance.isRunning ? .green : .gray).frame(width: 8, height: 8)
            Button(instance.isRunning ? "Stop" : "Start") {
                instance.isRunning ? instance.stop() : instance.start()
            }
            Button(role: .destructive) { manager.remove(instance) } label: {
                Image(systemName: "trash")
            }
        }
    }

    private var routing: some View {
        HStack(spacing: 12) {
            Picker("Input", selection: $instance.inputDevice) {
                Text("—").tag(AudioDevice?.none)
                ForEach(manager.inputDevices) { dev in
                    Text(dev.name).tag(AudioDevice?.some(dev))
                }
            }
            Picker("Channel", selection: $instance.inputChannel) {
                ForEach(0..<max(1, instance.inputDevice?.inputChannels ?? 1), id: \.self) { c in
                    Text("Ch \(c + 1)").tag(c)
                }
            }
            .frame(width: 110)
            Picker("Output", selection: $instance.outputDevice) {
                Text("—").tag(AudioDevice?.none)
                ForEach(manager.outputDevices) { dev in
                    Text(dev.name).tag(AudioDevice?.some(dev))
                }
            }
        }
    }

    private var stages: some View {
        HStack(spacing: 16) {
            Toggle("Feedback", isOn: $instance.feedbackEnabled)
            Toggle("Denoise", isOn: $instance.denoiseEnabled)
            Toggle("Neural", isOn: $instance.useNeuralDenoise)
                .disabled(!instance.neuralAvailable || !instance.denoiseEnabled)
                .help(instance.neuralAvailable
                      ? "Use the CoreML mask (ANE) instead of the classical denoiser"
                      : "CoreML model not loaded")
            Toggle("Dereverb", isOn: $instance.dereverbEnabled)
            Spacer()
            Label("\(instance.activeNotches) notches", systemImage: "scissors")
                .foregroundStyle(.secondary).font(.caption)
        }
        .toggleStyle(.switch)
    }

    private var meters: some View {
        HStack(spacing: 12) {
            LevelMeter(label: "in", level: instance.inputLevel)
            LevelMeter(label: "out", level: instance.outputLevel)
        }
    }
}

struct LevelMeter: View {
    let label: String
    let level: Float
    var body: some View {
        HStack(spacing: 6) {
            Text(label).font(.caption2).foregroundStyle(.secondary).frame(width: 22, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3).fill(Color.gray.opacity(0.2))
                    RoundedRectangle(cornerRadius: 3)
                        .fill(level > 0.9 ? Color.red : Color.green)
                        .frame(width: geo.size.width * CGFloat(min(1, level)))
                }
            }
            .frame(height: 8)
        }
    }
}
