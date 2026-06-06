import SwiftUI

struct ContentView: View {
    @EnvironmentObject var manager: InstanceManager
    @State private var showTraining = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            if manager.instances.isEmpty {
                emptyState
            } else {
                ScrollView {
                    VStack(spacing: 12) {
                        ForEach(manager.instances) { instance in
                            InstanceRowView(instance: instance)
                                .environmentObject(manager)
                        }
                    }
                    .padding()
                }
            }
        }
        .sheet(isPresented: $showTraining) {
            TrainingView(trainer: manager.trainer).environmentObject(manager)
        }
    }

    private var toolbar: some View {
        HStack {
            Text("Cleer").font(.headline)
            Spacer()
            Button {
                manager.refreshDevices()
            } label: { Label("Refresh devices", systemImage: "arrow.clockwise") }
            Button {
                manager.startAll()
            } label: { Label("Start all", systemImage: "play.fill") }
            Button {
                manager.stopAll()
            } label: { Label("Stop all", systemImage: "stop.fill") }
            Button {
                showTraining = true
            } label: { Label("Personalise", systemImage: "brain.head.profile") }
            Button {
                manager.addInstance()
            } label: { Label("Add instance", systemImage: "plus") }
                .keyboardShortcut("n", modifiers: .command)
        }
        .padding(8)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "waveform.badge.mic").font(.system(size: 48)).foregroundStyle(.secondary)
            Text("Live vocal → Mac speakers").font(.title3)
            Text("One click: take your mic, clean it up, play it out the Mac speakers.")
                .foregroundStyle(.secondary)
            Button {
                manager.quickStartMicToSpeakers()
            } label: { Label("Start mic → speakers", systemImage: "play.fill") }
                .buttonStyle(.borderedProminent)
            Text("Tip: on open speakers a live mic can ring — keep the volume moderate, "
                 + "or use headphones. (Suppressing that ring is exactly what Feedback does.)")
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 360)
            Button("Set up manually instead") { manager.addInstance() }
                .buttonStyle(.link)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
