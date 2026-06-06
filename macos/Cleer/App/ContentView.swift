import SwiftUI

struct ContentView: View {
    @EnvironmentObject var manager: InstanceManager

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
            Text("No instances yet").font(.title3)
            Text("Add an instance, assign it to an input device and channel.")
                .foregroundStyle(.secondary)
            Button("Add instance") { manager.addInstance() }
                .buttonStyle(.borderedProminent)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
