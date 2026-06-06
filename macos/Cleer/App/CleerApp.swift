import SwiftUI

@main
struct CleerApp: App {
    @StateObject private var manager = InstanceManager()

    var body: some Scene {
        WindowGroup("Cleer — De-Feedback") {
            ContentView()
                .environmentObject(manager)
                .frame(minWidth: 720, minHeight: 480)
        }
        .windowStyle(.titleBar)
    }
}
