#pragma once
#include <array>
#include <atomic>
#include <juce_audio_processors/juce_audio_processors.h>
#include "dsp/CleerEngine.h"

// De-feedback for live vocals as a VST3/AU: adaptive howl notching plus a
// downward expander, both time-domain, zero added latency. The DSP core in
// Source/dsp/ is framework-free; this class only maps parameters onto it.
class CleerAudioProcessor : public juce::AudioProcessor
{
public:
    CleerAudioProcessor();

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Cleer"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    // Shared state for the editor, written from the audio thread with relaxed
    // atomics (metering data — freshness matters, ordering doesn't).
    static constexpr int kSpecBins = 513;   // win 1024 / 2 + 1
    static constexpr int kMaxUiNotches = 16;

    std::atomic<int> uiNotchCount { 0 };
    std::atomic<float> uiGainReductionDB { 0.0f };
    std::atomic<float> uiInPeakDB { -100.0f };
    std::atomic<float> uiOutPeakDB { -100.0f };
    std::array<std::atomic<float>, kSpecBins> uiSpectrumDB {};
    std::array<std::atomic<float>, kMaxUiNotches> uiNotchFreq {};
    std::array<std::atomic<float>, kMaxUiNotches> uiNotchDepth {};
    std::atomic<int> uiNotchN { 0 };

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    cleer::Engine engine;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CleerAudioProcessor)
};
