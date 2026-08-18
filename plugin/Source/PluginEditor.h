#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"

// v1 editor: the stock generic parameter panel plus a live status strip
// (active notch count + expander gain reduction). A custom UI can replace
// this without touching the processor.
class CleerAudioProcessorEditor : public juce::AudioProcessorEditor,
                                  private juce::Timer
{
public:
    explicit CleerAudioProcessorEditor (CleerAudioProcessor& p)
        : AudioProcessorEditor (&p), processor (p), panel (p)
    {
        addAndMakeVisible (panel);
        addAndMakeVisible (status);
        status.setJustificationType (juce::Justification::centredLeft);
        setSize (420, 460);
        startTimerHz (10);
    }

    void resized() override
    {
        auto area = getLocalBounds();
        status.setBounds (area.removeFromBottom (26).reduced (8, 2));
        panel.setBounds (area);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    }

private:
    void timerCallback() override
    {
        const int notches = processor.uiNotchCount.load (std::memory_order_relaxed);
        const float gr = processor.uiGainReductionDB.load (std::memory_order_relaxed);
        status.setText (juce::String ("notches: ") + juce::String (notches)
                            + "    gate: " + juce::String (gr, 1) + " dB",
                        juce::dontSendNotification);
    }

    CleerAudioProcessor& processor;
    juce::GenericAudioProcessorEditor panel;
    juce::Label status;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CleerAudioProcessorEditor)
};
