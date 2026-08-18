#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace
{
// Parameter IDs, versioned for host automation compatibility.
constexpr auto kFbOn        = "fb_on";
constexpr auto kFbSens      = "fb_sens";      // prominence threshold, dB above floor
constexpr auto kFbDepth     = "fb_depth";     // max notch depth, dB
constexpr auto kFbQ         = "fb_q";
constexpr auto kNoiseOn     = "noise_on";
constexpr auto kNoiseAmount = "noise_amount";
constexpr auto kGateOn      = "gate_on";
constexpr auto kGateThresh  = "gate_thresh";
constexpr auto kGateRatio   = "gate_ratio";
constexpr auto kGateRange   = "gate_range";   // max attenuation, dB (maps to -floorDB)
constexpr auto kGateAttack  = "gate_attack";
constexpr auto kGateRelease = "gate_release";
constexpr auto kGateHold    = "gate_hold";
} // namespace

CleerAudioProcessor::CleerAudioProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMS", createLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout CleerAudioProcessor::createLayout()
{
    using FloatParam = juce::AudioParameterFloat;
    using BoolParam = juce::AudioParameterBool;

    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<BoolParam> (juce::ParameterID { kFbOn, 1 }, "Feedback Kill", true));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kFbSens, 1 }, "FB Sensitivity",
        juce::NormalisableRange<float> (10.0f, 30.0f, 0.5f), 18.0f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kFbDepth, 1 }, "FB Max Depth",
        juce::NormalisableRange<float> (6.0f, 36.0f, 1.0f), 24.0f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kFbQ, 1 }, "FB Notch Q",
        juce::NormalisableRange<float> (8.0f, 40.0f, 1.0f), 28.0f));

    layout.add (std::make_unique<BoolParam> (juce::ParameterID { kNoiseOn, 1 }, "Noise Removal", true));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kNoiseAmount, 1 }, "Noise Amount",
        juce::NormalisableRange<float> (0.0f, 100.0f, 1.0f), 90.0f,
        juce::AudioParameterFloatAttributes().withLabel ("%")));

    layout.add (std::make_unique<BoolParam> (juce::ParameterID { kGateOn, 1 }, "Gate", true));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kGateThresh, 1 }, "Gate Threshold",
        juce::NormalisableRange<float> (-70.0f, -20.0f, 0.5f), -42.0f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kGateRatio, 1 }, "Gate Ratio",
        juce::NormalisableRange<float> (1.0f, 10.0f, 0.1f), 3.0f));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kGateRange, 1 }, "Gate Range",
        juce::NormalisableRange<float> (6.0f, 40.0f, 1.0f), 24.0f,
        juce::AudioParameterFloatAttributes().withLabel ("dB")));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kGateAttack, 1 }, "Gate Attack",
        juce::NormalisableRange<float> (0.5f, 20.0f, 0.1f, 0.5f), 2.0f,
        juce::AudioParameterFloatAttributes().withLabel ("ms")));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kGateRelease, 1 }, "Gate Release",
        juce::NormalisableRange<float> (20.0f, 500.0f, 1.0f, 0.5f), 150.0f,
        juce::AudioParameterFloatAttributes().withLabel ("ms")));
    layout.add (std::make_unique<FloatParam> (
        juce::ParameterID { kGateHold, 1 }, "Gate Hold",
        juce::NormalisableRange<float> (0.0f, 200.0f, 1.0f), 40.0f,
        juce::AudioParameterFloatAttributes().withLabel ("ms")));

    return layout;
}

void CleerAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    engine.prepare (sampleRate, getTotalNumInputChannels(), samplesPerBlock);
    // Feedback + gate are latency-free; the RNNoise stage adds ~10 ms of
    // frame latency while enabled. Reported dynamically when toggled.
    engine.noiseEnabled = apvts.getRawParameterValue ("noise_on")->load() > 0.5f;
    setLatencySamples (engine.latencySamples());
}

bool CleerAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto in = layouts.getMainInputChannelSet();
    const auto out = layouts.getMainOutputChannelSet();
    return in == out
        && (in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo());
}

void CleerAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    auto boolValue = [this] (const char* id) {
        return apvts.getRawParameterValue (id)->load() > 0.5f;
    };
    auto value = [this] (const char* id) {
        return (double) apvts.getRawParameterValue (id)->load();
    };

    engine.feedbackEnabled = boolValue (kFbOn);
    engine.gateEnabled = boolValue (kGateOn);

    const bool noiseOn = boolValue (kNoiseOn);
    if (noiseOn != engine.noiseEnabled)
    {
        engine.noiseEnabled = noiseOn;
        setLatencySamples (engine.latencySamples());
    }
    engine.setNoiseMix (value (kNoiseAmount) / 100.0);

    cleer::FeedbackSuppressor::Params fb;   // starts at the tuned defaults
    fb.prominenceDB = value (kFbSens);
    fb.maxDepthDB = value (kFbDepth);
    fb.q = value (kFbQ);
    engine.setFeedbackParams (fb);

    cleer::Expander::Params ex;
    ex.thresholdDB = value (kGateThresh);
    ex.ratio = value (kGateRatio);
    ex.floorDB = -value (kGateRange);
    ex.attackMs = value (kGateAttack);
    ex.releaseMs = value (kGateRelease);
    ex.holdMs = value (kGateHold);
    engine.setExpanderParams (ex);

    const int numCh = buffer.getNumChannels();
    const int n = buffer.getNumSamples();

    float inPeak = 0.0f;
    for (int c = 0; c < numCh; ++c)
        inPeak = std::max (inPeak, buffer.getMagnitude (c, 0, n));

    engine.processBlock (buffer.getArrayOfWritePointers(), numCh, n);

    float outPeak = 0.0f;
    for (int c = 0; c < numCh; ++c)
        outPeak = std::max (outPeak, buffer.getMagnitude (c, 0, n));

    uiInPeakDB.store (juce::Decibels::gainToDecibels (inPeak, -100.0f), std::memory_order_relaxed);
    uiOutPeakDB.store (juce::Decibels::gainToDecibels (outPeak, -100.0f), std::memory_order_relaxed);
    uiNotchCount.store (engine.activeNotchCount(), std::memory_order_relaxed);
    uiGainReductionDB.store ((float) engine.gainReductionDB(), std::memory_order_relaxed);
    uiVoiceProb.store (engine.voiceProbability(), std::memory_order_relaxed);

    if (const auto* spec = engine.uiSpectrumDB())
    {
        const int nb = std::min (kSpecBins, (int) spec->size());
        for (int k = 0; k < nb; ++k)
            uiSpectrumDB[(size_t) k].store ((float) (*spec)[(size_t) k], std::memory_order_relaxed);
    }

    float nf[kMaxUiNotches], nd[kMaxUiNotches];
    const int nn = engine.uiNotches (nf, nd, kMaxUiNotches);
    for (int i = 0; i < nn; ++i)
    {
        uiNotchFreq[(size_t) i].store (nf[i], std::memory_order_relaxed);
        uiNotchDepth[(size_t) i].store (nd[i], std::memory_order_relaxed);
    }
    uiNotchN.store (nn, std::memory_order_relaxed);
}

void CleerAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto xml = apvts.copyState().createXml())
        copyXmlToBinary (*xml, destData);
}

void CleerAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessorEditor* CleerAudioProcessor::createEditor()
{
    return new CleerAudioProcessorEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CleerAudioProcessor();
}
