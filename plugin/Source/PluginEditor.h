#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"

// Custom editor: live input spectrum with notch markers, IN/OUT/GR metering
// with a running in-vs-out difference readout, and knobs for both sections.
// All display data comes from the processor's relaxed-atomic ui* fields; the
// editor never touches the DSP objects.

namespace cleerui
{
// Palette — the plugin commits to a single dark look.
const juce::Colour bg        { 0xff14161a };
const juce::Colour panel     { 0xff1c1f24 };
const juce::Colour panelLine { 0xff2a2e35 };
const juce::Colour text      { 0xffc8cdd4 };
const juce::Colour textDim   { 0xff7a8089 };
const juce::Colour cyan      { 0xff4ec9e0 };
const juce::Colour amber     { 0xffe0a54e };
const juce::Colour green     { 0xff5ec97a };
const juce::Colour red       { 0xffe05a5a };

inline float freqToX (float freq, float x0, float w)
{
    const float lo = std::log10 (40.0f), hi = std::log10 (20000.0f);
    return x0 + w * (std::log10 (juce::jlimit (40.0f, 20000.0f, freq)) - lo) / (hi - lo);
}

struct LookAndFeel : juce::LookAndFeel_V4
{
    LookAndFeel()
    {
        setColour (juce::Slider::textBoxTextColourId, text);
        setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
        setColour (juce::Slider::textBoxHighlightColourId, cyan.withAlpha (0.3f));
        setColour (juce::TextEditor::backgroundColourId, juce::Colours::transparentBlack);
        setColour (juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
        setColour (juce::TextEditor::focusedOutlineColourId, cyan.withAlpha (0.5f));
        setColour (juce::Label::textColourId, textDim);
        setColour (juce::TooltipWindow::backgroundColourId, panel);
    }

    void drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                           float pos, float startAngle, float endAngle,
                           juce::Slider& s) override
    {
        const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat().reduced (4.0f);
        const float radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) / 2.0f;
        const auto centre = bounds.getCentre();
        const float angle = startAngle + pos * (endAngle - startAngle);
        const float arcR = radius - 2.0f;
        const auto accent = s.findColour (juce::Slider::rotarySliderFillColourId);

        juce::Path track;
        track.addCentredArc (centre.x, centre.y, arcR, arcR, 0.0f, startAngle, endAngle, true);
        g.setColour (panelLine);
        g.strokePath (track, juce::PathStrokeType (3.0f, juce::PathStrokeType::curved,
                                                   juce::PathStrokeType::rounded));

        juce::Path value;
        value.addCentredArc (centre.x, centre.y, arcR, arcR, 0.0f, startAngle, angle, true);
        g.setColour (accent);
        g.strokePath (value, juce::PathStrokeType (3.0f, juce::PathStrokeType::curved,
                                                   juce::PathStrokeType::rounded));

        const auto tip = centre.getPointOnCircumference (arcR - 4.0f, angle);
        g.setColour (text);
        g.drawLine ({ centre.getPointOnCircumference (arcR * 0.45f, angle), tip }, 2.0f);
    }

    void drawToggleButton (juce::Graphics& g, juce::ToggleButton& b,
                           bool highlighted, bool) override
    {
        const auto r = b.getLocalBounds().toFloat().reduced (1.0f);
        const bool on = b.getToggleState();
        const auto accent = on ? cyan : panelLine;
        g.setColour (on ? accent.withAlpha (0.18f) : panel);
        g.fillRoundedRectangle (r, r.getHeight() / 2.0f);
        g.setColour (highlighted ? accent.brighter (0.3f) : accent);
        g.drawRoundedRectangle (r, r.getHeight() / 2.0f, 1.2f);
        g.setColour (on ? text : textDim);
        g.setFont (juce::FontOptions (11.0f, juce::Font::bold));
        g.drawText (b.getButtonText(), r, juce::Justification::centred);
    }
};

// A labelled rotary knob bound to one parameter.
struct Knob : juce::Component
{
    Knob (juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId,
          const juce::String& title, juce::Colour accent)
    {
        slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        // Colours must be set before setTextBoxStyle: the text box label
        // copies them at creation, before the custom LookAndFeel is attached.
        slider.setColour (juce::Slider::rotarySliderFillColourId, accent);
        slider.setColour (juce::Slider::textBoxTextColourId, text);
        slider.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        slider.setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
        slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 62, 14);
        addAndMakeVisible (slider);

        name.setText (title, juce::dontSendNotification);
        name.setJustificationType (juce::Justification::centred);
        name.setFont (juce::FontOptions (10.5f, juce::Font::bold));
        addAndMakeVisible (name);

        attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
            apvts, paramId, slider);
    }

    void resized() override
    {
        auto area = getLocalBounds();
        name.setBounds (area.removeFromTop (14));
        slider.setBounds (area);
    }

    juce::Slider slider;
    juce::Label name;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
};

// Live input spectrum with grid, floor shading and notch markers.
struct SpectrumView : juce::Component, juce::Timer
{
    explicit SpectrumView (CleerAudioProcessor& p) : processor (p)
    {
        smoothed.fill (-100.0f);
        setOpaque (true);
        startTimerHz (30);
    }

    void timerCallback() override
    {
        // Windowed-DFT full-scale sine sits ~ 20*log10(win/4) above dBFS.
        constexpr float offset = 48.2f;
        for (int k = 0; k < CleerAudioProcessor::kSpecBins; ++k)
        {
            const float v = processor.uiSpectrumDB[(size_t) k].load (std::memory_order_relaxed) - offset;
            float& s = smoothed[(size_t) k];
            s += (v > s ? 0.55f : 0.20f) * (v - s);   // fast rise, slow fall
        }
        notchN = juce::jmin (processor.uiNotchN.load (std::memory_order_relaxed),
                             CleerAudioProcessor::kMaxUiNotches);
        for (int i = 0; i < notchN; ++i)
        {
            notchFreq[(size_t) i] = processor.uiNotchFreq[(size_t) i].load (std::memory_order_relaxed);
            notchDepth[(size_t) i] = processor.uiNotchDepth[(size_t) i].load (std::memory_order_relaxed);
        }
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        const auto r = getLocalBounds().toFloat();
        g.fillAll (bg);
        g.setColour (panel);
        g.fillRoundedRectangle (r, 6.0f);

        const float x0 = r.getX() + 6.0f, w = r.getWidth() - 12.0f;
        const float yTop = r.getY() + 6.0f, h = r.getHeight() - 24.0f;
        const float dbLo = -90.0f, dbHi = 0.0f;
        auto dbToY = [&] (float db) {
            return yTop + h * (1.0f - (juce::jlimit (dbLo, dbHi, db) - dbLo) / (dbHi - dbLo));
        };

        // Grid.
        g.setFont (juce::FontOptions (9.0f));
        for (float f : { 100.0f, 1000.0f, 10000.0f })
        {
            const float x = freqToX (f, x0, w);
            g.setColour (panelLine.withAlpha (0.6f));
            g.drawVerticalLine ((int) x, yTop, yTop + h);
            g.setColour (textDim);
            g.drawText (f >= 1000.0f ? juce::String (f / 1000.0f, 0) + "k" : juce::String ((int) f),
                        (int) x - 14, (int) (yTop + h) + 2, 28, 12, juce::Justification::centred);
        }
        for (float db : { -60.0f, -30.0f })
        {
            g.setColour (panelLine.withAlpha (0.4f));
            g.drawHorizontalLine ((int) dbToY (db), x0, x0 + w);
        }

        // Spectrum: one point per pixel column, max of the bins mapping there.
        const double sr = processor.getSampleRate() > 0.0 ? processor.getSampleRate() : 48000.0;
        juce::Path path;
        bool started = false;
        float prevX = -10.0f, colMax = dbLo;
        for (int k = 1; k < CleerAudioProcessor::kSpecBins; ++k)
        {
            const float f = (float) (k * sr / 1024.0);
            if (f < 40.0f || f > 20000.0f)
                continue;
            const float x = freqToX (f, x0, w);
            colMax = juce::jmax (colMax, smoothed[(size_t) k]);
            if (x - prevX < 1.0f)
                continue;
            const float y = dbToY (colMax);
            if (! started) { path.startNewSubPath (x, y); started = true; }
            else            path.lineTo (x, y);
            prevX = x;
            colMax = dbLo;
        }
        if (started)
        {
            juce::Path fill (path);
            fill.lineTo (prevX, yTop + h);
            fill.lineTo (x0, yTop + h);
            fill.closeSubPath();
            g.setColour (cyan.withAlpha (0.10f));
            g.fillPath (fill);
            g.setColour (cyan.withAlpha (0.85f));
            g.strokePath (path, juce::PathStrokeType (1.4f));
        }

        // Notch markers: line down from the top, brightness follows depth.
        for (int i = 0; i < notchN; ++i)
        {
            const float depth = -notchDepth[(size_t) i];        // positive dB
            if (depth < 0.5f)
                continue;
            const float x = freqToX (notchFreq[(size_t) i], x0, w);
            const float alpha = juce::jlimit (0.25f, 1.0f, depth / 24.0f);
            g.setColour (red.withAlpha (alpha));
            g.drawVerticalLine ((int) x, yTop, yTop + h);
            g.fillEllipse (x - 2.5f, yTop + 2.0f, 5.0f, 5.0f);
            g.setFont (juce::FontOptions (9.0f));
            g.drawText (juce::String ((int) notchFreq[(size_t) i]) + " Hz",
                        (int) x + 4, (int) yTop + 2, 52, 11, juce::Justification::left);
        }
    }

    CleerAudioProcessor& processor;
    std::array<float, CleerAudioProcessor::kSpecBins> smoothed {};
    std::array<float, CleerAudioProcessor::kMaxUiNotches> notchFreq {}, notchDepth {};
    int notchN = 0;
};

// IN / OUT / GR bars plus the in-vs-out difference readout.
struct MeterView : juce::Component, juce::Timer
{
    explicit MeterView (CleerAudioProcessor& p) : processor (p)
    {
        setOpaque (true);
        startTimerHz (30);
    }

    void timerCallback() override
    {
        auto follow = [] (float& state, float target) {
            state += (target > state ? 0.7f : 0.12f) * (target - state);
        };
        follow (inDB, processor.uiInPeakDB.load (std::memory_order_relaxed));
        follow (outDB, processor.uiOutPeakDB.load (std::memory_order_relaxed));
        follow (grDB, processor.uiGainReductionDB.load (std::memory_order_relaxed));
        notchCount = processor.uiNotchCount.load (std::memory_order_relaxed);
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        const auto r = getLocalBounds().toFloat();
        g.fillAll (bg);
        g.setColour (panel);
        g.fillRoundedRectangle (r, 6.0f);

        auto area = getLocalBounds().reduced (8);
        auto footer = area.removeFromBottom (34);
        auto labels = area.removeFromBottom (14);

        const int barW = area.getWidth() / 3;
        auto drawBar = [&] (juce::Rectangle<int> cell, float db, float lo, float hi,
                            juce::Colour col, const char* name, juce::Rectangle<int> labelCell,
                            bool topDown)
        {
            auto bar = cell.reduced (cell.getWidth() / 3, 0).toFloat();
            g.setColour (bg);
            g.fillRoundedRectangle (bar, 3.0f);
            const float frac = juce::jlimit (0.0f, 1.0f, (db - lo) / (hi - lo));
            auto lit = topDown ? bar.removeFromTop (bar.getHeight() * (1.0f - frac))
                               : bar.removeFromBottom (bar.getHeight() * frac);
            g.setColour (col);
            g.fillRoundedRectangle (lit, 3.0f);
            g.setColour (textDim);
            g.setFont (juce::FontOptions (9.5f, juce::Font::bold));
            g.drawText (name, labelCell, juce::Justification::centred);
        };

        auto c1 = area.removeFromLeft (barW), l1 = labels.removeFromLeft (barW);
        auto c2 = area.removeFromLeft (barW), l2 = labels.removeFromLeft (barW);
        drawBar (c1, inDB, -60.0f, 0.0f, green, "IN", l1, false);
        drawBar (c2, outDB, -60.0f, 0.0f, green, "OUT", l2, false);
        drawBar (area, grDB, -40.0f, 0.0f, amber, "GATE", labels, true);

        const float diff = outDB - inDB;
        g.setColour (text);
        g.setFont (juce::FontOptions (12.0f, juce::Font::bold));
        g.drawText ((diff <= -0.05f ? juce::String (diff, 1) : "0.0") + " dB",
                    footer.removeFromTop (16), juce::Justification::centred);
        g.setColour (textDim);
        g.setFont (juce::FontOptions (9.0f));
        g.drawText (juce::String ("DIFF   |   ") + juce::String (notchCount) + " NOTCH",
                    footer, juce::Justification::centred);
    }

    CleerAudioProcessor& processor;
    float inDB = -100.0f, outDB = -100.0f, grDB = 0.0f;
    int notchCount = 0;
};

} // namespace cleerui

class CleerAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit CleerAudioProcessorEditor (CleerAudioProcessor& p)
        : AudioProcessorEditor (&p),
          spectrum (p), meters (p),
          fbSens (p.apvts, "fb_sens", "SENS", cleerui::cyan),
          fbDepth (p.apvts, "fb_depth", "DEPTH", cleerui::cyan),
          fbQ (p.apvts, "fb_q", "Q", cleerui::cyan),
          gateThresh (p.apvts, "gate_thresh", "THRESH", cleerui::amber),
          gateRatio (p.apvts, "gate_ratio", "RATIO", cleerui::amber),
          gateRange (p.apvts, "gate_range", "RANGE", cleerui::amber),
          gateAttack (p.apvts, "gate_attack", "ATTACK", cleerui::amber),
          gateRelease (p.apvts, "gate_release", "RELEASE", cleerui::amber),
          gateHold (p.apvts, "gate_hold", "HOLD", cleerui::amber)
    {
        setLookAndFeel (&lnf);

        addAndMakeVisible (spectrum);
        addAndMakeVisible (meters);
        for (auto* k : { &fbSens, &fbDepth, &fbQ, &gateThresh, &gateRatio,
                         &gateRange, &gateAttack, &gateRelease, &gateHold })
            addAndMakeVisible (*k);

        fbOn.setButtonText ("KILL");
        gateOn.setButtonText ("GATE");
        addAndMakeVisible (fbOn);
        addAndMakeVisible (gateOn);
        fbOnAtt = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
            p.apvts, "fb_on", fbOn);
        gateOnAtt = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
            p.apvts, "gate_on", gateOn);

        setSize (780, 468);
    }

    ~CleerAudioProcessorEditor() override { setLookAndFeel (nullptr); }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (cleerui::bg);

        g.setColour (cleerui::text);
        g.setFont (juce::FontOptions (20.0f, juce::Font::bold));
        g.drawText ("CLEER", 16, 8, 120, 26, juce::Justification::left);
        g.setColour (cleerui::textDim);
        g.setFont (juce::FontOptions (11.0f));
        g.drawText ("de-feedback", 92, 12, 140, 20, juce::Justification::left);

        // Section panels behind the knobs.
        g.setColour (cleerui::panel);
        g.fillRoundedRectangle (fbPanel.toFloat(), 6.0f);
        g.fillRoundedRectangle (gatePanel.toFloat(), 6.0f);
        g.setColour (cleerui::textDim);
        g.setFont (juce::FontOptions (10.0f, juce::Font::bold));
        g.drawText ("FEEDBACK", fbPanel.getX() + 10, fbPanel.getY() + 6, 100, 12,
                    juce::Justification::left);
        g.drawText ("EXPANDER / GATE", gatePanel.getX() + 10, gatePanel.getY() + 6, 140, 12,
                    juce::Justification::left);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (14);
        area.removeFromTop (28);   // header

        auto top = area.removeFromTop (216);
        meters.setBounds (top.removeFromRight (128));
        top.removeFromRight (10);
        spectrum.setBounds (top);

        area.removeFromTop (10);
        fbPanel = area.removeFromLeft (272);
        area.removeFromLeft (10);
        gatePanel = area;

        auto layoutPanel = [] (juce::Rectangle<int> panel, juce::ToggleButton& toggle,
                               std::initializer_list<cleerui::Knob*> knobs)
        {
            toggle.setBounds (panel.getRight() - 64, panel.getY() + 6, 54, 20);
            auto row = panel.reduced (8, 0).withTrimmedTop (30).withTrimmedBottom (8);
            const int kw = row.getWidth() / (int) knobs.size();
            for (auto* k : knobs)
                k->setBounds (row.removeFromLeft (kw).reduced (2, 0));
        };

        layoutPanel (fbPanel, fbOn, { &fbSens, &fbDepth, &fbQ });
        layoutPanel (gatePanel, gateOn, { &gateThresh, &gateRatio, &gateRange,
                                          &gateAttack, &gateRelease, &gateHold });
    }

private:
    cleerui::LookAndFeel lnf;

    cleerui::SpectrumView spectrum;
    cleerui::MeterView meters;

    cleerui::Knob fbSens, fbDepth, fbQ;
    cleerui::Knob gateThresh, gateRatio, gateRange, gateAttack, gateRelease, gateHold;

    juce::ToggleButton fbOn, gateOn;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> fbOnAtt, gateOnAtt;

    juce::Rectangle<int> fbPanel, gatePanel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CleerAudioProcessorEditor)
};
