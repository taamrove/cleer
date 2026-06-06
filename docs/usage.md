# Using Cleer: live mix, and testing with a recorded vocal

Cleer just needs an **input device + channel** and an **output device**. Two
setups:

## A. Live use (your mix) — interface in, interface out

1. Add an instance (or use **Start mic → speakers** and change the devices).
2. **Input** = your audio interface, **Channel** = the vocal/mix channel.
3. **Output** = your interface's outputs (to the PA / monitors) — or Mac
   speakers.
4. Toggle **Feedback / Denoise / Neural / Dereverb** as needed and hit **Start**.

Input and output can be the *same* interface or *two different* interfaces —
Cleer runs separate capture/playback engines and drift-corrects between them.
For two interfaces, an **Aggregate Device** gives sample-accurate sync.

## B. Test with a recorded vocal take (via BlackHole → Mac speakers)

You want: a recorded vocal plays → into Cleer → processed → out the Mac
speakers so you can hear the difference. Route the player's audio into a virtual
device (BlackHole) that Cleer reads as its input.

### 1. Install BlackHole (virtual audio cable)
```bash
brew install blackhole-2ch
```
(or download from https://existential.audio/blackhole/). You'll now have a
**"BlackHole 2ch"** device in Audio MIDI Setup.

### 2. Send the vocal take INTO BlackHole
Pick whichever matches how you're playing the take:

- **From a DAW** (best): set the vocal track's (or master's) output bus to
  **BlackHole 2ch**.
- **From a file player** (QuickTime / Music / browser): set the **system
  output** to **BlackHole 2ch** (System Settings ▸ Sound ▸ Output), then play
  the file. The player's audio now goes into BlackHole instead of the speakers.

> Don't worry that the system output is BlackHole — Cleer writes its **output
> straight to the speakers device**, independent of the system default, so
> you'll still hear the processed result.

### 3. Configure Cleer
- **Input** = `BlackHole 2ch`, **Channel** = `Ch 1`
- **Output** = `MacBook Speakers` (or your "Built-in Output")
- **Start**.

Play the take → you hear the **processed** vocal on the Mac speakers. Flip the
master **Process / Bypass** switch to A/B against the raw take.

### Notes
- This path is great for testing **Denoise** and **Dereverb**. To test
  **Feedback** suppression you need actual howl in the recording (or do the live
  mic-into-speakers test).
- BlackHole and the built-in speakers are different clocks; the ring buffer
  drift-corrects, so a few minutes of playback stays in sync.
- No feedback risk here (it's a file, not an open mic), so you can turn the
  speaker volume up to hear detail.

## C. Hear it on speakers AND keep monitoring elsewhere
If you want the processed audio in more than one place, make a **Multi-Output
Device** in Audio MIDI Setup (e.g. Speakers + interface) and pick it as Cleer's
output.
