// Offline runner: pull a WAV through the exact plugin chain (feedback
// notches -> RNNoise -> expander) and write the result next to the input.
// The C++ twin of reference/tools/process_wav.py, for listening tests and
// A/B against the commercial plugin on real material.
//
//   cleer_process INPUT.wav [--out OUT.wav] [--start S] [--dur S]
//                 [--noise-mix 0..1] [--no-fb] [--no-noise] [--no-gate]
//
// Output is latency-compensated (the RNNoise frame delay is trimmed), so
// input and output stay sample-aligned for difference listening.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../Source/dsp/CleerEngine.h"

namespace
{
struct Wav
{
    double sampleRate = 48000.0;
    int channels = 1;
    std::vector<std::vector<double>> data;   // [channel][sample]
};

bool readWav (const char* path, Wav& w, double startS, double durS)
{
    FILE* f = std::fopen (path, "rb");
    if (f == nullptr)
        return false;

    auto rd32 = [&] { uint32_t v = 0; std::fread (&v, 4, 1, f); return v; };
    auto rd16 = [&] { uint16_t v = 0; std::fread (&v, 2, 1, f); return v; };

    char tag[5] = {};
    std::fread (tag, 1, 4, f);
    rd32();
    std::fread (tag, 1, 4, f);
    if (std::memcmp (tag, "WAVE", 4) != 0)
        { std::fclose (f); return false; }

    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t sr = 0;
    long dataPos = -1;
    uint32_t dataLen = 0;
    while (std::fread (tag, 1, 4, f) == 4)
    {
        const uint32_t len = rd32();
        if (std::memcmp (tag, "fmt ", 4) == 0)
        {
            fmt = rd16();
            ch = rd16();
            sr = rd32();
            rd32();
            rd16();
            bits = rd16();
            std::fseek (f, (long) len - 16, SEEK_CUR);
        }
        else if (std::memcmp (tag, "data", 4) == 0)
        {
            dataPos = std::ftell (f);
            dataLen = len;
            std::fseek (f, (long) len + (len & 1), SEEK_CUR);
        }
        else
            std::fseek (f, (long) len + (len & 1), SEEK_CUR);
    }
    if (dataPos < 0 || ch == 0 || sr == 0 || (fmt != 1 && fmt != 3))
        { std::fclose (f); return false; }

    const int bytesPer = bits / 8;
    const uint64_t totalFrames = dataLen / (uint64_t) (bytesPer * ch);
    uint64_t startF = (uint64_t) (startS * sr);
    if (startF > totalFrames)
        startF = totalFrames;
    uint64_t count = durS > 0 ? (uint64_t) (durS * sr) : totalFrames - startF;
    if (startF + count > totalFrames)
        count = totalFrames - startF;

    std::fseek (f, dataPos + (long) (startF * bytesPer * ch), SEEK_SET);
    std::vector<uint8_t> raw (count * (uint64_t) bytesPer * ch);
    const size_t got = std::fread (raw.data(), 1, raw.size(), f);
    std::fclose (f);
    const uint64_t frames = got / (uint64_t) (bytesPer * ch);

    w.sampleRate = sr;
    w.channels = ch;
    w.data.assign (ch, std::vector<double> (frames, 0.0));
    const uint8_t* p = raw.data();
    for (uint64_t i = 0; i < frames; ++i)
        for (int c = 0; c < ch; ++c)
        {
            double v = 0.0;
            if (fmt == 3 && bits == 32)
            {
                float fl;
                std::memcpy (&fl, p, 4);
                v = fl;
            }
            else if (bits == 16)
            {
                int16_t s;
                std::memcpy (&s, p, 2);
                v = s / 32768.0;
            }
            else if (bits == 24)
            {
                int32_t s = (int32_t) (p[0] | (p[1] << 8) | (p[2] << 16));
                if (s & 0x800000)
                    s -= 0x1000000;
                v = s / 8388608.0;
            }
            else if (bits == 32)
            {
                int32_t s;
                std::memcpy (&s, p, 4);
                v = s / 2147483648.0;
            }
            w.data[(size_t) c][i] = v;
            p += bytesPer;
        }
    return true;
}

bool writeWav24 (const char* path, const Wav& w)
{
    FILE* f = std::fopen (path, "wb");
    if (f == nullptr)
        return false;
    const uint64_t frames = w.data.empty() ? 0 : w.data[0].size();
    const uint32_t dataLen = (uint32_t) (frames * 3 * (uint64_t) w.channels);
    auto wr32 = [&] (uint32_t v) { std::fwrite (&v, 4, 1, f); };
    auto wr16 = [&] (uint16_t v) { std::fwrite (&v, 2, 1, f); };
    std::fwrite ("RIFF", 1, 4, f);
    wr32 (36 + dataLen);
    std::fwrite ("WAVEfmt ", 1, 8, f);
    wr32 (16);
    wr16 (1);
    wr16 ((uint16_t) w.channels);
    wr32 ((uint32_t) w.sampleRate);
    wr32 ((uint32_t) (w.sampleRate * 3 * w.channels));
    wr16 ((uint16_t) (3 * w.channels));
    wr16 (24);
    std::fwrite ("data", 1, 4, f);
    wr32 (dataLen);
    for (uint64_t i = 0; i < frames; ++i)
        for (int c = 0; c < w.channels; ++c)
        {
            double v = w.data[(size_t) c][i];
            v = v > 1.0 ? 1.0 : (v < -1.0 ? -1.0 : v);
            const int32_t s = (int32_t) std::lround (v * 8388607.0);
            const uint8_t b[3] = { (uint8_t) (s & 0xff), (uint8_t) ((s >> 8) & 0xff),
                                   (uint8_t) ((s >> 16) & 0xff) };
            std::fwrite (b, 1, 3, f);
        }
    std::fclose (f);
    return true;
}

double rmsDB (const std::vector<double>& x, size_t a, size_t b)
{
    double acc = 0.0;
    for (size_t i = a; i < b && i < x.size(); ++i)
        acc += x[i] * x[i];
    return 20.0 * std::log10 (std::sqrt (acc / (double) (b - a)) + 1e-12);
}
} // namespace

int main (int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf ("usage: cleer_process INPUT.wav [--out OUT.wav] [--start S] [--dur S]\n"
                     "                     [--noise-mix 0..1] [--no-fb] [--no-noise] [--no-gate]\n");
        return 2;
    }
    std::string inPath = argv[1], outPath;
    double start = 0.0, dur = 0.0, noiseMix = 0.9;
    bool fb = true, noise = true, gate = true;
    for (int i = 2; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&] { return (i + 1 < argc) ? std::atof (argv[++i]) : 0.0; };
        if (a == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (a == "--start") start = next();
        else if (a == "--dur") dur = next();
        else if (a == "--noise-mix") noiseMix = next();
        else if (a == "--no-fb") fb = false;
        else if (a == "--no-noise") noise = false;
        else if (a == "--no-gate") gate = false;
    }
    if (outPath.empty())
    {
        outPath = inPath;
        const auto dot = outPath.rfind ('.');
        outPath = outPath.substr (0, dot) + "_cleer.wav";
    }

    Wav w;
    if (! readWav (inPath.c_str(), w, start, dur))
    {
        std::printf ("failed to read %s\n", inPath.c_str());
        return 1;
    }
    const size_t n = w.data[0].size();
    std::printf ("in:  %s  %.0f Hz, %d ch, %.1f s\n", inPath.c_str(), w.sampleRate, w.channels,
                 n / w.sampleRate);

    Wav orig = w;

    cleer::Engine e;
    e.prepare (w.sampleRate, w.channels, 4096);
    e.feedbackEnabled = fb;
    e.noiseEnabled = noise;
    e.gateEnabled = gate;
    e.setNoiseMix (noiseMix);

    std::vector<double*> ptrs ((size_t) w.channels);
    int maxNotches = 0;
    for (size_t i = 0; i < n; i += 4096)
    {
        const int len = (int) (n - i < 4096 ? n - i : 4096);
        for (int c = 0; c < w.channels; ++c)
            ptrs[(size_t) c] = w.data[(size_t) c].data() + i;
        // Engine wants float; run through a float staging block.
        static std::vector<std::vector<float>> stage;
        stage.assign ((size_t) w.channels, {});
        std::vector<float*> fptrs ((size_t) w.channels);
        for (int c = 0; c < w.channels; ++c)
        {
            auto& s = stage[(size_t) c];
            s.resize ((size_t) len);
            for (int k = 0; k < len; ++k)
                s[(size_t) k] = (float) ptrs[(size_t) c][k];
            fptrs[(size_t) c] = s.data();
        }
        e.processBlock (fptrs.data(), w.channels, len);
        for (int c = 0; c < w.channels; ++c)
            for (int k = 0; k < len; ++k)
                ptrs[(size_t) c][k] = fptrs[(size_t) c][(size_t) k];
        maxNotches = std::max (maxNotches, e.activeNotchCount());
    }

    // Latency-compensate so out lines up with in.
    const int lat = e.latencySamples();
    if (lat > 0)
        for (auto& chData : w.data)
            chData.erase (chData.begin(), chData.begin() + std::min ((size_t) lat, chData.size()));

    std::printf ("chain: fb=%d noise=%d(mix %.2f) gate=%d, latency %d smp, max notches %d\n",
                 fb, noise, noiseMix, gate, lat, maxNotches);
    const size_t sec = (size_t) w.sampleRate;
    for (size_t s0 = 0; s0 + sec <= w.data[0].size() && s0 < 10 * sec; s0 += 2 * sec)
        std::printf ("  %4.0fs  in %6.1f dB   out %6.1f dB   diff %+5.1f dB\n",
                     start + s0 / w.sampleRate,
                     rmsDB (orig.data[0], s0, s0 + sec),
                     rmsDB (w.data[0], s0, s0 + sec),
                     rmsDB (w.data[0], s0, s0 + sec) - rmsDB (orig.data[0], s0, s0 + sec));

    if (! writeWav24 (outPath.c_str(), w))
    {
        std::printf ("failed to write %s\n", outPath.c_str());
        return 1;
    }
    std::printf ("out: %s\n", outPath.c_str());
    return 0;
}
