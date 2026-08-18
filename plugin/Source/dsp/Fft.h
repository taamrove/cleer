#pragma once
#include <cmath>
#include <vector>

namespace cleer
{

// Minimal iterative radix-2 complex FFT (double precision). It runs once per
// 256-sample hop per channel on a 1024-point window, so raw speed is a
// non-issue; what matters is having zero dependencies so the DSP core builds
// and tests without JUCE or Accelerate.
class Fft
{
public:
    explicit Fft (int size) : n (size)
    {
        levels = 0;
        for (int s = n; s > 1; s >>= 1)
            ++levels;

        bitrev.resize ((size_t) n);
        for (int i = 0; i < n; ++i)
        {
            int r = 0;
            for (int b = 0; b < levels; ++b)
                r = (r << 1) | ((i >> b) & 1);
            bitrev[(size_t) i] = r;
        }

        cosTab.resize ((size_t) n / 2);
        sinTab.resize ((size_t) n / 2);
        for (int i = 0; i < n / 2; ++i)
        {
            const double a = 2.0 * M_PI * (double) i / (double) n;
            cosTab[(size_t) i] = std::cos (a);
            sinTab[(size_t) i] = std::sin (a);
        }
    }

    int size() const { return n; }

    // In-place forward FFT on split re/im arrays of length size().
    void forward (std::vector<double>& re, std::vector<double>& im) const
    {
        for (int i = 0; i < n; ++i)
        {
            const int j = bitrev[(size_t) i];
            if (j > i)
            {
                std::swap (re[(size_t) i], re[(size_t) j]);
                std::swap (im[(size_t) i], im[(size_t) j]);
            }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            const int half = len / 2;
            const int step = n / len;
            for (int i = 0; i < n; i += len)
            {
                for (int j = 0; j < half; ++j)
                {
                    const int k = j * step;
                    const double wr = cosTab[(size_t) k];
                    const double wi = -sinTab[(size_t) k];
                    const size_t a = (size_t) (i + j);
                    const size_t b = a + (size_t) half;
                    const double tr = re[b] * wr - im[b] * wi;
                    const double ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }

private:
    int n;
    int levels;
    std::vector<int> bitrev;
    std::vector<double> cosTab, sinTab;
};

} // namespace cleer
