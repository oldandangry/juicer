// JuicerProcessing.h
#pragma once
#define JUICER_ENABLE_COUPLERS 1

#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#include "SpectralMath.h"
#include "WorkingState.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"
#include "OutputEncoding.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace JuicerProc {
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst);
}

namespace JuicerProc {
    // Render a small print preview to outRGB (RGB, row-major), measuring from the source image.
    // outRGB size must be outW*outH*3. srcBounds is the full image bounds used for sampling.
    void renderPrintPreviewToBuffer(
        const OFX::Image* src, const OfxRectI& srcBounds,
        float* outRGB, int outW, int outH,
        const Print::Params& prm,
        const Print::Runtime* prt,
        const WorkingState* ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        const OutputEncoding::Params& outputEncoding = OutputEncoding::Params{});
}


// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    explicit JuicerProcessor(OFX::ImageEffect& effect);

    void setSrcDst(OFX::Image* src, OFX::Image* dst);
    void setRenderWindowRect(const OfxRectI& rect);
    void setComponents(int n);
    void setScannerParams(const Scanner::Params& p);
    void setPrintParams(const Print::Params& p);
    void setDirRuntime(const Couplers::Runtime& rt);
    void setWorkingState(const WorkingState* ws, bool wsReady);
    void setPrintRuntime(const Print::Runtime* prt, bool printReady);
    void setExposure(float exposureScale);
    void setOutputEncoding(const OutputEncoding::Params& p);

    void multiThreadProcessImages(OfxRectI procWindow) override;

private:
    OFX::Image* _srcImg;
    int _nComponents;

    Scanner::Params _scannerParams;
    Print::Params _printParams;
    Couplers::Runtime _dirRT;

    const Print::Runtime* _prt;
    const WorkingState* _ws;
    bool _wsReady;
    bool _printReady;

    float _exposureScale;
    OutputEncoding::Params _outputEncoding;
};
// Test-facing wrappers to access internal spatial utilities without changing production behavior.
namespace JuicerProcTest {

    // Build a separable Gaussian kernel and return it for assertions.
    // Why inline: ensures tests link even if production TU isn't linked.
    inline void makeGaussianKernel(float sigma, std::vector<float>& outKernel) {
        outKernel.clear();
        if (!(std::isfinite(sigma)) || sigma <= 0.5f) {
            outKernel.push_back(1.0f);
            return;
        }
        const int radiusRaw = std::max(1, int(std::ceil(3.0f * sigma)));
        const int radius = std::min(radiusRaw, 75); // cap at 75 taps each side
        outKernel.resize(size_t(2 * radius + 1));
        const float s2 = sigma * sigma * 2.0f;
        float wsum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float w = std::exp(-(i * i) / s2);
            outKernel[size_t(i + radius)] = w;
            wsum += w;
        }
        for (float& w : outKernel) w /= wsum;
    }

    // Curve sanity check (monotonic X, matching sizes), mirrors internal curve_ok().
    inline bool curveOk(const Spectral::Curve& c) {
        const size_t N = c.lambda_nm.size();
        if (N < 2 || c.linear.size() != N) return false;
        float prev = c.lambda_nm[0];
        if (!std::isfinite(prev)) return false;
        for (size_t i = 1; i < N; ++i) {
            const float xi = c.lambda_nm[i];
            if (!std::isfinite(xi)) return false;
            if (xi < prev) return false; // allow duplicates, never decreasing
            prev = xi;
        }
        return true;
    }

    // Safe density sampling at logE, mirrors internal sample_density_at_logE_safe().
    inline float sampleDensityAtLogESafe(const Spectral::Curve& c, float le) {
        const size_t N = c.lambda_nm.size();
        if (N == 0 || c.linear.size() != N) return 0.0f;

        const float xmin = c.lambda_nm.front();
        const float xmax = c.lambda_nm.back();
        float xq = le;
        if (!std::isfinite(xq)) xq = xmin;
        if (xq <= xmin) return c.linear.front();
        if (xq >= xmax) return c.linear.back();

        size_t i1 = size_t(std::lower_bound(c.lambda_nm.begin(), c.lambda_nm.end(), xq)
            - c.lambda_nm.begin());
        if (i1 == 0)   return c.linear.front();
        if (i1 >= N)   return c.linear.back();
        const size_t i0 = i1 - 1;

        const float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
        const float y0 = c.linear[i0], y1 = c.linear[i1];

        if (!std::isfinite(x0) || !std::isfinite(x1) || !std::isfinite(y0) || !std::isfinite(y1)) {
            return std::isfinite(y0) ? y0 : 0.0f; // conservative fallback
        }
        const float denom = (x1 - x0);
        if (!(denom > 0.0f) || !std::isfinite(denom)) return y0;
        const float t = (xq - x0) / denom;
        return y0 + t * (y1 - y0);
    }
}

