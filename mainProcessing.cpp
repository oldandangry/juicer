// JuicerProcessing.cpp

#define JUICER_ENABLE_COUPLERS 1
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Resolve OFX support library C++ wrappers — suppress MSVC C5040 for dynamic exception specs
#pragma warning(push)
#pragma warning(disable: 5040)
#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#pragma warning(pop)
#include "SpectralMath.h"
#include "WorkingState.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"
#include "mainProcessing.h"

namespace JuicerProc {

    // Copied from main.cpp helper, unchanged behavior.
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst) {
        const OfxRectI bounds = src->getBounds();
        const OFX::PixelComponentEnum comps = src->getPixelComponents();
        const OFX::BitDepthEnum depth = src->getPixelDepth();

        int nComponents = 0;
        switch (comps) {
        case OFX::ePixelComponentRGBA: nComponents = 4; break;
        case OFX::ePixelComponentRGB:  nComponents = 3; break;
        case OFX::ePixelComponentAlpha:nComponents = 1; break;
        default: return;
        }
        int bytesPerComp = 0;
        switch (depth) {
        case OFX::eBitDepthUByte:  bytesPerComp = 1; break;
        case OFX::eBitDepthUShort: bytesPerComp = 2; break;
        case OFX::eBitDepthFloat:  bytesPerComp = 4; break;
        default: return;
        }
        const size_t bytesPerPixel = size_t(nComponents * bytesPerComp);
        for (int y = bounds.y1; y < bounds.y2; ++y) {
            for (int x = bounds.x1; x < bounds.x2; ++x) {
                const void* s = src->getPixelAddress(x, y);
                void* d = dst->getPixelAddress(x, y);
                if (!s || !d) continue;
                std::memcpy(d, s, bytesPerPixel);
            }
        }
    }

    // Separable Gaussian kernel builder, with radius cap for safety.
    static void buildGaussianKernel(float sigma, std::vector<float>& kernel) {
        kernel.clear();
        if (!(std::isfinite(sigma)) || sigma <= 0.5f) {
            kernel.push_back(1.0f);
            return;
        }

        const int radiusRaw = std::max(1, int(std::ceil(3.0f * sigma)));
        const int radius = std::min(radiusRaw, 75); // cap at 75 taps each side

        kernel.resize(size_t(2 * radius + 1));
        const float s2 = sigma * sigma * 2.0f;
        float wsum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            float w = std::exp(-(i * i) / s2);
            kernel[size_t(i + radius)] = w;
            wsum += w;
        }
        for (float& w : kernel) w /= wsum;
    }


    // Separable blur, unchanged.
    static void blurChannelSeparable(const std::vector<float>& src, std::vector<float>& tmp, std::vector<float>& dst,
        int width, int height, const std::vector<float>& k) {
        // Horizontal
        tmp.assign(size_t(width * height), 0.0f);
        const int radius = int(k.size() / 2);
        for (int y = 0; y < height; ++y) {
            const float* srow = &src[size_t(y * width)];
            float* trow = &tmp[size_t(y * width)];
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int j = -radius; j <= radius; ++j) {
                    int xx = std::min(std::max(x + j, 0), width - 1);
                    acc += srow[xx] * k[size_t(j + radius)];
                }
                trow[x] = acc;
            }
        }
        // Vertical
        dst.assign(size_t(width * height), 0.0f);
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                float acc = 0.0f;
                for (int j = -radius; j <= radius; ++j) {
                    int yy = std::min(std::max(y + j, 0), height - 1);
                    acc += tmp[size_t(yy * width + x)] * k[size_t(j + radius)];
                }
                dst[size_t(y * width + x)] = acc;
            }
        }
    }
}

// --- Spatial DIR: defensive curve utilities (monotonic + robust interpolation) ---

static inline bool curve_ok(const Spectral::Curve& c) {
    const size_t N = c.lambda_nm.size();
    if (N < 2 || c.linear.size() != N) return false;
    float prev = c.lambda_nm[0];
    if (!std::isfinite(prev)) return false;
    for (size_t i = 1; i < N; ++i) {
        float xi = c.lambda_nm[i];
        if (!std::isfinite(xi)) return false;
        if (xi < prev) return false; // allow duplicates (xi == prev), but never decreasing
        prev = xi;
    }
    return true;
}

static inline float sample_density_at_logE_safe(const Spectral::Curve& c, float le) {
    const size_t N = c.lambda_nm.size();
    if (N == 0 || c.linear.size() != N) return 0.0f;

    const float xmin = c.lambda_nm.front();
    const float xmax = c.lambda_nm.back();
    float xq = le;
    if (!std::isfinite(xq)) xq = xmin;
    if (xq <= xmin) return c.linear.front();
    if (xq >= xmax) return c.linear.back();

    // lower_bound assumes non-decreasing X
    size_t i1 = size_t(std::lower_bound(c.lambda_nm.begin(), c.lambda_nm.end(), xq) - c.lambda_nm.begin());
    if (i1 == 0) return c.linear.front();
    if (i1 >= N) return c.linear.back();
    size_t i0 = i1 - 1;

    float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
    float y0 = c.linear[i0], y1 = c.linear[i1];

    // Any non-finite -> conservative fallback
    if (!std::isfinite(x0) || !std::isfinite(x1) || !std::isfinite(y0) || !std::isfinite(y1)) {
        return std::isfinite(y0) ? y0 : 0.0f;
    }
    const float denom = (x1 - x0);
    if (!(denom > 0.0f) || !std::isfinite(denom)) return y0;
    const float t = (xq - x0) / denom;
    return y0 + t * (y1 - y0);
}


namespace JuicerProc {
    // Note: print preview should be measured at unit scene exposure for compensation;
    // caller passes exposureScale=1.0f to avoid compounding EV.
    void renderPrintPreviewToBuffer(
        const OFX::Image* src, const OfxRectI& srcBounds,
        float* outRGB, int outW, int outH,
        const Print::Params& prm,
        const Print::Runtime* prt,
        const WorkingState* ws,
        const Couplers::Runtime& dirRT,
        float exposureScale)
    {
        if (!src || !outRGB || outW <= 0 || outH <= 0 || !prt || !ws) return;

        const int W = srcBounds.x2 - srcBounds.x1;
        const int H = srcBounds.y2 - srcBounds.y1;
        if (W <= 0 || H <= 0) return;

        // Map preview pixel centers to source coordinates (nearest neighbor)
        for (int yy = 0; yy < outH; ++yy) {
            for (int xx = 0; xx < outW; ++xx) {
                const float fx = (xx + 0.5f) / float(outW);
                const float fy = (yy + 0.5f) / float(outH);

                int sx = srcBounds.x1 + std::min(std::max(int(std::floor(fx * W)), 0), W - 1);
                int sy = srcBounds.y1 + std::min(std::max(int(std::floor(fy * H)), 0), H - 1);

                const float* srcPix = reinterpret_cast<const float*>(src->getPixelAddress(sx, sy));
                float* dstPix = &outRGB[(size_t(yy) * outW + size_t(xx)) * 3];

                if (!srcPix) {
                    dstPix[0] = dstPix[1] = dstPix[2] = 0.0f;
                    continue;
                }

                float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
                float rgbOut[3] = { rgbIn[0], rgbIn[1], rgbIn[2] };

                // Simulate print pixel using current params, runtime, and working state
                Print::simulate_print_pixel(
                    rgbIn, prm,
                    *prt, dirRT, *ws,
                    /*exposureScale*/ exposureScale,
                    rgbOut);

                dstPix[0] = rgbOut[0];
                dstPix[1] = rgbOut[1];
                dstPix[2] = rgbOut[2];
            }
        }
    }
} // namespace JuicerProc

// JuicerProcessor method definitions matching JuicerProcessing.h

JuicerProcessor::JuicerProcessor(OFX::ImageEffect& effect)
    : OFX::ImageProcessor(effect)
    , _srcImg(nullptr)
    , _nComponents(0)
    , _prt(nullptr)
    , _ws(nullptr)
    , _wsReady(false)
    , _printReady(false)
    , _exposureScale(1.0f)    
{
}

void JuicerProcessor::setSrcDst(OFX::Image* src, OFX::Image* dst) {
    _srcImg = src;
    setDstImg(dst);
}

void JuicerProcessor::setRenderWindowRect(const OfxRectI& rect) { setRenderWindow(rect); }
void JuicerProcessor::setComponents(int n) { _nComponents = n; }
void JuicerProcessor::setScannerParams(const Scanner::Params& p) { _scannerParams = p; }
void JuicerProcessor::setPrintParams(const Print::Params& p) { _printParams = p; }
void JuicerProcessor::setDirRuntime(const Couplers::Runtime& rt) { _dirRT = rt; }
void JuicerProcessor::setWorkingState(const WorkingState* ws, bool wsReady) {
    _ws = ws;
    _wsReady = wsReady;
    // Align DIR normalization constants to per-instance maxima if available
    if (_wsReady && _ws) {
        _dirRT.dMax[0] = (std::isfinite(_ws->dMax[0]) && _ws->dMax[0] > 1e-4f) ? _ws->dMax[0] : 1.0f;
        _dirRT.dMax[1] = (std::isfinite(_ws->dMax[1]) && _ws->dMax[1] > 1e-4f) ? _ws->dMax[1] : 1.0f;
        _dirRT.dMax[2] = (std::isfinite(_ws->dMax[2]) && _ws->dMax[2] > 1e-4f) ? _ws->dMax[2] : 1.0f;
    }
}
void JuicerProcessor::setPrintRuntime(const Print::Runtime* prt, bool printReady) { _prt = prt; _printReady = printReady; }
void JuicerProcessor::setExposure(float exposureScale) {
    _exposureScale = exposureScale;
}

void JuicerProcessor::multiThreadProcessImages(OfxRectI procWindow) {
    if (!_srcImg || !_dstImg) return;

    const int tileW = procWindow.x2 - procWindow.x1;
    const int tileH = procWindow.y2 - procWindow.y1;
    const bool doSpatial = std::isfinite(_dirRT.spatialSigmaPixels) && (_dirRT.spatialSigmaPixels > 0.5f);

    auto curvesReady = [&]()->bool {
        if (!_ws) return false;
        return curve_ok(_ws->densB) && curve_ok(_ws->densG) && curve_ok(_ws->densR);
        };

    if (_dirRT.active && doSpatial && _nComponents >= 3 && _wsReady && _ws && curvesReady()) {        
        std::vector<float> logE_B(size_t(tileW * tileH));
        std::vector<float> logE_G(size_t(tileW * tileH));
        std::vector<float> logE_R(size_t(tileW * tileH));

        std::vector<float> aY(size_t(tileW * tileH));
        std::vector<float> aM(size_t(tileW * tileH));
        std::vector<float> aC(size_t(tileW * tileH));

        // Pass A
        for (int yy = 0; yy < tileH; ++yy) {
            if (_effect.abort()) break;
            for (int xx = 0; xx < tileW; ++xx) {
                const int x = procWindow.x1 + xx;
                const int y = procWindow.y1 + yy;
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                const size_t idx = size_t(yy * tileW + xx);

                if (!srcPix) {
                    logE_B[idx] = logE_G[idx] = logE_R[idx] = 0.0f;
                    aY[idx] = aM[idx] = aC[idx] = 0.0f;
                    continue;
                }

                float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
                float E[3];
                Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
                    rgbIn, E, _exposureScale,
                    (_ws && _ws->tablesView.K > 0 ? &_ws->tablesView : nullptr),
                    (_ws && _ws->spdReady ? _ws->spdSInv : nullptr),
                    (int)std::clamp(_ws ? _ws->spectralMode : 0, 0, 1),
                    (_ws && (_ws->exposureModel == 1) && _ws->spdReady),
                    _ws->sensB, _ws->sensG, _ws->sensR);

                float leB = std::log10(std::max(E[0], 1e-6f)) + _ws->logEOffB;
                float leG = std::log10(std::max(E[1], 1e-6f)) + _ws->logEOffG;
                float leR = std::log10(std::max(E[2], 1e-6f)) + _ws->logEOffR;

                auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
                    if (c.lambda_nm.empty()) return le;
                    const float xmin = c.lambda_nm.front();
                    const float xmax = c.lambda_nm.back();
                    if (!std::isfinite(le)) return xmin;
                    return std::min(std::max(le, xmin), xmax);
                    };
                leB = clamp_logE_to_curve(_ws->densB, leB);
                leG = clamp_logE_to_curve(_ws->densG, leG);
                leR = clamp_logE_to_curve(_ws->densR, leR);

                logE_B[idx] = leB;
                logE_G[idx] = leG;
                logE_R[idx] = leR;

                float D_Y = sample_density_at_logE_safe(_ws->densB, leB);
                float D_M = sample_density_at_logE_safe(_ws->densG, leG);
                float D_C = sample_density_at_logE_safe(_ws->densR, leR);

                // Use the single, robust path (parity with agx)
                float aCorr[3];
                Couplers::ApplyInputLogE io{ { leB, leG, leR }, { D_Y, D_M, D_C } };
                Couplers::compute_logE_corrections(io, _dirRT, aCorr);
                aY[idx] = aCorr[0];
                aM[idx] = aCorr[1];
                aC[idx] = aCorr[2];
            }
        }

        // Blur corrections
        std::vector<float> k;
        JuicerProc::buildGaussianKernel(_dirRT.spatialSigmaPixels, k);
        std::vector<float> tmp, aY_blur, aM_blur, aC_blur;
        JuicerProc::blurChannelSeparable(aY, tmp, aY_blur, tileW, tileH, k);
        JuicerProc::blurChannelSeparable(aM, tmp, aM_blur, tileW, tileH, k);
        JuicerProc::blurChannelSeparable(aC, tmp, aC_blur, tileW, tileH, k);

        // NaN scrub and conservative clamp on blurred corrections
        auto scrubClamp = [](std::vector<float>& v) {
            for (float& t : v) {
                if (!std::isfinite(t)) t = 0.0f;
                // keep corrections within plausible design bounds
                if (t < -10.0f) t = -10.0f;
                if (t > 10.0f) t = 10.0f;
            }
        };
        scrubClamp(aY_blur);
        scrubClamp(aM_blur);
        scrubClamp(aC_blur);

        // Pass B
        for (int yy = 0; yy < tileH; ++yy) {
            if (_effect.abort()) break;
            for (int xx = 0; xx < tileW; ++xx) {
                const int x = procWindow.x1 + xx;
                const int y = procWindow.y1 + yy;

                // Fetch both src and dst addresses *per pixel* to avoid stride/tiling assumptions
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (!dstPix || !srcPix) {
                    continue;
                }

                float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
                float rgbOut[3] = { rgbIn[0],  rgbIn[1],  rgbIn[2] };

                const size_t idx = size_t(yy * tileW + xx);
                float leB2 = logE_B[idx] - aY_blur[idx];
                float leG2 = logE_G[idx] - aM_blur[idx];
                float leR2 = logE_R[idx] - aC_blur[idx];

                auto clamp_to = [](float le, const Spectral::Curve& c)->float {
                    if (c.lambda_nm.empty()) return le;
                    const float xmin = c.lambda_nm.front();
                    const float xmax = c.lambda_nm.back();
                    if (!std::isfinite(le)) return xmin;
                    return std::min(std::max(le, xmin), xmax);
                    };
                leB2 = clamp_to(leB2, _ws->densB);
                leG2 = clamp_to(leG2, _ws->densG);
                leR2 = clamp_to(leR2, _ws->densR);

                float D_cmy[3];
                D_cmy[0] = sample_density_at_logE_safe(_ws->densB, leB2);
                D_cmy[1] = sample_density_at_logE_safe(_ws->densG, leG2);
                D_cmy[2] = sample_density_at_logE_safe(_ws->densR, leR2);

                // NaN scrub on sampled densities (non-negative, finite)
                for (int i = 0; i < 3; ++i) {
                    float v = D_cmy[i];
                    if (!std::isfinite(v) || v < 0.0f) v = 0.0f;
                    if (v > 1000.0f) v = 1000.0f;
                    D_cmy[i] = v;
                }

                if (_printParams.bypass || !_printReady || !_prt) {
                    float XYZ[3];
                    if (_ws->hasBaseline && _ws->tablesView.hasBaseline) {
                        const float w = Spectral::neutral_blend_weight_from_DWG_rgb(rgbIn);
                        Spectral::dyes_to_XYZ_with_baseline_given_tables(_ws->tablesView, D_cmy, w, XYZ);
                    }
                    else {
                        Spectral::dyes_to_XYZ_given_tables(_ws->tablesView, D_cmy, XYZ);
                    }
                    Spectral::XYZ_to_DWG_linear(XYZ, rgbOut);
                }
                else {
                    // Spatial DIR path must continue from D_cmy to preserve blurred corrections.
                    const float wNeutral = Spectral::neutral_blend_weight_from_DWG_rgb(rgbIn);
                    thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;

                    // 1) Negative transmittance from corrected densities
                    Print::negative_T_from_dyes(*_ws, D_cmy, Tneg, wNeutral);

                    // Per-instance spectral shape to avoid global races and size mismatches
                    const int K = _ws->tablesView.K;
                    if (K <= 0) {
                        // Defensive early-out: cannot proceed with spectral integration
                        dstPix[0] = rgbOut[0];
                        dstPix[1] = rgbOut[1];
                        dstPix[2] = rgbOut[2];
                        if (_nComponents == 4) dstPix[3] = srcPix[3];
                        continue; // NO pointer arithmetic
                    }

                    // Ensure Tneg length is at least K (pad if needed)
                    if (int(Tneg.size()) < K) Tneg.resize(size_t(K), 0.0f);

                    // 2) Enlarger illuminant exposure (no print exposure yet)
                    Ee_expose.resize(size_t(K));
                    for (int i = 0; i < K; ++i) {
                        const float Ee = (_prt->illumEnlarger.linear.size() > size_t(i))
                            ? _prt->illumEnlarger.linear[i]
                            : 1.0f;
                        Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
                    }

                    // 3) Apply spectral dichroic filters with Y/M/C neutral values (agx density-as-OD)
                    Ee_filtered.resize(size_t(K));
                    auto blend_filter = [](float curveVal, float amount)->float {
                        const float a = std::isfinite(amount) ? std::clamp(amount, 0.0f, 8.0f) : 0.0f;
                        const float c = std::isfinite(curveVal) ? std::clamp(curveVal, 1e-6f, 1.0f) : 1.0f;
                        return std::pow(c, a);
                        };
                    for (int i = 0; i < K; ++i) {
                        const float fY = blend_filter(
                            (_prt->filterY.linear.size() > size_t(i)) ? _prt->filterY.linear[i] : 1.0f,
                            _printParams.yFilter);
                        const float fM = blend_filter(
                            (_prt->filterM.linear.size() > size_t(i)) ? _prt->filterM.linear[i] : 1.0f,
                            _printParams.mFilter);
                        const float fC = blend_filter(
                            (_prt->filterC.linear.size() > size_t(i)) ? _prt->filterC.linear[i] : 1.0f,
                            _printParams.cFilter);
                        const float fTotal = std::max(0.0f, std::min(1.0f, fY * fM * fC));
                        Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
                    }

                    // 4) Contract to per-channel raw exposures using print paper sensitivities
                    float raw[3];
                    Print::raw_exposures_from_filtered_light(
                        _prt->profile, Ee_filtered, raw, _ws->tablesView.deltaLambda);

                    // 5) Apply print exposure ONCE (agx: raw *= exposure) + green-only comp factor
                    raw[0] *= _printParams.exposure;
                    raw[1] *= _printParams.exposure;
                    raw[2] *= _printParams.exposure;
                    {
                        const float gFactor = std::isfinite(_printParams.exposureCompGFactor)
                            ? std::max(0.0f, _printParams.exposureCompGFactor)
                            : 1.0f;
                        raw[1] *= gFactor;
                    }

                    // 6) Map to print densities via calibrated per-channel logE offsets and DC curves
                    float D_print[3];
                    Print::print_densities_from_Eprint(_prt->profile, raw, D_print);

                    // 7) Print transmittance
                    Print::print_T_from_dyes(_prt->profile, D_print, Tprint);
                    if (int(Tprint.size()) < K) Tprint.resize(size_t(K), 0.0f);

                    // 8) Viewing illuminant and integration to DWG
                    Ee_viewed.resize(size_t(K));
                    for (int i = 0; i < K; ++i) {
                        const float Ev = (_prt->illumView.linear.size() > size_t(i))
                            ? _prt->illumView.linear[i]
                            : 1.0f;
                        Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
                    }
                    float XYZ[3];
                    Spectral::Ee_to_XYZ_given_tables(_ws->tablesView, Ee_viewed, XYZ);
                    Spectral::XYZ_to_DWG_linear(XYZ, rgbOut);
                }

                // Write out (no pointer increments)
                dstPix[0] = rgbOut[0];
                dstPix[1] = rgbOut[1];
                dstPix[2] = rgbOut[2];
                if (_nComponents == 4) dstPix[3] = srcPix[3];
            }
        }
        return;
    }

    // Fallback path
    for (int y = procWindow.y1; y < procWindow.y2; ++y) {
        if (_effect.abort()) break;

        if (_nComponents >= 3) {
            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (!dstPix || !srcPix) continue;

                float rgbIn[3] = { srcPix[0], srcPix[1], srcPix[2] };
                float rgbOut[3] = { rgbIn[0],  rgbIn[1],  rgbIn[2] };

                if (_wsReady && _ws) {
                    if (_printParams.bypass || !_printReady || !_prt) {
                        Scanner::simulate_scanner(
                            rgbIn, rgbOut,
                            _scannerParams, _dirRT, *_ws,
                            _exposureScale);
                    }
                    else {
                        Print::simulate_print_pixel(
                            rgbIn, _printParams,
                            *_prt, _dirRT, *_ws,
                            _exposureScale,
                            rgbOut);
                    }
                }

                dstPix[0] = rgbOut[0];
                dstPix[1] = rgbOut[1];
                dstPix[2] = rgbOut[2];
                if (_nComponents == 4) dstPix[3] = srcPix[3];
            }
        }        
        else if (_nComponents == 1) {
            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                float* dstPix = reinterpret_cast<float*>(_dstImg->getPixelAddress(x, y));
                const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
                if (dstPix && srcPix) dstPix[0] = srcPix[0];
            }
        }
    }
}
