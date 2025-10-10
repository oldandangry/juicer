// JuicerProcessing.cpp

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

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

namespace {

    struct SpatialDIRWorkspace {
        std::vector<float> logE_B, logE_G, logE_R;
        std::vector<float> corrY, corrM, corrC;
        std::vector<float> corrYBlur, corrMBlur, corrCBlur;
        std::vector<float> tmp; // reused for separable blur scratch
    };

    template <typename FetchRGB, typename AbortCheck>
    void buildSpatialDIRCorrections(
        int width, int height,
        const WorkingState& ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        FetchRGB&& fetchRGB,
        AbortCheck&& abortCheck,
        SpatialDIRWorkspace& work)
    {
        const size_t total = size_t(width) * size_t(height);
        work.logE_B.assign(total, 0.0f);
        work.logE_G.assign(total, 0.0f);
        work.logE_R.assign(total, 0.0f);
        work.corrY.assign(total, 0.0f);
        work.corrM.assign(total, 0.0f);
        work.corrC.assign(total, 0.0f);

        auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };

        auto select_exposure_curve = [](const Spectral::Curve& neg, const Spectral::Curve& base)
            -> const Spectral::Curve& {
            return neg.linear.empty() ? base : neg;
            };
        const Spectral::Curve& sensB_forExposure = select_exposure_curve(ws.negSensB, ws.sensB);
        const Spectral::Curve& sensG_forExposure = select_exposure_curve(ws.negSensG, ws.sensG);
        const Spectral::Curve& sensR_forExposure = select_exposure_curve(ws.negSensR, ws.sensR);

        // Pass A: sample exposures, convert to logE, compute DIR corrections per pixel.
        for (int yy = 0; yy < height; ++yy) {
            if (abortCheck()) break;
            for (int xx = 0; xx < width; ++xx) {
                const size_t idx = size_t(yy) * size_t(width) + size_t(xx);
                float rgbIn[3] = { 0.0f, 0.0f, 0.0f };
                if (!fetchRGB(xx, yy, rgbIn)) {
                    work.logE_B[idx] = work.logE_G[idx] = work.logE_R[idx] = 0.0f;
                    work.corrY[idx] = work.corrM[idx] = work.corrC[idx] = 0.0f;
                    continue;
                }

                float E[3];
                const SpectralTables* tablesSPD =
                    (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
                Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
                    rgbIn, E, 1.0f,
                    tablesSPD,
                    (ws.spdReady ? ws.spdSInv : nullptr),
                    ws.spdReady,
                    sensB_forExposure,
                    sensG_forExposure,
                    sensR_forExposure);

                const float sExp = (std::isfinite(exposureScale) ? std::max(0.0f, exposureScale) : 1.0f);
                if (sExp != 1.0f) {
                    E[0] *= sExp; E[1] *= sExp; E[2] *= sExp;
                }

                float leB = std::log10(std::max(0.0f, E[0]) + 1e-10f) + ws.logEOffB;
                float leG = std::log10(std::max(0.0f, E[1]) + 1e-10f) + ws.logEOffG;
                float leR = std::log10(std::max(0.0f, E[2]) + 1e-10f) + ws.logEOffR;

                leB = clamp_logE_to_curve(ws.densB, leB);
                leG = clamp_logE_to_curve(ws.densG, leG);
                leR = clamp_logE_to_curve(ws.densR, leR);

                work.logE_B[idx] = leB;
                work.logE_G[idx] = leG;
                work.logE_R[idx] = leR;

                float D_Y = sample_density_at_logE_safe(ws.densB, leB);
                float D_M = sample_density_at_logE_safe(ws.densG, leG);
                float D_C = sample_density_at_logE_safe(ws.densR, leR);

                float aCorr[3];
                Couplers::ApplyInputLogE io{ { leB, leG, leR }, { D_Y, D_M, D_C } };
                Couplers::compute_logE_corrections(io, dirRT, aCorr);
                for (float& v : aCorr) {
                    if (!std::isfinite(v)) v = 0.0f;
                }
                work.corrY[idx] = aCorr[0];
                work.corrM[idx] = aCorr[1];
                work.corrC[idx] = aCorr[2];
            }
        }

        // Blur corrections spatially (shared between preview + render path)
        std::vector<float> kernel;
        JuicerProc::buildGaussianKernel(dirRT.spatialSigmaPixels, kernel);
        JuicerProc::blurChannelSeparable(work.corrY, work.tmp, work.corrYBlur, width, height, kernel);
        JuicerProc::blurChannelSeparable(work.corrM, work.tmp, work.corrMBlur, width, height, kernel);
        JuicerProc::blurChannelSeparable(work.corrC, work.tmp, work.corrCBlur, width, height, kernel);

        auto scrubClamp = [](std::vector<float>& v) {
            for (float& t : v) {
                if (!std::isfinite(t)) t = 0.0f;
                if (t < -10.0f) t = -10.0f;
                if (t > 10.0f) t = 10.0f;
            }
            };
        scrubClamp(work.corrYBlur);
        scrubClamp(work.corrMBlur);
        scrubClamp(work.corrCBlur);
    }
}


namespace JuicerProc {
    // Note: print preview should be measured at unit scene exposure for compensation;
    // caller passes exposureScale=1.0f to avoid compounding EV.
    void renderPrintPreviewToBuffer(
        const OFX::Image* src, const OfxRectI& srcBounds,
        float* outRGB, int outW, int outH,
        const Scanner::Params& scanPrm,
        const Print::Params& prm,
        const Print::Runtime* prt,
        const WorkingState* ws,
        const Couplers::Runtime& dirRT,
        float exposureScale,
        const OutputEncoding::Params& outputEncoding)
    {
        if (!src || !outRGB || outW <= 0 || outH <= 0 || !ws) return;

        const int W = srcBounds.x2 - srcBounds.x1;
        const int H = srcBounds.y2 - srcBounds.y1;
        if (W <= 0 || H <= 0) return;

        const bool useScanner = prm.bypass || !prt;
        const bool curvesReady = curve_ok(ws->densB) && curve_ok(ws->densG) && curve_ok(ws->densR);
        const bool doSpatial = (!useScanner) && dirRT.active && std::isfinite(dirRT.spatialSigmaPixels)
            && dirRT.spatialSigmaPixels > 0.5f && curvesReady;
        float midgrayScale[3] = { 1.0f, 1.0f, 1.0f };
        float kMid_spectral = 1.0f;
        if (!useScanner && prt) {
            const float exposureCompScale = prm.exposureCompensationEnabled
                ? prm.exposureCompensationScale
                : 1.0f;
            kMid_spectral = Print::compute_exposure_factor_midgray(*ws, *prt, prm, exposureCompScale, midgrayScale);
        }

        const size_t totalPix = size_t(outW) * size_t(outH);
        std::vector<float> sampledRGB(totalPix * 3, 0.0f);
        std::vector<uint8_t> sampleValid(totalPix, 0);

        // Map preview pixel centers to source coordinates (bilinear interpolation)
        for (int yy = 0; yy < outH; ++yy) {
            for (int xx = 0; xx < outW; ++xx) {
                const float fx = (xx + 0.5f) / float(outW);
                const float fy = (yy + 0.5f) / float(outH);

                float sampleX = srcBounds.x1 + fx * float(W);
                float sampleY = srcBounds.y1 + fy * float(H);

                sampleX = std::min(std::max(sampleX, float(srcBounds.x1)), float(srcBounds.x2 - 1));
                sampleY = std::min(std::max(sampleY, float(srcBounds.y1)), float(srcBounds.y2 - 1));

                int x0 = int(std::floor(sampleX));
                int y0 = int(std::floor(sampleY));
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float tx = sampleX - float(x0);
                float ty = sampleY - float(y0);

                if (x1 >= srcBounds.x2) {
                    x1 = x0;
                    tx = 0.0f;
                }
                if (y1 >= srcBounds.y2) {
                    y1 = y0;
                    ty = 0.0f;
                }

                const float* src00 = reinterpret_cast<const float*>(src->getPixelAddress(x0, y0));
                const float* src10 = reinterpret_cast<const float*>(src->getPixelAddress(x1, y0));
                const float* src01 = reinterpret_cast<const float*>(src->getPixelAddress(x0, y1));
                const float* src11 = reinterpret_cast<const float*>(src->getPixelAddress(x1, y1));

                const float w00 = (1.0f - tx) * (1.0f - ty);
                const float w10 = tx * (1.0f - ty);
                const float w01 = (1.0f - tx) * ty;
                const float w11 = tx * ty;

                float rgbIn[3] = { 0.0f, 0.0f, 0.0f };

                if (src00) {
                    rgbIn[0] += src00[0] * w00;
                    rgbIn[1] += src00[1] * w00;
                    rgbIn[2] += src00[2] * w00;
                }
                if (src10) {
                    rgbIn[0] += src10[0] * w10;
                    rgbIn[1] += src10[1] * w10;
                    rgbIn[2] += src10[2] * w10;
                }
                if (src01) {
                    rgbIn[0] += src01[0] * w01;
                    rgbIn[1] += src01[1] * w01;
                    rgbIn[2] += src01[2] * w01;
                }
                if (src11) {
                    rgbIn[0] += src11[0] * w11;
                    rgbIn[1] += src11[1] * w11;
                    rgbIn[2] += src11[2] * w11;
                }            
                

                const bool hasSample = (src00 || src10 || src01 || src11);
                const size_t idx = size_t(yy) * size_t(outW) + size_t(xx);
                if (hasSample) {
                    sampleValid[idx] = 1;
                    sampledRGB[idx * 3 + 0] = rgbIn[0];
                    sampledRGB[idx * 3 + 1] = rgbIn[1];
                    sampledRGB[idx * 3 + 2] = rgbIn[2];
                }
            }
        }

        SpatialDIRWorkspace previewSpatial;
        const bool haveSpatial = doSpatial;
        if (haveSpatial) {
            auto fetchSample = [&](int xx, int yy, float rgb[3])->bool {
                const size_t idx = size_t(yy) * size_t(outW) + size_t(xx);
                if (!sampleValid[idx]) return false;
                rgb[0] = sampledRGB[idx * 3 + 0];
                rgb[1] = sampledRGB[idx * 3 + 1];
                rgb[2] = sampledRGB[idx * 3 + 2];
                return true;
                };
            auto abortNever = []() -> bool { return false; };
            buildSpatialDIRCorrections(
                outW, outH,
                *ws,
                dirRT,
                exposureScale,
                fetchSample,
                abortNever,
                previewSpatial);
        }

        for (int yy = 0; yy < outH; ++yy) {
            for (int xx = 0; xx < outW; ++xx) {
                const size_t idx = size_t(yy) * size_t(outW) + size_t(xx);
                float* dstPix = &outRGB[idx * 3];
                if (!sampleValid[idx]) {
                    dstPix[0] = dstPix[1] = dstPix[2] = 0.0f;
                    continue;
                }

                float rgbIn[3] = {
                    sampledRGB[idx * 3 + 0],
                    sampledRGB[idx * 3 + 1],
                    sampledRGB[idx * 3 + 2]
                };

                float rgbOut[3] = { rgbIn[0], rgbIn[1], rgbIn[2] };

                if (useScanner) {
                    Scanner::simulate_scanner(
                        rgbIn, rgbOut,
                        scanPrm, dirRT, *ws,
                        exposureScale);
                }
                else if (haveSpatial && prt) {
                    float leB2 = previewSpatial.logE_B[idx] - previewSpatial.corrYBlur[idx];
                    float leG2 = previewSpatial.logE_G[idx] - previewSpatial.corrMBlur[idx];
                    float leR2 = previewSpatial.logE_R[idx] - previewSpatial.corrCBlur[idx];

                    auto clamp_to = [](float le, const Spectral::Curve& c)->float {
                        if (c.lambda_nm.empty()) return le;
                        const float xmin = c.lambda_nm.front();
                        const float xmax = c.lambda_nm.back();
                        if (!std::isfinite(le)) return xmin;
                        return std::min(std::max(le, xmin), xmax);
                        };
                    leB2 = clamp_to(leB2, ws->densB);
                    leG2 = clamp_to(leG2, ws->densG);
                    leR2 = clamp_to(leR2, ws->densR);

                    float D_cmy[3];
                    D_cmy[0] = sample_density_at_logE_safe(ws->densB, leB2);
                    D_cmy[1] = sample_density_at_logE_safe(ws->densG, leG2);
                    D_cmy[2] = sample_density_at_logE_safe(ws->densR, leR2);

                    for (int i = 0; i < 3; ++i) {
                        float v = D_cmy[i];
                        if (!std::isfinite(v) || v < 0.0f) v = 0.0f;
                        if (v > 1000.0f) v = 1000.0f;
                        D_cmy[i] = v;
                    }

                    thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
                    thread_local std::vector<float> Tpreflash, Ee_preflash;

                    Print::negative_T_from_dyes(*ws, D_cmy, Tneg);

                    const int K = std::min(ws->tablesView.K, ws->tablesPrint.K);
                    if (K > 0) {
                        if (int(Tneg.size()) < K) Tneg.resize(size_t(K), 0.0f);

                        Ee_expose.resize(size_t(K));
                        for (int i = 0; i < K; ++i) {
                            const float Ee = (prt->illumEnlarger.linear.size() > size_t(i))
                                ? prt->illumEnlarger.linear[i]
                                : 1.0f;
                            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
                        }

                        Ee_filtered.resize(size_t(K));
                        const float yAmount = Print::compose_dichroic_amount(prt->neutralY, prm.yFilter);
                        const float mAmount = Print::compose_dichroic_amount(prt->neutralM, prm.mFilter);
                        const float cAmount = Print::compose_dichroic_amount(prt->neutralC, 0.0f);
                        for (int i = 0; i < K; ++i) {
                            const float fY = Print::blend_dichroic_filter_linear(
                                (prt->filterY.linear.size() > size_t(i)) ? prt->filterY.linear[i] : 1.0f,
                                yAmount);
                            const float fM = Print::blend_dichroic_filter_linear(
                                (prt->filterM.linear.size() > size_t(i)) ? prt->filterM.linear[i] : 1.0f,
                                mAmount);
                            const float fC = Print::blend_dichroic_filter_linear(
                                (prt->filterC.linear.size() > size_t(i)) ? prt->filterC.linear[i] : 1.0f,
                                cAmount);
                            const float fTotal = fY * fM * fC;
                            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
                        }

                        float raw[3];
                        Print::raw_exposures_from_filtered_light(
                            prt->profile, Ee_filtered, raw);

                        const float rawScale = prm.exposure * kMid_spectral;
                        raw[0] *= rawScale * midgrayScale[0];
                        raw[1] *= rawScale * midgrayScale[1];
                        raw[2] *= rawScale * midgrayScale[2];

                        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
                            float rawPre[3];
                            Print::compute_preflash_raw(*prt, *ws, Tpreflash, Ee_preflash, rawPre);
                            raw[0] += rawPre[0] * prm.preflashExposure * midgrayScale[0];
                            raw[1] += rawPre[1] * prm.preflashExposure * midgrayScale[1];
                            raw[2] += rawPre[2] * prm.preflashExposure * midgrayScale[2];
                        }

                        float D_print[3];
                        Print::print_densities_from_Eprint(prt->profile, raw, D_print);

                        Print::print_T_from_dyes(prt->profile, D_print, Tprint);
                        if (int(Tprint.size()) < K) Tprint.resize(size_t(K), 0.0f);

                        Ee_viewed.resize(size_t(K));
                        for (int i = 0; i < K; ++i) {
                            const float Ev = (prt->illumView.linear.size() > size_t(i))
                                ? prt->illumView.linear[i]
                                : 1.0f;
                            Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
                        }
                        float XYZ[3];
                        Spectral::Ee_to_XYZ_given_tables(ws->tablesPrint, Ee_viewed, XYZ);
                        Spectral::XYZ_to_DWG_linear_adapted(ws->tablesPrint, XYZ, rgbOut);
                        rgbOut[0] = std::max(0.0f, rgbOut[0]);
                        rgbOut[1] = std::max(0.0f, rgbOut[1]);
                        rgbOut[2] = std::max(0.0f, rgbOut[2]);
                    }
                    else {
                        // Fallback to non-spatial print simulation if tables are invalid.
                        Print::simulate_print_pixel(
                            rgbIn, prm,
                            *prt, dirRT, *ws,
                            exposureScale,
                            kMid_spectral,
                            midgrayScale,
                            rgbOut);
                    }
                }
                else {                    
                    Print::simulate_print_pixel(
                        rgbIn, prm,
                        *prt, dirRT, *ws,
                        /*exposureScale*/ exposureScale,
                        kMid_spectral,
                        midgrayScale,
                        rgbOut);
                }

                OutputEncoding::applyEncoding(outputEncoding, rgbOut);
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
    , _outputEncoding{}
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

void JuicerProcessor::setOutputEncoding(const OutputEncoding::Params& p) {
    _outputEncoding = p;
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
        SpatialDIRWorkspace dirWorkspace;
        auto fetchRGB = [&](int xx, int yy, float rgb[3])->bool {
            const int x = procWindow.x1 + xx;
            const int y = procWindow.y1 + yy;
            const float* srcPix = reinterpret_cast<const float*>(_srcImg->getPixelAddress(x, y));
            if (!srcPix) return false;
            rgb[0] = srcPix[0];
            rgb[1] = srcPix[1];
            rgb[2] = srcPix[2];
            return true;
        };
        auto abortCheck = [&]() -> bool { return _effect.abort(); };
        buildSpatialDIRCorrections(
            tileW, tileH,
            *_ws,
            _dirRT,
            _exposureScale,
            fetchRGB,
            abortCheck,
            dirWorkspace);

        // Precompute spectral midgray factor once (AgX parity), constant across the tile.
        const float exposureCompScale = _printParams.exposureCompensationEnabled
            ? _printParams.exposureCompensationScale
            : 1.0f;
        float midgrayScale[3] = { 1.0f, 1.0f, 1.0f };
        const float kMid_spectral = Print::compute_exposure_factor_midgray(*_ws, *_prt, _printParams, exposureCompScale, midgrayScale);

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
                float leB2 = dirWorkspace.logE_B[idx] - dirWorkspace.corrYBlur[idx];
                float leG2 = dirWorkspace.logE_G[idx] - dirWorkspace.corrMBlur[idx];
                float leR2 = dirWorkspace.logE_R[idx] - dirWorkspace.corrCBlur[idx];

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
                    const SpectralTables* tables = nullptr;
                    if (_ws->tablesScan.K > 0) {
                        tables = &_ws->tablesScan;
                    }
                    else if (_ws->tablesView.K > 0) {
                        tables = &_ws->tablesView;
                    }

                    if (tables) {
                        float XYZ[3] = { 0.0f, 0.0f, 0.0f };
                        const bool useBaseline = _ws->hasBaseline && tables->hasBaseline;
                        if (useBaseline) {
                            Spectral::dyes_to_XYZ_with_baseline_given_tables(*tables, D_cmy, XYZ);
                        }
                        else {
                            Spectral::dyes_to_XYZ_given_tables(*tables, D_cmy, XYZ);                            
                        }
                        const float exposureScaleSafe =
                            (std::isfinite(_exposureScale) && _exposureScale > 0.0f)
                            ? _exposureScale
                            : 1.0f;
                        const SpectralTables* tablesSPD =
                            (_ws->spdReady && _ws->tablesRef.K > 0) ? &_ws->tablesRef : nullptr;
                        const Spectral::Curve& sensB_forExposure =
                            _ws->negSensB.linear.empty() ? _ws->sensB : _ws->negSensB;
                        const Spectral::Curve& sensG_forExposure =
                            _ws->negSensG.linear.empty() ? _ws->sensG : _ws->negSensG;
                        const Spectral::Curve& sensR_forExposure =
                            _ws->negSensR.linear.empty() ? _ws->sensR : _ws->negSensR;
                        const float autoGain = Scanner::compute_auto_exposure_gain(
                            _scannerParams,
                            *_ws,
                            *tables,
                            tablesSPD,
                            sensB_forExposure,
                            sensG_forExposure,
                            sensR_forExposure,
                            exposureScaleSafe);
                        XYZ[0] *= autoGain;
                        XYZ[1] *= autoGain;
                        XYZ[2] *= autoGain;
                        Spectral::XYZ_to_DWG_linear_adapted(*tables, XYZ, rgbOut);
                        rgbOut[0] = std::max(0.0f, rgbOut[0]);
                        rgbOut[1] = std::max(0.0f, rgbOut[1]);
                        rgbOut[2] = std::max(0.0f, rgbOut[2]);
                    }
                }
                else {
                    // Spatial DIR path must continue from D_cmy to preserve blurred corrections.
                    thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
                    thread_local std::vector<float> Tpreflash, Ee_preflash;

                    // 1) Negative transmittance from corrected densities
                    Print::negative_T_from_dyes(*_ws, D_cmy, Tneg);

                    // Per-instance spectral shape to avoid global races and size mismatches
                    const int K = std::min(_ws->tablesView.K, _ws->tablesPrint.K);
                    if (K <= 0) {
                        // Defensive early-out: cannot proceed with spectral integration
                        OutputEncoding::applyEncoding(_outputEncoding, rgbOut);
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
                    const float yAmount = Print::compose_dichroic_amount(_prt->neutralY, _printParams.yFilter);
                    const float mAmount = Print::compose_dichroic_amount(_prt->neutralM, _printParams.mFilter);
                    const float cAmount = Print::compose_dichroic_amount(_prt->neutralC, 0.0f);
                    for (int i = 0; i < K; ++i) {
                        const float fY = Print::blend_dichroic_filter_linear(
                            (_prt->filterY.linear.size() > size_t(i)) ? _prt->filterY.linear[i] : 1.0f,
                            yAmount);
                        const float fM = Print::blend_dichroic_filter_linear(
                            (_prt->filterM.linear.size() > size_t(i)) ? _prt->filterM.linear[i] : 1.0f,
                            mAmount);
                        const float fC = Print::blend_dichroic_filter_linear(
                            (_prt->filterC.linear.size() > size_t(i)) ? _prt->filterC.linear[i] : 1.0f,
                            cAmount);
                        const float fTotal = fY * fM * fC;
                        Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
                    }

                    // 4) Contract to per-channel raw exposures using print paper sensitivities
                    float raw[3];
                    Print::raw_exposures_from_filtered_light(
                        _prt->profile, Ee_filtered, raw);

                    // 5) Apply print exposure ONCE (agx: raw *= exposure) + midgray compensation (vector)
                    const float rawScale = _printParams.exposure * kMid_spectral;
                    raw[0] *= rawScale * midgrayScale[0];
                    raw[1] *= rawScale * midgrayScale[1];
                    raw[2] *= rawScale * midgrayScale[2];

                    if (std::isfinite(_printParams.preflashExposure) && _printParams.preflashExposure > 0.0f) {
                        float rawPre[3];
                        Print::compute_preflash_raw(*_prt, *_ws, Tpreflash, Ee_preflash, rawPre);
                        raw[0] += rawPre[0] * _printParams.preflashExposure * midgrayScale[0];
                        raw[1] += rawPre[1] * _printParams.preflashExposure * midgrayScale[1];
                        raw[2] += rawPre[2] * _printParams.preflashExposure * midgrayScale[2];
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
                    Spectral::Ee_to_XYZ_given_tables(_ws->tablesPrint, Ee_viewed, XYZ);
                    Spectral::XYZ_to_DWG_linear_adapted(_ws->tablesPrint, XYZ, rgbOut);
                    rgbOut[0] = std::max(0.0f, rgbOut[0]);
                    rgbOut[1] = std::max(0.0f, rgbOut[1]);
                    rgbOut[2] = std::max(0.0f, rgbOut[2]);
                }

                // Write out (no pointer increments)
                OutputEncoding::applyEncoding(_outputEncoding, rgbOut);
                dstPix[0] = rgbOut[0];
                dstPix[1] = rgbOut[1];
                dstPix[2] = rgbOut[2];
                if (_nComponents == 4) dstPix[3] = srcPix[3];
            }
        }
        return;
    }

    float midgrayScale[3] = { 1.0f, 1.0f, 1.0f };
    float kMid_spectral = 0.0f;
    if (_wsReady && _ws && _printReady && _prt && !_printParams.bypass) {
        const float exposureCompScale = _printParams.exposureCompensationEnabled
            ? _printParams.exposureCompensationScale
            : 1.0f;
        kMid_spectral = Print::compute_exposure_factor_midgray(*_ws, *_prt, _printParams, exposureCompScale, midgrayScale);
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
                            kMid_spectral,
                            midgrayScale,
                            rgbOut);
                    }
                }

                OutputEncoding::applyEncoding(_outputEncoding, rgbOut);
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
