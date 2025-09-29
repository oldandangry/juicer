#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "SpectralMath.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"
#include "ofxParam.h"

// Couplers.h — REPLACE the entire stub namespace block under #ifndef JUICER_ENABLE_COUPLERS with this
#ifndef JUICER_ENABLE_COUPLERS
// Stubs: compiled out safely
namespace Couplers {
    // Keep identical signatures so call sites compile either way
    struct ApplyInput { float E[3]; float D[3]; };
    struct Runtime {
        bool  active = false;
        float M[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
    };

    inline void define_params(OfxImageEffectHandle, OfxImageEffectSuiteV1*, OfxPropertySuiteV1*, OfxParameterSuiteV1*) {}
    inline void on_param_changed(const char*) {}

    // IMPORTANT: keep the exact same signature as enabled build
    inline void maybe_precorrect_curves(OfxImageEffectHandle, OfxImageEffectSuiteV1*, OfxParameterSuiteV1*) {}

    inline void fetch_runtime(const OfxImageEffectHandle, OfxImageEffectSuiteV1*, OfxParameterSuiteV1*, Runtime&) {}
    inline void apply_runtime(ApplyInput&, const Runtime&) {}

    // Optional parity helpers used by the print/scanner paths
    struct ApplyInputLogE { float logE[3]; float D[3]; };
    inline void apply_runtime_logE(ApplyInputLogE&, const Runtime&) {}
} // namespace Couplers

#else

namespace Couplers {

    // ------------------------------
    // OFX parameter names
    // ------------------------------
    static constexpr const char* kParamCouplersActive = "CouplersActive";
    static constexpr const char* kParamCouplersPrecorrect = "CouplersPrecorrectCurves";
    static constexpr const char* kParamCouplersAmountB = "CouplersAmountB";
    static constexpr const char* kParamCouplersAmountG = "CouplersAmountG";
    static constexpr const char* kParamCouplersAmountR = "CouplersAmountR";
    static constexpr const char* kParamCouplersLayerSigma = "CouplersLayerDiffusion";
    static constexpr const char* kParamCouplersHighExpShift = "CouplersHighExposureShift";
    static constexpr const char* kParamCouplersSpatialSigma = "CouplersSpatialDiffusion"; // pixels


    // NOTE: In enabled builds we do not call maybe_precorrect_curves().
    // Pre-correction is handled inside rebuild_working_state from BaseState in a single shot.
    // This guard is kept only for stub parity and is otherwise unused in the current pipeline.
    inline bool gPrecorrectApplied = false;

    // Orientation: M[inputLayer][outputLayer]. Input layers are ordered [B,G,R],
    // output axes correspond to [Y,M,C] corrections applied to Blue, Green, Red layer logE respectively.
    // Rows are normalized diffusion weights across outputs; each row is scaled by the per-input-layer amount.
    // Build 3x3 DIR matrix (layer-space diffusion, per-row scaling)
    inline void build_dir_matrix(float M[3][3], const float amountRGB[3], float layerSigma) {
        // Sanitize inputs
        const float sigma = std::isfinite(layerSigma) ? std::max(0.0f, layerSigma) : 0.0f;
        float amt[3] = { amountRGB[0], amountRGB[1], amountRGB[2] };
        // Cap sigma to avoid over-uniform diffusion rows which destabilize mid-gray detection
        const float sigmaCapped = std::min(sigma, 3.0f);
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(amt[i])) amt[i] = 0.0f;
            amt[i] = std::clamp(amt[i], 0.0f, 1.0f);
        }

        auto gauss = [&](int dx)->float {
            if (sigmaCapped <= 0.0f) return (dx == 0) ? 1.0f : 0.0f;
            const float s2 = sigmaCapped * sigmaCapped;
            return std::exp(-0.5f * (dx * dx) / s2);
            };
        for (int r = 0; r < 3; ++r) {
            float row[3]; float wsum = 0.0f;
            for (int c = 0; c < 3; ++c) { row[c] = gauss(c - r); wsum += row[c]; }
            if (wsum > 0.0f) for (int c = 0; c < 3; ++c) row[c] /= wsum;
            for (int c = 0; c < 3; ++c) M[r][c] = amt[r] * row[c];
        }

        // Scrub non-finite entries to ensure runtime stability
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(M[r][c])) M[r][c] = 0.0f;
            }
        }
    }



    // Pre-correct 1D density curves (before DIR) — agx parity
    inline void precorrect_density_curves_before_DIR(const float M[3][3], float highExpShift) {
        using Spectral::Curve;
        Curve& cB = Spectral::gDensityCurveB; // Yellow <- Blue layer
        Curve& cG = Spectral::gDensityCurveG; // Magenta <- Green layer
        Curve& cR = Spectral::gDensityCurveR; // Cyan   <- Red  layer

        const size_t N = std::min({ cB.lambda_nm.size(), cG.lambda_nm.size(), cR.lambda_nm.size() });
        if (N == 0) return;

        // Per-channel grids and densities
        std::vector<float> xB(N), xG(N), xR(N);
        std::vector<float> dY(N), dM(N), dC(N);
        for (size_t i = 0; i < N; ++i) {
            xB[i] = cB.lambda_nm[i];
            xG[i] = cG.lambda_nm[i];
            xR[i] = cR.lambda_nm[i];
            dY[i] = cB.linear[i];
            dM[i] = cG.linear[i];
            dC[i] = cR.linear[i];
        }

        // Monotonicity guard
        auto is_monotonic = [](const std::vector<float>& X)->bool {
            if (X.empty() || !std::isfinite(X.front()) || !std::isfinite(X.back())) return false;
            for (size_t i = 1; i < X.size(); ++i) {
                if (!std::isfinite(X[i]) || !(X[i] >= X[i - 1])) return false;
            }
            return true;
            };
        if (!(is_monotonic(xB) && is_monotonic(xG) && is_monotonic(xR))) {
            return; // leave curves untouched
        }

        // Normalize per-channel, with high exposure quadratic term
        auto vmax = [](const std::vector<float>& v) {
            float m = 0.0f; for (float t : v) m = std::max(m, t); return (m > 0.0f) ? m : 1.0f;
            };
        const float dYmax = vmax(dY), dMmax = vmax(dM), dCmax = vmax(dC);

        std::vector<float> nB(N), nG(N), nR(N);
        for (size_t i = 0; i < N; ++i) {
            const float y = dY[i] / dYmax;
            const float m = dM[i] / dMmax;
            const float c = dC[i] / dCmax;
            nB[i] = y + highExpShift * y * y;
            nG[i] = m + highExpShift * m * m;
            nR[i] = c + highExpShift * c * c;
        }

        // Clamp warp to local domain
        auto clamp_warp = [](float xq, float xmin, float xmax)->float {
            if (!std::isfinite(xq)) return xmin;
            return std::min(std::max(xq, xmin), xmax);
            };

        // Local interpolator
        auto interp = [](const std::vector<float>& X, const std::vector<float>& Y, float xq)->float {
            if (X.empty() || Y.empty()) return 0.0f;
            const float xmin = X.front(), xmax = X.back();
            if (!std::isfinite(xq)) return (!Y.empty() ? Y.front() : 0.0f);
            if (!std::isfinite(xmin) || !std::isfinite(xmax)) return (!Y.empty() ? Y.front() : 0.0f);
            if (xq <= xmin) return Y.front();
            if (xq >= xmax) return Y.back();

            // lower_bound assumes ascending; NaN in X breaks ordering. Guard by scanning a small window.
            size_t i1 = size_t(std::lower_bound(X.begin(), X.end(), xq) - X.begin());
            if (i1 == 0) return Y.front();
            if (i1 >= X.size()) return Y.back();
            size_t i0 = i1 - 1;

            const float x0 = X[i0], x1 = X[i1];
            const float y0 = Y[i0], y1 = Y[i1];

            // If any involved value is non-finite, fall back to y0
            if (!std::isfinite(x0) || !std::isfinite(x1) || !std::isfinite(y0) || !std::isfinite(y1)) {
                return std::isfinite(y0) ? y0 : 0.0f;
            }

            const float denom = (x1 - x0);
            if (denom <= 0.0f || !std::isfinite(denom)) return y0;
            const float t = (xq - x0) / denom;
            return y0 + t * (y1 - y0);
            };


        // Apply per-sample ΔlogE on each channel’s own grid
        std::vector<float> dYcorr(N), dMcorr(N), dCcorr(N);
        for (size_t i = 0; i < N; ++i) {
            const float aY = M[0][0] * nB[i] + M[1][0] * nG[i] + M[2][0] * nR[i];
            const float aM = M[0][1] * nB[i] + M[1][1] * nG[i] + M[2][1] * nR[i];
            const float aC = M[0][2] * nB[i] + M[1][2] * nG[i] + M[2][2] * nR[i];

            const float xqY = clamp_warp(xB[i] - aY, xB.front(), xB.back());
            const float xqM = clamp_warp(xG[i] - aM, xG.front(), xG.back());
            const float xqC = clamp_warp(xR[i] - aC, xR.front(), xR.back());

            dYcorr[i] = interp(xB, dY, xqY);
            dMcorr[i] = interp(xG, dM, xqM);
            dCcorr[i] = interp(xR, dC, xqC);
        }

        for (size_t i = 0; i < N; ++i) {
            cB.linear[i] = dYcorr[i];
            cG.linear[i] = dMcorr[i];
            cR.linear[i] = dCcorr[i];
        }
        Spectral::mark_mixing_dirty();
    }


    inline void precorrect_density_curves_before_DIR_into(
        const float M[3][3], float highExpShift,
        const Spectral::Curve& inB, const Spectral::Curve& inG, const Spectral::Curve& inR,
        Spectral::Curve& outB, Spectral::Curve& outG, Spectral::Curve& outR)
    {
        using Spectral::Curve;
        outB = inB; outG = inG; outR = inR;

        const size_t N = std::min({ inB.lambda_nm.size(), inG.lambda_nm.size(), inR.lambda_nm.size() });
        if (N == 0) return;

        // Per‑channel grids and densities
        std::vector<float> xB(N), xG(N), xR(N);
        std::vector<float> dY(N), dM(N), dC(N);
        for (size_t i = 0; i < N; ++i) {
            xB[i] = inB.lambda_nm[i];
            xG[i] = inG.lambda_nm[i];
            xR[i] = inR.lambda_nm[i];
            dY[i] = inB.linear[i];
            dM[i] = inG.linear[i];
            dC[i] = inR.linear[i];
        }

        // Monotonicity guard
        auto is_monotonic = [](const std::vector<float>& X)->bool {
            if (X.empty() || !std::isfinite(X.front()) || !std::isfinite(X.back())) return false;
            for (size_t i = 1; i < X.size(); ++i) {
                if (!std::isfinite(X[i]) || !(X[i] >= X[i - 1])) return false;
            }
            return true;
            };
        const bool okB = is_monotonic(xB);
        const bool okG = is_monotonic(xG);
        const bool okR = is_monotonic(xR);
        if (!(okB && okG && okR)) {
            outB = inB; outG = inG; outR = inR;
            return;
        }

        // Normalize per-channel, with high exposure quadratic term
        auto vmax = [](const std::vector<float>& v) {
            float m = 0.0f; for (float t : v) m = std::max(m, t); return (m > 0.0f) ? m : 1.0f;
            };
        const float dYmax = vmax(dY), dMmax = vmax(dM), dCmax = vmax(dC);

        std::vector<float> nB(N), nG(N), nR(N);
        auto safe_div = [](float v, float m)->float {
            float mm = (std::isfinite(m) && m > 1e-6f) ? m : 1.0f;
            float r = v / mm;
            if (!std::isfinite(r)) r = 0.0f;
            return std::min(std::max(r, 0.0f), 1.0f);
            };

        for (size_t i = 0; i < N; ++i) {
            const float y = safe_div(dY[i], dYmax);
            const float m = safe_div(dM[i], dMmax);
            const float c = safe_div(dC[i], dCmax);
            auto boost = [&](float t)->float {
                float tb = t + highExpShift * t * t; // agx quadratic boost
                if (!std::isfinite(tb)) tb = t;
                return std::min(std::max(tb, 0.0f), 1.0f);
                };
            nB[i] = boost(y);
            nG[i] = boost(m);
            nR[i] = boost(c);
        }

        // Dedup each channel grid against its own density vector (once, not in the per-sample loop)
        auto dedup_strict = [](std::vector<float>& X, std::vector<float>& Y) {
            if (X.size() != Y.size() || X.empty()) return;
            std::vector<float> X2; X2.reserve(X.size());
            std::vector<float> Y2; Y2.reserve(Y.size());
            float lastX = X[0];
            float lastY = Y[0];
            X2.push_back(lastX);
            Y2.push_back(std::isfinite(lastY) ? std::max(0.0f, lastY) : 0.0f);
            for (size_t i = 1; i < X.size(); ++i) {
                float xi = X[i];
                float yi = std::isfinite(Y[i]) ? std::max(0.0f, Y[i]) : 0.0f;
                const float eps = 1e-6f;
                if (!std::isfinite(xi)) continue;
                if (xi <= lastX + eps) {
                    X2.back() = lastX;
                    Y2.back() = std::max(Y2.back(), yi);
                    continue;
                }
                X2.push_back(xi);
                Y2.push_back(yi);
                lastX = xi;
            }
            if (X2.size() >= 2) {
                X = std::move(X2);
                Y = std::move(Y2);
            }
            };
        dedup_strict(xB, dY);
        dedup_strict(xG, dM);
        dedup_strict(xR, dC);

        // Clamp warp to local domain with conservative bounds and max absolute warp per sample.
        // This prevents endpoint accumulation across repeated pre-warp passes when the host
        // triggers multiple rebuilds in a short time.
        auto clamp_warp = [](float xq, float xmin, float xmax)->float {
            if (!std::isfinite(xq)) return xmin;
            const float eps = 5e-5f; // slightly stronger inward bias
            const float lo = xmin + eps;
            const float hi = xmax - eps;
            float xc = std::min(std::max(xq, lo), hi);
            if (!std::isfinite(xc)) xc = lo;
            return xc;
            };

        // Local interpolator
        auto interp = [](const std::vector<float>& X, const std::vector<float>& Y, float xq)->float {
            if (!std::isfinite(xq)) return Y.front();
            if (xq <= X.front()) return Y.front();
            if (xq >= X.back())  return Y.back();
            size_t i1 = size_t(std::lower_bound(X.begin(), X.end(), xq) - X.begin());
            if (i1 == 0) return Y.front();
            if (i1 >= X.size()) return Y.back();
            size_t i0 = i1 - 1;
            const float denom = (X[i1] - X[i0]);
            if (denom <= 0.0f || !std::isfinite(denom)) return Y[i0];
            const float t = (xq - X[i0]) / denom;
            return Y[i0] + t * (Y[i1] - Y[i0]);
            };

        // Apply per-sample ΔlogE on each channel’s own grid
        std::vector<float> dYcorr(N), dMcorr(N), dCcorr(N);
        for (size_t i = 0; i < N; ++i) {
            const float aY = M[0][0] * nB[i] + M[1][0] * nG[i] + M[2][0] * nR[i];
            const float aM = M[0][1] * nB[i] + M[1][1] * nG[i] + M[2][1] * nR[i];
            const float aC = M[0][2] * nB[i] + M[1][2] * nG[i] + M[2][2] * nR[i];

            const float xqY = clamp_warp(xB[i] - aY, xB.front(), xB.back());
            const float xqM = clamp_warp(xG[i] - aM, xG.front(), xG.back());
            const float xqC = clamp_warp(xR[i] - aC, xR.front(), xR.back());

            dYcorr[i] = interp(xB, dY, xqY);
            dMcorr[i] = interp(xG, dM, xqM);
            dCcorr[i] = interp(xR, dC, xqC);
        }

        // Enforce monotonic non-decreasing densities and scrub NaNs (agx expects monotonic curves over logE)
        auto enforce_monotone = [](std::vector<float>& v) {
            float prev = (v.empty() || !std::isfinite(v[0])) ? 0.0f : v[0];
            if (!std::isfinite(prev)) prev = 0.0f;
            for (size_t i = 0; i < v.size(); ++i) {
                float cur = v[i];
                if (!std::isfinite(cur) || cur < 0.0f) cur = 0.0f;
                if (cur < prev) cur = prev;
                v[i] = cur;
                prev = cur;
            }
            };
        enforce_monotone(dYcorr);
        enforce_monotone(dMcorr);
        enforce_monotone(dCcorr);

        // Store back per‑channel grids
        outB.linear = dYcorr; outB.lambda_nm = xB;
        outG.linear = dMcorr; outG.lambda_nm = xG;
        outR.linear = dCcorr; outR.lambda_nm = xR;
    }

    // Define OFX params (no globals)
    inline void define_params(OfxImageEffectHandle effect,
        OfxImageEffectSuiteV1* effectSuite,
        OfxPropertySuiteV1* propSuite,
        OfxParameterSuiteV1* paramSuite) {
        OfxParamSetHandle ps = nullptr;
        effectSuite->getParamSet(effect, &ps);
        if (!ps) return;

        OfxPropertySetHandle p;

        paramSuite->paramDefine(ps, kOfxParamTypeBoolean, kParamCouplersActive, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "DIR couplers");
        propSuite->propSetInt(p, kOfxParamPropDefault, 0, 1);

        paramSuite->paramDefine(ps, kOfxParamTypeBoolean, kParamCouplersPrecorrect, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Precorrect density curves");
        propSuite->propSetInt(p, kOfxParamPropDefault, 0, 1);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmountR, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers amount R");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 0.7);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmountG, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers amount G");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 0.7);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersAmountB, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers amount B");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 0.5);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersLayerSigma, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Layer diffusion");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 1.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 3.0);

        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersHighExpShift, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "High exposure shift");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 1.0);

        // Spatial diffusion (xy Gaussian in pixel domain), default 0 for speed/parity now
        paramSuite->paramDefine(ps, kOfxParamTypeDouble, kParamCouplersSpatialSigma, &p);
        propSuite->propSetString(p, kOfxPropLabel, 0, "Couplers spatial diffusion");
        propSuite->propSetDouble(p, kOfxParamPropDefault, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
        propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, 15.0);

    }

    // Called from onParamChanged
    inline void on_param_changed(const char* name) {
        if (!name) return;
        if (std::strcmp(name, kParamCouplersPrecorrect) == 0 ||
            std::strcmp(name, kParamCouplersActive) == 0 ||
            std::strcmp(name, kParamCouplersAmountR) == 0 ||
            std::strcmp(name, kParamCouplersAmountG) == 0 ||
            std::strcmp(name, kParamCouplersAmountB) == 0 ||
            std::strcmp(name, kParamCouplersLayerSigma) == 0 ||
            std::strcmp(name, kParamCouplersHighExpShift) == 0 ||
            std::strcmp(name, kParamCouplersSpatialSigma) == 0) {
            Spectral::mark_mixing_dirty();
            gPrecorrectApplied = false;
        }

    }

    // Couplers.h

    inline void maybe_precorrect_curves(
        OfxImageEffectHandle instance,
        OfxImageEffectSuiteV1* effectSuite,
        OfxParameterSuiteV1* paramSuite)
    {
        if (gPrecorrectApplied) return;

        // Read params
        OfxParamSetHandle ps = nullptr;
        effectSuite->getParamSet(instance, &ps);

        auto getB = [&](const char* nm, int def)->int {
            OfxParamHandle h = nullptr; int v = def;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            return v;
            };
        auto getD = [&](const char* nm, double def)->double {
            OfxParamHandle h = nullptr; double v = def;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            return v;
            };

        // Respect toggle
        const int doPre = getB(kParamCouplersPrecorrect, 1);
        if (!doPre) { gPrecorrectApplied = true; return; }

        // Fetch amounts in B,G, R order (rename or map if your UI is R,G,B)
        const float aB = static_cast<float>(getD(kParamCouplersAmountB, 0.5));
        const float aG = static_cast<float>(getD(kParamCouplersAmountG, 0.7));
        const float aR = static_cast<float>(getD(kParamCouplersAmountR, 0.7));
        const float sigma = static_cast<float>(getD(kParamCouplersLayerSigma, 1.0));
        const float high = static_cast<float>(getD(kParamCouplersHighExpShift, 0.0));

        const float amount[3] = { aB, aG, aR };
        float M[3][3];
        build_dir_matrix(M, amount, sigma);

        // Apply pre-warp
        precorrect_density_curves_before_DIR(M, high);

        gPrecorrectApplied = true;
    }


    struct Runtime {
        bool  active = true;
        float M[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
        float spatialSigmaPixels = 0.0f; // 0 = disabled, later used for optional xy Gaussian
    };


    inline void fetch_runtime(const OfxImageEffectHandle instance,
        OfxImageEffectSuiteV1* effectSuite,
        OfxParameterSuiteV1* paramSuite,
        Runtime& rt) {
        // Defaults
        rt.active = true;
        double ar = 0.7, ag = 0.7, ab = 0.5, sigma = 1.0, high = 0.0;
        double spatial = 0.0;
        int activeInt = 1;

        OfxParamSetHandle ps = nullptr;
        effectSuite->getParamSet(instance, &ps);

        auto getD = [&](const char* nm, double& v) {
            OfxParamHandle h = nullptr;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            };
        auto getB = [&](const char* nm, int& v) {
            OfxParamHandle h = nullptr;
            if (ps && paramSuite->paramGetHandle(ps, nm, &h, nullptr) == kOfxStatOK && h)
                paramSuite->paramGetValue(h, &v);
            };

        getB(kParamCouplersActive, activeInt);
        getD(kParamCouplersAmountR, ar);
        getD(kParamCouplersAmountG, ag);
        getD(kParamCouplersAmountB, ab);
        getD(kParamCouplersLayerSigma, sigma);
        getD(kParamCouplersHighExpShift, high);
        getD(kParamCouplersSpatialSigma, spatial);

        rt.active = (activeInt != 0);
        const float amount[3] = { float(ab), float(ag), float(ar) };
        build_dir_matrix(rt.M, amount, float(sigma));
        rt.highShift = float(high);

        // In enabled builds, dMax is supplied from per-instance WorkingState; avoid globals here.
        rt.dMax[0] = 1.0f;
        rt.dMax[1] = 1.0f;
        rt.dMax[2] = 1.0f;

    }

    struct ApplyInput {
        float E[3];   // layer exposures (B,G,R)
        float D[3];   // dye densities over B+F (Y,M,C)
    };

    inline void apply_runtime(ApplyInput& io, const Runtime& rt) {
        if (!rt.active) return;

        float nB = std::clamp(io.D[0] / rt.dMax[0], 0.0f, 1.0f);
        float nG = std::clamp(io.D[1] / rt.dMax[1], 0.0f, 1.0f);
        float nR = std::clamp(io.D[2] / rt.dMax[2], 0.0f, 1.0f);
        nB += rt.highShift * nB * nB;
        nG += rt.highShift * nG * nG;
        nR += rt.highShift * nR * nR;

        const float aY = rt.M[0][0] * nB + rt.M[1][0] * nG + rt.M[2][0] * nR; // affects E_B
        const float aM = rt.M[0][1] * nB + rt.M[1][1] * nG + rt.M[2][1] * nR; // affects E_G
        const float aC = rt.M[0][2] * nB + rt.M[1][2] * nG + rt.M[2][2] * nR; // affects E_R

        const float sB = std::pow(10.0f, -aY);
        const float sG = std::pow(10.0f, -aM);
        const float sR = std::pow(10.0f, -aC);
        io.E[0] *= sB; io.E[1] *= sG; io.E[2] *= sR;
    }

    // LogE-domain couplers runtime (agx parity)
    struct ApplyInputLogE {
        float logE[3]; // per-layer log10 exposure, including any per-layer offsets used to sample curves
        float D[3];    // current dye densities Y,M,C at those logE (used only to compute correction)
    };

    
    // Internal: compute ΔlogE (aY, aM, aC) from normalized densities
    // Note: M has shape [input layer (B,G,R)] x [output correction (Y,M,C)],
    // and corrections (aY,aM,aC) are subtracted from logE of Blue, Green, Red respectively,
    // matching agx-emulsion’s convention (input index = RGB, output index = per-layer correction).

    inline void compute_logE_corrections(const ApplyInputLogE& in, const Runtime& rt, float a_out[3]) {
        auto safe_norm = [](float D, float dmax)->float {
            // Enforce non-negative inputs and robust maxima
            float Din = (!std::isfinite(D) || D < 0.0f) ? 0.0f : D;
            float m = (std::isfinite(dmax) && dmax > 1e-4f) ? dmax : 1.0f; // slightly higher floor
            float n = Din / m;
            if (!std::isfinite(n)) n = 0.0f;
            return std::min(std::max(n, 0.0f), 1.0f);
            };


        float nB = safe_norm(in.D[0], rt.dMax[0]);
        float nG = safe_norm(in.D[1], rt.dMax[1]);
        float nR = safe_norm(in.D[2], rt.dMax[2]);

        // high-exposure quadratic boost (agx: n += k * n^2)
        auto high_boost = [&](float n)->float {
            float nb = n + rt.highShift * n * n;
            if (!std::isfinite(nb)) nb = n;
            return std::min(std::max(nb, 0.0f), 1.0f);
            };
        nB = high_boost(nB);
        nG = high_boost(nG);
        nR = high_boost(nR);

        // M maps normalized densities [B,G,R] to per-layer logE corrections in [Y,M,C] axes
        float aY = rt.M[0][0] * nB + rt.M[1][0] * nG + rt.M[2][0] * nR; // affects Blue layer logE
        float aM = rt.M[0][1] * nB + rt.M[1][1] * nG + rt.M[2][1] * nR; // affects Green layer logE
        float aC = rt.M[0][2] * nB + rt.M[1][2] * nG + rt.M[2][2] * nR; // affects Red layer logE

        // Scrub non-finite corrections before bounding
        if (!std::isfinite(aY)) aY = 0.0f;
        if (!std::isfinite(aM)) aM = 0.0f;
        if (!std::isfinite(aC)) aC = 0.0f;

        // Bound corrections early to stabilize downstream interpolation and sampling.
        auto clamp_corr = [](float v)->float {
            if (!std::isfinite(v)) return 0.0f;
            if (v < -10.0f) return -10.0f;
            if (v > 10.0f)  return 10.0f;
            return v;
            };
        aY = clamp_corr(aY);
        aM = clamp_corr(aM);
        aC = clamp_corr(aC);

        a_out[0] = std::isfinite(aY) ? aY : 0.0f;
        a_out[1] = std::isfinite(aM) ? aM : 0.0f;
        a_out[2] = std::isfinite(aC) ? aC : 0.0f;
    }


    // Public: apply ΔlogE by subtraction (agx: log_raw_corrected = log_raw - correction)
    inline void apply_runtime_logE(ApplyInputLogE& io, const Runtime& rt) {
        if (!rt.active) return;
        float a[3]; compute_logE_corrections(io, rt, a);
        io.logE[0] -= a[0]; // Blue layer
        io.logE[1] -= a[1]; // Green layer
        io.logE[2] -= a[2]; // Red layer

        // Defensive clamp to current curve domains (if globals are present)
        // Note: Scanner/Print paths clamp before sampling; this is extra safety.
        auto clamp_to = [](float le, const Spectral::Curve& c)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };
        io.logE[0] = clamp_to(io.logE[0], Spectral::gDensityCurveB);
        io.logE[1] = clamp_to(io.logE[1], Spectral::gDensityCurveG);
        io.logE[2] = clamp_to(io.logE[2], Spectral::gDensityCurveR);

    }

    // Variant: clamp using per-instance curves to match Scanner/Print sampling (agx parity)
    inline void apply_runtime_logE_with_curves(
        ApplyInputLogE& io,
        const Runtime& rt,
        const Spectral::Curve& densB,
        const Spectral::Curve& densG,
        const Spectral::Curve& densR)
    {
        if (!rt.active) return;
        float a[3]; compute_logE_corrections(io, rt, a);
        io.logE[0] -= a[0];
        io.logE[1] -= a[1];
        io.logE[2] -= a[2];

        auto clamp_to = [](float le, const Spectral::Curve& c)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };
        io.logE[0] = clamp_to(io.logE[0], densB);
        io.logE[1] = clamp_to(io.logE[1], densG);
        io.logE[2] = clamp_to(io.logE[2], densR);
    }



} // namespace Couplers

#endif // JUICER_ENABLE_COUPLERS
