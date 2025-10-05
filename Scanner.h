// Scanner.h
#pragma once

#include "SpectralTables.h"
#include "SpectralMath.h"
#include "Couplers.h"
#include "WorkingState.h"

// Forward-declare WorkingState so we don't create header cycles
struct WorkingState;

namespace Scanner {

    struct Params {
        bool enabled = false;
        bool autoExposure = true;
        float targetY = 0.18f;
    };

    inline void fetch_params(OfxParameterSuiteV1* paramSuite, OfxParamSetHandle paramSet, Params& out) {
        if (!paramSuite || !paramSet) { out = {}; out.autoExposure = true; out.targetY = 0.18f; return; }
        OfxParamHandle hEnabled = nullptr, hAutoExp = nullptr, hTargetY = nullptr;
        paramSuite->paramGetHandle(paramSet, "ScannerEnabled", &hEnabled, nullptr);
        paramSuite->paramGetHandle(paramSet, "ScannerAutoExposure", &hAutoExp, nullptr);
        paramSuite->paramGetHandle(paramSet, "ScannerTargetY", &hTargetY, nullptr);
        int enabled = 0, autoExp = 1;
        double targetY = 0.18;
        if (hEnabled) paramSuite->paramGetValue(hEnabled, &enabled);
        if (hAutoExp) paramSuite->paramGetValue(hAutoExp, &autoExp);
        if (hTargetY) paramSuite->paramGetValue(hTargetY, &targetY);
        out.enabled = (enabled != 0);
        out.autoExposure = (autoExp != 0);
        out.targetY = static_cast<float>(targetY);
    }

    // Helper: sample densities from per-instance curves at given per-layer logE
    inline void sample_densities_from_ws(const WorkingState& ws, const float logE[3], float D_cmy[3]) {
        D_cmy[0] = Spectral::sample_density_at_logE(ws.densB, logE[0]); // Y from B layer curve
        D_cmy[1] = Spectral::sample_density_at_logE(ws.densG, logE[1]); // M from G layer curve
        D_cmy[2] = Spectral::sample_density_at_logE(ws.densR, logE[2]); // C from R layer curve
    }

    inline void simulate_scanner(
        const float rgbIn[3],
        float rgbOut[3],
        const Scanner::Params& scannerParams,
        const Couplers::Runtime& dirRT,
        const WorkingState& ws,
        float exposureScale)
    {
        if (!scannerParams.enabled) {
            rgbOut[0] = rgbIn[0];
            rgbOut[1] = rgbIn[1];
            rgbOut[2] = rgbIn[2];
            return;
        }

        // DWG → layer exposures (per-instance SPD vs Matrix, no global toggle)
        float E[3];
        const SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
        Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbIn, E, exposureScale,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        // Per-instance logE offsets (no globals)
        float logE[3] = {
            std::log10(std::max(E[0], 1e-6f)) + ws.logEOffB,
            std::log10(std::max(E[1], 1e-6f)) + ws.logEOffG,
            std::log10(std::max(E[2], 1e-6f)) + ws.logEOffR
        };

        // Domain clamp helper before sampling
        auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };

        // Densities from per-instance curves
        float D_cmy[3];
        float logE_clamped[3] = {
            clamp_logE_to_curve(ws.densB, logE[0]),
            clamp_logE_to_curve(ws.densG, logE[1]),
            clamp_logE_to_curve(ws.densR, logE[2]),
        };
        sample_densities_from_ws(ws, logE_clamped, D_cmy);

#ifdef JUICER_ENABLE_COUPLERS
        // Always apply local DIR corrections when enabled (agx parity), even if the build precorrected
        // density curves and spatial diffusion is disabled.
        if (dirRT.active) {
            Couplers::ApplyInputLogE io{ {logE[0], logE[1], logE[2]}, {D_cmy[0], D_cmy[1], D_cmy[2]} };
            // Per-instance clamp variant: aligns clamp domain to the same curves we sample
            Couplers::apply_runtime_logE_with_curves(io, dirRT, ws.densB, ws.densG, ws.densR);
            float logE2_clamped[3] = { io.logE[0], io.logE[1], io.logE[2] };
            sample_densities_from_ws(ws, logE2_clamped, D_cmy);
        }
#endif

        // Spectral integration using scanner viewing tables (fallback to view tables if missing)
        const SpectralTables* tables = nullptr;
        if (ws.tablesScan.K > 0) {
            tables = &ws.tablesScan;
        }
        else if (ws.tablesView.K > 0) {
            tables = &ws.tablesView;
        }

        if (!tables) {
            rgbOut[0] = rgbIn[0];
            rgbOut[1] = rgbIn[1];
            rgbOut[2] = rgbIn[2];
            return;
        }

        float XYZ[3] = { 0.0f, 0.0f, 0.0f };
        if (ws.hasBaseline && tables->hasBaseline) {
            Spectral::dyes_to_XYZ_with_baseline_given_tables(*tables, D_cmy, XYZ);
        }
        else {
            Spectral::dyes_to_XYZ_given_tables(*tables, D_cmy, XYZ);
        }

        float autoGain = 1.0f;
        if (scannerParams.autoExposure) {
            const float targetY = (std::isfinite(scannerParams.targetY) && scannerParams.targetY > 0.0f)
                ? scannerParams.targetY
                : 0.184f;

            const float exposureScaleSafe = (std::isfinite(exposureScale) && exposureScale > 0.0f)
                ? exposureScale
                : 1.0f;

            struct AutoExposureCache {
                const WorkingState* ws = nullptr;
                uint64_t buildCounter = 0;
                const SpectralTables* tables = nullptr;
                const SpectralTables* tablesSPD = nullptr;
                bool spdReady = false;
                float exposureScale = 1.0f;
                float targetY = 0.184f;
                float gain = 1.0f;
                bool valid = false;
            };

            thread_local AutoExposureCache cache;

            auto nearly_equal = [](float a, float b) {
                const float diff = std::fabs(a - b);
                const float absA = std::fabs(a);
                const float absB = std::fabs(b);
                const float scale = std::max(1.0f, std::max(absA, absB));
                return diff <= scale * 1e-4f;
                };

            const bool cacheHit = cache.valid &&
                cache.ws == &ws &&
                cache.buildCounter == ws.buildCounter &&
                cache.tables == tables &&
                cache.tablesSPD == tablesSPD &&
                cache.spdReady == ws.spdReady &&
                nearly_equal(cache.exposureScale, exposureScaleSafe) &&
                nearly_equal(cache.targetY, targetY);

            if (cacheHit) {
                autoGain = cache.gain;
            }
            else {
                float rgbRef[3] = { 0.184f, 0.184f, 0.184f };
                float Eref[3];
                Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
                    rgbRef, Eref, exposureScaleSafe,
                    tablesSPD,
                    (ws.spdReady ? ws.spdSInv : nullptr),
                    ws.spdReady,
                    ws.sensB, ws.sensG, ws.sensR);

                float logERef[3] = {
                    std::log10(std::max(Eref[0], 1e-6f)) + ws.logEOffB,
                    std::log10(std::max(Eref[1], 1e-6f)) + ws.logEOffG,
                    std::log10(std::max(Eref[2], 1e-6f)) + ws.logEOffR
                };

                float logERefClamped[3] = {
                    clamp_logE_to_curve(ws.densB, logERef[0]),
                    clamp_logE_to_curve(ws.densG, logERef[1]),
                    clamp_logE_to_curve(ws.densR, logERef[2])
                };

                float Dref[3];
                sample_densities_from_ws(ws, logERefClamped, Dref);

                float XYZref[3] = { 0.0f, 0.0f, 0.0f };
                if (ws.hasBaseline && tables->hasBaseline) {
                    Spectral::dyes_to_XYZ_with_baseline_given_tables(*tables, Dref, XYZref);
                }
                else {
                    Spectral::dyes_to_XYZ_given_tables(*tables, Dref, XYZref);
                }

                const float Yref = XYZref[1];
                if (std::isfinite(Yref) && Yref > 0.0f) {
                    autoGain = targetY / Yref;
                }

                if (!std::isfinite(autoGain) || autoGain <= 0.0f) {
                    autoGain = 1.0f;
                }
                else {
                    const float minGain = 1.0f / 32.0f;
                    const float maxGain = 32.0f;
                    if (autoGain < minGain) autoGain = minGain;
                    if (autoGain > maxGain) autoGain = maxGain;
                }

                cache.ws = &ws;
                cache.buildCounter = ws.buildCounter;
                cache.tables = tables;
                cache.tablesSPD = tablesSPD;
                cache.spdReady = ws.spdReady;
                cache.exposureScale = exposureScaleSafe;
                cache.targetY = targetY;
                cache.gain = autoGain;
                cache.valid = true;
            }
        }

        XYZ[0] *= autoGain;
        XYZ[1] *= autoGain;
        XYZ[2] *= autoGain;

        // XYZ → DWG with chromatic adaptation based on the scan illuminant
        Spectral::XYZ_to_DWG_linear_adapted(*tables, XYZ, rgbOut);
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);
    }

} // namespace Scanner
