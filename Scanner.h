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
        (void)scannerParams; // MVP: not used yet

        // DWG → layer exposures (per-instance SPD vs Matrix, no global toggle)
        float E[3];
        Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbIn, E, exposureScale,
            (ws.tablesView.K > 0 ? &ws.tablesView : nullptr),
            (ws.spdReady ? ws.spdSInv : nullptr),
            (int)std::clamp(ws.spectralMode, 0, 1),
            (ws.exposureModel == 1) && ws.spdReady,
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
        // Skip local (non-spatial) DIR when curves were precorrected and spatial diffusion is disabled
        const bool allowLocalDIR = (!ws.dirPrecorrected) || (dirRT.spatialSigmaPixels > 0.5f);
        if (dirRT.active && allowLocalDIR) {
            Couplers::ApplyInputLogE io{ {logE[0], logE[1], logE[2]}, {D_cmy[0], D_cmy[1], D_cmy[2]} };
            // Per-instance clamp variant: aligns clamp domain to the same curves we sample
            Couplers::apply_runtime_logE_with_curves(io, dirRT, ws.densB, ws.densG, ws.densR);
            float logE2_clamped[3] = { io.logE[0], io.logE[1], io.logE[2] };
            sample_densities_from_ws(ws, logE2_clamped, D_cmy);
        }
#endif

        // Spectral integration (still uses global tables for eps/CMFs/illuminant)
        float XYZ[3];
        if (ws.hasBaseline && ws.tablesView.hasBaseline) {
            const float w = Spectral::neutral_blend_weight_from_DWG_rgb(rgbIn);
            Spectral::dyes_to_XYZ_with_baseline_given_tables(ws.tablesView, D_cmy, w, XYZ);
        }
        else {
            Spectral::dyes_to_XYZ_given_tables(ws.tablesView, D_cmy, XYZ);
        }

        // XYZ → DWG + contrast
        Spectral::XYZ_to_DWG_linear(XYZ, rgbOut);        
    }

} // namespace Scanner
