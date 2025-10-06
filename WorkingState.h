// WorkingState.h
#pragma once

#include <cstdint>
#include "SpectralTables.h"
#include "SpectralMath.h"   // for Spectral::Curve
#include "Couplers.h"

// Forward declaration
namespace Print {
    struct Runtime;
}

// Per‑instance, derived state used for rendering.
// Built from BaseState in rebuild_working_state() and never mutated in render().
struct WorkingState {
    // Density curves (after rebuild transforms)
    Spectral::Curve densB;
    Spectral::Curve densG;
    Spectral::Curve densR;

    // Sensitivity curves (post-balance scaling)
    Spectral::Curve sensB;
    Spectral::Curve sensG;
    Spectral::Curve sensR;

    // Negative sensitivity curves before neutral balancing (AgX parity use)
    Spectral::Curve negSensB;
    Spectral::Curve negSensG;
    Spectral::Curve negSensR;

    // Baseline curves and flag
    Spectral::Curve baseMin;
    Spectral::Curve baseMid;
    bool hasBaseline = false;

    // Per-channel logE offsets (mid-gray alignment)
    float logEOffB = 0.0f;
    float logEOffG = 0.0f;
    float logEOffR = 0.0f;

    // DIR runtime (per-pixel)
    Couplers::Runtime dirRT;

    // Whether DIR pre-correction was applied to density curves in this build
    bool dirPrecorrected = false;

    // Normalization constants (frozen from pre-DIR balanced curves)
    float dMax[3] = { 1.0f, 1.0f, 1.0f };

    // Per-instance precomputed spectral tables for the viewing illuminant
    SpectralTables tablesView;

    // Per-instance spectral tables for the print paper viewing path
    SpectralTables tablesPrint;

    // Per-instance spectral tables for the film reference illuminant (SPD reconstruction)
    SpectralTables tablesRef;

    // Per-instance spectral tables for scanner/negative viewing (D50 parity)
    SpectralTables tablesScan;

    // SPD reconstruction per-instance (non-global)
    float spdSInv[9] = { 1,0,0, 0,1,0, 0,0,1 };
    bool  spdReady = false;

    // Snapshot of print runtime used for this WorkingState build (immutable during render)
    std::unique_ptr<Print::Runtime> printRT;

    Spectral::NegativeCouplerParams negParams{};

    // Versioning for atomic swap / debugging
    std::uint64_t buildCounter = 0;
};
