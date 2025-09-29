#pragma once
#include <vector>

// Standalone definition of the per-instance spectral tables.
// No dependency on WorkingState.h or SpectralMath.h.
struct SpectralTables {
    // Wavelength axis
    std::vector<float> lambda;
    int   K = 0;
    float deltaLambda = 5.0f;
    float invYn = 1.0f;

    // Illuminant-weighted CMFs (Ax, Ay, Az) and raw CMFs
    std::vector<float> Ax, Ay, Az;
    std::vector<float> Xbar, Ybar, Zbar;

    // Dye extinction tables
    std::vector<float> epsY, epsM, epsC;

    // Baseline (optional) and flag
    std::vector<float> baseMin, baseMid;
    bool hasBaseline = false;
};
