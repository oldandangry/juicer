// SpectralMath.h
#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include "NpyLoader.h"
#include <cstdint>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "SpectralTables.h"

// Forward declaration to avoid circular dependency with WorkingState.h
struct SpectralTables;

namespace Spectral {
    // --- Precompute change detector state ---
    inline uint64_t gIllumVersion = 0;
    inline uint64_t gLastPrecomputeIllumVersion = ~uint64_t{ 0 };
    inline bool gPrecomputeDirty = true;
    // Add shape versioning to trigger precompute when gShape wavelengths change
    inline uint64_t gShapeVersion = 0;
    inline uint64_t gLastPrecomputeShapeVersion = ~uint64_t{ 0 };
    // Version for curve mixing/unmixing changes
    inline uint64_t gMixVersion = 0;
    inline void mark_mixing_dirty() {
        ++gMixVersion;
    }
    inline int gLastIllumChoice = -1; // -1 = unknown, avoids recompute on first frame
   
    // SpectralMath.h (near the top, inside namespace Spectral, before any use)
    struct BaselineCtx {
        bool hasBaseline;
        float w;
        const float* baseMin;
        const float* baseMid;
    };

#if defined(__AVX2__)
    void integrate_dyes_to_XYZ_avx2(
        float dY, float dM, float dC,
        const float* epsY, const float* epsM, const float* epsC,
        const float* Ax, const float* Ay, const float* Az,
        int K,
        const BaselineCtx& base,
        float XYZ_out[3]);
#endif

    // Forward declare

    struct Curve;

    inline void layerExposures_from_sceneSPD_with_curves(
        const std::vector<float>& Ee,
        const Curve& sB, const Curve& sG, const Curve& sR,
        float E[3], float exposureScale);


    // -----------------------------------------------------------------------------
    // DIR coupler hook — default no‑op
    // In agx‑emulsion, this adjusts per‑dye densities based on interlayer coupler effects.
    // Here we just leave densities unchanged.
    // D[0] = Yellow dye density (from blue layer)
    // D[1] = Magenta dye density (from green layer)
    // D[2] = Cyan dye density (from red layer)
    // E[0..2] = layer exposures (logE or linear, depending on your pipeline)
    // -----------------------------------------------------------------------------
    inline void apply_dir_couplers(float D[3], const float E_in[3]) {
#ifdef JUICER_ENABLE_COUPLERS
        // We only define the call site here; the actual param fetch is done at the effect layer.
        // So keep this as a no-op and use the runtime hook inside density_curve_rgb_dwg.
        (void)D; (void)E_in;
#else
        (void)D; (void)E_in;
#endif
    }



    // Mark tables dirty (call when any spectral input that affects tables changes)
    inline void mark_spectral_tables_dirty() {
        gPrecomputeDirty = true;
    }

    void precompute_spectral_tables();
    void disable_hanatos_if_shape_mismatch();

    // Dynamic spectral shape descriptor (prep for arbitrary K)
    struct SpectralShape {
        std::vector<float> wavelengths; // monotonically increasing, in nm
        float lambdaMin = 0.0f;
        float lambdaMax = 0.0f;
        float delta = 0.0f;             // assumed uniform for now
        int   K = 0;

        inline void clear() {
            wavelengths.clear();
            lambdaMin = lambdaMax = delta = 0.0f;
            K = 0;
        }
    };

    inline SpectralShape gShape;

    // Wavelength domain (nm)
    static constexpr float kLambdaMin = 400.0f;
    static constexpr float kLambdaMax = 800.0f;
    static constexpr float kDelta = 5.0f;

    // Fixed wavelength grid precompute (400..800 nm step 5) => 81 bands
    static constexpr int kNumSamples = static_cast<int>((kLambdaMax - kLambdaMin) / kDelta) + 1;
    static_assert(kNumSamples == 81, "Spectral grid must be 400..800 nm at 5 nm.");

    // Dynamic wavelength grid — sized at runtime from gShape.K
    inline std::vector<float> gEpsYTable;
    inline std::vector<float> gEpsMTable;
    inline std::vector<float> gEpsCTable;
    inline std::vector<float> gXbarTable;
    inline std::vector<float> gYbarTable;
    inline std::vector<float> gZbarTable;
    inline std::vector<float> gBaselineMinTable;
    inline std::vector<float> gBaselineMidTable;
    inline std::vector<float> gIllumTable;

    inline std::vector<float> gAx; // Ee * xbar
    inline std::vector<float> gAy; // Ee * ybar
    inline std::vector<float> gAz; // Ee * zbar
    inline float gInvYn = 1.0f;    // 1 / gYnNorm

    inline constexpr float kLn10 = 2.302585092994046f;

    // Wavelength axis for sampling
    inline std::vector<float> gLambda;

    // Normalization for Y (illuminant-weighted)
    inline float gYnNorm = 1.0f;

    // Global deltaLambda used for integrals (derived from gShape)
    inline float gDeltaLambda = kDelta;

    inline float compute_delta_from_shape(const SpectralShape& s) {
        if (s.K <= 1 || s.wavelengths.size() < 2) return kDelta;
        // Estimate mean Δλ to be robust to tiny non-uniformities
        double sum = 0.0;
        for (int i = 1; i < s.K; ++i) sum += static_cast<double>(s.wavelengths[i] - s.wavelengths[i - 1]);
        const double mean = sum / static_cast<double>(s.K - 1);
        return (mean > 0.0) ? static_cast<float>(mean) : kDelta;
    }


    // Effective sensitivity gains (computed in precompute)
    // Store per-layer correction so neutral exposures match green under the viewing illuminant.
    inline float gSensCorrB = 1.0f;
    inline float gSensCorrG = 1.0f;
    inline float gSensCorrR = 1.0f;

    // Forward declaration because precompute_spectral_tables() calls it
    inline float illuminant_E(float lambda);

    inline float gaussian(float x, float mu, float sigma) {
        const float t = (x - mu) / sigma;
        return std::exp(-0.5f * t * t);
    }

    // -------------------------------------------------------------------------
    // Simple curve container for measured sensitivities
    // -------------------------------------------------------------------------
    struct Curve {
        std::vector<float> lambda_nm;
        std::vector<float> linear; // linearised and peak-normalised

        inline void build_from_log10_pairs(const std::vector<std::pair<float, float>>& samples) {
            lambda_nm.clear();
            linear.clear();
            lambda_nm.reserve(samples.size());
            linear.reserve(samples.size());
            float peak = 0.0f;
            for (auto& p : samples) {
                lambda_nm.push_back(p.first);
                float lin = std::pow(10.0f, p.second);
                linear.push_back(lin);
                if (lin > peak) peak = lin;
            }
            if (peak > 0.0f) {
                for (auto& v : linear) v /= peak;
            }
        }

        inline void build_from_linear_pairs(const std::vector<std::pair<float, float>>& samples) {
            lambda_nm.clear();
            linear.clear();
            if (samples.empty()) return;

            // Make a sorted copy
            std::vector<std::pair<float, float>> s = samples;
            std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first < b.first; });

            lambda_nm.reserve(s.size());
            linear.reserve(s.size());
            for (auto& p : s) {
                lambda_nm.push_back(p.first);
                linear.push_back(std::max(0.0f, p.second));
            }
        }


        inline float sample(float lambda) const {
            const size_t n = lambda_nm.size();
            if (n == 0) return 0.0f;
            if (lambda <= lambda_nm.front()) return linear.front();
            if (lambda >= lambda_nm.back()) return linear.back();
            size_t i1 = 1;
            while (i1 < n && lambda_nm[i1] < lambda) ++i1;
            size_t i0 = i1 - 1;
            float x0 = lambda_nm[i0], x1 = lambda_nm[i1];
            float y0 = linear[i0], y1 = linear[i1];
            float t = (lambda - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    };

    // ---- Centralized resampling to SpectralShape ----

// Linear sample arbitrary (lambda, value) pairs at 'lambda' with endpoint clamp.
    inline float sample_linear_pairs(const std::vector<std::pair<float, float>>& pairs, float lambda) {
        const size_t n = pairs.size();
        if (n == 0) return 0.0f;
        if (lambda <= pairs.front().first) return pairs.front().second;
        if (lambda >= pairs.back().first)  return pairs.back().second;
        size_t i1 = 1;
        while (i1 < n && pairs[i1].first < lambda) ++i1;
        const size_t i0 = i1 - 1;
        const float x0 = pairs[i0].first, x1 = pairs[i1].first;
        const float y0 = pairs[i0].second, y1 = pairs[i1].second;
        const float t = (lambda - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    // Resample x->y pairs to the current SpectralShape axis.
    inline std::vector<std::pair<float, float>>
        resample_pairs_to_shape(const std::vector<std::pair<float, float>>& in, const SpectralShape& s = gShape) {
        std::vector<std::pair<float, float>> out;
        if (in.empty() || s.K <= 0 || s.wavelengths.empty()) return out;
        out.reserve(static_cast<size_t>(s.K));
        // Ensure input is sorted by wavelength
        std::vector<std::pair<float, float>> sorted = in;
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (int i = 0; i < s.K; ++i) {
            const float l = s.wavelengths[i];
            const float v = std::max(0.0f, sample_linear_pairs(sorted, l));
            out.emplace_back(l, v);
        }
        return out;
    }

    // Build a curve pinned to SpectralShape from linear pairs
    inline void build_curve_on_shape_from_linear_pairs(Curve& curve,
        const std::vector<std::pair<float, float>>& pairs,
        const SpectralShape& s = gShape) {
        auto res = resample_pairs_to_shape(pairs, s);
        curve.lambda_nm.clear();
        curve.linear.clear();
        curve.lambda_nm.reserve(res.size());
        curve.linear.reserve(res.size());
        for (auto& p : res) {
            curve.lambda_nm.push_back(p.first);
            curve.linear.push_back(std::max(0.0f, p.second));
        }
    }

    // Build a curve pinned to SpectralShape from log10 pairs (convert to linear, peak-normalize, then resample)
    inline void build_curve_on_shape_from_log10_pairs(Curve& curve,
        const std::vector<std::pair<float, float>>& log10pairs,
        const SpectralShape& s = gShape) {
        if (log10pairs.empty() || s.K <= 0 || s.wavelengths.empty()) {
            curve.lambda_nm.clear();
            curve.linear.clear();
            return;
        }
        // Convert to linear first
        std::vector<std::pair<float, float>> lin;
        lin.reserve(log10pairs.size());
        float peak = 0.0f;
        for (auto& p : log10pairs) {
            const float val = std::pow(10.0f, p.second);
            lin.emplace_back(p.first, val);
            if (val > peak) peak = val;
        }
        if (peak > 0.0f) {
            for (auto& p : lin) p.second /= peak;
        }
        build_curve_on_shape_from_linear_pairs(curve, lin, s);
    }

    // Holds a custom illuminant spectrum (if empty, precompute_spectral_tables uses equal-energy)
    inline Curve gIlluminantCurve;

    // Install a custom illuminant from wavelength/value pairs (linear power)
    inline void set_illuminant_from_pairs(const std::vector<std::pair<float, float>>& pairs) {
        // Build pinned curve
        Curve newCurve;
        build_curve_on_shape_from_linear_pairs(newCurve, pairs);

        // If identical to current gIlluminantCurve, skip dirtying
        const bool sameSize =
            newCurve.linear.size() == gIlluminantCurve.linear.size() &&
            newCurve.lambda_nm.size() == gIlluminantCurve.lambda_nm.size();

        bool identical = sameSize;
        if (identical) {
            // Compare lambda axis first (shapes should match)
            for (size_t i = 0; i < newCurve.lambda_nm.size(); ++i) {
                if (newCurve.lambda_nm[i] != gIlluminantCurve.lambda_nm[i]) {
                    identical = false; break;
                }
            }
            // Compare spectrum with a tight epsilon (mean‑power normalization should make this exact or very close)
            if (identical) {
                constexpr float eps = 1e-6f;
                for (size_t i = 0; i < newCurve.linear.size(); ++i) {
                    if (std::fabs(newCurve.linear[i] - gIlluminantCurve.linear[i]) > eps) {
                        identical = false; break;
                    }
                }
            }
        }

        if (identical) {
            return; // no change
        }

        // Install and dirty
        gIlluminantCurve = std::move(newCurve);
        ++gIllumVersion;
        mark_spectral_tables_dirty();
    }



    inline Curve gSensBlue, gSensGreen, gSensRed;

    inline void set_layer_sensitivities(
        const std::vector<std::pair<float, float>>& blue_log10,
        const std::vector<std::pair<float, float>>& green_log10,
        const std::vector<std::pair<float, float>>& red_log10)
    {
        build_curve_on_shape_from_log10_pairs(gSensBlue, blue_log10);
        build_curve_on_shape_from_log10_pairs(gSensGreen, green_log10);
        build_curve_on_shape_from_log10_pairs(gSensRed, red_log10);
    }


    // --- Per-layer density curves (log10 relative exposure → density) ---
    inline Curve gDensityCurveB, gDensityCurveG, gDensityCurveR;

    // Global offset to shift logE domain when sampling density curves
    // (~0.3 ≈ 1 stop shift to the right)
    inline float gDensityCurveLogEOffset = 0.0f;
    inline float gDensityCurveLogEOffsetB = 0.0f;
    inline float gDensityCurveLogEOffsetG = 0.0f;
    inline float gDensityCurveLogEOffsetR = 0.0f;

    inline void set_density_curve_logE_offset(float offset) {
        gDensityCurveLogEOffset = offset;
    }

    // Helper: sort pairs by x, build curve
    inline void sort_and_build(Curve& curve, const std::vector<std::pair<float, float>>& pairs) {
        if (pairs.empty()) {
            curve.lambda_nm.clear();
            curve.linear.clear();
            return;
        }
        std::vector<std::pair<float, float>> sorted = pairs;
        std::sort(sorted.begin(), sorted.end(),
            [](auto& a, auto& b) { return a.first < b.first; });

        curve.lambda_nm.clear();
        curve.linear.clear();
        curve.lambda_nm.reserve(sorted.size());
        curve.linear.reserve(sorted.size());
        for (auto& p : sorted) {
            curve.lambda_nm.push_back(p.first);
            curve.linear.push_back(std::max(0.0f, p.second));
        }
    }

    // Helper: find logE at mid-density
    inline float find_mid_gray_logE(const Curve& c) {
        if (c.lambda_nm.empty()) return 0.0f;
        float dmin = c.linear.front();
        float dmax = c.linear.back();
        float target = dmin + 0.5f * (dmax - dmin);
        // Linear search for segment containing target
        for (size_t i = 1; i < c.lambda_nm.size(); ++i) {
            float y0 = c.linear[i - 1], y1 = c.linear[i];
            if ((y0 <= target && target <= y1) || (y1 <= target && target <= y0)) {
                float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                float t = (target - y0) / (y1 - y0);
                return x0 + t * (x1 - x0);
            }
        }
        return c.lambda_nm.front();
    }

    inline float gMidB = 0.0f;
    inline float gMidG = 0.0f;
    inline float gMidR = 0.0f;

    inline void set_density_curves(
        const std::vector<std::pair<float, float>>& b_pairs,
        const std::vector<std::pair<float, float>>& g_pairs,
        const std::vector<std::pair<float, float>>& r_pairs)
    {
        sort_and_build(gDensityCurveB, b_pairs);
        sort_and_build(gDensityCurveG, g_pairs);
        sort_and_build(gDensityCurveR, r_pairs);

        // Store mid-gray logE for each layer
        gMidB = find_mid_gray_logE(gDensityCurveB);
        gMidG = find_mid_gray_logE(gDensityCurveG);
        gMidR = find_mid_gray_logE(gDensityCurveR);
    }

    // Align red/blue density curves to green at logE = 0 (agx balance_density-like)
    // Shifts R and B curves along logE so that their densities at logE=0 match green's density at logE=0.
    // Assumes curve.lambda_nm holds logE domain and curve.linear holds densities.
    inline void balance_density_curves_to_green_zero() {
        // Require curves to exist
        if (gDensityCurveG.lambda_nm.empty() ||
            gDensityCurveR.lambda_nm.empty() ||
            gDensityCurveB.lambda_nm.empty())
            return;

        auto interp_density = [](const Curve& c, float x) -> float {
            if (c.lambda_nm.empty()) return 0.0f;
            // Sample using the same linear interpolation as Curve::sample,
            // but the x-domain is logE (stored in lambda_nm)
            const size_t n = c.lambda_nm.size();
            if (x <= c.lambda_nm.front()) return c.linear.front();
            if (x >= c.lambda_nm.back())  return c.linear.back();
            size_t i1 = 1;
            while (i1 < n && c.lambda_nm[i1] < x) ++i1;
            const size_t i0 = i1 - 1;
            const float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
            const float y0 = c.linear[i0], y1 = c.linear[i1];
            const float t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
            };

        auto find_logE_for_density = [](const Curve& c, float targetY) -> float {
            if (c.lambda_nm.empty()) return 0.0f;
            const size_t n = c.lambda_nm.size();
            for (size_t i = 1; i < n; ++i) {
                const float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                const float y0 = c.linear[i - 1], y1 = c.linear[i];
                // check segment contains target
                if ((y0 <= targetY && targetY <= y1) || (y1 <= targetY && targetY <= y0)) {
                    const float t = (y1 != y0) ? (targetY - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            // Fallback to closest end
            return c.lambda_nm.front();
            };

        // Target is green density at logE = 0
        const float targetG = interp_density(gDensityCurveG, 0.0f);

        // Compute shifts for R and B so that density(logE + shift) == targetG at logE == 0 => find x such that c(x) = targetG
        const float shiftR = -find_logE_for_density(gDensityCurveR, targetG);
        const float shiftB = -find_logE_for_density(gDensityCurveB, targetG);

        auto shift_curve_in_x = [](Curve& c, float dx) {
            if (c.lambda_nm.empty() || dx == 0.0f) return;
            for (auto& x : c.lambda_nm) x += dx;
            };

        shift_curve_in_x(gDensityCurveR, shiftR);
        shift_curve_in_x(gDensityCurveB, shiftB);

        // Recompute mids for diagnostics if needed
        gMidR = find_mid_gray_logE(gDensityCurveR);
        gMidB = find_mid_gray_logE(gDensityCurveB);

        mark_mixing_dirty();
    }


    inline void mean_power_normalize(std::vector<float>& spd) {
        if (spd.empty()) return;
        double sum = 0.0;
        for (float v : spd) sum += static_cast<double>(v);
        double mean = sum / static_cast<double>(spd.size());
        if (mean > 0.0) {
            for (float& v : spd) v = static_cast<float>(v / mean);
        }
    }


    // Utility: Load wavelength/value pairs from a CSV file
    inline std::vector<std::pair<float, float>> load_csv_pairs(const std::string& path) {
        std::vector<std::pair<float, float>> data;
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV: " + path);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            // Strip comments starting at # or ;
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos) line.erase(p);
                };
            strip_comment('#');
            strip_comment(';');
            std::istringstream ss(line);
            float x = 0.0f, y = 0.0f;
            if (!(ss >> x)) continue;
            // Skip optional comma/semicolon
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();
            if (!(ss >> y)) continue;
            data.emplace_back(x, y);
        }
        return data;
    }

    struct CMFTriplets {
        std::vector<std::pair<float, float>> xbar;
        std::vector<std::pair<float, float>> ybar;
        std::vector<std::pair<float, float>> zbar;
    };

    inline CMFTriplets load_csv_triplets(const std::string& path) {
        CMFTriplets out;
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV: " + path);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos) line.erase(p);
                };
            strip_comment('#');
            strip_comment(';');

            std::istringstream ss(line);
            float l = 0.0f, xv = 0.0f, yv = 0.0f, zv = 0.0f;

            if (!(ss >> l)) continue;
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();

            if (!(ss >> xv)) continue;
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();

            if (!(ss >> yv)) continue;
            while (ss.peek() == ',' || ss.peek() == ';') ss.get();

            if (!(ss >> zv)) continue;

            out.xbar.emplace_back(l, xv);
            out.ybar.emplace_back(l, yv);
            out.zbar.emplace_back(l, zv);
        }
        return out;
    }

    // Load a single-column CSV of wavelengths (nm). Ignores comments (# or ;) and empty lines.
    inline std::vector<float> load_csv_single(const std::string& path) {
        std::vector<float> data;
        std::ifstream file(path);
        if (!file.is_open()) {
            // Return empty to allow fallback without throwing
            return data;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            // Strip comments
            auto strip_comment = [&](char c) {
                size_t p = line.find(c);
                if (p != std::string::npos) line.erase(p);
                };
            strip_comment('#');
            strip_comment(';');
            std::istringstream ss(line);
            float v = 0.0f;
            if (!(ss >> v)) continue;
            data.push_back(v);
        }
        return data;
    }

    inline SpectralShape make_fixed_shape_400_800_5nm() {
        SpectralShape s;
        s.wavelengths.reserve(kNumSamples);
        for (int i = 0; i < kNumSamples; ++i) {
            s.wavelengths.push_back(kLambdaMin + i * kDelta);
        }
        s.K = kNumSamples;
        s.lambdaMin = kLambdaMin;
        s.lambdaMax = kLambdaMax;
        s.delta = kDelta;
        return s;
    }

    inline bool shape_matches_fixed_grid(const SpectralShape& s) {
        if (s.K != kNumSamples) return false;
        if (s.wavelengths.size() != static_cast<size_t>(kNumSamples)) return false;
        for (int i = 0; i < kNumSamples; ++i) {
            const float l = kLambdaMin + i * kDelta;
            if (std::abs(s.wavelengths[i] - l) > 1e-3f) return false;
        }
        return true;
    }

    // === [NEW] Densitometer Crosstalk Utilities ===
// Computes the 3x3 densitometer crosstalk matrix for CMY dyes given Status A responsivities
    inline void compute_densitometer_crosstalk_matrix(
        const std::vector<std::vector<float>>& densitometerResp, // [3][nWavelengths]
        const std::vector<std::vector<float>>& dyeDensity,       // [nWavelengths][3] CMY
        float M[3][3])
    {
        const size_t nW = dyeDensity.size();
        for (int i = 0; i < 3; ++i) { // densitometer RGB
            for (int j = 0; j < 3; ++j) { // CMY dyes
                double num = 0.0;
                double den = 0.0;
                for (size_t w = 0; w < nW; ++w) {
                    const double T = std::pow(10.0, -dyeDensity[w][j]);
                    num += densitometerResp[i][w] * T;
                    den += densitometerResp[i][w];
                }
                M[i][j] = static_cast<float>(-std::log10(num / den));
            }
        }
    }

    // Applies inverse crosstalk matrix to per‑logE density curves
    inline void unmix_density_curves(
        std::vector<std::array<float, 3>>& curves, // [nLogE][3] CMY densities
        const float M[3][3])
    {
        // Invert M
        Eigen::Matrix3f mat{}; // value-init to zero internal POD storage
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat(i, j) = M[i][j];

        Eigen::Matrix3f inv = mat.inverse();

        for (auto& sample : curves) {
            Eigen::Vector3f v(sample[0], sample[1], sample[2]); // ctor fully initializes
            Eigen::Vector3f unmixed = inv * v;
            for (int k = 0; k < 3; ++k)
                sample[k] = std::max(0.0f, unmixed[k]);
        }

    }


    // Initialize gShape from wavelengths.csv if available; otherwise use fixed grid.
    // In Step B, we require exact match to the fixed grid, otherwise we fall back.
    inline void initialize_spectral_shape_from_csv(const std::string& wavelengthsCsvPath) {
        SpectralShape loaded;
        auto ws = load_csv_single(wavelengthsCsvPath);
        if (!ws.empty()) {
            // Sort and deduplicate tiny glitches
            std::sort(ws.begin(), ws.end());
            ws.erase(std::unique(ws.begin(), ws.end(), [](float a, float b) { return std::abs(a - b) < 1e-5f; }), ws.end());
            loaded.wavelengths = std::move(ws);
            loaded.K = static_cast<int>(loaded.wavelengths.size());
            loaded.lambdaMin = loaded.wavelengths.front();
            loaded.lambdaMax = loaded.wavelengths.back();
            // Estimate delta from first two, assume uniform for Step B
            loaded.delta = (loaded.K >= 2) ? (loaded.wavelengths[1] - loaded.wavelengths[0]) : 0.0f;
        }

        SpectralShape fixed = make_fixed_shape_400_800_5nm();
        if (shape_matches_fixed_grid(loaded)) {
            gShape = std::move(loaded);
        }
        else {
            // Fall back to fixed to ensure no behavior change in Step B
            gShape = std::move(fixed);
        }
    }

    // Initialize gShape directly from a CMF CSV’s wavelength column (agx-style axis ownership).
    inline void initialize_spectral_shape_from_cmf_triplets(const CMFTriplets& cmf) {
        SpectralShape loaded;

        // Prefer ybar’s wavelength list; fallback: xbar or zbar if needed.
        const std::vector<std::pair<float, float>>* col = nullptr;
        if (!cmf.ybar.empty()) col = &cmf.ybar;
        else if (!cmf.xbar.empty()) col = &cmf.xbar;
        else if (!cmf.zbar.empty()) col = &cmf.zbar;

        if (col && !col->empty()) {
            loaded.wavelengths.reserve(col->size());
            for (const auto& p : *col) loaded.wavelengths.push_back(p.first);

            // Ensure sorted unique
            std::sort(loaded.wavelengths.begin(), loaded.wavelengths.end());
            loaded.wavelengths.erase(
                std::unique(loaded.wavelengths.begin(), loaded.wavelengths.end(),
                    [](float a, float b) { return std::abs(a - b) < 1e-5f; }),
                loaded.wavelengths.end()
            );

            loaded.K = static_cast<int>(loaded.wavelengths.size());
            loaded.lambdaMin = loaded.wavelengths.front();
            loaded.lambdaMax = loaded.wavelengths.back();
            loaded.delta = (loaded.K >= 2) ? (loaded.wavelengths[1] - loaded.wavelengths[0]) : 0.0f;

            // Install and bump shape version if changed
            bool changed = (gShape.K != loaded.K) ||
                (gShape.lambdaMin != loaded.lambdaMin) ||
                (gShape.lambdaMax != loaded.lambdaMax) ||
                (gShape.wavelengths != loaded.wavelengths);

            gShape = std::move(loaded);
            if (changed) {
                ++gShapeVersion;
                mark_spectral_tables_dirty();
            }
        }
    }

    // --- Dye extinction curves ---
    inline Curve gEpsY, gEpsM, gEpsC;

    inline void set_dye_extinctions_linear(
        const std::vector<std::pair<float, float>>& y_linear,
        const std::vector<std::pair<float, float>>& m_linear,
        const std::vector<std::pair<float, float>>& c_linear) {
        build_curve_on_shape_from_linear_pairs(gEpsY, y_linear);
        build_curve_on_shape_from_linear_pairs(gEpsM, m_linear);
        build_curve_on_shape_from_linear_pairs(gEpsC, c_linear);
    }
    // set_dye_extinctions_log10 is NOT for agx-emulsion dye_density_* CSVs.
    // Those CSVs are linear optical densities already.
    inline void set_dye_extinctions_log10(
        const std::vector<std::pair<float, float>>& y_log10,
        const std::vector<std::pair<float, float>>& m_log10,
        const std::vector<std::pair<float, float>>& c_log10) {
        build_curve_on_shape_from_log10_pairs(gEpsY, y_log10);
        build_curve_on_shape_from_log10_pairs(gEpsM, m_log10);
        build_curve_on_shape_from_log10_pairs(gEpsC, c_log10);
    }

    inline float eps_yellow(float lambda) {
        if (!gEpsY.lambda_nm.empty()) return gEpsY.sample(lambda);
        return 1.4f * gaussian(lambda, 440.0f, 25.0f);
    }
    inline float eps_magenta(float lambda) {
        if (!gEpsM.lambda_nm.empty()) return gEpsM.sample(lambda);
        return 1.2f * gaussian(lambda, 540.0f, 30.0f);
    }
    inline float eps_cyan(float lambda) {
        if (!gEpsC.lambda_nm.empty()) return gEpsC.sample(lambda);
        return 1.1f * gaussian(lambda, 610.0f, 35.0f);
    }


    // -------------------------------------------------------------------------
    // CIE 1931 CMFs (Gaussian fallback).
    // -------------------------------------------------------------------------
    inline Curve gXBar, gYBar, gZBar;

    inline void set_cie_1931_2deg_cmf(
        const std::vector<std::pair<float, float>>& xbar,
        const std::vector<std::pair<float, float>>& ybar,
        const std::vector<std::pair<float, float>>& zbar)
    {
        build_curve_on_shape_from_linear_pairs(gXBar, xbar);
        build_curve_on_shape_from_linear_pairs(gYBar, ybar);
        build_curve_on_shape_from_linear_pairs(gZBar, zbar);
    }

    // --- Per‑pixel SPD reconstruction from DWG RGB (rectified CMF basis) ---

    // Build S = ∫ [max(0, CMF)]^T * [CMF] dλ and precompute its inverse once.
    // We’ll do it lazily the first time we need it.
    inline bool gSPDInit = false;
    inline float gS_inv[9]; // row-major inverse of 3x3 S

    inline void compute_S_inverse_once() {
        if (gSPDInit) return;

        // Accumulate S over the fixed grid with Δλ = kDelta
        double Sxx = 0, Sxy = 0, Sxz = 0;
        double Syx = 0, Syy = 0, Syz = 0;
        double Szx = 0, Szy = 0, Szz = 0;

        for (int i = 0; i < gShape.K; ++i) {
            const float bx = std::max(0.0f, gXbarTable[i]);
            const float by = std::max(0.0f, gYbarTable[i]);
            const float bz = std::max(0.0f, gZbarTable[i]);
            const float x = gXbarTable[i];
            const float y = gYbarTable[i];
            const float z = gZbarTable[i];

            Sxx += bx * x; Sxy += bx * y; Sxz += bx * z;
            Syx += by * x; Syy += by * y; Syz += by * z;
            Szx += bz * x; Szy += bz * y; Szz += bz * z;
        }
        const double dl = static_cast<double>(gDeltaLambda);
        Sxx *= dl; Sxy *= dl; Sxz *= dl;
        Syx *= dl; Syy *= dl; Syz *= dl;
        Szx *= dl; Szy *= dl; Szz *= dl;


        // Invert S via adjugate
        const double det =
            Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx);

        // Robustness: if ill‑conditioned, fall back to identity
        if (std::fabs(det) < 1e-20) {
            gS_inv[0] = 1; gS_inv[1] = 0; gS_inv[2] = 0;
            gS_inv[3] = 0; gS_inv[4] = 1; gS_inv[5] = 0;
            gS_inv[6] = 0; gS_inv[7] = 0; gS_inv[8] = 1;
            gSPDInit = true;
            return;
        }

        const double invDet = 1.0 / det;
        const double invSxx = (Syy * Szz - Syz * Szy) * invDet;
        const double invSxy = (Sxz * Szy - Sxy * Szz) * invDet;
        const double invSxz = (Sxy * Syz - Sxz * Syy) * invDet;
        const double invSyx = (Syz * Szx - Syx * Szz) * invDet;
        const double invSyy = (Sxx * Szz - Sxz * Szx) * invDet;
        const double invSyz = (Sxz * Syx - Sxx * Syz) * invDet;
        const double invSzx = (Syx * Szy - Syy * Szx) * invDet;
        const double invSzy = (Sxy * Szx - Sxx * Szy) * invDet;
        const double invSzz = (Sxx * Syy - Sxy * Syx) * invDet;

        gS_inv[0] = static_cast<float>(invSxx);
        gS_inv[1] = static_cast<float>(invSxy);
        gS_inv[2] = static_cast<float>(invSxz);
        gS_inv[3] = static_cast<float>(invSyx);
        gS_inv[4] = static_cast<float>(invSyy);
        gS_inv[5] = static_cast<float>(invSyz);
        gS_inv[6] = static_cast<float>(invSzx);
        gS_inv[7] = static_cast<float>(invSzy);
        gS_inv[8] = static_cast<float>(invSzz);

        gSPDInit = true;
    }

    inline void DWG_linear_to_XYZ(const float RGB[3], float XYZ[3]);

    inline void reconstruct_Ee_from_DWG_RGB(const float rgbDWG[3], std::vector<float>& Ee_out) {
        compute_S_inverse_once();

        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        XYZ[0] = std::max(0.0f, XYZ[0]);
        XYZ[1] = std::max(0.0f, XYZ[1]);
        XYZ[2] = std::max(0.0f, XYZ[2]);

        float cx = gS_inv[0] * XYZ[0] + gS_inv[1] * XYZ[1] + gS_inv[2] * XYZ[2];
        float cy = gS_inv[3] * XYZ[0] + gS_inv[4] * XYZ[1] + gS_inv[5] * XYZ[2];
        float cz = gS_inv[6] * XYZ[0] + gS_inv[7] * XYZ[1] + gS_inv[8] * XYZ[2];

        cx = std::max(0.0f, cx);
        cy = std::max(0.0f, cy);
        cz = std::max(0.0f, cz);

        const int K = gShape.K;
        Ee_out.resize(K);

        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i) {
            const float bx = std::max(0.0f, gXbarTable[i]);
            const float by = std::max(0.0f, gYbarTable[i]);
            const float bz = std::max(0.0f, gZbarTable[i]);

            const float Ei = std::max(1e-6f, cx * bx + cy * by + cz * bz);
            Ee_out[i] = Ei;
            Y_recon += static_cast<double>(Ei) * static_cast<double>(gYbarTable[i]);
        }

        if (Y_recon > 1e-20) {
            const float s = static_cast<float>(XYZ[1] / Y_recon);
            for (int i = 0; i < K; ++i)
                Ee_out[i] = std::max(0.0f, s * Ee_out[i]);
        }
    }
    // -------------------------------------------------------------------------
    // Hanatos 2025 spectral upsampling (direct spectra LUT, no bases)
    // -------------------------------------------------------------------------
    inline int gSpectralUpsamplingMode = 0; // 0: Hanatos (default), 1: CMF-basis
    inline bool gHanatosAvailable = false;
    
    // Ensure precomputed tables are up-to-date with current illuminant and shape.
    // Safe to call per-render; it will no-op when nothing changed.
    inline void ensure_precomputed_up_to_date() {
        const bool need =
            gPrecomputeDirty ||
            (gLastPrecomputeIllumVersion != gIllumVersion) ||
            (gLastPrecomputeShapeVersion != gShapeVersion);

        if (!need) return;

        precompute_spectral_tables();
        disable_hanatos_if_shape_mismatch();

        gLastPrecomputeIllumVersion = gIllumVersion;
        gLastPrecomputeShapeVersion = gShapeVersion;
        gPrecomputeDirty = false;
    }



    // Tri->quad mapping identical to agx-emulsion
    inline void tri2quad(float tx, float ty, float& qx, float& qy) {
        const float denom = std::max(1.0f - tx, 1e-10f);
        float x = (1.0f - tx);
        x = x * x;
        float y = ty / denom;
        qx = std::clamp(x, 0.0f, 1.0f);
        qy = std::clamp(y, 0.0f, 1.0f);
    }
    inline void quad2tri(float qx, float qy, float& tx, float& ty) {
        const float s = std::sqrt(std::max(0.0f, qx));
        tx = 1.0f - s;
        ty = qy * s;
    }

    inline NpySpectraLUT gHanSpectra;

    inline bool load_hanatos_spectra_lut(const std::string& path) {
        const bool ok = load_npy_spectra_lut(path, gHanSpectra);
        if (!ok) {
            gHanSpectra = NpySpectraLUT{};
            gHanatosAvailable = false;
            return false;
        }
        if (gHanSpectra.size <= 0 || gHanSpectra.numSamples != gShape.K) {
            // Mismatch: clear and mark unavailable
            gHanSpectra = NpySpectraLUT{};
            gHanatosAvailable = false;
            return false;
        }
        gHanatosAvailable = true;
        return true;
    }

    inline bool hanatos_shape_matches_current() {
        return (gHanSpectra.size > 0) && (gHanSpectra.numSamples == gShape.K);
    }

    inline void disable_hanatos_if_shape_mismatch() {
        if (gHanatosAvailable && !hanatos_shape_matches_current()) {
            gHanatosAvailable = false;
        }
    }    

    inline void hanatos_bilinear_spectrum(float qx, float qy, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (gHanSpectra.size <= 0) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
            return;
        }

        const int N = gHanSpectra.size;
        const float fx = std::clamp(qx, 0.0f, 1.0f) * (N - 1);
        const float fy = std::clamp(qy, 0.0f, 1.0f) * (N - 1);
        const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, N - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, N - 1);
        const int x1 = std::min(x0 + 1, N - 1);
        const int y1 = std::min(y0 + 1, N - 1);
        const float tx = fx - x0;
        const float ty = fy - y0;

        auto at = [&](int i, int j, int k) -> float {
            size_t idx = ((static_cast<size_t>(i) * N + j) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            return gHanSpectra.data[idx];
            };

        thread_local std::vector<float> s00, s10, s01, s11;
        if ((int)s00.size() != K) {
            s00.resize(K); s10.resize(K); s01.resize(K); s11.resize(K);
        }

        for (int k = 0; k < K; ++k) {
            s00[k] = at(x0, y0, k);
            s10[k] = at(x1, y0, k);
            s01[k] = at(x0, y1, k);
            s11[k] = at(x1, y1, k);
        }

        for (int k = 0; k < K; ++k) {
            const float a = s00[k] * (1.0f - tx) + s10[k] * tx;
            const float b = s01[k] * (1.0f - tx) + s11[k] * tx;
            Ee_out[k] = std::max(0.0f, a * (1.0f - ty) + b * ty);
        }
    }

    // Reconstruct Ee via Hanatos spectra LUT from DWG RGB (match agx-emulsion behavior)
    inline void reconstruct_Ee_from_DWG_RGB_hanatos(const float rgbDWG[3], std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (!gHanatosAvailable || !hanatos_shape_matches_current()) {
            reconstruct_Ee_from_DWG_RGB(rgbDWG, Ee_out);
            return;
        }

        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        XYZ[0] = std::max(0.0f, XYZ[0]);
        XYZ[1] = std::max(0.0f, XYZ[1]);
        XYZ[2] = std::max(0.0f, XYZ[2]);
        const float b = XYZ[0] + XYZ[1] + XYZ[2];

        float x = 1.0f / 3.0f, y = 1.0f / 3.0f;
        if (b > 1e-12f) {
            x = XYZ[0] / b;
            y = XYZ[1] / b;
        }
        const float sxy = x + y;
        if (sxy > 1.0f) {
            const float s = 1.0f / (sxy + 1e-6f);
            x *= s; y *= s;
        }
        x = std::clamp(x, 0.0f, 1.0f);
        y = std::clamp(y, 0.0f, 1.0f - x);

        float qx, qy;
        tri2quad(x, y, qx, qy);

        hanatos_bilinear_spectrum(qx, qy, Ee_out);

        if (b > 0.0f) {
            for (int i = 0; i < K; ++i) Ee_out[i] *= b;
        }

        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i)
            Y_recon += static_cast<double>(Ee_out[i]) * static_cast<double>(gYbarTable[i]);

        if (Y_recon > 1e-30) {
            const float s = static_cast<float>(XYZ[1] / Y_recon);
            for (int i = 0; i < K; ++i)
                Ee_out[i] = std::max(0.0f, s * Ee_out[i]);
        }
    }


    inline float cie_xbar(float lambda) {
        if (!gXBar.lambda_nm.empty()) return gXBar.sample(lambda);
        return 1.0f * gaussian(lambda, 595.0f, 40.0f) + 0.25f * gaussian(lambda, 445.0f, 20.0f);
    }
    inline float cie_ybar(float lambda) {
        if (!gYBar.lambda_nm.empty()) return gYBar.sample(lambda);
        return 1.0f * gaussian(lambda, 555.0f, 30.0f);
    }
    inline float cie_zbar(float lambda) {
        if (!gZBar.lambda_nm.empty()) return gZBar.sample(lambda);
        return 1.2f * gaussian(lambda, 445.0f, 25.0f);
    }

    // --- Baseline spectral densities (global, not per-dye) ---
    inline Curve gBaseMin, gBaseMid;
    inline bool gHasBaseline = false;

    inline void set_negative_baseline_linear(
        const std::vector<std::pair<float, float>>& min_linear,
        const std::vector<std::pair<float, float>>& mid_linear)
    {
        build_curve_on_shape_from_linear_pairs(gBaseMin, min_linear);
        build_curve_on_shape_from_linear_pairs(gBaseMid, mid_linear);
        gHasBaseline = !gBaseMin.lambda_nm.empty() && !gBaseMid.lambda_nm.empty();
    }

    inline void set_negative_baseline_log10(
        const std::vector<std::pair<float, float>>& min_log10,
        const std::vector<std::pair<float, float>>& mid_log10)
    {
        build_curve_on_shape_from_log10_pairs(gBaseMin, min_log10);
        build_curve_on_shape_from_log10_pairs(gBaseMid, mid_log10);
        gHasBaseline = !gBaseMin.lambda_nm.empty() && !gBaseMid.lambda_nm.empty();
    }

    inline void precompute_spectral_tables() {
        // No more forced fallback to 400–800 nm. Respect gShape as provided by CMF wavelengths.
        // Validate basic invariants.
        if (gShape.K <= 0 || gShape.wavelengths.size() != static_cast<size_t>(gShape.K)) {
            // Minimal safe fallback: 380–780 nm @ 5 nm (agx-emulsion)
            SpectralShape s;
            s.wavelengths.reserve(81);
            for (int i = 0; i < 81; ++i) s.wavelengths.push_back(380.0f + 5.0f * i);
            s.K = 81; s.lambdaMin = 380.0f; s.lambdaMax = 780.0f; s.delta = 5.0f;
            gShape = std::move(s);
            ++gShapeVersion;
        }

        disable_hanatos_if_shape_mismatch();

        // Set Δλ for all subsequent integrals
        gDeltaLambda = compute_delta_from_shape(gShape);

        // Ensure storage is allocated for current K
        const int K = gShape.K;
        gLambda.resize(K);
        gEpsYTable.resize(K);
        gEpsMTable.resize(K);
        gEpsCTable.resize(K);
        gXbarTable.resize(K);
        gYbarTable.resize(K);
        gZbarTable.resize(K);
        gBaselineMinTable.resize(K);
        gBaselineMidTable.resize(K);
        gIllumTable.resize(K);
        gAx.resize(K);
        gAy.resize(K);
        gAz.resize(K);

        // 1) Wavelength axis from gShape
        for (int i = 0; i < K; ++i) {
            gLambda[i] = gShape.wavelengths[i];
        }

        // 2) Sample all curves on the working grid (size-safe)
        for (int i = 0; i < K; ++i) {
            const float l = gLambda[i];

            // Measured dye extinctions: if pinned exactly to K, use direct indexing; otherwise sample by wavelength.
            if (!gEpsY.linear.empty() && (int)gEpsY.linear.size() == K) gEpsYTable[i] = gEpsY.linear[i];
            else gEpsYTable[i] = eps_yellow(l);

            if (!gEpsM.linear.empty() && (int)gEpsM.linear.size() == K) gEpsMTable[i] = gEpsM.linear[i];
            else gEpsMTable[i] = eps_magenta(l);

            if (!gEpsC.linear.empty() && (int)gEpsC.linear.size() == K) gEpsCTable[i] = gEpsC.linear[i];
            else gEpsCTable[i] = eps_cyan(l);

            // CMFs with the same size-safe rule
            if (!gXBar.linear.empty() && (int)gXBar.linear.size() == K) gXbarTable[i] = gXBar.linear[i];
            else gXbarTable[i] = cie_xbar(l);

            if (!gYBar.linear.empty() && (int)gYBar.linear.size() == K) gYbarTable[i] = gYBar.linear[i];
            else gYbarTable[i] = cie_ybar(l);

            if (!gZBar.linear.empty() && (int)gZBar.linear.size() == K) gZbarTable[i] = gZBar.linear[i];
            else gZbarTable[i] = cie_zbar(l);

            // Baseline (size-safe: only direct index if sizes match)
            gBaselineMinTable[i] = (gHasBaseline && (int)gBaseMin.linear.size() == K) ? gBaseMin.linear[i] : 0.0f;
            gBaselineMidTable[i] = (gHasBaseline && (int)gBaseMid.linear.size() == K) ? gBaseMid.linear[i] : 0.0f;

            // Illuminant: only direct index if sizes match; otherwise fallback to equal-energy for this sample
            if (!gIlluminantCurve.linear.empty() && (int)gIlluminantCurve.linear.size() == K) {
                gIllumTable[i] = gIlluminantCurve.linear[i];
            }
            else {
                gIllumTable[i] = illuminant_E(l); // equal-energy fallback
            }
        }


        // 3) Precompute Ee*CMFs and Yn normalization
        float Yn = 0.0f;
        for (int i = 0; i < K; ++i) {
            const float Ee = gIllumTable[i];
            const float x = gXbarTable[i];
            const float y = gYbarTable[i];
            const float z = gZbarTable[i];

            gAx[i] = Ee * x;
            gAy[i] = Ee * y;
            gAz[i] = Ee * z;

            Yn += gAy[i];

            gSPDInit = false;
        }

        gYnNorm = (Yn > 0.0f) ? Yn : 1.0f;
        gInvYn = 1.0f / gYnNorm;

        // Per-layer neutral exposures under current illuminant Ee
        auto neutral_exposure = [&](const Curve& sens) -> float {
            if (sens.lambda_nm.empty()) return 1.0f;
            double num = 0.0;
            double den = 0.0;
            for (int i = 0; i < gShape.K; ++i) {
                const float Ee = gIllumTable[i];
                const float s = sens.sample(gLambda[i]);
                num += static_cast<double>(Ee) * static_cast<double>(s);
                den += static_cast<double>(Ee);
            }
            return (den > 0.0) ? static_cast<float>(num / den) : 1.0f;
            };

        const float nB = neutral_exposure(gSensBlue);
        const float nG = neutral_exposure(gSensGreen);
        const float nR = neutral_exposure(gSensRed);

        gSensCorrB = (nB > 0.0f) ? (nG / nB) : 1.0f;
        gSensCorrG = 1.0f;
        gSensCorrR = (nR > 0.0f) ? (nG / nR) : 1.0f;
    }


    inline float baseline_density(float lambda, float w) {
        if (!gHasBaseline) return 0.0f;
        const float d0 = gBaseMin.sample(lambda);
        const float d1 = gBaseMid.sample(lambda);
        const float t = std::clamp(w, 0.0f, 1.0f);
        return d0 + t * (d1 - d0);
    }

    // INSERT in SpectralMath.h inside namespace Spectral, new helpers (non-global builder).

    inline void build_tables_from_curves_non_global(
        const Curve& epsY, const Curve& epsM, const Curve& epsC,
        const Curve& xbar, const Curve& ybar, const Curve& zbar,
        const Curve& illumView,
        const Curve& baseMin, const Curve& baseMid, bool hasBaseline,
        SpectralTables& T)
    {
        const int K = gShape.K;
        T.K = K;
        T.lambda = gShape.wavelengths;
        T.deltaLambda = compute_delta_from_shape(gShape);

        T.epsY.resize(K);
        T.epsM.resize(K);
        T.epsC.resize(K);
        T.Xbar.resize(K);
        T.Ybar.resize(K);
        T.Zbar.resize(K);
        T.Ax.resize(K);
        T.Ay.resize(K);
        T.Az.resize(K);

        const bool hasIll = (!illumView.linear.empty() && (int)illumView.linear.size() == K);
        double Yn = 0.0;
        for (int i = 0; i < K; ++i) {
            const float l = T.lambda[i];

            T.epsY[i] = (!epsY.linear.empty() && (int)epsY.linear.size() == K) ? epsY.linear[i] : eps_yellow(l);
            T.epsM[i] = (!epsM.linear.empty() && (int)epsM.linear.size() == K) ? epsM.linear[i] : eps_magenta(l);
            T.epsC[i] = (!epsC.linear.empty() && (int)epsC.linear.size() == K) ? epsC.linear[i] : eps_cyan(l);

            T.Xbar[i] = (!xbar.linear.empty() && (int)xbar.linear.size() == K) ? xbar.linear[i] : cie_xbar(l);
            T.Ybar[i] = (!ybar.linear.empty() && (int)ybar.linear.size() == K) ? ybar.linear[i] : cie_ybar(l);
            T.Zbar[i] = (!zbar.linear.empty() && (int)zbar.linear.size() == K) ? zbar.linear[i] : cie_zbar(l);

            const float Ee = hasIll ? illumView.linear[i] : 1.0f;
            T.Ax[i] = Ee * T.Xbar[i];
            T.Ay[i] = Ee * T.Ybar[i];
            T.Az[i] = Ee * T.Zbar[i];
            Yn += T.Ay[i];
        }
        T.invYn = (Yn > 0.0) ? (1.0f / (float)Yn) : 1.0f;

        T.hasBaseline = hasBaseline &&
            (int)baseMin.linear.size() == K &&
            (int)baseMid.linear.size() == K;
        if (T.hasBaseline) {
            T.baseMin = baseMin.linear;
            T.baseMid = baseMid.linear;
        }
        else {
            T.baseMin.assign(K, 0.0f);
            T.baseMid.assign(K, 0.0f);
        }
    }

    inline void compute_S_inverse_from_tables(const SpectralTables& T, float S_inv_out[9]) {
        // S = ∫ [max(0, CMF)]^T * [CMF] dλ under viewing axis in T
        double Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            const float x = T.Xbar[i], y = T.Ybar[i], z = T.Zbar[i];
            const float bx = std::max(0.0f, x);
            const float by = std::max(0.0f, y);
            const float bz = std::max(0.0f, z);
            Sxx += bx * x; Sxy += bx * y; Sxz += bx * z;
            Syx += by * x; Syy += by * y; Syz += by * z;
            Szx += bz * x; Szy += bz * y; Szz += bz * z;
        }
        const double dl = static_cast<double>(T.deltaLambda);
        Sxx *= dl; Sxy *= dl; Sxz *= dl; Syx *= dl; Syy *= dl; Syz *= dl; Szx *= dl; Szy *= dl; Szz *= dl;

        const double det = Sxx * (Syy * Szz - Syz * Szy) - Sxy * (Syx * Szz - Syz * Szx) + Sxz * (Syx * Szy - Syy * Szx);
        if (std::fabs(det) < 1e-20) {
            S_inv_out[0] = 1; S_inv_out[1] = 0; S_inv_out[2] = 0;
            S_inv_out[3] = 0; S_inv_out[4] = 1; S_inv_out[5] = 0;
            S_inv_out[6] = 0; S_inv_out[7] = 0; S_inv_out[8] = 1;
            return;
        }
        const double invDet = 1.0 / det;
        S_inv_out[0] = static_cast<float>((Syy * Szz - Syz * Szy) * invDet);
        S_inv_out[1] = static_cast<float>((Sxz * Szy - Sxy * Szz) * invDet);
        S_inv_out[2] = static_cast<float>((Sxy * Syz - Sxz * Syy) * invDet);
        S_inv_out[3] = static_cast<float>((Syz * Szx - Syx * Szz) * invDet);
        S_inv_out[4] = static_cast<float>((Sxx * Szz - Sxz * Szx) * invDet);
        S_inv_out[5] = static_cast<float>((Sxz * Syx - Sxx * Syz) * invDet);
        S_inv_out[6] = static_cast<float>((Syx * Szy - Syy * Szx) * invDet);
        S_inv_out[7] = static_cast<float>((Sxy * Szx - Sxx * Szy) * invDet);
        S_inv_out[8] = static_cast<float>((Sxx * Syy - Sxy * Syx) * invDet);
    }

    inline void reconstruct_Ee_from_DWG_RGB_with_tables(
        const float rgbDWG[3],
        const SpectralTables& T,
        const float S_inv[9],
        std::vector<float>& Ee_out)
    {
        float XYZ[3];
        DWG_linear_to_XYZ(rgbDWG, XYZ);
        XYZ[0] = std::max(0.0f, XYZ[0]);
        XYZ[1] = std::max(0.0f, XYZ[1]);
        XYZ[2] = std::max(0.0f, XYZ[2]);

        float cx = S_inv[0] * XYZ[0] + S_inv[1] * XYZ[1] + S_inv[2] * XYZ[2];
        float cy = S_inv[3] * XYZ[0] + S_inv[4] * XYZ[1] + S_inv[5] * XYZ[2];
        float cz = S_inv[6] * XYZ[0] + S_inv[7] * XYZ[1] + S_inv[8] * XYZ[2];
        cx = std::max(0.0f, cx);
        cy = std::max(0.0f, cy);
        cz = std::max(0.0f, cz);

        const int K = T.K;
        Ee_out.resize(K);
        double Y_recon = 0.0;
        for (int i = 0; i < K; ++i) {
            const float bx = std::max(0.0f, T.Xbar[i]);
            const float by = std::max(0.0f, T.Ybar[i]);
            const float bz = std::max(0.0f, T.Zbar[i]);
            const float Ei = std::max(1e-6f, cx * bx + cy * by + cz * bz);
            Ee_out[i] = Ei;
            Y_recon += static_cast<double>(Ei) * static_cast<double>(T.Ybar[i]);
        }
        if (Y_recon > 1e-20) {
            const float s = static_cast<float>(XYZ[1] / Y_recon);
            for (int i = 0; i < K; ++i)
                Ee_out[i] = std::max(0.0f, s * Ee_out[i]);
        }
    }

    // --- Forward declarations to satisfy helper below ---
    extern bool gUseSPDExposure; // defined later as inline variable

    inline void rgbDWG_to_layerExposures(const float rgbDWG[3], float E[3], float exposureScale);
    inline void applyLayerSensitivityBalance(float E[3]);
    inline void layerExposures_from_sceneSPD(const std::vector<float>& Ee, float E[3], float exposureScale);


    // Per-instance SPD path: never consult a global switch.
    // Only enter SPD if T and S_inv are valid and the caller requests it.
    // Per-instance SPD path: caller must explicitly request SPD via useSPDExposure
    inline void rgbDWG_to_layerExposures_flex_with_tables(
        const float rgbDWG[3], float E[3], float exposureScale,
        const SpectralTables* T, const float* S_inv /* size 9 */,
        int spectralMode /*0 Hanatos, 1 CMF-basis*/,
        bool useSPDExposure)
    {
        const bool spdOk = useSPDExposure && T && S_inv && T->K > 0;

        if (!spdOk) {
            rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);
            applyLayerSensitivityBalance(E);
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(T->K);

        if (spectralMode == 0 && gHanatosAvailable) {
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene);
        }
        else {
            reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, *T, S_inv, Ee_scene);
        }

        layerExposures_from_sceneSPD(Ee_scene, E, exposureScale);
        applyLayerSensitivityBalance(E);
    }

    // Per-instance SPD exposure using per-instance sensitivity curves
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale,
        const SpectralTables* T, const float* S_inv /* size 9 */,
        int spectralMode /*0 Hanatos, 1 CMF-basis*/,
        bool useSPDExposure,
        const Curve& sB, const Curve& sG, const Curve& sR)
    {
        const bool spdOk = useSPDExposure && T && S_inv && T->K > 0;
        if (!spdOk) {
            rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);
            applyLayerSensitivityBalance(E);
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(T->K);

        if (spectralMode == 0 && gHanatosAvailable) {
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene);
        }
        else {
            reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, *T, S_inv, Ee_scene);
        }

        // Integrate with per-instance sensitivity curves (no globals)
        layerExposures_from_sceneSPD_with_curves(Ee_scene, sB, sG, sR, E, exposureScale);
        applyLayerSensitivityBalance(E);
    }



    inline void dyes_to_XYZ_given_tables(
        const SpectralTables& T,
        const float dyes[3],        
        float XYZ[3])
    {
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            const float Dlambda = dyes[0] * T.epsY[i] + dyes[1] * T.epsM[i] + dyes[2] * T.epsC[i];
            const float Tlambda = std::exp(-kLn10 * Dlambda);
            X += Tlambda * T.Ax[i];
            Y += Tlambda * T.Ay[i];
            Z += Tlambda * T.Az[i];
        }
        const float s = T.deltaLambda * T.invYn;
        XYZ[0] = (float)(X * s);
        XYZ[1] = (float)(Y * s);
        XYZ[2] = (float)(Z * s);
    }

    inline void dyes_to_XYZ_with_baseline_given_tables(
        const SpectralTables& T,
        const float dyes[3],
        float neutralW,
        float XYZ[3])
    {
        const float w = std::clamp(neutralW, 0.0f, 1.0f);
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            const float dBase = T.hasBaseline ? (T.baseMin[i] + w * (T.baseMid[i] - T.baseMin[i])) : 0.0f;
            const float Dlambda = (dyes[0] + dBase) * T.epsY[i]
                + (dyes[1] + dBase) * T.epsM[i]
                + (dyes[2] + dBase) * T.epsC[i];
            const float Tlambda = std::exp(-kLn10 * Dlambda);
            X += Tlambda * T.Ax[i];
            Y += Tlambda * T.Ay[i];
            Z += Tlambda * T.Az[i];
        }
        const float s = T.deltaLambda * T.invYn;
        XYZ[0] = (float)(X * s);
        XYZ[1] = (float)(Y * s);
        XYZ[2] = (float)(Z * s);
    }

    // -----------------------------------------------------------------------------
    // Integrate spectral irradiance under CMFs (stored as Spectral::Curve) to XYZ
    // -----------------------------------------------------------------------------
    inline void Ee_to_XYZ_given_cmf(
        const std::vector<float>& Ee,
        const Spectral::Curve& xbar,
        const Spectral::Curve& ybar,
        const Spectral::Curve& zbar,
        float XYZ[3])
    {
        assert(Ee.size() == xbar.linear.size() &&
            Ee.size() == ybar.linear.size() &&
            Ee.size() == zbar.linear.size());

        double X = 0.0, Y = 0.0, Z = 0.0;
        for (size_t i = 0; i < Ee.size(); ++i) {
            const double E = Ee[i];
            X += E * xbar.linear[i];
            Y += E * ybar.linear[i];
            Z += E * zbar.linear[i];
        }

        // Compute Δλ from the global spectral shape
        const float deltaLambda = (Spectral::gShape.lambdaMax - Spectral::gShape.lambdaMin)
            / float(Spectral::gShape.K - 1);
        const float s = deltaLambda * Spectral::gInvYn;

        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
    }

    // Integrate spectral irradiance with per-instance tables (viewing axis and normalization)
    inline void Ee_to_XYZ_given_tables(
        const SpectralTables& T,
        const std::vector<float>& Ee,
        float XYZ[3])
    {
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        const int N = static_cast<int>(Ee.size());
        for (int i = 0; i < K; ++i) {
            const float e = (i < N) ? Ee[i] : 0.0f;
            X += static_cast<double>(e) * static_cast<double>(T.Xbar[i]);
            Y += static_cast<double>(e) * static_cast<double>(T.Ybar[i]);
            Z += static_cast<double>(e) * static_cast<double>(T.Zbar[i]);
        }
        const float s = T.deltaLambda * T.invYn;
        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
    }



    // -------------------------------------------------------------------------
    // Illuminant (equal-energy placeholder). Substitute scanning illuminant as needed.
    // -------------------------------------------------------------------------
    inline float illuminant_E(float /*lambda*/) { return 1.0f; }

    inline void fill_viewing_illuminant_Ee(float gain, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);
        for (int i = 0; i < K; ++i) {
            Ee_out[i] = std::max(0.0f, gain * gIllumTable[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Three-layer negative: sensitivities (now using measured curves).
    // -------------------------------------------------------------------------
    inline float sens_blue(float lambda) { return gSensBlue.sample(lambda); }
    inline float sens_green(float lambda) { return gSensGreen.sample(lambda); }
    inline float sens_red(float lambda) { return gSensRed.sample(lambda); }

    // -------------------------------------------------------------------------
    // Effective layer gain helpers
    // -------------------------------------------------------------------------
    inline float effective_layer_gain(const Curve& c) {
        if (c.lambda_nm.empty()) return 1.0f;
        float num = 0.0f, den = 0.0f;
        for (int i = 0; i < gShape.K; ++i) {
            const float Ee = gIllumTable[i];
            const float s = c.sample(gLambda[i]);
            num += Ee * s;
            den += Ee;
        }
        return (den > 0.0f) ? (num / den) : 1.0f;
    }



    inline void applyLayerSensitivityBalance(float E[3]) {
        E[0] *= gSensCorrB; // blue layer -> yellow dye
        E[1] *= gSensCorrG; // green layer -> magenta dye
        E[2] *= gSensCorrR; // red layer -> cyan dye
    }

    // -------------------------------------------------------------------------
    // DaVinci Wide Gamut linear RGB <-> CIE 1931 XYZ (official, DWG linear)
    // Source: Blackmagic Design "DaVinci Wide Gamut Intermediate" document.
    // These are row-major, used as M * v (row dot vector).
    // -------------------------------------------------------------------------
    struct Mat3 {
        float m[9];
        inline void mul(const float v[3], float out[3]) const {
            out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
            out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
            out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
        }
    };

    // -------------------------------------------------------------------------
    // Mapping from DWG Linear RGB to layer “exposure” vector E = [E_B, E_G, E_R].
    // -------------------------------------------------------------------------
    inline void rgbDWG_to_layerExposures(const float rgbDWG[3], float E[3], float exposureScale) {
        const float r = std::max(0.0f, rgbDWG[0]);
        const float g = std::max(0.0f, rgbDWG[1]);
        const float b = std::max(0.0f, rgbDWG[2]);
        const float M[9] = {
            0.05f, 0.09f, 0.75f,
            0.10f, 0.80f, 0.15f,
            0.85f, 0.12f, 0.05f
        };
        E[0] = exposureScale * (M[0] * r + M[1] * g + M[2] * b);
        E[1] = exposureScale * (M[3] * r + M[4] * g + M[5] * b);
        E[2] = exposureScale * (M[6] * r + M[7] * g + M[8] * b);
        E[0] = std::max(0.0f, E[0]);
        E[1] = std::max(0.0f, E[1]);
        E[2] = std::max(0.0f, E[2]);
    }
    // HIDDEN MESSAGE 2
    // -------------------------------------------------------------------------
    // Dye coupler model: per-layer H-D curve + small interlayer masking + base densities
    // -------------------------------------------------------------------------
    struct NegativeCouplerParams {
        // Per-dye maximum density (dynamic range before base; tweak per stock)
        float DmaxY, DmaxM, DmaxC;
        // Base (orange mask) densities
        float baseY, baseM, baseC;
        // Per-layer curve steepness (k) for log1p mapping
        float kB, kG, kR;
        // 3x3 masking matrix applied to [dY0, dM0, dC0] (post H-D)
        // Row-major: [ YY YM YC; MY MM MC; CY CM CC ]
        float mask[9];
    };

    // Default prototype negative (placeholder values; tune later)
    inline NegativeCouplerParams gNegParams = {
        // Dmax per dye (kept conservative for now)
        1.10f, 1.00f, 1.30f,
        // Base mask densities (preserve your current baseline look)
        0.04f, 0.02f, 0.06f,
        // H-D curve steepness (larger = quicker rise toward Dmax)
        6.0f, 6.0f, 6.0f,
        // Masking (small off-diagonals; close to your current values)
         0.98f, -0.06f, -0.02f,
        -0.03f,  0.98f, -0.05f,
        -0.02f, -0.04f,  0.98f
    };

    // Simple H-D-like curve: monotonic, saturating, smooth
    inline float hd_curve(float E, float Dmax, float k) {
        E = std::max(0.0f, E);
        if (k <= 0.0f) return std::min(Dmax, E);
        const float denom = std::log1p(k);
        if (denom <= 0.0f) return 0.0f;
        const float t = std::log1p(k * E) / denom; // 0..1 for E in 0..1 (approx), saturates smoothly
        return Dmax * std::max(0.0f, t);
    }

    // Compute layer exposures E = [E_B, E_G, E_R] by integrating a scene SPD Ee[i]
    // with the measured layer sensitivity curves on the fixed grid.    
    inline void layerExposures_from_sceneSPD(const std::vector<float>& Ee, float E[3], float exposureScale) {
        const int K = gShape.K;
        double Eb = 0.0, Eg = 0.0, Er = 0.0;

        for (int i = 0; i < K; ++i) {
            const float l = gLambda[i];

            const float sB = gSensBlue.linear[i];
            const float sG = gSensGreen.linear[i];
            const float sR = gSensRed.linear[i];

            const float e = (i < static_cast<int>(Ee.size())) ? Ee[i] : 0.0f;
            Eb += static_cast<double>(e) * static_cast<double>(sB);
            Eg += static_cast<double>(e) * static_cast<double>(sG);
            Er += static_cast<double>(e) * static_cast<double>(sR);
        }

        const double dl = static_cast<double>(gDeltaLambda) * static_cast<double>(std::max(0.0f, exposureScale));
        E[0] = std::max(0.0f, static_cast<float>(Eb * dl));
        E[1] = std::max(0.0f, static_cast<float>(Eg * dl));
        E[2] = std::max(0.0f, static_cast<float>(Er * dl));
    }


    // Integrate Ee with per-instance sensitivities curves (non-global)
    inline void layerExposures_from_sceneSPD_with_curves(
        const std::vector<float>& Ee, const Spectral::Curve& sB, const Spectral::Curve& sG, const Spectral::Curve& sR,
        float E[3], float exposureScale)
    {
        const int K = gShape.K;
        double Eb = 0.0, Eg = 0.0, Er = 0.0;
        for (int i = 0; i < K; ++i) {
            const float l = gLambda[i];
            const float e = (i < static_cast<int>(Ee.size())) ? Ee[i] : 0.0f;
            Eb += static_cast<double>(e) * static_cast<double>(sB.linear.empty() ? 0.0f : sB.linear[i]);
            Eg += static_cast<double>(e) * static_cast<double>(sG.linear.empty() ? 0.0f : sG.linear[i]);
            Er += static_cast<double>(e) * static_cast<double>(sR.linear.empty() ? 0.0f : sR.linear[i]);
        }
        const double dl = static_cast<double>(gDeltaLambda) * static_cast<double>(std::max(0.0f, exposureScale));
        E[0] = std::max(0.0f, static_cast<float>(Eb * dl));
        E[1] = std::max(0.0f, static_cast<float>(Eg * dl));
        E[2] = std::max(0.0f, static_cast<float>(Er * dl));
    }


    // Toggle: choose how to get E from DWG RGB
    inline bool gUseSPDExposure = false; // false = current 3x3 matrix path; true = reconstruct SPD path

    // Compute E from DWG via either matrix mapping (current path) or SPD integration (new path)
    inline void rgbDWG_to_layerExposures_flex(const float rgbDWG[3], float E[3], float exposureScale) {
        if (!gUseSPDExposure) {
            rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);
            applyLayerSensitivityBalance(E);
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(gShape.K);

        // SPD path: reconstruct the scene SPD per pixel then integrate with sensitivities
        if (gSpectralUpsamplingMode == 0) {
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene);
        }
        else {
            reconstruct_Ee_from_DWG_RGB(rgbDWG, Ee_scene);
        }
        layerExposures_from_sceneSPD(Ee_scene, E, exposureScale);
        applyLayerSensitivityBalance(E);
    }

    // Density curves from agx-emulsion's csv data.

    inline void density_curve_rgb_dwg(
        const float rgbDWG[3],
        float exposureScale,
        float D_out[3],
        float D_hd_nomask_out[3] = nullptr)

    {
        // 1) DWG -> layer exposures via selected method (matrix vs SPD)
        //    SPD path integrates reconstructed SPD with measured sensitivities.
        float E[3];
        rgbDWG_to_layerExposures_flex(rgbDWG, E, exposureScale);


        // 3) H-D curves from CSVs (no masking/base yet): Y<-B, M<-G, C<-R
        auto eval_density_curve = [](const Curve& curve, float exposure, float offset) -> float {
            if (curve.lambda_nm.empty()) {
                // Fallback to old hd_curve if no CSV loaded
                return hd_curve(exposure, 1.0f, 6.0f);
            }
            exposure = std::max(exposure, 1e-6f); // avoid log of zero
            float logE = std::log10(exposure) + offset;
            return curve.sample(logE);
            };

        // Apply per-layer offsets; these densities are “over B+F” (do NOT add base here)
        const float dY0 = eval_density_curve(gDensityCurveB, E[0], gDensityCurveLogEOffsetB); // Blue layer → Yellow dye
        const float dM0 = eval_density_curve(gDensityCurveG, E[1], gDensityCurveLogEOffsetG); // Green layer → Magenta dye
        const float dC0 = eval_density_curve(gDensityCurveR, E[2], gDensityCurveLogEOffsetR); // Red layer → Cyan dye

#ifdef JUICER_ENABLE_COUPLERS
        // Non-spatial DIR: adjust E using dY0,dM0,dC0 then re-evaluate densities
        
        // Fetching runtime params requires the instance; we do that in juicer-001 during render.
        // Here we just expose the container type; a helper in render will perform apply_runtime.
        // So do nothing here; actual scaling will be done in juicer-001 before calling this function when enabled.
#endif
        
        if (D_hd_nomask_out) {
            D_hd_nomask_out[0] = dY0;
            D_hd_nomask_out[1] = dM0;
            D_hd_nomask_out[2] = dC0;
        }

        // Apply DIR couplers (default no‑op); keep over‑B+F semantics
        float D_tmp[3] = { dY0, dM0, dC0 };
        apply_dir_couplers(D_tmp, E);

        D_out[0] = D_tmp[0];
        D_out[1] = D_tmp[1];
        D_out[2] = D_tmp[2];
    }

    inline void density_curve_from_exposures(const float E[3], float D_out[3]) {
        auto eval = [](const Curve& c, float e, float off) {
            if (c.lambda_nm.empty()) return hd_curve(e, 1.0f, 6.0f);
            e = std::max(e, 1e-6f);
            float logE = std::log10(e) + off;
            return c.sample(logE);
            };
        D_out[0] = eval(gDensityCurveB, E[0], gDensityCurveLogEOffsetB); // Yellow
        D_out[1] = eval(gDensityCurveG, E[1], gDensityCurveLogEOffsetG); // Magenta
        D_out[2] = eval(gDensityCurveR, E[2], gDensityCurveLogEOffsetR); // Cyan
    }

    // LogE-domain density sampling helpers (agx DIR parity) ===
    // Insert below density_curve_from_exposures(...) in Spectral namespace.

    inline float sample_density_at_logE(const Curve& c, float logE) {
        const size_t n = c.lambda_nm.size();
        if (n == 0) return 0.0f;
        if (!std::isfinite(logE)) {
            return c.linear.front(); // safe fallback
        }
        if (logE <= c.lambda_nm.front()) return c.linear.front();
        if (logE >= c.lambda_nm.back())  return c.linear.back();
        size_t i1 = 1;
        while (i1 < n && c.lambda_nm[i1] < logE) ++i1;
        const size_t i0 = i1 - 1;
        const float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
        const float y0 = c.linear[i0], y1 = c.linear[i1];
        const float t = (logE - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    // Balance negative sensitivities and density curves under a given reference illuminant.
    // Uses gSensBlue/gSensGreen/gSensRed and shifts R/B density curves to match G at logE = 0.
    inline void balance_negative_under_reference(const Curve& illumRef) {
        const int K = gShape.K;
        if (K <= 0) return;

        // --- Guard against empty or mismatched reference curves ---
        if (illumRef.linear.size() != static_cast<size_t>(K)) {
            // Synthesize a pinned equal-energy curve to avoid OOB access
            Curve eq;
            eq.lambda_nm = gShape.wavelengths;
            eq.linear.assign(K, 1.0f);
            balance_negative_under_reference(eq);
            return;
        }

        // Guard: sensitivities must be pinned to K before in-place scaling
        const bool sens_ok =
            gSensBlue.linear.size() == static_cast<size_t>(K) &&
            gSensGreen.linear.size() == static_cast<size_t>(K) &&
            gSensRed.linear.size() == static_cast<size_t>(K);

        if (!sens_ok) {
            // Nothing to do safely; avoid OOB writes
            return;
        }


        // 1) Compute neutral exposures under reference illuminant using measured sensitivities

        double nB = 0.0, nG = 0.0, nR = 0.0;
        for (int i = 0; i < K; ++i) {
            const double Ee = static_cast<double>(illumRef.linear[i]);
            const double sB = static_cast<double>(gSensBlue.linear[i]);
            const double sG = static_cast<double>(gSensGreen.linear[i]);
            const double sR = static_cast<double>(gSensRed.linear[i]);
            nB += Ee * sB;
            nG += Ee * sG;
            nR += Ee * sR;
        }
        if (nB <= 1e-20 || nG <= 1e-20 || nR <= 1e-20) return;

        // 2) Correction factors to match green
        const float corrB = static_cast<float>(nG / nB);
        const float corrR = static_cast<float>(nG / nR);

        // 3) Apply correction by scaling the sensitivity curves (in-place)
        for (int i = 0; i < K; ++i) {
            gSensBlue.linear[i] *= corrB;
            gSensRed.linear[i] *= corrR;
            // gSensGreen stays as reference
        }

        // 4) Align density curves so R/B match G at logE = 0
        auto interp_density_at = [](const Curve& c, float x)->float {
            const size_t n = c.lambda_nm.size();
            if (n == 0) return 0.0f;
            if (x <= c.lambda_nm.front()) return c.linear.front();
            if (x >= c.lambda_nm.back())  return c.linear.back();
            size_t i1 = 1;
            while (i1 < n && c.lambda_nm[i1] < x) ++i1;
            const size_t i0 = i1 - 1;
            const float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
            const float y0 = c.linear[i0], y1 = c.linear[i1];
            const float t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
            };

        auto find_logE_for_density_local = [&](const Curve& c, float targetY)->float {
            const size_t n = c.lambda_nm.size();
            if (n < 2) return 0.0f;
            for (size_t i = 1; i < n; ++i) {
                const float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                const float y0 = c.linear[i - 1], y1 = c.linear[i];
                if ((y0 <= targetY && targetY <= y1) || (y1 <= targetY && targetY <= y0)) {
                    const float t = (y1 != y0) ? (targetY - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            // Fallback to closest endpoint
            const float d0 = std::abs(targetY - c.linear.front());
            const float d1 = std::abs(targetY - c.linear.back());
            return (d0 < d1) ? c.lambda_nm.front() : c.lambda_nm.back();
            };

        const float targetG = interp_density_at(gDensityCurveG, 0.0f);
        const float shiftB = -find_logE_for_density_local(gDensityCurveB, targetG);
        const float shiftR = -find_logE_for_density_local(gDensityCurveR, targetG);

        for (float& x : gDensityCurveB.lambda_nm) x += shiftB;
        for (float& x : gDensityCurveR.lambda_nm) x += shiftR;

        mark_mixing_dirty();
    }

    // Non-global variant: balances sensitivities and shifts B/R density curve domains
    // so that at logE = 0 their densities match the green curve’s density.
    // Inputs: sensB/G/R_in and densB/G/R_in; Output: sensB/G/R_out and densB/G/R_out.
    // illumRef must be pinned to current gShape.
    inline void balance_negative_under_reference_non_global(
        const Curve& illumRef,
        const Curve& sensB_in, const Curve& sensG_in, const Curve& sensR_in,
        const Curve& densB_in, const Curve& densG_in, const Curve& densR_in,
        Curve& sensB_out, Curve& sensG_out, Curve& sensR_out,
        Curve& densB_out, Curve& densG_out, Curve& densR_out)
    {
        const int K = gShape.K;
        if (K <= 0 || illumRef.linear.size() != static_cast<size_t>(K)) {
            // Fallback: shallow copies
            sensB_out = sensB_in; sensG_out = sensG_in; sensR_out = sensR_in;
            densB_out = densB_in; densG_out = densG_in; densR_out = densR_in;
            return;
        }

        // Copy inputs
        sensB_out = sensB_in; sensG_out = sensG_in; sensR_out = sensR_in;
        densB_out = densB_in; densG_out = densG_in; densR_out = densR_in;

        // 1) Neutral exposures under reference illuminant
        auto neutral_exposure = [&](const Curve& s)->double {
            if (s.linear.size() != static_cast<size_t>(K)) return 1.0;
            double n = 0.0, d = 0.0;
            for (int i = 0; i < K; ++i) {
                const double Ee = (double)illumRef.linear[i];
                const double ss = (double)s.linear[i];
                n += Ee * ss;
                d += Ee;
            }
            return (d > 0.0) ? n / d : 1.0;
            };
        const double nB = neutral_exposure(sensB_in);
        const double nG = neutral_exposure(sensG_in);
        const double nR = neutral_exposure(sensR_in);

        if (nB <= 1e-20 || nG <= 1e-20 || nR <= 1e-20) return;

        const float corrB = (float)(nG / nB);
        const float corrR = (float)(nG / nR);

        // 2) Apply sensitivity scaling (in-place on copies)
        if (sensB_out.linear.size() == static_cast<size_t>(K))
            for (int i = 0; i < K; ++i) sensB_out.linear[i] *= corrB;
        if (sensR_out.linear.size() == static_cast<size_t>(K))
            for (int i = 0; i < K; ++i) sensR_out.linear[i] *= corrR;

        // 3) Shift B/R density domains to align with G at logE=0
        auto interp_density_at = [](const Curve& c, float x)->float {
            const size_t n = c.lambda_nm.size();
            if (n == 0) return 0.0f;
            if (x <= c.lambda_nm.front()) return c.linear.front();
            if (x >= c.lambda_nm.back())  return c.linear.back();
            size_t i1 = 1; while (i1 < n && c.lambda_nm[i1] < x) ++i1;
            const size_t i0 = i1 - 1;
            const float x0 = c.lambda_nm[i0], x1 = c.lambda_nm[i1];
            const float y0 = c.linear[i0], y1 = c.linear[i1];
            const float t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
            };
        auto find_logE_for_density_local = [&](const Curve& c, float targetY)->float {
            const size_t n = c.lambda_nm.size();
            if (n < 2) return 0.0f;
            for (size_t i = 1; i < n; ++i) {
                const float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                const float y0 = c.linear[i - 1], y1 = c.linear[i];
                if ((y0 <= targetY && targetY <= y1) || (y1 <= targetY && targetY <= y0)) {
                    const float t = (y1 != y0) ? (targetY - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            const float d0 = std::abs(targetY - (c.linear.empty() ? 0.0f : c.linear.front()));
            const float d1 = std::abs(targetY - (c.linear.empty() ? 0.0f : c.linear.back()));
            return (d0 < d1) ? (c.lambda_nm.empty() ? 0.0f : c.lambda_nm.front())
                : (c.lambda_nm.empty() ? 0.0f : c.lambda_nm.back());
            };

        const float targetG = interp_density_at(densG_out, 0.0f);

        const float shiftB = -find_logE_for_density_local(densB_out, targetG);
        const float shiftR = -find_logE_for_density_local(densR_out, targetG);

        for (float& x : densB_out.lambda_nm) x += shiftB;
        for (float& x : densR_out.lambda_nm) x += shiftR;
    }


    // Sample Y,M,C densities at given per-layer logE, directly from the
    // (already pre-warped) 1D density curves. These curves should be the
    // “before-DIR” curves if you apply pre-warp on stock load.
    inline void density_from_logE_with_offsets(const float logE[3], float D_out[3]) {
        D_out[0] = sample_density_at_logE(gDensityCurveB, logE[0]); // Y <- B layer
        D_out[1] = sample_density_at_logE(gDensityCurveG, logE[1]); // M <- G layer
        D_out[2] = sample_density_at_logE(gDensityCurveR, logE[2]); // C <- R layer
    }


    // Map layer exposures (E_B, E_G, E_R) to dye densities (D_Y, D_M, D_C)
    // 1) H-D curve per layer: Y<-B, M<-G, C<-R
    // 2) Apply masking matrix
    // 3) Add base densities
    inline void exposures_to_dyes(const float E[3], float D[3]) {
        D[0] = std::max(0.0f, hd_curve(E[0], gNegParams.DmaxY, gNegParams.kB)); // Y <- B
        D[1] = std::max(0.0f, hd_curve(E[1], gNegParams.DmaxM, gNegParams.kG)); // M <- G
        D[2] = std::max(0.0f, hd_curve(E[2], gNegParams.DmaxC, gNegParams.kR)); // C <- R
    }

    // -------------------------------------------------------------------------
    // Masking coupler matrix 
    // -------------------------------------------------------------------------
    inline void applyMaskingCoupler(const float E[3], float D[3]) {
        // Deprecated: linear-only coupler replaced by non-linear H-D + masking model
        exposures_to_dyes(E, D);
    }


    inline void dyes_to_XYZ_given_Ee(const float* dyes, float XYZ[3]) {
        BaselineCtx base;
        base.hasBaseline = false;
        base.w = 0.0f;
        base.baseMin = nullptr;
        base.baseMid = nullptr;

#if defined(__AVX2__)
        integrate_dyes_to_XYZ_avx2(
            dyes[0], dyes[1], dyes[2],
            gEpsYTable.data(), gEpsMTable.data(), gEpsCTable.data(),
            gAx.data(), gAy.data(), gAz.data(),
            gShape.K, base, XYZ);
#else
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = gShape.K;
        for (int i = 0; i < K; ++i) {
            const float Dlambda = dyes[0] * gEpsYTable[i]
                + dyes[1] * gEpsMTable[i]
                + dyes[2] * gEpsCTable[i];
            const float T = std::exp(-kLn10 * Dlambda);
            X += T * gAx[i];
            Y += T * gAy[i];
            Z += T * gAz[i];
        }
        const float s = gDeltaLambda * gInvYn;
        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
#endif
    }


    inline void dyes_to_XYZ_given_Ee_with_baseline(const float* dyes, float neutralW, float XYZ[3]) {
        BaselineCtx base;
        base.hasBaseline = gHasBaseline;
        base.w = std::clamp(neutralW, 0.0f, 1.0f);
        base.baseMin = gBaselineMinTable.data();
        base.baseMid = gBaselineMidTable.data();

#if defined(__AVX2__)
        integrate_dyes_to_XYZ_avx2(
            dyes[0], dyes[1], dyes[2],
            gEpsYTable.data(), gEpsMTable.data(), gEpsCTable.data(),
            gAx.data(), gAy.data(), gAz.data(),
            gShape.K, base, XYZ);
#else
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = gShape.K;
        for (int i = 0; i < K; ++i) {
            const float dBase = gHasBaseline
                ? (gBaselineMinTable[i] + base.w * (gBaselineMidTable[i] - gBaselineMinTable[i]))
                : 0.0f;

            const float Dlambda = (dyes[0] + dBase) * gEpsYTable[i]
                + (dyes[1] + dBase) * gEpsMTable[i]
                + (dyes[2] + dBase) * gEpsCTable[i];
            const float T = std::exp(-kLn10 * Dlambda);
            X += T * gAx[i];
            Y += T * gAy[i];
            Z += T * gAz[i];
        }
        const float s = gDeltaLambda * gInvYn;
        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
#endif
    }


    inline Mat3 gDWG_RGB_to_XYZ = { {
        0.70062239f,  0.14877482f,  0.10105872f,
        0.27411851f,  0.87363190f, -0.14775041f,
       -0.09896291f, -0.13789533f,  1.32591599f
    } };
    inline Mat3 gDWG_XYZ_to_RGB = { {
        1.51667204f, -0.28147805f, -0.14696363f,
       -0.46491710f,  1.25142378f,  0.17488461f,
        0.06484905f,  0.10913934f,  0.76141462f
    } };

    inline void XYZ_to_DWG_linear(const float XYZ[3], float RGB[3]) {
        gDWG_XYZ_to_RGB.mul(XYZ, RGB);
        RGB[0] = std::max(0.0f, RGB[0]);
        RGB[1] = std::max(0.0f, RGB[1]);
        RGB[2] = std::max(0.0f, RGB[2]);
    }

    inline void DWG_linear_to_XYZ(const float RGB[3], float XYZ[3]) {
        gDWG_RGB_to_XYZ.mul(RGB, XYZ);
    }

    inline float neutral_blend_weight_from_DWG_rgb(const float rgbDWG[3]) {
        // Y = row 2 of DWG_RGB_to_XYZ dot rgb
        const float Y = std::max(0.0f,
            gDWG_RGB_to_XYZ.m[3] * rgbDWG[0] +
            gDWG_RGB_to_XYZ.m[4] * rgbDWG[1] +
            gDWG_RGB_to_XYZ.m[5] * rgbDWG[2]);
        const float w = Y / 0.18f;
        return std::clamp(w, 0.0f, 1.0f);
    }



    inline void dwgRGB_through_negative(const float rgbIn[3], float exposureScale, float rgbOut[3]) {
        // 1) Scene → dye densities over B+F (apply Exposure here). Do NOT add base here.
        float D_curve_over_BF[3];
        density_curve_rgb_dwg(rgbIn, /*exposureScale*/ exposureScale, D_curve_over_BF);

        // 2) Use over‑B+F curves directly; baseline is added spectrally at integration time.
        float D[3] = { D_curve_over_BF[0], D_curve_over_BF[1], D_curve_over_BF[2] };        

        // 3) Neutral blend weight.
        const float w = neutral_blend_weight_from_DWG_rgb(rgbIn);

        // 4) Integrate under Ee_view using per-call normalization above.
        float XYZ[3];
        if (gHasBaseline) {
            dyes_to_XYZ_given_Ee_with_baseline(D, w, XYZ);
        }
        else {
            dyes_to_XYZ_given_Ee(D, XYZ);
        }

        // 6) XYZ → DWG
        XYZ_to_DWG_linear(XYZ, rgbOut);
    }





    // -------------------------------------------------------------------------
    // NEW: Convenience function to go DWG RGB -> dye densities with sensitivity gains
    // -------------------------------------------------------------------------
    inline void dwgRGB_to_dyes(const float rgbDWG[3], float exposureScale, float D[3]) {
        float E[3];
        rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);
        applyMaskingCoupler(E, D);
    }

} // namespace Spectral
