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
#include <atomic>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "SpectralTables.h"
#include <mutex>

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
        const float* baseMin;
        const float* baseMid;
        float mix;
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

    struct DirRuntimeSnapshot {
        bool active = false;
        float M[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };
        float highShift = 0.0f;
        float dMax[3] = { 1.0f, 1.0f, 1.0f };
    };

#ifdef JUICER_ENABLE_COUPLERS
    inline std::atomic<DirRuntimeSnapshot> gDirRuntimeSnapshot{ DirRuntimeSnapshot{} };
    inline void set_dir_runtime_snapshot(const DirRuntimeSnapshot& snap) {
        gDirRuntimeSnapshot.store(snap, std::memory_order_release);
    }
    inline DirRuntimeSnapshot get_dir_runtime_snapshot() {
        return gDirRuntimeSnapshot.load(std::memory_order_acquire);
    }
#else
    inline DirRuntimeSnapshot gDirRuntimeSnapshot{};
    inline void set_dir_runtime_snapshot(const DirRuntimeSnapshot&) {}
    inline DirRuntimeSnapshot get_dir_runtime_snapshot() { return gDirRuntimeSnapshot; }
#endif   
    

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
                linear.push_back(p.second);
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
            const float v = sample_linear_pairs(sorted, l);
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
            curve.linear.push_back(p.second);
        }
    }

    // Build a curve pinned to SpectralShape from log10 pairs (convert to linear and resample without renormalizing).
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
        for (auto& p : log10pairs) {
            float val = std::pow(10.0f, p.second);
            if (!std::isfinite(val) || val < 0.0f)
                val = 0.0f;
            lin.emplace_back(p.first, val);
            
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

    inline float mitchell_weight(float x) {
        constexpr float B = 1.0f / 3.0f;
        constexpr float C = 1.0f / 3.0f;
        x = std::abs(x);
        if (x < 1.0f) {
            const float x2 = x * x;
            const float x3 = x2 * x;
            return ((12.0f - 9.0f * B - 6.0f * C) * x3 +
                (-18.0f + 12.0f * B + 6.0f * C) * x2 +
                (6.0f - 2.0f * B)) * (1.0f / 6.0f);
        }
        if (x < 2.0f) {
            const float x2 = x * x;
            const float x3 = x2 * x;
            return ((-B - 6.0f * C) * x3 +
                (6.0f * B + 30.0f * C) * x2 +
                (-12.0f * B - 48.0f * C) * x +
                (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
        }
        return 0.0f;
    }

    inline int reflect_index(int idx, int size) {
        if (size <= 0) return 0;
        while (idx < 0 || idx >= size) {
            if (idx < 0) {
                idx = -idx - 1;
            }
            else {
                idx = 2 * size - idx - 1;
            }
        }
        return idx;
    }

    inline void hanatos_cubic_spectrum(float qx, float qy, std::vector<float>& Ee_out) {
        const int K = gShape.K;
        Ee_out.resize(K);

        if (gHanSpectra.size <= 0) {
            std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);
            return;
        }

        const int N = gHanSpectra.size;
        const float fx = std::clamp(qx, 0.0f, 1.0f) * (N - 1);
        const float fy = std::clamp(qy, 0.0f, 1.0f) * (N - 1);
        const int baseX = std::clamp(static_cast<int>(std::floor(fx)), 0, N - 1);
        const int baseY = std::clamp(static_cast<int>(std::floor(fy)), 0, N - 1);
        const float tx = fx - static_cast<float>(baseX);
        const float ty = fy - static_cast<float>(baseY);

        auto at = [&](int i, int j, int k) -> float {
            size_t idx = ((static_cast<size_t>(i) * N + j) * static_cast<size_t>(gHanSpectra.numSamples) + k);
            return gHanSpectra.data[idx];
        };

        thread_local std::vector<float> tapBuffer;
        const size_t required = static_cast<size_t>(K) * 16;
        if (tapBuffer.size() != required) {
            tapBuffer.resize(required);
        }

        float wx[4];
        float wy[4];
        int xIndices[4];
        int yIndices[4];

        for (int i = 0; i < 4; ++i) {
            const int offset = i - 1;
            wx[i] = mitchell_weight(static_cast<float>(offset) - tx);
            xIndices[i] = reflect_index(baseX + offset, N);
            wy[i] = mitchell_weight(static_cast<float>(offset) - ty);
            yIndices[i] = reflect_index(baseY + offset, N);
        }

        int tap = 0;
        for (int j = 0; j < 4; ++j) {
            const int yIdx = yIndices[j];
            for (int i = 0; i < 4; ++i) {
                const int xIdx = xIndices[i];
                float* dest = tapBuffer.data() + static_cast<size_t>(tap) * static_cast<size_t>(K);
                for (int k = 0; k < K; ++k) {
                    dest[k] = at(xIdx, yIdx, k);
                }
                ++tap;
            }
        }

        std::fill(Ee_out.begin(), Ee_out.end(), 0.0f);

        float totalWeight = 0.0f;
        tap = 0;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                const float w = wx[i] * wy[j];
                totalWeight += w;
                const float* src = tapBuffer.data() + static_cast<size_t>(tap) * static_cast<size_t>(K);
                for (int k = 0; k < K; ++k) {
                    Ee_out[k] += src[k] * w;
                }
                ++tap;
            }
        }

        if (totalWeight > 0.0f) {
            const float invW = 1.0f / totalWeight;
            for (int k = 0; k < K; ++k) {
                Ee_out[k] *= invW;
            }
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

        hanatos_cubic_spectrum(qx, qy, Ee_out);

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
        double sumAx = 0.0;
        double sumAy = 0.0;
        double sumAz = 0.0;
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
            sumAx += T.Ax[i];
            sumAy += T.Ay[i];
            sumAz += T.Az[i];
        }
        T.invYn = (Yn > 0.0) ? (1.0f / (float)Yn) : 1.0f;
        const double scale = static_cast<double>(T.invYn);
        T.whiteXYZ[0] = static_cast<float>(scale * sumAx);
        T.whiteXYZ[1] = static_cast<float>(scale * sumAy);
        T.whiteXYZ[2] = static_cast<float>(scale * sumAz);

        T.hasBaseline = hasBaseline &&
            (int)baseMin.linear.size() == K;
        if (T.hasBaseline) {
            T.baseMin = baseMin.linear;
            if ((int)baseMid.linear.size() == K) {
                T.baseMid = baseMid.linear;
            }
            else {
                T.baseMid.assign(K, 0.0f);
            }
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
    inline void layerExposures_from_sceneSPD(const std::vector<float>& Ee, float E[3], float exposureScale);


    // Per-instance SPD path: never consult a global switch.
    // Only enter SPD if T and S_inv are valid and the caller requests it.
    // Per-instance SPD path: caller must explicitly request SPD via useSPDExposure
    inline void rgbDWG_to_layerExposures_flex_with_tables(
        const float rgbDWG[3], float E[3], float exposureScale,
        const SpectralTables* T, const float* S_inv /* size 9 */,
        bool useSPDExposure)
    {
        const bool spdOk = useSPDExposure && T && S_inv && T->K > 0;

        if (!spdOk) {
            rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);            
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(T->K);

        if (gHanatosAvailable) {
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene);
        }
        else {
            reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, *T, S_inv, Ee_scene);
        }

        layerExposures_from_sceneSPD(Ee_scene, E, exposureScale);        
    }

    // Per-instance SPD exposure using per-instance sensitivity curves
    inline void rgbDWG_to_layerExposures_from_tables_with_curves(
        const float rgbDWG[3], float E[3], float exposureScale,
        const SpectralTables* T, const float* S_inv /* size 9 */,        
        bool useSPDExposure,
        const Curve& sB, const Curve& sG, const Curve& sR)
    {
        const bool spdOk = useSPDExposure && T && S_inv && T->K > 0;
        if (!spdOk) {
            rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);            
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(T->K);

        if (gHanatosAvailable) {
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene);
        }
        else {
            reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, *T, S_inv, Ee_scene);
        }

        // Integrate with per-instance sensitivity curves (no globals)
        layerExposures_from_sceneSPD_with_curves(Ee_scene, sB, sG, sR, E, exposureScale);        
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
        const float s = T.invYn;
        XYZ[0] = (float)(X * s);
        XYZ[1] = (float)(Y * s);
        XYZ[2] = (float)(Z * s);
    }

    inline void dyes_to_XYZ_with_baseline_given_tables(
        const SpectralTables& T,
        const float dyes[3],        
        float XYZ[3])
    {        
        double X = 0.0, Y = 0.0, Z = 0.0;
        const int K = T.K;
        for (int i = 0; i < K; ++i) {
            const float baseSpectral = T.hasBaseline
                ? T.baseMin[i]
                : 0.0f;
            const float Dlambda = dyes[0] * T.epsY[i]
                + dyes[1] * T.epsM[i]
                + dyes[2] * T.epsC[i]
                + baseSpectral;
            const float Tlambda = std::exp(-kLn10 * Dlambda);
            X += Tlambda * T.Ax[i];
            Y += Tlambda * T.Ay[i];
            Z += Tlambda * T.Az[i];
        }
        const float s = T.invYn;
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
        const float s = Spectral::gInvYn;

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
        const float s = T.invYn;
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
    inline NegativeCouplerParams make_default_neg_params() {
        return {
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
    }

    inline std::mutex gNegParamsMutex;
    inline NegativeCouplerParams gNegParams = make_default_neg_params();

    inline void set_neg_params(const NegativeCouplerParams& params) {
        std::lock_guard<std::mutex> lock(gNegParamsMutex);
        gNegParams = params;
    }

    inline NegativeCouplerParams get_neg_params() {
        std::lock_guard<std::mutex> lock(gNegParamsMutex);
        return gNegParams;
    }

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
            return;
        }

        thread_local std::vector<float> Ee_scene;
        Ee_scene.resize(gShape.K);

        // SPD path: reconstruct the scene SPD per pixel then integrate with sensitivities
        if (gHanatosAvailable) {
            reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee_scene);
        }
        else {
            reconstruct_Ee_from_DWG_RGB(rgbDWG, Ee_scene);
        }
        layerExposures_from_sceneSPD(Ee_scene, E, exposureScale);        
    }    

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

    // Map layer exposures (E_B, E_G, E_R) to dye densities (D_Y, D_M, D_C)
    // 1) H-D curve per layer: Y<-B, M<-G, C<-R
    // 2) Apply masking matrix
    // 3) Add base densities
    inline void exposures_to_dyes_with_params(const float E[3], float D[3], const NegativeCouplerParams& negParams) {
        D[0] = std::max(0.0f, hd_curve(E[0], negParams.DmaxY, negParams.kB)); // Y <- B
        D[1] = std::max(0.0f, hd_curve(E[1], negParams.DmaxM, negParams.kG)); // M <- G
        D[2] = std::max(0.0f, hd_curve(E[2], negParams.DmaxC, negParams.kR)); // C <- R
    }
    inline void exposures_to_dyes(const float E[3], float D[3]) {
        const NegativeCouplerParams negParams = get_neg_params();
        exposures_to_dyes_with_params(E, D, negParams);
    }

    // -------------------------------------------------------------------------
    // Masking coupler matrix 
    // -------------------------------------------------------------------------
    inline void applyMaskingCoupler(const float E[3], float D[3]) {        
        const NegativeCouplerParams negParams = get_neg_params();
        exposures_to_dyes_with_params(E, D, negParams);
        const float* M = negParams.mask;
        const float y = M[0] * D[0] + M[1] * D[1] + M[2] * D[2] + negParams.baseY;
        const float m = M[3] * D[0] + M[4] * D[1] + M[5] * D[2] + negParams.baseM;
        const float c = M[6] * D[0] + M[7] * D[1] + M[8] * D[2] + negParams.baseC;
        D[0] = std::max(0.0f, y);
        D[1] = std::max(0.0f, m);
        D[2] = std::max(0.0f, c);
    }


    inline void dyes_to_XYZ_given_Ee(const float* dyes, float XYZ[3]) {
        BaselineCtx base;
        base.hasBaseline = false;
        base.baseMin = nullptr;
        base.baseMid = nullptr;
        base.mix = 0.0f;

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
        const float s = gInvYn;
        XYZ[0] = static_cast<float>(X * s);
        XYZ[1] = static_cast<float>(Y * s);
        XYZ[2] = static_cast<float>(Z * s);
#endif
    }


    inline void dyes_to_XYZ_given_Ee_with_baseline(const float* dyes, float neutralW, float XYZ[3]) {
        const float mix = std::clamp(neutralW, 0.0f, 1.0f);
        BaselineCtx base;
        const bool tablesReady =
            !gBaselineMinTable.empty() &&
            !gBaselineMidTable.empty() &&
            gBaselineMinTable.size() == static_cast<size_t>(gShape.K) &&
            gBaselineMidTable.size() == static_cast<size_t>(gShape.K);

        base.hasBaseline = gHasBaseline && tablesReady;
        base.baseMin = tablesReady ? gBaselineMinTable.data() : nullptr;
        base.baseMid = tablesReady ? gBaselineMidTable.data() : nullptr;
        base.mix = mix;

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
            float baseSpectral = 0.0f;
            if (base.hasBaseline && base.baseMin && base.baseMid) {
                const float minV = base.baseMin[i];
                const float midV = base.baseMid[i];
                baseSpectral = minV + base.mix * (midV - minV);
            }

            const float Dlambda = dyes[0] * gEpsYTable[i]
                + dyes[1] * gEpsMTable[i]
                + dyes[2] * gEpsCTable[i]
                + baseSpectral;
            const float T = std::exp(-kLn10 * Dlambda);
            X += T * gAx[i];
            Y += T * gAy[i];
            Z += T * gAz[i];
        }
        const float s = gInvYn;
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
    //CAT02/Von Kries based chromatic adaptation (matches colour.XYZ_to_RGB default)
    inline const float gDWG_WhitePoint_XYZ[3] = {
        0.950455f, 1.0f, 1.089058f
    };

    inline void chromatic_adapt_XYZ_CAT02(
        const float XYZ[3],
        const float srcWhiteXYZ[3],
        const float dstWhiteXYZ[3],
        float outXYZ[3])
    {
        static const float M[9] = {
             0.7328000f,  0.4296000f, -0.1624000f,
            -0.7036000f,  1.6975000f,  0.0061000f,
             0.0030000f,  0.0136000f,  0.9834000f
        };
        static const float M_inv[9] = {
            1.0961238f, -0.2788690f,  0.1827452f,
            0.4543690f,  0.4735332f,  0.0720978f,
           -0.0096276f, -0.0056980f,  1.0153256f
        };

        auto mul3 = [](const float m[9], const float v[3], float dst[3]) {
            dst[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
            dst[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
            dst[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
            };

        auto sanitize = [](float v) -> float {
            return std::isfinite(v) ? std::max(0.0f, v) : 0.0f;
            };

        float srcWhite[3] = {
            sanitize(srcWhiteXYZ[0]),
            sanitize(srcWhiteXYZ[1]),
            sanitize(srcWhiteXYZ[2])
        };
        float dstWhite[3] = {
            sanitize(dstWhiteXYZ[0]),
            sanitize(dstWhiteXYZ[1]),
            sanitize(dstWhiteXYZ[2])
        };

        const float srcY = (srcWhite[1] > 0.0f) ? srcWhite[1] : 1.0f;
        const float dstY = (dstWhite[1] > 0.0f) ? dstWhite[1] : 1.0f;
        const float srcScale = 1.0f / srcY;
        const float dstScale = 1.0f / dstY;
        srcWhite[0] *= srcScale; srcWhite[1] = 1.0f; srcWhite[2] *= srcScale;
        dstWhite[0] *= dstScale; dstWhite[1] = 1.0f; dstWhite[2] *= dstScale;

        float srcLMS[3];
        float dstLMS[3];
        float XYZ_LMS[3];
        mul3(M, srcWhite, srcLMS);
        mul3(M, dstWhite, dstLMS);
        mul3(M, XYZ, XYZ_LMS);

        float scale[3];
        scale[0] = (srcLMS[0] > 1e-6f) ? (dstLMS[0] / srcLMS[0]) : 1.0f;
        scale[1] = (srcLMS[1] > 1e-6f) ? (dstLMS[1] / srcLMS[1]) : 1.0f;
        scale[2] = (srcLMS[2] > 1e-6f) ? (dstLMS[2] / srcLMS[2]) : 1.0f;

        float adaptedLMS[3] = {
            scale[0] * XYZ_LMS[0],
            scale[1] * XYZ_LMS[1],
            scale[2] * XYZ_LMS[2]
        };

        mul3(M_inv, adaptedLMS, outXYZ);
    }

    inline void XYZ_to_DWG_linear_adapted(
        const SpectralTables& tables,
        const float XYZ[3],
        float RGB[3])
    {
        float srcWhite[3] = {
            tables.whiteXYZ[0],
            tables.whiteXYZ[1],
            tables.whiteXYZ[2]
        };
        const float sum = srcWhite[0] + srcWhite[1] + srcWhite[2];
        if (!std::isfinite(sum) || sum <= 0.0f) {
            srcWhite[0] = gDWG_WhitePoint_XYZ[0];
            srcWhite[1] = gDWG_WhitePoint_XYZ[1];
            srcWhite[2] = gDWG_WhitePoint_XYZ[2];
        }

        float adaptedXYZ[3];
        chromatic_adapt_XYZ_CAT02(XYZ, srcWhite, gDWG_WhitePoint_XYZ, adaptedXYZ);
        gDWG_XYZ_to_RGB.mul(adaptedXYZ, RGB);
    }

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

    // -------------------------------------------------------------------------
    // NEW: Convenience function to go DWG RGB -> dye densities with sensitivity gains
    // -------------------------------------------------------------------------
    inline void dwgRGB_to_dyes(const float rgbDWG[3], float exposureScale, float D[3]) {
        float E[3];
        rgbDWG_to_layerExposures(rgbDWG, E, exposureScale);
        applyMaskingCoupler(E, D);
    }

} // namespace Spectral
