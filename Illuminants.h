// Illuminants.h
#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include "SpectralMath.h"
#include <iostream>


namespace Spectral {

    inline void set_illuminant_equal_energy();

    // Build an illuminant and set it into Spectral::gIlluminantCurve via set_illuminant_from_pairs.
    // During precompute_spectral_tables(), gIllumTable will be filled from this curve if present.

    // --------------------------
    // Helpers: normalization
    // --------------------------
    inline void normalize_mean_power(std::vector<std::pair<float, float>>& pairs) {
        if (pairs.empty()) return;
        float sum = 0.0f;
        for (const auto& p : pairs) sum += p.second;
        const float mean = sum / static_cast<float>(pairs.size());
        if (mean > 0.0f) {
            for (auto& p : pairs) p.second /= mean;
        }
    }

    // --------------------------
    // Physics: Planck blackbody
    // --------------------------
    inline float planck_blackbody(float lambda_nm, float T_kelvin) {
        // Spectral radiance up to a scale factor; we normalize later.
        // lambda in meters
        const double lambda_m = static_cast<double>(lambda_nm) * 1e-9;
        const double c = 2.99792458e8;
        const double h = 6.62607015e-34;
        const double k = 1.380649e-23;

        const double c1 = 2.0 * h * c * c;
        const double c2 = h * c / k;
        const double denom = std::exp(c2 / (lambda_m * static_cast<double>(T_kelvin))) - 1.0;
        const double L = (denom > 0.0) ? c1 / (std::pow(lambda_m, 5) * denom) : 0.0;
        return static_cast<float>(L);
    }

    // --------------------------
    // Placeholder filters
    // --------------------------
    inline float schott_KG3_transmission(float /*lambda_nm*/) {
        // TODO: replace with measured KG3 transmission; unity means "no attenuation".
        return 1.0f;
    }
    inline float generic_lens_transmission(float lambda_nm) {
        // Smooth attenuation: ~0.88 at 400 nm, ~0.95 at 550 nm, ~0.92 at 700 nm.
        const float blue = 1.0f - 0.12f * std::exp(-0.5f * std::pow((lambda_nm - 420.0f) / 35.0f, 2.0f));
        const float red = 1.0f - 0.06f * std::exp(-0.5f * std::pow((lambda_nm - 700.0f) / 60.0f, 2.0f));
        float t = blue * red;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        return t;
    }

    // Optional: allow user to feed measured filter curves (wavelength, transmission 0..1).
    // These are multiplied into the illuminant build if provided.
    inline Curve gFilterKG3Curve;      // leave empty to use schott_KG3_transmission()
    inline Curve gLensTransmissionCurve; // leave empty to use generic_lens_transmission()

    inline void set_filter_KG3_from_pairs(const std::vector<std::pair<float, float>>& pairs) {
        build_curve_on_shape_from_linear_pairs(gFilterKG3Curve, pairs);
    }
    inline void set_lens_transmission_from_pairs(const std::vector<std::pair<float, float>>& pairs) {
        build_curve_on_shape_from_linear_pairs(gLensTransmissionCurve, pairs);
    }

    // --------------------------
    // CSV-based illuminant loading
    // --------------------------
    inline bool load_illuminant_from_csv_pairs_mean_normalized(
        const std::vector<std::pair<float, float>>& inPairs,
        std::vector<std::pair<float, float>>& outPairs)
    {
        outPairs.clear();
        if (inPairs.empty()) return false;

        // Resample to current SpectralShape
        outPairs = Spectral::resample_pairs_to_shape(inPairs, Spectral::gShape);
        if (outPairs.empty()) return false;

        // Mean-power normalize (agx-emulsion style)
        normalize_mean_power(outPairs);
        return true;
    }

    // Cache for D65 loaded from disk and pinned to current SpectralShape
    inline Curve gIllumD65CurveLoaded;

    inline void set_illuminant_from_loaded_curve_or_pairs(const Curve& curve,
        const std::vector<std::pair<float, float>>& pairs)
    {
        if (!curve.lambda_nm.empty()) {
            // Curve is already pinned to SpectralShape
            const bool sameSize =
                curve.linear.size() == gIlluminantCurve.linear.size() &&
                curve.lambda_nm.size() == gIlluminantCurve.lambda_nm.size();

            bool identical = sameSize;
            if (identical) {
                for (size_t i = 0; i < curve.lambda_nm.size(); ++i) {
                    if (curve.lambda_nm[i] != gIlluminantCurve.lambda_nm[i]) {
                        identical = false; break;
                    }
                }
                if (identical) {
                    constexpr float eps = 1e-6f;
                    for (size_t i = 0; i < curve.linear.size(); ++i) {
                        if (std::fabs(curve.linear[i] - gIlluminantCurve.linear[i]) > eps) {
                            identical = false; break;
                        }
                    }
                }
            }

            if (identical) {
                return; // no change
            }

            gIlluminantCurve = curve;
            ++gIllumVersion;
            mark_spectral_tables_dirty();
            return;
        }
        if (!pairs.empty()) {
            set_illuminant_from_pairs(pairs);
            return;
        }
        // If both are empty, leave to caller to fall back.
    }

    // --------------------------
    // Builders for illuminants
    // --------------------------
    inline void build_blackbody_illuminant(float T_kelvin, bool apply_KG3, bool apply_lens) {
        if (gShape.K <= 0 || gShape.wavelengths.empty()) return;

        std::vector<std::pair<float, float>> pairs;
        pairs.reserve(static_cast<size_t>(gShape.K));

        for (int i = 0; i < gShape.K; ++i) {
            const float l = gShape.wavelengths[i];
            float E = planck_blackbody(l, T_kelvin);

            if (apply_KG3) {
                const float tKG3 = (!gFilterKG3Curve.lambda_nm.empty())
                    ? gFilterKG3Curve.sample(l)
                    : schott_KG3_transmission(l);
                E *= tKG3;
            }
            if (apply_lens) {
                const float tLens = (!gLensTransmissionCurve.lambda_nm.empty())
                    ? gLensTransmissionCurve.sample(l)
                    : generic_lens_transmission(l);
                E *= tLens;
            }
            pairs.emplace_back(l, E);
        }

        // Normalize average power to 1.0 (agx-emulsion style)
        normalize_mean_power(pairs);

        // Install into Spectral
        set_illuminant_from_pairs(pairs);
    }
    inline Spectral::Curve build_curve_D65_pinned(const std::string& csvPath) {
        Spectral::Curve c;
        auto pairs = Spectral::load_csv_pairs(csvPath);
        if (pairs.empty()) {
            c.lambda_nm = Spectral::gShape.wavelengths;
            c.linear.assign(Spectral::gShape.K, 1.0f);
            return c;
        }
        auto pinned_pairs = Spectral::resample_pairs_to_shape(pairs, Spectral::gShape);
        double sum = 0.0;
        for (auto& p : pinned_pairs) sum += p.second;
        const double mean = (pinned_pairs.empty() ? 1.0 : sum / pinned_pairs.size());
        c.lambda_nm = Spectral::gShape.wavelengths;
        c.linear.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            c.linear[i] = (mean > 0.0) ? static_cast<float>(pinned_pairs[i].second / mean)
                : pinned_pairs[i].second;
        }
        return c;
    }

    inline Spectral::Curve build_curve_D55_pinned(const std::string& csvPath) {
        Spectral::Curve c;
        auto pairs = Spectral::load_csv_pairs(csvPath);
        if (pairs.empty()) {
            c.lambda_nm = Spectral::gShape.wavelengths;
            c.linear.assign(Spectral::gShape.K, 1.0f);
            return c;
        }
        auto pinned_pairs = Spectral::resample_pairs_to_shape(pairs, Spectral::gShape);
        double sum = 0.0;
        for (auto& p : pinned_pairs) sum += p.second;
        const double mean = (pinned_pairs.empty() ? 1.0 : sum / pinned_pairs.size());
        c.lambda_nm = Spectral::gShape.wavelengths;
        c.linear.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            c.linear[i] = (mean > 0.0) ? static_cast<float>(pinned_pairs[i].second / mean)
                : pinned_pairs[i].second;
        }
        return c;
    }

    inline Spectral::Curve build_curve_D50_pinned(const std::string& csvPath) {
        Spectral::Curve c;
        auto pairs = Spectral::load_csv_pairs(csvPath);
        if (pairs.empty()) {
            c.lambda_nm = Spectral::gShape.wavelengths;
            c.linear.assign(Spectral::gShape.K, 1.0f);
            return c;
        }
        auto pinned_pairs = Spectral::resample_pairs_to_shape(pairs, Spectral::gShape);
        double sum = 0.0;
        for (auto& p : pinned_pairs) sum += p.second;
        const double mean = (pinned_pairs.empty() ? 1.0 : sum / pinned_pairs.size());
        c.lambda_nm = Spectral::gShape.wavelengths;
        c.linear.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            c.linear[i] = (mean > 0.0) ? static_cast<float>(pinned_pairs[i].second / mean)
                : pinned_pairs[i].second;
        }
        return c;
    }

    inline Spectral::Curve build_curve_TH_KG3_L_pinned(
        const std::string& kg3CsvPath, const std::string& lensCsvPath)
    {
        Spectral::Curve c;
        if (Spectral::gShape.K <= 0 || Spectral::gShape.wavelengths.empty()) {
            c.lambda_nm.clear(); c.linear.clear();
            return c;
        }

        // 1) 3200K blackbody
        std::vector<float> bb(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            bb[i] = planck_blackbody(Spectral::gShape.wavelengths[i], 3200.0f);
        }

        // 2) KG3 filter (resampled, unity fallback, exception-safe)
        std::vector<std::pair<float, float>> kg3_pairs;
        try { kg3_pairs = Spectral::load_csv_pairs(kg3CsvPath); }
        catch (...) { kg3_pairs.clear(); }
        if (kg3_pairs.empty()) {
            kg3_pairs = {
                { Spectral::gShape.wavelengths.front(), 1.0f },
                { Spectral::gShape.wavelengths.back(),  1.0f }
            };
        }
        auto kg3_pinned = Spectral::resample_pairs_to_shape(kg3_pairs, Spectral::gShape);

        // 3) Lens transmission (resampled, unity fallback, exception-safe)
        std::vector<std::pair<float, float>> lens_pairs;
        try { lens_pairs = Spectral::load_csv_pairs(lensCsvPath); }
        catch (...) { lens_pairs.clear(); }
        if (lens_pairs.empty()) {
            lens_pairs = {
                { Spectral::gShape.wavelengths.front(), 1.0f },
                { Spectral::gShape.wavelengths.back(),  1.0f }
            };
        }
        auto lens_pinned = Spectral::resample_pairs_to_shape(lens_pairs, Spectral::gShape);

        // 4) Multiply and mean-power normalize
        std::vector<float> combined(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            combined[i] = bb[i] * kg3_pinned[i].second * lens_pinned[i].second;
        }
        Spectral::mean_power_normalize(combined);

        // 5) Pin to shape without touching globals
        c.lambda_nm = Spectral::gShape.wavelengths;
        c.linear = std::move(combined);
        return c;
    }


    inline Spectral::Curve build_curve_equal_energy_pinned() {
        Spectral::Curve c;
        if (Spectral::gShape.K > 0 && !Spectral::gShape.wavelengths.empty()) {
            c.lambda_nm = Spectral::gShape.wavelengths;
            c.linear.assign(Spectral::gShape.K, 1.0f);
        }
        return c;
    }  
            

    inline std::vector<std::pair<float, float>> resample_to_plugin_grid(
        const std::vector<std::pair<float, float>>& inPairs)
    {
        // Delegates to SpectralShape-aware resampling
        return Spectral::resample_pairs_to_shape(inPairs, Spectral::gShape);
    }


    inline void set_illuminant_D65(const std::string& csvPath) {
        std::vector<std::pair<float, float>> d65_pairs;
        try {
            d65_pairs = load_csv_pairs(csvPath);
        }
        catch (...) {
            d65_pairs.clear();
        }

        if (d65_pairs.empty()) {
            set_illuminant_equal_energy();
            std::cerr << "[Illuminants] Warning: D65 CSV missing/invalid, using equal-energy fallback.\n";
            return;
        }

        auto pinned_pairs = resample_pairs_to_shape(d65_pairs, gShape);

        // Mean-power normalize
        double sum = 0.0;
        for (auto& p : pinned_pairs) sum += p.second;
        const double mean = pinned_pairs.empty() ? 1.0 : (sum / pinned_pairs.size());

        gIlluminantCurve.lambda_nm = gShape.wavelengths;
        gIlluminantCurve.linear.resize(gShape.K);
        for (int i = 0; i < gShape.K; ++i) {
            gIlluminantCurve.linear[i] = (float)((mean > 0.0) ? (pinned_pairs[i].second / mean) : pinned_pairs[i].second);
        }
        ++gIllumVersion;
        mark_spectral_tables_dirty();
    }

    inline void set_illuminant_D55(const std::string& csvPath) {
        std::vector<std::pair<float, float>> d55_pairs;
        try {
            d55_pairs = load_csv_pairs(csvPath);
        }
        catch (...) {
            d55_pairs.clear();
        }

        if (d55_pairs.empty()) {
            set_illuminant_equal_energy();
            std::cerr << "[Illuminants] Warning: D55 CSV missing/invalid, using equal-energy fallback.\n";
            return;
        }

        auto pinned_pairs = resample_pairs_to_shape(d55_pairs, gShape);

        // Mean-power normalize
        double sum = 0.0;
        for (auto& p : pinned_pairs) sum += p.second;
        const double mean = pinned_pairs.empty() ? 1.0 : (sum / pinned_pairs.size());

        gIlluminantCurve.lambda_nm = gShape.wavelengths;
        gIlluminantCurve.linear.resize(gShape.K);
        for (int i = 0; i < gShape.K; ++i) {
            gIlluminantCurve.linear[i] = (float)((mean > 0.0) ? (pinned_pairs[i].second / mean) : pinned_pairs[i].second);
        }
        ++gIllumVersion;
        mark_spectral_tables_dirty();
    }

    inline void set_illuminant_D50(const std::string& csvPath) {
        std::vector<std::pair<float, float>> d50_pairs;
        try {
            d50_pairs = load_csv_pairs(csvPath);
        }
        catch (...) {
            d50_pairs.clear();
        }

        if (d50_pairs.empty()) {
            set_illuminant_equal_energy();
            std::cerr << "[Illuminants] Warning: D50 CSV missing/invalid, using equal-energy fallback.\n";
            return;
        }

        auto pinned_pairs = resample_pairs_to_shape(d50_pairs, gShape);

        // Mean-power normalize
        double sum = 0.0;
        for (auto& p : pinned_pairs) sum += p.second;
        const double mean = pinned_pairs.empty() ? 1.0 : (sum / pinned_pairs.size());

        gIlluminantCurve.lambda_nm = gShape.wavelengths;
        gIlluminantCurve.linear.resize(gShape.K);
        for (int i = 0; i < gShape.K; ++i) {
            gIlluminantCurve.linear[i] = (float)((mean > 0.0) ? (pinned_pairs[i].second / mean) : pinned_pairs[i].second);
        }
        ++gIllumVersion;
        mark_spectral_tables_dirty();
    }
    // HIDDEN MESSAGE 3
    inline void set_illuminant_TH_KG3_L(const std::string& kg3CsvPath, const std::string& lensCsvPath) {
        // 1) Blackbody 3200K
        std::vector<float> bb(gShape.K);
        for (int i = 0; i < gShape.K; ++i) {
            bb[i] = planck_blackbody(gShape.wavelengths[i], 3200.0f);
        }

        // 2) KG3 filter (catch errors and fallback to unity range)
        std::vector<std::pair<float, float>> kg3_pairs;
        try {
            kg3_pairs = load_csv_pairs(kg3CsvPath);
        }
        catch (...) {
            kg3_pairs.clear();
        }
        if (kg3_pairs.empty()) {
            std::cerr << "[Illuminants] Warning: KG3 CSV missing/invalid, using unity transmission.\n";
            kg3_pairs = {
                { gShape.wavelengths.front(), 1.0f },
                { gShape.wavelengths.back(),  1.0f }
            };
        }
        auto kg3_pinned = resample_pairs_to_shape(kg3_pairs, gShape);

        // 3) Lens transmission
        std::vector<std::pair<float, float>> lens_pairs;
        try {
            lens_pairs = load_csv_pairs(lensCsvPath);
        }
        catch (...) {
            lens_pairs.clear();
        }
        if (lens_pairs.empty()) {
            std::cerr << "[Illuminants] Warning: Lens CSV missing/invalid, using unity transmission.\n";
            lens_pairs = {
                { gShape.wavelengths.front(), 1.0f },
                { gShape.wavelengths.back(),  1.0f }
            };
        }
        auto lens_pinned = resample_pairs_to_shape(lens_pairs, gShape);

        // 4) Multiply and mean-power normalize
        std::vector<float> combined(gShape.K);
        for (int i = 0; i < gShape.K; ++i) {
            combined[i] = bb[i] * kg3_pinned[i].second * lens_pinned[i].second;
        }
        Spectral::mean_power_normalize(combined);

        // 5) Assign
        gIlluminantCurve.lambda_nm = gShape.wavelengths;
        gIlluminantCurve.linear = std::move(combined);
        ++gIllumVersion;
        mark_spectral_tables_dirty();
    }


    // --------------------------
    // Public presets
    // --------------------------
    inline void set_illuminant_equal_energy() {
        // Pin equal-energy to the current shape to avoid empty-curve dereferences downstream
        if (gShape.K > 0 && !gShape.wavelengths.empty()) {
            gIlluminantCurve.lambda_nm = gShape.wavelengths;
            gIlluminantCurve.linear.assign(gShape.K, 1.0f);
        }
        else {
            gIlluminantCurve.lambda_nm.clear();
            gIlluminantCurve.linear.clear();
        }
        ++gIllumVersion;
        mark_spectral_tables_dirty();
    }


    inline void set_illuminant_TH_KG3_L() {
        build_blackbody_illuminant(3200.0f, /*apply_KG3*/ true, /*apply_lens*/ true);
    }


    inline void set_illuminant_T_incandescent() {
        // Rough incandescent ~2856K, no filters
        build_blackbody_illuminant(2856.0f, /*apply_KG3*/ false, /*apply_lens*/ false);
    }


    inline void set_illuminant_BB(float T_kelvin) {
        build_blackbody_illuminant(T_kelvin, /*apply_KG3*/ false, /*apply_lens*/ false);
    }



} // namespace Spectral
