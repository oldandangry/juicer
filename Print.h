// Print.h
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include "SpectralTables.h"
#include "SpectralMath.h"
#include "Illuminants.h"
#include "WorkingState.h"
#include "ProfileJSONLoader.h"
#include "AkimaInterpolator.h"
#include <sstream>
#include <algorithm>


namespace JuicerTrace {
    void write(const char* tag, const std::string& msg);
}

// Forward-declare WorkingState to avoid header cycles
struct WorkingState;

namespace Print {

    constexpr float kEnlargerSteps = 170.0f;

    inline float blend_dichroic_filter_linear(float curveVal, float normalizedAmount) {
        const float c = std::isfinite(curveVal) ? curveVal : 1.0f;
        const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
        return 1.0f - (1.0f - c) * a;
    }

    inline float compose_dichroic_amount(float neutralAmount, float deltaAmount) {
        const float neutral = std::isfinite(neutralAmount)
            ? std::clamp(neutralAmount, 0.0f, 1.0f)
            : 0.0f;
        float deltaSteps = std::isfinite(deltaAmount) ? deltaAmount : 0.0f;
        const float shiftLimit = kEnlargerSteps;
        deltaSteps = std::clamp(deltaSteps, -shiftLimit, shiftLimit);
        const float totalSteps = neutral * kEnlargerSteps + deltaSteps;
        return totalSteps / kEnlargerSteps;
    }

    struct Profile {
        Spectral::Curve epsC, epsM, epsY;     // print dye extinction (OD/λ)
        Spectral::Curve dcC, dcM, dcY;        // logE -> D (C,M,Y)
        Spectral::Curve baseMin, baseMid;     // optional print baseline (D-min/mid)
        bool hasBaseline = false;
        bool glareRemoved = false;
        float logEOffC = 0.0f, logEOffM = 0.0f, logEOffY = 0.0f; // legacy per-channel logE offsets (unused)

        // Print paper spectral log-sensitivity (B/G/R on disk => Y/M/C mapping), pinned to shape (log10 domain in CSVs)
        Spectral::Curve sensY_log, sensM_log, sensC_log;
    };


    struct Params {
        bool bypass = false;     // if true, bypass print (show negative)
        float exposure = 1.0f;   // enlarger exposure scalar
        float preflashExposure = 0.0f; // additional uniform print exposure (linear scale)
        float yFilter = 0.0f;    // delta from neutral baseline in Durst steps (±170)
        float mFilter = 0.0f;
        float cFilter = 0.0f;    // optional

        // Whether print exposure compensation is enabled in the UI.
        bool exposureCompensationEnabled = false;
        // Slider EV scale (2^EV) used for the mid-gray probe when compensation is enabled.
        float exposureCompensationScale = 1.0f;
    };


    struct Runtime {
        Profile profile;
        Spectral::Curve illumEnlarger; // pinned to gShape
        Spectral::Curve illumView;     // pinned to gShape

        // New: dichroic filter transmittance curves (normalized 0..1, pinned to gShape)
        Spectral::Curve filterY;
        Spectral::Curve filterM;
        Spectral::Curve filterC;
        // Neutral baseline scalars (0..1) used for compensation probes
        float neutralY = 1.0f;
        float neutralM = 1.0f;
        float neutralC = 1.0f;
    };

    // Profile validity: spectral-domain curves must match current spectral shape (K).
    // Density curves (dcC/M/Y) are in log-exposure domain, so they only need to be non-empty.
    inline bool profile_is_valid(const Profile& p) {
        const int K = Spectral::gShape.K;
        if (K <= 0) return false;

        auto spectral_curve_ok = [K](const Spectral::Curve& c) -> bool {
            return !c.linear.empty() && static_cast<int>(c.linear.size()) == K;
            };
        auto logE_curve_ok = [](const Spectral::Curve& c) -> bool {
            return !c.linear.empty();
            };

        // Require spectral print dye EPS
        if (!spectral_curve_ok(p.epsC) ||
            !spectral_curve_ok(p.epsM) ||
            !spectral_curve_ok(p.epsY)) {
            return false;
        }
        // Require print paper density curves (logE -> D)
        if (!logE_curve_ok(p.dcC) ||
            !logE_curve_ok(p.dcM) ||
            !logE_curve_ok(p.dcY)) {
            return false;
        }
        // Require print paper spectral log-sensitivities pinned to shape
        const bool sensOK =
            spectral_curve_ok(p.sensY_log) &&
            spectral_curve_ok(p.sensM_log) &&
            spectral_curve_ok(p.sensC_log);
        if (!sensOK) {
            return false;
        }

        if (p.hasBaseline) {
            if (!spectral_curve_ok(p.baseMin) || !spectral_curve_ok(p.baseMid)) {
                return false;
            }
        }
        return true;
    }
    //HIDDEN MESSAGE - I NEVER READ THE CODE
    inline void load_profile_from_dir(const std::string& dir, Profile& out, const std::string& jsonProfilePath = std::string()) {
        using Spectral::load_csv_pairs;
        using Spectral::build_curve_on_shape_from_linear_pairs;
        using Spectral::build_curve_on_shape_from_log10_pairs;

        auto load_pairs_silent = [](const std::string& path) {
            try { return Spectral::load_csv_pairs(path); }
            catch (...) { return std::vector<std::pair<float, float>>{}; }
            };

        // Define helper BEFORE any calls
        auto pad_to_shape_domain = [](std::vector<std::pair<float, float>>& pairs) {
            if (pairs.empty() || Spectral::gShape.wavelengths.empty()) return;
            const float Lmin = Spectral::gShape.lambdaMin;
            const float Lmax = Spectral::gShape.lambdaMax;

            std::sort(pairs.begin(), pairs.end(),
                [](auto& a, auto& b) { return a.first < b.first; });

            const float firstL = pairs.front().first;
            const float lastL = pairs.back().first;
            const float firstV = pairs.front().second;
            const float lastV = pairs.back().second;

            if (firstL > Lmin) {
                pairs.insert(pairs.begin(), { Lmin, firstV });
            }
            if (lastL < Lmax) {
                pairs.emplace_back(Lmax, lastV);
            }
            };

        // Now load EPS
        auto c_eps = load_pairs_silent(dir + "dye_density_c.csv");
        auto m_eps = load_pairs_silent(dir + "dye_density_m.csv");
        auto y_eps = load_pairs_silent(dir + "dye_density_y.csv");

        // Load sensitivities
        auto b_sens = load_pairs_silent(dir + "log_sensitivity_b.csv"); // Blue -> Yellow
        auto g_sens = load_pairs_silent(dir + "log_sensitivity_g.csv"); // Green -> Magenta
        auto r_sens = load_pairs_silent(dir + "log_sensitivity_r.csv"); // Red   -> Cyan

        // Pad all to shape domain
        pad_to_shape_domain(c_eps);
        pad_to_shape_domain(m_eps);
        pad_to_shape_domain(y_eps);
        pad_to_shape_domain(b_sens);
        pad_to_shape_domain(g_sens);
        pad_to_shape_domain(r_sens);

        // Build curves
        build_curve_on_shape_from_linear_pairs(out.epsC, c_eps);
        build_curve_on_shape_from_linear_pairs(out.epsM, m_eps);
        build_curve_on_shape_from_linear_pairs(out.epsY, y_eps);

        build_curve_on_shape_from_log10_pairs(out.sensY_log, b_sens);
        build_curve_on_shape_from_log10_pairs(out.sensM_log, g_sens);
        build_curve_on_shape_from_log10_pairs(out.sensC_log, r_sens);

        // --- Diagnostic logging to juicer_trace.txt ---
        {           

            auto fmt_pairs = [](const std::vector<std::pair<float, float>>& pairs) {
                std::ostringstream s;
                s << pairs.size();
                if (!pairs.empty()) {
                    float xmin = pairs.front().first;
                    float xmax = pairs.front().first;
                    float ymin = pairs.front().second;
                    float ymax = pairs.front().second;
                    for (size_t i = 1; i < pairs.size(); ++i) {
                        xmin = std::min(xmin, pairs[i].first);
                        xmax = std::max(xmax, pairs[i].first);
                        ymin = std::min(ymin, pairs[i].second);
                        ymax = std::max(ymax, pairs[i].second);
                    }
                    s << " [x:" << xmin << "..." << xmax
                        << " y:" << ymin << "..." << ymax << "]";
                }
                return s.str();
                };

            std::ostringstream oss;
            oss << "PROFILE_LOAD begin"
                << " shape.K=" << Spectral::gShape.K
                << " lambdaMin=" << Spectral::gShape.lambdaMin
                << " lambdaMax=" << Spectral::gShape.lambdaMax
                << " dir='" << dir << "'";
            JuicerTrace::write("PRINT", oss.str());

            // Raw EPS files (un-padded) sizes
            {
                std::ostringstream s;
                s << "EPS raw sizes C=" << fmt_pairs(c_eps)
                    << " M=" << fmt_pairs(m_eps)
                    << " Y=" << fmt_pairs(y_eps);
                JuicerTrace::write("PRINT", s.str());
            }
        }
        // --- End diagnostic logging ---

        // density curves are linear D vs logE (not log10 D)
        // agx-emulsion convention for PRINT stocks: files are B/G/R; map B→Y, G→M, R→C

        // Declare CMY in-memory curves (double precision)
        std::vector<std::pair<double, double>> c_dc;
        std::vector<std::pair<double, double>> m_dc;
        std::vector<std::pair<double, double>> y_dc;

        // helper: promote float->double
        auto promote_pairs = [](const std::vector<std::pair<float, float>>& in) {
            std::vector<std::pair<double, double>> out;
            out.reserve(in.size());
            for (auto& p : in) out.emplace_back(static_cast<double>(p.first),
                static_cast<double>(p.second));
            return out;
            };

        // helper: demote double->float
        auto demote_pairs = [](const std::vector<std::pair<double, double>>& in) {
            std::vector<std::pair<float, float>> out;
            out.reserve(in.size());
            for (auto& p : in) out.emplace_back(static_cast<float>(p.first),
                static_cast<float>(p.second));
            return out;
            };

        // Canonical agx-emulsion print stock: BGR on disk
        auto b_dc = load_pairs_silent(dir + "density_curve_b.csv"); // Blue -> Yellow
        auto g_dc = load_pairs_silent(dir + "density_curve_g.csv"); // Green -> Magenta
        auto r_dc = load_pairs_silent(dir + "density_curve_r.csv"); // Red   -> Cyan

        if (!r_dc.empty()) c_dc = promote_pairs(r_dc); // R -> C
        if (!g_dc.empty()) m_dc = promote_pairs(g_dc); // G -> M
        if (!b_dc.empty()) y_dc = promote_pairs(b_dc); // B -> Y

        // Build sorted curves into profile (convert to float for sort_and_build)
        if (!c_dc.empty()) Spectral::sort_and_build(out.dcC, demote_pairs(c_dc));
        if (!m_dc.empty()) Spectral::sort_and_build(out.dcM, demote_pairs(m_dc));
        if (!y_dc.empty()) Spectral::sort_and_build(out.dcY, demote_pairs(y_dc));

        // Monotonic sanitization (non-decreasing) and strict deduplication in logE domain for print dc curves
        auto sanitize_curve = [](Spectral::Curve& c) {
            float prev = (c.linear.empty() || !std::isfinite(c.linear[0])) ? 0.0f : c.linear[0];
            for (size_t i = 0; i < c.linear.size(); ++i) {
                float cur = c.linear[i];
                if (!std::isfinite(cur) || cur < 0.0f) cur = 0.0f;
                if (cur < prev) cur = prev;
                c.linear[i] = cur;
                prev = cur;
            }
            };
        auto dedup_strict_curve = [](Spectral::Curve& c) {
            if (c.lambda_nm.size() != c.linear.size() || c.lambda_nm.empty()) return;
            std::vector<float> X = c.lambda_nm;
            std::vector<float> Y = c.linear;
            std::vector<float> X2; X2.reserve(X.size());
            std::vector<float> Y2; Y2.reserve(Y.size());
            float lastX = X[0];
            float lastY = std::isfinite(Y[0]) ? std::max(0.0f, Y[0]) : 0.0f;
            X2.push_back(lastX);
            Y2.push_back(lastY);
            const float eps = 1e-6f;
            for (size_t i = 1; i < X.size(); ++i) {
                float xi = X[i];
                float yi = std::isfinite(Y[i]) ? std::max(0.0f, Y[i]) : 0.0f;
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
                c.lambda_nm = std::move(X2);
                c.linear = std::move(Y2);
            }
            };

        sanitize_curve(out.dcY);
        sanitize_curve(out.dcM);
        sanitize_curve(out.dcC);
        dedup_strict_curve(out.dcY);
        dedup_strict_curve(out.dcM);
        dedup_strict_curve(out.dcC);

        std::vector<std::pair<float, float>> bmin, bmid;
        try {
            bmin = Spectral::load_csv_pairs(dir + "dye_density_min.csv");
        }
        catch (...) {
            bmin.clear(); // missing is OK
        }
        try {
            bmid = Spectral::load_csv_pairs(dir + "dye_density_mid.csv");
        }
        catch (...) {
            bmid.clear(); // missing is OK
        }

        std::vector<std::pair<float, float>> jsonBaseMin;
        std::vector<std::pair<float, float>> jsonBaseMid;
        float baselineScale = 1.0f;
        if (!jsonProfilePath.empty()) {
            Profiles::AgxFilmProfile profileJson;
            if (Profiles::load_agx_film_profile_json(jsonProfilePath, profileJson)) {
                baselineScale = std::isfinite(profileJson.dyeDensityMinFactor)
                    ? profileJson.dyeDensityMinFactor
                    : 1.0f;
                jsonBaseMin = std::move(profileJson.baseMin);
                jsonBaseMid = std::move(profileJson.baseMid);
            }
        }

        auto baselineMinPairs = std::move(bmin);
        auto baselineMidPairs = std::move(bmid);
        if (baselineMinPairs.empty() && !jsonBaseMin.empty()) {
            baselineMinPairs = std::move(jsonBaseMin);
        }
        if (baselineMidPairs.empty() && !jsonBaseMid.empty()) {
            baselineMidPairs = std::move(jsonBaseMid);
        }
        if (!baselineMinPairs.empty()) pad_to_shape_domain(baselineMinPairs);
        if (!baselineMidPairs.empty()) pad_to_shape_domain(baselineMidPairs);


        build_curve_on_shape_from_linear_pairs(out.epsC, c_eps);
        build_curve_on_shape_from_linear_pairs(out.epsM, m_eps);
        build_curve_on_shape_from_linear_pairs(out.epsY, y_eps);

        if (!baselineMinPairs.empty() && !baselineMidPairs.empty()) {
            if (baselineScale != 1.0f) {
                for (auto& sample : baselineMinPairs) {
                    if (std::isfinite(sample.second)) {
                        sample.second *= baselineScale;
                    }
                }
            }
            build_curve_on_shape_from_linear_pairs(out.baseMin, baselineMinPairs);
            build_curve_on_shape_from_linear_pairs(out.baseMid, baselineMidPairs);
            out.hasBaseline = true;
        }
        else {
            out.baseMin.lambda_nm.clear();
            out.baseMin.linear.clear();
            out.baseMid.lambda_nm.clear();
            out.baseMid.linear.clear();
            out.hasBaseline = false;
        }

        // Preserve log-exposure offsets authored in the profile (agx parity).
        out.logEOffC = 0.0f;
        out.logEOffM = 0.0f;
        out.logEOffY = 0.0f;
    }

    // Build an illuminant pinned to shape from choice. Choices align with your UI (0:D65,1:D50,2:TH-KG3-L,3:Equal)
    inline void build_illuminant_from_choice(int choice, Runtime& rt, const std::string& dataDir, bool forEnlarger) {
        // Build pinned curve locally without touching Spectral globals
        Spectral::Curve c;
        if (choice == 0) {
            c = Spectral::build_curve_D65_pinned(dataDir + "Illuminants\\D65.csv");
        }
        else if (choice == 1) {
            c = Spectral::build_curve_D50_pinned(dataDir + "Illuminants\\D50.csv");
        }
        else if (choice == 2) {
            try {
                c = Spectral::build_curve_TH_KG3_L_pinned(
                    dataDir + "Filter\\KG3.csv",
                    dataDir + "Lens\\lens.csv");
            }
            catch (...) {
                // Fallback to equal-energy if any exception slips through
                c = Spectral::build_curve_equal_energy_pinned();
            }
        }

        else {
            c = Spectral::build_curve_equal_energy_pinned();
        }

        if (forEnlarger) {
            rt.illumEnlarger = std::move(c);
        }
        else {
            rt.illumView = std::move(c);
        }
    }

    inline void load_dichroic_filters_from_csvs(
        const std::string& dirYMC, Runtime& rt)
    {
        auto load_pairs_silent = [](const std::string& path) {
            try { return Spectral::load_csv_pairs(path); }
            catch (...) { return std::vector<std::pair<float, float>>{}; }
            };        

        // Try Durst Digital Light first
        std::string yPath = dirYMC + "filter_y.csv";
        std::string mPath = dirYMC + "filter_m.csv";
        std::string cPath = dirYMC + "filter_c.csv";

        auto y_pairs = load_pairs_silent(yPath);
        auto m_pairs = load_pairs_silent(mPath);
        auto c_pairs = load_pairs_silent(cPath);

        // If not found, try Edmund Optics / Thorlabs fallbacks (same filenames under their dirs)
        if (y_pairs.empty() || m_pairs.empty() || c_pairs.empty()) {
            std::string alt1 = dirYMC; // allow caller to pass different vendor dirs if desired
            // Identity fallback will be used below if still empty.
        }        

        rt.filterY.lambda_nm = Spectral::gShape.wavelengths;
        rt.filterM.lambda_nm = Spectral::gShape.wavelengths;
        rt.filterC.lambda_nm = Spectral::gShape.wavelengths;

        rt.filterY.linear.resize(Spectral::gShape.K);
        rt.filterM.linear.resize(Spectral::gShape.K);
        rt.filterC.linear.resize(Spectral::gShape.K);

        auto sample_curve = [](const std::vector<std::pair<float, float>>& pairs,
            Spectral::Curve& dst) {
                if (Spectral::gShape.K <= 0) {
                    dst.linear.clear();
                    dst.lambda_nm.clear();
                    return;
                }
                dst.linear.assign((size_t)Spectral::gShape.K, 1.0f);
                if (pairs.size() < 2) {
                    return;
                }
                std::vector<float> x, y;
                x.reserve(pairs.size());
                y.reserve(pairs.size());
                for (const auto& p : pairs) {
                    if (!std::isfinite(p.first) || !std::isfinite(p.second)) continue;
                    x.push_back(p.first);
                    y.push_back(p.second * (1.0f / 100.0f));
                }
                if (x.size() < 2) {
                    return;
                }
                Interpolation::AkimaInterpolator interp;
                if (!interp.build(x, y)) {
                    return;
                }
                for (int i = 0; i < Spectral::gShape.K; ++i) {
                    const float wl = Spectral::gShape.wavelengths[i];
                    const float v = interp.evaluate(wl);
                    dst.linear[(size_t)i] = std::isfinite(v) ? v : 1.0f;
                }
            };

        sample_curve(y_pairs, rt.filterY);
        sample_curve(m_pairs, rt.filterM);
        sample_curve(c_pairs, rt.filterC);

        // Diagnostics
        {
            std::ostringstream oss;
            oss << "DICHROICS loaded K=" << Spectral::gShape.K
                << " Y/M/C first="
                << (rt.filterY.linear.empty() ? -1.0f : rt.filterY.linear.front()) << "/"
                << (rt.filterM.linear.empty() ? -1.0f : rt.filterM.linear.front()) << "/"
                << (rt.filterC.linear.empty() ? -1.0f : rt.filterC.linear.front());
            JuicerTrace::write("PRINT", oss.str());
        }
    }



    // Compute negative transmittance T_neg(λ) from over-B+F densities D_neg using per-instance data
    inline void negative_T_from_dyes(const WorkingState& ws,
        const float D_neg[3],
        std::vector<float>& Tneg_out)
    {
        // Use the instance's spectral length; do NOT gate off gShape here.
        const int K = ws.tablesView.K;
        Tneg_out.assign((size_t)std::max(K, 0), 1.0f);
        if (K <= 0) return;

        // Per-instance baseline (if available)
        const bool hasBL = ws.hasBaseline &&
            (int)ws.baseMin.linear.size() == K;

        // Access per-wavelength epsilon for Y/M/C negative dyes; lambdas fallback to 0 if missing.
        auto epsY_at = [&](int i)->float {
            return (i < (int)ws.tablesView.epsY.size()) ? ws.tablesView.epsY[i] : 0.0f;
            };
        auto epsM_at = [&](int i)->float {
            return (i < (int)ws.tablesView.epsM.size()) ? ws.tablesView.epsM[i] : 0.0f;
            };
        auto epsC_at = [&](int i)->float {
            return (i < (int)ws.tablesView.epsC.size()) ? ws.tablesView.epsC[i] : 0.0f;
            };

        for (int i = 0; i < K; ++i) {
            const float baseSpectral = hasBL
                ? ws.baseMin.linear[i]
                : 0.0f;

            const float Dlambda = D_neg[0] * epsY_at(i)
                + D_neg[1] * epsM_at(i)
                + D_neg[2] * epsC_at(i)
                + baseSpectral;

            Tneg_out[i] = std::exp(-Spectral::kLn10 * Dlambda);
        }
    }


    // MVP: derive print channel exposures E_print[3] from Ee_expose(λ) using print dye extinctions as proxies
    inline void exposures_for_print_channels(const Runtime& rt,
        const std::vector<float>& Ee_expose,
        float Eprint[3])
    {
        const int K = Spectral::gShape.K;
        double Ec = 0.0, Em = 0.0, Ey = 0.0;

        for (int i = 0; i < K; ++i) {
            const float e = Ee_expose[i];
            const float lc = rt.profile.epsC.linear.empty() ? 0.0f : rt.profile.epsC.linear[i];
            const float lm = rt.profile.epsM.linear.empty() ? 0.0f : rt.profile.epsM.linear[i];
            const float ly = rt.profile.epsY.linear.empty() ? 0.0f : rt.profile.epsY.linear[i];

            Ec += static_cast<double>(e) * static_cast<double>(lc);
            Em += static_cast<double>(e) * static_cast<double>(lm);
            Ey += static_cast<double>(e) * static_cast<double>(ly);
        }

        Eprint[0] = std::max(0.0f, static_cast<float>(Ey)); // Y
        Eprint[1] = std::max(0.0f, static_cast<float>(Em)); // M
        Eprint[2] = std::max(0.0f, static_cast<float>(Ec)); // C
    }

    // Compute per-channel raw exposures via spectral sensitivity contraction:
    inline void raw_exposures_from_filtered_light(
        const Profile& p,
        const std::vector<float>& Ee_filtered,
        float raw[3])
    {
        // Convert log10 sensitivities to linear sensitivity per wavelength
        const int K = Spectral::gShape.K;
        double rY = 0.0, rM = 0.0, rC = 0.0;
        for (int i = 0; i < K; ++i) {
            const float sY = p.sensY_log.linear.empty() ? 0.0f : std::pow(10.0f, p.sensY_log.linear[i]);
            const float sM = p.sensM_log.linear.empty() ? 0.0f : std::pow(10.0f, p.sensM_log.linear[i]);
            const float sC = p.sensC_log.linear.empty() ? 0.0f : std::pow(10.0f, p.sensC_log.linear[i]);

            const float e = Ee_filtered[i];
            rY += static_cast<double>(e) * static_cast<double>(sY);
            rM += static_cast<double>(e) * static_cast<double>(sM);
            rC += static_cast<double>(e) * static_cast<double>(sC);
        }

        raw[0] = std::max(0.0f, static_cast<float>(rY)); // Y
        raw[1] = std::max(0.0f, static_cast<float>(rM)); // M
        raw[2] = std::max(0.0f, static_cast<float>(rC)); // C
    }

    inline void compute_preflash_raw(
        const Runtime& rt,
        const WorkingState& ws,
        std::vector<float>& Tpre,
        std::vector<float>& Ee_pre,
        float rawOut[3])
    {
        rawOut[0] = rawOut[1] = rawOut[2] = 0.0f;
        const int K = Spectral::gShape.K;
        if (K <= 0) return;
        if (ws.tablesView.K <= 0) return;

        const float Dbase[3] = { 0.0f, 0.0f, 0.0f };
        negative_T_from_dyes(ws, Dbase, Tpre);
        if ((int)Tpre.size() < K) {
            Tpre.resize((size_t)K, 1.0f);
        }

        Ee_pre.resize((size_t)K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, 0.0f);
        const float mAmount = compose_dichroic_amount(rt.neutralM, 0.0f);
        const float cAmount = compose_dichroic_amount(rt.neutralC, 0.0f);

        for (int i = 0; i < K; ++i) {
            const float Ee = (rt.illumEnlarger.linear.size() > size_t(i))
                ? rt.illumEnlarger.linear[i]
                : 1.0f;
            const float t = Tpre[i];
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_pre[i] = std::max(0.0f, Ee * t * fTotal);
        }

        raw_exposures_from_filtered_light(rt.profile, Ee_pre, rawOut);
    }

    // Midgray compensation factor computed spectrally (AgX parity): always normalize RAW so a neutral mid-gray is
    // mapped to unity exposure. When the print exposure compensation toggle is enabled the camera exposure EV scale
    // is injected, otherwise the EV term is ignored. This uses the same negative development, enlarger illuminant
    // (including dichroic filters), and print paper sensitivities as the print leg.
    inline float compute_exposure_factor_midgray(
        const WorkingState& ws,
        const Runtime& rt,
        const Params& prm,
        float exposureCompScale)
    {
        // If slider EV scale is not meaningful, skip compensation.
        if (!std::isfinite(exposureCompScale) || exposureCompScale <= 0.0f) return 1.0f;

        // 1) Midgray DWG rgb at canonical brightness (AgX parity: constant 18.4% reflectance)
        const float rgbMid[3] = { 0.184f, 0.184f, 0.184f };

        // 2) DWG → per-layer exposures (negative leg); apply camera EV exactly once here.
        //    NOTE: Do not pre-scale rgbMid by cameraExposureScale — avoids double-applying EV.
        float E[3];
        const SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
        Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbMid, E, 1.0f,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);
        const float exposureScale = prm.exposureCompensationEnabled ? exposureCompScale : 1.0f;
        E[0] *= exposureScale; E[1] *= exposureScale; E[2] *= exposureScale;

        // 3) LogE with per-layer offsets, clamp to domain, sample negative densities
        auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };
        const float logE[3] = {
            std::log10(std::max(0.0f, E[0]) + 1e-10f) + ws.logEOffB,
            std::log10(std::max(0.0f, E[1]) + 1e-10f) + ws.logEOffG,
            std::log10(std::max(0.0f, E[2]) + 1e-10f) + ws.logEOffR
        };
        float logE_clamped[3] = {
            clamp_logE_to_curve(ws.densB, logE[0]),
            clamp_logE_to_curve(ws.densG, logE[1]),
            clamp_logE_to_curve(ws.densR, logE[2]),
        };
        float D_neg[3] = {
            Spectral::sample_density_at_logE(ws.densB, logE_clamped[0]),
            Spectral::sample_density_at_logE(ws.densG, logE_clamped[1]),
            Spectral::sample_density_at_logE(ws.densR, logE_clamped[2]),
        };

        // 4) Negative transmittance (baseline applied per stock)        
        thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered;
        negative_T_from_dyes(ws, D_neg, Tneg);

        // 5) Enlarger illuminant exposure
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // 6) Apply Y/M/C dichroic filters using Durst wheel linear blending (agx parity)
        Ee_filtered.resize(Spectral::gShape.K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, prm.yFilter);
        const float mAmount = compose_dichroic_amount(rt.neutralM, prm.mFilter);
        const float cAmount = compose_dichroic_amount(rt.neutralC, prm.cFilter);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // 7) RAW via print paper sensitivities (log domain → linear sensitivity)
        float raw[3];
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw);

        // 8) Factor is inverse of midgray green RAW; guard for tiny values
        const float g = std::max(1e-12f, raw[1]);
        return 1.0f / g;
    }



    inline float sample_dc(const Spectral::Curve& dc, float logE) {
        return Spectral::sample_density_at_logE(dc, logE);
    }

    // Build print densities from print exposures (parity with agx-emulsion)
    inline void print_densities_from_Eprint(const Profile& p, const float Eprint[3], float D_print[3]) {
        auto safe_log10 = [](float v)->float { return std::log10(std::max(0.0f, v) + 1e-10f); };
        float lEy = safe_log10(Eprint[0]);
        float lEm = safe_log10(Eprint[1]);
        float lEc = safe_log10(Eprint[2]);

        // Clamp to density curve logE domain to prevent sampling off-curve (parity with agx guardrails)
        auto clamp_to_dc = [](float le, const Spectral::Curve& dc)->float {
            if (dc.lambda_nm.empty()) return le;
            const float xmin = dc.lambda_nm.front();
            const float xmax = dc.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };
        lEy = clamp_to_dc(lEy, p.dcY);
        lEm = clamp_to_dc(lEm, p.dcM);
        lEc = clamp_to_dc(lEc, p.dcC);

        D_print[0] = sample_dc(p.dcY, lEy);
        D_print[1] = sample_dc(p.dcM, lEm);
        D_print[2] = sample_dc(p.dcC, lEc);

        // Densities are optical densities (OD); clamp to non-negative to prevent T>1.
        D_print[0] = std::max(0.0f, D_print[0]);
        D_print[1] = std::max(0.0f, D_print[1]);
        D_print[2] = std::max(0.0f, D_print[2]);

    }    


    inline void print_T_from_dyes(const Profile& p, const float D_print[3], std::vector<float>& Tprint_out) {
        const int K = Spectral::gShape.K;
        Tprint_out.resize(K);
        for (int i = 0; i < K; ++i) {
            const float baseSpectral = p.hasBaseline
                ? p.baseMin.linear[i]
                : 0.0f;
            const float Dlambda = D_print[0] * p.epsY.linear[i]
                + D_print[1] * p.epsM.linear[i]
                + D_print[2] * p.epsC.linear[i]
                + baseSpectral;
            Tprint_out[i] = std::exp(-Spectral::kLn10 * Dlambda);
        }
    }
    // >>> BEGIN INSERT: test-only probe helpers (Print.h) REMOVE AFTER USE
#ifdef JUICER_TESTS

    struct Probe {
        std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
        float raw[3]{ 0,0,0 };
        float D_print[3]{ 0,0,0 };
        float XYZ[3]{ 0,0,0 };
    };

    // Start from *negative* logE (already camera-exposed), then run the *print* leg.
    // This isolates the print path (filters → raw → DC curves → Tprint → viewing → XYZ → DWG).
    inline void simulate_print_from_logE_for_test(
        const float logE_neg[3],
        const Params& prm,
        const Runtime& rt,
        const Couplers::Runtime& /*dirRT*/,   // DIR is applied on negative leg in main path; off here to isolate print
        const WorkingState& ws,
        float exposureScale,              // not used when starting at logE
        float rgbOut[3],
        Probe* probe /*nullable*/)
    {
        // 1) Sample negative densities from per-instance negative curves.
        auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };

        float leB = clamp_logE_to_curve(ws.densB, logE_neg[0]);
        float leG = clamp_logE_to_curve(ws.densG, logE_neg[1]);
        float leR = clamp_logE_to_curve(ws.densR, logE_neg[2]);

        float D_neg[3] = {
            Spectral::sample_density_at_logE(ws.densB, leB), // Y from B-layer curve
            Spectral::sample_density_at_logE(ws.densG, leG), // M from G-layer curve
            Spectral::sample_density_at_logE(ws.densR, leR)  // C from R-layer curve
        };

        // 2) Negative transmittance with per-stock baseline        
        thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
        thread_local std::vector<float> Tpreflash, Ee_preflash;
        negative_T_from_dyes(ws, D_neg, Tneg);

        // 3) Enlarger exposure (neutral illuminant here; print exposure happens later on RAW)
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // 4) Apply neutral Y/M/C dichroic filters to enlarger light
        Ee_filtered.resize(Spectral::gShape.K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, prm.yFilter);
        const float mAmount = compose_dichroic_amount(rt.neutralM, prm.mFilter);
        const float cAmount = compose_dichroic_amount(rt.neutralC, prm.cFilter);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // 5) Contract to channel RAW using print sensitometries
        float raw[3] = { 0,0,0 };
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw);

        // Apply print exposure (agx parity)
        const float expPrint = std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;
        const float exposureCompScale = prm.exposureCompensationEnabled
            ? prm.exposureCompensationScale
            : 1.0f;
        const float kMid = compute_exposure_factor_midgray(ws, rt, prm, exposureCompScale);
        const float rawScale = expPrint * kMid;
        raw[0] *= rawScale; raw[1] *= rawScale; raw[2] *= rawScale;

        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
            float rawPre[3];
            compute_preflash_raw(rt, ws, Tpreflash, Ee_preflash, rawPre);
            raw[0] += rawPre[0] * prm.preflashExposure;
            raw[1] += rawPre[1] * prm.preflashExposure;
            raw[2] += rawPre[2] * prm.preflashExposure;
        }

        // 6) RAW → print densities via DC curves and per-channel logE offsets
        float D_print[3] = { 0,0,0 };
        print_densities_from_Eprint(rt.profile, raw, D_print);

        // 7) Print transmittance
        print_T_from_dyes(rt.profile, D_print, Tprint);

        // 8) Viewing illuminant and integration to XYZ
        Ee_viewed.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ev = rt.illumView.linear.empty() ? 1.0f : rt.illumView.linear[i];
            Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
        }

        float XYZ[3] = { 0,0,0 };
        Spectral::Ee_to_XYZ_given_tables(ws.tablesView, Ee_viewed, XYZ);

        // XYZ → DWG with chromatic adaptation
        Spectral::XYZ_to_DWG_linear_adapted(ws.tablesView, XYZ, rgbOut);
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);

        if (probe) {
            probe->Tneg = Tneg;
            probe->Ee_expose = Ee_expose;
            probe->Ee_filtered = Ee_filtered;
            probe->Tprint = Tprint;
            probe->Ee_viewed = Ee_viewed;
            probe->raw[0] = raw[0]; probe->raw[1] = raw[1]; probe->raw[2] = raw[2];
            probe->D_print[0] = D_print[0]; probe->D_print[1] = D_print[1]; probe->D_print[2] = D_print[2];
            probe->XYZ[0] = XYZ[0]; probe->XYZ[1] = XYZ[1]; probe->XYZ[2] = XYZ[2];
        }
    }

#endif // JUICER_TESTS
// <<< END INSERT REMOVE AFTER USE


    // Full pixel pipeline when print is active
    inline void simulate_print_pixel(const float rgbIn[3],
        const Params& prm,
        const Runtime& rt,
        const Couplers::Runtime& dirRT,
        const WorkingState& ws,
        float exposureScale,
        float rgbOut[3])
    {
        // 1) Negative densities with DIR in logE domain (per-instance SPD vs Matrix)
        float E[3];
        const SpectralTables* tablesSPD =
            (ws.spdReady && ws.tablesRef.K > 0) ? &ws.tablesRef : nullptr;
        Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbIn, E, 1.0f,
            tablesSPD,
            (ws.spdReady ? ws.spdSInv : nullptr),
            ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        // Apply camera exposure explicitly on negative leg to ensure parity and avoid hidden double-handling.
        {
            const float sExp = (std::isfinite(exposureScale) ? std::max(0.0f, exposureScale) : 1.0f);
            if (sExp != 1.0f) { E[0] *= sExp; E[1] *= sExp; E[2] *= sExp; }
        }



        float logE[3] = {
            std::log10(std::max(0.0f, E[0]) + 1e-10f) + ws.logEOffB,
            std::log10(std::max(0.0f, E[1]) + 1e-10f) + ws.logEOffG,
            std::log10(std::max(0.0f, E[2]) + 1e-10f) + ws.logEOffR
        };

        // Domain clamp helper before sample_ws usage
        auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = c.lambda_nm.front();
            const float xmax = c.lambda_nm.back();
            if (!std::isfinite(le)) return xmin;
            return std::min(std::max(le, xmin), xmax);
            };

        auto sample_ws = [&](const float le[3], float D_out[3]) {
            D_out[0] = Spectral::sample_density_at_logE(ws.densB, le[0]); // Y
            D_out[1] = Spectral::sample_density_at_logE(ws.densG, le[1]); // M
            D_out[2] = Spectral::sample_density_at_logE(ws.densR, le[2]); // C
            };

        float D_neg[3];
        float logE_clamped[3] = {
            clamp_logE_to_curve(ws.densB, logE[0]),
            clamp_logE_to_curve(ws.densG, logE[1]),
            clamp_logE_to_curve(ws.densR, logE[2]),
                };
                sample_ws(logE_clamped, D_neg);

#ifdef JUICER_ENABLE_COUPLERS
                // Always apply local DIR corrections when active (agx parity), even if curves were
                // precorrected and the spatial diffusion sigma is zero.
                if (dirRT.active) {
                    Couplers::ApplyInputLogE io{ {logE[0], logE[1], logE[2]}, {D_neg[0], D_neg[1], D_neg[2]} };
                    // Per-instance clamp variant: aligns clamp domain to the same curves we sample
                    Couplers::apply_runtime_logE_with_curves(io, dirRT, ws.densB, ws.densG, ws.densR);
                    float logE2_clamped[3] = { io.logE[0], io.logE[1], io.logE[2] };
                    sample_ws(logE2_clamped, D_neg);
                }
#endif

        // 2) Negative transmittance with optional baseline blend        
        thread_local std::vector<float> Tneg, Ee_expose, Tprint, Ee_viewed;
        thread_local std::vector<float> Tpreflash, Ee_preflash;
        negative_T_from_dyes(ws, D_neg, Tneg);

        // 3) Ee_expose = Ee_enlarger * T_neg * exposure
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // 4) Reduce to per-channel print exposures and apply neutral Y/M/C filters
        // Apply spectral Y/M/C dichroic filters to enlarger light before per-channel reduction
        thread_local std::vector<float> Ee_filtered;
        Ee_filtered.resize(Spectral::gShape.K);
        const float yAmount = compose_dichroic_amount(rt.neutralY, prm.yFilter);
        const float mAmount = compose_dichroic_amount(rt.neutralM, prm.mFilter);
        const float cAmount = compose_dichroic_amount(rt.neutralC, prm.cFilter);
        // Blend each wheel between identity (1.0) and its full transmittance curve
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_dichroic_filter_linear(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], yAmount);
            const float fM = blend_dichroic_filter_linear(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], mAmount);
            const float fC = blend_dichroic_filter_linear(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], cAmount);
            const float fTotal = fY * fM * fC;
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // Compute per-channel raw via sensitivity contraction (agx parity)
        float raw[3];
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw);

        // Apply print exposure scaling to raw (agx: raw *= print_exposure)
        const float expPrint = std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;        

        // Spectral midgray compensation (AgX parity): scale RAW vector by factor = 1 / RAW_midgray_green.
        // Compute once per call, independent of pixel content.
        const float exposureCompScale = prm.exposureCompensationEnabled
            ? prm.exposureCompensationScale
            : 1.0f;
        const float kMid = compute_exposure_factor_midgray(ws, rt, prm, exposureCompScale);
        const float rawScale = expPrint * kMid;
        raw[0] *= rawScale; raw[1] *= rawScale; raw[2] *= rawScale;

        if (std::isfinite(prm.preflashExposure) && prm.preflashExposure > 0.0f) {
            float rawPre[3];
            compute_preflash_raw(rt, ws, Tpreflash, Ee_preflash, rawPre);
            raw[0] += rawPre[0] * prm.preflashExposure;
            raw[1] += rawPre[1] * prm.preflashExposure;
            raw[2] += rawPre[2] * prm.preflashExposure;
        }


        // Map raw to print densities via per-channel logE offsets and DC curves
        float D_print[3];
        print_densities_from_Eprint(rt.profile, raw, D_print);

        // 6) Print transmittance
        print_T_from_dyes(rt.profile, D_print, Tprint);

        // 7) Multiply print transmittance by viewing illuminant to get Ee_viewed
        Ee_viewed.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ev = rt.illumView.linear.empty() ? 1.0f : rt.illumView.linear[i];
            Ee_viewed[i] = std::max(0.0f, Ev * Tprint[i]);
        }

        // Integrate to XYZ using per-instance viewing axis and normalization
        float XYZ[3];
        Spectral::Ee_to_XYZ_given_tables(ws.tablesView, Ee_viewed, XYZ);

        // XYZ -> DWG with chromatic adaptation
        Spectral::XYZ_to_DWG_linear_adapted(ws.tablesView, XYZ, rgbOut);
        rgbOut[0] = std::max(0.0f, rgbOut[0]);
        rgbOut[1] = std::max(0.0f, rgbOut[1]);
        rgbOut[2] = std::max(0.0f, rgbOut[2]);

    }

} // namespace Print
