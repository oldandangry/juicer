// Print.h
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include "SpectralTables.h"
#include "SpectralMath.h"
#include "Illuminants.h"
#include "WorkingState.h"
#include <sstream>
#include <algorithm>

namespace JuicerTrace {
    void write(const char* tag, const std::string& msg);
}

// Forward-declare WorkingState to avoid header cycles
struct WorkingState;

namespace Print {

    struct Profile {
        Spectral::Curve epsC, epsM, epsY;     // print dye extinction (OD/λ)
        Spectral::Curve dcC, dcM, dcY;        // logE -> D (C,M,Y)
        Spectral::Curve baseMin, baseMid;     // optional print baseline (D-min/mid)
        bool hasBaseline = false;
        bool glareRemoved = false;
        float logEOffC = 0.0f, logEOffM = 0.0f, logEOffY = 0.0f; // per-channel logE offsets

        // Print paper spectral log-sensitivity (B/G/R on disk => Y/M/C mapping), pinned to shape (log10 domain in CSVs)
        Spectral::Curve sensY_log, sensM_log, sensC_log;
    };


    struct Params {
        bool bypass = false;     // if true, bypass print (show negative)
        float exposure = 1.0f;   // enlarger exposure scalar
        float yFilter = 1.0f;    // neutral filter scalars (MVP)
        float mFilter = 1.0f;
        float cFilter = 1.0f;    // optional

        // Agx parity: print exposure compensation applies via green channel only.
        // This factor multiplies raw[1] (green channel) before log10 and development.
        float exposureCompGFactor = 1.0f;
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
    inline void load_profile_from_dir(const std::string& dir, Profile& out) {
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

        // Pad baselines to match the current spectral shape domain if present
        if (!bmin.empty()) pad_to_shape_domain(bmin);
        if (!bmid.empty()) pad_to_shape_domain(bmid);


        build_curve_on_shape_from_linear_pairs(out.epsC, c_eps);
        build_curve_on_shape_from_linear_pairs(out.epsM, m_eps);
        build_curve_on_shape_from_linear_pairs(out.epsY, y_eps);

        if (!bmin.empty() && !bmid.empty()) {
            build_curve_on_shape_from_linear_pairs(out.baseMin, bmin);
            build_curve_on_shape_from_linear_pairs(out.baseMid, bmid);
            out.hasBaseline = true;
        }
        else {
            out.baseMin.lambda_nm.clear();
            out.baseMin.linear.clear();
            out.baseMid.lambda_nm.clear();
            out.baseMid.linear.clear();
            out.hasBaseline = false;
        }

        // Calibrate print logE offsets to place mids (simple midpoint heuristic)
        auto find_mid = [](const Spectral::Curve& c)->float {
            if (c.lambda_nm.empty()) return 0.0f;
            float dmin = c.linear.front();
            float dmax = c.linear.back();
            float target = dmin + 0.5f * (dmax - dmin);
            // find x such that c(x)=target
            const size_t n = c.lambda_nm.size();
            for (size_t i = 1; i < n; ++i) {
                float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                float y0 = c.linear[i - 1], y1 = c.linear[i];
                if ((y0 <= target && target <= y1) || (y1 <= target && target <= y0)) {
                    float t = (y1 != y0) ? (target - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            return c.lambda_nm.front();
            };
        out.logEOffC = find_mid(out.dcC);
        out.logEOffM = find_mid(out.dcM);
        out.logEOffY = find_mid(out.dcY);
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

        auto normalize_pct_pairs_to_unit = [](std::vector<std::pair<float, float>>& pairs) {
            for (auto& p : pairs) {
                float v = p.second * (1.0f / 100.0f);
                if (!std::isfinite(v) || v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                p.second = v;
            }
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

        // Normalize percent to unit and pin to current shape
        if (!y_pairs.empty()) normalize_pct_pairs_to_unit(y_pairs);
        if (!m_pairs.empty()) normalize_pct_pairs_to_unit(m_pairs);
        if (!c_pairs.empty()) normalize_pct_pairs_to_unit(c_pairs);

        auto y_pinned = Spectral::resample_pairs_to_shape(y_pairs, Spectral::gShape);
        auto m_pinned = Spectral::resample_pairs_to_shape(m_pairs, Spectral::gShape);
        auto c_pinned = Spectral::resample_pairs_to_shape(c_pairs, Spectral::gShape);

        rt.filterY.lambda_nm = Spectral::gShape.wavelengths;
        rt.filterM.lambda_nm = Spectral::gShape.wavelengths;
        rt.filterC.lambda_nm = Spectral::gShape.wavelengths;

        rt.filterY.linear.resize(Spectral::gShape.K);
        rt.filterM.linear.resize(Spectral::gShape.K);
        rt.filterC.linear.resize(Spectral::gShape.K);

        for (int i = 0; i < Spectral::gShape.K; ++i) {
            float fy = (!y_pinned.empty()) ? std::max(0.0f, std::min(1.0f, y_pinned[i].second)) : 1.0f;
            float fm = (!m_pinned.empty()) ? std::max(0.0f, std::min(1.0f, m_pinned[i].second)) : 1.0f;
            float fc = (!c_pinned.empty()) ? std::max(0.0f, std::min(1.0f, c_pinned[i].second)) : 1.0f;
            rt.filterY.linear[i] = fy;
            rt.filterM.linear[i] = fm;
            rt.filterC.linear[i] = fc;
        }

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



    // Compute negative transmittance T_neg(λ) from over‑B+F densities D_neg using per‑instance data
    inline void negative_T_from_dyes(const WorkingState& ws,
        const float D_neg[3],
        std::vector<float>& Tneg_out,
        float neutralW = 0.0f)
    {
        const int K = Spectral::gShape.K;
        Tneg_out.assign((size_t)K, 1.0f);

        if (K <= 0) return;

        // Per‑instance baseline (if available)
        const bool hasBL = ws.hasBaseline &&
            (int)ws.baseMin.linear.size() == K &&
            (int)ws.baseMid.linear.size() == K;

        // Access per‑instance epsilon (negative dyes) from tablesView if present.
        // We defensively handle cases where the tables may not expose eps; in that case, we leave Tneg as 1.0.
        const bool haveTables = (ws.tablesView.K > 0) &&
            ((int)ws.tablesView.K == K);

        // Guard: if no tables, return neutral
        if (!haveTables) return;

        // Expected: ws.tablesView provides per‑wavelength epsilon for Y/M/C negative dyes
        // We access via helper lambdas that fallback to 0.0f if arrays are missing.
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
            const float dBase = hasBL
                ? (ws.baseMin.linear[i] + neutralW * (ws.baseMid.linear[i] - ws.baseMin.linear[i]))
                : 0.0f;

            const float Dy = (D_neg[0] + dBase);
            const float Dm = (D_neg[1] + dBase);
            const float Dc = (D_neg[2] + dBase);

            const float Dlambda = Dy * epsY_at(i)
                + Dm * epsM_at(i)
                + Dc * epsC_at(i);

            Tneg_out[i] = std::exp(-Spectral::kLn10 * Dlambda);
        }
    }


    // MVP: derive print channel exposures E_print[3] from Ee_expose(λ) using print dye extinctions as proxies
    inline void exposures_for_print_channels(const Runtime& rt,
        const std::vector<float>& Ee_expose,
        float Eprint[3],
        float deltaLambda)   
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

        const double dl = static_cast<double>(deltaLambda);

        Eprint[0] = std::max(0.0f, static_cast<float>(Ey * dl)); // Y
        Eprint[1] = std::max(0.0f, static_cast<float>(Em * dl)); // M
        Eprint[2] = std::max(0.0f, static_cast<float>(Ec * dl)); // C
    }

    // Compute per-channel raw exposures via spectral sensitivity contraction:
    inline void raw_exposures_from_filtered_light(
        const Profile& p,
        const std::vector<float>& Ee_filtered,
        float raw[3],
        float deltaLambda)
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
        const double dl = static_cast<double>(deltaLambda);
        raw[0] = std::max(0.0f, static_cast<float>(rY * dl)); // Y
        raw[1] = std::max(0.0f, static_cast<float>(rM * dl)); // M
        raw[2] = std::max(0.0f, static_cast<float>(rC * dl)); // C
    }


    inline float sample_dc(const Spectral::Curve& dc, float logE) {
        return Spectral::sample_density_at_logE(dc, logE);
    }

    // Build print densities from print exposures and per-channel offsets
    inline void print_densities_from_Eprint(const Profile& p, const float Eprint[3], float D_print[3]) {
        auto safe_log10 = [](float v)->float { return std::log10(std::max(1e-6f, v)); };
        float lEy = safe_log10(Eprint[0]) + p.logEOffY;
        float lEm = safe_log10(Eprint[1]) + p.logEOffM;
        float lEc = safe_log10(Eprint[2]) + p.logEOffC;

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

    // Sensitivity-balanced calibration of per-channel print logE offsets using viewing axis tables.
// Sets p.logEOffY/M/C = mid_logE_channel - log10(channel_sensitivity), where channel_sensitivity
// is the illuminant-weighted integral of the print dye extinction under the viewing CMFs.
    inline void calibrate_print_logE_offsets_from_tables(const SpectralTables& T, Profile& p) {
        const int K = T.K;
        if (K <= 0) return;

        // Require print dye extinction curves to be pinned to current shape.
        const bool epsOK =
            (int)p.epsY.linear.size() == K &&
            (int)p.epsM.linear.size() == K &&
            (int)p.epsC.linear.size() == K;

        // Require viewing axis components.
        const bool axisOK =
            (int)T.Ybar.size() == K &&
            (int)T.Xbar.size() == K &&
            (int)T.Zbar.size() == K;

        if (!epsOK || !axisOK) return;

        // Effective per-channel sensitivity under viewing axis (weight by Ybar; scale with Δλ).
        auto chan_gain = [&](const Spectral::Curve& eps)->double {
            double g = 0.0;
            for (int i = 0; i < K; ++i)
                g += static_cast<double>(eps.linear[i]) * static_cast<double>(T.Ybar[i]);
            g *= static_cast<double>(T.deltaLambda);
            return g;
            };

        const double sY = std::max(1e-12, chan_gain(p.epsY));
        const double sM = std::max(1e-12, chan_gain(p.epsM));
        const double sC = std::max(1e-12, chan_gain(p.epsC));

        // Mid-logE positions per channel (unchanged sampling domain, robust to monotone curves).
        auto find_mid = [](const Spectral::Curve& c)->float {
            if (c.lambda_nm.empty()) return 0.0f;
            float dmin = c.linear.front();
            float dmax = c.linear.back();
            float target = dmin + 0.5f * (dmax - dmin);
            const size_t n = c.lambda_nm.size();
            for (size_t i = 1; i < n; ++i) {
                float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                float y0 = c.linear[i - 1], y1 = c.linear[i];
                if ((y0 <= target && target <= y1) || (y1 <= target && target <= y0)) {
                    float t = (y1 != y0) ? (target - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            return c.lambda_nm.front();
            };

        const float leMidY = find_mid(p.dcY);
        const float leMidM = find_mid(p.dcM);
        const float leMidC = find_mid(p.dcC);

        // Sensitivity-balanced offsets
        p.logEOffY = leMidY - static_cast<float>(std::log10(sY));
        p.logEOffM = leMidM - static_cast<float>(std::log10(sM));
        p.logEOffC = leMidC - static_cast<float>(std::log10(sC));
    }

    // Calibrate per-channel print logE offsets using print paper sensitivity under viewing axis.
// Sets p.logEOffY/M/C = mid_logE_channel - log10(channel_sensitivity), where channel_sensitivity
// is the illuminant-weighted integral of the print paper sensitivity under the viewing CMFs.
    inline void calibrate_print_logE_offsets_from_profile(const SpectralTables& T, Profile& p) {
        const int K = T.K;
        if (K <= 0) return;

        // Require print paper log sensitivities to be pinned to current shape.
        const bool sensOK =
            (int)p.sensY_log.linear.size() == K &&
            (int)p.sensM_log.linear.size() == K &&
            (int)p.sensC_log.linear.size() == K;

        // Require viewing axis components from tables.
        const bool axisOK =
            (int)T.Ybar.size() == K &&
            (int)T.Xbar.size() == K &&
            (int)T.Zbar.size() == K;

        if (!sensOK || !axisOK) return;

        // Effective per-channel sensitivity under viewing Ybar; use 10^(sens_log) to get linear sensitivity.
        auto chan_gain_from_print = [&](const Spectral::Curve& sensLog)->double {
            double g = 0.0;
            for (int i = 0; i < K; ++i) {
                const double sLin = std::pow(10.0, (double)sensLog.linear[i]);
                g += sLin * (double)T.Ybar[i];
            }
            g *= (double)T.deltaLambda;
            return std::max(1e-12, g);
            };

        const double sY = chan_gain_from_print(p.sensY_log);
        const double sM = chan_gain_from_print(p.sensM_log);
        const double sC = chan_gain_from_print(p.sensC_log);

        // Mid-logE positions per channel (robust to monotone curves).
        auto find_mid = [](const Spectral::Curve& c)->float {
            if (c.lambda_nm.empty()) return 0.0f;
            const float dmin = c.linear.front();
            const float dmax = c.linear.back();
            const float target = dmin + 0.5f * (dmax - dmin);
            const size_t n = c.lambda_nm.size();
            for (size_t i = 1; i < n; ++i) {
                const float x0 = c.lambda_nm[i - 1], x1 = c.lambda_nm[i];
                const float y0 = c.linear[i - 1], y1 = c.linear[i];
                if ((y0 <= target && target <= y1) || (y1 <= target && target <= y0)) {
                    const float t = (y1 != y0) ? (target - y0) / (y1 - y0) : 0.0f;
                    return x0 + t * (x1 - x0);
                }
            }
            return c.lambda_nm.front();
            };

        const float leMidY = find_mid(p.dcY);
        const float leMidM = find_mid(p.dcM);
        const float leMidC = find_mid(p.dcC);

        // Sensitivity-balanced offsets (print domain).
        p.logEOffY = leMidY - static_cast<float>(std::log10(sY));
        p.logEOffM = leMidM - static_cast<float>(std::log10(sM));
        p.logEOffC = leMidC - static_cast<float>(std::log10(sC));
    }


    inline void print_T_from_dyes(const Profile& p, const float D_print[3], std::vector<float>& Tprint_out) {
        const int K = Spectral::gShape.K;
        Tprint_out.resize(K);
        for (int i = 0; i < K; ++i) {
            float dBase = p.hasBaseline
                ? (p.baseMin.linear[i]) // MVP: fixed baseline min; mid weighting can come later
                : 0.0f;
            const float Dlambda = (D_print[0] + dBase) * (p.epsY.linear[i])
                + (D_print[1] + dBase) * (p.epsM.linear[i])
                + (D_print[2] + dBase) * (p.epsC.linear[i]);
            Tprint_out[i] = std::exp(-Spectral::kLn10 * Dlambda);
        }
    }

    // >>> BEGIN INSERT: test-only probe helpers (Print.h) REMOVE AFTER USE
#ifdef JUICER_TESTS
    namespace Print {

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
            float /*exposureScale*/,              // not used when starting at logE
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

            // 2) Negative transmittance with baseline blend
            const float rgbInDummy[3] = { 0.5f, 0.5f, 0.5f }; // used only for neutral weight
            const float wNeutral = Spectral::neutral_blend_weight_from_DWG_rgb(rgbInDummy);
            thread_local std::vector<float> Tneg, Ee_expose, Ee_filtered, Tprint, Ee_viewed;
            negative_T_from_dyes(ws, D_neg, Tneg, wNeutral);

            // 3) Enlarger exposure (neutral illuminant here; print exposure happens later on RAW)
            Ee_expose.resize(Spectral::gShape.K);
            for (int i = 0; i < Spectral::gShape.K; ++i) {
                const float Ee = rt.illumEnlarger.linear.empty() ? 1.0f : rt.illumEnlarger.linear[i];
                Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
            }

            // 4) Apply neutral Y/M/C dichroic filters to enlarger light
            Ee_filtered.resize(Spectral::gShape.K);
            auto blend_filter = [](float curveVal, float amount)->float {
                const float a = std::isfinite(amount) ? std::clamp(amount, 0.0f, 8.0f) : 0.0f;
                const float c = std::isfinite(curveVal) ? std::clamp(curveVal, 1e-6f, 1.0f) : 1.0f;
                return std::pow(c, a);
                };
            for (int i = 0; i < Spectral::gShape.K; ++i) {
                const float fY = blend_filter(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], prm.yFilter);
                const float fM = blend_filter(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], prm.mFilter);
                const float fC = blend_filter(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], prm.cFilter);
                const float fTotal = std::max(0.0f, std::min(1.0f, fY * fM * fC));
                Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
            }

            // 5) Contract to channel RAW using print sensitometries
            float raw[3] = { 0,0,0 };
            raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw, ws.tablesView.deltaLambda);

            // Apply print exposure (agx parity)
            raw[0] *= std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;
            raw[1] *= std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;
            raw[2] *= std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;

            // Mid-gray compensation via green channel only
            const float gFactor = std::isfinite(prm.exposureCompGFactor) ? std::max(0.0f, prm.exposureCompGFactor) : 1.0f;
            raw[1] *= gFactor;

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

            // XYZ → DWG
            Spectral::XYZ_to_DWG_linear(XYZ, rgbOut);

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

    } // namespace Print
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
        Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbIn, E, 1.0f,
            (ws.tablesView.K > 0 ? &ws.tablesView : nullptr),
            (ws.spdReady ? ws.spdSInv : nullptr),
            (int)std::clamp(ws.spectralMode, 0, 1),
            (ws.exposureModel == 1) && ws.spdReady,
            ws.sensB, ws.sensG, ws.sensR);

        // Apply camera exposure explicitly on negative leg to ensure parity and avoid hidden double-handling.
        {
            const float sExp = (std::isfinite(exposureScale) ? std::max(0.0f, exposureScale) : 1.0f);
            if (sExp != 1.0f) { E[0] *= sExp; E[1] *= sExp; E[2] *= sExp; }
        }



        float logE[3] = {
            std::log10(std::max(E[0], 1e-6f)) + ws.logEOffB,
            std::log10(std::max(E[1], 1e-6f)) + ws.logEOffG,
            std::log10(std::max(E[2], 1e-6f)) + ws.logEOffR
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
                // Skip local (non-spatial) DIR when curves were precorrected and spatial diffusion is disabled
                const bool allowLocalDIR = (!ws.dirPrecorrected) || (dirRT.spatialSigmaPixels > 0.5f);
                if (dirRT.active && allowLocalDIR) {
                    Couplers::ApplyInputLogE io{ {logE[0], logE[1], logE[2]}, {D_neg[0], D_neg[1], D_neg[2]} };
                    // Per-instance clamp variant: aligns clamp domain to the same curves we sample
                    Couplers::apply_runtime_logE_with_curves(io, dirRT, ws.densB, ws.densG, ws.densR);
                    float logE2_clamped[3] = { io.logE[0], io.logE[1], io.logE[2] };
                    sample_ws(logE2_clamped, D_neg);
                }
#endif

        // 2) Negative transmittance with optional baseline blend
        const float wNeutral = Spectral::neutral_blend_weight_from_DWG_rgb(rgbIn);
        thread_local std::vector<float> Tneg, Ee_expose, Tprint, Ee_viewed;
        negative_T_from_dyes(ws, D_neg, Tneg, wNeutral);

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

        // Blend each wheel between identity (1.0) and its full transmittance curve
        auto blend_filter = [](float curveVal, float amount)->float {
            // AgX parity: optical density scaling → transmittance = curve^amount
            const float a = std::isfinite(amount) ? std::clamp(amount, 0.0f, 8.0f) : 0.0f;
            const float c = std::isfinite(curveVal) ? std::clamp(curveVal, 1e-6f, 1.0f) : 1.0f;
            return std::pow(c, a);
            };

        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_filter(rt.filterY.linear.empty() ? 1.0f : rt.filterY.linear[i], prm.yFilter);
            const float fM = blend_filter(rt.filterM.linear.empty() ? 1.0f : rt.filterM.linear[i], prm.mFilter);
            const float fC = blend_filter(rt.filterC.linear.empty() ? 1.0f : rt.filterC.linear[i], prm.cFilter);
            const float fTotal = std::max(0.0f, std::min(1.0f, fY * fM * fC));
            Ee_filtered[i] = std::max(0.0f, Ee_expose[i] * fTotal);
        }

        // Compute per-channel raw via sensitivity contraction (agx parity)
        float raw[3];
        raw_exposures_from_filtered_light(rt.profile, Ee_filtered, raw, ws.tablesView.deltaLambda);

        // Apply print exposure scaling to raw (agx: raw *= print_exposure)
        raw[0] *= std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;
        raw[1] *= std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;
        raw[2] *= std::isfinite(prm.exposure) ? std::max(0.0f, prm.exposure) : 1.0f;

        // Agx parity: apply mid-gray compensation via green channel only (before log10 and development).
        // raw *= [1, exposureCompGFactor, 1]
        {
            const float gFactor = std::isfinite(prm.exposureCompGFactor) ? std::max(0.0f, prm.exposureCompGFactor) : 1.0f;
            raw[1] *= gFactor;
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

        // XYZ -> DWG
        Spectral::XYZ_to_DWG_linear(XYZ, rgbOut);

    }

} // namespace Print
