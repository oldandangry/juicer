#include "JuicerState.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

extern const std::string gDataDir;

namespace {
    static const char* kPrintPaperOptions[] = {
        "2383",
        "2393"
    };
    static constexpr int kNumPrintPapers = static_cast<int>(sizeof(kPrintPaperOptions) / sizeof(kPrintPaperOptions[0]));

    struct FilmStockDefinition {
        const char* optionLabel;
        const char* folderName;
        const char* jsonKey;
    };

    static constexpr FilmStockDefinition kFilmStocks[] = {
        { "Vision3 250D", "Vision3 250D", "kodak_vision3_250d_uc" },
        { "Vision3 50D",  "Vision3 50D",  "kodak_vision3_50d_uc" },
        { "Vision3 200T", "Vision3 200T", "kodak_vision3_200t_uc" },
        { "Vision3 500T", "Vision3 500T", "kodak_vision3_500t_uc" },
        { "Portra 400",   "Portra_400",   "kodak_portra_400_auc" }
    };

    static constexpr int kNumFilmStocks = static_cast<int>(sizeof(kFilmStocks) / sizeof(kFilmStocks[0]));

    static inline const FilmStockDefinition& film_stock_for_index(int filmIndex) {
        if (filmIndex < 0 || filmIndex >= kNumFilmStocks) {
            filmIndex = 0;
        }
        return kFilmStocks[filmIndex];
    }

    static inline const char* film_json_key_for_folder(const std::string& folder) {
        for (const FilmStockDefinition& stock : kFilmStocks) {
            if (folder == stock.folderName) {
                return stock.jsonKey;
            }
        }
        return nullptr;
    }

    static std::string last_path_segment(const std::string& path) {
        if (path.empty()) return {};
        size_t end = path.find_last_not_of("\\/");
        if (end == std::string::npos) return {};
        size_t start = path.find_last_of("\\/", end);
        if (start == std::string::npos) {
            return path.substr(0, end + 1);
        }
        return path.substr(start + 1, end - start);
    }

    static std::string sanitize_densitometer_type(const std::string& type) {
        std::string out;
        out.reserve(type.size());
        bool lastUnderscore = false;
        for (char ch : type) {
            unsigned char uc = static_cast<unsigned char>(ch);
            if (std::isalnum(uc)) {
                out.push_back(static_cast<char>(std::tolower(uc)));
                lastUnderscore = false;
            }
            else if (ch == '_' || ch == '-' || ch == ' ') {
                if (!lastUnderscore && !out.empty()) {
                    out.push_back('_');
                    lastUnderscore = true;
                }
            }
        }
        while (!out.empty() && out.back() == '_') {
            out.pop_back();
        }
        return out;
    }

    static bool load_resampled_channel(const std::string& path, std::vector<float>& outChannel) {
        try {
            auto pairs = Spectral::load_csv_pairs(path);
            if (pairs.empty()) return false;

            auto pinned = Spectral::resample_pairs_to_shape(pairs, Spectral::gShape);
            if (pinned.empty() || static_cast<int>(pinned.size()) != Spectral::gShape.K)
                return false;

            outChannel.resize(Spectral::gShape.K);
            for (int i = 0; i < Spectral::gShape.K; ++i)
                outChannel[i] = std::max(0.0f, pinned[i].second);

            return true;
        }
        catch (const std::exception& e) {
            std::ofstream log("C:/temp/juicer_densitometer_warning.txt", std::ios::app);
            if (log.is_open())
                log << "[load_resampled_channel] Exception: " << e.what()
                << " while loading " << path << "\n";
            return false;
        }
        catch (...) {
            std::ofstream log("C:/temp/juicer_densitometer_warning.txt", std::ios::app);
            if (log.is_open())
                log << "[load_resampled_channel] Unknown exception while loading "
                << path << "\n";
            return false;
        }
    }
}

uint64_t hash_params(const ParamSnapshot& p) {
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
        };
    uint64_t h = 0;
    h = mix(h, static_cast<uint64_t>(p.filmStockIndex));
    h = mix(h, static_cast<uint64_t>(p.printPaperIndex));
    h = mix(h, static_cast<uint64_t>(p.refIll));
    h = mix(h, static_cast<uint64_t>(p.viewIll));
    h = mix(h, static_cast<uint64_t>(p.enlIll));
    h = mix(h, static_cast<uint64_t>(p.couplersActive));
    h = mix(h, static_cast<uint64_t>(p.couplersPrecorrect));
    h = mix(h, static_cast<uint64_t>(p.couplersAmount * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioR * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioG * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.ratioB * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.sigma * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.high * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.spatialSigma * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.unmix));
    return h;
}

std::string print_dir_for_index(int index) {
    if (index < 0 || index >= kNumPrintPapers) index = 0;
    return gDataDir + std::string("Print\\") + kPrintPaperOptions[index] + "\\";
}

std::string stock_dir_for_index(int index) {
    const FilmStockDefinition& stock = film_stock_for_index(index);
    return gDataDir + std::string("Stock\\") + stock.folderName + "\\";
}

const char* negative_json_key_for_stock_index(int filmIndex) {
    return film_stock_for_index(filmIndex).jsonKey;
}

int film_stock_option_count() {
    return kNumFilmStocks;
}

const char* film_stock_option_label(int index) {
    return film_stock_for_index(index).optionLabel;
}

int print_paper_option_count() {
    return kNumPrintPapers;
}

const char* print_paper_option_label(int index) {
    if (index < 0 || index >= kNumPrintPapers) index = 0;
    return kPrintPaperOptions[index];
}

bool load_film_stock_into_base(const std::string& stockDir, InstanceState& S) {
    JTRACE_SCOPE("STOCK", std::string("load_film_stock_into_base: ") + stockDir);

    std::vector<std::pair<float, float>> c_data;
    std::vector<std::pair<float, float>> m_data;
    std::vector<std::pair<float, float>> y_data;

    std::vector<std::pair<float, float>> r_sens;
    std::vector<std::pair<float, float>> g_sens;
    std::vector<std::pair<float, float>> b_sens;

    std::vector<std::pair<float, float>> dmin;
    std::vector<std::pair<float, float>> dmid;

    std::vector<std::pair<float, float>> dc_r;
    std::vector<std::pair<float, float>> dc_g;
    std::vector<std::pair<float, float>> dc_b;

    S.base.densitometerType.clear();
    S.base.densityMidNeutral.clear();

    const std::string folder = last_path_segment(stockDir);
    if (folder.empty()) {
        JTRACE("STOCK", "stock folder empty; cannot resolve JSON profile");
        return false;
    }

    const char* jsonKey = film_json_key_for_folder(folder);
    if (!jsonKey) {
        JTRACE("STOCK", std::string("no agx profile key for folder: ") + folder);
        return false;
    }
    const std::string jsonPath = gDataDir + std::string("profiles\\") + jsonKey + std::string(".json");
    Profiles::AgxFilmProfile profile;
    if (!Profiles::load_agx_film_profile_json(jsonPath, profile)) {
        JTRACE("STOCK", std::string("failed to load agx profile json: ") + jsonKey);
        return false;
    }

    c_data = std::move(profile.dyeC);
    m_data = std::move(profile.dyeM);
    y_data = std::move(profile.dyeY);
    r_sens = std::move(profile.logSensR);
    g_sens = std::move(profile.logSensG);
    b_sens = std::move(profile.logSensB);
    dc_r = std::move(profile.densityCurveR);
    dc_g = std::move(profile.densityCurveG);
    dc_b = std::move(profile.densityCurveB);
    dmin = std::move(profile.baseMin);
    dmid = std::move(profile.baseMid);
    S.base.densitometerType = sanitize_densitometer_type(profile.densitometer);
    S.base.densityMidNeutral = std::move(profile.densityMidNeutral);
    S.base.dirCouplers = profile.dirCouplers;
    S.base.maskingCouplers = profile.maskingCouplers;
    JTRACE("STOCK", std::string("loaded agx profile json: ") + jsonKey);

    {
        auto sz = [](const auto& v) { return static_cast<int>(v.size()); };
        std::ostringstream oss;
        oss << "c/m/y=" << sz(c_data) << "/" << sz(m_data) << "/" << sz(y_data)
            << " sens r/g/b=" << sz(r_sens) << "/" << sz(g_sens) << "/" << sz(b_sens)
            << " dens r/g/b=" << sz(dc_r) << "/" << sz(dc_g) << "/" << sz(dc_b)
            << " base min/mid=" << sz(dmin) << "/" << sz(dmid);
        JTRACE("STOCK", oss.str());
    }

    if (c_data.empty()) JTRACE("STOCK", "dye_density_c missing/empty");
    if (m_data.empty()) JTRACE("STOCK", "dye_density_m missing/empty");
    if (y_data.empty()) JTRACE("STOCK", "dye_density_y missing/empty");
    if (r_sens.empty()) JTRACE("STOCK", "log_sensitivity_r missing/empty");
    if (g_sens.empty()) JTRACE("STOCK", "log_sensitivity_g missing/empty");
    if (b_sens.empty()) JTRACE("STOCK", "log_sensitivity_b missing/empty");
    if (dc_r.empty()) JTRACE("STOCK", "density_curve_r missing/empty");
    if (dc_g.empty()) JTRACE("STOCK", "density_curve_g missing/empty");
    if (dc_b.empty()) JTRACE("STOCK", "density_curve_b missing/empty");

    const bool okCore =
        !c_data.empty() && !m_data.empty() && !y_data.empty() &&
        !r_sens.empty() && !g_sens.empty() && !b_sens.empty() &&
        !dc_r.empty() && !dc_g.empty() && !dc_b.empty();
    if (!okCore) {
        JTRACE("STOCK", "okCore=false (required assets missing)");
        return false;
    }

    Spectral::build_curve_on_shape_from_linear_pairs(S.base.epsY, y_data);
    Spectral::build_curve_on_shape_from_linear_pairs(S.base.epsM, m_data);
    Spectral::build_curve_on_shape_from_linear_pairs(S.base.epsC, c_data);

    Spectral::build_curve_on_shape_from_log10_pairs(S.base.sensB, b_sens);
    Spectral::build_curve_on_shape_from_log10_pairs(S.base.sensG, g_sens);
    Spectral::build_curve_on_shape_from_log10_pairs(S.base.sensR, r_sens);

    Spectral::sort_and_build(S.base.densB, dc_b);
    Spectral::sort_and_build(S.base.densG, dc_g);
    Spectral::sort_and_build(S.base.densR, dc_r);

    auto subtract_baseline_floor = [](Spectral::Curve& curve) {
        if (curve.linear.empty()) return;
        float minVal = FLT_MAX;
        for (float v : curve.linear) {
            if (std::isfinite(v) && v < minVal) {
                minVal = v;
            }
        }
        if (!std::isfinite(minVal) || minVal == FLT_MAX || minVal == 0.0f) {
            return;
        }
        for (float& v : curve.linear) {
            v = std::max(0.0f, v - minVal);
        }
        };
    subtract_baseline_floor(S.base.densB);
    subtract_baseline_floor(S.base.densG);
    subtract_baseline_floor(S.base.densR);

    Spectral::build_curve_on_shape_from_linear_pairs(S.base.baseMin, dmin);
    Spectral::build_curve_on_shape_from_linear_pairs(S.base.baseMid, dmid);
    S.base.hasBaseline = !S.base.baseMin.linear.empty();

    {
        std::ostringstream oss;
        oss << "epsY/M/C K=" << static_cast<int>(S.base.epsY.linear.size())
            << "/" << static_cast<int>(S.base.epsM.linear.size())
            << "/" << static_cast<int>(S.base.epsC.linear.size())
            << " sensB/G/R K=" << static_cast<int>(S.base.sensB.linear.size())
            << "/" << static_cast<int>(S.base.sensG.linear.size())
            << "/" << static_cast<int>(S.base.sensR.linear.size())
            << " densB/G/R K=" << static_cast<int>(S.base.densB.linear.size())
            << "/" << static_cast<int>(S.base.densG.linear.size())
            << "/" << static_cast<int>(S.base.densR.linear.size())
            << " baseMin/baseMid K=" << static_cast<int>(S.base.baseMin.linear.size())
            << "/" << static_cast<int>(S.base.baseMid.linear.size())
            << " hasBaseline=" << (S.base.hasBaseline ? 1 : 0)
            << " densType=" << (S.base.densitometerType.empty() ? std::string("none") : S.base.densitometerType);
        JTRACE("STOCK", oss.str());
    }

    JTRACE("STOCK", std::string("loaded OK; baseline=") + (S.base.hasBaseline ? "1" : "0"));

    return true;
}

void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P) {
    auto sanitize_curve = [](Spectral::Curve& c) {
        for (float& v : c.linear) {
            if (!std::isfinite(v)) v = 0.0f;
            if (v < 0.0f) v = 0.0f;
        }
        };

    auto estimate_base_dyes = [](const Spectral::Curve& baseCurve,
        const Spectral::Curve& epsY,
        const Spectral::Curve& epsM,
        const Spectral::Curve& epsC) -> std::array<float, 3>
        {
            std::array<float, 3> result{ 0.0f, 0.0f, 0.0f };
            const size_t K = baseCurve.linear.size();
            if (K == 0 || epsY.linear.size() != K || epsM.linear.size() != K || epsC.linear.size() != K) {
                return result;
            }

            double ATA[3][3] = { {0.0,0.0,0.0}, {0.0,0.0,0.0}, {0.0,0.0,0.0} };
            double ATb[3] = { 0.0, 0.0, 0.0 };

            for (size_t i = 0; i < K; ++i) {
                const double ay = epsY.linear[i];
                const double am = epsM.linear[i];
                const double ac = epsC.linear[i];
                const double b = baseCurve.linear[i];
                if (!std::isfinite(ay) || !std::isfinite(am) || !std::isfinite(ac) || !std::isfinite(b)) {
                    continue;
                }
                const double vec[3] = { ay, am, ac };
                for (int r = 0; r < 3; ++r) {
                    ATb[r] += vec[r] * b;
                    for (int c = 0; c < 3; ++c) {
                        ATA[r][c] += vec[r] * vec[c];
                    }
                }
            }

            double mat[3][4];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    mat[r][c] = ATA[r][c];
                }
                mat[r][3] = ATb[r];
            }

            for (int i = 0; i < 3; ++i) {
                int pivot = i;
                double maxAbs = std::fabs(mat[i][i]);
                for (int r = i + 1; r < 3; ++r) {
                    const double absVal = std::fabs(mat[r][i]);
                    if (absVal > maxAbs) {
                        maxAbs = absVal;
                        pivot = r;
                    }
                }
                if (maxAbs < 1e-9) {
                    return result;
                }
                if (pivot != i) {
                    for (int c = i; c < 4; ++c) {
                        std::swap(mat[i][c], mat[pivot][c]);
                    }
                }
                const double inv = 1.0 / mat[i][i];
                for (int c = i; c < 4; ++c) {
                    mat[i][c] *= inv;
                }
                for (int r = 0; r < 3; ++r) {
                    if (r == i) continue;
                    const double factor = mat[r][i];
                    for (int c = i; c < 4; ++c) {
                        mat[r][c] -= factor * mat[i][c];
                    }
                }
            }

            for (int i = 0; i < 3; ++i) {
                float v = static_cast<float>(mat[i][3]);
                if (!std::isfinite(v) || v < 0.0f) {
                    v = 0.0f;
                }
                result[i] = v;
            }

            return result;
        };

    JTRACE_SCOPE("BUILD", "rebuild_working_state");

    std::unique_lock<std::mutex> lk(S.m);

    {
        std::ostringstream oss;
        oss << "enter with baseLoaded=" << (S.baseLoaded ? 1 : 0);
        JTRACE("BUILD", oss.str());
    }

    Print::build_illuminant_from_choice(P.enlIll, S.printRT, S.dataDir, /*forEnlarger*/true);
    Print::build_illuminant_from_choice(P.viewIll, S.printRT, S.dataDir, /*forEnlarger*/false);
    {
        std::ostringstream oss;
        oss << "Enl illum K=" << static_cast<int>(S.printRT.illumEnlarger.linear.size())
            << " View illum K=" << static_cast<int>(S.printRT.illumView.linear.size());
        JTRACE("BUILD", oss.str());
    }

    Spectral::Curve epsY = S.base.epsY;
    Spectral::Curve epsM = S.base.epsM;
    Spectral::Curve epsC = S.base.epsC;
    Spectral::Curve sensB = S.base.sensB;
    Spectral::Curve sensG = S.base.sensG;
    Spectral::Curve sensR = S.base.sensR;
    Spectral::Curve densB = S.base.densB;
    Spectral::Curve densG = S.base.densG;
    Spectral::Curve densR = S.base.densR;
    Spectral::Curve baseMin = S.base.baseMin;
    Spectral::Curve baseMid = S.base.baseMid;
    Spectral::Curve illumRef;
    bool hasRefIlluminant = false;
    const bool hasBaseline = S.base.hasBaseline;

    if (P.unmix) {
        JTRACE("BUILD", "unmix: entered");

        auto load_resp = [&](char channel)->std::vector<float> {
            std::vector<float> out;
            auto try_load = [&](const std::string& rel) {
                if (load_resampled_channel(S.dataDir + rel, out)) {
                    JTRACE("BUILD", std::string("unmix: loaded ") + rel + " size=" + std::to_string(out.size()));
                    return true;
                }
                return false;
                };
            const std::string& densType = S.base.densitometerType;
            if (!densType.empty()) {
                std::string rel = std::string("densitometer\\") + densType + "\\responsivity_";
                rel.push_back(channel);
                rel += ".csv";
                if (try_load(rel)) {
                    return out;
                }
            }

            const std::string fallback = std::string("densitometer\\responsivity_") + channel + ".csv";
            if (!try_load(fallback)) {
                JTRACE("BUILD", std::string("unmix: failed to load responsivity for ") + channel);
            }
            return out;
            };
        std::vector<float> respB = load_resp('b');
        std::vector<float> respG = load_resp('g');
        std::vector<float> respR = load_resp('r');

        if ((int)respB.size() == Spectral::gShape.K &&
            (int)respG.size() == Spectral::gShape.K &&
            (int)respR.size() == Spectral::gShape.K)
        {
            JTRACE("BUILD", "unmix: responsivity sizes match gShape.K=" + std::to_string(Spectral::gShape.K));

            const int K = Spectral::gShape.K;
            std::vector<std::vector<float>> dyeDensityCMY((size_t)K, std::vector<float>(3, 0.0f));
            for (int i = 0; i < K; ++i) {
                dyeDensityCMY[(size_t)i][0] = epsC.linear[i];
                dyeDensityCMY[(size_t)i][1] = epsM.linear[i];
                dyeDensityCMY[(size_t)i][2] = epsY.linear[i];
            }
            JTRACE("BUILD", "unmix: dyeDensityCMY populated");

            std::vector<std::vector<float>> densitometerResp(3, std::vector<float>(Spectral::gShape.K));
            for (int i = 0; i < Spectral::gShape.K; ++i) {
                densitometerResp[0][i] = respR[i];
                densitometerResp[1][i] = respG[i];
                densitometerResp[2][i] = respB[i];
            }
            JTRACE("BUILD", "unmix: densitometerResp populated");

            float M[3][3];
            Spectral::compute_densitometer_crosstalk_matrix(densitometerResp, dyeDensityCMY, M);
            {
                std::ostringstream oss;
                oss << "unmix: crosstalk matrix = ["
                    << M[0][0] << "," << M[0][1] << "," << M[0][2] << " ; "
                    << M[1][0] << "," << M[1][1] << "," << M[1][2] << " ; "
                    << M[2][0] << "," << M[2][1] << "," << M[2][2] << "]";
                JTRACE("BUILD", oss.str());
            }

            const size_t nL = std::min({ densR.linear.size(), densG.linear.size(), densB.linear.size() });
            std::vector<std::array<float, 3>> curvesCMY(nL);
            for (size_t i = 0; i < nL; ++i) {
                curvesCMY[i] = { densR.linear[i], densG.linear[i], densB.linear[i] };
            }
            JTRACE("BUILD", "unmix: curvesCMY packed, nL=" + std::to_string(nL));

            Spectral::unmix_density_curves(curvesCMY, M);
            JTRACE("BUILD", "unmix: curves unmixed");

            for (size_t i = 0; i < nL; ++i) {
                densR.linear[i] = curvesCMY[i][0];
                densG.linear[i] = curvesCMY[i][1];
                densB.linear[i] = curvesCMY[i][2];
            }
            JTRACE("BUILD", "unmix: densR/G/B replaced");

            sanitize_curve(densB);
            sanitize_curve(densG);
            sanitize_curve(densR);
            JTRACE("BUILD", "unmix: densR/G/B sanitized");

        }
        else {
            JTRACE("BUILD", "unmix: responsivity sizes do NOT match gShape.K");
        }
    }

    Spectral::Curve sensB_beforeBalance = sensB;
    Spectral::Curve sensG_beforeBalance = sensG;
    Spectral::Curve sensR_beforeBalance = sensR;
    {
        Print::Runtime tmpRT;
        Print::build_illuminant_from_choice(P.refIll, tmpRT, S.dataDir, /*forEnlarger*/false);
        illumRef = tmpRT.illumView;
        hasRefIlluminant =
            (!illumRef.linear.empty() && (int)illumRef.linear.size() == Spectral::gShape.K);
        Spectral::Curve sensB_bal, sensG_bal, sensR_bal;
        Spectral::Curve densB_bal, densG_bal, densR_bal;
        Spectral::balance_negative_under_reference_non_global(
            tmpRT.illumView, sensB, sensG, sensR, densB, densG, densR,
            sensB_bal, sensG_bal, sensR_bal, densB_bal, densG_bal, densR_bal);
        sensB = std::move(sensB_bal);
        sensG = std::move(sensG_bal);
        sensR = std::move(sensR_bal);
        densB = std::move(densB_bal);
        densG = std::move(densG_bal);
        densR = std::move(densR_bal);

        JTRACE("BUILD", "balance under reference complete");
    }

    sanitize_curve(densB);
    sanitize_curve(densG);
    sanitize_curve(densR);

    if (P.couplersPrecorrect) {
        float amountRGB[3] = {
            static_cast<float>(std::clamp(P.ratioB, 0.0, 1.0)),
            static_cast<float>(std::clamp(P.ratioG, 0.0, 1.0)),
            static_cast<float>(std::clamp(P.ratioR, 0.0, 1.0))
        };
        Couplers::Runtime couplersRuntime{};
        Couplers::build_dir_matrix(couplersRuntime.M, amountRGB, static_cast<float>(P.sigma));
        Couplers::precorrect_density_curves_before_DIR(couplersRuntime.M, static_cast<float>(P.high));
        sanitize_curve(Spectral::gDensityCurveB);
        sanitize_curve(Spectral::gDensityCurveG);
        sanitize_curve(Spectral::gDensityCurveR);
        JTRACE("BUILD", "precorrect: densR/G/B sanitized");
        Spectral::enforce_monotonic_density_curves();
        JTRACE("BUILD", "precorrect: monotonicity enforced on densR/G/B");
        Spectral::deduplicate_density_curve_grid();
        JTRACE("BUILD", "precorrect: grid dedup applied to densR/G/B");
    }

    const Profiles::DirCouplersProfile& dirCfg = S.base.dirCouplers;
    if (P.couplersActive && dirCfg.valid) {
        Couplers::Runtime runtime;
        runtime.active = true;
        runtime.highShift = static_cast<float>(P.high);
        runtime.spatialSigmaPixels = static_cast<float>(P.spatialSigma);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                runtime.M[r][c] = dirCfg.dirMatrix[r][c];
        runtime.dMax[0] = dirCfg.maxDensityB;
        runtime.dMax[1] = dirCfg.maxDensityG;
        runtime.dMax[2] = dirCfg.maxDensityR;
        Couplers::apply_runtime_to_density_curves(runtime);
        sanitize_curve(Spectral::gDensityCurveB);
        sanitize_curve(Spectral::gDensityCurveG);
        sanitize_curve(Spectral::gDensityCurveR);
        JTRACE("BUILD", "DIR couplers applied and sanitized");
    }

    Spectral::build_working_tables_from_base(
        instance,
        S.dataDir,
        epsY, epsM, epsC,
        sensB, sensG, sensR,
        densB, densG, densR,
        baseMin, baseMid,
        hasBaseline,
        sensB_beforeBalance, sensG_beforeBalance, sensR_beforeBalance,
        illumRef,
        hasRefIlluminant,
        S.printRT,
        S.base.densityMidNeutral,
        S.base.maskingCouplers,
        S);
}