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
    h = mix(h, static_cast<uint64_t>(p.spatialSigmaMicrometers * 10000.0));
    h = mix(h, static_cast<uint64_t>(p.unmix));
    return h;
}

std::string print_dir_for_index(int index) {
    if (index < 0 || index >= kNumPrintPapers) index = 0;
    return gDataDir + std::string("Print\\") + kPrintPaperOptions[index] + "\\";
}

std::string stock_dir_for_index(int index) {
    const FilmStockDefinition& stock = film_stock_for_index(index);
    return gDataDir + std::string("profiles\\") + stock.folderName + "\\";
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
    if (!dc_r.empty() && !dc_g.empty() && !dc_b.empty()) {
        std::ostringstream oss;
        oss << "density curves loaded from JSON '" << jsonKey << "' samples R/G/B="
            << dc_r.size() << "/" << dc_g.size() << "/" << dc_b.size();
        JTRACE("STOCK", oss.str());
    }
    else {
        JTRACE("STOCK", std::string("density curves missing in JSON profile: ") + jsonKey);
    }

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
        JTRACE("STOCK", "film stock load aborted due to missing JSON density curves");
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

    bool precorrectApplied = false;
    Couplers::Runtime dirRT{};
    dirRT.active = (P.couplersActive != 0);
    {
        auto clampRatio = [](double v) -> float {
            if (!std::isfinite(v) || v < 0.0) return 0.0f;
            if (v > 1.0) return 1.0f;
            return static_cast<float>(v);
            };
        auto clampAmount = [](double v) -> float {
            if (!std::isfinite(v) || v < 0.0) return 0.0f;
            if (v > 2.0) return 2.0f;
            return static_cast<float>(v);
            };
        const float amountScale = clampAmount(P.couplersAmount);
        const float amount[3] = {
            amountScale * clampRatio(P.ratioB),
            amountScale * clampRatio(P.ratioG),
            amountScale * clampRatio(P.ratioR)
        };
        Couplers::build_dir_matrix(dirRT.M, amount, static_cast<float>(P.sigma));
        dirRT.highShift = static_cast<float>(P.high);
        dirRT.spatialSigmaMicrometers = static_cast<float>(P.spatialSigmaMicrometers);
        dirRT.spatialSigmaPixels = 0.0f;

        if (dirRT.active && P.couplersPrecorrect) {
            Spectral::Curve densB_corr, densG_corr, densR_corr;
            Couplers::precorrect_density_curves_before_DIR_into(
                dirRT.M, dirRT.highShift,
                densB, densG, densR,
                densB_corr, densG_corr, densR_corr);
            densB = std::move(densB_corr);
            densG = std::move(densG_corr);
            densR = std::move(densR_corr);
            precorrectApplied = true;
        }

        sanitize_curve(densB);
        sanitize_curve(densG);
        sanitize_curve(densR);
        JTRACE("BUILD", "precorrect: densR/G/B sanitized");

        auto enforce_monotone_curve = [](Spectral::Curve& c) {
            float prev = (c.linear.empty() || !std::isfinite(c.linear[0])) ? 0.0f : c.linear[0];
            for (size_t i = 0; i < c.linear.size(); ++i) {
                float cur = c.linear[i];
                if (!std::isfinite(cur) || cur < 0.0f) cur = 0.0f;
                if (cur < prev) cur = prev;
                c.linear[i] = cur;
                prev = cur;
            }
            };
        enforce_monotone_curve(densB);
        enforce_monotone_curve(densG);
        enforce_monotone_curve(densR);

        JTRACE("BUILD", "precorrect: monotonicity enforced on densR/G/B");
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
        dedup_strict_curve(densB);
        dedup_strict_curve(densG);
        dedup_strict_curve(densR);

        JTRACE("BUILD", "precorrect: grid dedup applied to densR/G/B");

        auto vmax = [](const Spectral::Curve& c) {
            float m = 0.0f;
            for (float v : c.linear) m = std::max(m, v);
            if (!std::isfinite(m) || m <= 1e-4f) m = 1.0f;
            if (m > 1000.0f) m = 1000.0f;
            return m;
            };
        dirRT.dMax[0] = vmax(densB);
        dirRT.dMax[1] = vmax(densG);
        dirRT.dMax[2] = vmax(densR);

        {
            std::ostringstream oss;
            oss << "DIR active=" << (dirRT.active ? 1 : 0)
                << " sigma=" << static_cast<float>(P.sigma) << " high=" << static_cast<float>(P.high)
                << " dMax=" << dirRT.dMax[0] << "," << dirRT.dMax[1] << "," << dirRT.dMax[2]
                << " precorrect=" << (P.couplersPrecorrect ? 1 : 0);
            JTRACE("BUILD", oss.str());
        }
    }

    Spectral::NegativeCouplerParams negParams;
    negParams.DmaxY = dirRT.dMax[0];
    negParams.DmaxM = dirRT.dMax[1];
    negParams.DmaxC = dirRT.dMax[2];
    negParams.baseY = 0.0f;
    negParams.baseM = 0.0f;
    negParams.baseC = 0.0f;
    negParams.kB = 6.0f;
    negParams.kG = 6.0f;
    negParams.kR = 6.0f;
    {
        const float defaultMask[9] = {
            0.98f, -0.06f, -0.02f,
           -0.03f,  0.98f, -0.05f,
           -0.02f, -0.04f,  0.98f };
        std::copy(std::begin(defaultMask), std::end(defaultMask), negParams.mask);
    }

    if (S.base.hasBaseline) {
        const auto baseCoeffs = estimate_base_dyes(baseMin, epsY, epsM, epsC);
        negParams.baseY = baseCoeffs[0];
        negParams.baseM = baseCoeffs[1];
        negParams.baseC = baseCoeffs[2];
    }

    const Profiles::DirCouplersProfile& dirCfg = S.base.dirCouplers;
    if (dirCfg.hasData) {
        auto computeK = [&](int idx) -> float {
            float amount = std::isfinite(dirCfg.amount) ? dirCfg.amount : 1.0f;
            if (amount <= 0.0f) amount = 0.1f;
            float ratio = dirCfg.ratioRGB[idx];
            if (!std::isfinite(ratio) || ratio <= 0.0f) ratio = 1.0f;
            float k = 6.0f * amount * ratio;
            if (!std::isfinite(k) || k <= 0.0f) k = 6.0f;
            return std::clamp(k, 1.0f, 24.0f);
            };
        negParams.kB = computeK(0);
        negParams.kG = computeK(1);
        negParams.kR = computeK(2);
    }

    const bool hasMaskingData = S.base.maskingCouplers.hasData;
    if (dirCfg.hasData || hasMaskingData) {
        float amountRGB[3] = { 1.0f, 1.0f, 1.0f };
        if (dirCfg.hasData) {
            for (int i = 0; i < 3; ++i) {
                float ratio = dirCfg.ratioRGB[i];
                if (!std::isfinite(ratio)) ratio = 1.0f;
                if (ratio < 0.0f) ratio = 0.0f;
                float amount = std::isfinite(dirCfg.amount) ? dirCfg.amount : 1.0f;
                if (amount < 0.0f) amount = 0.0f;
                amountRGB[i] = std::clamp(amount * ratio, 0.0f, 1.0f);
            }
        }

        float dirMatrix[3][3] = { {0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f} };
#ifdef JUICER_ENABLE_COUPLERS
        Couplers::build_dir_matrix(dirMatrix, amountRGB, dirCfg.hasData ? dirCfg.diffusionInterlayer : 0.0f);
#else
        auto build_dir_matrix_local = [](float M[3][3], const float amountValues[3], float layerSigma) {
            const float sigma = std::isfinite(layerSigma) ? std::max(0.0f, layerSigma) : 0.0f;
            float amt[3] = { amountValues[0], amountValues[1], amountValues[2] };
            const float sigmaCapped = std::min(sigma, 3.0f);
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(amt[i])) amt[i] = 0.0f;
                amt[i] = std::clamp(amt[i], 0.0f, 1.0f);
            }
            auto gauss = [sigmaCapped](int dx) -> float {
                if (sigmaCapped <= 0.0f) {
                    return (dx == 0) ? 1.0f : 0.0f;
                }
                const float s2 = sigmaCapped * sigmaCapped;
                return std::exp(-0.5f * (dx * dx) / s2);
                };
            for (int r = 0; r < 3; ++r) {
                float row[3];
                float wsum = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    row[c] = gauss(c - r);
                    wsum += row[c];
                }
                if (wsum > 0.0f) {
                    for (int c = 0; c < 3; ++c) {
                        row[c] /= wsum;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    M[r][c] = amt[r] * row[c];
                }
            }
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(M[r][c])) {
                        M[r][c] = 0.0f;
                    }
                }
            }
            };
        build_dir_matrix_local(dirMatrix, amountRGB, dirCfg.hasData ? dirCfg.diffusionInterlayer : 0.0f);
#endif

        std::array<float, 3> gaussianSum{ 0.0f, 0.0f, 0.0f };
        if (hasMaskingData) {
            for (int ch = 0; ch < 3; ++ch) {
                const auto& channel = S.base.maskingCouplers.gaussianModel[ch];
                for (const auto& tri : channel) {
                    float amp = tri[2];
                    if (!std::isfinite(amp)) amp = 0.0f;
                    gaussianSum[ch] += std::fabs(amp);
                }
            }
        }

        const float maskScaleBase = 0.25f;
        const float maskScaleOffset = hasMaskingData ? 0.05f : 0.0f;
        for (int r = 0; r < 3; ++r) {
            float sum = hasMaskingData ? gaussianSum[r] : 0.0f;
            if (!std::isfinite(sum)) sum = 0.0f;
            float amountRow = dirCfg.hasData ? amountRGB[r] : 1.0f;
            if (!std::isfinite(amountRow) || amountRow < 0.0f) amountRow = 0.0f;
            float scale = maskScaleBase * (sum + maskScaleOffset);
            if (dirCfg.hasData) {
                scale *= amountRow;
            }
            scale = std::clamp(scale, 0.0f, 0.25f);
            for (int c = 0; c < 3; ++c) {
                float val = dirMatrix[r][c];
                if (!std::isfinite(val)) {
                    val = (r == c) ? 1.0f : 0.0f;
                }
                const float delta = scale * val;
                if (r == c) {
                    float diag = 1.0f - delta;
                    if (!std::isfinite(diag)) diag = 1.0f;
                    if (diag < 0.0f) diag = 0.0f;
                    negParams.mask[r * 3 + c] = diag;
                }
                else {
                    float off = -delta;
                    if (!std::isfinite(off)) off = 0.0f;
                    off = std::clamp(off, -1.0f, 1.0f);
                    negParams.mask[r * 3 + c] = off;
                }
            }
        }
    }

    WorkingState* target = S.inactive();
    while (S.renderWS.load(std::memory_order_acquire) == target &&
        S.rendersInFlight.load(std::memory_order_acquire) > 0)
    {
        S.renderCv.wait(lk);
        target = S.inactive();
    }
    target->negParams = negParams;

    Spectral::build_tables_from_curves_non_global(
        /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
        /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
        /*illumView*/ S.printRT.illumView,
        /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
        target->tablesView);

    if (hasRefIlluminant) {
        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ illumRef,
            /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
            target->tablesRef);
    }
    else {
        target->tablesRef = SpectralTables{};
    }

    if (Print::profile_is_valid(S.printRT.profile) &&
        S.printRT.illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K))
    {
        Spectral::build_tables_from_curves_non_global(
            /*epsY*/ S.printRT.profile.epsY,
            /*epsM*/ S.printRT.profile.epsM,
            /*epsC*/ S.printRT.profile.epsC,
            /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
            /*illumView*/ S.printRT.illumView,
            /*baseMin*/ S.printRT.profile.baseMin,
            /*baseMid*/ S.printRT.profile.baseMid,
            /*hasBaseline*/ S.printRT.profile.hasBaseline,
            target->tablesPrint);
    }
    else {
        target->tablesPrint = SpectralTables{};
    }

    Spectral::Curve illumScan = Spectral::build_curve_D50_pinned(S.dataDir + "Illuminants\\D50.csv");
    Spectral::build_tables_from_curves_non_global(
        /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
        /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
        /*illumView*/ illumScan,
        /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
        target->tablesScan);

    if (hasRefIlluminant && target->tablesRef.K > 0) {
        Spectral::compute_S_inverse_from_tables(target->tablesRef, target->spdSInv);
        target->spdReady = true;
    }
    else {
        target->spdReady = false;
    }

    if (target->tablesView.K <= 0) {
        JTRACE("BUILD", "tablesView not ready; will cause wsReady=0 in render.");
    }
    if (hasRefIlluminant) {
        if (!target->spdReady || target->tablesRef.K <= 0) {
            JTRACE("BUILD", "tablesRef or spdSInv not ready; SPD exposure disabled.");
        }
    }
    else {
        JTRACE("BUILD", "reference illuminant missing; SPD exposure disabled.");
    }

    {
        auto finiteCurve = [](const Spectral::Curve& c)->bool {
            for (float v : c.linear) if (!std::isfinite(v)) return false;
            return true;
            };
        const bool ok_dens = finiteCurve(densB) && finiteCurve(densG) && finiteCurve(densR);
        const bool ok_sens = finiteCurve(sensB) && finiteCurve(sensG) && finiteCurve(sensR);
        const bool ok_base = finiteCurve(baseMin) && finiteCurve(baseMid);
        const bool ok_tables =
            (target->tablesView.K == Spectral::gShape.K) &&
            (target->tablesScan.K == Spectral::gShape.K) &&
            (!target->spdReady || target->tablesRef.K == Spectral::gShape.K);

        std::ostringstream oss;
        oss << "pre-Ecal: ok_dens=" << (ok_dens ? 1 : 0)
            << " ok_sens=" << (ok_sens ? 1 : 0)
            << " ok_base=" << (ok_base ? 1 : 0)
            << " ok_tables=" << (ok_tables ? 1 : 0)
            << " spdReady=" << (target->spdReady ? 1 : 0);
        JTRACE("BUILD", oss.str());

        if (!(ok_dens && ok_sens && ok_base && ok_tables)) {
            JTRACE("BUILD", "pre-Ecal: invalid inputs; aborting rebuild to avoid crash");
            return;
        }
    }

    {
        bool ok_spd = true;
        for (int i = 0; i < 9; ++i) {
            if (!std::isfinite(target->spdSInv[i])) { ok_spd = false; break; }
        }
        const bool ok_invYn =
            std::isfinite(target->tablesView.invYn) && target->tablesView.invYn > 0.0f &&
            std::isfinite(target->tablesScan.invYn) && target->tablesScan.invYn > 0.0f &&
            (!target->spdReady || (std::isfinite(target->tablesRef.invYn) && target->tablesRef.invYn > 0.0f));

        std::ostringstream oss;
        oss << "pre-Ecal: ok_spd=" << (ok_spd ? 1 : 0)
            << " ok_invYn=" << (ok_invYn ? 1 : 0);
        JTRACE("BUILD", oss.str());

        if (!ok_spd || !ok_invYn) {
            JTRACE("BUILD", "pre-Ecal: invalid S_inv or invYn; aborting rebuild");
            return;
        }
    }

    const float offB = 0.0f;
    const float offG = 0.0f;
    const float offR = 0.0f;

    {
        std::ostringstream oss;
        oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR << " (agx parity)";
        JTRACE("BUILD", oss.str());
    }

    target->densB = std::move(densB);
    target->densG = std::move(densG);
    target->densR = std::move(densR);
    target->sensB = std::move(sensB);
    target->sensG = std::move(sensG);
    target->sensR = std::move(sensR);
    target->negSensB = std::move(sensB_beforeBalance);
    target->negSensG = std::move(sensG_beforeBalance);
    target->negSensR = std::move(sensR_beforeBalance);
    target->baseMin = std::move(baseMin);
    target->baseMid = std::move(baseMid);
    target->hasBaseline = hasBaseline;

    target->logEOffB = offB;
    target->logEOffG = offG;
    target->logEOffR = offR;

    target->dirRT = dirRT;
    target->dirPrecorrected = precorrectApplied;
    target->dMax[0] = dirRT.dMax[0];
    target->dMax[1] = dirRT.dMax[1];
    target->dMax[2] = dirRT.dMax[2];

    {
        const WorkingState* prev = S.activeWS.load(std::memory_order_acquire);
        if (prev && prev->buildCounter > 0) {
            const bool sameCouplers =
                (prev->dirRT.highShift == target->dirRT.highShift) &&
                (prev->dirRT.M[0][0] == target->dirRT.M[0][0]) &&
                (prev->dirRT.M[0][1] == target->dirRT.M[0][1]) &&
                (prev->dirRT.M[0][2] == target->dirRT.M[0][2]) &&
                (prev->dirRT.M[1][0] == target->dirRT.M[1][0]) &&
                (prev->dirRT.M[1][1] == target->dirRT.M[1][1]) &&
                (prev->dirRT.M[1][2] == target->dirRT.M[1][2]) &&
                (prev->dirRT.M[2][0] == target->dirRT.M[2][0]) &&
                (prev->dirRT.M[2][1] == target->dirRT.M[2][1]) &&
                (prev->dirRT.M[2][2] == target->dirRT.M[2][2]) &&
                (prev->dirRT.spatialSigmaMicrometers == target->dirRT.spatialSigmaMicrometers) &&
                (prev->dirRT.spatialSigmaPixels == target->dirRT.spatialSigmaPixels);

            if (sameCouplers && prev->dirPrecorrected == target->dirPrecorrected) {
                target->densB = prev->densB;
                target->densG = prev->densG;
                target->densR = prev->densR;
                target->dMax[0] = prev->dMax[0];
                target->dMax[1] = prev->dMax[1];
                target->dMax[2] = prev->dMax[2];
                target->dirRT.dMax[0] = prev->dirRT.dMax[0];
                target->dirRT.dMax[1] = prev->dirRT.dMax[1];
                target->dirRT.dMax[2] = prev->dirRT.dMax[2];
                target->negParams = prev->negParams;
            }
        }
    }

    {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float v = target->dirRT.M[r][c];
                if (!std::isfinite(v)) v = 0.0f;
                if (v < -10.0f) v = -10.0f;
                if (v > 10.0f)  v = 10.0f;
                target->dirRT.M[r][c] = v;
            }
        }
        for (int i = 0; i < 3; ++i) {
            float v = target->dMax[i];
            if (!std::isfinite(v) || v <= 1e-4f) v = 1.0f;
            if (v > 1000.0f) v = 1000.0f;
            target->dMax[i] = v;
            target->dirRT.dMax[i] = v;
        }
    }

    Spectral::set_neg_params(target->negParams);

    {
        Spectral::DirRuntimeSnapshot snap;
        snap.active = target->dirRT.active;
        snap.highShift = target->dirRT.highShift;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                snap.M[r][c] = target->dirRT.M[r][c];
            }
        }
        snap.dMax[0] = target->dirRT.dMax[0];
        snap.dMax[1] = target->dirRT.dMax[1];
        snap.dMax[2] = target->dirRT.dMax[2];
        Spectral::set_dir_runtime_snapshot(snap);
    }

    target->printRT = std::make_unique<Print::Runtime>(S.printRT);

    ++target->buildCounter;

    {
        std::ostringstream oss;
        oss << "WorkingState build #" << target->buildCounter;
        JTRACE("BUILD", oss.str());
    }

    S.activeWS.store(target, std::memory_order_release);
    {
        std::ostringstream oss;
        oss << "activeWS swapped; buildCounter=" << (long long)S.activeBuildCounter;
        JTRACE("BUILD", oss.str());
    }
    S.activeBuildCounter = target->buildCounter;
}