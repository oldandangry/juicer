#define JUICER_ENABLE_COUPLERS 1

#include <mutex>
#include <cctype>

// OpenFX 1.4 canonical headers (no non-spec feature macros)
#include "ofxCore.h"        // OfxHost, OfxPlugin, kOfxAction*
#include "ofxProperty.h"    // OfxPropertySuiteV1, property keys
#include "ofxParam.h"       // OfxParameterSuiteV1, kOfxTypeParameter
#include "ofxImageEffect.h" // OfxImageEffectSuiteV1, OfxRectI, OfxRectD, image effect props
#include "ofxPixels.h"      // Pixel/rect helpers used by some 1.4 distributions

// Resolve OFX support library C++ wrappers (OpenFX 1.4 compliant)
#pragma warning(push)
#pragma warning(disable: 5040)
#include "ofxsCore.h"
#include "ofxsImageEffect.h"
#include "ofxsParam.h"
#include "ofxsProcessing.h"
#include "ofxsMemory.h"
#include "ofxsLog.h"
#pragma warning(pop)
// Note: These headers provide the factory macros, ImageEffect base, descriptors,
// Param wrappers, Clip/Image RAII, RenderArguments, and optional processors.


#ifndef kOfxActionInstanceChanged
#error "Missing kOfxActionInstanceChanged in ofxCore.h (OpenFX 1.4)."
#endif
#ifndef kOfxPropType
#error "Missing kOfxPropType in ofxCore.h (OpenFX 1.4)."
#endif
#ifndef kOfxTypeParameter
#error "Missing kOfxTypeParameter in ofxParam.h (OpenFX 1.4)."
#endif
#include <string.h>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <array>
#include <vector>
#include "Couplers.h"
#include "SpectralTables.h"
#include "SpectralMath.h"
#include <iostream>
#include "Illuminants.h"
#include <sstream> // for diagnostics
#include <algorithm>
#include "Scanner.h"
#include "Print.h"
#include "WorkingState.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <unordered_map>
#include "mainProcessing.h"
#include "NeutralFilters.h"
#include "ProfileJSONLoader.h"
#include <utility>

// Plugin data directory (immutable)
static const std::string gDataDir =
R"(C:\Program Files\Common Files\OFX\Plugins\Juicer.ofx.bundle\Contents\Resources\)";


// --- One-time spectral global init fencing ---
namespace {
    static std::once_flag gSpectralGlobalsOnce;
    static void init_spectral_globals_once() {
        try {
            const auto cmf = Spectral::load_csv_triplets(gDataDir + "cie1931_2deg.csv");
            Spectral::initialize_spectral_shape_from_cmf_triplets(cmf);
            Spectral::set_cie_1931_2deg_cmf(cmf.xbar, cmf.ybar, cmf.zbar);
            Spectral::ensure_precomputed_up_to_date();            
            Spectral::disable_hanatos_if_shape_mismatch();
        }
        catch (...) {
            // Leave globals as-is; instance guards will pass-through if shape is invalid.
        }

        // Hanatos LUT: load once and set availability flag atomically.
        try {
            const std::string lutPath = gDataDir + "irradiance_xy_tc.npy";
            const bool lutOK = Spectral::load_hanatos_spectra_lut(lutPath);
            Spectral::gHanatosAvailable = lutOK;            
        }
        catch (...) {
            Spectral::gHanatosAvailable = false;
        }

        // KG3 filter fallback: set once if not present; safe idempotently.
        std::vector<std::pair<float, float>> kg3_pairs_raw;
        try { kg3_pairs_raw = Spectral::load_csv_pairs(gDataDir + "Filter\\KG3.csv"); }
        catch (...) { kg3_pairs_raw.clear(); }
        if (kg3_pairs_raw.empty()) {
            kg3_pairs_raw = {
                { Spectral::gShape.lambdaMin, 1.0f },
                { Spectral::gShape.lambdaMax, 1.0f }
            };
        }
        Spectral::set_filter_KG3_from_pairs(kg3_pairs_raw);
    }
}

// --- TRACE LOGGING (surgical; remove by deleting this section) ---
namespace JuicerTrace {
    inline std::mutex& mtx() {
        static std::mutex m;
        return m;
    }
    inline void write(const char* tag, const std::string& msg) {
        std::lock_guard<std::mutex> lk(mtx());
        std::ofstream f("C:/temp/juicer_trace.txt", std::ios::app);
        if (f.is_open()) {
            f << tag << " | " << msg << "\n";
        }
    }
    struct Scope {
        const char* tag;
        std::string name;
        Scope(const char* t, std::string n) : tag(t), name(std::move(n)) { write(tag, "BEGIN " + name); }
        ~Scope() { write(tag, "END   " + name); }
    };
}
#define JTRACE(tag, msg) ::JuicerTrace::write(tag, (msg))
#define JTRACE_SCOPE(tag, name) ::JuicerTrace::Scope _juicer_scope_guard_((tag), (name))

struct BaseState {
    // Immutable stock data as loaded from CSVs
    Spectral::Curve epsY, epsM, epsC;         // dye extinctions (linear OD/λ)
    Spectral::Curve sensB, sensG, sensR;      // log sensitivity curves (pinned and linearized)
    Spectral::Curve densB, densG, densR;      // density curves (logE -> D)
    Spectral::Curve baseMin, baseMid;         // negative baseline min/mid
    bool hasBaseline = false;
    std::string densitometerType;
    std::vector<float> densityMidNeutral;
    Profiles::DirCouplersProfile dirCouplers;
    Profiles::MaskingCouplersProfile maskingCouplers;
};


// Parameters snapshot (subset) to hash
struct ParamSnapshot {
    int filmStockIndex = 0;
    int printPaperIndex = 0;
    int refIll = 0;
    int viewIll = 0;
    int enlIll = 2;    
    int couplersActive = 1;
    int couplersPrecorrect = 1;
    double aR = 0.7, aG = 0.7, aB = 0.5;
    double sigma = 1.0, high = 0.0;
    int unmix = 1;
    double spatialSigma = 0.0;
};

static inline uint64_t hash_params(const ParamSnapshot& p) {
    // Simple 64-bit mix
    auto mix = [](uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); return h; };
    uint64_t h = 0;
    h = mix(h, (uint64_t)p.filmStockIndex);
    h = mix(h, (uint64_t)p.printPaperIndex);
    h = mix(h, (uint64_t)p.refIll);
    h = mix(h, (uint64_t)p.viewIll);
    h = mix(h, (uint64_t)p.enlIll);    
    h = mix(h, (uint64_t)p.couplersActive);
    h = mix(h, (uint64_t)p.couplersPrecorrect);
    h = mix(h, (uint64_t)(p.aR * 10000.0));
    h = mix(h, (uint64_t)(p.aG * 10000.0));
    h = mix(h, (uint64_t)(p.aB * 10000.0));
    h = mix(h, (uint64_t)(p.sigma * 10000.0));
    h = mix(h, (uint64_t)(p.high * 10000.0));
    h = mix(h, (uint64_t)(p.spatialSigma * 10000.0));
    h = mix(h, (uint64_t)p.unmix);
    return h;
}

struct InstanceState {
    std::mutex m;
    BaseState base;
    // Double-buffered working states
    WorkingState workA;
    WorkingState workB;
    std::atomic<WorkingState*> activeWS{ nullptr };
    uint64_t activeBuildCounter = 0;

    ParamSnapshot lastParams;
    uint64_t lastHash = 0;

    // print runtime for simulate_print (profile+illum pinned)
    Print::Runtime printRT;

    // Guard flags to suppress re-entrant changedParam during bootstrap and programmatic UI updates
    bool suppressParamEvents = false;
    bool inBootstrap = false;


    // data dir
    std::string dataDir;
    // flag indicating base is ready
    bool baseLoaded = false;

    // Helper: get inactive buffer to build into
    WorkingState* inactive() {
        WorkingState* a = activeWS.load(std::memory_order_acquire);
        return (a == &workA) ? &workB : &workA;
    }
};

// Step 1 registry: allows legacy free functions to resolve state without C suites.
namespace JuicerRegistry {
    static std::mutex MTX;
    static std::unordered_map<OfxImageEffectHandle, InstanceState*> MAP;

    inline void set(OfxImageEffectHandle h, InstanceState* s) {
        std::lock_guard<std::mutex> lk(MTX);
        MAP[h] = s;
    }
    inline InstanceState* get(OfxImageEffectHandle h) {
        std::lock_guard<std::mutex> lk(MTX);
        auto it = MAP.find(h);
        return (it == MAP.end()) ? nullptr : it->second;
    }
    inline void erase(OfxImageEffectHandle h) {
        std::lock_guard<std::mutex> lk(MTX);
        MAP.erase(h);
    }
}

// Small helper: resample a (λ, v) CSV to gShape and flatten to a single channel vector
static bool load_resampled_channel(const std::string& path,
    std::vector<float>& outChannel)
{
    try
    {
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
    catch (const std::exception& e)
    {
        std::ofstream log("C:/temp/juicer_densitometer_warning.txt", std::ios::app);
        if (log.is_open())
            log << "[load_resampled_channel] Exception: " << e.what()
            << " while loading " << path << "\n";
        return false;
    }
    catch (...)
    {
        std::ofstream log("C:/temp/juicer_densitometer_warning.txt", std::ios::app);
        if (log.is_open())
            log << "[load_resampled_channel] Unknown exception while loading "
            << path << "\n";
        return false;
    }
}

// Placeholder parameter names (Step 1)
#define kParamExposure "Exposure"   // EV
#define kParamContrast "Contrast"   // unitless
#define kParamSpectralMode "SpectralUpsampling"
#define kParamViewingIllum  "ViewingIlluminant"
#define kParamEnlargerIlluminant "EnlargerIlluminant"

// Film stock parameter
#define kParamFilmStock "FilmStock"

// Print paper parameter
#define kParamPrintPaper "PrintPaper"

// Print paper options (folders under Resources/Print)
static const char* kPrintPaperOptions[] = {
    "2383",
    "2393"
};
static constexpr int kNumPrintPapers = sizeof(kPrintPaperOptions) / sizeof(kPrintPaperOptions[0]);

static inline std::string print_dir_for_index(int index) {
    if (index < 0 || index >= kNumPrintPapers) index = 0;
    return gDataDir + std::string("Print\\") + kPrintPaperOptions[index] + "\\";
}

// Map our film stock options to agx JSON keys
static inline const char* negative_json_key_for_stock_index(int filmIndex) {
    // kFilmStockOptions: "Vision3 250D", "Vision3 50D", "Vision3 200T", "Vision3 500T"
    switch (filmIndex) {
    case 0: return "kodak_vision3_250d_uc";
    case 1: return "kodak_vision3_50d_uc";
    case 2: return "kodak_vision3_200t_uc";
    case 3: return "kodak_vision3_500t_uc";
    default: return "kodak_vision3_250d_uc";
    }
}

// Map print paper selection to agx JSON paper keys
static inline const char* print_json_key_for_paper_index(int paperIndex) {
    switch (paperIndex) {
    case 0: return "kodak_2383_uc";
    case 1: return "kodak_2393_uc";
    default: return "kodak_2383_uc";
    }
}

static std::vector<std::string> enlarger_illuminant_keys_for_choice(int choice) {
    switch (choice) {
    case 0: return { "D65", "d65" };
    case 1: return { "D50", "d50" };
    case 2: return { "TH-KG3-L", "th-kg3-l" };
    case 3: return { "EqualEnergy", "equal_energy", "Equal energy" };
    default: break;
    }
    return {};
}

// Film stock folder names (must match folder names under Resources/Stock)
static const char* kFilmStockOptions[] = {
    "Vision3 250D",
    "Vision3 50D",
    "Vision3 200T",
    "Vision3 500T"
};
static constexpr int kNumFilmStocks = sizeof(kFilmStockOptions) / sizeof(kFilmStockOptions[0]);

// Resolve stock index -> folder path under Resources/Stock
static std::string stock_dir_for_index(int index) {
    if (index < 0 || index >= kNumFilmStocks) index = 0; // default to 250D
    return gDataDir + std::string("Stock\\") + kFilmStockOptions[index] + "\\";
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

static const char* film_json_key_for_folder(const std::string& folder) {
    if (folder == "Vision3 250D") return "kodak_vision3_250d_uc";
    if (folder == "Vision3 50D") return "kodak_vision3_50d_uc";
    if (folder == "Vision3 200T") return "kodak_vision3_200t_uc";
    if (folder == "Vision3 500T") return "kodak_vision3_500t_uc";
    return nullptr;
}

/// Reload film stock assets into BaseState (no global mutation). Returns true on success.
static bool load_film_stock_into_base(const std::string& stockDir, InstanceState& S) {
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
        auto sz = [](const auto& v) { return (int)v.size(); };
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

    // Pin to current shape without touching Spectral globals
    Spectral::build_curve_on_shape_from_linear_pairs(S.base.epsY, y_data);
    Spectral::build_curve_on_shape_from_linear_pairs(S.base.epsM, m_data);
    Spectral::build_curve_on_shape_from_linear_pairs(S.base.epsC, c_data);

    Spectral::build_curve_on_shape_from_log10_pairs(S.base.sensB, b_sens);
    Spectral::build_curve_on_shape_from_log10_pairs(S.base.sensG, g_sens);
    Spectral::build_curve_on_shape_from_log10_pairs(S.base.sensR, r_sens);

    Spectral::sort_and_build(S.base.densB, dc_b);
    Spectral::sort_and_build(S.base.densG, dc_g);
    Spectral::sort_and_build(S.base.densR, dc_r);

    // Remove per-channel baseline floor (B+F) from density curves once up-front
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
        oss << "epsY/M/C K=" << (int)S.base.epsY.linear.size()
            << "/" << (int)S.base.epsM.linear.size()
            << "/" << (int)S.base.epsC.linear.size()
            << " sensB/G/R K=" << (int)S.base.sensB.linear.size()
            << "/" << (int)S.base.sensG.linear.size()
            << "/" << (int)S.base.sensR.linear.size()
            << " densB/G/R K=" << (int)S.base.densB.linear.size()
            << "/" << (int)S.base.densG.linear.size()
            << "/" << (int)S.base.densR.linear.size()
            << " baseMin/baseMid K=" << (int)S.base.baseMin.linear.size()
            << "/" << (int)S.base.baseMid.linear.size()
            << " hasBaseline=" << (S.base.hasBaseline ? 1 : 0)
            << " densType=" << (S.base.densitometerType.empty() ? std::string("none") : S.base.densitometerType);
        JTRACE("STOCK", oss.str());
    }

    JTRACE("STOCK", std::string("loaded OK; baseline=") + (S.base.hasBaseline ? "1" : "0"));

    return true;
}

// Apply all transforms from Base -> Working -> Spectral globals
static void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P) {

    // Local helper to scrub NaN/Inf and negatives from density curves
    auto sanitize_curve = [](Spectral::Curve& c) {
        for (float& v : c.linear) {
            if (!std::isfinite(v)) v = 0.0f;
            if (v < 0.0f) v = 0.0f; // enforce non-negativity
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

    std::lock_guard<std::mutex> lk(S.m);

    // INSERT AFTER std::lock_guard<std::mutex> lk(S.m);
    {
        std::ostringstream oss;
        oss << "enter with baseLoaded=" << (S.baseLoaded ? 1 : 0);
        JTRACE("BUILD", oss.str());
    }  


    // 0) Build per-instance illuminants (no global writes)
    Print::build_illuminant_from_choice(P.enlIll, S.printRT, S.dataDir, /*forEnlarger*/true);
    Print::build_illuminant_from_choice(P.viewIll, S.printRT, S.dataDir, /*forEnlarger*/false);
    {
        std::ostringstream oss;
        oss << "Enl illum K=" << (int)S.printRT.illumEnlarger.linear.size()
            << " View illum K=" << (int)S.printRT.illumView.linear.size();
        JTRACE("BUILD", oss.str());
    }


    // Local working copies start from the immutable base
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
    Spectral::Curve illumRef; // film reference illuminant used for SPD reconstruction
    bool hasRefIlluminant = false;
    const bool hasBaseline = S.base.hasBaseline;

    // 1) Optional unmix BEFORE balance (non-destructive)
    if (P.unmix) {
        JTRACE("BUILD", "unmix: entered");

        // Load Status A (or chosen) responsivities pinned to shape
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

            // Make dye spectral “densities” for crosstalk estimation from eps curves
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

            // Pack CMY density curves along logE samples (R->C, G->M, B->Y)
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


    // 2) Balance under Reference Illuminant (non-global)
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

    // Sanitize densities before computing DIR runtime and pre-warp
    sanitize_curve(densB);
    sanitize_curve(densG);
    sanitize_curve(densR);


    // 3) DIR runtime + optional pre-warp (non-global)
    bool precorrectApplied = false;
    Couplers::Runtime dirRT{};
    dirRT.active = (P.couplersActive != 0);
    {
        const float amount[3] = { (float)P.aB, (float)P.aG, (float)P.aR };
        Couplers::build_dir_matrix(dirRT.M, amount, (float)P.sigma);
        dirRT.highShift = (float)P.high;
        dirRT.spatialSigmaPixels = static_cast<float>(P.spatialSigma);

        // NOTE: dMax must reference the same-stage (balanced + optionally precorrected) curves used downstream.
        // Compute dMax AFTER optional pre-warp and monotonic enforcement to avoid tiny/invalid maxima.

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

        // Sanitize after precorrect (or no-op if precorrect is off)
        sanitize_curve(densB);
        sanitize_curve(densG);
        sanitize_curve(densR);
        JTRACE("BUILD", "precorrect: densR/G/B sanitized");

        // Ensure monotonic non-decreasing densities before mid-gray search
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

        // Compute dMax from corrected/balanced curves (parity with agx’s nanmax on current stage),
        // enforcing a small positive floor and capping absurd values to stabilize normalization.
        auto vmax = [](const Spectral::Curve& c) {
            float m = 0.0f;
            for (float v : c.linear) m = std::max(m, v);
            if (!std::isfinite(m) || m <= 1e-4f) m = 1.0f; // floor to 1.0 if tiny/invalid
            if (m > 1000.0f) m = 1000.0f;                  // cap pathological maxima
            return m;
            };
        dirRT.dMax[0] = vmax(densB); // Y
        dirRT.dMax[1] = vmax(densG); // M
        dirRT.dMax[2] = vmax(densR); // C

        {
            std::ostringstream oss;
            oss << "DIR active=" << (dirRT.active ? 1 : 0)
                << " sigma=" << (float)P.sigma << " high=" << (float)P.high
                << " dMax=" << dirRT.dMax[0] << "," << dirRT.dMax[1] << "," << dirRT.dMax[2]
                << " precorrect=" << (P.couplersPrecorrect ? 1 : 0);
            JTRACE("BUILD", oss.str());
        }
    }


    // 4) Recompute per-channel logE offsets (SPD exposure path) for midscale neutral using per-instance tables
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
    target->negParams = negParams;


    // Build per-instance spectral tables for current viewing illuminant
    Spectral::build_tables_from_curves_non_global(
        /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
        /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
        /*illumView*/ S.printRT.illumView,
        /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
        target->tablesView);

    // Build per-instance spectral tables for the film reference illuminant (SPD reconstruction)
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

    // Build scanner tables using film-negative viewing illuminant (AgX parity uses D50)
    Spectral::Curve illumScan = Spectral::build_curve_D50_pinned(S.dataDir + "Illuminants\\D50.csv");
    Spectral::build_tables_from_curves_non_global(
        /*epsY*/ epsY, /*epsM*/ epsM, /*epsC*/ epsC,
        /*xbar*/ Spectral::gXBar, /*ybar*/ Spectral::gYBar, /*zbar*/ Spectral::gZBar,
        /*illumView*/ illumScan,
        /*baseMin*/ baseMin, /*baseMid*/ baseMid, /*hasBaseline*/ hasBaseline,
        target->tablesScan);

    // Compute per-instance SPD S^-1 for reconstruction
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
        std::ostringstream oss;
        oss << "tablesView K=" << target->tablesView.K
            << " invYn=" << target->tablesView.invYn
            << " tablesRef K=" << target->tablesRef.K
            << " invYnRef=" << target->tablesRef.invYn
            << " hasRefIll=" << (hasRefIlluminant ? 1 : 0)
            << " spdReady=" << (target->spdReady ? 1 : 0);
        JTRACE("BUILD", oss.str());
    }    

    // INSERT: diagnostics and hard guards before Ecal/mid-gray
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

    // HARD GUARDS: S_inv contents and invYn validity
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
    // Agx parity: do NOT align mid-gray in the negative via per-channel logE offsets.
    // Development tone placement happens via the density curves themselves; mid-gray factor
    // is applied later in the print raw domain (print exposure compensation).

    // Set per-channel negative logE offsets to zero, keeping density curves’ native domain.
    const float offB = 0.0f;
    const float offG = 0.0f;
    const float offR = 0.0f;

    {
        std::ostringstream oss;
        oss << "negative logE offsets B/G/R=" << offB << "/" << offG << "/" << offR << " (agx parity)";
        JTRACE("BUILD", oss.str());
    }

    // 6) Snapshot data into WorkingState
    target->densB = std::move(densB);
    target->densG = std::move(densG);
    target->densR = std::move(densR);
    target->sensB = std::move(sensB);
    target->sensG = std::move(sensG);
    target->sensR = std::move(sensR);
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

    // Guard: freeze pre-warp stage within one change cycle to avoid drift.
    // If the previous WorkingState had the same coupler params hash, reuse its dens curves
    // instead of recomputing precorrect repeatedly during host-triggered dirty loops.
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
                (prev->dirRT.spatialSigmaPixels == target->dirRT.spatialSigmaPixels);

            // If coupler runtime is identical, keep precorrected curves as-is to avoid re-warp thrash
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

    // Hard scrub: ensure matrix and dMax are finite and within sane bounds
    {
        // Matrix: finite entries only
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float v = target->dirRT.M[r][c];
                if (!std::isfinite(v)) v = 0.0f;
                // cap absurd values to keep ΔlogE within design bounds
                if (v < -10.0f) v = -10.0f;
                if (v > 10.0f)  v = 10.0f;
                target->dirRT.M[r][c] = v;
            }
        }
        // dMax: enforce floor and cap again at snapshot
        for (int i = 0; i < 3; ++i) {
            float v = target->dMax[i];
            if (!std::isfinite(v) || v <= 1e-4f) v = 1.0f;
            if (v > 1000.0f) v = 1000.0f;
            target->dMax[i] = v;
            target->dirRT.dMax[i] = v; // keep rt copy coherent
        }
    }

    Spectral::gNegParams = target->negParams;

    // Publish DIR runtime snapshot for global spectral helpers (negative preview path).
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

    // Snapshot print runtime into WorkingState to avoid render-time data races
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

// === Resolve support library effect shell (Step 1) ===
// NOTE: This class centralizes instance state and defers actual IO/suite migration to Step 2.
// It delegates to your existing logic to avoid functional changes.

class JuicerEffect : public OFX::ImageEffect {
public:
    explicit JuicerEffect(OfxImageEffectHandle handle)
        : OFX::ImageEffect(handle)
    {
        // Cache clips (wrappers) for Step 2; safe even if render still uses legacy path.
        try {
            _src = fetchClip(kOfxImageEffectSimpleSourceClipName); // "Source"
            _dst = fetchClip(kOfxImageEffectOutputClipName);       // "Output"
        }
        catch (...) {
            _src = nullptr;
            _dst = nullptr;
        }

        // Cache parameter handles (wrappers)
        try {
            _pExposure = fetchDoubleParam(kParamExposure);
            _pFilmStock = fetchChoiceParam(kParamFilmStock);
            _pPrintPaper = fetchChoiceParam(kParamPrintPaper);            
            _pRefIll = fetchChoiceParam("ReferenceIlluminant");
            _pViewIll = fetchChoiceParam("ViewingIlluminant");
            _pEnlIll = fetchChoiceParam("EnlargerIlluminant");

            _pUnmix = fetchBooleanParam("UnmixDensities");

#ifdef JUICER_ENABLE_COUPLERS
            _pCouplersActive = fetchBooleanParam(Couplers::kParamCouplersActive);
            _pCouplersPrecorrect = fetchBooleanParam(Couplers::kParamCouplersPrecorrect);
            _pCouplersAmountR = fetchDoubleParam(Couplers::kParamCouplersAmountR);
            _pCouplersAmountG = fetchDoubleParam(Couplers::kParamCouplersAmountG);
            _pCouplersAmountB = fetchDoubleParam(Couplers::kParamCouplersAmountB);
            _pCouplersSigma = fetchDoubleParam(Couplers::kParamCouplersLayerSigma);
            _pCouplersHigh = fetchDoubleParam(Couplers::kParamCouplersHighExpShift);
            _pCouplersSpatialSigma = fetchDoubleParam(Couplers::kParamCouplersSpatialSigma);
#endif

            _pScanEnabled = fetchBooleanParam("ScannerEnabled");
            _pScanAuto = fetchBooleanParam("ScannerAutoExposure");
            _pScanTargetY = fetchDoubleParam("ScannerTargetY");

            _pPrintBypass = fetchBooleanParam("PrintBypass");
            _pPrintExposure = fetchDoubleParam("PrintExposure");
            _pPrintPreflash = fetchDoubleParam("PrintPreflash");
            _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
            _pEnlargerY = fetchDoubleParam("EnlargerY");
            _pEnlargerM = fetchDoubleParam("EnlargerM");            
        }
        catch (...) {
            // Safe: any missing param will remain nullptr and defaults are used in snapshot/usage paths.
        }

        // Own per-instance state
        _state = std::make_unique<InstanceState>();
        _state->dataDir = gDataDir;
        _state->activeWS.store(&_state->workA, std::memory_order_release);
        _state->activeBuildCounter = 0;

        // Optional compatibility registry
        JuicerRegistry::set(handle, _state.get());

        // Defer heavy bootstrap until first param change

    }

    ~JuicerEffect() override {
        // Mirror destroyInstance() guards without touching C suites.        
        JuicerRegistry::erase(this->getHandle());
        _state.reset();
    }    

    void render(const OFX::RenderArguments& args) override {
        // Fetch images via wrappers
        std::unique_ptr<OFX::Image> srcImg(_src ? _src->fetchImage(args.time) : nullptr);
        std::unique_ptr<OFX::Image> dstImg(_dst ? _dst->fetchImage(args.time) : nullptr);
        if (!srcImg || !dstImg) return;

        // Components and depth
        const OFX::PixelComponentEnum comps = srcImg->getPixelComponents();
        const OFX::BitDepthEnum depth = srcImg->getPixelDepth();
        if (depth != OFX::eBitDepthFloat) {
            JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
            return;
        }

        const int nComponents =
            (comps == OFX::ePixelComponentRGBA) ? 4 :
            (comps == OFX::ePixelComponentRGB) ? 3 :
            (comps == OFX::ePixelComponentAlpha) ? 1 : 0;

        if (nComponents == 0) {
            JuicerProc::copyNonFloatRect(srcImg.get(), dstImg.get());
            return;
        }

        // ROI: args.renderWindow if provided; otherwise use image bounds
        OfxRectI roi = args.renderWindow;
        if (roi.x1 == roi.x2 && roi.y1 == roi.y2) {
            roi = srcImg->getBounds();
        }
        const int width = roi.x2 - roi.x1;
        const int height = roi.y2 - roi.y1;
        if (width <= 0 || height <= 0) return;

        // Parameter reads
        double exposureSliderEV = 0.0;
        if (_pExposure) _pExposure->getValue(exposureSliderEV);
               
        float exposureSliderScale = 1.0f;
        if (std::isfinite(exposureSliderEV)) {
            exposureSliderScale = static_cast<float>(std::pow(2.0, exposureSliderEV));
        }
        // Defer composing exposureScale until after the auto-exposure block.
        float exposureScale = 1.0f;

        // Prepare branch params
        Scanner::Params scannerParams;
        {
            // Use wrappers to read scanner params
            bool scanEnabled = false, scanAuto = true;
            double scanY = 0.18;
            if (_pScanEnabled) _pScanEnabled->getValue(scanEnabled);
            if (_pScanAuto)    _pScanAuto->getValue(scanAuto);
            if (_pScanTargetY) _pScanTargetY->getValue(scanY);
            scannerParams.enabled = scanEnabled;
            scannerParams.autoExposure = scanAuto;
            scannerParams.targetY = static_cast<float>(scanY);
        }

        Print::Params printParams;
        {
            bool bypass = true;
            double pexp = 1.0, preflash = 0.0, y = 1.0, m = 1.0;
            if (_pPrintBypass)   _pPrintBypass->getValue(bypass);
            if (_pPrintExposure) _pPrintExposure->getValue(pexp);
            if (_pPrintPreflash) _pPrintPreflash->getValue(preflash);
            if (_pEnlargerY)     _pEnlargerY->getValue(y);
            if (_pEnlargerM)     _pEnlargerM->getValue(m);            
            auto clampShift = [](double v) -> double {
                if (!std::isfinite(v)) return 0.0;
                const double limit = static_cast<double>(Print::kEnlargerSteps);
                return std::clamp(v, -limit, limit);
                };
            printParams.bypass = bypass;
            printParams.exposure = static_cast<float>(pexp);
            printParams.preflashExposure = static_cast<float>(preflash);
            printParams.yFilter = static_cast<float>(clampShift(y));
            printParams.mFilter = static_cast<float>(clampShift(m));            
        }

        // --- Auto exposure compensation (camera) and print exposure compensation ---
        {
            double autoEV = 0.0;

            // Center-weighted Y measurement (agx-emulsion): sigma ≈ 0.2, aspect-normalized, target Y=0.184
            auto measure_center_weighted_Y_DWG = [&](OFX::Image* img, const OfxRectI& bounds)->double {
                if (!img) return 0.0;
                const int W = bounds.x2 - bounds.x1;
                const int H = bounds.y2 - bounds.y1;
                if (W <= 0 || H <= 0) return 0.0;

                const double sigma = 0.2;
                // Aspect-normalized coordinates, centered at 0
                auto maskAt = [&](int x, int y)->double {
                    const double nx = (double(x) + 0.5) / double(W) - 0.5;
                    const double ny = (double(y) + 0.5) / double(H) - 0.5;
                    const double normX = nx * (double(W) >= double(H) ? (double(H) / double(W)) : 1.0);
                    const double normY = ny * (double(H) >= double(W) ? (double(W) / double(H)) : 1.0);
                    const double r2 = normX * normX + normY * normY;
                    return std::exp(-r2 / (2.0 * sigma * sigma));
                    };

                // Sum mask
                double sumMask = 0.0;
                for (int yy = bounds.y1; yy < bounds.y2; ++yy)
                    for (int xx = bounds.x1; xx < bounds.x2; ++xx)
                        sumMask += maskAt(xx - bounds.x1, yy - bounds.y1);
                if (sumMask <= 0.0) return 0.0;

                // Accumulate Y using DWG→XYZ row 2 (exact DWG-linear luminance)
                double sumY = 0.0;
                for (int yy = bounds.y1; yy < bounds.y2; ++yy) {
                    for (int xx = bounds.x1; xx < bounds.x2; ++xx) {
                        const float* pix = reinterpret_cast<const float*>(img->getPixelAddress(xx, yy));
                        if (!pix) continue;
                        // Y = row 2 of DWG_RGB_to_XYZ dot rgb
                        const double Y = std::max(0.0,
                            (double)Spectral::gDWG_RGB_to_XYZ.m[3] * pix[0] +
                            (double)Spectral::gDWG_RGB_to_XYZ.m[4] * pix[1] +
                            (double)Spectral::gDWG_RGB_to_XYZ.m[5] * pix[2]);
                        const double w = maskAt(xx - bounds.x1, yy - bounds.y1);
                        sumY += Y * w;
                    }
                }
                return sumY / sumMask;
                };

            // Camera auto exposure compensation: adjust scene EV to reach target Y=0.184
            if (scannerParams.autoExposure) {
                OfxRectI fullBounds = srcImg->getBounds();
                const double Yexp = measure_center_weighted_Y_DWG(srcImg.get(), fullBounds);
                // We will use ScannerTargetY as the target; fallback to 0.184 if invalid (agx parity).
                double targetY = 0.184;
                if (std::isfinite(scannerParams.targetY) && scannerParams.targetY > 0.0) {
                    targetY = static_cast<double>(scannerParams.targetY);
                }                
                double evComp = 0.0;
                if (Yexp > 0.0 && targetY > 0.0) {
                    const double exposureRatio = Yexp / targetY;
                    evComp = -std::log(exposureRatio) / std::log(2.0);
                }
                if (!std::isfinite(evComp)) evComp = 0.0;
                autoEV = evComp;
            }
            // Compose total camera EV (agx parity): autoEV + slider EV
            double sliderEV = std::isfinite(exposureSliderEV) ? exposureSliderEV : 0.0;
            const double totalEV = autoEV + sliderEV;
            exposureScale = static_cast<float>(std::pow(2.0, totalEV));
        }

#ifdef JUICER_ENABLE_COUPLERS
        Couplers::Runtime dirRT{};
        if (_state && _state->baseLoaded) {
            const WorkingState* wsCur = _state->activeWS.load(std::memory_order_acquire);
            if (wsCur && wsCur->buildCounter > 0) {
                dirRT = wsCur->dirRT;
                // Safety: ensure dMax are positive and finite (agx uses nanmax per channel)
                for (int i = 0; i < 3; ++i) {
                    float v = dirRT.dMax[i];
                    if (!std::isfinite(v) || v <= 0.0f) v = 1.0f;
                    if (v > 1000.0f) v = 1000.0f;
                    dirRT.dMax[i] = v;
                }
                // Safety: scrub matrix entries before per-pixel math (only once)
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        float m = dirRT.M[r][c];
                        if (!std::isfinite(m)) m = 0.0f;
                        if (m < -10.0f) m = -10.0f;
                        if (m > 10.0f)  m = 10.0f;
                        dirRT.M[r][c] = m;
                    }
                }
            }
            // Cap spatial sigma to a sane maximum to avoid impractical kernels
            if (!std::isfinite(dirRT.spatialSigmaPixels) || dirRT.spatialSigmaPixels < 0.0f)
                dirRT.spatialSigmaPixels = 0.0f;
            if (dirRT.spatialSigmaPixels > 25.0f)
                dirRT.spatialSigmaPixels = 25.0f;
        }
#else
        Couplers::Runtime dirRT{};
#endif


        const WorkingState* ws = (_state && _state->baseLoaded)
            ? _state->activeWS.load(std::memory_order_acquire)
            : nullptr;
        const Print::Runtime* prt = (ws && ws->buildCounter > 0 && ws->printRT) ? ws->printRT.get() : nullptr;
        

        const bool wsReady = (ws && ws->buildCounter > 0 &&
            ws->tablesView.K == Spectral::gShape.K &&
            ws->tablesView.epsY.size() == static_cast<size_t>(Spectral::gShape.K) &&
            ws->tablesView.epsM.size() == static_cast<size_t>(Spectral::gShape.K) &&
            ws->tablesView.epsC.size() == static_cast<size_t>(Spectral::gShape.K) &&
            ws->baseMin.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&            
            !ws->densB.lambda_nm.empty() && !ws->densB.linear.empty() &&
            !ws->densG.lambda_nm.empty() && !ws->densG.linear.empty() &&
            !ws->densR.lambda_nm.empty() && !ws->densR.linear.empty());

        const bool printReady =
            (prt != nullptr) &&
            Print::profile_is_valid(prt->profile) &&
            prt->illumView.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
            prt->illumEnlarger.linear.size() == static_cast<size_t>(Spectral::gShape.K) &&
            wsReady;
        if (!printReady) {
            if (prt && !Print::profile_is_valid(prt->profile)) {
                JTRACE("PRINT", "profile invalid: missing print sensitivities or curves; bypassing print path");
            }
            else if (prt) {
                JTRACE("PRINT", "print runtime invalid: illuminants not pinned or working tables not ready; bypassing print path");
            }
        }


        // --- Print exposure compensation via spectral mid-gray probe (agx parity) ---
        {
            bool printComp = false;
            if (_pPrintExposureComp) { bool pc = false; _pPrintExposureComp->getValue(pc); printComp = pc; }

            printParams.exposureCompensationEnabled = printComp;
            printParams.exposureCompensationScale = printComp ? exposureSliderScale : 1.0f;            
        }

        // Tile-based multithreaded processing via OFX::ImageProcessor
        JuicerProcessor proc(*this);
        proc.setSrcDst(srcImg.get(), dstImg.get());
        proc.setComponents(nComponents);
        proc.setScannerParams(scannerParams);
        proc.setPrintParams(printParams);
        proc.setDirRuntime(dirRT);
        proc.setWorkingState(ws, wsReady);
        proc.setPrintRuntime(prt, printReady);
        proc.setExposure(exposureScale);
        proc.setRenderWindowRect(roi);
        proc.setGPURenderArgs(args);
        // Dispatch to support library's threaded/tiled CPU path
        proc.process();
    }


    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override {
        // Suppress recursion while we are programmatically setting params
        if (_state && _state->suppressParamEvents) {
            JTRACE("BUILD", std::string("changedParam suppressed for '") + paramName + "'");
            return;
        }
        if (_state && _state->inBootstrap) {
            JTRACE("BUILD", std::string("changedParam ignored during bootstrap for '") + paramName + "'");
            return;
        }
        onParamsPossiblyChanged(paramName.c_str());
    }

private:
    OFX::Clip* _src = nullptr;
    OFX::Clip* _dst = nullptr;

    // Cached params (wrappers)
    OFX::DoubleParam* _pExposure = nullptr;
    OFX::ChoiceParam* _pFilmStock = nullptr;
    OFX::ChoiceParam* _pPrintPaper = nullptr;    
    OFX::ChoiceParam* _pRefIll = nullptr;
    OFX::ChoiceParam* _pViewIll = nullptr;
    OFX::ChoiceParam* _pEnlIll = nullptr;

    OFX::BooleanParam* _pUnmix = nullptr;

#ifdef JUICER_ENABLE_COUPLERS
    OFX::BooleanParam* _pCouplersActive = nullptr;
    OFX::BooleanParam* _pCouplersPrecorrect = nullptr;
    OFX::DoubleParam* _pCouplersAmountR = nullptr;
    OFX::DoubleParam* _pCouplersAmountG = nullptr;
    OFX::DoubleParam* _pCouplersAmountB = nullptr;
    OFX::DoubleParam* _pCouplersSigma = nullptr;
    OFX::DoubleParam* _pCouplersHigh = nullptr;
    OFX::DoubleParam* _pCouplersSpatialSigma = nullptr;
#endif

    // Scanner and print params
    OFX::BooleanParam* _pScanEnabled = nullptr;
    OFX::BooleanParam* _pScanAuto = nullptr;
    OFX::DoubleParam* _pScanTargetY = nullptr;

    OFX::BooleanParam* _pPrintBypass = nullptr;
    OFX::DoubleParam* _pPrintExposure = nullptr;
    OFX::DoubleParam* _pPrintPreflash = nullptr;
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;    

    std::unique_ptr<InstanceState> _state; // owned per instance

    ParamSnapshot snapshotParams() const {
        ParamSnapshot P;
        if (_pFilmStock)      _pFilmStock->getValue(P.filmStockIndex);
        if (_pPrintPaper)     _pPrintPaper->getValue(P.printPaperIndex);
        if (_pRefIll)         _pRefIll->getValue(P.refIll);
        if (_pViewIll)        _pViewIll->getValue(P.viewIll);
        if (_pEnlIll)         _pEnlIll->getValue(P.enlIll);        
        if (_pUnmix) { bool v = true; _pUnmix->getValue(v); P.unmix = v ? 1 : 0; }
#ifdef JUICER_ENABLE_COUPLERS
        if (_pCouplersActive) { bool v = true; _pCouplersActive->getValue(v); P.couplersActive = v ? 1 : 0; }
        if (_pCouplersPrecorrect) { bool v = true; _pCouplersPrecorrect->getValue(v); P.couplersPrecorrect = v ? 1 : 0; }
        if (_pCouplersAmountR)    _pCouplersAmountR->getValue(P.aR);
        if (_pCouplersAmountG)    _pCouplersAmountG->getValue(P.aG);
        if (_pCouplersAmountB)    _pCouplersAmountB->getValue(P.aB);
        if (_pCouplersSigma)      _pCouplersSigma->getValue(P.sigma);
        if (_pCouplersHigh)       _pCouplersHigh->getValue(P.high);
        double spatial = 0.0;
        if (_pCouplersSpatialSigma) _pCouplersSpatialSigma->getValue(spatial);
        P.spatialSigma = spatial;
#endif
        return P;
    }

    void onParamsPossiblyChanged(const char* changedNameOrNull);
    void bootstrap_after_attach();

    void applyNeutralFilters(const ParamSnapshot& P, bool resetFilterParams, bool ensureExposureComp);
        
};

void JuicerEffect::bootstrap_after_attach() {
    // Initialize Spectral globals exactly once per process.
    std::call_once(gSpectralGlobalsOnce, init_spectral_globals_once);
    JTRACE("BUILD", "spectral globals ensured once; proceeding to profile and film stock load");
    // Suppress re-entrant param events during bootstrap
    _state->inBootstrap = true;
    _state->suppressParamEvents = true;


    _state->printRT = Print::Runtime{};
    ParamSnapshot P = snapshotParams();

    // Load selected print paper profile
    const std::string printDir = print_dir_for_index(P.printPaperIndex);
    const char* paperKey = print_json_key_for_paper_index(P.printPaperIndex);
    const std::string printProfileJson = paperKey
        ? gDataDir + std::string("profiles\\") + paperKey + std::string(".json")
        : std::string();
    Print::load_profile_from_dir(printDir, _state->printRT.profile, printProfileJson);

    // Illuminants from current params
    Print::build_illuminant_from_choice(P.enlIll, _state->printRT, _state->dataDir, /*forEnlarger*/true);
    Print::build_illuminant_from_choice(P.viewIll, _state->printRT, _state->dataDir, /*forEnlarger*/false);

    // Load dichroic filters (Durst Digital Light by default)
    try {
        const std::string durstDir = _state->dataDir + "Filter\\dichroics\\durst_digital_light\\";
        Print::load_dichroic_filters_from_csvs(durstDir, _state->printRT);
    }
    catch (...) {
        // Identity fallback is already handled in loader via 1.0 curves
        JuicerTrace::write("PRINT", "DICHROICS: exception during load; using identity filters");
    }

    applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/true);    


    // Film stock load + initial rebuild
    const std::string stockDir = stock_dir_for_index(P.filmStockIndex);
    _state->baseLoaded = load_film_stock_into_base(stockDir, *_state);

    if (_state->baseLoaded) {
        rebuild_working_state(this->getHandle(), *_state, P);
        _state->lastParams = P;
        _state->lastHash = hash_params(P);
    }
    else {
        JTRACE("STOCK", "bootstrap: failed to load film stock; deferring rebuild");
    }

    // Re-enable changedParam handling now that bootstrap is complete
    _state->suppressParamEvents = false;
    _state->inBootstrap = false;
}

void JuicerEffect::applyNeutralFilters(const ParamSnapshot& P, bool resetFilterParams, bool ensureExposureComp) {
    if (!_state) {
        return;
    }

    const char* paperKey = print_json_key_for_paper_index(P.printPaperIndex);
    const char* negativeKey = negative_json_key_for_stock_index(P.filmStockIndex);
    const std::vector<std::string> illumKeys = enlarger_illuminant_keys_for_choice(P.enlIll);

    float neutralY = Print::kDefaultNeutralY;
    float neutralM = Print::kDefaultNeutralM;
    float neutralC = Print::kDefaultNeutralC;
    bool loaded = false;

    if (paperKey && negativeKey && !illumKeys.empty()) {
        const std::string jsonPath = gDataDir + std::string("Print\\enlarger_neutral_ymc_filters.json");
        std::tuple<float, float, float> ymc{};
        for (const std::string& illumKey : illumKeys) {
            if (illumKey.empty()) {
                continue;
            }
            if (load_enlarger_neutral_filters(jsonPath, paperKey, illumKey, negativeKey, ymc)) {
                neutralY = std::clamp(std::get<0>(ymc), 0.0f, 1.0f);
                neutralM = std::clamp(std::get<1>(ymc), 0.0f, 1.0f);
                neutralC = std::clamp(std::get<2>(ymc), 0.0f, 1.0f);
                loaded = true;
                JTRACE("PRINT", "Neutral filters loaded for " + std::string(illumKey)
                    + " Y/M/C=" + std::to_string(neutralY) + "/" + std::to_string(neutralM)
                    + "/" + std::to_string(neutralC));
                break;
            }
        }
    }

    if (!loaded) {
        JTRACE("PRINT", "Neutral filters missing for illuminant; using Durst defaults Y/M/C="
            + std::to_string(neutralY) + "/" + std::to_string(neutralM) + "/" + std::to_string(neutralC));
    }

    _state->printRT.neutralY = neutralY;
    _state->printRT.neutralM = neutralM;
    _state->printRT.neutralC = neutralC;

    if (resetFilterParams) {
        const bool wasSuppressed = _state->suppressParamEvents;
        _state->suppressParamEvents = true;
        if (_pEnlargerY) _pEnlargerY->setValue(0.0);
        if (_pEnlargerM) _pEnlargerM->setValue(0.0);        
        _state->suppressParamEvents = wasSuppressed;
    }

    if (ensureExposureComp && _pPrintExposureComp) {
        bool exposureToggle = false;
        _pPrintExposureComp->getValue(exposureToggle);
        if (!exposureToggle) {
            const bool wasSuppressed = _state->suppressParamEvents;
            _state->suppressParamEvents = true;
            _pPrintExposureComp->setValue(true);
            _state->suppressParamEvents = wasSuppressed;
        }
    }
}

void JuicerEffect::onParamsPossiblyChanged(const char* changedNameOrNull) {
    if (!_state) return;
    // Suppress re-entrant param handling while programmatic changes are in flight
    if (_state->suppressParamEvents) {
        JTRACE("BUILD", "onParamsPossiblyChanged suppressed");
        return;
    }

    // If bootstrap hasn’t run yet, run it once now
    if (_state->lastHash == 0 && !_state->baseLoaded) {
        bootstrap_after_attach();
    }

    ParamSnapshot P = snapshotParams();
    const uint64_t h = hash_params(P);

    // Reload print profile if print paper changed
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamPrintPaper) == 0) {
        const std::string printDir = print_dir_for_index(P.printPaperIndex);
        const char* paperKey = print_json_key_for_paper_index(P.printPaperIndex);
        const std::string printProfileJson = paperKey
            ? gDataDir + std::string("profiles\\") + paperKey + std::string(".json")
            : std::string();
        Print::load_profile_from_dir(printDir, _state->printRT.profile, printProfileJson);

        // Reload dichroic filters (keep vendor default; can be parameterized later)
        try {
            const std::string durstDir = _state->dataDir + "Filter\\dichroics\\durst_digital_light\\";
            Print::load_dichroic_filters_from_csvs(durstDir, _state->printRT);
        }
        catch (...) {
            JuicerTrace::write("PRINT", "DICHROICS: reload failed; identity filters");
        }

        applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/false);

        // Force working state snapshot to carry updated printRT profile into render
        if (_state->baseLoaded) {
            rebuild_working_state(this->getHandle(), *_state, P);
            _state->lastParams = P;
            _state->lastHash = hash_params(P);
        }
    }

    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamEnlargerIlluminant) == 0) {
        applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/false);
    }

    // Reload BaseState if stock changed
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamFilmStock) == 0) {
        const std::string stockDir = stock_dir_for_index(P.filmStockIndex);
        _state->baseLoaded = load_film_stock_into_base(stockDir, *_state);
    }
    // If film stock changed, also re-apply neutral filters for current print paper
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamFilmStock) == 0 && _state->baseLoaded) {
        applyNeutralFilters(P, /*resetFilterParams*/true, /*ensureExposureComp*/false);
        
    }
    else if (P.filmStockIndex != _state->lastParams.filmStockIndex) {
        const std::string stockDir = stock_dir_for_index(P.filmStockIndex);
        _state->baseLoaded = load_film_stock_into_base(stockDir, *_state);
    }

    // Rebuild if any effective param changed
    if (_state->baseLoaded && h != _state->lastHash) {
        rebuild_working_state(this->getHandle(), *_state, P);
        _state->lastParams = P;
        _state->lastHash = h;
    }    

#ifdef JUICER_ENABLE_COUPLERS
    if (changedNameOrNull) {
        using namespace Couplers;
        const bool isCouplerParam =
            std::strcmp(changedNameOrNull, kParamCouplersActive) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersPrecorrect) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmountR) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmountG) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersAmountB) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersLayerSigma) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersHighExpShift) == 0 ||
            std::strcmp(changedNameOrNull, kParamCouplersSpatialSigma) == 0;

        if (isCouplerParam) {
            // Avoid cascading rebuild loops: mark mixing dirty only if hash actually changes.
            ParamSnapshot Pnew = snapshotParams();
            const uint64_t hnew = hash_params(Pnew);
            if (hnew != _state->lastHash) {
                Couplers::on_param_changed(changedNameOrNull);
            }
        }
    }
#endif    

}

// === Resolve support library factory (Step 1) ===
// The factory owns the plugin identity and wires descriptor/instance creation.

// === Resolve support library factory (Resolve pattern) ===
// Define a factory that derives from PluginFactoryHelper and wires ID/version.
#define kPluginIdentifier "com.juicer.Juicer"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

class JuicerPluginFactory : public OFX::PluginFactoryHelper<JuicerPluginFactory> {
public:
    JuicerPluginFactory()
        : OFX::PluginFactoryHelper<JuicerPluginFactory>(kPluginIdentifier,
            kPluginVersionMajor,
            kPluginVersionMinor) {
    }

    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override;
    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context) override;
};


void JuicerPluginFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    // Label/group
    desc.setLabels("Juicer", "Juicer", "Juicer");
    desc.setPluginGrouping("Negative-juice");

    // Contexts
    desc.addSupportedContext(OFX::eContextFilter);

    // Pixel depths
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    // Flags
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(true);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);
}

void JuicerPluginFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context)
{
    if (context != OFX::eContextFilter) return;

    // Clips
    {
        OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
        src->addSupportedComponent(OFX::ePixelComponentRGBA);
        src->addSupportedComponent(OFX::ePixelComponentRGB);
        src->setSupportsTiles(true);
        src->setOptional(false);
    }
    {
        OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
        dst->addSupportedComponent(OFX::ePixelComponentRGBA);
        dst->addSupportedComponent(OFX::ePixelComponentRGB);
        dst->setSupportsTiles(true);
    }

    // Parameters — mirror current define semantics (names, defaults, ranges)
    // Exposure
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(kParamExposure);
        p->setLabel("Exposure");
        p->setDefault(0.0);
        p->setRange(-8.0, 8.0);
        p->setDisplayRange(-4.0, 4.0);
    }    

    // Film stock (choice)
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamFilmStock);
        p->setLabel("Film stock");
        for (int i = 0; i < kNumFilmStocks; ++i) p->appendOption(kFilmStockOptions[i]);
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }
    // Spectral upsampling
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamSpectralMode);
        p->setLabel("Spectral upsampling");
        p->appendOption("Hanatos");        
        p->setDefault(0);
    }    
    // Unmix densities
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("UnmixDensities");
        p->setLabel("Unmix densities");
        p->setDefault(true);
    }

#ifdef JUICER_ENABLE_COUPLERS
    // Couplers (DIR) parameters — wrapper descriptors matching Couplers::define_params
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(Couplers::kParamCouplersActive);
        p->setLabel("DIR couplers");
        p->setDefault(true);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(Couplers::kParamCouplersPrecorrect);
        p->setLabel("Precorrect density curves");
        p->setDefault(true);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountR);
        p->setLabel("Couplers amount R");
        p->setDefault(0.7);
        p->setRange(0.0, 1.0);
        p->setDisplayRange(0.0, 1.0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountG);
        p->setLabel("Couplers amount G");
        p->setDefault(0.7);
        p->setRange(0.0, 1.0);
        p->setDisplayRange(0.0, 1.0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountB);
        p->setLabel("Couplers amount B");
        p->setDefault(0.5);
        p->setRange(0.0, 1.0);
        p->setDisplayRange(0.0, 1.0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersLayerSigma);
        p->setLabel("Layer diffusion");
        p->setDefault(1.0);
        p->setRange(0.0, 3.0);
        p->setDisplayRange(0.0, 3.0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersHighExpShift);
        p->setLabel("High exposure shift");
        p->setDefault(0.0);
        p->setRange(0.0, 1.0);
        p->setDisplayRange(0.0, 1.0);
        p->setEvaluateOnChange(true);
    }

    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersSpatialSigma);
        p->setLabel("Couplers spatial diffusion");
        p->setDefault(0.0);
        p->setRange(0.0, 15.0);
        p->setDisplayRange(0.0, 15.0);
        p->setEvaluateOnChange(true);
    }

#endif

    // Scanner group
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("ScannerEnabled");
        p->setLabel("Scanner enabled");
        p->setDefault(false);
    }
    {
        OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("ScannerAutoExposure");
        p->setLabel("Auto exposure compensation");
        p->setDefault(true);
    }
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("ScannerTargetY");
        p->setLabel("Scanner target Y");
        p->setDefault(0.184);
        p->setDisplayRange(0.01, 1.0);
    }

    // Print group
    {
        OFX::GroupParamDescriptor* grpPrint = nullptr;
        {
            grpPrint = desc.defineGroupParam("PrintGroup");
            grpPrint->setLabel("Print");
        }
        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamPrintPaper);
            p->setLabel("Print paper");
            // Options map to folder names and JSON keys
            p->appendOption("2383");
            p->appendOption("2393");
            p->setDefault(0);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("PrintBypass");
            p->setLabel("Bypass print");
            p->setDefault(true);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("PrintExposure");
            p->setLabel("Print exposure");
            p->setDefault(1.0);
            p->setDisplayRange(0.1, 10.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("PrintPreflash");
            p->setLabel("Print preflash");
            p->setDefault(0.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("PrintExposureCompensation");
            p->setLabel("Print exposure compensation");
            p->setDefault(true);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerY");
            p->setLabel("Enlarger Y");
            p->setDefault(0.0);
            p->setDisplayRange(-Print::kEnlargerSteps, Print::kEnlargerSteps);
            p->setIncrement(1.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerM");
            p->setLabel("Enlarger M");
            p->setDefault(0.0);
            p->setDisplayRange(-Print::kEnlargerSteps, Print::kEnlargerSteps);
            p->setIncrement(1.0);
            if (grpPrint) p->setParent(*grpPrint);
        }        
    }

    // Illuminants
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("ReferenceIlluminant");
        p->setLabel("Reference illuminant");
        p->appendOption("D65");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("Equal energy");
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("EnlargerIlluminant");
        p->setLabel("Enlarger illuminant");
        p->appendOption("D65");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("Equal energy");
        p->setDefault(2);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("ViewingIlluminant");
        p->setLabel("Viewing illuminant");
        p->appendOption("D65");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("Equal energy");
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }
}

OFX::ImageEffect* JuicerPluginFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum /*context*/)
{
    // Create our effect instance (constructor attaches InstanceState + runs bootstrap).
    return new JuicerEffect(handle);
}

// Resolve support library entry: register our factory.
void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& arr) {
    static JuicerPluginFactory factory;
    arr.push_back(&factory);
}
