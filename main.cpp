#define JUICER_ENABLE_COUPLERS 1

#include <mutex>

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
#include "Couplers.h"
#include "SpectralTables.h"
#include "SpectralMath.h"
#include <iostream>
#include "Illuminants.h"
#include <sstream> // for diagnostics
#include <array>
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
            // Do NOT flip gSpectralUpsamplingMode globally here; instances carry spectralMode themselves.
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
};


// Parameters snapshot (subset) to hash
struct ParamSnapshot {
    int filmStockIndex = 0;
    int printPaperIndex = 0;
    int refIll = 0;
    int viewIll = 0;
    int enlIll = 2;
    int spectralMode = 0;   // 0 Hanatos, 1 CMF-basis
    int exposureModel = 1;  // 0 Matrix, 1 SPD
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
    h = mix(h, (uint64_t)p.spectralMode);
    h = mix(h, (uint64_t)p.exposureModel);
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
#define kParamSpectralMode "SpectralUpsampling" // 0: Hanatos, 1: CMF-basis
#define kParamExposureModel "ExposureModel" // 0: Matrix, 1: SPD
#define kParamViewingIllum  "ViewingIlluminant"

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

// Safe CSV loader that fails silently (no exceptions)
static std::vector<std::pair<float, float>> load_pairs_silent(const std::string& path) {
    try {
        return Spectral::load_csv_pairs(path);
    }
    catch (...) {
        return {};
    }
}

/// Reload film stock assets into BaseState (no global mutation). Returns true on success.
static bool load_film_stock_into_base(const std::string& stockDir, InstanceState& S) {
    JTRACE_SCOPE("STOCK", std::string("load_film_stock_into_base: ") + stockDir);   

    auto y_data = load_pairs_silent(stockDir + "dye_density_y.csv");
    auto m_data = load_pairs_silent(stockDir + "dye_density_m.csv");
    auto c_data = load_pairs_silent(stockDir + "dye_density_c.csv");

    auto b_sens = load_pairs_silent(stockDir + "log_sensitivity_b.csv");
    auto g_sens = load_pairs_silent(stockDir + "log_sensitivity_g.csv");
    auto r_sens = load_pairs_silent(stockDir + "log_sensitivity_r.csv");

    auto dmin = load_pairs_silent(stockDir + "dye_density_min.csv");
    auto dmid = load_pairs_silent(stockDir + "dye_density_mid.csv");

    auto dc_b = load_pairs_silent(stockDir + "density_curve_b.csv");
    auto dc_g = load_pairs_silent(stockDir + "density_curve_g.csv");
    auto dc_r = load_pairs_silent(stockDir + "density_curve_r.csv");

    {
        auto sz = [](const auto& v) { return (int)v.size(); };
        std::ostringstream oss;
        oss << "y/m/c=" << sz(y_data) << "/" << sz(m_data) << "/" << sz(c_data)
            << " sens b/g/r=" << sz(b_sens) << "/" << sz(g_sens) << "/" << sz(r_sens)
            << " dens b/g/r=" << sz(dc_b) << "/" << sz(dc_g) << "/" << sz(dc_r)
            << " base min/mid=" << sz(dmin) << "/" << sz(dmid);
        JTRACE("STOCK", oss.str());
    }

    if (y_data.empty()) JTRACE("STOCK", "dye_density_y.csv missing/empty");
    if (m_data.empty()) JTRACE("STOCK", "dye_density_m.csv missing/empty");
    if (c_data.empty()) JTRACE("STOCK", "dye_density_c.csv missing/empty");
    if (b_sens.empty()) JTRACE("STOCK", "log_sensitivity_b.csv missing/empty");
    if (g_sens.empty()) JTRACE("STOCK", "log_sensitivity_g.csv missing/empty");
    if (r_sens.empty()) JTRACE("STOCK", "log_sensitivity_r.csv missing/empty");
    if (dc_b.empty()) JTRACE("STOCK", "density_curve_b.csv missing/empty");
    if (dc_g.empty()) JTRACE("STOCK", "density_curve_g.csv missing/empty");
    if (dc_r.empty()) JTRACE("STOCK", "density_curve_r.csv missing/empty");

    const bool okCore =
        !y_data.empty() && !m_data.empty() && !c_data.empty() &&
        !b_sens.empty() && !g_sens.empty() && !r_sens.empty() &&
        !dc_b.empty() && !dc_g.empty() && !dc_r.empty();
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
            << " hasBaseline=" << (S.base.hasBaseline ? 1 : 0);
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
        auto load_resp = [&](const std::string& rel)->std::vector<float> {
            std::vector<float> out;
            (void)load_resampled_channel(S.dataDir + rel, out);
            JTRACE("BUILD", std::string("unmix: loaded ") + rel + " size=" + std::to_string(out.size()));
            return out;
            };
        std::vector<float> respB = load_resp("densitometer\\responsivity_b.csv");
        std::vector<float> respG = load_resp("densitometer\\responsivity_g.csv");
        std::vector<float> respR = load_resp("densitometer\\responsivity_r.csv");

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
                densitometerResp[0][i] = respB[i];
                densitometerResp[1][i] = respG[i];
                densitometerResp[2][i] = respR[i];
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
    WorkingState* target = S.inactive();


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

    // Calibrate print profile per-channel logE offsets to viewing axis (agx parity)
    if (Print::profile_is_valid(S.printRT.profile)) {
        // Use print paper sensitivity under viewing Ybar to place mids correctly.
        Print::calibrate_print_logE_offsets_from_profile(target->tablesView, S.printRT.profile);
        std::ostringstream oss;
        oss << "calibrated print logE offsets Y/M/C="
            << S.printRT.profile.logEOffY << "/"
            << S.printRT.profile.logEOffM << "/"
            << S.printRT.profile.logEOffC;
        JTRACE("BUILD", oss.str());
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

    // Honor print profile logE offsets as loaded from disk (parity with agx).
    if (Print::profile_is_valid(S.printRT.profile)) {
        std::ostringstream oss;
        oss << "print logE offsets (profile) Y/M/C="
            << S.printRT.profile.logEOffY << "/"
            << S.printRT.profile.logEOffM << "/"
            << S.printRT.profile.logEOffC;
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

    target->spectralMode = std::clamp(P.spectralMode, 0, 1);
    target->exposureModel = P.exposureModel;


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
            _pSpectralMode = fetchChoiceParam(kParamSpectralMode);
            _pExposureModel = fetchChoiceParam(kParamExposureModel);
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
            _pPrintExposureComp = fetchBooleanParam("PrintExposureCompensation");
            _pEnlargerY = fetchDoubleParam("EnlargerY");
            _pEnlargerM = fetchDoubleParam("EnlargerM");
            _pEnlargerC = fetchDoubleParam("EnlargerC");
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
            double pexp = 1.0, y = 1.0, m = 1.0, c = 1.0;
            if (_pPrintBypass)   _pPrintBypass->getValue(bypass);
            if (_pPrintExposure) _pPrintExposure->getValue(pexp);
            if (_pEnlargerY)     _pEnlargerY->getValue(y);
            if (_pEnlargerM)     _pEnlargerM->getValue(m);
            if (_pEnlargerC)     _pEnlargerC->getValue(c);
            printParams.bypass = bypass;
            printParams.exposure = static_cast<float>(pexp);
            printParams.yFilter = static_cast<float>(y);
            printParams.mFilter = static_cast<float>(m);
            printParams.cFilter = static_cast<float>(c);
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
            const double totalEV = autoEV + exposureSliderEV;
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


        // --- Print exposure compensation via mid-gray factor from Exposure slider EV (agx parity) ---
        {
            bool printComp = false;
            if (_pPrintExposureComp) { bool pc = false; _pPrintExposureComp->getValue(pc); printComp = pc; }

            if (printComp && printReady && !printParams.bypass) {
                // Use camera exposure compensation EV (slider) only to derive the mid-gray factor under neutral filters.
                double exposureCompEV = 0.0;
                if (_pExposure) _pExposure->getValue(exposureCompEV);

                const WorkingState* wsCur = ws;
                const Print::Runtime* prtCur = prt;

                const float factorG = compute_midgray_factor_green_neutral(*_state, wsCur, prtCur, exposureCompEV);

                // Agx parity: apply mid-gray factor via green channel only.
                printParams.exposureCompGFactor = factorG;

                // Diagnostics
                JTRACE("PRINT", std::string("mid-gray comp factorG=") + std::to_string(factorG));
            }
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
        onParamsPossiblyChanged(paramName.c_str());
    }

private:
    OFX::Clip* _src = nullptr;
    OFX::Clip* _dst = nullptr;

    // Cached params (wrappers)
    OFX::DoubleParam* _pExposure = nullptr;
    OFX::ChoiceParam* _pFilmStock = nullptr;
    OFX::ChoiceParam* _pPrintPaper = nullptr;
    OFX::ChoiceParam* _pSpectralMode = nullptr;
    OFX::ChoiceParam* _pExposureModel = nullptr;
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
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;
    OFX::DoubleParam* _pEnlargerC = nullptr;

    std::unique_ptr<InstanceState> _state; // owned per instance

    ParamSnapshot snapshotParams() const {
        ParamSnapshot P;
        if (_pFilmStock)      _pFilmStock->getValue(P.filmStockIndex);
        if (_pPrintPaper)     _pPrintPaper->getValue(P.printPaperIndex);
        if (_pRefIll)         _pRefIll->getValue(P.refIll);
        if (_pViewIll)        _pViewIll->getValue(P.viewIll);
        if (_pEnlIll)         _pEnlIll->getValue(P.enlIll);
        if (_pSpectralMode)   _pSpectralMode->getValue(P.spectralMode);
        if (_pExposureModel)  _pExposureModel->getValue(P.exposureModel);
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

    // Insert directly above: void onParamsPossiblyChanged(const char* changedNameOrNull);
    void fitNeutralEnlargerFilters() {
        if (!_state || !_state->baseLoaded) return;
        const WorkingState* ws = _state->activeWS.load(std::memory_order_acquire);
        const Print::Runtime* prt = (ws && ws->buildCounter > 0 && ws->printRT) ? ws->printRT.get() : nullptr;
        if (!ws || !prt) return;
        if (!Print::profile_is_valid(prt->profile)) return;
        if (prt->illumView.linear.size() != (size_t)Spectral::gShape.K ||
            prt->illumEnlarger.linear.size() != (size_t)Spectral::gShape.K) return;

        // Start from current UI values
        double y = 1.0, m = 1.0, c = 1.0, pexp = 1.0;
        if (_pPrintExposure) _pPrintExposure->getValue(pexp);
        if (_pEnlargerY) _pEnlargerY->getValue(y);
        if (_pEnlargerM) _pEnlargerM->getValue(m);
        if (_pEnlargerC) _pEnlargerC->getValue(c);

        // Force YM-only fitting parity: use current neutral cyan from UI (set via JSON), not hardcoded.
        if (_pEnlargerC) { double cN = 0.35; _pEnlargerC->getValue(cN); c = cN; }

        // Use a mid-grey DWG input for fitting
        const float rgbMid[3] = { 0.18f, 0.18f, 0.18f };

        // Iterative ratio balancing over a few passes to equalize DWG channels
        float fY = (float)y, fM = (float)m, fC = (float)c;
        const int maxIter = 8;
        for (int it = 0; it < maxIter; ++it) {
            // Simulate print with current filters
            Print::Params prm;
            prm.bypass = false;
            prm.exposure = (float)pexp;
            prm.yFilter = fY;
            prm.mFilter = fM;
            prm.cFilter = 1.0f;

            float rgbOut[3] = { rgbMid[0], rgbMid[1], rgbMid[2] };
            // Safe call path mirrors JuicerProcessing fallback (non-spatial) compute
            Print::simulate_print_pixel(rgbMid, prm, *prt, ws->dirRT, *ws, /*exposureScale*/ 1.0f, rgbOut);

            // If any non-finite, break
            if (!std::isfinite(rgbOut[0]) || !std::isfinite(rgbOut[1]) || !std::isfinite(rgbOut[2])) break;

            // Target equal RGB; adjust filters inversely proportional to channel deviation
            const float mean = (rgbOut[0] + rgbOut[1] + rgbOut[2]) * (1.0f / 3.0f);
            auto safeAdj = [&](float f, float v)->float {
                const float eps = 1e-4f;
                if (mean <= eps) return f;
                float ratio = mean / std::max(v, eps);
                // Dampen and clamp adjustments
                ratio = std::clamp(ratio, 0.5f, 2.0f);
                float fout = f * ratio;
                if (!std::isfinite(fout)) fout = f;
                return std::clamp(fout, 0.05f, 8.0f);
                };
            fY = safeAdj(fY, rgbOut[0]);
            fM = safeAdj(fM, rgbOut[1]);
            fC = safeAdj(fC, rgbOut[2]);
        }

        // Commit fitted filters back to UI params (use the fitted fC, not hardcoded 0.35)
        if (_pEnlargerY) _pEnlargerY->setValue(fY);
        if (_pEnlargerM) _pEnlargerM->setValue(fM);
        if (_pEnlargerC) _pEnlargerC->setValue(fC);

    }    

    // Compute the agx mid-gray compensation factor using neutral enlarger filters and current print runtime.
    // Uses camera exposure compensation EV (Exposure slider), not autoEV, and never measures the preview.
    float compute_midgray_factor_green_neutral(const InstanceState& state, const WorkingState* ws, const Print::Runtime* prt, double exposureCompEV) {
        if (!ws || !prt) return 1.0f;
        if (!Print::profile_is_valid(prt->profile)) return 1.0f;
        if (ws->tablesView.K != Spectral::gShape.K) return 1.0f;

        // Build a DWG mid-gray patch scaled by the camera exposure compensation EV only (agx parity).
        const float midBase = 0.184f;
        const float midScale = static_cast<float>(std::pow(2.0, exposureCompEV));
        const float rgbMid[3] = { midBase * midScale, midBase * midScale, midBase * midScale };

        // Negative E (no DIR on the probe) at scene EV=0 here
        float E[3];
        Spectral::rgbDWG_to_layerExposures_from_tables_with_curves(
            rgbMid, E, /*exposureScale*/ 1.0f,
            (ws->tablesView.K > 0 ? &ws->tablesView : nullptr),
            (ws->spdReady ? ws->spdSInv : nullptr),
            (int)std::clamp(ws->spectralMode, 0, 1),
            (ws->exposureModel == 1) && ws->spdReady,
            ws->sensB, ws->sensG, ws->sensR);

        // Negative logE offsets are zero by design
        const float logE_mid[3] = {
            std::log10f(std::max(E[0], 1e-6f)),
            std::log10f(std::max(E[1], 1e-6f)),
            std::log10f(std::max(E[2], 1e-6f))
        };

        auto clamp_logE_to_curve = [](const Spectral::Curve& c, float le)->float {
            if (c.lambda_nm.empty()) return le;
            const float xmin = static_cast<float>(c.lambda_nm.front());
            const float xmax = static_cast<float>(c.lambda_nm.back());
            if (!std::isfinite(le)) return xmin;
            if (le < xmin) return xmin;
            if (le > xmax) return xmax;
            return le;
            };

        float logE_clamped[3] = {
            clamp_logE_to_curve(ws->densB, logE_mid[0]),
            clamp_logE_to_curve(ws->densG, logE_mid[1]),
            clamp_logE_to_curve(ws->densR, logE_mid[2]),
        };

        float D_neg[3];
        D_neg[0] = Spectral::sample_density_at_logE(ws->densB, logE_clamped[0]); // Y
        D_neg[1] = Spectral::sample_density_at_logE(ws->densG, logE_clamped[1]); // M
        D_neg[2] = Spectral::sample_density_at_logE(ws->densR, logE_clamped[2]); // C

        std::vector<float> Tneg, Ee_expose, Ee_filtered;

        // Negative T from dyes (baseline included by ws.hasBaseline)
        Print::negative_T_from_dyes(*ws, D_neg, Tneg);

        // Enlarger illuminant exposure
        Ee_expose.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float Ee = prt->illumEnlarger.linear.empty() ? 1.0f : prt->illumEnlarger.linear[i];
            Ee_expose[i] = std::max(0.0f, Ee * Tneg[i]);
        }

        // Use persisted neutral baselines from runtime, independent of current UI filter positions
        double yN = prt->neutralY;
        double mN = prt->neutralM;
        double cN = prt->neutralC;

        auto blend_filter = [](float curveVal, float amount)->float {
            // AgX parity: optical density scaling → transmittance = curve^amount
            const float a = std::isfinite(amount) ? std::clamp(amount, 0.0f, 8.0f) : 0.0f;
            const float c = std::isfinite(curveVal) ? std::clamp(curveVal, 1e-6f, 1.0f) : 1.0f;
            return std::pow(c, a);
            };

        float yF = static_cast<float>(yN);
        float mF = static_cast<float>(mN);
        float cF = static_cast<float>(cN);

        Ee_filtered.resize(Spectral::gShape.K);
        for (int i = 0; i < Spectral::gShape.K; ++i) {
            const float fY = blend_filter(prt->filterY.linear.empty() ? 1.0f : prt->filterY.linear[i], yF);
            const float fM = blend_filter(prt->filterM.linear.empty() ? 1.0f : prt->filterM.linear[i], mF);
            const float fC = blend_filter(prt->filterC.linear.empty() ? 1.0f : prt->filterC.linear[i], cF);
            const float fTotal = (fY * fM * fC);
            Ee_filtered[i] = (Ee_expose[i] > 0.0f && fTotal > 0.0f) ? (Ee_expose[i] * fTotal) : 0.0f;
        }


        float rawMid[3];
        Print::raw_exposures_from_filtered_light(prt->profile, Ee_filtered, rawMid, ws->tablesView.deltaLambda);

        // agx-emulsion parity: factor = 1 / raw_midgray_green (under neutral filters).
        const float denom = std::max(1e-6f, rawMid[1]);
        const float factor = 1.0f / denom;
        return std::isfinite(factor) ? factor : 1.0f;
    }


    void onParamsPossiblyChanged(const char* changedNameOrNull);
    void bootstrap_after_attach();
        
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
    Print::load_profile_from_dir(printDir, _state->printRT.profile);

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


    // Apply enlarger neutral filters from JSON if illuminant is TH-KG3-L (index 2)
    if (_pEnlIll) {
        int enlChoice = 2;
        _pEnlIll->getValue(enlChoice);
        if (enlChoice == 2) {
            const char* paperKey = print_json_key_for_paper_index(P.printPaperIndex);
            const char* negKey = negative_json_key_for_stock_index(P.filmStockIndex);
            const std::string jsonPath = gDataDir + std::string("Print\\enlarger_neutral_ymc_filters.json");

            std::tuple<float, float, float> ymc{};
            if (load_enlarger_neutral_filters(jsonPath, paperKey, "TH-KG3-L", negKey, ymc)) {
                float y = std::get<0>(ymc), m = std::get<1>(ymc), c = std::get<2>(ymc);
                if (_pEnlargerY) _pEnlargerY->setValue(y);
                if (_pEnlargerM) _pEnlargerM->setValue(m);
                if (_pEnlargerC) _pEnlargerC->setValue(c);
                JTRACE("PRINT", "Applied neutral filters (JSON) Y/M/C=" + std::to_string(y) + "/" + std::to_string(m) + "/" + std::to_string(c));
                if (_pPrintExposureComp) _pPrintExposureComp->setValue(true);

                // Persist neutral baselines in runtime
                _state->printRT.neutralY = y;
                _state->printRT.neutralM = m;
                _state->printRT.neutralC = c;
            }
            else {
                JTRACE("PRINT", "Neutral filters JSON not found or incomplete; leaving defaults");

                // Fallback: persist current UI values or defaults
                const float yDef = 0.96f, mDef = 0.69f, cDef = 0.35f;
                _state->printRT.neutralY = (_pEnlargerY ? (float)_pEnlargerY->getValue() : yDef);
                _state->printRT.neutralM = (_pEnlargerM ? (float)_pEnlargerM->getValue() : mDef);
                _state->printRT.neutralC = (_pEnlargerC ? (float)_pEnlargerC->getValue() : cDef);
            }
        }
    }


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
        Print::load_profile_from_dir(printDir, _state->printRT.profile);

        // Reload dichroic filters (keep vendor default; can be parameterized later)
        try {
            const std::string durstDir = _state->dataDir + "Filter\\dichroics\\durst_digital_light\\";
            Print::load_dichroic_filters_from_csvs(durstDir, _state->printRT);
        }
        catch (...) {
            JuicerTrace::write("PRINT", "DICHROICS: reload failed; identity filters");
        }

        // Re-apply neutral filters from JSON if enlarger is TH-KG3-L
        int enlChoice = 2;
        if (_pEnlIll) _pEnlIll->getValue(enlChoice);
        if (enlChoice == 2) {
            const char* paperKey = print_json_key_for_paper_index(P.printPaperIndex);
            const char* negKey = negative_json_key_for_stock_index(P.filmStockIndex);
            const std::string jsonPath = gDataDir + std::string("Print\\enlarger_neutral_ymc_filters.json");

            std::tuple<float, float, float> ymc{};
            if (load_enlarger_neutral_filters(jsonPath, paperKey, "TH-KG3-L", negKey, ymc)) {
                float y = std::get<0>(ymc), m = std::get<1>(ymc), c = std::get<2>(ymc);

                _state->suppressParamEvents = true; // prevent recursion
                if (_pEnlargerY) _pEnlargerY->setValue(y);
                if (_pEnlargerM) _pEnlargerM->setValue(m);
                if (_pEnlargerC) _pEnlargerC->setValue(c);
                _state->suppressParamEvents = false;

                JTRACE("PRINT", "Re-applied neutral filters (JSON) Y/M/C="
                    + std::to_string(y) + "/" + std::to_string(m) + "/" + std::to_string(c));

                // Persist neutral baselines in runtime
                _state->printRT.neutralY = y;
                _state->printRT.neutralM = m;
                _state->printRT.neutralC = c;
            }
            else {
                JTRACE("PRINT", "Neutral filters JSON not found or incomplete on paper change");

                // Fallback: persist current UI values or defaults
                const float yDef = 0.96f, mDef = 0.69f, cDef = 0.35f;
                _state->printRT.neutralY = (_pEnlargerY ? (float)_pEnlargerY->getValue() : yDef);
                _state->printRT.neutralM = (_pEnlargerM ? (float)_pEnlargerM->getValue() : mDef);
                _state->printRT.neutralC = (_pEnlargerC ? (float)_pEnlargerC->getValue() : cDef);
            }
        }

        // Force working state snapshot to carry updated printRT profile into render
        if (_state->baseLoaded) {
            rebuild_working_state(this->getHandle(), *_state, P);
            _state->lastParams = P;
            _state->lastHash = hash_params(P);
        }

    }

    // Reload BaseState if stock changed
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamFilmStock) == 0) {
        const std::string stockDir = stock_dir_for_index(P.filmStockIndex);
        _state->baseLoaded = load_film_stock_into_base(stockDir, *_state);
    }
    // If film stock changed, also re-apply neutral filters for current print paper
    if (changedNameOrNull && std::strcmp(changedNameOrNull, kParamFilmStock) == 0 && _state->baseLoaded) {
        int enlChoice = 2;
        if (_pEnlIll) _pEnlIll->getValue(enlChoice);
        if (enlChoice == 2) {
            const char* paperKey = print_json_key_for_paper_index(P.printPaperIndex);
            const char* negKey = negative_json_key_for_stock_index(P.filmStockIndex);
            const std::string jsonPath = gDataDir + std::string("Print\\enlarger_neutral_ymc_filters.json");

            std::tuple<float, float, float> ymc{};
            if (load_enlarger_neutral_filters(jsonPath, paperKey, "TH-KG3-L", negKey, ymc)) {
                float y = std::get<0>(ymc), m = std::get<1>(ymc), c = std::get<2>(ymc);
                _state->suppressParamEvents = true;
                if (_pEnlargerY) _pEnlargerY->setValue(y);
                if (_pEnlargerM) _pEnlargerM->setValue(m);
                if (_pEnlargerC) _pEnlargerC->setValue(c);
                _state->suppressParamEvents = false;
                JTRACE("PRINT", "Re-applied neutral filters after stock change Y/M/C=" + std::to_string(y) + "/" + std::to_string(m) + "/" + std::to_string(c));
            }
            else {
                JTRACE("PRINT", "Neutral filters JSON missing on stock change; applying realistic defaults");
                const float yDef = 0.96f, mDef = 0.69f, cDef = 0.35f;
                _state->printRT.neutralY = yDef;
                _state->printRT.neutralM = mDef;
                _state->printRT.neutralC = cDef;
                // Do not overwrite user UI values here; runtime neutral baselines are used by mid-gray computation.
            }
        }
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

    if (changedNameOrNull && std::strcmp(changedNameOrNull, "PrintFitNeutral") == 0) {
        // Fit and set neutral Y/M/C enlarger filters to remove gross blue/cyan bias
        fitNeutralEnlargerFilters();
    }

}
// HIDDEN MESSAGE KUKHUVUD


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
        p->appendOption("CMF-basis");
        p->setDefault(0);
    }
    // Exposure model
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamExposureModel);
        p->setLabel("Exposure model");
        p->appendOption("Matrix");
        p->appendOption("SPD");
        p->setDefault(1);
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
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("PrintExposureCompensation");
            p->setLabel("Print exposure compensation");
            p->setDefault(true);
            if (grpPrint) p->setParent(*grpPrint);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerY");
            p->setLabel("Enlarger Y");
            p->setDefault(0.96);
            p->setDisplayRange(0.25, 4.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerM");
            p->setLabel("Enlarger M");
            p->setDefault(0.69);
            p->setDisplayRange(0.25, 4.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("EnlargerC");
            p->setLabel("Enlarger C (neutral baseline)");
            p->setDefault(0.35);
            p->setDisplayRange(0.25, 4.0);
            if (grpPrint) p->setParent(*grpPrint);
        }
        {
            OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam("PrintFitNeutral");
            p->setLabel("Fit neutral");
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
