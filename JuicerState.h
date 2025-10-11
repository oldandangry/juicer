#pragma once

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Couplers.h"
#include "Print.h"
#include "ProfileJSONLoader.h"
#include "SpectralMath.h"
#include "WorkingState.h"
#include "ofxImageEffect.h"

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
        Scope(const char* t, std::string n) : tag(t), name(std::move(n)) {
            write(tag, "BEGIN " + name);
        }
        ~Scope() {
            write(tag, "END   " + name);
        }
    };
} // namespace JuicerTrace

#define JTRACE(tag, msg) ::JuicerTrace::write(tag, (msg))
#define JTRACE_SCOPE(tag, name) ::JuicerTrace::Scope _juicer_scope_guard_((tag), (name))

extern const std::string gDataDir;

struct BaseState {
    Spectral::Curve epsY, epsM, epsC;
    Spectral::Curve sensB, sensG, sensR;
    Spectral::Curve densB, densG, densR;
    Spectral::Curve baseMin, baseMid;
    bool hasBaseline = false;
    std::string densitometerType;
    std::vector<float> densityMidNeutral;
    Profiles::DirCouplersProfile dirCouplers;
    Profiles::MaskingCouplersProfile maskingCouplers;
};

struct ParamSnapshot {
    int filmStockIndex = 0;
    int printPaperIndex = 0;
    int refIll = 0;
    int viewIll = 0;
    int enlIll = 3;
    int couplersActive = 1;
    int couplersPrecorrect = 1;
    double couplersAmount = 1.0;
    double ratioR = 0.7, ratioG = 0.7, ratioB = 0.5;
    double sigma = 1.0, high = 0.0;
    int unmix = 1;
    double spatialSigmaMicrometers = 0.0;
};

constexpr int kFactoryCouplersActive = 1;
constexpr double kFactoryCouplersAmount = 1.0;
constexpr double kFactoryCouplersRatioR = 0.7;
constexpr double kFactoryCouplersRatioG = 0.7;
constexpr double kFactoryCouplersRatioB = 0.5;
constexpr double kFactoryCouplersSigma = 1.0;
constexpr double kFactoryCouplersHigh = 0.0;
constexpr double kFactoryCouplersSpatialSigma = 0.0;

uint64_t hash_params(const ParamSnapshot& p);

struct CouplerDirtyFlags {
    bool active = false;
    bool amount = false;
    bool ratioR = false;
    bool ratioG = false;
    bool ratioB = false;
    bool sigma = false;
    bool high = false;
    bool spatialSigma = false;
};

uint64_t hash_params(const ParamSnapshot& p);

struct InstanceState {
    std::mutex m;
    BaseState base;
    WorkingState workA;
    WorkingState workB;
    std::atomic<WorkingState*> activeWS{ nullptr };
    std::atomic<int> rendersInFlight{ 0 };
    std::atomic<WorkingState*> renderWS{ nullptr };
    std::condition_variable renderCv;
    uint64_t activeBuildCounter = 0;

    ParamSnapshot lastParams;
    uint64_t lastHash = 0;

    Print::Runtime printRT;

    bool suppressParamEvents = false;
    bool inBootstrap = false;

    std::string dataDir;
    bool baseLoaded = false;

    CouplerDirtyFlags couplerDirty;

    bool couplerProfileSpatialSigmaValid = false;
    double couplerProfileSpatialSigmaMicrometers = 0.0;

    // Cache for DIR spatial sigma conversion (canonical project dimensions)
    std::atomic<bool> spatialSigmaCacheValid{ false };
    std::atomic<double> spatialSigmaCanonicalWidth{ 0.0 };
    std::atomic<double> spatialSigmaCanonicalHeight{ 0.0 };
    std::atomic<double> spatialSigmaFilmLongEdgeMm{ 0.0 };
    std::atomic<float> spatialSigmaMicrometers{ 0.0f };
    std::atomic<float> spatialSigmaPixelsCanonical{ 0.0f };

    // Auto-exposure cache (per frame / build)
    std::mutex autoExposureMutex;
    bool autoExposureCacheValid = false;
    double autoExposureCacheTime = std::numeric_limits<double>::quiet_NaN();
    double autoExposureCacheTargetY = 0.184;
    bool autoExposureCacheAutoEnabled = false;
    uint64_t autoExposureCacheBuildCounter = 0;
    OfxRectI autoExposureCacheBounds{ 0, 0, 0, 0 };
    double autoExposureCacheEV = 0.0;
    bool autoExposureCanonicalValid = false;
    OfxRectI autoExposureCanonicalBounds{ 0, 0, 0, 0 };

    WorkingState* inactive() {
        WorkingState* a = activeWS.load(std::memory_order_acquire);
        return (a == &workA) ? &workB : &workA;
    }
};

std::string print_dir_for_index(int index);
std::string stock_dir_for_index(int index);
const char* negative_json_key_for_stock_index(int filmIndex);
int film_stock_option_count();
const char* film_stock_option_label(int index);
int print_paper_option_count();
const char* print_paper_option_label(int index);
bool load_film_stock_into_base(const std::string& stockDir, InstanceState& S);
void rebuild_working_state(OfxImageEffectHandle instance, InstanceState& S, const ParamSnapshot& P);
