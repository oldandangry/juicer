#pragma once

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <fstream>
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
    double spatialSigma = 0.0;
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
