#include "JuicerEffect.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Couplers.h"
#include "Illuminants.h"
#include "NeutralFilters.h"
#include "OutputEncoding.h"
#include "Print.h"
#include "ParamNames.h"
#include "Scanner.h"
#include "SpectralMath.h"
#include "SpectralTables.h"
#include "mainProcessing.h"

namespace {
    static std::once_flag gSpectralGlobalsOnce;

    void init_spectral_globals_once() {
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
        case 1: return { "D55", "d55" };
        case 2: return { "D50", "d50" };
        case 3: return { "TH-KG3-L", "th-kg3-l" };
        case 4: return { "EqualEnergy", "equal_energy", "Equal energy" };
        default: break;
        }
        return {};
    }

} // namespace

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
} // namespace JuicerRegistry

JuicerEffect::JuicerEffect(OfxImageEffectHandle handle)
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
        _pOutputColorSpace = fetchChoiceParam(kParamOutputColorSpace);
        _pOutputCctfEncoding = fetchBooleanParam(kParamOutputCctfEncoding);
        _pOutputLinearPassThrough = fetchBooleanParam(kParamOutputLinearPassThrough);

        _pUnmix = fetchBooleanParam("UnmixDensities");

#ifdef JUICER_ENABLE_COUPLERS
        _pCouplersActive = fetchBooleanParam(Couplers::kParamCouplersActive);
        _pCouplersPrecorrect = fetchBooleanParam(Couplers::kParamCouplersPrecorrect);
        _pCouplersAmount = fetchDoubleParam(Couplers::kParamCouplersAmount);
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
        _pScanFilmLongEdge = fetchDoubleParam(JuicerParams::kScannerFilmLongEdgeMm);

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

JuicerEffect::~JuicerEffect() {
    // Mirror destroyInstance() guards without touching C suites.
    JuicerRegistry::erase(this->getHandle());
    _state.reset();
}

void JuicerEffect::render(const OFX::RenderArguments& args) {
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

    const OfxRectI fullBounds = srcImg->getBounds();

    // ROI: args.renderWindow if provided; otherwise use image bounds
    OfxRectI roi = args.renderWindow;
    if (roi.x1 == roi.x2 && roi.y1 == roi.y2) {
        roi = fullBounds;
    }
    const int width = roi.x2 - roi.x1;
    const int height = roi.y2 - roi.y1;
    if (width <= 0 || height <= 0) return;
    const int fullWidth = fullBounds.x2 - fullBounds.x1;
    const int fullHeight = fullBounds.y2 - fullBounds.y1;

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
        double scanFilmMm = 36.0;
        if (_pScanEnabled) _pScanEnabled->getValue(scanEnabled);
        if (_pScanAuto)    _pScanAuto->getValue(scanAuto);
        if (_pScanTargetY) _pScanTargetY->getValue(scanY);
        if (_pScanFilmLongEdge) _pScanFilmLongEdge->getValue(scanFilmMm);
        scannerParams.enabled = scanEnabled;
        scannerParams.autoExposure = scanAuto;
        scannerParams.targetY = static_cast<float>(scanY);
        scannerParams.filmLongEdgeMm =
            (std::isfinite(scanFilmMm) && scanFilmMm > 0.0)
            ? static_cast<float>(scanFilmMm)
            : 36.0f;
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

    OutputEncoding::Params outputEncodingParams;
    {
        int csIndex = OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
        if (_pOutputColorSpace) _pOutputColorSpace->getValue(csIndex);
        bool applyCctf = true;
        if (_pOutputCctfEncoding) _pOutputCctfEncoding->getValue(applyCctf);
        bool preserveLinear = false;
        if (_pOutputLinearPassThrough) _pOutputLinearPassThrough->getValue(preserveLinear);
        outputEncodingParams.colorSpace = OutputEncoding::colorSpaceFromIndex(csIndex);
        outputEncodingParams.applyCctfEncoding = applyCctf;
        outputEncodingParams.preserveLinearRange = preserveLinear;
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
        auto valid_dim = [](double v) -> bool {
            return std::isfinite(v) && v > 0.0;
            };
        double canonicalWidth = 0.0;
        double canonicalHeight = 0.0;
        if (_src) {
            canonicalWidth = static_cast<double>(_src->getCanonicalWidth());
            canonicalHeight = static_cast<double>(_src->getCanonicalHeight());
        }
        if (!valid_dim(canonicalWidth) || !valid_dim(canonicalHeight)) {
            if (_state) {
                const double cachedW = _state->spatialSigmaCanonicalWidth.load(std::memory_order_acquire);
                const double cachedH = _state->spatialSigmaCanonicalHeight.load(std::memory_order_acquire);
                if (valid_dim(cachedW) && valid_dim(cachedH)) {
                    canonicalWidth = cachedW;
                    canonicalHeight = cachedH;
                }
            }
        }
        if (!valid_dim(canonicalWidth)) canonicalWidth = static_cast<double>(fullWidth);
        if (!valid_dim(canonicalHeight)) canonicalHeight = static_cast<double>(fullHeight);

        const double filmLongEdgeMm =
            (std::isfinite(scannerParams.filmLongEdgeMm) && scannerParams.filmLongEdgeMm > 0.0f)
            ? static_cast<double>(scannerParams.filmLongEdgeMm)
            : 36.0;
        float canonicalSigmaPixels = 0.0f;
        const float sigmaMicrometers = dirRT.spatialSigmaMicrometers;
        if (sigmaMicrometers > 0.0f && valid_dim(canonicalWidth) && valid_dim(canonicalHeight)) {
            const auto nearly_equal_double = [](double a, double b) {
                const double diff = std::fabs(a - b);
                const double scale = std::max({ 1.0, std::fabs(a), std::fabs(b) });
                return diff <= scale * 1e-9;
                };
            const auto nearly_equal_float = [](float a, float b) {
                const float diff = std::fabs(a - b);
                const float scale = std::max({ 1.0f, std::fabs(a), std::fabs(b) });
                return diff <= scale * 1e-6f;
                };

            bool cacheHit = false;
            if (_state) {
                const bool cacheValid = _state->spatialSigmaCacheValid.load(std::memory_order_acquire);
                if (cacheValid) {
                    const float cachedMic = _state->spatialSigmaMicrometers.load(std::memory_order_relaxed);
                    const double cachedW = _state->spatialSigmaCanonicalWidth.load(std::memory_order_relaxed);
                    const double cachedH = _state->spatialSigmaCanonicalHeight.load(std::memory_order_relaxed);
                    const double cachedFilm = _state->spatialSigmaFilmLongEdgeMm.load(std::memory_order_relaxed);
                    const float cachedSigma = _state->spatialSigmaPixelsCanonical.load(std::memory_order_relaxed);
                    if (nearly_equal_float(cachedMic, sigmaMicrometers) &&
                        nearly_equal_double(cachedW, canonicalWidth) &&
                        nearly_equal_double(cachedH, canonicalHeight) &&
                        nearly_equal_double(cachedFilm, filmLongEdgeMm)) {
                        canonicalSigmaPixels = cachedSigma;
                        cacheHit = true;
                    }
                }
            }

            if (!cacheHit) {
                canonicalSigmaPixels = Couplers::spatial_sigma_pixels_from_micrometers(
                    sigmaMicrometers,
                    filmLongEdgeMm,
                    canonicalWidth,
                    canonicalHeight);
                if (_state) {
                    _state->spatialSigmaCanonicalWidth.store(canonicalWidth, std::memory_order_release);
                    _state->spatialSigmaCanonicalHeight.store(canonicalHeight, std::memory_order_release);
                    _state->spatialSigmaFilmLongEdgeMm.store(filmLongEdgeMm, std::memory_order_release);
                    _state->spatialSigmaMicrometers.store(sigmaMicrometers, std::memory_order_release);
                    _state->spatialSigmaPixelsCanonical.store(canonicalSigmaPixels, std::memory_order_release);
                    _state->spatialSigmaCacheValid.store(true, std::memory_order_release);
                }
            }

            double scaleX = (std::isfinite(args.renderScale.x) && args.renderScale.x > 0.0)
                ? args.renderScale.x
                : 1.0;
            double scaleY = (std::isfinite(args.renderScale.y) && args.renderScale.y > 0.0)
                ? args.renderScale.y
                : 1.0;
            double renderScaleFactor = std::max(scaleX, scaleY);
            if (!(std::isfinite(renderScaleFactor)) || renderScaleFactor <= 0.0) {
                renderScaleFactor = 1.0;
            }
            float scaledSigma = canonicalSigmaPixels * static_cast<float>(renderScaleFactor);
            if (!std::isfinite(scaledSigma) || scaledSigma < 0.0f) scaledSigma = 0.0f;
            if (scaledSigma > 25.0f) scaledSigma = 25.0f;
            dirRT.spatialSigmaPixels = scaledSigma;
        }
        else {
            dirRT.spatialSigmaPixels = 0.0f;
        }
    }
#else
    Couplers::Runtime dirRT{};
#endif

    WorkingState* activeWorkingState = (_state && _state->baseLoaded)
        ? _state->activeWS.load(std::memory_order_acquire)
        : nullptr;
    const WorkingState* ws = activeWorkingState;
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
        (ws && ws->tablesPrint.K == Spectral::gShape.K) &&
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
    proc.setOutputEncoding(outputEncodingParams);
    proc.setRenderWindowRect(roi);
    proc.setGPURenderArgs(args);

    struct RenderGuard {
        InstanceState* state;
        WorkingState* ws;
        RenderGuard(InstanceState* s, WorkingState* w) : state(s), ws(w) {
            if (state) {
                state->rendersInFlight.fetch_add(1, std::memory_order_acq_rel);
                state->renderWS.store(w, std::memory_order_release);
            }
        }
        ~RenderGuard() {
            if (state) {
                const int prev = state->rendersInFlight.fetch_sub(1, std::memory_order_acq_rel);
                if (prev <= 1) {
                    WorkingState* expected = ws;
                    state->renderWS.compare_exchange_strong(
                        expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
                }
                state->renderCv.notify_all();
            }
        }
    } guard(_state.get(), activeWorkingState);

    // Dispatch to support library's threaded/tiled CPU path
    proc.process();
}

void JuicerEffect::changedParam(const OFX::InstanceChangedArgs&, const std::string& paramName) {
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

ParamSnapshot JuicerEffect::snapshotParams() const {
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
    if (_pCouplersAmount)  _pCouplersAmount->getValue(P.couplersAmount);
    if (_pCouplersAmountR) _pCouplersAmountR->getValue(P.ratioR);
    if (_pCouplersAmountG) _pCouplersAmountG->getValue(P.ratioG);
    if (_pCouplersAmountB) _pCouplersAmountB->getValue(P.ratioB);
    if (_pCouplersSigma)      _pCouplersSigma->getValue(P.sigma);
    if (_pCouplersHigh)       _pCouplersHigh->getValue(P.high);
    double spatial = 0.0;
    if (_pCouplersSpatialSigma) _pCouplersSpatialSigma->getValue(spatial);
    P.spatialSigmaMicrometers = spatial;
#endif
    return P;
}

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
            std::strcmp(changedNameOrNull, kParamCouplersAmount) == 0 ||
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