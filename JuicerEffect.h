#pragma once

#include <memory>
#include <string>

#include "ofxsImageEffect.h"
#include "JuicerState.h"

namespace OFX {
    class Clip;
    class Image;
    class DoubleParam;
    class ChoiceParam;
    class BooleanParam;
    struct InstanceChangedArgs;
    struct RenderArguments;
} // namespace OFX

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

// Output encoding parameters
#define kParamOutputColorSpace "OutputColorSpace"
#define kParamOutputCctfEncoding "OutputCctfEncoding"
#define kParamOutputLinearPassThrough "OutputLinearPassThrough"

class JuicerEffect : public OFX::ImageEffect {
public:
    explicit JuicerEffect(OfxImageEffectHandle handle);
    ~JuicerEffect() override;

    void render(const OFX::RenderArguments& args) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;

private:
    ParamSnapshot snapshotParams() const;
    void onParamsPossiblyChanged(const char* changedNameOrNull);
    void bootstrap_after_attach();
    void applyNeutralFilters(const ParamSnapshot& P, bool resetFilterParams, bool ensureExposureComp);
#ifdef JUICER_ENABLE_COUPLERS
    void applyCouplerProfileDefaults(ParamSnapshot& P);
#endif

    OFX::Clip* _src = nullptr;
    OFX::Clip* _dst = nullptr;

    // Cached params (wrappers)
    OFX::DoubleParam* _pExposure = nullptr;
    OFX::ChoiceParam* _pFilmStock = nullptr;
    OFX::ChoiceParam* _pPrintPaper = nullptr;
    OFX::ChoiceParam* _pRefIll = nullptr;
    OFX::ChoiceParam* _pViewIll = nullptr;
    OFX::ChoiceParam* _pEnlIll = nullptr;
    OFX::ChoiceParam* _pOutputColorSpace = nullptr;
    OFX::BooleanParam* _pOutputCctfEncoding = nullptr;
    OFX::BooleanParam* _pOutputLinearPassThrough = nullptr;

    OFX::BooleanParam* _pUnmix = nullptr;

#ifdef JUICER_ENABLE_COUPLERS
    OFX::BooleanParam* _pCouplersActive = nullptr;
    OFX::BooleanParam* _pCouplersPrecorrect = nullptr;
    OFX::DoubleParam* _pCouplersAmount = nullptr;
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
    OFX::DoubleParam* _pScanFilmLongEdge = nullptr;

    OFX::BooleanParam* _pPrintBypass = nullptr;
    OFX::DoubleParam* _pPrintExposure = nullptr;
    OFX::DoubleParam* _pPrintPreflash = nullptr;
    OFX::BooleanParam* _pPrintExposureComp = nullptr;
    OFX::DoubleParam* _pEnlargerY = nullptr;
    OFX::DoubleParam* _pEnlargerM = nullptr;

    std::unique_ptr<InstanceState> _state;
};