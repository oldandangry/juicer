

// OpenFX 1.4 canonical headers (no non-spec feature macros)
#include "ofxCore.h"        // OfxHost, OfxPlugin, kOfxAction*
#include "ofxProperty.h"    // OfxPropertySuiteV1, property keys
#include "ofxParam.h"       // OfxParameterSuiteV1, kOfxTypeParameter
#include "ofxImageEffect.h" // OfxImageEffectSuiteV1, OfxRectI, OfxRectD, image effect props
#include "ofxPixels.h"      // Pixel/rect helpers used by some 1.4 distributions

#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

#include "Couplers.h"
#include "JuicerEffect.h"
#include "JuicerState.h"
#include "OutputEncoding.h"
#include "ParamNames.h"
#include "Print.h"
#include "ProfileJSONLoader.h"

namespace {

    static std::string computeDataDir()
    {
        namespace fs = std::filesystem;

#if defined(_WIN32)
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&computeDataDir),
            &module)) {
            return std::string();
        }

        std::wstring buffer(MAX_PATH, L'\0');
        DWORD length = 0;
        for (;;) {
            SetLastError(ERROR_SUCCESS);
            length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (length == 0) {
                return std::string();
            }
            if (length < buffer.size()) {
                buffer.resize(length);
                break;
            }
            if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
                buffer.resize(length);
                break;
            }
            buffer.resize(buffer.size() * 2);
        }

        fs::path modulePath(buffer);
        fs::path moduleDir = modulePath.parent_path();
        if (moduleDir.empty()) {
            return std::string();
        }
        fs::path contentsDir = moduleDir.parent_path();
        if (contentsDir.empty()) {
            return std::string();
        }

        fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
        resourcesDir.make_preferred();
        std::wstring native = resourcesDir.native();
        if (!native.empty() && native.back() != L'\\') {
            native.push_back(L'\\');
        }

        if (native.empty()) {
            return std::string();
        }

        int required = WideCharToMultiByte(CP_UTF8, 0, native.c_str(), static_cast<int>(native.size()), nullptr, 0, nullptr, nullptr);
        if (required <= 0) {
            return std::string();
        }

        std::string path(static_cast<size_t>(required), '\0');
        WideCharToMultiByte(CP_UTF8, 0, native.c_str(), static_cast<int>(native.size()), path.data(), required, nullptr, nullptr);
        return path;
#else
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void*>(&computeDataDir), &info) == 0 || info.dli_fname == nullptr) {
            return std::string();
        }

        fs::path modulePath(info.dli_fname);
        fs::path moduleDir = modulePath.parent_path();
        if (moduleDir.empty()) {
            return std::string();
        }
        fs::path contentsDir = moduleDir.parent_path();
        if (contentsDir.empty()) {
            return std::string();
        }

        fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
        resourcesDir.make_preferred();
        std::string path = resourcesDir.u8string();
        if (!path.empty() && path.back() != '/') {
            path.push_back('/');
        }
        return path;
#endif
    }

} // namespace

// Plugin data directory (immutable)
const std::string gDataDir = computeDataDir();

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
        const int stockCount = film_stock_option_count();
        for (int i = 0; i < stockCount; ++i) p->appendOption(film_stock_option_label(i));
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
        p->setDefault(false);
    }

#ifdef JUICER_ENABLE_COUPLERS
    // Couplers (DIR) parameters — wrapper descriptors matching Couplers::define_params
    {
        OFX::GroupParamDescriptor* grpCouplers = desc.defineGroupParam(Couplers::kParamCouplersGroup);
        if (grpCouplers) grpCouplers->setLabel("DIR couplers");

        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(Couplers::kParamCouplersActive);
            p->setLabel("Active");
            p->setDefault(true);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(Couplers::kParamCouplersPrecorrect);
            p->setLabel("Precorrect density curves");
            p->setDefault(true);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmount);
            p->setLabel("Couplers amount");
            p->setDefault(1.0);
            p->setRange(0.0, 2.0);
            p->setDisplayRange(0.0, 2.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountR);
            p->setLabel("Couplers ratio R");
            p->setDefault(0.7);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountG);
            p->setLabel("Couplers ratio G");
            p->setDefault(0.7);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersAmountB);
            p->setLabel("Couplers ratio B");
            p->setDefault(0.5);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersLayerSigma);
            p->setLabel("Layer diffusion");
            p->setDefault(1.0);
            p->setRange(0.0, 3.0);
            p->setDisplayRange(0.0, 3.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersHighExpShift);
            p->setLabel("High exposure shift");
            p->setDefault(0.0);
            p->setRange(0.0, 1.0);
            p->setDisplayRange(0.0, 1.0);
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(Couplers::kParamCouplersSpatialSigma);
            p->setLabel("Couplers spatial diffusion (\xC2\xB5m)");
            p->setDefault(0.0);
            p->setRange(0.0, 50.0);
            p->setDisplayRange(0.0, 50.0);
            p->setHint("Micrometers of DIR spatial diffusion; scaled by the Scanner film long edge parameter.");
            if (grpCouplers) p->setParent(*grpCouplers);
            p->setEvaluateOnChange(true);
        }
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
    {
        OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(JuicerParams::kScannerFilmLongEdgeMm);
        p->setLabel("Film long edge (mm)");
        p->setDefault(36.0);
        p->setRange(1.0, 400.0);
        p->setDisplayRange(24.0, 320.0);
        p->setHint("Longest film dimension in millimeters; used to convert DIR spatial diffusion micrometers to pixels.");
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
            const int paperCount = print_paper_option_count();
            for (int i = 0; i < paperCount; ++i) p->appendOption(print_paper_option_label(i));
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
        p->appendOption("D55");
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
        p->appendOption("D55");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("Equal energy");
        p->setDefault(3);
        p->setEvaluateOnChange(true);
    }
    {
        OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("ViewingIlluminant");
        p->setLabel("Viewing illuminant");
        p->appendOption("D65");
        p->appendOption("D55");
        p->appendOption("D50");
        p->appendOption("TH-KG3-L");
        p->appendOption("Equal energy");
        p->setDefault(0);
        p->setEvaluateOnChange(true);
    }

    // Output encoding group
    {
        OFX::GroupParamDescriptor* grpOutput = desc.defineGroupParam("OutputEncodingGroup");
        if (grpOutput) grpOutput->setLabel("Output encoding");

        {
            OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamOutputColorSpace);
            p->setLabel("Output color space");
            for (std::size_t i = 0; i < OutputEncoding::kColorSpaceCount; ++i) {
                p->appendOption(OutputEncoding::kColorSpaceLabels[i]);
            }
            p->setDefault(OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB));
            if (grpOutput) p->setParent(*grpOutput);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(kParamOutputCctfEncoding);
            p->setLabel("Apply output CCTF");
            p->setDefault(true);
            if (grpOutput) p->setParent(*grpOutput);
            p->setEvaluateOnChange(true);
        }
        {
            OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(kParamOutputLinearPassThrough);
            p->setLabel("Output linear pass-through");
            p->setDefault(false);
            if (grpOutput) p->setParent(*grpOutput);
            p->setEvaluateOnChange(true);
        }
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
