// JuicerProcessing.h
#pragma once
#define JUICER_ENABLE_COUPLERS 1

#include "ofxsProcessing.h"
#include "ofxsImageEffect.h"
#include "SpectralMath.h"
#include "WorkingState.h"
#include "Print.h"
#include "Scanner.h"
#include "Couplers.h"

namespace JuicerProc {
    void copyNonFloatRect(OFX::Image* src, OFX::Image* dst);
}

namespace JuicerProc {
    // Render a small print preview to outRGB (RGB, row-major), measuring from the source image.
    // outRGB size must be outW*outH*3. srcBounds is the full image bounds used for sampling.
    void renderPrintPreviewToBuffer(
        const OFX::Image* src, const OfxRectI& srcBounds,
        float* outRGB, int outW, int outH,
        const Print::Params& prm,
        const Print::Runtime* prt,
        const WorkingState* ws,
        const Couplers::Runtime& dirRT,
        float exposureScale);
}


// Full class declaration
class JuicerProcessor : public OFX::ImageProcessor {
public:
    explicit JuicerProcessor(OFX::ImageEffect& effect);

    void setSrcDst(OFX::Image* src, OFX::Image* dst);
    void setRenderWindowRect(const OfxRectI& rect);
    void setComponents(int n);
    void setScannerParams(const Scanner::Params& p);
    void setPrintParams(const Print::Params& p);
    void setDirRuntime(const Couplers::Runtime& rt);
    void setWorkingState(const WorkingState* ws, bool wsReady);
    void setPrintRuntime(const Print::Runtime* prt, bool printReady);
    void setExposure(float exposureScale);

    void multiThreadProcessImages(OfxRectI procWindow) override;

private:
    OFX::Image* _srcImg;
    int _nComponents;

    Scanner::Params _scannerParams;
    Print::Params _printParams;
    Couplers::Runtime _dirRT;

    const Print::Runtime* _prt;
    const WorkingState* _ws;
    bool _wsReady;
    bool _printReady;

    float _exposureScale;    
};
