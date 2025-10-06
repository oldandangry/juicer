#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace Profiles {

    struct DirCouplersProfile {
        bool hasData = false;
        bool active = false;
        float amount = 1.0f;
        std::array<float, 3> ratioRGB{ {1.0f, 1.0f, 1.0f} };
        float diffusionInterlayer = 0.0f;
        float diffusionSizeUm = 0.0f;
        float highExposureShift = 0.0f;
    };

    struct MaskingCouplersProfile {
        bool hasData = false;
        std::vector<float> crossOverPoints;
        std::vector<float> transitionWidths;
        std::array<std::vector<std::array<float, 3>>, 3> gaussianModel{};
    };

    struct AgxFilmProfile {
        std::vector<std::pair<float, float>> dyeC;
        std::vector<std::pair<float, float>> dyeM;
        std::vector<std::pair<float, float>> dyeY;
        std::vector<std::pair<float, float>> baseMin;
        std::vector<std::pair<float, float>> baseMid;
        float dyeDensityMinFactor = 1.0f;
        std::array<float, 3> gammaFactor{ {1.0f, 1.0f, 1.0f} };
        bool hasGammaFactor = false;

        std::string densitometer;
        std::vector<float> densityMidNeutral;

        std::vector<std::pair<float, float>> logSensR;
        std::vector<std::pair<float, float>> logSensG;
        std::vector<std::pair<float, float>> logSensB;
                
        std::vector<std::pair<float, float>> densityCurveR;
        std::vector<std::pair<float, float>> densityCurveG;
        std::vector<std::pair<float, float>> densityCurveB;

        DirCouplersProfile dirCouplers;
        MaskingCouplersProfile maskingCouplers;
    };

    bool load_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile);

} // namespace Profiles
