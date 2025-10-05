#pragma once

#include <string>
#include <utility>
#include <vector>

namespace Profiles {

    struct AgxFilmProfile {
        std::vector<std::pair<float, float>> dyeC;
        std::vector<std::pair<float, float>> dyeM;
        std::vector<std::pair<float, float>> dyeY;
        std::vector<std::pair<float, float>> baseMin;
        std::vector<std::pair<float, float>> baseMid;
        float dyeDensityMinFactor = 1.0f;

        std::string densitometer;
        std::vector<float> densityMidNeutral;

        std::vector<std::pair<float, float>> logSensR;
        std::vector<std::pair<float, float>> logSensG;
        std::vector<std::pair<float, float>> logSensB;

        std::vector<std::pair<float, float>> densityCurveB;
        std::vector<std::pair<float, float>> densityCurveG;
        std::vector<std::pair<float, float>> densityCurveR;
    };

    bool load_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile);

} // namespace Profiles
