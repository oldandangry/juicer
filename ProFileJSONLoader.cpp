#include "ProfileJSONLoader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"

namespace Profiles {
    namespace {

        using Json = nlohmann::json;

        void replace_all(std::string& str, const std::string& from, const std::string& to) {
            if (from.empty()) return;
            size_t pos = 0;
            while ((pos = str.find(from, pos)) != std::string::npos) {
                str.replace(pos, from.size(), to);
                pos += to.size();
            }
        }

        bool parse_json_file(const std::string& path, Json& out) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                return false;
            }
            std::ostringstream oss;
            oss << file.rdbuf();
            std::string text = oss.str();

            // agx-emulsion profiles encode missing spectral samples as NaN.
            // Replace with JSON null so the parser can ingest the document.
            replace_all(text, "NaN", "null");

            out = Json::parse(text, nullptr, false);
            return !out.is_discarded();
        }

        float value_or_default(const Json& node, float defaultValue) {
            if (node.is_number_float() || node.is_number_integer()) {
                float v = static_cast<float>(node.get<double>());
                if (std::isfinite(v)) {
                    return v;
                }
            }
            return defaultValue;
        }

    } // namespace

    bool load_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile) {
        outProfile = AgxFilmProfile{};

        Json root;
        if (!parse_json_file(jsonPath, root)) {
            return false;
        }

        if (root.contains("info")) {
            const Json& info = root["info"];
            if (info.contains("densitometer") && info["densitometer"].is_string()) {
                outProfile.densitometer = info["densitometer"].get<std::string>();
            }
            if (info.contains("density_midscale_neutral")) {
                const Json& mid = info["density_midscale_neutral"];
                outProfile.densityMidNeutral.clear();
                if (mid.is_array()) {
                    for (const auto& v : mid) {
                        float val = value_or_default(v, std::numeric_limits<float>::quiet_NaN());
                        if (std::isfinite(val)) {
                            outProfile.densityMidNeutral.push_back(val);
                        }
                    }
                }
                else {
                    float val = value_or_default(mid, std::numeric_limits<float>::quiet_NaN());
                    if (std::isfinite(val)) {
                        outProfile.densityMidNeutral.push_back(val);
                    }
                }
            }
        }

        if (!root.contains("data")) {
            return false;
        }
        const Json& data = root["data"];

        if (!data.contains("wavelengths") || !data.contains("dye_density")) {
            return false;
        }

        const Json& wavelengths = data["wavelengths"];
        const Json& dyeDensity = data["dye_density"];
        if (!wavelengths.is_array() || !dyeDensity.is_array()) {
            return false;
        }

        const size_t sampleCount = std::min(wavelengths.size(), dyeDensity.size());
        outProfile.dyeC.reserve(sampleCount);
        outProfile.dyeM.reserve(sampleCount);
        outProfile.dyeY.reserve(sampleCount);
        outProfile.baseMin.reserve(sampleCount);
        outProfile.baseMid.reserve(sampleCount);

        for (size_t i = 0; i < sampleCount; ++i) {
            float wl = value_or_default(wavelengths[i], std::numeric_limits<float>::quiet_NaN());
            if (!std::isfinite(wl)) {
                continue;
            }

            const Json& row = dyeDensity[i];
            if (!row.is_array()) {
                continue;
            }

            const float valC = (row.size() > 0) ? value_or_default(row[0], 0.0f) : 0.0f;
            const float valM = (row.size() > 1) ? value_or_default(row[1], 0.0f) : 0.0f;
            const float valY = (row.size() > 2) ? value_or_default(row[2], 0.0f) : 0.0f;
            const float valMin = (row.size() > 3) ? value_or_default(row[3], 0.0f) : 0.0f;
            const float valMid = (row.size() > 4) ? value_or_default(row[4], 0.0f) : 0.0f;

            outProfile.dyeC.emplace_back(wl, valC);
            outProfile.dyeM.emplace_back(wl, valM);
            outProfile.dyeY.emplace_back(wl, valY);
            outProfile.baseMin.emplace_back(wl, valMin);
            outProfile.baseMid.emplace_back(wl, valMid);
        }

        // Log sensitivity curves (RGB order in the source profile).
        if (data.contains("log_sensitivity")) {
            const Json& logSens = data["log_sensitivity"];
            if (logSens.is_array()) {
                const size_t sensCount = std::min(wavelengths.size(), logSens.size());
                outProfile.logSensR.reserve(sensCount);
                outProfile.logSensG.reserve(sensCount);
                outProfile.logSensB.reserve(sensCount);
                for (size_t i = 0; i < sensCount; ++i) {
                    float wl = value_or_default(wavelengths[i], std::numeric_limits<float>::quiet_NaN());
                    if (!std::isfinite(wl)) {
                        continue;
                    }
                    const Json& row = logSens[i];
                    if (!row.is_array()) {
                        continue;
                    }
                    const float valR = (row.size() > 0) ? value_or_default(row[0], -10.0f) : -10.0f;
                    const float valG = (row.size() > 1) ? value_or_default(row[1], -10.0f) : -10.0f;
                    const float valB = (row.size() > 2) ? value_or_default(row[2], -10.0f) : -10.0f;
                    outProfile.logSensR.emplace_back(wl, valR);
                    outProfile.logSensG.emplace_back(wl, valG);
                    outProfile.logSensB.emplace_back(wl, valB);
                }
            }
        }

        // Density curves (log exposure domain shared by all channels).
        if (!data.contains("log_exposure") || !data.contains("density_curves")) {
            return false;
        }
        const Json& logExposure = data["log_exposure"];
        const Json& densityCurves = data["density_curves"];
        if (!logExposure.is_array() || !densityCurves.is_array()) {
            return false;
        }
        const size_t curveCount = std::min(logExposure.size(), densityCurves.size());
        outProfile.densityCurveB.reserve(curveCount);
        outProfile.densityCurveG.reserve(curveCount);
        outProfile.densityCurveR.reserve(curveCount);
        for (size_t i = 0; i < curveCount; ++i) {
            float logE = value_or_default(logExposure[i], std::numeric_limits<float>::quiet_NaN());
            if (!std::isfinite(logE)) {
                continue;
            }
            const Json& row = densityCurves[i];
            if (!row.is_array()) {
                continue;
            }
            const float valR = (row.size() > 0) ? value_or_default(row[0], 0.0f) : 0.0f;
            const float valG = (row.size() > 1) ? value_or_default(row[1], 0.0f) : 0.0f;
            const float valB = (row.size() > 2) ? value_or_default(row[2], 0.0f) : 0.0f;
            // JSON stores curves in RGB order; Juicer keeps them as B/G/R for sampling.
            outProfile.densityCurveR.emplace_back(logE, valR);
            outProfile.densityCurveG.emplace_back(logE, valG);
            outProfile.densityCurveB.emplace_back(logE, valB);
        }

        // Require the core spectral assets to be present.
        const bool haveDyes = !outProfile.dyeY.empty();
        const bool haveCurves = !outProfile.densityCurveB.empty();
        const bool haveSens = !outProfile.logSensR.empty() &&
            !outProfile.logSensG.empty() && !outProfile.logSensB.empty();
        return haveDyes && haveCurves && haveSens;
    }

} // namespace Profiles