#include "ProfileJSONLoader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <string_view>

#include "nlohmann/json.hpp"

namespace Profiles {
    namespace {

        using Json = nlohmann::json;

        bool is_identifier_char(char c) {
            return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
        }

        void sanitize_non_finite_literals(std::string& text) {
            static constexpr std::array<std::pair<std::string_view, std::string_view>, 15> kReplacements{ {
                { "-Infinity", "null" },
                { "+Infinity", "null" },
                { "Infinity", "null" },
                { "-INF", "null" },
                { "+INF", "null" },
                { "INF", "null" },
                { "-inf", "null" },
                { "+inf", "null" },
                { "inf", "null" },
                { "-NaN", "null" },
                { "+NaN", "null" },
                { "NaN", "null" },
                { "-nan", "null" },
                { "+nan", "null" },
                { "nan", "null" }
            } };

            bool inString = false;
            bool escaping = false;

            for (std::size_t i = 0; i < text.size();) {
                const char c = text[i];
                if (inString) {
                    if (escaping) {
                        escaping = false;
                        ++i;
                        continue;
                    }
                    if (c == '\\') {
                        escaping = true;
                        ++i;
                        continue;
                    }
                    if (c == '"') {
                        inString = false;
                    }
                    ++i;
                    continue;
                }

                if (c == '"') {
                    inString = true;
                    ++i;
                    continue;
                }

                bool replaced = false;
                for (const auto& [token, replacement] : kReplacements) {
                    if (text.compare(i, token.size(), token) == 0) {
                        const std::size_t end = i + token.size();
                        const bool hasPrev = i > 0 && is_identifier_char(text[i - 1]);
                        const bool hasNext = end < text.size() && is_identifier_char(text[end]);
                        if (!hasPrev && !hasNext) {
                            text.replace(i, token.size(), replacement);
                            i += replacement.size();
                            replaced = true;
                            break;
                        }
                    }
                }
                if (!replaced) {
                    ++i;
                }
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

            // agx-emulsion profiles encode missing spectral samples as non-finite literals.
            // Replace with JSON null so the parser can ingest the document.
            sanitize_non_finite_literals(text);

            out = Json::parse(text, nullptr, false);
            return !out.is_discarded();
        }

        std::optional<float> parse_optional_float(const Json& node) {
            if (node.is_number_float() || node.is_number_integer()) {
                float v = static_cast<float>(node.get<double>());
                if (std::isfinite(v)) {
                    return v;
                }
            }
            else if (node.is_string()) {
                const std::string& str = node.get_ref<const std::string&>();
                const char* begin = str.c_str();
                char* end = nullptr;
                errno = 0;
                const float v = std::strtof(begin, &end);
                if (end != begin && end == begin + str.size() && errno == 0 && std::isfinite(v)) {
                    return v;
                }
            }
            else if (node.is_boolean()) {
                return node.get<bool>() ? 1.0f : 0.0f;
            }
            return std::nullopt;
        }

        void append_pair_if_present(std::vector<std::pair<float, float>>& target, float x, const Json& node) {
            if (auto opt = parse_optional_float(node)) {
                target.emplace_back(x, *opt);
            }
        }

        void parse_dir_couplers(const Json& node, DirCouplersProfile& outProfile) {
            outProfile = DirCouplersProfile{};
            if (!node.is_object()) {
                return;
            }

            bool any = false;

            if (node.contains("active") && node["active"].is_boolean()) {
                outProfile.active = node["active"].get<bool>();
                any = true;
            }
            if (auto amount = parse_optional_float(node.value("amount", Json{}))) {
                outProfile.amount = *amount;
                any = true;
            }
            if (node.contains("ratio_rgb") && node["ratio_rgb"].is_array()) {
                const Json& arr = node["ratio_rgb"];
                for (size_t i = 0; i < std::min<size_t>(3, arr.size()); ++i) {
                    if (auto val = parse_optional_float(arr[i])) {
                        outProfile.ratioRGB[i] = *val;
                        any = true;
                    }
                }
            }
            if (auto diff = parse_optional_float(node.value("diffusion_interlayer", Json{}))) {
                outProfile.diffusionInterlayer = *diff;
                any = true;
            }
            if (auto diffSize = parse_optional_float(node.value("diffusion_size_um", Json{}))) {
                outProfile.diffusionSizeUm = *diffSize;
                any = true;
            }
            if (auto high = parse_optional_float(node.value("high_exposure_shift", Json{}))) {
                outProfile.highExposureShift = *high;
                any = true;
            }

            outProfile.hasData = any;
        }

        void parse_masking_couplers(const Json& node, MaskingCouplersProfile& outProfile) {
            outProfile = MaskingCouplersProfile{};
            if (!node.is_object()) {
                return;
            }

            bool any = false;

            if (node.contains("cross_over_points") && node["cross_over_points"].is_array()) {
                const Json& arr = node["cross_over_points"];
                outProfile.crossOverPoints.clear();
                for (const auto& v : arr) {
                    if (auto val = parse_optional_float(v)) {
                        outProfile.crossOverPoints.push_back(*val);
                        any = true;
                    }
                }
            }

            if (node.contains("transition_widths") && node["transition_widths"].is_array()) {
                const Json& arr = node["transition_widths"];
                outProfile.transitionWidths.clear();
                for (const auto& v : arr) {
                    if (auto val = parse_optional_float(v)) {
                        outProfile.transitionWidths.push_back(*val);
                        any = true;
                    }
                }
            }

            if (node.contains("gaussian_model") && node["gaussian_model"].is_array()) {
                const Json& gm = node["gaussian_model"];
                for (size_t ch = 0; ch < std::min<size_t>(3, gm.size()); ++ch) {
                    const Json& channel = gm[ch];
                    auto& dest = outProfile.gaussianModel[ch];
                    dest.clear();
                    if (!channel.is_array()) {
                        continue;
                    }
                    for (const auto& peak : channel) {
                        if (!peak.is_array() || peak.size() < 3) {
                            continue;
                        }
                        std::array<float, 3> triplet{ 0.0f, 0.0f, 0.0f };
                        bool ok = true;
                        for (size_t k = 0; k < 3; ++k) {
                            if (auto val = parse_optional_float(peak[k])) {
                                triplet[k] = *val;
                            }
                            else {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            dest.push_back(triplet);
                            any = true;
                        }
                    }
                }
            }

            outProfile.hasData = any;
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
                        if (auto val = parse_optional_float(v)) {
                            outProfile.densityMidNeutral.push_back(*val);
                        }
                    }
                }
                else if (auto val = parse_optional_float(mid)) {
                    outProfile.densityMidNeutral.push_back(*val);
                }
            }
        }

        if (!root.contains("data")) {
            return false;
        }
        const Json& data = root["data"];

        if (data.contains("tune")) {
            const Json& tune = data["tune"];
            if (tune.contains("dye_density_min_factor")) {
                if (auto val = parse_optional_float(tune["dye_density_min_factor"])) {
                    outProfile.dyeDensityMinFactor = *val;
                }
            }
            if (tune.contains("gamma_factor")) {
                const Json& gf = tune["gamma_factor"];
                std::array<float, 3> gamma{ {1.0f, 1.0f, 1.0f} };
                bool any = false;
                if (auto scalar = parse_optional_float(gf)) {
                    const float v = *scalar;
                    if (std::isfinite(v) && v > 0.0f) {
                        gamma.fill(v);
                        any = true;
                    }
                }
                else if (gf.is_array()) {
                    const size_t count = std::min<size_t>(3, gf.size());
                    for (size_t i = 0; i < count; ++i) {
                        if (auto val = parse_optional_float(gf[i])) {
                            const float v = *val;
                            if (std::isfinite(v) && v > 0.0f) {
                                gamma[i] = v;
                                any = true;
                            }
                        }
                    }
                }
                if (any) {
                    outProfile.gammaFactor = gamma;
                    outProfile.hasGammaFactor = true;
                }
            }
        }

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
            std::optional<float> wlOpt = parse_optional_float(wavelengths[i]);
            if (!wlOpt) {
                continue;
            }
            const float wl = *wlOpt;

            const Json& row = dyeDensity[i];
            if (!row.is_array()) {
                continue;
            }

            if (row.size() > 0) append_pair_if_present(outProfile.dyeC, wl, row[0]);
            if (row.size() > 1) append_pair_if_present(outProfile.dyeM, wl, row[1]);
            if (row.size() > 2) append_pair_if_present(outProfile.dyeY, wl, row[2]);
            if (row.size() > 3) append_pair_if_present(outProfile.baseMin, wl, row[3]);
            if (row.size() > 4) append_pair_if_present(outProfile.baseMid, wl, row[4]);
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
                    std::optional<float> wlOpt = parse_optional_float(wavelengths[i]);
                    if (!wlOpt) {
                        continue;
                    }
                    const float wl = *wlOpt;
                    const Json& row = logSens[i];
                    if (!row.is_array()) {
                        continue;
                    }
                    if (row.size() > 0) append_pair_if_present(outProfile.logSensR, wl, row[0]);
                    if (row.size() > 1) append_pair_if_present(outProfile.logSensG, wl, row[1]);
                    if (row.size() > 2) append_pair_if_present(outProfile.logSensB, wl, row[2]);
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
        outProfile.densityCurveR.reserve(curveCount);
        outProfile.densityCurveG.reserve(curveCount);
        outProfile.densityCurveB.reserve(curveCount);
        for (size_t i = 0; i < curveCount; ++i) {
            std::optional<float> logEOpt = parse_optional_float(logExposure[i]);
            if (!logEOpt) {
                continue;
            }
            const float logE = *logEOpt;
            const Json& row = densityCurves[i];
            if (!row.is_array()) {
                continue;
            }
            if (row.size() > 0) append_pair_if_present(outProfile.densityCurveR, logE, row[0]);
            if (row.size() > 1) append_pair_if_present(outProfile.densityCurveG, logE, row[1]);
            if (row.size() > 2) append_pair_if_present(outProfile.densityCurveB, logE, row[2]);
        }

        // Require the core spectral assets to be present.
        const bool haveDyes = !outProfile.dyeC.empty() &&
            !outProfile.dyeM.empty() && !outProfile.dyeY.empty();
        const bool haveCurves = !outProfile.densityCurveR.empty() &&
            !outProfile.densityCurveG.empty() && !outProfile.densityCurveB.empty();
        const bool haveSens = !outProfile.logSensR.empty() &&
            !outProfile.logSensG.empty() && !outProfile.logSensB.empty();
        if (!(haveDyes && haveCurves && haveSens)) {
            return false;
        }

        if (root.contains("dir_couplers")) {
            parse_dir_couplers(root["dir_couplers"], outProfile.dirCouplers);
        }

        if (root.contains("masking_couplers")) {
            parse_masking_couplers(root["masking_couplers"], outProfile.maskingCouplers);
        }

        return true;
    }

} // namespace Profiles