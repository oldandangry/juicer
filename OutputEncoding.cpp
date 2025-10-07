#include "OutputEncoding.h"

#include <algorithm>
#include <cmath>

namespace OutputEncoding {
    namespace {

        struct Mat3 {
            float m[9];
            void mul(const float in[3], float out[3]) const {
                out[0] = m[0] * in[0] + m[1] * in[1] + m[2] * in[2];
                out[1] = m[3] * in[0] + m[4] * in[1] + m[5] * in[2];
                out[2] = m[6] * in[0] + m[7] * in[1] + m[8] * in[2];
            }
        };

        constexpr Mat3 kDWG_to_sRGB = {
            {
                1.89840485f, -0.79207266f, -0.10648889f,
               -0.16874849f,  1.48888814f, -0.32004050f,
               -0.12149930f, -0.31568951f,  1.43726326f
            }
        };

        inline float clamp01(float v) {
            if (!std::isfinite(v)) {
                return 0.0f;
            }
            if (v <= 0.0f) return 0.0f;
            if (v >= 1.0f) return 1.0f;
            return v;
        }

        inline float encode_sRGB(float v) {
            const float linear = clamp01(v);
            if (linear <= 0.0031308f) {
                return 12.92f * linear;
            }
            const float encoded = 1.055f * static_cast<float>(std::pow(linear, 1.0f / 2.4f)) - 0.055f;
            return clamp01(encoded);
        }

        inline float encode_DaVinciIntermediate(float v) {
            if (!std::isfinite(v)) {
                return 0.0f;
            }

            constexpr float kDI_A = 0.0075f;
            constexpr float kDI_B = 7.0f;
            constexpr float kDI_C = 0.07329248f;
            constexpr float kDI_M = 10.44426855f;
            constexpr float kDI_LIN_CUT = 0.00262409f;

            const float linear = std::max(0.0f, v);
            if (linear <= kDI_LIN_CUT) {
                return clamp01(linear * kDI_M);
            }

            const float encoded = (static_cast<float>(std::log2(linear + kDI_A)) + kDI_B) * kDI_C;
            return clamp01(encoded);
        }

        inline float applyEncodingChannel(ColorSpace cs, float v) {
            switch (cs) {
            case ColorSpace::sRGB:
                return encode_sRGB(v);
            case ColorSpace::DaVinciWideGamutIntermediate:
                return encode_DaVinciIntermediate(v);
            default:
                return clamp01(v);
            }
        }

        void convertFromDWG(ColorSpace cs, const float in[3], float out[3]) {
            switch (cs) {
            case ColorSpace::sRGB:
                kDWG_to_sRGB.mul(in, out);
                break;
            case ColorSpace::DaVinciWideGamutIntermediate:
                out[0] = in[0];
                out[1] = in[1];
                out[2] = in[2];
                break;
            default:
                out[0] = in[0];
                out[1] = in[1];
                out[2] = in[2];
                break;
            }
        }

        bool hasEncoding(ColorSpace cs) {
            switch (cs) {
            case ColorSpace::sRGB:
            case ColorSpace::DaVinciWideGamutIntermediate:
                return true;
            default:
                return false;
            }
        }
    }

    void applyEncoding(const Params& params, float rgb[3]) {
        float converted[3];
        convertFromDWG(params.colorSpace, rgb, converted);

        if (params.preserveLinearRange) {
            rgb[0] = converted[0];
            rgb[1] = converted[1];
            rgb[2] = converted[2];
            return;
        }

        if (params.applyCctfEncoding && hasEncoding(params.colorSpace)) {
            rgb[0] = applyEncodingChannel(params.colorSpace, converted[0]);
            rgb[1] = applyEncodingChannel(params.colorSpace, converted[1]);
            rgb[2] = applyEncodingChannel(params.colorSpace, converted[2]);
        }
        else {
            rgb[0] = clamp01(converted[0]);
            rgb[1] = clamp01(converted[1]);
            rgb[2] = clamp01(converted[2]);
        }
    }

} // namespace OutputEncoding