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

        void convertFromDWG(ColorSpace cs, const float in[3], float out[3]) {
            switch (cs) {
            case ColorSpace::sRGB:
                kDWG_to_sRGB.mul(in, out);
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
            rgb[0] = encode_sRGB(converted[0]);
            rgb[1] = encode_sRGB(converted[1]);
            rgb[2] = encode_sRGB(converted[2]);
        }
        else {
            rgb[0] = clamp01(converted[0]);
            rgb[1] = clamp01(converted[1]);
            rgb[2] = clamp01(converted[2]);
        }
    }

} // namespace OutputEncoding