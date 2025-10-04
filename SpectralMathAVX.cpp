#include "SpectralMath.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace Spectral {

#if defined(__AVX2__)

    // Fast exp approximation for AVX2
    __m256 exp_ps_approx(__m256 x) {
        const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
        const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f);
        const __m256 c1 = _mm256_set1_ps(1.0f);
        const __m256 c2 = _mm256_set1_ps(0.499999940f);
        const __m256 c3 = _mm256_set1_ps(0.166665524f);
        const __m256 c4 = _mm256_set1_ps(0.0416573475f);
        const __m256 c5 = _mm256_set1_ps(0.00830111025f);

        __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = _mm256_fnmadd_ps(kf, ln2, x);

        __m256 r2 = _mm256_mul_ps(r, r);
        __m256 p = _mm256_fmadd_ps(c5, r, c4);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, c1);
        __m256 er = _mm256_fmadd_ps(p, r2, _mm256_add_ps(r, c1));

        __m256i ki = _mm256_cvtps_epi32(kf);
        ki = _mm256_add_epi32(ki, _mm256_set1_epi32(127));
        ki = _mm256_slli_epi32(ki, 23);
        __m256 two_k = _mm256_castsi256_ps(ki);

        return _mm256_mul_ps(er, two_k);
    }

    float hsum_avx(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 vsum = _mm_add_ps(vlow, vhigh);
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        float out;
        _mm_store_ss(&out, vsum);
        return out;
    }

    void integrate_dyes_to_XYZ_avx2(
        float dY, float dM, float dC,
        const float* epsY, const float* epsM, const float* epsC,
        const float* Ax, const float* Ay, const float* Az,
        int K,
        const BaselineCtx& base,
        float XYZ_out[3])
    {
        const __m256 kLN10 = _mm256_set1_ps(kLn10);
        const __m256 dYv = _mm256_set1_ps(dY);
        const __m256 dMv = _mm256_set1_ps(dM);
        const __m256 dCv = _mm256_set1_ps(dC);

        __m256 sumX = _mm256_setzero_ps();
        __m256 sumY = _mm256_setzero_ps();
        __m256 sumZ = _mm256_setzero_ps();

        int i = 0;
        const int step = 8;
        for (; i + step <= K; i += step) {
            __m256 eY = _mm256_loadu_ps(epsY + i);
            __m256 eM = _mm256_loadu_ps(epsM + i);
            __m256 eC = _mm256_loadu_ps(epsC + i);

            __m256 Dlambda = _mm256_fmadd_ps(dYv, eY, _mm256_fmadd_ps(dMv, eM, _mm256_mul_ps(dCv, eC)));

            if (base.hasBaseline && base.base) {
                __m256 b = _mm256_loadu_ps(base.base + i);
                __m256 scale = _mm256_set1_ps(base.scale);
                __m256 dBase = _mm256_mul_ps(scale, b);

                Dlambda = _mm256_add_ps(Dlambda, dBase);
            }

            __m256 T = exp_ps_approx(_mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_mul_ps(kLN10, Dlambda)));

            __m256 Xv = _mm256_mul_ps(T, _mm256_loadu_ps(Ax + i));
            __m256 Yv = _mm256_mul_ps(T, _mm256_loadu_ps(Ay + i));
            __m256 Zv = _mm256_mul_ps(T, _mm256_loadu_ps(Az + i));

            sumX = _mm256_add_ps(sumX, Xv);
            sumY = _mm256_add_ps(sumY, Yv);
            sumZ = _mm256_add_ps(sumZ, Zv);
        }

        float X = hsum_avx(sumX);
        float Y = hsum_avx(sumY);
        float Z = hsum_avx(sumZ);

        for (; i < K; ++i) {
            float Dlambda = dY * epsY[i] + dM * epsM[i] + dC * epsC[i];
            if (base.hasBaseline && base.base) {
                Dlambda += base.scale * base.base[i];
            }
            float T = std::exp(-kLn10 * Dlambda);
            X += T * Ax[i];
            Y += T * Ay[i];
            Z += T * Az[i];
        }

        float scale = gDeltaLambda * gInvYn;
        XYZ_out[0] = X * scale;
        XYZ_out[1] = Y * scale;
        XYZ_out[2] = Z * scale;
    }

#endif // __AVX2__

} // namespace Spectral
