#include "AkimaInterpolator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Interpolation {
    namespace {
        struct AkimaPoint {
            float x;
            float y;
            size_t index;
        };

        bool build_sorted_unique_points(const std::vector<float>& x,
            const std::vector<float>& y,
            std::vector<AkimaPoint>& out) {
            out.clear();
            const size_t n = x.size();
            if (n != y.size() || n < 2) {
                return false;
            }
            out.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const float xi = x[i];
                const float yi = y[i];
                if (!std::isfinite(xi) || !std::isfinite(yi)) {
                    continue;
                }
                out.push_back({ xi, yi, i });
            }
            if (out.size() < 2) {
                return false;
            }
            std::sort(out.begin(), out.end(), [](const AkimaPoint& a, const AkimaPoint& b) {
                if (a.x == b.x) {
                    return a.index < b.index;
                }
                return a.x < b.x;
                });
            auto newEnd = std::unique(out.begin(), out.end(), [](const AkimaPoint& a, const AkimaPoint& b) {
                return a.x == b.x;
                });
            out.erase(newEnd, out.end());
            if (out.size() < 2) {
                return false;
            }
            for (size_t i = 1; i < out.size(); ++i) {
                if (out[i].x <= out[i - 1].x) {
                    return false;
                }
            }
            return true;
        }

        float hermite_segment(float x0, float x1,
            float y0, float y1,
            float m0, float m1,
            float x) {
            const double h = static_cast<double>(x1) - static_cast<double>(x0);
            if (h <= 0.0) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            const double t = (static_cast<double>(x) - static_cast<double>(x0)) / h;
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            const double h10 = t3 - 2.0 * t2 + t;
            const double h01 = -2.0 * t3 + 3.0 * t2;
            const double h11 = t3 - t2;
            const double result =
                h00 * y0 +
                h10 * h * m0 +
                h01 * y1 +
                h11 * h * m1;
            return static_cast<float>(result);
        }
    }

    bool AkimaInterpolator::build(const std::vector<float>& x, const std::vector<float>& y) {
        std::vector<AkimaPoint> pts;
        if (!build_sorted_unique_points(x, y, pts)) {
            _xs.clear();
            _ys.clear();
            _slopes.clear();
            return false;
        }

        const size_t n = pts.size();
        _xs.resize(n);
        _ys.resize(n);
        for (size_t i = 0; i < n; ++i) {
            _xs[i] = pts[i].x;
            _ys[i] = pts[i].y;
        }

        _slopes.assign(n, 0.0f);
        if (n == 2) {
            const float dx = _xs[1] - _xs[0];
            if (dx == 0.0f) {
                _xs.clear();
                _ys.clear();
                _slopes.clear();
                return false;
            }
            const float slope = (_ys[1] - _ys[0]) / dx;
            _slopes[0] = slope;
            _slopes[1] = slope;
            return true;
        }

        const size_t mCount = n + 3;
        std::vector<double> m(mCount, 0.0);
        for (size_t i = 0; i + 1 < n; ++i) {
            const double dx = static_cast<double>(_xs[i + 1]) - static_cast<double>(_xs[i]);
            if (dx <= 0.0) {
                _xs.clear();
                _ys.clear();
                _slopes.clear();
                return false;
            }
            const double dy = static_cast<double>(_ys[i + 1]) - static_cast<double>(_ys[i]);
            m[i + 2] = dy / dx;
        }
        m[1] = 2.0 * m[2] - m[3];
        m[0] = 2.0 * m[1] - m[2];
        m[mCount - 2] = 2.0 * m[mCount - 3] - m[mCount - 4];
        m[mCount - 1] = 2.0 * m[mCount - 2] - m[mCount - 3];

        std::vector<double> t(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            t[i] = 0.5 * (m[i + 3] + m[i]);
        }

        std::vector<double> dm(mCount - 1, 0.0);
        for (size_t i = 0; i + 1 < mCount; ++i) {
            dm[i] = std::abs(m[i + 1] - m[i]);
        }

        const double break_mult = 1e-9;
        std::vector<double> f1(n, 0.0);
        std::vector<double> f2(n, 0.0);
        std::vector<double> f12(n, 0.0);
        double max_f12 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            f1[i] = dm[i + 2];
            f2[i] = dm[i];
            f12[i] = f1[i] + f2[i];
            if (f12[i] > max_f12) {
                max_f12 = f12[i];
            }
        }
        const double cutoff = break_mult * max_f12;

        for (size_t i = 0; i < n; ++i) {
            if (f12[i] > cutoff) {
                const double denom = f12[i];
                const double numer = f2[i] * (m[i + 2] - m[i + 1]);
                t[i] = m[i + 1] + numer / denom;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            _slopes[i] = static_cast<float>(t[i]);
        }
        return true;
    }

    float AkimaInterpolator::evaluate(float x) const {
        if (_xs.size() < 2) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (!(x >= _xs.front() && x <= _xs.back())) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        auto it = std::upper_bound(_xs.begin(), _xs.end(), x);
        size_t idx = static_cast<size_t>(std::distance(_xs.begin(), it));
        if (idx == 0) {
            idx = 1;
        }
        if (idx >= _xs.size()) {
            idx = _xs.size() - 1;
        }
        const size_t i0 = idx - 1;
        const size_t i1 = idx;
        return hermite_segment(_xs[i0], _xs[i1], _ys[i0], _ys[i1], _slopes[i0], _slopes[i1], x);
    }

    void AkimaInterpolator::evaluate_many(const std::vector<float>& xs, std::vector<float>& out) const {
        out.resize(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            out[i] = evaluate(xs[i]);
        }
    }

}