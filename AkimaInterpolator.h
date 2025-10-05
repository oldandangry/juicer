#pragma once

#include <vector>

namespace Interpolation {

    class AkimaInterpolator {
    public:
        bool build(const std::vector<float>& x, const std::vector<float>& y);
        float evaluate(float x) const;
        void evaluate_many(const std::vector<float>& xs, std::vector<float>& out) const;
        bool empty() const { return _xs.empty(); }
    private:
        std::vector<float> _xs;
        std::vector<float> _ys;
        std::vector<float> _slopes;
    };

}
