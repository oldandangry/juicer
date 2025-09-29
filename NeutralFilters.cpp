#include "NeutralFilters.h"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

// Naive text search parser, tailored to the known json schema in agx-emulsion.
// This avoids external dependencies and keeps the parsing deterministic.
static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool load_enlarger_neutral_filters(
    const std::string& jsonPath,
    const std::string& paperKey,
    const std::string& illuminantKey,
    const std::string& negativeKey,
    std::tuple<float, float, float>& outYMC)
{
    std::ifstream f(jsonPath);
    if (!f.is_open()) return false;
    std::ostringstream oss;
    oss << f.rdbuf();
    std::string s = oss.str();

    // Normalize for case-insensitive search safety on keys.
    std::string S = to_lower(s);
    std::string kPaper = to_lower(paperKey);
    std::string kIll = to_lower(illuminantKey);
    std::string kNeg = to_lower(negativeKey);

    auto find_key_after = [&](const std::string& hay, size_t pos, const std::string& key)->size_t {
        const std::string needle = "\"" + key + "\"";
        return hay.find(needle, pos);
        };

    size_t p0 = find_key_after(S, 0, kPaper);
    if (p0 == std::string::npos) return false;
    size_t p1 = find_key_after(S, p0, kIll);
    if (p1 == std::string::npos) return false;
    size_t p2 = find_key_after(S, p1, kNeg);
    if (p2 == std::string::npos) return false;

    size_t arrStart = S.find('[', p2);
    if (arrStart == std::string::npos) return false;
    size_t arrEnd = S.find(']', arrStart);
    if (arrEnd == std::string::npos) return false;

    std::string arr = s.substr(arrStart + 1, arrEnd - arrStart - 1);
    float vals[3] = { 0.f, 0.f, 0.f };
    int count = 0;
    size_t cur = 0;
    while (cur < arr.size() && count < 3) {
        while (cur < arr.size() && std::isspace((unsigned char)arr[cur])) ++cur;
        size_t next = arr.find(',', cur);
        std::string tok;
        if (next == std::string::npos) {
            tok = arr.substr(cur);
            cur = arr.size();
        }
        else {
            tok = arr.substr(cur, next - cur);
            cur = next + 1;
        }
        size_t b = 0, e = tok.size();
        while (b < e && std::isspace((unsigned char)tok[b])) ++b;
        while (e > b && std::isspace((unsigned char)tok[e - 1])) --e;
        tok = tok.substr(b, e - b);
        try {
            vals[count] = std::stof(tok);
        }
        catch (...) {
            return false;
        }
        if (!std::isfinite(vals[count])) return false;
        if (vals[count] < 0.f) vals[count] = 0.f;
        if (vals[count] > 1.f) vals[count] = 1.f;
        ++count;
    }
    if (count != 3) return false;

    outYMC = std::make_tuple(vals[0], vals[1], vals[2]);
    return true;
}
