// SPDX-License-Identifier: MIT
// Offline checks for the candidate shape gate and the recurrence/blocked algebra.
#include "chunk_gated_delta_net_policy.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

constexpr int tokens = 256, tile = 16, width = 4;
using Vec = std::array<double, width>;
using Seq = std::array<Vec, tokens>;
using Scalar = std::array<double, tokens>;
using State = std::array<Vec, width>;
struct Result { Seq out; State state; };

double dot(const Vec & a, const Vec & b) {
    double v = 0;
    for (int i = 0; i < width; ++i) v += a[i] * b[i];
    return v;
}

Result recurrent(const Seq & q, const Seq & k, const Seq & v, const Scalar & g, const Scalar & b, State s) {
    Result r{};
    for (int t = 0; t < tokens; ++t) {
        const double d = std::exp(g[t]);
        for (int h = 0; h < width; ++h) {
            auto & row = s[h];
            const double err = b[t] * (v[t][h] - d * dot(row, k[t]));
            for (int j = 0; j < width; ++j) row[j] = d * row[j] + err * k[t][j];
            r.out[t][h] = dot(row, q[t]);
        }
    }
    r.state = s;
    return r;
}

Result chunked(const Seq & q, const Seq & k, const Seq & v, const Scalar & g, const Scalar & b, State s) {
    Result r{};
    for (int base = 0; base < tokens; base += tile) {
        std::array<double, tile> prefix{};
        for (int i = 0; i < tile; ++i) prefix[i] = g[base+i] + (i ? prefix[i-1] : 0.0);
        for (int h = 0; h < width; ++h) {
            Vec & state = s[h];
            std::array<double, tile> innovation{};
            for (int i = 0; i < tile; ++i) {
                const int t = base+i;
                double pred = std::exp(prefix[i]) * dot(state, k[t]);
                for (int j = 0; j < i; ++j)
                    pred += std::exp(prefix[i]-prefix[j]) * dot(k[t], k[base+j]) * innovation[j];
                innovation[i] = b[t] * (v[t][h] - pred);
                double out = std::exp(prefix[i]) * dot(state, q[t]);
                for (int j = 0; j <= i; ++j)
                    out += std::exp(prefix[i]-prefix[j]) * dot(q[t], k[base+j]) * innovation[j];
                r.out[t][h] = out;
            }
            for (int w = 0; w < width; ++w) {
                state[w] *= std::exp(prefix.back());
                for (int i = 0; i < tile; ++i)
                    state[w] += std::exp(prefix.back()-prefix[i]) * k[base+i][w] * innovation[i];
            }
        }
    }
    r.state = s;
    return r;
}

template <class A> void equivalent(const A & a, const A & b) {
    for (size_t i=0; i<a.size(); ++i)
        for (int j=0; j<width; ++j) assert(std::fabs(a[i][j]-b[i][j]) < 1e-12);
}

int main() {
    static_assert(ggml_cdna2_gdn_shape(true, true, 256, 1, 1, 8, 24, 128, 128, 1));
    static_assert(ggml_cdna2_gdn_shape(true, true, 256, 1, 1, 16, 48, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, false, 256, 1, 1, 8, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(false, true, 256, 1, 1, 8, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 1, 1, 1, 8, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 255, 1, 1, 8, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 256, 2, 1, 8, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 256, 1, 2, 8, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 256, 1, 1, 0, 24, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 256, 1, 1, 8, 25, 128, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 256, 1, 1, 8, 24, 64, 128, 1));
    static_assert(!ggml_cdna2_gdn_shape(true, true, 256, 1, 1, 8, 24, 128, 128, 128));
    Seq q{}, k{}, v{}, second{};
    Scalar g{}, b{};
    for (int t=0; t<tokens; ++t) {
        g[t] = -0.03 - ((t % 7) * 0.02);
        b[t] = 0.2 + ((t % 11) * 0.04);
        for (int j=0; j<width; ++j) {
            q[t][j] = 0.3 * std::sin(0.17 * (t + 2*j));
            k[t][j] = 0.3 * std::cos(0.11 * (t + 3*j));
            v[t][j] = std::sin(0.03 * (t + 5*j));
            second[t][j] = std::cos(0.07 * (t - 7*j));
        }
    }
    const Result init = recurrent(q, k, v, g, b, {});
    const Result init_blocked = chunked(q, k, v, g, b, {});
    equivalent(init.out, init_blocked.out);
    equivalent(init.state, init_blocked.state);
    const Result continuation = recurrent(q, k, second, g, b, init.state);
    const Result continued_blocked = chunked(q, k, second, g, b, init_blocked.state);
    equivalent(continuation.out, continued_blocked.out);
    equivalent(continuation.state, continued_blocked.state);
    const Result fresh = recurrent(q, k, second, g, b, {});
    const Result fresh_blocked = chunked(q, k, second, g, b, {});
    equivalent(fresh.out, fresh_blocked.out);
    equivalent(fresh.state, fresh_blocked.state);
    assert(std::fabs(fresh.out[0][0]-continued_blocked.out[0][0]) > 1e-4);
    std::cout << "dispatch + fp64 output/state reference: fresh, continued, reset PASS\n";
}
