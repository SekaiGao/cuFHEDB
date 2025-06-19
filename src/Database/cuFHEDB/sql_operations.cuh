#pragma once

#include "ARCEDB/utils/types.h"
#include "FHE/coreFHE.cuh"
#include <cstring>
#include <limits>

// this file contains the SQL operations implemented in cuFHEDB

namespace cufhedb {

// LIKE: word-wise matching via equality + OR
template <typename C, typename EK>
void likeOperator(const std::vector<C>& tokens, const C& pattern, C& out, const EK& ek, uint32_t sid) {
    equality(tokens[0], pattern, out, sid);
    for (size_t i = 1; i < tokens.size(); ++i) {
        C tmp; equality(tokens[i], pattern, tmp, sid);
        HomOR(out, tmp, out, ek, sid);
    }
}


// Join + left-filter mask
template <typename C, typename M, typename EK>
void equiJoinAndMask(const std::vector<C>& L, const std::vector<C>& R,
                     const std::vector<M>& LF, std::vector<M>& J, const EK& ek) {
    size_t n = L.size(), m = R.size(); J.resize(n * m);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        uint32_t sid = omp_get_thread_num();
        for (size_t j = 0; j < m; ++j) {
            M eq; equality(L[i], R[j], eq, sid);
            HomAND(eq, LF[i], J[i*m + j], ek, sid);
        }
    }
}


// GROUP BY equality into G groups
template <typename C, typename EK>
void groupByEquality(const std::vector<C>& keys, const std::vector<C>& preds,
                     const std::vector<C>& filt, std::vector<std::vector<C>>& out,
                     const EK& ek) {
    size_t n = filt.size(), G = preds.size();
    out.assign(G, std::vector<C>(n));
    #pragma omp parallel for
    for (size_t g = 0; g < G; ++g) {
        uint32_t sid = omp_get_thread_num();
        for (size_t i = 0; i < n; ++i) {
            C tmp; equality(keys[i], preds[g], tmp, sid);
            HomAND(tmp, filt[i], out[g][i], ek, sid);
        }
    }
}


// COUNT aggregation: sum a binary mask
template <typename C>
void aggregateCount(const std::vector<C>& mask, C& sum) {
    sum = C{};
    for (auto& m : mask) for (size_t k = 0; k <= C::n; ++k) sum[k] += m[k];
}


// ORDER BY DESC: rank = # of values greater than self
template <typename C, typename EK, typename SK>
void homomorphicOrderByDesc(const std::vector<C>& vals, std::vector<C>& idx,
                            int cmpBits, int liftBits, const EK& ek, const SK& sk) {
    size_t N = vals.size(); idx.assign(N, C{});
    for (size_t i = 0; i < N; ++i) for (size_t j = 0; j < N; ++j) if (i != j) {
        C cmp; less_than(vals[i], vals[j], cmpBits, cmp, ek, 0);
        C lifted; lift_and_and(cmp, cmp, lifted, liftBits, ek, sk);
        for (size_t k = 0; k <= C::n; ++k) idx[i][k] += lifted[k];
    }
}

};
