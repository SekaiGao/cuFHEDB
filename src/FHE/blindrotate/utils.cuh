#pragma once

#include "FHE/externalproduct/ExternalProduct_gpu.cuh"

namespace cufhedb {

constexpr int Lvl0_n = 636;
constexpr int Lvl0_n80 = 496;
using Lvl0_T = uint16_t;

template<class P>
__device__ inline double4 PolynomialMulByXaiMinusOne(uint32_t *poly, const uint32_t &a, const int &idx, const int &digit) {
  constexpr typename P::T totaloffset = 2181562368 + 8192;
  constexpr typename P::T digits = 32;
  constexpr typename P::T mask = 63;
  constexpr typename P::T halfBg = 32;
  constexpr typename P::T Bgbit = 6;
  register typename P::T shiftbits = digits - (digit + 1) * Bgbit;

  uint32_t index0 = 4 * idx;
  register uint32_t temp[4];

  // PolynomialMulByXaiMinusOne
  if (a < P::n) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      if (index < a)
        temp[i] = -poly[index - a + P::n] - poly[index];
      else
        temp[i] = poly[index - a] - poly[index];
    }
  } else {
    const typename P::T aa = a - P::n;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      if (index < aa)
        temp[i] = poly[index - aa + P::n] - poly[index];
      else
        temp[i] = -poly[index - aa] - poly[index];
    }
  }

  // DecompositionPolynomial
  register double4 res;
  res.x = __int2double_rn((((temp[0] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.y = __int2double_rn((((temp[1] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.z = __int2double_rn((((temp[2] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.w = __int2double_rn((((temp[3] + totaloffset) >> shiftbits) & mask) - halfBg);
  return res;
}

template<class P>
__device__ inline double4 PolynomialMulByXaiMinusOne(uint64_t *poly, const uint32_t &a, const int &idx, const int &digit) {
  constexpr typename P::T totaloffset = 9241421688455823360 + 134217728;
  constexpr typename P::T digits = 64;
  constexpr typename P::T mask = 511;
  constexpr typename P::T halfBg = 256;
  constexpr typename P::T Bgbit = 9;
  register typename P::T shiftbits = digits - (digit + 1) * Bgbit;

  uint32_t index0 = 4 * idx;
  register uint64_t temp[4];

  // PolynomialMulByXaiMinusOne
  if (a < P::n) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      if (index < a)
        temp[i] = -poly[index - a + P::n] - poly[index];
      else
        temp[i] = poly[index - a] - poly[index];
    }
  } else {
    const typename P::T aa = a - P::n;
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      if (index < aa)
        temp[i] = poly[index - aa + P::n] - poly[index];
      else
        temp[i] = -poly[index - aa] - poly[index];
    }
  }

  // DecompositionPolynomial
  register double4 res;
  res.x = __ll2double_rn((((temp[0] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.y = __ll2double_rn((((temp[1] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.z = __ll2double_rn((((temp[2] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.w = __ll2double_rn((((temp[3] + totaloffset) >> shiftbits) & mask) - halfBg);
  return res;
}

// TV = u+uX+...+uX^n
template<class P, uint32_t halfblk>
__device__ inline void PolynomialMulByXai_upolygen(typename P::T *res, const typename P::T &u, const uint32_t &a, const int &idx, const int &blk) {
  uint32_t index0 = 4 * idx;

  if (blk / halfblk == 0) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      res[index] = 0;
    }
  } else {
    // PolynomialMulByXai
    if (a < P::n) {
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index = index0 + i;
        if (index < a)
          res[index] = -u;
        else
          res[index] = u;
      }
    } else {
      const typename P::T aa = a - P::n;
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index = index0 + i;
        if (index < aa)
          res[index] = u;
        else
          res[index] = -u;
      }
    }
  }
}

template<class P>
__device__ inline void PolynomialMulByXaiCGu(typename P::T *res, const typename P::T &u, const uint32_t &a, const int &idx, const int&batch) {
  uint32_t index0 = 4 * idx;

  if (batch == 0) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      res[index] = 0;
    }
  } else {
    // PolynomialMulByXai
    if (a < P::n) {
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index = index0 + i;
        if (index < a)
          res[index] = -u;
        else
          res[index] = u;
      }
    } else {
      const typename P::T aa = a - P::n;
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index = index0 + i;
        if (index < aa)
          res[index] = u;
        else
          res[index] = -u;
      }
    }
  }
}

template<class P, uint32_t halfblk>
__device__ inline void PolynomialMulByXai_gpolygen(typename P::T *res, const typename P::T &scale_bits, const uint32_t &a, const int &idx, const int &blk) {
  uint32_t index0 = 4 * idx;
  constexpr typename P::T padding_bits = 6; // P :: nbit - plain_bits;
  typename P::T poly = 1ULL << scale_bits;
  if (blk / halfblk == 0) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      res[index] = 0;
    }
  } else {
    // PolynomialMulByXai
    if (a < P::n) {
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        typename P::T index = index0 + i;
        if (index < a)
          res[index] = -poly * ((index - a + P::n) >> padding_bits);
        else
          res[index] = poly * ((index - a) >> padding_bits);
      }
    } else {
      const typename P::T aa = a - P::n;
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        typename P::T index = index0 + i;
        if (index < aa)
          res[index] = poly * ((index - aa + P::n) >> padding_bits);
        else
          res[index] = -poly * ((index - aa) >> padding_bits);
      }
    }
  }
}

template<class P>
__device__ inline void PolynomialMulByXaiCGg(typename P::T *res, const typename P::T &scale_bits, const uint32_t &a, const int &idx, const int &batch) {
  uint32_t index0 = 4 * idx;
  constexpr typename P::T padding_bits = 6; // P :: nbit - plain_bits;
  typename P::T poly = 1ULL << scale_bits;
  if (batch == 0) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t index = index0 + i;
      res[index] = 0;
    }
  } else {
    // PolynomialMulByXai
    if (a < P::n) {
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        typename P::T index = index0 + i;
        if (index < a)
          res[index] = -poly * ((index - a + P::n) >> padding_bits);
        else
          res[index] = poly * ((index - a) >> padding_bits);
      }
    } else {
      const typename P::T aa = a - P::n;
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        typename P::T index = index0 + i;
        if (index < aa)
          res[index] = poly * ((index - aa + P::n) >> padding_bits);
        else
          res[index] = -poly * ((index - aa) >> padding_bits);
      }
    }
  }
}

};