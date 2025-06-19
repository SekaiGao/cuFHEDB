#pragma once
#include "FHE/blindrotate/BlindRotate_gpu.cuh"
#include <type_traits>

namespace cufhedb {

// extract LWE ciphertext from a RLWE ciphertext's constant term
template<class P>
__device__ inline void SampleExtractIndex(typename P::T *tlwe, typename P::T *trlwe, const int &idx) {

  int chunk = P::n >> 4;
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    int index = idx + j * chunk;
    tlwe[index] = -trlwe[P::n - index];
  }
  tlwe[0] = trlwe[0];
  tlwe[P::n] = trlwe[P::n];
}

__device__ inline void Load(double4 *out, double4 *in, const int &idx) {
  out[idx] = in[idx];
  out[idx + 64] = in[idx + 64];
  out[idx + 128] = in[idx + 128];
  out[idx + 192] = in[idx + 192];
}

__device__ inline void Load64(double4 *out, double4 *in, const int &idx) {
  out[idx] = in[idx];
  out[idx + 128] = in[idx + 128];
  out[idx + 256] = in[idx + 256];
  out[idx + 384] = in[idx + 384];
}

// Fused Multiply-Add in Fourier Domain
__device__ inline void FMAInFD2(double4 *sres0, double4 *sres1, double4 *ga2, double4 *gb, const int &idx) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int base_idx = idx + (i << 6);

    double4 re0 = gb[base_idx];
    double4 im0 = gb[base_idx + 128];

    double4 re1 = ga2[base_idx];
    double4 im1 = ga2[base_idx + 128];
    CplxMul(re0, im0, re1, im1);

    double4 re2 = sres0[base_idx];
    double4 im2 = sres0[base_idx + 128];

    re1.x += re2.x;
    re1.y += re2.y;
    re1.z += re2.z;
    re1.w += re2.w;
    im1.x += im2.x;
    im1.y += im2.y;
    im1.z += im2.z;
    im1.w += im2.w;

    sres0[base_idx] = re1;
    sres0[base_idx + 128] = im1;

    re1 = ga2[base_idx + 256];
    im1 = ga2[base_idx + 384];
    CplxMul(re0, im0, re1, im1);

    re2 = sres1[base_idx];
    im2 = sres1[base_idx + 128];

    re1.x += re2.x;
    re1.y += re2.y;
    re1.z += re2.z;
    re1.w += re2.w;
    im1.x += im2.x;
    im1.y += im2.y;
    im1.z += im2.z;
    im1.w += im2.w;

    sres1[base_idx] = re1;
    sres1[base_idx + 128] = im1;
  }
}

__device__ inline void FMAInFD64(double4 *sres0, double4 *sres1, double4 *ga2, double4 *gb, const int &idx) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int base_idx = idx + (i << 7);

    double4 re0 = gb[base_idx];
    double4 im0 = gb[base_idx + 256];

    double4 re1 = ga2[base_idx];
    double4 im1 = ga2[base_idx + 256];
    CplxMul(re0, im0, re1, im1);

    double4 re2 = sres0[base_idx];
    double4 im2 = sres0[base_idx + 256];

    re1.x += re2.x;
    re1.y += re2.y;
    re1.z += re2.z;
    re1.w += re2.w;
    im1.x += im2.x;
    im1.y += im2.y;
    im1.z += im2.z;
    im1.w += im2.w;

    sres0[base_idx] = re1;
    sres0[base_idx + 256] = im1;

    re1 = ga2[base_idx + 512];
    im1 = ga2[base_idx + 768];
    CplxMul(re0, im0, re1, im1);

    re2 = sres1[base_idx];
    im2 = sres1[base_idx + 256];

    re1.x += re2.x;
    re1.y += re2.y;
    re1.z += re2.z;
    re1.w += re2.w;
    im1.x += im2.x;
    im1.y += im2.y;
    im1.z += im2.z;
    im1.w += im2.w;

    sres1[base_idx] = re1;
    sres1[base_idx + 256] = im1;
  }
}

template<class P>
__global__ void PolynomialMulByXai(uint32_t *res, const uint32_t u, const uint32_t a) {
  int idx = threadIdx.x; // Thread index
  int ns8 = P::n >> 3;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (idx < ns8) {
      int index = idx + j * ns8;
      uint32_t temp;
      if (a == 0) {
        temp = u;
      } else if (a < P::n) {
        if (index < a) {
          temp = -u;
        } else {
          temp = u;
        }
      } else {
        const uint32_t aa = a - P::n;
        if (index < aa) {
          temp = u;
        } else {
          temp = -u;
        }
      }
      res[index] = 0;
      res[index + P::n] = temp;
    }
  }
}

__global__ void SampleExtractIndex_Kernel(uint32_t *tlwe_d, uint32_t *inout_direct_d, const int32_t N) {
  int idx = threadIdx.x;
  int ns8 = N/8;
  if (idx >= ns8) // 128
    return;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    int index = idx + j * ns8;
    if (index != 0) {
      tlwe_d[index] = -inout_direct_d[N - index];
    } else {
      tlwe_d[0] = inout_direct_d[0];
      tlwe_d[N] = inout_direct_d[N];
    }
  }
}
};