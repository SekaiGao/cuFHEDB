#pragma once
#include "FHE/fft/fft_gpu.cuh"
#include "FHE/fft/ifft_gpu.cuh"

namespace cufhedb {

template <class P>
__device__ inline double4 DecompositionPolynomial(uint4 *val, const int &idx, const int &digit) {
  constexpr typename P::T totaloffset = 2181562368 + 8192;
  constexpr typename P::T digits = 32;
  constexpr typename P::T mask = 63;
  constexpr typename P::T halfBg = 32;
  constexpr typename P::T Bgbit = 6;
  register typename P::T shiftbits = digits - (digit + 1) * Bgbit;

  register uint4 temp = val[idx];
  register double4 res;
  res.x = __int2double_rn((((temp.x + totaloffset) >> shiftbits) & mask) - halfBg);
  res.y = __int2double_rn((((temp.y + totaloffset) >> shiftbits) & mask) - halfBg);
  res.z = __int2double_rn((((temp.z + totaloffset) >> shiftbits) & mask) - halfBg);
  res.w = __int2double_rn((((temp.w + totaloffset) >> shiftbits) & mask) - halfBg);

  return res;
}

template <class P>
__device__ inline double4 DecompositionPolynomial(uint64_t *val, const int &digit) {
  constexpr typename P::T totaloffset = 9241421688455823360 + 134217728;
  constexpr typename P::T digits = 64;
  constexpr typename P::T mask = 511;
  constexpr typename P::T halfBg = 256;
  constexpr typename P::T Bgbit = 9;
  register typename P::T shiftbits = digits - (digit + 1) * Bgbit;

  register double4 res;
  res.x = __ll2double_rn((((val[0] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.y = __ll2double_rn((((val[1] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.z = __ll2double_rn((((val[2] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.w = __ll2double_rn((((val[3] + totaloffset) >> shiftbits) & mask) - halfBg);

  return res;
}

// trgswfft * polyfft
__device__ inline void MulInFD(double4 *ifftb, double4 *pre, double4 *pim, const int &idx) {
  double4 *__restrict__ ifftb_pre = ifftb;
  double4 *__restrict__ ifftb_pim = (double4 *)(ifftb_pre + 128);

  // CplxMul
  register double4 re0, re1, im0, im1;
  #pragma unroll 2
  for (int i = 0; i < 2; ++i) {
    int idx0 = (idx << 1) + i;
    re0 = pre[idx0];
    im0 = pim[idx0];

    re1 = ifftb_pre[idx0];
    im1 = ifftb_pim[idx0];

    // trgswffti0 * decpolyfft
    CplxMul(re0, im0, re1, im1);

    pre[idx0] = re1;
    pim[idx0] = im1;
  }
}

__device__ inline void MulInFD2(double4 *trgswffti, double4 *pre, double4 *pim, const int &idx) {
  double4 *__restrict__ trgswffti0_pre = trgswffti;
  double4 *__restrict__ trgswffti0_pim = (double4 *)(trgswffti0_pre + 128);
  double4 *__restrict__ trgswffti1_pre = (double4 *)(trgswffti0_pim + 128);
  double4 *__restrict__ trgswffti1_pim = (double4 *)(trgswffti1_pre + 128);

  // CplxMul
  register double4 re0, re1, im0, im1;
  #pragma unroll 2
  for (int i = 0; i < 2; ++i) {
    int idx0 = (idx << 1) + i;
    re0 = pre[idx0];
    im0 = pim[idx0];

    re1 =trgswffti0_pre[idx0];
    im1 = trgswffti0_pim[idx0]; 

    // trgswffti0 * decpolyfft
    CplxMul(re0, im0, re1, im1);

    pre[idx0] = re1;
    pim[idx0] = im1;

    re1 = trgswffti1_pre[idx0];
    im1 = trgswffti1_pim[idx0]; 

    // trgswffti1 * decpolyfft
    CplxMul(re0, im0, re1, im1);

    pre[idx0 + 128] = re1;
    pim[idx0 + 128] = im1;
  }
}

__device__ inline void MulInFD64(double4 *trgswffti, double4 *pre, double4 *pim, const int &idx) {
  double4 *__restrict__ trgswffti0_pre = trgswffti;
  double4 *__restrict__ trgswffti0_pim = (double4 *)(trgswffti0_pre + 256);
  double4 *__restrict__ trgswffti1_pre = (double4 *)(trgswffti0_pim + 256);
  double4 *__restrict__ trgswffti1_pim = (double4 *)(trgswffti1_pre + 256);

  // CplxMul
  register double4 re0, re1, im0, im1;
  #pragma unroll 2
  for (int i = 0; i < 2; ++i) {
    int idx0 = (idx << 1) + i;
    re0 = pre[idx0];
    im0 = pim[idx0];

    re1 = trgswffti0_pre[idx0];
    im1 = trgswffti0_pim[idx0];

    // trgswffti0 * decpolyfft
    CplxMul(re0, im0, re1, im1);

    pre[idx0] = re1;
    pim[idx0] = im1;

    re1 = trgswffti1_pre[idx0];
    im1 = trgswffti1_pim[idx0];

    // trgswffti1 * decpolyfft
    CplxMul(re0, im0, re1, im1);

    pre[idx0 + 256] = re1;
    pim[idx0 + 256] = im1;
  }
}
};