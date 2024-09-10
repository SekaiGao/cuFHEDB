#pragma once
#include "ExternalProduct_gpu.cuh"

namespace cufft {

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
  res.x = int32_t((((temp[0] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.y = int32_t((((temp[1] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.z = int32_t((((temp[2] + totaloffset) >> shiftbits) & mask) - halfBg);
  res.w = int32_t((((temp[3] + totaloffset) >> shiftbits) & mask) - halfBg);
  return res;
}


template<class P>
__device__ inline void PolynomialMulByXai_upolygen(uint32_t *res, const uint32_t &u, const uint32_t &a, const int &idx, const int &blk) {
  uint32_t index0 = 4 * idx;

  if (blk / 3 == 0) {
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
__device__ inline void PolynomialMulByXai_gpolygen(uint32_t *res, const uint32_t &scale_bits, const uint32_t &a, const int &idx, const int &blk) {
  uint32_t index0 = 4 * idx;
  constexpr uint32_t padding_bits = 6;// P :: nbit - plain_bits;
  uint32_t poly = 1ULL << scale_bits; 
  if (blk / 3 == 0) {
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
          res[index] = -poly * ((index - a + P::n) >> padding_bits);
        else
          res[index] = poly * ((index - a) >> padding_bits);
      }
    } else {
      const typename P::T aa = a - P::n;
      #pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index = index0 + i;
        if (index < aa)
          res[index] = poly * ((index - aa + P::n) >> padding_bits);
        else
          res[index] = -poly * ((index - aa) >> padding_bits);
      }
    }
  }
}

template<class P>
__global__ void __launch_bounds__(64, 6) CMUXFFTwithPolynomialMulByXaiMinusOne(uint32_t *trlwe, double *trgswfft, const uint32_t a, double *buf, const int32_t Ns2) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int l = blk % 3;
  int idx0 = idx << 1;

  int offset = (blk/3) ? 1024 : 0;
  
  // in
  uint32_t *__restrict__ in_rev_dre = (uint32_t *)(trlwe + offset);
  
  // (X^a-1)*acc[i]

  // load to SMEM
  shared_pre[idx0] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0, l);
  shared_pre[idx0 + 1] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 1, l);
  shared_pim[idx0] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 128, l);
  shared_pim[idx0 + 1] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 129, l);
  
  // BK[i]*((X^a-1)*acc[i])

  // IFFT
  ifft1024(shared_pre, shared_pim, Ns2, idx);

  double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 2048 * blk);
  
  // MulInFD
  MulInFD(trgswffti, shared_pre, shared_pim, idx);

  // load to GMEM
  double4 *bufi = (double4 *)(buf + 2048 * blk);

  bufi[idx0] = shared_pre[idx0];
  bufi[idx0 + 1] = shared_pre[idx0 + 1];
  bufi[idx0 + 128] = shared_pim[idx0];
  bufi[idx0 + 129] = shared_pim[idx0 + 1];

  bufi[idx0 + 256] = shared_pre[idx0 + 128];
  bufi[idx0 + 257] = shared_pre[idx0 + 129];
  bufi[idx0 + 384] = shared_pim[idx0 + 128];
  bufi[idx0 + 385] = shared_pim[idx0 + 129];

  // lock-free inter-block sync
  __syncblocks(1, Syncin1, Syncout1);

  if (blk < 2) {

    // sum trlwefft
    uint32_t sumoffset = blk ? 1024 : 0;
    double4 *bufblk = reinterpret_cast<double4 *>(buf + sumoffset);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int idx0 = 2 * idx + i;
      // real part
      register double4 buf0 = bufblk[idx0];
      register double4 buf1 = bufblk[idx0 + 512];
      register double4 buf2 = bufblk[idx0 + 2 * 512];
      register double4 buf3 = bufblk[idx0 + 3 * 512];
      register double4 buf4 = bufblk[idx0 + 4 * 512];
      register double4 buf5 = bufblk[idx0 + 5 * 512];

      register double4 sumr, sumi;
      sumr.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x;
      sumr.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y;
      sumr.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z;
      sumr.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w;

      //imag part
      buf0 = bufblk[idx0 + 128];
      buf1 = bufblk[idx0 + 512 + 128];
      buf2 = bufblk[idx0 + 2 * 512 + 128];
      buf3 = bufblk[idx0 + 3 * 512 + 128];
      buf4 = bufblk[idx0 + 4 * 512 + 128];
      buf5 = bufblk[idx0 + 5 * 512 + 128];

      sumi.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x;
      sumi.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y;
      sumi.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z;
      sumi.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w;
      
      // to SMEM
      shared_pre[idx0] = sumr;
      shared_pim[idx0] = sumi;
    }

    // out
    uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(trlwe + sumoffset);
    uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);

    // fft1024(restrlwefft)
    fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }
}

//BlindRotate
template<class P>
__global__ void __launch_bounds__(64, 6) BlindRotate(uint32_t *trlwe, uint32_t *tlwe, double *BK, const uint32_t u, double *buf, const int32_t Ns2) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int l = blk % 3;
  int idx0 = idx << 1;

  int offset = (blk/3) ? 1024 : 0;
  uint32_t sumoffset = blk ? 1024 : 0;

  // in
  uint32_t *__restrict__ in_rev_dre = (uint32_t *)(trlwe + offset);
  // out
  uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(trlwe + sumoffset);
  uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);


  constexpr uint32_t roundoffset = 1048576;
  constexpr uint32_t trgswlen = 6 * 2 * 1024;
  const uint32_t b = 2048 - (tlwe[672] >> 21);

  PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0, blk);
  PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0 + 1, blk);
  PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0 + 128, blk);
  PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0 + 129, blk);

  #pragma unroll
  for (int i0 = 0; i0 < 672; ++i0) {
  
    const uint32_t a = (tlwe[i0] + roundoffset) >> 21;

    if (a == 0)
      continue;

    double *trgswfft = (double *)(BK + i0 * trgswlen);

    // (X^a-1)*acc[i]

    // load to SMEM
    shared_pre[idx0] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0, l);
    shared_pre[idx0 + 1] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 1, l);
    shared_pim[idx0] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 128, l);
    shared_pim[idx0 + 1] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 129, l);
  
    // BK[i]*((X^a-1)*acc[i])

    // IFFT
    ifft1024(shared_pre, shared_pim, Ns2, idx);

    double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 2048 * blk);
  
    // MulInFD
    MulInFD(trgswffti, shared_pre, shared_pim, idx);

    // load to GMEM
    double4 *bufi = (double4 *)(buf + 2048 * blk);

    bufi[idx0] = shared_pre[idx0];
    bufi[idx0 + 1] = shared_pre[idx0 + 1];
    bufi[idx0 + 128] = shared_pim[idx0];
    bufi[idx0 + 129] = shared_pim[idx0 + 1];

    bufi[idx0 + 256] = shared_pre[idx0 + 128];
    bufi[idx0 + 257] = shared_pre[idx0 + 129];
    bufi[idx0 + 384] = shared_pim[idx0 + 128];
    bufi[idx0 + 385] = shared_pim[idx0 + 129];

    // lock-free inter-block sync
    __syncblocks(i0 + 2, Syncin1, Syncout1);

    if (blk < 2) {
      // sum of trlwefft(polynomial addition)
      double4 *bufblk = reinterpret_cast<double4 *>(buf + sumoffset);
      register double4 buf0, buf1, buf2, buf3, buf4, buf5;
      register double4 sumr, sumi;
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        int idx1 = idx0 + i;
        // real part
        buf0 = bufblk[idx1];
        buf1 = bufblk[idx1 + 512];
        buf2 = bufblk[idx1 + 1024];
        buf3 = bufblk[idx1 + 1536];
        buf4 = bufblk[idx1 + 2048];
        buf5 = bufblk[idx1 + 2560];

        sumr.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x;
        sumr.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y;
        sumr.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z;
        sumr.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w;

        // imag part
        buf0 = bufblk[idx1 + 128];
        buf1 = bufblk[idx1 + 640];
        buf2 = bufblk[idx1 + 1152];
        buf3 = bufblk[idx1 + 1664];
        buf4 = bufblk[idx1 + 2176];
        buf5 = bufblk[idx1 + 2688];

        sumi.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x;
        sumi.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y;
        sumi.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z;
        sumi.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w;

        // to SMEM
        shared_pre[idx1] = sumr;
        shared_pim[idx1] = sumi;
      }

      // fft1024(restrlwefft)
      fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
    }

    __syncblocks(i0 + 1, Syncin1, Syncout1);
  }
}

template<class P>
__device__ inline void SampleExtractIndex(uint32_t *tlwe, uint32_t *trlwe, const int &idx) {

  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    int index = idx + j * 64;
    tlwe[index] = -trlwe[P::n - index];
  }
  tlwe[0] = trlwe[0];
  tlwe[P::n] = trlwe[P::n];
}

template<class P>
__global__ void __launch_bounds__(64, 6) GateBootstrappingTLWE2TLWEFFT(uint32_t *res, uint32_t *trlwe, uint32_t *tlwe, double *BK, const uint32_t u, double *buf, const int32_t Ns2, volatile int *Syncin, volatile int *Syncout, bool isIde) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int l = blk % 3;
  int idx0 = idx << 1;

  int offset = (blk/3) ? 1024 : 0;
  uint32_t sumoffset = blk ? 1024 : 0;

  // in
  uint32_t *__restrict__ in_rev_dre = (uint32_t *)(trlwe + offset);
  // out
  uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(trlwe + sumoffset);
  uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);


  constexpr uint32_t roundoffset = 1048576;
  constexpr uint32_t trgswlen = 6 * 2 * 1024;
  const uint32_t b = 2048 - (tlwe[672] >> 21);

  if (isIde) {
    PolynomialMulByXai_gpolygen<P>(in_rev_dre, u, b, idx0, blk);
    PolynomialMulByXai_gpolygen<P>(in_rev_dre, u, b, idx0 + 1, blk);
    PolynomialMulByXai_gpolygen<P>(in_rev_dre, u, b, idx0 + 128, blk);
    PolynomialMulByXai_gpolygen<P>(in_rev_dre, u, b, idx0 + 129, blk);
  } else {
    PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0, blk);
    PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0 + 1, blk);
    PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0 + 128, blk);
    PolynomialMulByXai_upolygen<P>(in_rev_dre, u, b, idx0 + 129, blk);
  }

  #pragma unroll
  for (int i = 0; i < 672; ++i) {
  
    const uint32_t a = (tlwe[i] + roundoffset) >> 21;

    if (a == 0)
      continue;

    // (X^a-1)*acc[i]

    // load to SMEM
    shared_pre[idx0] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0, l);
    shared_pre[idx0 + 1] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 1, l);
    shared_pim[idx0] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 128, l);
    shared_pim[idx0 + 1] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx0 + 129, l);
  
    // BK[i]*((X^a-1)*acc[i])

    // IFFT
    ifft1024(shared_pre, shared_pim, Ns2, idx);

    double4 *__restrict__ trgswffti = (double4 *)(BK + i * trgswlen + 2048 * blk);
  
    // MulInFD
    MulInFD(trgswffti, shared_pre, shared_pim, idx);

    // load to GMEM
    double4 *bufi = (double4 *)(buf + 2048 * blk);

    bufi[idx0] = shared_pre[idx0];
    bufi[idx0 + 1] = shared_pre[idx0 + 1];
    bufi[idx0 + 128] = shared_pim[idx0];
    bufi[idx0 + 129] = shared_pim[idx0 + 1];

    bufi[idx0 + 256] = shared_pre[idx0 + 128];
    bufi[idx0 + 257] = shared_pre[idx0 + 129];
    bufi[idx0 + 384] = shared_pim[idx0 + 128];
    bufi[idx0 + 385] = shared_pim[idx0 + 129];

    // lock-free inter-block sync
    __syncblocks(i + 2, Syncin, Syncout);

    if (blk < 2) {
      // sum of trlwefft(polynomial addition)
      double4 *bufblk = reinterpret_cast<double4 *>(buf + sumoffset);
      register double4 buf0, buf1, buf2, buf3, buf4, buf5;
      register double4 sumr, sumi;
      #pragma unroll
      for (int i0 = 0; i0 < 2; ++i0) {
        int idx1 = idx0 + i0;
        // real part
        buf0 = bufblk[idx1];
        buf1 = bufblk[idx1 + 512];
        buf2 = bufblk[idx1 + 1024];
        buf3 = bufblk[idx1 + 1536];
        buf4 = bufblk[idx1 + 2048];
        buf5 = bufblk[idx1 + 2560];

        sumr.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x;
        sumr.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y;
        sumr.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z;
        sumr.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w;

        // imag part
        buf0 = bufblk[idx1 + 128];
        buf1 = bufblk[idx1 + 640];
        buf2 = bufblk[idx1 + 1152];
        buf3 = bufblk[idx1 + 1664];
        buf4 = bufblk[idx1 + 2176];
        buf5 = bufblk[idx1 + 2688];

        sumi.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x;
        sumi.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y;
        sumi.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z;
        sumi.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w;

        // to SMEM
        shared_pre[idx1] = sumr;
        shared_pim[idx1] = sumi;
      }

      // fft1024(restrlwefft)
      fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
    }

    __syncblocks(i + 1, Syncin, Syncout);
  }

  if (blk == 0) {
    SampleExtractIndex<P>(res, trlwe, idx);
  }
}

};