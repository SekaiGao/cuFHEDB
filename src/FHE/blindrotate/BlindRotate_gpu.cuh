#pragma once
#include "utils.cuh"

// code for blind rotate

namespace cufhedb {
  
template<class P>
__global__ void __launch_bounds__(64, 6) CMUXFFTwithPolynomialMulByXaiMinusOne(uint32_t *trlwe, double *trgswfft, const Lvl0_T a, double *buf, const int32_t Ns2, volatile int *Syncin, volatile int *Syncout) {
  
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
  
  // MulInFD2
  MulInFD2(trgswffti, shared_pre, shared_pim, idx);

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
  __syncblocks(1, Syncin, Syncout);

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

    // fft1024fma(restrlwefft)
    fft1024fma(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }
}

// Lvl1
// BlindRotate: acc[i+1] = BK[i]*((X^a-1)*acc[i])
template<class P>
__global__ void __launch_bounds__(64, 6) BlindRotate(uint32_t *trlwe, Lvl0_T *tlwe, double *BK, const uint32_t u, double *buf, const int32_t Ns2, volatile int *Syncin, volatile int *Syncout) {
  
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

  constexpr Lvl0_T roundoffset = 16;
  constexpr uint32_t trgswlen = 6 * 2 * 1024;
  const uint32_t b = 2048 - (tlwe[Lvl0_n] >> 5);

  PolynomialMulByXai_upolygen<P, 3>(in_rev_dre, u, b, idx0, blk);
  PolynomialMulByXai_upolygen<P, 3>(in_rev_dre, u, b, idx0 + 1, blk);
  PolynomialMulByXai_upolygen<P, 3>(in_rev_dre, u, b, idx0 + 128, blk);
  PolynomialMulByXai_upolygen<P, 3>(in_rev_dre, u, b, idx0 + 129, blk);

  #pragma unroll
  for (int i0 = 0; i0 < Lvl0_n; ++i0) {
  
    const uint32_t a = (tlwe[i0] + roundoffset) >> 5;

    if (a == 0)
      continue;

    double *trgswfft = (double *)(BK + i0 * trgswlen);

    // (X^a-1)*acc[i]

    // load to SMEM
    shared_pre[idx] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx, l);
    shared_pre[idx + 64] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx + 64, l);
    shared_pim[idx] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx + 128, l);
    shared_pim[idx + 64] = PolynomialMulByXaiMinusOne<P>(in_rev_dre, a, idx + 192, l);
  
    // BK[i]*((X^a-1)*acc[i])

    // IFFT
    ifft1024(shared_pre, shared_pim, Ns2, idx);

    double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 2048 * blk);
  
    // MulInFD2
    MulInFD2(trgswffti, shared_pre, shared_pim, idx);

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
    __syncblocks(i0 + 2, Syncin, Syncout);

    if (blk < 2) {
      // summation of trlwefft(polynomial addition)
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

      // fft1024fma(restrlwefft)
      fft1024fma(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
    }

    __syncblocks(i0 + 1, Syncin, Syncout);
  }
}

}; // namespace cufhedb