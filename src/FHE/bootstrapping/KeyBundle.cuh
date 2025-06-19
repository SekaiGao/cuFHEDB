#pragma once
#include "utils.cuh"

// Key bundle generate for bootstrapping unroll

namespace cufhedb {
template<class P>
__global__ void __launch_bounds__(64, 6) KeyBundleFFT(Lvl0_T *tlwe, double *BK, double *Xai, double *KBK, double *onetrgsw) {

  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int l = blk % 3;
  int idx0 = idx << 1;
  constexpr int Addends_size = Lvl0_n / 12;
  int oft = blk * Addends_size;

  constexpr Lvl0_T roundoffset = 16;
  constexpr uint32_t trgswlen = 6 * 2 * 1024;

  double4 *Xai4 = (double4 *)Xai;
  double4 *KBKb = (double4 *)(KBK + oft * trgswlen);
  double4 *__restrict__ BKb = (double4 *)(BK + 3 * oft * trgswlen);

  #pragma unroll
  for (int i = 0; i < Addends_size; ++i) {
    const uint32_t a1 = (tlwe[2 * (i + oft)] + roundoffset) >> 5;
    const uint32_t a2 = (tlwe[2 * (i + oft) + 1] + roundoffset) >> 5;
    double4 *__restrict__ BK0 = BKb + i * 9216;
    double4 *__restrict__ BK1 = BK0 + 3072;
    double4 *__restrict__ BK2 = BK1 + 3072;
    double4 *KBKi = KBKb + i * 3072;
    double4 *XaiMinusOne;
    double4 *BKj;
    #pragma unroll
    for (int j = 0; j < 6; ++j) {
      double4 *__restrict__ onetrgswj = (double4 *)(onetrgsw + j * 2048);
      // trgsw(1)
      Load(shared_pre, onetrgswj, idx);   
      Load(shared_pim, onetrgswj + 256, idx);

      if (a1 + a2) {
        BKj = BK0 + j * 512;
        XaiMinusOne = Xai4 + ((a1 + a2) & 2047) * 256; // X^(a1 + a2) - 1
        FMAInFD2(shared_pre, shared_pim, BKj, XaiMinusOne, idx); // (X^(a1 + a2) - 1) * BK0
      }
      if (a1) {
        BKj = BK1 + j * 512;
        XaiMinusOne = Xai4 + (a1 & 2047) * 256;
        FMAInFD2(shared_pre, shared_pim, BKj, XaiMinusOne, idx); // (X^a1 - 1) * BK1
      }     
      if (a2) {
        BKj = BK2 + j * 512;
        XaiMinusOne = Xai4 + (a2 & 2047) * 256;
        FMAInFD2(shared_pre, shared_pim, BKj, XaiMinusOne, idx); // (X^a2 - 1) * BK2
      }

      double4 *KBKij = KBKi + j * 512;
      Load(KBKij, shared_pre, idx);
      Load(KBKij + 256, shared_pim, idx);
    }
  }

}

template<class P, bool isIde>
__global__ void __launch_bounds__(64, 6) Bootstrapping(uint32_t *res, uint32_t *trlwe, Lvl0_T *tlwe, double *KBK, const uint32_t u, double *buf, const int32_t Ns2) {

  cg::grid_group grid = cg::this_grid();

  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int l = blk % 3;
  int idx0 = idx << 1;
  
  int offset = (blk/3) ? 1024 : 0;
  uint32_t sumoffset = blk ? 1024 : 0;

  // in
  uint32_t *__restrict__ in_rev_dre0 = (uint32_t *)(trlwe + offset);
  // out
  uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(trlwe + sumoffset);
  uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);

  constexpr uint32_t trgswlen = 6 * 2 * 1024;

  const uint32_t b = 2048 - (tlwe[Lvl0_n] >> 5);

  if(blk == 0 || blk ==3)
  {
  if constexpr (isIde) {
    PolynomialMulByXai_gpolygen<P, 3>(in_rev_dre0, u, b, idx, blk);
    PolynomialMulByXai_gpolygen<P, 3>(in_rev_dre0, u, b, idx + 64, blk);
    PolynomialMulByXai_gpolygen<P, 3>(in_rev_dre0, u, b, idx + 128, blk);
    PolynomialMulByXai_gpolygen<P, 3>(in_rev_dre0, u, b, idx + 192, blk);
  } else {
    PolynomialMulByXai_upolygen<P, 3>(in_rev_dre0, u, b, idx, blk);
    PolynomialMulByXai_upolygen<P, 3>(in_rev_dre0, u, b, idx + 64, blk);
    PolynomialMulByXai_upolygen<P, 3>(in_rev_dre0, u, b, idx + 128, blk);
    PolynomialMulByXai_upolygen<P, 3>(in_rev_dre0, u, b, idx + 192, blk);
  }
  }

  grid.sync();

  // in
  uint4 *__restrict__ in_rev_dre = (uint4 *)(trlwe + offset);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);

  #pragma unroll
  for (int i = 0; i < Lvl0_n / 2; ++i) { //

    // load to SMEM
    shared_pre[idx] = DecompositionPolynomial<P>(in_rev_dre, idx, l);
    shared_pre[idx + 64] = DecompositionPolynomial<P>(in_rev_dre, idx + 64, l);
    shared_pim[idx] = DecompositionPolynomial<P>(in_rev_dim, idx, l);
    shared_pim[idx + 64] = DecompositionPolynomial<P>(in_rev_dim, idx + 64, l);

    // IFFT
    ifft1024(shared_pre, shared_pim, Ns2, idx);

    double4 *__restrict__ trgswffti = (double4 *)(KBK + i * trgswlen + 2048 * blk);
  
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

    grid.sync();

    if (blk < 2) {
      // sum of trlwefft(polynomial accumulation)
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

    grid.sync();
  }

  if (blk == 0) {
    SampleExtractIndex<P>(res, trlwe, idx);
  }
}
};