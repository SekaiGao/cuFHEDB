#pragma once
#include "syncblocks.cuh"
#include "utils.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// code for external product

namespace cufhedb {

// External Product: res(trlwe) = trgsw * trlwe
// for Lvl1
template<class P>
__global__ void __launch_bounds__(64, 6) ExternalProduct(uint32_t *res, uint32_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2, volatile int *Syncin, volatile int *Syncout) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int offset = (blk/3) ? 1024 : 0;
  
  // in
  uint4 *__restrict__ in_rev_dre = (uint4 *)(trlwe + offset);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);

  int l = blk % 3;

  // load to SMEM
  int idx0 = idx << 1;
  shared_pre[idx] = DecompositionPolynomial<P>(in_rev_dre, idx, l);
  shared_pre[idx + 64] = DecompositionPolynomial<P>(in_rev_dre, idx + 64, l);
  shared_pim[idx] = DecompositionPolynomial<P>(in_rev_dim, idx, l);
  shared_pim[idx + 64] = DecompositionPolynomial<P>(in_rev_dim, idx + 64, l);

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
  __syncblocks(3, Syncin, Syncout);

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
    uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(res + sumoffset);
    uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);

    // fft1024(restrlwefft)
    fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }

}

// using cooperative group
template <class P>
__global__ void __launch_bounds__(64, 6) ExternalProductCG(uint32_t *res, uint32_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2) {
  
  cg::grid_group grid = cg::this_grid();
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int offset = (blk / 3) ? 1024 : 0;
  
  uint4 *__restrict__ in_rev_dre = (uint4 *)(trlwe + offset);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);

  int l = blk % 3;

  int idx0 = idx << 1;
  shared_pre[idx] = DecompositionPolynomial<P>(in_rev_dre, idx, l);
  shared_pre[idx + 64] = DecompositionPolynomial<P>(in_rev_dre, idx + 64, l);
  shared_pim[idx] = DecompositionPolynomial<P>(in_rev_dim, idx, l);
  shared_pim[idx + 64] = DecompositionPolynomial<P>(in_rev_dim, idx + 64, l);

  // IFFT 
  ifft1024(shared_pre, shared_pim, Ns2, idx);

  double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 2048 * blk);
  
  MulInFD2(trgswffti, shared_pre, shared_pim, idx);

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
    uint32_t sumoffset = blk ? 1024 : 0;
    double4 *bufblk = reinterpret_cast<double4 *>(buf + sumoffset);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int idx0 = 2 * idx + i;
      // real
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

      // image
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

      shared_pre[idx0] = sumr;
      shared_pim[idx0] = sumi;
    }

    uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(res + sumoffset);
    uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);

    // fft1024(restrlwefft)
    fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }
}

// for Lvl2
template<class P>
__global__ void __launch_bounds__(128, 8) ExternalProduct(uint64_t *res, uint64_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2, volatile int *Syncin, volatile int *Syncout) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[512];
  __shared__ double4 shared_pim[512];

  int offset = (blk/4) ? 2048 : 0;
  
  // in
  uint64_t *__restrict__ in_rev_dre = (uint64_t *)(trlwe + offset);
  uint64_t *__restrict__ in_rev_dim = (uint64_t *)(in_rev_dre + 1024);

  int l = blk % 4;

  // load to SMEM
  int idx0 = idx << 1;
  shared_pre[idx0] = DecompositionPolynomial<P>(&in_rev_dre[4 * idx0], l);
  shared_pre[idx0 + 1] = DecompositionPolynomial<P>(&in_rev_dre[4 * (idx0 + 1)], l);
  shared_pim[idx0] = DecompositionPolynomial<P>(&in_rev_dim[4 * idx0], l);
  shared_pim[idx0 + 1] = DecompositionPolynomial<P>(&in_rev_dim[4 * (idx0 + 1)], l);

  // IFFT
  ifft2048(shared_pre, shared_pim, Ns2, idx);

  double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 4096 * blk);
  
  // MulInFD2
  MulInFD64(trgswffti, shared_pre, shared_pim, idx);

  // load to GMEM
  double4 *bufi = (double4 *)(buf + 4096 * blk);

  bufi[idx0] = shared_pre[idx0];
  bufi[idx0 + 1] = shared_pre[idx0 + 1];
  bufi[idx0 + 256] = shared_pim[idx0];
  bufi[idx0 + 257] = shared_pim[idx0 + 1];

  bufi[idx0 + 512] = shared_pre[idx0 + 256];
  bufi[idx0 + 513] = shared_pre[idx0 + 257];
  bufi[idx0 + 768] = shared_pim[idx0 + 256];
  bufi[idx0 + 769] = shared_pim[idx0 + 257];

  // lock-free inter-block sync
  __syncblocks(3, Syncin, Syncout);

  if (blk < 2) {

    // sum trlwefft
    uint32_t sumoffset = blk ? 2048 : 0;
    double4 *bufblk = reinterpret_cast<double4 *>(buf + sumoffset);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int idx0 = 2 * idx + i;
      // real part
      register double4 buf0 = bufblk[idx0];
      register double4 buf1 = bufblk[idx0 + 1024];
      register double4 buf2 = bufblk[idx0 + 2 * 1024];
      register double4 buf3 = bufblk[idx0 + 3 * 1024];
      register double4 buf4 = bufblk[idx0 + 4 * 1024];
      register double4 buf5 = bufblk[idx0 + 5 * 1024];
      register double4 buf6 = bufblk[idx0 + 6 * 1024];
      register double4 buf7 = bufblk[idx0 + 7 * 1024];

      register double4 sumr, sumi;
      sumr.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x + buf6.x + buf7.x;
      sumr.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y + buf6.y + buf7.y;
      sumr.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z + buf6.z + buf7.z;
      sumr.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w + buf6.w + buf7.w;

      //imag part
      buf0 = bufblk[idx0 + 256];
      buf1 = bufblk[idx0 + 1024 + 256];
      buf2 = bufblk[idx0 + 2 * 1024 + 256];
      buf3 = bufblk[idx0 + 3 * 1024 + 256];
      buf4 = bufblk[idx0 + 4 * 1024 + 256];
      buf5 = bufblk[idx0 + 5 * 1024 + 256];
      buf6 = bufblk[idx0 + 6 * 1024 + 256];
      buf7 = bufblk[idx0 + 7 * 1024 + 256];

      sumi.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x + buf6.x + buf7.x;
      sumi.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y + buf6.y + buf7.y;
      sumi.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z + buf6.z + buf7.z;
      sumi.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w + buf6.w + buf7.w;
      
      // to SMEM
      shared_pre[idx0] = sumr;
      shared_pim[idx0] = sumi;


    }

    // out
    uint64_t *__restrict__ out_direct_dre = (uint64_t *)(res + sumoffset);
    uint64_t *__restrict__ out_direct_dim = (uint64_t *)(out_direct_dre + 1024);

    // fft2048(restrlwefft)
    fft2048(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }

}

template<class P>
__global__ void __launch_bounds__(128, 8) ExternalProductCG(uint64_t *res, uint64_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2) {

  cg::grid_group grid = cg::this_grid();

  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[512];
  __shared__ double4 shared_pim[512];

  int offset = (blk/4) ? 2048 : 0;
  
  // in
  uint64_t *__restrict__ in_rev_dre = (uint64_t *)(trlwe + offset);
  uint64_t *__restrict__ in_rev_dim = (uint64_t *)(in_rev_dre + 1024);

  int l = blk % 4;

  // load to SMEM
  int idx0 = idx << 1;
  shared_pre[idx0] = DecompositionPolynomial<P>(&in_rev_dre[4 * idx0], l);
  shared_pre[idx0 + 1] = DecompositionPolynomial<P>(&in_rev_dre[4 * (idx0 + 1)], l);
  shared_pim[idx0] = DecompositionPolynomial<P>(&in_rev_dim[4 * idx0], l);
  shared_pim[idx0 + 1] = DecompositionPolynomial<P>(&in_rev_dim[4 * (idx0 + 1)], l);

  // IFFT
  ifft2048(shared_pre, shared_pim, Ns2, idx);

  double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 4096 * blk);
  
  // MulInFD2
  MulInFD64(trgswffti, shared_pre, shared_pim, idx);

  // load to GMEM
  double4 *bufi = (double4 *)(buf + 4096 * blk);

  bufi[idx0] = shared_pre[idx0];
  bufi[idx0 + 1] = shared_pre[idx0 + 1];
  bufi[idx0 + 256] = shared_pim[idx0];
  bufi[idx0 + 257] = shared_pim[idx0 + 1];

  bufi[idx0 + 512] = shared_pre[idx0 + 256];
  bufi[idx0 + 513] = shared_pre[idx0 + 257];
  bufi[idx0 + 768] = shared_pim[idx0 + 256];
  bufi[idx0 + 769] = shared_pim[idx0 + 257];

  grid.sync();

  if (blk < 2) {

    // sum trlwefft
    uint32_t sumoffset = blk ? 2048 : 0;
    double4 *bufblk = reinterpret_cast<double4 *>(buf + sumoffset);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int idx0 = 2 * idx + i;
      // real part
      register double4 buf0 = bufblk[idx0];
      register double4 buf1 = bufblk[idx0 + 1024];
      register double4 buf2 = bufblk[idx0 + 2 * 1024];
      register double4 buf3 = bufblk[idx0 + 3 * 1024];
      register double4 buf4 = bufblk[idx0 + 4 * 1024];
      register double4 buf5 = bufblk[idx0 + 5 * 1024];
      register double4 buf6 = bufblk[idx0 + 6 * 1024];
      register double4 buf7 = bufblk[idx0 + 7 * 1024];

      register double4 sumr, sumi;
      sumr.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x + buf6.x + buf7.x;
      sumr.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y + buf6.y + buf7.y;
      sumr.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z + buf6.z + buf7.z;
      sumr.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w + buf6.w + buf7.w;

      //imag part
      buf0 = bufblk[idx0 + 256];
      buf1 = bufblk[idx0 + 1024 + 256];
      buf2 = bufblk[idx0 + 2 * 1024 + 256];
      buf3 = bufblk[idx0 + 3 * 1024 + 256];
      buf4 = bufblk[idx0 + 4 * 1024 + 256];
      buf5 = bufblk[idx0 + 5 * 1024 + 256];
      buf6 = bufblk[idx0 + 6 * 1024 + 256];
      buf7 = bufblk[idx0 + 7 * 1024 + 256];

      sumi.x = buf0.x + buf1.x + buf2.x + buf3.x + buf4.x + buf5.x + buf6.x + buf7.x;
      sumi.y = buf0.y + buf1.y + buf2.y + buf3.y + buf4.y + buf5.y + buf6.y + buf7.y;
      sumi.z = buf0.z + buf1.z + buf2.z + buf3.z + buf4.z + buf5.z + buf6.z + buf7.z;
      sumi.w = buf0.w + buf1.w + buf2.w + buf3.w + buf4.w + buf5.w + buf6.w + buf7.w;
      
      // to SMEM
      shared_pre[idx0] = sumr;
      shared_pim[idx0] = sumi;


    }

    // out
    uint64_t *__restrict__ out_direct_dre = (uint64_t *)(res + sumoffset);
    uint64_t *__restrict__ out_direct_dim = (uint64_t *)(out_direct_dre + 1024);

    // fft2048(restrlwefft)
    fft2048(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }

}

// without kernel fusion
template<class P>
__global__ void __launch_bounds__(64, 6) MulByTRGSWFFT(uint32_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int l = blk % 3;
  int idx0 = idx << 1;

  int offset = (blk/3) ? 1024 : 0;
  
  // in
  uint4 *__restrict__ in_rev_dre = (uint4 *)(trlwe + offset);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);

  // (X^a-1)*acc[i]

  // load to SMEM
  shared_pre[idx0] = DecompositionPolynomial<P>(in_rev_dre, idx0, l);
  shared_pre[idx0 + 1] = DecompositionPolynomial<P>(in_rev_dre, idx0 + 1, l);
  shared_pim[idx0] = DecompositionPolynomial<P>(in_rev_dim, idx0, l);
  shared_pim[idx0 + 1] = DecompositionPolynomial<P>(in_rev_dim, idx0 + 1, l);

  // BK[i]*((X^a-1)*acc[i])

  // IFFT
  ifft1024(shared_pre, shared_pim, Ns2, idx);

  double4 *__restrict__ trgswffti = (double4 *)(trgswfft + 2048 * blk);
  
  // MulInFD
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

  __threadfence();
}

template<class P>
__global__ void __launch_bounds__(64, 2) Reduction(uint32_t *trlwe, double *buf, const int32_t Ns2) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[256];
  __shared__ double4 shared_pim[256];

  int idx0 = idx << 1;

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

    // imag part
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

template <class P>
__global__ __launch_bounds__(384)
void ExternalProduct_th(uint32_t *res, uint32_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2)
{
    int tid = threadIdx.x;
    int p   = tid / 64;   
    int idx = tid % 64;   

    // offset and decomposition level
    int offset = (p / 3) ? 1024 : 0;
    int l = p % 3;

    __shared__ double4 shared_pre[256];
    __shared__ double4 shared_pim[256];

    // —— Decomposition —— 
    uint4 *__restrict__ in_rev_dre = reinterpret_cast<uint4*>(trlwe + offset);
    uint4 *__restrict__ in_rev_dim = in_rev_dre + 128;

    shared_pre[idx   ] = DecompositionPolynomial<P>(in_rev_dre,    idx,     l);
    shared_pre[idx+64] = DecompositionPolynomial<P>(in_rev_dre,    idx+64,  l);
    shared_pim[idx   ] = DecompositionPolynomial<P>(in_rev_dim,    idx,     l);
    shared_pim[idx+64] = DecompositionPolynomial<P>(in_rev_dim,    idx+64,  l);

    __syncthreads();

    // —— IFFT —— 
    ifft1024(shared_pre, shared_pim, Ns2, idx);


    double4 *__restrict__ trgswffti = 
        reinterpret_cast<double4*>(trgswfft + 2048 * p);
    MulInFD2(trgswffti, shared_pre, shared_pim, idx);

    double4 *bufi = reinterpret_cast<double4*>(buf + 2048 * p);
    int idx0 = idx << 1;
    bufi[idx0      ] = shared_pre[idx0];
    bufi[idx0 + 1  ] = shared_pre[idx0 + 1];
    bufi[idx0 + 128] = shared_pim[idx0];
    bufi[idx0 + 129] = shared_pim[idx0 + 1];
    bufi[idx0 + 256] = shared_pre[idx0 + 128];
    bufi[idx0 + 257] = shared_pre[idx0 + 129];
    bufi[idx0 + 384] = shared_pim[idx0 + 128];
    bufi[idx0 + 385] = shared_pim[idx0 + 129];

    __syncthreads();

    if (p < 2) {
        uint32_t sumoffset = p ? 1024 : 0;
        double4 *bufblk = reinterpret_cast<double4*>(buf + sumoffset);

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx0s = (idx << 1) + i;

            double4 b0 = bufblk[idx0s];
            double4 b1 = bufblk[idx0s + 512];
            double4 b2 = bufblk[idx0s + 2*512];
            double4 b3 = bufblk[idx0s + 3*512];
            double4 b4 = bufblk[idx0s + 4*512];
            double4 b5 = bufblk[idx0s + 5*512];
            double4 sumr;
            sumr.x = b0.x + b1.x + b2.x + b3.x + b4.x + b5.x;
            sumr.y = b0.y + b1.y + b2.y + b3.y + b4.y + b5.y;
            sumr.z = b0.z + b1.z + b2.z + b3.z + b4.z + b5.z;
            sumr.w = b0.w + b1.w + b2.w + b3.w + b4.w + b5.w;

            b0 = bufblk[idx0s + 128];
            b1 = bufblk[idx0s + 512 + 128];
            b2 = bufblk[idx0s + 2*512 + 128];
            b3 = bufblk[idx0s + 3*512 + 128];
            b4 = bufblk[idx0s + 4*512 + 128];
            b5 = bufblk[idx0s + 5*512 + 128];
            double4 sumi;
            sumi.x = b0.x + b1.x + b2.x + b3.x + b4.x + b5.x;
            sumi.y = b0.y + b1.y + b2.y + b3.y + b4.y + b5.y;
            sumi.z = b0.z + b1.z + b2.z + b3.z + b4.z + b5.z;
            sumi.w = b0.w + b1.w + b2.w + b3.w + b4.w + b5.w;

            shared_pre[idx0s] = sumr;
            shared_pim[idx0s] = sumi;
        }

        __syncthreads();

    
        uint4 *__restrict__ out_dre = 
            reinterpret_cast<uint4*>(res + sumoffset);
        uint4 *__restrict__ out_dim = out_dre + 128;
        fft1024(out_dre, out_dim, shared_pre, shared_pim, Ns2, idx);
    }
}


}; // namespace cufhedb