#pragma once
#include "fft_gpu.cuh"

namespace cufft {

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

//polynomial multiplication: polya * polyb -> ifftpolya * ifftpolyb -> ifftres -> res
template<class P>
__global__ void __launch_bounds__(64, 2) PolyMul(uint32_t *res, uint32_t *a, double *ifftb, const int32_t Ns2) {
  
  int idx = threadIdx.x;
  int blk = blockIdx.x;

  __shared__ double4 shared_pre[128];
  __shared__ double4 shared_pim[128];

  int offset = blk ? 1024 : 0;
  
  // in
  uint4 *__restrict__ in_rev_dre = (uint4 *)(a + offset);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);
  // out
  uint4 *__restrict__ out_direct_dre = (uint4 *)(res + offset);
  uint4 *__restrict__ out_direct_dim = (uint4 *)(out_direct_dre + 128);

  // load to SMEM
  int idx0 = idx << 1;
  shared_pre[idx0] = uint4ToDouble4(in_rev_dre[idx0]);
  shared_pre[idx0 + 1] = uint4ToDouble4(in_rev_dre[idx0 + 1]);
  shared_pim[idx0] = uint4ToDouble4(in_rev_dim[idx0]);
  shared_pim[idx0 + 1] = uint4ToDouble4(in_rev_dim[idx0 + 1]);

  double4 *__restrict__ ifftb4 = (double4 *)(ifftb);

  // IFFT
  ifft1024(shared_pre, shared_pim, Ns2, idx);

  // MulInFD
  MulInFD(ifftb4, shared_pre, shared_pim, idx);

  // FFT
  fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);

}

// synchronizing all blocks in the kernel
#if 0
// this version will cause deadlock when blocks > SMs
__device__ inline void __syncblocks(int goalVal, volatile int *Syncin, volatile int *Syncout) {
  int idx = threadIdx.x;
  int blk = blockIdx.x;
  int numblk = gridDim.x;
  
  // lock-free inter-block sync
  if (idx == 0) {
    Syncin[blk] = goalVal; 
    __threadfence();
  }

  if (blk == 0) {
    for (int i = 0; i < numblk; ++i) {
      while (Syncin[i] != goalVal) {// busy wait
      }
    }

    // When all blocks are complete, blk 0 updates Syncout
    for (int i = 0; i < numblk; ++i) {
      Syncout[i] = goalVal;
    }
    __threadfence();
  }

  __syncthreads();

  if (idx == 0) {
    while (Syncout[blk] != goalVal) {// busy wait
      __threadfence_block();
    }
    Syncin[blk] = 0;
    Syncout[blk] = 0;
    __threadfence();
  }

  __syncthreads();
}
#else
// using this version for syncblocks should make sure blocks < 2*SMs
__device__ inline void __syncblocks(int goalVal, volatile int *Syncin, volatile int *Syncout) {
    int idx = threadIdx.x;
    int blk = blockIdx.x;
    int numblk = gridDim.x;

    // lock-free inter-block sync
    if (idx == 0) {
        Syncin[blk] = goalVal; 
        // ensure the write visible to all blocks
        __threadfence();  
    }

    // block 0 check whether all blocks have written their status
    if (blk == 0 && idx == 0) {
      volatile int complete = 0;
      // busy wait
      while (complete != numblk) {
        complete = 0;
        // check
        for (int i = 0; i < numblk; ++i) {
          if (Syncin[i] == goalVal) {
            ++complete;
          }
        }
      }
    }
    
    __syncthreads(); 

    if (blk == 0 && idx == 0) {
        for (int i = 0; i < numblk; ++i) {
          // notify all blocks synchronization is complete
          Syncout[i] = goalVal;  
        }
        __threadfence();
    }

    __syncthreads(); 

    if (idx == 0) {
        while (Syncout[blk] != goalVal) {
          // ensure consistency within the block
          __threadfence_block();  
        }
        // reset Syncin and Syncout to 0 
        Syncin[blk] = 0;
        Syncout[blk] = 0;
        __threadfence();  
    }

    __syncthreads();
}
#endif

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
  shared_pre[idx0] = DecompositionPolynomial<P>(in_rev_dre, idx0, l);
  shared_pre[idx0 + 1] = DecompositionPolynomial<P>(in_rev_dre, idx0 + 1, l);
  shared_pim[idx0] = DecompositionPolynomial<P>(in_rev_dim, idx0, l);
  shared_pim[idx0 + 1] = DecompositionPolynomial<P>(in_rev_dim, idx0 + 1, l);

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

};