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
  res.x = (((temp.x + totaloffset) >> shiftbits) & mask) - halfBg;
  res.y = (((temp.y + totaloffset) >> shiftbits) & mask) - halfBg;
  res.z = (((temp.z + totaloffset) >> shiftbits) & mask) - halfBg;
  res.w = (((temp.w + totaloffset) >> shiftbits) & mask) - halfBg;
  return res;
}


__device__ inline void MulInFD(double4 *trgswffti, double4 *pre, double4 *pim, const int &idx) {
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

#if 0
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
#endif

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



template<class P>
__global__ void __launch_bounds__(64, 6) ExternalProduct(uint32_t *res, uint32_t *trlwe, double *trgswfft, double *buf, const int32_t Ns2) {
  
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
    uint4 *__restrict__ out_direct_dre = reinterpret_cast<uint4 *>(res + sumoffset);
    uint4 *__restrict__ out_direct_dim = reinterpret_cast<uint4 *>(out_direct_dre + 128);

    // fft1024(restrlwefft)
    fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
  }

}

};