#pragma once
#include "utils.cuh"
// Negacyclic cuIFFT

// Warp shuffle eliminates all synchronization in FFT.
#define USE_WARP_SHUFFLE false


namespace cufhedb {

// 1024-point folding IFFT
__device__ inline void ifft1024(double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d;

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    register int idx0, idx1, tidx, i, j;

    {
      idx0 = idx;
      idx1 = idx0 + 64;
      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // multiply by omb^j
      tidx = idx0 << 1;
      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // w * cplx
      CplxMul(tcs, tsn, re, im);

      tidx = idx1 << 1;
      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // w * cplx
      CplxMul(tcs, tsn, re1, im1);

      // unroll
      // 512-point DFT
      tidx = 256 + (idx << 1);

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      // 256-point DFT
      i = idx >> 5; // idx / halfnn4; // quotient
      j = idx & 31; // remainder
      idx0 = (i << 6) + j;
      idx1 = idx0 + 32;
      tidx = 384 + (j << 1);

      __syncthreads();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      // 128-point DFT
      i = idx >> 4; // idx / halfnn4; // quotient
      j = idx & 15; // remainder
      idx0 = (i << 5) + j;
      idx1 = idx0 + 16;
      tidx = 448 + (j << 1);

      __syncwarp();
      //__threadfence_block();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      // 64-point DFT
      i = idx >> 3; // idx / halfnn4; // quotient
      j = idx & 7;  // remainder
      idx0 = (i << 4) + j;
      idx1 = idx0 + 8;
      tidx = 480 + (j << 1);

      __syncwarp();
      //__threadfence_block();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      // 32-point DFT
      i = idx >> 2; // idx / halfnn4; // quotient
      j = idx & 3;  // remainder
      idx0 = (i << 3) + j;
      idx1 = idx0 + 4;
      tidx = 496 + (j << 1);

      __syncwarp();
      //__threadfence_block();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      // 16-point DFT
      i = idx >> 1; // idx / halfnn4; // quotient
      j = idx & 1;  // remainder
      idx0 = (i << 2) + j;
      idx1 = idx0 + 2;
      tidx = 504 + (j << 1);

      __syncwarp();
      //__threadfence_block();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      // 8-point DFT
      idx0 = idx << 1;
      idx1 = idx0 + 1;
      tidx = 508;

      __syncwarp();
      //__threadfence_block();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      // 4 & 2-point DFT
      InvFFT4n2(re, im);
      InvFFT4n2(re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] =im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;
    }
}

// 2048-point folding IFFT
__device__ inline void ifft2048(double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d64;

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    // multiply by omb^j
    {
      #pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        int tidx = idx0 << 1;

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        // w * cplx
        CplxMul(tcs, tsn, re, im);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }

    // general loop
    {
      #pragma unroll 
      for (int32_t k = 7; k >= 0; --k) {
        int32_t halfnn4 = 1 << k;
        int32_t i = idx >> k;//idx / halfnn4; // quotient
        int32_t j = idx % halfnn4;  //& (halfnn4 - 1); // remainder
        int32_t idx0 = i * (halfnn4 << 1) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = Ns2 - (halfnn4 << 2) + (j << 1);

        __syncthreads();

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        re1 = shared_pre[idx1];
        im1 = shared_pim[idx1];

        // size nn
        InvCplxFma(tcs, tsn, re, im, re1, im1);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
        shared_pre[idx1] = re1;
        shared_pim[idx1] = im1;

      }
    }
    
    // size4 & size2
    {
      #pragma unroll 2
      for(int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        re = shared_pre[idx0];
        im = shared_pim[idx0];

        // size4 & size2
        InvFFT4n2(re,im);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }
}


#if USE_WARP_SHUFFLE

// for warp shuffle
__device__ inline void InvAdd(double4 &re0, double4 &im0, double4 &re1, double4 &im1) {
    // cplx0 = cplx0 + cplx1
    re0.x += re1.x;
    re0.y += re1.y;
    re0.z += re1.z;
    re0.w += re1.w;

    im0.x += im1.x;
    im0.y += im1.y;
    im0.z += im1.z;
    im0.w += im1.w;
}

__device__ inline void InvSub(double4 &tcs, double4 &tsn, double4 &re0, double4 &im0, double4 &re1, double4 &im1) {
    register double4 tmp0, tmp1;

    // cplx1 = cplx0 - cplx1
    tmp0.x = re0.x - re1.x;
    tmp0.y = re0.y - re1.y;
    tmp0.z = re0.z - re1.z;
    tmp0.w = re0.w - re1.w;

    tmp1.x = im0.x - im1.x;
    tmp1.y = im0.y - im1.y;
    tmp1.z = im0.z - im1.z;
    tmp1.w = im0.w - im1.w;

    re1.x = fma(tmp0.x, tcs.x, -tmp1.x * tsn.x);
    re1.y = fma(tmp0.y, tcs.y, -tmp1.y * tsn.y);
    re1.z = fma(tmp0.z, tcs.z, -tmp1.z * tsn.z);
    re1.w = fma(tmp0.w, tcs.w, -tmp1.w * tsn.w);

    im1.x = fma(tmp0.x, tsn.x, tmp1.x * tcs.x);
    im1.y = fma(tmp0.y, tsn.y, tmp1.y * tcs.y);
    im1.z = fma(tmp0.z, tsn.z, tmp1.z * tcs.z);
    im1.w = fma(tmp0.w, tsn.w, tmp1.w * tcs.w);
}

__device__ inline double4 warpShuffleXorDouble4(double4 var, int laneMask) {
  var.x = __shfl_xor_sync(0xFFFFFFFF, var.x, laneMask);
  var.y = __shfl_xor_sync(0xFFFFFFFF, var.y, laneMask);
  var.z = __shfl_xor_sync(0xFFFFFFFF, var.z, laneMask);
  var.w = __shfl_xor_sync(0xFFFFFFFF, var.w, laneMask);
  return var;
}

// using warp shuffle
__global__ void __launch_bounds__(64, 1) ifft(double * out_rev_d, uint32_t * in_rev_d, const int32_t Ns2) {

    int ns16 = 64;// threads needed
    int idx = threadIdx.x;

    if (idx >= ns16)
        return;

    // convert to double4
    __shared__ double4 shared_pre[128];
    __shared__ double4 shared_pim[128];

    // in
    uint4 *__restrict__ in_rev_dre = (uint4 *)in_rev_d;
    uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);
    // out
    double4 *__restrict__ out_rev_dre = (double4 *)out_rev_d;
    double4 *__restrict__ out_rev_dim = (double4 *)(out_rev_dre + 128);
    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d;

    register int32_t i, j, idx0, idx1, tidx;
    register double4 re, re1, re2, re3;
    register double4 im, im1, im2, im3;
    register double4 tsn, tcs;

    // load to SM
    // multiply by omb^j
    {
      idx0 = idx;
      idx1 = idx0 + 64;
      re = uint4ToDouble4(in_rev_dre[idx0]);
      re1 = uint4ToDouble4(in_rev_dre[idx1]);
      im = uint4ToDouble4(in_rev_dim[idx0]);
      im1 = uint4ToDouble4(in_rev_dim[idx1]);

      tidx = idx0 << 1;
      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // w * cplx
      CplxMul(tcs, tsn, re, im);

      tidx = idx1 << 1;
      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

        // w * cplx
      CplxMul(tcs, tsn, re1, im1);

      // 512-point DFT
      tidx = 256 + (idx << 1);

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;
    }

    // 256-point
    {
      i = idx >> 5; // quotient
      j = idx & 31; // remainder
      idx0 = (i << 6) + j;
      idx1 = idx0 + 32;
      tidx = 384 + (j << 1);

      __syncthreads();

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      InvCplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;
    }
    __syncthreads();

    {
      idx0 = idx;
      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re2 = shared_pre[idx0 + 64];
      im2 = shared_pim[idx0 + 64];

      // warp shuffle
      #pragma once
      for (int k = 4; k >=0 ;--k){
        int halfnn = 1 << k;
        int32_t i = idx >> k;      // quotient
        int32_t j = idx % halfnn; // remainder
        int32_t tidx = Ns2 - (halfnn << 2) + (j << 1);
        
        re1 = warpShuffleXorDouble4(re, halfnn);
        im1 = warpShuffleXorDouble4(im, halfnn);
        re3 = warpShuffleXorDouble4(re2, halfnn);
        im3 = warpShuffleXorDouble4(im2, halfnn);

        if ((idx0 >> k) % 2 == 0) {
          InvAdd(re, im, re1, im1);
          InvAdd(re2, im2, re3, im3);
      } else {
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];
        InvSub(tcs, tsn, re1, im1, re, im);
        InvSub(tcs, tsn, re3, im3, re2, im2);
      }
    }
      // size4 & size2
      InvFFT4n2(re, im);
      InvFFT4n2(re2, im2);

      out_rev_dre[idx0] = re;
      out_rev_dim[idx0] = im;
      out_rev_dre[idx0 + 64] = re2;
      out_rev_dim[idx0 + 64] = im2;
    }   
}
#else
__global__ void __launch_bounds__(64, 1) ifft(double * out_rev_d, uint32_t * in_rev_d, const int32_t Ns2) {

    int ns16 = Ns2 >> 3;// threads needed
    int idx = threadIdx.x;

    if (idx >= ns16)
        return;

    // convert to double4
    __shared__ double4 shared_pre[128];
    __shared__ double4 shared_pim[128];

    // in
    uint4 *__restrict__ in_rev_dre = (uint4 *)in_rev_d;
    uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);
    // out
    double4 *__restrict__ out_rev_dre = (double4 *)out_rev_d;
    double4 *__restrict__ out_rev_dim = (double4 *)(out_rev_dre + 128);
    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d;
    int idx0 = idx;
    int idx1 = idx0 + 64;
    shared_pre[idx0] = uint4ToDouble4(in_rev_dre[idx0]);
    shared_pre[idx1] = uint4ToDouble4(in_rev_dre[idx1]);
    shared_pim[idx0] = uint4ToDouble4(in_rev_dim[idx0]);
    shared_pim[idx1] = uint4ToDouble4(in_rev_dim[idx1]);

    ifft1024(shared_pre, shared_pim, Ns2, idx);

    idx0 = idx << 1;
    idx1 = idx0 + 1;

    out_rev_dre[idx0] = shared_pre[idx0];
    out_rev_dim[idx0] = shared_pim[idx0];
    out_rev_dre[idx1] = shared_pre[idx1];
    out_rev_dim[idx1] = shared_pim[idx1];
}
#endif

// inter-thread batch
template<int32_t batch_size>
__global__ void batch_ifft_th(double * out_rev_d, uint32_t * in_rev_d, const int32_t Ns2) {
  int idx = threadIdx.x;
  int batch = idx >> 6;

  // convert to double4
  __shared__ double4 shared_pre[batch_size * 128];
  __shared__ double4 shared_pim[batch_size * 128];

  // in
  uint4 *__restrict__ in_rev_dre = (uint4 *)(in_rev_d + 1024 * batch);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);
  // out
  double4 *__restrict__ out_rev_dre = (double4 *)(out_rev_d + 1024 * batch);
  double4 *__restrict__ out_rev_dim = (double4 *)(out_rev_dre + 128);

  double4 *pre = shared_pre + 128 * batch;
  double4 *pim = shared_pim + 128 * batch;
  int idx0 = idx & 63;
  int idx1 = idx0 + 64;
  pre[idx0] = uint4ToDouble4(in_rev_dre[idx0]);
  pre[idx1] = uint4ToDouble4(in_rev_dre[idx1]);
  pim[idx0] = uint4ToDouble4(in_rev_dim[idx0]);
  pim[idx1] = uint4ToDouble4(in_rev_dim[idx1]);

  //ifft
  ifft1024(pre, pim, Ns2, idx0);

  idx0 = idx0 << 1;
  idx1 = idx0 + 1;

  out_rev_dre[idx0] = pre[idx0];
  out_rev_dre[idx1] = pre[idx1];
  out_rev_dim[idx0] = pim[idx0];
  out_rev_dim[idx1] = pim[idx1];
}

// inter-block batch
__global__ void batch_ifft_blk(double * out_rev_d, uint32_t * in_rev_d, const int32_t Ns2) {
  int idx = threadIdx.x;
  int batch = blockIdx.x;


  // convert to double4
  __shared__ double4 shared_pre[128];
  __shared__ double4 shared_pim[128];

  // in
  uint4 *__restrict__ in_rev_dre = (uint4 *)(in_rev_d + 1024 * batch);
  uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);
  // out
  double4 *__restrict__ out_rev_dre = (double4 *)(out_rev_d + 1024 * batch);
  double4 *__restrict__ out_rev_dim = (double4 *)(out_rev_dre + 128);

  double4 *pre = shared_pre;
  double4 *pim = shared_pim;
  int idx0 = idx;
  int idx1 = idx0 + 64;
  shared_pre[idx0] = uint4ToDouble4(in_rev_dre[idx0]);
  shared_pre[idx1] = uint4ToDouble4(in_rev_dre[idx1]);
  shared_pim[idx0] = uint4ToDouble4(in_rev_dim[idx0]);
  shared_pim[idx1] = uint4ToDouble4(in_rev_dim[idx1]);

  //ifft
  ifft1024(shared_pre, shared_pim, Ns2, idx0);

  idx0 = idx0 << 1;
  idx1 = idx0 + 1;

  out_rev_dre[idx0] = shared_pre[idx0];
  out_rev_dre[idx1] = shared_pre[idx1];
  out_rev_dim[idx0] = shared_pim[idx0];
  out_rev_dim[idx1] = shared_pim[idx1];
}


// for Lvl2
__global__ void __launch_bounds__(128, 1) ifft(double * out_rev_d, uint64_t * in_rev_d, const int32_t Ns2) {

    int ns16 = Ns2 >> 3;// threads needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ns16)
        return;

    // convert to double4
    __shared__ double4 shared_pre[256];
    __shared__ double4 shared_pim[256];

    // in
    uint64_t *__restrict__ in_rev_dre = (uint64_t *)in_rev_d;
    uint64_t *__restrict__ in_rev_dim = (uint64_t *)(in_rev_dre + 1024);
    // out
    double4 *__restrict__ out_rev_dre = (double4 *)out_rev_d;
    double4 *__restrict__ out_rev_dim = (double4 *)(out_rev_dre + 256);
    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d64;

    // load to SM
    {
      int idx0 = idx << 1;
      shared_pre[idx0] = uint4ToDouble4(&in_rev_dre[4 * idx0]);
      shared_pre[idx0 + 1] = uint4ToDouble4(&in_rev_dre[4 * (idx0 + 1)]);
      shared_pim[idx0] = uint4ToDouble4(&in_rev_dim[4 * idx0]);
      shared_pim[idx0 + 1] = uint4ToDouble4(&in_rev_dim[4 * (idx0 + 1)]);
    }

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    // multiply by omb^j
    {
      #pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        int tidx = idx0 << 1;

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        // w * cplx
        CplxMul(tcs, tsn, re, im);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }

    // general loop
    {
      #pragma unroll 
      for (int32_t k = 7; k >= 0; --k) {
        int32_t halfnn4 = 1 << k;
        int32_t i = idx >> k;//idx / halfnn4; // quotient
        int32_t j = idx % halfnn4;//& (halfnn4 - 1); // remainder
        int32_t idx0 = i * (halfnn4 << 1) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = Ns2 - (halfnn4 << 2) + (j << 1);

        __syncthreads();

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        re1 = shared_pre[idx1];
        im1 = shared_pim[idx1];

        // size nn
        InvCplxFma(tcs, tsn, re, im, re1, im1);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
        shared_pre[idx1] = re1;
        shared_pim[idx1] = im1;
      }
    }

    // size4 & size2
    {
      #pragma unroll 2
      for(int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        re = shared_pre[idx0];
        im = shared_pim[idx0];

        // size4 & size2
        InvFFT4n2(re,im);

        out_rev_dre[idx0] = re;
        out_rev_dim[idx0] = im;
      }
    }
}


#if 1
__device__ inline void negacyclic_ifft(double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d;

    register double4 re, re1, re2, re3;
    register double4 im, im1, im2, im3;
    register double4 tsn, tcs;

    // multiply by omb^j
    {
      #pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        int tidx = idx0 << 1;

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        // w * cplx
        cufhedb::CplxMul(tcs, tsn, re, im);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
      __syncthreads();
    }

    // general loop
    {
      for (int32_t k = 6; k >= 0; --k) {
        int32_t halfnn4 = 1 << k;
        int32_t i = idx / halfnn4; 
        int32_t j = idx % halfnn4;  
        int32_t idx0 = i * (2 * halfnn4) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = (Ns2 - (4 * halfnn4) + 2 * j)%Ns2;

        __syncthreads();

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        re1 = shared_pre[idx1];
        im1 = shared_pim[idx1];

        re2 = shared_pre[idx0 + 128];
        im2 = shared_pim[idx0 + 128];
        re3 = shared_pre[idx1 + 128];
        im3 = shared_pim[idx1 + 128];

        // size nn
        cufhedb::InvCplxFma(tcs, tsn, re, im, re1, im1);
        cufhedb::InvCplxFma(tcs, tsn, re2, im2, re3, im3);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
        shared_pre[idx1] = re1;
        shared_pim[idx1] = im1;

        shared_pre[idx0 + 128] = re2;
        shared_pim[idx0 + 128] = im2;
        shared_pre[idx1 + 128] = re3;
        shared_pim[idx1 + 128] = im3;
      }
    }
    
    __syncthreads();

    // size4 & size2
    {
      for(int i = 0; i < 2; ++i) {
        int32_t idx0 = 2*idx + i;
        re = shared_pre[idx0];
        im = shared_pim[idx0];

        re1 = shared_pre[idx0+128];
        im1 = shared_pim[idx0+128];

        // size4 & size2
        cufhedb::InvFFT4n2(re, im);
        cufhedb::InvFFT4n2(re1, im1);

        re.x += re1.x;
        im.x += im1.x;
        re.y -= re1.y;
        im.y -= im1.y;
        re.z += re1.z;
        im.z += im1.z;
        re.w -= re1.w;
        im.w -= im1.w;

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }  
}

__global__ void __launch_bounds__(64, 1) negacyclic_ifft(double * out_rev_d, uint32_t * in_rev_d, const int32_t Ns2) {

    int ns16 = Ns2 / 8;// threads needed
    int idx = threadIdx.x;

    if (idx >= ns16)
        return;

    // convert to double4
    __shared__ double4 shared_pre[256];
    __shared__ double4 shared_pim[256];

    // in
    uint4 *__restrict__ in_rev_dre = (uint4 *)in_rev_d;
    uint4 *__restrict__ in_rev_dim = (uint4 *)(in_rev_dre + 128);
    // out
    double4 *__restrict__ out_rev_dre = (double4 *)out_rev_d;
    double4 *__restrict__ out_rev_dim = (double4 *)(out_rev_dre + 128);
    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d;
    int idx0 = idx;
    int idx1 = idx0 + 64;
    shared_pre[idx0] = cufhedb::uint4ToDouble4(in_rev_dre[idx0]);
    shared_pre[idx1] = cufhedb::uint4ToDouble4(in_rev_dre[idx1]);
    shared_pim[idx0] = cufhedb::uint4ToDouble4(in_rev_dim[idx0]);
    shared_pim[idx1] = cufhedb::uint4ToDouble4(in_rev_dim[idx1]);

    shared_pre[idx0+128] = negaDouble4(uint4ToDouble4(in_rev_dre[idx0]));
    shared_pre[idx1+128] = negaDouble4(uint4ToDouble4(in_rev_dre[idx1]));
    shared_pim[idx0+128] = negaDouble4(uint4ToDouble4(in_rev_dim[idx0]));
    shared_pim[idx1+128] = negaDouble4(uint4ToDouble4(in_rev_dim[idx1]));

    negacyclic_ifft(shared_pre, shared_pim, Ns2, idx);

    idx0 = idx << 1;
    idx1 = idx0 + 1;

    out_rev_dre[idx0] = shared_pre[idx0];
    out_rev_dim[idx0] = shared_pim[idx0];
    out_rev_dre[idx1] = shared_pre[idx1];
    out_rev_dim[idx1] = shared_pim[idx1];
}
#endif

};