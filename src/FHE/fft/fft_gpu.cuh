#pragma once
#include "utils.cuh"

// Negacyclic cuFFT

namespace cufhedb {

// 1024-point folding FFT + 4x threads with warp shuffle
__device__ inline void fft1024_4xwarp(uint *out_direct_dre, uint *out_direct_dim, double *shared_pre, double *shared_pim, const int32_t Ns2, const int &idx) {

    int ns8 = Ns2 >> 2; 

    // trig table
    const double *__restrict__ trig_table = (double *)tables_direct_d;

    register double re, re1;
    register double im, im1;
    register double tsn, tcs;

    register double _2sN = 1./512.;

    double trig_table4[6] = {1., 0., 1., 0., 0., -1.};

    {
      int idx0 = idx << 1;
      int idx1 = idx0 + 1;

      // normalization
      re = shared_pre[idx0] * _2sN;
      im = shared_pim[idx0] * _2sN;
      re1 = shared_pre[idx1] * _2sN;
      im1 = shared_pim[idx1] * _2sN;

      // nn-point FFT
      #pragma unroll
      for (int32_t nn = 2; nn <= Ns2; nn *= 2) { 
        int32_t halfnn = nn / 2;
        int32_t i = idx / halfnn; // quotient
        int32_t j = idx % halfnn; // remainder
        idx0 = i * (2 * halfnn) + j;
        idx1 = idx0 + halfnn;

        int32_t halfnn4 = nn / 8;
        int32_t tidx = (halfnn4 + (idx/4) % halfnn4 - 1) * 8 + idx % 4;

        if (nn == 2) {
          tcs = trig_table4[0];
          tsn = trig_table4[1];
        } else if (nn == 4) {
          tcs = trig_table4[2 + j];
          tsn = trig_table4[2 + j + 2];
        } else {
          tcs = trig_table[tidx];
          tsn = trig_table[tidx + 4];
        }

        int32_t max_nn = 32; // 32-point is the largest DFT fully handled within a warp
        int32_t halfnn2 = nn / 4;
        double re0, im0;
        if (nn <= max_nn) { // warp shuffle

          unsigned mask = __activemask();        
          int lane = idx & 31;        
          int partner = lane ^ halfnn2; // partner thread ID

          double nbr_re  = __shfl_xor_sync(mask, re,  halfnn2);
          double nbr_re1 = __shfl_xor_sync(mask, re1, halfnn2);
          double nbr_im  = __shfl_xor_sync(mask, im,  halfnn2);
          double nbr_im1 = __shfl_xor_sync(mask, im1, halfnn2); 


          if (lane < partner) {
              re1 = nbr_re;
              im1 = nbr_im;
          } 
          else if (lane > partner) {
              re = nbr_re1;
              im = nbr_im1;
          }
          
          // butterfly unit
          CplxFma(tcs, tsn, re, im, re1, im1);
          

          if (nn == max_nn) {
            shared_pre[idx0] = re;
            shared_pim[idx0] = im;
            shared_pre[idx1] = re1;
            shared_pim[idx1] = im1;

            __syncthreads();
          }

        } else {
          re = shared_pre[idx0];
          im = shared_pim[idx0];
          re1 = shared_pre[idx1];
          im1 = shared_pim[idx1];
          
          // size nn
          CplxFma(tcs, tsn, re, im, re1, im1);

          shared_pre[idx0] = re;
          shared_pim[idx0] = im;
          shared_pre[idx1] = re1;
          shared_pim[idx1] = im1;

          __syncthreads();
        }
      }
    }

    // multiply by omb^j
    {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = 2 * idx + i;
        int32_t tidx = (ns8 + idx0 / 4 - 1) * 8 + idx0 % 4;

        re = shared_pre[idx0];
        im = shared_pim[idx0];

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 4];

        // w * cplx
        CplxMul(tcs, tsn, re, im);
        
        // load back
        out_direct_dre[idx0] = __double2ll_rn(re);
        out_direct_dim[idx0] = __double2ll_rn(im);
      }

    }
}


// 1024-point folding FFT
__device__ inline void fft1024(uint4 *out_direct_dre, uint4 *out_direct_dim, double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d;

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    int i, j, tidx;
    // load to SM
    {
      int idx0 = idx << 1;
      int idx1 = idx0 + 1;
      re = shared_pre[idx0];
      re1 = shared_pre[idx1];
      im = shared_pim[idx0];
      im1 = shared_pim[idx1];

      // 2 & 4-point DFT
      constexpr double _2sN = 1. / 512.;

      re = make_double4(re.x * _2sN, re.y * _2sN, re.z * _2sN, re.w * _2sN);
      im = make_double4(im.x * _2sN, im.y * _2sN, im.z * _2sN, im.w * _2sN);
      re1 = make_double4(re1.x * _2sN, re1.y * _2sN, re1.z * _2sN, re1.w * _2sN);
      im1 = make_double4(im1.x * _2sN, im1.y * _2sN, im1.z * _2sN, im1.w * _2sN);

      FFT2n4(re, im);
      FFT2n4(re1, im1);

      // unroll
      // 8-point DFT
      tcs = trig_table[0];
      tsn = trig_table[1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      //__syncthreads();
      __syncwarp();

      // 16-point DFT
      i = idx >> 1; // quotient
      j = idx & 1;  // remainder
      idx0 = (i << 2) + j;
      idx1 = idx0 + 2;
      tidx = (1 + j) << 1;

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      //__syncthreads();
      __syncwarp();

      // 32-point DFT
      i = idx >> 2; // quotient
      j = idx & 3;  // remainder
      idx0 = (i << 3) + j;
      idx1 = idx0 + 4;
      tidx = (3 + j) << 1;

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      //__syncthreads();
      __syncwarp();

      // 64-point DFT
      i = idx >> 3; // quotient
      j = idx & 7;  // remainder
      idx0 = (i << 4) + j;
      idx1 = idx0 + 8;
      tidx = (7 + j) << 1;

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      //__syncthreads();
      __syncwarp();

      // 128-point DFT
      i = idx >> 4; // quotient
      j = idx & 15; // remainder
      idx0 = (i << 5) + j;
      idx1 = idx0 + 16;
      tidx = (15 + j) << 1;

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      //__syncthreads();
      __syncwarp();

      // 256-point DFT
      i = idx >> 5; // quotient
      j = idx & 31; // remainder
      idx0 = (i << 6) + j;
      idx1 = idx0 + 32;
      tidx = (31 + j) << 1;

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      shared_pre[idx0] = re;
      shared_pim[idx0] = im;
      shared_pre[idx1] = re1;
      shared_pim[idx1] = im1;

      __syncthreads();

      // 512-point DFT
      idx0 = idx;
      idx1 = idx + 64;
      tidx = (63 + idx) << 1;

      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      re = shared_pre[idx0];
      im = shared_pim[idx0];
      re1 = shared_pre[idx1];
      im1 = shared_pim[idx1];

      // size nn
      CplxFma(tcs, tsn, re, im, re1, im1);

      // multiply by omb^j
      tidx = (127 + idx0) << 1;
      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // w * cplx
      CplxMul(tcs, tsn, re, im);

      tidx = (127 + idx1) << 1;
      tcs = trig_table[tidx];
      tsn = trig_table[tidx + 1];

      // w * cplx
      CplxMul(tcs, tsn, re1, im1);

      // load back
      out_direct_dre[idx0] = double4ToUint4(re);
      out_direct_dim[idx0] = double4ToUint4(im);
      out_direct_dre[idx1] = double4ToUint4(re1);
      out_direct_dim[idx1] = double4ToUint4(im1);
    }
}

// 2048-point folding FFT
__device__ inline void fft2048(uint64_t *out_direct_dre, uint64_t *out_direct_dim, double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    int ns8 = Ns2 >> 2; 

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d64;

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    // size2 & size4
    {
      register double _2sN = 1./Ns2;
      
      #pragma unroll 2
      for(int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        re = shared_pre[idx0];
        im = shared_pim[idx0];

        re = make_double4(re.x * _2sN, re.y * _2sN, re.z * _2sN, re.w * _2sN);
        im = make_double4(im.x * _2sN, im.y * _2sN, im.z * _2sN, im.w * _2sN);

        // size2 & size4
        FFT2n4(re,im);


        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }


    // general loop
    {
      #pragma unroll 
      for (int32_t k = 0; k < 8; ++k) {
        int32_t halfnn4 = 1 << k; // halfnn / 4
        int32_t i = idx >> k; // quotient
        int32_t j = idx % halfnn4;// & (halfnn4 - 1); // remainder
        int32_t idx0 = i * (halfnn4 << 1) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = (halfnn4 + j - 1) << 1;

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        re1 = shared_pre[idx1];
        im1 = shared_pim[idx1];
        
        // size nn
        CplxFma(tcs, tsn, re, im, re1, im1);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
        shared_pre[idx1] = re1;
        shared_pim[idx1] = im1;

        __syncthreads();
      }
    }
    
    // multiply by omb^j
    {
      #pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        int tidx = (ns8 + idx0 - 1) << 1;

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        // w * cplx
        CplxMul(tcs, tsn, re, im);
        
        // load back
        double4ToUint4(&out_direct_dre[4 * idx0], re);
        double4ToUint4(&out_direct_dim[4 * idx0], im);
      }
    }
}

__device__ inline void fft1024fma(uint4 *out_direct_dre, uint4 *out_direct_dim, double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

  // trig table
  const double4 *__restrict__ trig_table = (double4 *)tables_direct_d;

  register double4 re, re1;
  register double4 im, im1;
  register double4 tsn, tcs;

  int i, j, tidx;
  // load to SM
  {
    int idx0 = idx << 1;
    int idx1 = idx0 + 1;
    re = shared_pre[idx0];
    re1 = shared_pre[idx1];
    im = shared_pim[idx0];
    im1 = shared_pim[idx1];

    // 2 & 4-point DFT
    constexpr double _2sN = 1. / 512.;

    re = make_double4(re.x * _2sN, re.y * _2sN, re.z * _2sN, re.w * _2sN);
    im = make_double4(im.x * _2sN, im.y * _2sN, im.z * _2sN, im.w * _2sN);
    re1 = make_double4(re1.x * _2sN, re1.y * _2sN, re1.z * _2sN, re1.w * _2sN);
    im1 = make_double4(im1.x * _2sN, im1.y * _2sN, im1.z * _2sN, im1.w * _2sN);

    FFT2n4(re, im);
    FFT2n4(re1, im1);

    // unroll
    // 8-point DFT
    tcs = trig_table[0];
    tsn = trig_table[1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    shared_pre[idx0] = re;
    shared_pim[idx0] = im;
    shared_pre[idx1] = re1;
    shared_pim[idx1] = im1;

    __syncwarp();

    // 16-point DFT
    i = idx >> 1; // quotient
    j = idx & 1;  // remainder
    idx0 = (i << 2) + j;
    idx1 = idx0 + 2;
    tidx = (1 + j) << 1;

    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    re = shared_pre[idx0];
    im = shared_pim[idx0];
    re1 = shared_pre[idx1];
    im1 = shared_pim[idx1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    shared_pre[idx0] = re;
    shared_pim[idx0] = im;
    shared_pre[idx1] = re1;
    shared_pim[idx1] = im1;

    __syncwarp();

    // 32-point DFT
    i = idx >> 2; // quotient
    j = idx & 3;  // remainder
    idx0 = (i << 3) + j;
    idx1 = idx0 + 4;
    tidx = (3 + j) << 1;

    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    re = shared_pre[idx0];
    im = shared_pim[idx0];
    re1 = shared_pre[idx1];
    im1 = shared_pim[idx1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    shared_pre[idx0] = re;
    shared_pim[idx0] = im;
    shared_pre[idx1] = re1;
    shared_pim[idx1] = im1;

    __syncwarp();

    // 64-point DFT
    i = idx >> 3; // quotient
    j = idx & 7;  // remainder
    idx0 = (i << 4) + j;
    idx1 = idx0 + 8;
    tidx = (7 + j) << 1;

    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    re = shared_pre[idx0];
    im = shared_pim[idx0];
    re1 = shared_pre[idx1];
    im1 = shared_pim[idx1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    shared_pre[idx0] = re;
    shared_pim[idx0] = im;
    shared_pre[idx1] = re1;
    shared_pim[idx1] = im1;

    __syncwarp();

    // 128-point DFT
    i = idx >> 4; // quotient
    j = idx & 15; // remainder
    idx0 = (i << 5) + j;
    idx1 = idx0 + 16;
    tidx = (15 + j) << 1;

    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    re = shared_pre[idx0];
    im = shared_pim[idx0];
    re1 = shared_pre[idx1];
    im1 = shared_pim[idx1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    shared_pre[idx0] = re;
    shared_pim[idx0] = im;
    shared_pre[idx1] = re1;
    shared_pim[idx1] = im1;

    __syncwarp();

    // 256-point DFT
    i = idx >> 5; // quotient
    j = idx & 31; // remainder
    idx0 = (i << 6) + j;
    idx1 = idx0 + 32;
    tidx = (31 + j) << 1;

    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    re = shared_pre[idx0];
    im = shared_pim[idx0];
    re1 = shared_pre[idx1];
    im1 = shared_pim[idx1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    shared_pre[idx0] = re;
    shared_pim[idx0] = im;
    shared_pre[idx1] = re1;
    shared_pim[idx1] = im1;

    __syncthreads();

    // 512-point DFT
    idx0 = idx;
    idx1 = idx + 64;
    tidx = (63 + idx) << 1;

    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    re = shared_pre[idx0];
    im = shared_pim[idx0];
    re1 = shared_pre[idx1];
    im1 = shared_pim[idx1];

    // size nn
    CplxFma(tcs, tsn, re, im, re1, im1);

    // multiply by omb^j
    tidx = (127 + idx0) << 1;
    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    // w * cplx
    CplxMul(tcs, tsn, re, im);

    tidx = (127 + idx1) << 1;
    tcs = trig_table[tidx];
    tsn = trig_table[tidx + 1];

    // w * cplx
    CplxMul(tcs, tsn, re1, im1);


    // load back
    register uint4 temp1, temp2, temp3, temp4;
    temp1 = double4ToUint4(re);
    temp2 = double4ToUint4(im);
    temp3 = out_direct_dre[idx0];
    temp4 = out_direct_dim[idx0];

    temp1.x += temp3.x;
    temp1.y += temp3.y;
    temp1.z += temp3.z;
    temp1.w += temp3.w;

    temp2.x += temp4.x;
    temp2.y += temp4.y;
    temp2.z += temp4.z;
    temp2.w += temp4.w;

    out_direct_dre[idx0] = temp1;
    out_direct_dim[idx0] = temp2;

    temp1 = double4ToUint4(re1);
    temp2 = double4ToUint4(im1);
    temp3 = out_direct_dre[idx1];
    temp4 = out_direct_dim[idx1];

    temp1.x += temp3.x;
    temp1.y += temp3.y;
    temp1.z += temp3.z;
    temp1.w += temp3.w;

    temp2.x += temp4.x;
    temp2.y += temp4.y;
    temp2.z += temp4.z;
    temp2.w += temp4.w;

    out_direct_dre[idx1] = temp1;
    out_direct_dim[idx1] = temp2;
  }

}

__device__ inline void fft2048fma(uint64_t *out_direct_dre, uint64_t *out_direct_dim, double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    int ns8 = Ns2 >> 2; 

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d64;

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    // size2 & size4
    {
      register double _2sN = 1./Ns2;
      
      #pragma unroll 2
      for(int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        re = shared_pre[idx0];
        im = shared_pim[idx0];

        // size2 & size4
        FFT2n4(re,im);

        re = make_double4(re.x * _2sN, re.y * _2sN, re.z * _2sN, re.w * _2sN);
        im = make_double4(im.x * _2sN, im.y * _2sN, im.z * _2sN, im.w * _2sN);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }

    // general loop
    {
      #pragma unroll 
      for (int32_t k = 0; k < 8; ++k) {
        int32_t halfnn4 = 1 << k;
        int32_t i = idx >> k; // quotient
        int32_t j = idx % halfnn4;// & (halfnn4 - 1); // remainder
        int32_t idx0 = i * (halfnn4 << 1) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = (halfnn4 + j - 1) << 1;

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        re1 = shared_pre[idx1];
        im1 = shared_pim[idx1];
        
        // size nn
        CplxFma(tcs, tsn, re, im, re1, im1);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
        shared_pre[idx1] = re1;
        shared_pim[idx1] = im1;

        __syncthreads();
      }
    }

    // multiply by omb^j
    register uint64_t temp1[4], temp2[4];
    {
      #pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        int tidx = (ns8 + idx0 - 1) << 1;

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        // w * cplx
        CplxMul(tcs, tsn, re, im);
        
        // load back
        double4ToUint4(temp1, re);
        double4ToUint4(temp2, im);
        add64_4(&out_direct_dre[4 * idx0], temp1);
        add64_4(&out_direct_dim[4 * idx0], temp2);
      }
      //__threadfence();
    }
}

//for Lvl1
__global__ void __launch_bounds__(64, 1) fft(uint32_t * out_direct_d, double * in_direct_d, const int32_t Ns2) {

    int ns8 = Ns2 >> 2; 
    int ns16 = Ns2 >> 3;// threads needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ns16)
        return;

    // convert to double4
    __shared__ double4 shared_pre[128];
    __shared__ double4 shared_pim[128];

    // in
    double4 *__restrict__ in_direct_dre = (double4 *)in_direct_d;
    double4 *__restrict__ in_direct_dim = (double4 *)(in_direct_dre + 128);
    // out
    uint4 *__restrict__ out_direct_dre = (uint4 *)out_direct_d;
    uint4 *__restrict__ out_direct_dim = (uint4 *)(out_direct_dre + 128);

    {  
      int idx0 = idx << 1;
      int idx1 = idx0 + 1;
      shared_pre[idx0] = in_direct_dre[idx0];
      shared_pre[idx1] = in_direct_dre[idx1];
      shared_pim[idx0] = in_direct_dim[idx0];
      shared_pim[idx1] = in_direct_dim[idx1];
    }

    fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
}

// Lvl1 warp shuffle
__global__ void __launch_bounds__(256, 1) fft4x_warp(uint32_t * out_direct_d, double * in_direct_d, const int32_t Ns2) {

    int ns = Ns2; 
    int ns2 = Ns2 >> 1;// threads needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ns2)
        return;

    __shared__ double shared_pre[4 * 128];
    __shared__ double shared_pim[4 * 128];

    // in
    double *__restrict__ in_direct_dre = (double *)in_direct_d;
    double *__restrict__ in_direct_dim = (double *)(in_direct_dre + 4 * 128);
    // out
    uint *__restrict__ out_direct_dre = (uint *)out_direct_d;
    uint *__restrict__ out_direct_dim = (uint *)(out_direct_dre + 4 * 128);

    {  
      int idx0 = idx << 1;
      int idx1 = idx0 + 1;
      shared_pre[idx0] = in_direct_dre[idx0];
      shared_pre[idx1] = in_direct_dre[idx1];
      shared_pim[idx0] = in_direct_dim[idx0];
      shared_pim[idx1] = in_direct_dim[idx1];
    }

    fft1024_4xwarp(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);
}

// for Lvl2
__global__ void __launch_bounds__(128, 1) fft(uint64_t *out_direct_d, double * in_direct_d, const int32_t Ns2) {

    int ns8 = Ns2 >> 2; 
    int ns16 = Ns2 >> 3;// threads needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ns16)
        return;

    // convert to double4
    __shared__ double4 shared_pre[256];
    __shared__ double4 shared_pim[256];

    // in
    double4 *__restrict__ in_direct_dre = (double4 *)in_direct_d;
    double4 *__restrict__ in_direct_dim = (double4 *)(in_direct_dre + 256);
    // out
    uint64_t *__restrict__ out_direct_dre = (uint64_t *)out_direct_d;
    uint64_t *__restrict__ out_direct_dim = (uint64_t *)(out_direct_dre + 1024);
    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d64;

    // load to SM
    {
      int idx0 = idx << 1;
      shared_pre[idx0] = in_direct_dre[idx0];
      shared_pre[idx0 + 1] = in_direct_dre[idx0 + 1];
      shared_pim[idx0] = in_direct_dim[idx0];
      shared_pim[idx0 + 1] = in_direct_dim[idx0 + 1];
    }

    register double4 re, re1;
    register double4 im, im1;
    register double4 tsn, tcs;

    // size2 & size4
    {
      register double _2sN = 1./Ns2;
      
      #pragma unroll 2
      for(int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        re = shared_pre[idx0];
        im = shared_pim[idx0];

        re = make_double4(re.x * _2sN, re.y * _2sN, re.z * _2sN, re.w * _2sN);
        im = make_double4(im.x * _2sN, im.y * _2sN, im.z * _2sN, im.w * _2sN);

        // size2 & size4
        FFT2n4(re,im);


        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
      }
    }


    // general loop
    {
      #pragma unroll 
      for (int32_t k = 0; k < 8; ++k) {
        int32_t halfnn4 = 1 << k; // halfnn / 4
        int32_t i = idx >> k; // quotient
        int32_t j = idx % halfnn4;// & (halfnn4 - 1); // remainder
        int32_t idx0 = i * (halfnn4 << 1) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = (halfnn4 + j - 1) << 1;

        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        re1 = shared_pre[idx1];
        im1 = shared_pim[idx1];
        
        // size nn
        CplxFma(tcs, tsn, re, im, re1, im1);

        shared_pre[idx0] = re;
        shared_pim[idx0] = im;
        shared_pre[idx1] = re1;
        shared_pim[idx1] = im1;

        __syncthreads();
      }
    }
    
    // multiply by omb^j
    {
      #pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        int32_t idx0 = (idx << 1) + i;
        int tidx = (ns8 + idx0 - 1) << 1;

        re = shared_pre[idx0];
        im = shared_pim[idx0];
        tcs = trig_table[tidx];
        tsn = trig_table[tidx + 1];

        // w * cplx
        CplxMul(tcs, tsn, re, im);
        
        // load back
        double4ToUint4(&out_direct_dre[4 * idx0], re);
        double4ToUint4(&out_direct_dim[4 * idx0], im);
      }
    }
}

};
