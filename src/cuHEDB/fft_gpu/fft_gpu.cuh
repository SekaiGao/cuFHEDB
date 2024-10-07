#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fenv.h>

// Negacyclic cuFFT/cuIFFT

#define ASM 0

__device__ alignas(32) double tables_direct_d[2048];
__device__ alignas(32) double tables_reverse_d[2048];
__device__ alignas(32) double tables_direct_d64[4096];
__device__ alignas(32) double tables_reverse_d64[4096];

__device__ volatile int Syncin1[6];
__device__ volatile int Syncout1[6];

// radix-2 FFT
namespace cufft {

// 2-point DFT
__device__ inline void FFT2(double4 &re, double4 &im) {
    asm volatile (
        "{\n\t"
        ".reg .f64 tmp;\n\t"

        // real part
        "sub.f64 tmp, %0, %1;\n\t"   // tmp = re.x - re.y
        "add.f64 %0, %0, %1;\n\t"    // re.x += re.y
        "mov.f64 %1, tmp;\n\t"       // re.y = tmp

        "sub.f64 tmp, %2, %3;\n\t"   // tmp = re.z - re.w
        "add.f64 %2, %2, %3;\n\t"    // re.z += re.w
        "mov.f64 %3, tmp;\n\t"       // re.w = tmp

        // imaginary part
        "sub.f64 tmp, %4, %5;\n\t"   // tmp = im.x - im.y
        "add.f64 %4, %4, %5;\n\t"    // im.x += im.y
        "mov.f64 %5, tmp;\n\t"       // im.y = tmp

        "sub.f64 tmp, %6, %7;\n\t"   // tmp = im.z - im.w
        "add.f64 %6, %6, %7;\n\t"    // im.z += im.w
        "mov.f64 %7, tmp;\n\t"       // im.w = tmp
        "}"
        : "+d"(re.x), "+d"(re.y), "+d"(re.z), "+d"(re.w),
          "+d"(im.x), "+d"(im.y), "+d"(im.z), "+d"(im.w)
        : 
    );
}

// 4-point DFT
__device__ inline void FFT4(double4 &re, double4 &im) {
    asm volatile (
        "{\n\t"
        ".reg .f64 tmp;\n\t"

        "sub.f64 tmp, %0, %2;\n\t"  // tmp = re.x - re.z
        "add.f64 %0, %0, %2;\n\t"   // re.x += re.z
        "mov.f64 %2, tmp;\n\t"      // re.z = tmp

        "sub.f64 tmp, %4, %6;\n\t"  // tmp = im.x - im.z
        "add.f64 %4, %4, %6;\n\t"   // im.x += im.z
        "mov.f64 %6, tmp;\n\t"      // im.z = tmp

        "sub.f64 tmp, %1, %7;\n\t"  // tmp = re.y - im.w
        "add.f64 %1, %1, %7;\n\t"   // re.y += im.w

        "add.f64 %7, %5, %3;\n\t"   // im.w = im.y + re.w
        "sub.f64 %5, %5, %3;\n\t"   // im.y -= re.w
        "mov.f64 %3, tmp;\n\t"      // re.w = tmp
        "}"
        : "+d"(re.x), "+d"(re.y), "+d"(re.z), "+d"(re.w),
          "+d"(im.x), "+d"(im.y), "+d"(im.z), "+d"(im.w)
        :
    );
}

// 2&4-point DFT
#if ASM
__device__ inline void FFT2n4(double4 &re, double4 &im) {
    asm volatile (
        "{\n\t"
        ".reg .f64 tmp;\n\t"
        //size 2 
        "sub.f64 tmp, %0, %1;\n\t"   // tmp = re.x - re.y
        "add.f64 %0, %0, %1;\n\t"    // re.x += re.y
        "mov.f64 %1, tmp;\n\t"       // re.y = tmp

        "sub.f64 tmp, %2, %3;\n\t"   // tmp = re.z - re.w
        "add.f64 %2, %2, %3;\n\t"    // re.z += re.w
        "mov.f64 %3, tmp;\n\t"       // re.w = tmp

        "sub.f64 tmp, %4, %5;\n\t"   // tmp = im.x - im.y
        "add.f64 %4, %4, %5;\n\t"    // im.x += im.y
        "mov.f64 %5, tmp;\n\t"       // im.y = tmp

        "sub.f64 tmp, %6, %7;\n\t"   // tmp = im.z - im.w
        "add.f64 %6, %6, %7;\n\t"    // im.z += im.w
        "mov.f64 %7, tmp;\n\t"       // im.w = tmp

        "sub.f64 tmp, %0, %2;\n\t"  // tmp = re.x - re.z
        "add.f64 %0, %0, %2;\n\t"   // re.x += re.z
        "mov.f64 %2, tmp;\n\t"      // re.z = tmp

        "sub.f64 tmp, %4, %6;\n\t"  // tmp = im.x - im.z
        "add.f64 %4, %4, %6;\n\t"   // im.x += im.z
        "mov.f64 %6, tmp;\n\t"      // im.z = tmp

        "sub.f64 tmp, %1, %7;\n\t"  // tmp = re.y - im.w
        "add.f64 %1, %1, %7;\n\t"   // re.y += im.w

        "add.f64 %7, %5, %3;\n\t"   // im.w = im.y + re.w
        "sub.f64 %5, %5, %3;\n\t"   // im.y -= re.w
        "mov.f64 %3, tmp;\n\t"      // re.w = tmp
        "}"
        : "+d"(re.x), "+d"(re.y), "+d"(re.z), "+d"(re.w),
          "+d"(im.x), "+d"(im.y), "+d"(im.z), "+d"(im.w)
        : 
    );
}
#else
__device__ inline void FFT2n4(double4 &re, double4 &im) {
  double tmp;

  tmp = re.x - re.y; 
  re.x += re.y;      
  re.y = tmp;        

  tmp = re.z - re.w; 
  re.z += re.w;      
  re.w = tmp;        

  tmp = im.x - im.y; 
  im.x += im.y;      
  im.y = tmp;        

  tmp = im.z - im.w; 
  im.z += im.w;      
  im.w = tmp;        

  tmp = re.x - re.z; 
  re.x += re.z;      
  re.z = tmp;        

  tmp = im.x - im.z; 
  im.x += im.z;      
  im.z = tmp;        

  tmp = re.y - im.w; 
  re.y += im.w;      

  im.w = im.y + re.w; 
  im.y -= re.w;       
  re.w = tmp;         
}
#endif

// 2&4-point Inverse DFT
__device__ inline void InvFFT4n2(double4 &re, double4 &im) {
  double tmp;

  tmp = re.x - re.z;
  re.x += re.z;
  re.z = tmp;

  tmp = im.x - im.z;
  im.x += im.z;
  im.z = tmp;

  tmp = re.y - re.w;
  re.y += re.w;

  re.w = im.w - im.y;
  im.y += im.w;
  im.w = tmp;

  tmp = re.x - re.y;
  re.x += re.y;
  re.y = tmp;

  tmp = re.z - re.w;
  re.z += re.w;
  re.w = tmp;

  tmp = im.x - im.y;
  im.x += im.y;
  im.y = tmp;

  tmp = im.z - im.w;
  im.z += im.w;
  im.w = tmp;
}

#if ASM
__device__ inline void CplxFma(double4 &tcs, double4 &tsn, double4 &re0, double4 &im0, double4 &re1, double4 &im1) {
    asm volatile (
        "{\n\t"
        ".reg .f64 tmp, tmp0;\n\t"

        // tmp = -im1 * tsn
        // tmp = fma(re1, tcs, tmp) = re1 * tcs - im1 * tsn
        // re1 = re0 - tmp 
        // re0 = re0 + tmp

        // tmp0 = re1 * tsn
        // tmp0 = fma(im1, tcs, tmp0) = re1 * tsn + im1 * tcs
        // im1 = im0 - tmp0
        // im0 = im0 + tmp0

        // x
        "mul.f64 tmp, %4, %20;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "fma.rn.f64 tmp, %0, %16, tmp;\n\t"
        "mul.f64 tmp0, %0, %20;\n\t"
        "fma.rn.f64 tmp0, %4, %16, tmp0;\n\t"
        "sub.f64 %0, %8, tmp;\n\t"
        "add.f64 %8, %8, tmp;\n\t"
        "sub.f64 %4, %12, tmp0;\n\t"
        "add.f64 %12, %12, tmp0;\n\t"
        // y
        "mul.f64 tmp, %5, %21;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "fma.rn.f64 tmp, %1, %17, tmp;\n\t"
        "mul.f64 tmp0, %1, %21;\n\t"
        "fma.rn.f64 tmp0, %5, %17, tmp0;\n\t"
        "sub.f64 %1, %9, tmp;\n\t"
        "add.f64 %9, %9, tmp;\n\t"
        "sub.f64 %5, %13, tmp0;\n\t"
        "add.f64 %13, %13, tmp0;\n\t"
        // z
        "mul.f64 tmp, %6, %22;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "fma.rn.f64 tmp, %2, %18, tmp;\n\t"
        "mul.f64 tmp0, %2, %22;\n\t"
        "fma.rn.f64 tmp0, %6, %18, tmp0;\n\t"
        "sub.f64 %2, %10, tmp;\n\t"
        "add.f64 %10, %10, tmp;\n\t"
        "sub.f64 %6, %14, tmp0;\n\t"
        "add.f64 %14, %14, tmp0;\n\t"
        // w
        "mul.f64 tmp, %7, %23;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "fma.rn.f64 tmp, %3, %19, tmp;\n\t"
        "mul.f64 tmp0, %3, %23;\n\t"
        "fma.rn.f64 tmp0, %7, %19, tmp0;\n\t"
        "sub.f64 %3, %11, tmp;\n\t"
        "add.f64 %11, %11, tmp;\n\t"
        "sub.f64 %7, %15, tmp0;\n\t"
        "add.f64 %15, %15, tmp0;\n\t"
        "}"
        : "+d"(re1.x), "+d"(re1.y), "+d"(re1.z), "+d"(re1.w),
          "+d"(im1.x), "+d"(im1.y), "+d"(im1.z), "+d"(im1.w),
          "+d"(re0.x), "+d"(re0.y), "+d"(re0.z), "+d"(re0.w),
          "+d"(im0.x), "+d"(im0.y), "+d"(im0.z), "+d"(im0.w)
        : "d"(tcs.x), "d"(tcs.y), "d"(tcs.z), "d"(tcs.w),
          "d"(tsn.x), "d"(tsn.y), "d"(tsn.z), "d"(tsn.w)
    );
}
#else
__device__ inline void CplxFma(double4 &tcs, double4 &tsn, double4 &re0, double4 &im0, double4 &re1, double4 &im1) {
    register double4 tmp0, tmp1;

    // re1*cos-im1*sin
    tmp0.x = fma(-im1.x, tsn.x, re1.x * tcs.x);
    tmp0.y = fma(-im1.y, tsn.y, re1.y * tcs.y);
    tmp0.z = fma(-im1.z, tsn.z, re1.z * tcs.z);
    tmp0.w = fma(-im1.w, tsn.w, re1.w * tcs.w);
    // re1*sin+im1*cos
    tmp1.x = fma(im1.x, tcs.x, re1.x * tsn.x);
    tmp1.y = fma(im1.y, tcs.y, re1.y * tsn.y);
    tmp1.z = fma(im1.z, tcs.z, re1.z * tsn.z);
    tmp1.w = fma(im1.w, tcs.w, re1.w * tsn.w);

    // cplx1 = cplx0 - w * cplx1
    re1.x = re0.x - tmp0.x;
    re1.y = re0.y - tmp0.y;
    re1.z = re0.z - tmp0.z;
    re1.w = re0.w - tmp0.w;

    im1.x = im0.x - tmp1.x;
    im1.y = im0.y - tmp1.y;
    im1.z = im0.z - tmp1.z;
    im1.w = im0.w - tmp1.w;

    // cplx0 = cplx0 + w * cplx1
    re0.x += tmp0.x;
    re0.y += tmp0.y;
    re0.z += tmp0.z;
    re0.w += tmp0.w;

    im0.x += tmp1.x;
    im0.y += tmp1.y;
    im0.z += tmp1.z;
    im0.w += tmp1.w;

}
#endif

__device__ inline void InvCplxFma(double4 &tcs, double4 &tsn, double4 &re0, double4 &im0, double4 &re1, double4 &im1) {
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

    // cplx0 = cplx0 + cplx1
    re0.x += re1.x;
    re0.y += re1.y;
    re0.z += re1.z;
    re0.w += re1.w;

    im0.x += im1.x;
    im0.y += im1.y;
    im0.z += im1.z;
    im0.w += im1.w;

    re1.x = fma(tmp0.x, tcs.x, -tmp1.x * tsn.x);
    re1.y = fma(tmp0.y, tcs.y, -tmp1.y * tsn.y);
    re1.z = fma(tmp0.z, tcs.z, -tmp1.z * tsn.z);
    re1.w = fma(tmp0.w, tcs.w, -tmp1.w * tsn.w);

    im1.x = fma(tmp0.x, tsn.x, tmp1.x * tcs.x);
    im1.y = fma(tmp0.y, tsn.y, tmp1.y * tcs.y);
    im1.z = fma(tmp0.z, tsn.z, tmp1.z * tcs.z);
    im1.w = fma(tmp0.w, tsn.w, tmp1.w * tcs.w);
}

// complex multiplication
#if ASM
__device__ inline void CplxMul(double4 &tcs, double4 &tsn, double4 &re1, double4 &im1) {
    asm volatile (
        "{\n\t"
        ".reg .f64 tmp, tmp0;\n\t"

        // tmp = -im1 * tsn
        // re1 = fma(re1, tcs, tmp) = re1 * tcs - im1 * tsn

        // tmp0 = re1 * tsn
        // im1 = fma(im1, tcs, tmp0) = re1 * tsn + im1 * tcs

        // x
        "mul.f64 tmp, %4, %12;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "mul.f64 tmp0, %0, %12;\n\t"
        "fma.rn.f64 %0, %0, %8, tmp;\n\t"
        "fma.rn.f64 %4, %4, %8, tmp0;\n\t"
        // y
        "mul.f64 tmp, %5, %13;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "mul.f64 tmp0, %1, %13;\n\t"
        "fma.rn.f64 %1, %1, %9, tmp;\n\t"
        "fma.rn.f64 %5, %5, %9, tmp0;\n\t"
        // z
        "mul.f64 tmp, %6, %14;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "mul.f64 tmp0, %2, %14;\n\t"
        "fma.rn.f64 %2, %2, %10, tmp;\n\t"
        "fma.rn.f64 %6, %6, %10, tmp0;\n\t"
        // w
        "mul.f64 tmp, %7, %15;\n\t"
        "neg.f64 tmp, tmp;\n\t"
        "mul.f64 tmp0, %3, %15;\n\t"
        "fma.rn.f64 %3, %3, %11, tmp;\n\t"
        "fma.rn.f64 %7, %7, %11, tmp0;\n\t"
        "}"
        : "+d"(re1.x), "+d"(re1.y), "+d"(re1.z), "+d"(re1.w),
          "+d"(im1.x), "+d"(im1.y), "+d"(im1.z), "+d"(im1.w)
        : "d"(tcs.x), "d"(tcs.y), "d"(tcs.z), "d"(tcs.w),
          "d"(tsn.x), "d"(tsn.y), "d"(tsn.z), "d"(tsn.w)
    );
}
#else 
__device__ inline void CplxMul(double4 &tcs, double4 &tsn, double4 &re1, double4 &im1) {
    register double4 tmp0, tmp1;
    // re1*cos
    tmp0.x = re1.x * tcs.x;
    tmp0.y = re1.y * tcs.y;
    tmp0.z = re1.z * tcs.z;
    tmp0.w = re1.w * tcs.w;
    // re1*sin
    tmp1.x = re1.x * tsn.x;
    tmp1.y = re1.y * tsn.y;
    tmp1.z = re1.z * tsn.z;
    tmp1.w = re1.w * tsn.w;
    // re1*cos-im1*sin
    re1.x = fma(-im1.x, tsn.x, tmp0.x);
    re1.y = fma(-im1.y, tsn.y, tmp0.y);
    re1.z = fma(-im1.z, tsn.z, tmp0.z);
    re1.w = fma(-im1.w, tsn.w, tmp0.w);
    // re1*sin+im1*cos
    im1.x = fma(im1.x, tcs.x, tmp1.x);
    im1.y = fma(im1.y, tcs.y, tmp1.y);
    im1.z = fma(im1.z, tcs.z, tmp1.z);
    im1.w = fma(im1.w, tcs.w, tmp1.w);
}
#endif

// convert float point to fixed point
__device__ inline uint64_t float2fixed(double &d) {
  uint64_t vals = __double_as_longlong(d);
  static const uint64_t valmask0 = 0x000FFFFFFFFFFFFFul;
  static const uint64_t valmask1 = 0x0010000000000000ul;
  static const uint16_t expmask0 = 0x07FFu;

  uint64_t val = (vals & valmask0) | valmask1; // mantissa on 53 bits
  uint16_t expo = (vals >> 52) & expmask0;     // exponent 11 bits
  // 1023 -> 52th pos -> 0th pos
  // 1075 -> 52th pos -> 52th pos
  int16_t trans = expo - 1075;
  uint64_t val2 = trans > 0 ? (val << trans) : (val >> -trans);
  vals = (vals >> 63) ? -val2 : val2;
  return vals;
}

// for uint32
__device__ inline uint4 double4ToUint4(double4 &d) {
  uint4 u;
  // round to nearest(or round to zero)
  u.x = __double2ll_rz(d.x);
  u.y = __double2ll_rz(d.y);
  u.z = __double2ll_rz(d.z);
  u.w = __double2ll_rz(d.w);
  return u;
}

// for uint64
__device__ inline void double4ToUint4(uint64_t *u, double4 &d) {
  
  u[0] = float2fixed(d.x);
  u[1] = float2fixed(d.y);
  u[2] = float2fixed(d.z);
  u[3] = float2fixed(d.w);
}

__device__ inline double4 uint4ToDouble4(uint4 &d) {
  double4 u;
  u.x = __int2double_rn(d.x);
  u.y = __int2double_rn(d.y);
  u.z = __int2double_rn(d.z);
  u.w = __int2double_rn(d.w);
  return u;
}

__device__ inline double4 uint4ToDouble4(uint64_t *d) {
  double4 u;
  u.x = __ll2double_rn(d[0]);
  u.y = __ll2double_rn(d[1]);
  u.z = __ll2double_rn(d[2]);
  u.w = __ll2double_rn(d[3]);
  return u;
}

__device__ inline void add64_4(uint64_t *a, uint64_t *b) {
  a[0] += b[0];
  a[1] += b[1];
  a[2] += b[2];
  a[3] += b[3];
}

__device__ inline void fft1024(uint4 *out_direct_dre, uint4 *out_direct_dim, double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    int ns8 = Ns2 >> 2; 

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d;

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
      for (int32_t k = 0; k < 7; ++k) {
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
        out_direct_dre[idx0] = double4ToUint4(re);
        out_direct_dim[idx0] = double4ToUint4(im);
      }
    }
}

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

__device__ inline void ifft1024(double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_reverse_d;

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
      for (int32_t k = 6; k >= 0; --k) {
        int32_t halfnn4 = 1 << k;
        int32_t i = idx >> k;//idx / halfnn4; // quotient
        int32_t j = idx % halfnn4;  //& (halfnn4 - 1); // remainder
        int32_t idx0 = i * (halfnn4 << 1) + j;
        int32_t idx1 = idx0 + halfnn4;
        int32_t tidx = Ns2 - (halfnn4 << 2) + (j << 1);

        __syncthreads();
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

__device__ inline void fft1024fma(uint4 *out_direct_dre, uint4 *out_direct_dim, double4 *shared_pre, double4 *shared_pim, const int32_t Ns2, const int &idx) {

    int ns8 = Ns2 >> 2; 

    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d;

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
      for (int32_t k = 0; k < 7; ++k) {
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
    register uint4 temp1, temp2;
    register uint4 temp3, temp4;
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
      }
      //__threadfence();
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
    // trig table
    const double4 *__restrict__ trig_table = (double4 *)tables_direct_d;

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
      for (int32_t k = 0; k < 7; ++k) {
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
        out_direct_dre[idx0] = double4ToUint4(re);
        out_direct_dim[idx0] = double4ToUint4(im);
      }
    }
}

__global__ void __launch_bounds__(64, 1) ifft(double * out_rev_d, uint32_t * in_rev_d, const int32_t Ns2) {

    int ns16 = Ns2 >> 3;// threads needed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

    // load to SM
    {
      int idx0 = idx << 1;
      shared_pre[idx0] = uint4ToDouble4(in_rev_dre[idx0]);
      shared_pre[idx0 + 1] = uint4ToDouble4(in_rev_dre[idx0 + 1]);
      shared_pim[idx0] = uint4ToDouble4(in_rev_dim[idx0]);
      shared_pim[idx0 + 1] = uint4ToDouble4(in_rev_dim[idx0 + 1]);
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
      for (int32_t k = 6; k >= 0; --k) {
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

};