#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fenv.h>
#include <string>
#include <array>
#include <vector>

template<class P>
__device__ void polynomialMulByXaiMinusOne(double4 *res, double *poly, const uint32_t a, int idx0, int idx1, int digit) {
  constexpr typename P::T offset = 2181562368;
  constexpr typename P::T roundoffset = 8192;
  constexpr typename P::T digits = 32;
  constexpr typename P::T mask = 63;
  constexpr typename P::T halfBg = 32;
  constexpr typename P::T Bgbit = 6;
  constexpr int32_t n = 1024;

  int index0 = 4 * idx1;
  bool flag1, flag2;
  double4 temp;
  double temp1[4];
  for (int i = 0; i < 4; ++i) {
    int index = index0 + i;
    if (index < n) {
      flag1 = index < a;
      flag2 = index < a - n;
    } else {
      flag1 = index < a + n;
      flag2 = index < a;
    }
    if (a < n) {
      if (flag1) {
        temp1[i] = -poly[index - a + n] - poly[index];
      } else {
        temp1[i] = poly[index - a] - poly[index];
      }
    } else {
      const uint32_t aa = a - n;
      if (flag2) {
        temp1[i] = poly[index - aa + n] - poly[index];
      } else {
        temp1[i] = -poly[index - aa] - poly[index];
      }
    }
    }
    temp.x = double(int32_t(((((uint32_t(int64_t(temp1[0])) + offset + roundoffset) >> (digits - (digit + 1) * Bgbit)) & mask) - halfBg)));
    temp.y = double(int32_t(((((uint32_t(int64_t(temp1[1])) + offset + roundoffset) >> (digits - (digit + 1) * Bgbit)) & mask) - halfBg)));
    temp.z = double(int32_t(((((uint32_t(int64_t(temp1[2])) + offset + roundoffset) >> (digits - (digit + 1) * Bgbit)) & mask) - halfBg)));
    temp.w = double(int32_t(((((uint32_t(int64_t(temp1[3])) + offset + roundoffset) >> (digits - (digit + 1) * Bgbit)) & mask) - halfBg)));
    res[idx0] = temp;
}

template<class P>
__global__ void trgswfftFMA_kernel3v(double *inout_rev_i, double *trig_tables, double *inout_rev_d, double *trgswffti, const uint32_t a, const int32_t Ns2) {
	  
    int ns4 = Ns2 / 2;
    int ns8 = Ns2 / 4; // threads needed(128)
    int ns16 = ns8 / 2;
    int idx = threadIdx.x;
    int blk = blockIdx.x; // block(3)

    if (idx >= ns8)
        return;
    
    // convert to double4
    double4 *out_rev_d4 = (double4 *)(&inout_rev_i[4*blk*Ns2]);
    double4 *trig_tables4 = (double4 *)trig_tables;
    __shared__ double4 shared_data[512];
    double4 *shared_data1 = &shared_data[256]; // Second half of shared_data
    __shared__ double4 shared_trig_tables[512]; 

    double4 *trgswffti0 = (double4 *)(&trgswffti[8 * blk * Ns2]);
    double4 *trgswffti1 = &trgswffti0[Ns2];

    // load to SM
    if (idx < ns16) {
		  int idx0 = 2*idx;
      polynomialMulByXaiMinusOne<P>(shared_data, inout_rev_d, a, idx0, idx0, blk);
      polynomialMulByXaiMinusOne<P>(shared_data, inout_rev_d, a, idx0 + 1, idx0 + 1, blk);
      polynomialMulByXaiMinusOne<P>(shared_data, inout_rev_d, a, idx0 + ns8, idx0 + ns8, blk);
      polynomialMulByXaiMinusOne<P>(shared_data, inout_rev_d, a, idx0 + ns8 + 1, idx0 + ns8 + 1, blk);
    } 
    else {
      int idx0 = 2 * idx - ns8;
      int idx1 = 2 * idx + ns8;
      polynomialMulByXaiMinusOne<P>(shared_data1, inout_rev_d, a, idx0, idx1, blk);
      polynomialMulByXaiMinusOne<P>(shared_data1, inout_rev_d, a, idx0 + 1, idx1 + 1, blk);
      polynomialMulByXaiMinusOne<P>(shared_data1, inout_rev_d, a, idx0 + ns8, idx1 + ns8, blk);
      polynomialMulByXaiMinusOne<P>(shared_data1, inout_rev_d, a, idx0 + ns8 + 1, idx1 + ns8 + 1, blk);
	  }


    __syncthreads();

	
    // multiply by omb^j
    if (idx < ns16) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        int dataIdx = 2 * idx + i;
        int trigIdx = dataIdx * 2;

        double4 re = shared_data[dataIdx];
        double4 im = shared_data[dataIdx + ns8];
        double4 tcs = trig_tables4[trigIdx]; // shared_trig_tables[trigIdx];
        double4 tsn = trig_tables4[trigIdx + 1]; // shared_trig_tables[trigIdx + 1];

        double4 tmp0, tmp1;

        // re*cos - im*sin
        tmp0.x = fma(re.x, tcs.x, -im.x * tsn.x);
        tmp0.y = fma(re.y, tcs.y, -im.y * tsn.y);
        tmp0.z = fma(re.z, tcs.z, -im.z * tsn.z);
        tmp0.w = fma(re.w, tcs.w, -im.w * tsn.w);

        // im*cos + re*sin
        tmp1.x = fma(im.x, tcs.x, re.x * tsn.x);
        tmp1.y = fma(im.y, tcs.y, re.y * tsn.y);
        tmp1.z = fma(im.z, tcs.z, re.z * tsn.z);
        tmp1.w = fma(im.w, tcs.w, re.w * tsn.w);

        //(re*cos-im*sin) + i (im*cos+re*sin)
        shared_data[dataIdx] = tmp0;
        shared_data[dataIdx + ns8] = tmp1;
      }
    } else {
	  #pragma unroll
      for (int i = 0; i < 2; i++) {
        int dataIdx = 2 * idx - ns8  + i;
        int trigIdx = dataIdx * 2;

        double4 re = shared_data1[dataIdx];
        double4 im = shared_data1[dataIdx + ns8];
        double4 tcs = trig_tables4[trigIdx]; // shared_trig_tables[trigIdx];
        double4 tsn = trig_tables4[trigIdx + 1]; // shared_trig_tables[trigIdx + 1];

        double4 tmp0, tmp1;

        // re*cos - im*sin
        tmp0.x = fma(re.x, tcs.x, -im.x * tsn.x);
        tmp0.y = fma(re.y, tcs.y, -im.y * tsn.y);
        tmp0.z = fma(re.z, tcs.z, -im.z * tsn.z);
        tmp0.w = fma(re.w, tcs.w, -im.w * tsn.w);

        // im*cos + re*sin
        tmp1.x = fma(im.x, tcs.x, re.x * tsn.x);
        tmp1.y = fma(im.y, tcs.y, re.y * tsn.y);
        tmp1.z = fma(im.z, tcs.z, re.z * tsn.z);
        tmp1.w = fma(im.w, tcs.w, re.w * tsn.w);

        //(re*cos-im*sin) + i (im*cos+re*sin)
        shared_data1[dataIdx] = tmp0;
        shared_data1[dataIdx + ns8] = tmp1;
      }
    }

    __syncthreads();

    // general loop
    if (idx < ns16) {
      #pragma unroll
      for (int32_t halfnn = Ns2/2; halfnn >= 4; halfnn /= 2) {
        int nns8 = halfnn / 4;
        
        int i = idx / nns8;
        int j = idx % nns8;
        int inout_idx = i * halfnn/2 + j;
        int table_idx = Ns2 - halfnn + 2 * j;

        double4 r0 = trig_tables4[table_idx]; // shared_trig_tables[table_idx];
        double4 r1 = trig_tables4[table_idx + 1]; // shared_trig_tables[table_idx + 1];

        double4 d00 = shared_data[inout_idx];
        double4 d01 = shared_data[inout_idx + ns8];
        double4 d10 = shared_data[inout_idx + nns8];
        double4 d11 = shared_data[inout_idx + ns8 + nns8];

        double4 tmp0, tmp1;

        // d00-d10
        tmp0.x = d00.x - d10.x;
        tmp0.y = d00.y - d10.y;
        tmp0.z = d00.z - d10.z;
        tmp0.w = d00.w - d10.w;
        // d01-d11
        tmp1.x = d01.x - d11.x;
        tmp1.y = d01.y - d11.y;
        tmp1.z = d01.z - d11.z;
        tmp1.w = d01.w - d11.w;

        // d00=d00+d10
        d00.x += d10.x;
        d00.y += d10.y;
        d00.z += d10.z;
        d00.w += d10.w;
        // d01=d01+d11
        d01.x += d11.x;
        d01.y += d11.y;
        d01.z += d11.z;
        d01.w += d11.w;
        // d10=tmp0*r0-tmp1*r1
        d10.x = fma(tmp0.x, r0.x, -tmp1.x * r1.x);
        d10.y = fma(tmp0.y, r0.y, -tmp1.y * r1.y);
        d10.z = fma(tmp0.z, r0.z, -tmp1.z * r1.z);
        d10.w = fma(tmp0.w, r0.w, -tmp1.w * r1.w);
        // d11=tmp0*r1+tmp1*r0
        d11.x = fma(tmp0.x, r1.x, tmp1.x * r0.x);
        d11.y = fma(tmp0.y, r1.y, tmp1.y * r0.y);
        d11.z = fma(tmp0.z, r1.z, tmp1.z * r0.z);
        d11.w = fma(tmp0.w, r1.w, tmp1.w * r0.w);
        
        shared_data[inout_idx] = d00;
        shared_data[inout_idx + ns8] = d01;
        shared_data[inout_idx + nns8] = d10;
        shared_data[inout_idx + ns8 + nns8] = d11;

        __syncthreads();
      }
    }else {
	  #pragma unroll
      for (int32_t halfnn = Ns2 / 2; halfnn >= 4; halfnn /= 2) {
        int nns8 = halfnn / 4;

        int i = (idx - ns16) / nns8;
        int j = (idx - ns16) % nns8;
        int inout_idx = i * halfnn / 2 + j;
        int table_idx = Ns2 - halfnn + 2 * j;

        double4 r0 = trig_tables4[table_idx]; // shared_trig_tables[table_idx];
        double4 r1 = trig_tables4[table_idx + 1]; // shared_trig_tables[table_idx + 1];

        double4 d00 = shared_data1[inout_idx];
        double4 d01 = shared_data1[inout_idx + ns8];
        double4 d10 = shared_data1[inout_idx + nns8];
        double4 d11 = shared_data1[inout_idx + ns8 + nns8];

        double4 tmp0, tmp1;

        // d00-d10
        tmp0.x = d00.x - d10.x;
        tmp0.y = d00.y - d10.y;
        tmp0.z = d00.z - d10.z;
        tmp0.w = d00.w - d10.w;
        // d01-d11
        tmp1.x = d01.x - d11.x;
        tmp1.y = d01.y - d11.y;
        tmp1.z = d01.z - d11.z;
        tmp1.w = d01.w - d11.w;

        // d00=d00+d10
        d00.x += d10.x;
        d00.y += d10.y;
        d00.z += d10.z;
        d00.w += d10.w;
        // d01=d01+d11
        d01.x += d11.x;
        d01.y += d11.y;
        d01.z += d11.z;
        d01.w += d11.w;
        // d10=tmp0*r0-tmp1*r1
        d10.x = fma(tmp0.x, r0.x, -tmp1.x * r1.x);
        d10.y = fma(tmp0.y, r0.y, -tmp1.y * r1.y);
        d10.z = fma(tmp0.z, r0.z, -tmp1.z * r1.z);
        d10.w = fma(tmp0.w, r0.w, -tmp1.w * r1.w);
        // d11=tmp0*r1+tmp1*r0
        d11.x = fma(tmp0.x, r1.x, tmp1.x * r0.x);
        d11.y = fma(tmp0.y, r1.y, tmp1.y * r0.y);
        d11.z = fma(tmp0.z, r1.z, tmp1.z * r0.z);
        d11.w = fma(tmp0.w, r1.w, tmp1.w * r0.w);

        shared_data1[inout_idx] = d00;
        shared_data1[inout_idx + ns8] = d01;
        shared_data1[inout_idx + nns8] = d10;
        shared_data1[inout_idx + ns8 + nns8] = d11;

        __syncthreads();
      }
    }

    // size2 & size4 & FMA
    if (idx < ns16) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        int idxi = 2*idx+i;
        double4 re = shared_data[idxi];
        double4 im = shared_data[idxi + ns8];

        double4 re1, im1;

        re1.x = re.x + re.z;
        re1.y = re.y + re.w;
        re1.z = re.x - re.z;
        re1.w = im.w - im.y;

        im1.x = im.x + im.z;
        im1.y = im.y + im.w;
        im1.z = im.x - im.z;
        im1.w = re.y - re.w;

        double4 are, aim;
        are.x = re1.x + re1.y;
        are.y = re1.x - re1.y;
        are.z = re1.z + re1.w;
        are.w = re1.z - re1.w;

        aim.x = im1.x + im1.y;
        aim.y = im1.x - im1.y;
        aim.z = im1.z + im1.w;
        aim.w = im1.z - im1.w;

        double4 bre1, bim1,bre2,bim2;
        
        bre1 = trgswffti0[idxi];
        bim1 = trgswffti0[idxi + ns8];
        bre2 = trgswffti0[idxi + ns4];
        bim2 = trgswffti0[idxi + ns8 + ns4];

        double4 aimbim, arebim;
        aimbim.x = aim.x * bim1.x;
        aimbim.y = aim.y * bim1.y;
        aimbim.z = aim.z * bim1.z;
        aimbim.w = aim.w * bim1.w;

        arebim.x = are.x * bim1.x;
        arebim.y = are.y * bim1.y;
        arebim.z = are.z * bim1.z;
        arebim.w = are.w * bim1.w;

        re.x = fma(are.x, bre1.x, -aimbim.x);
        re.y = fma(are.y, bre1.y, -aimbim.y);
        re.z = fma(are.z, bre1.z, -aimbim.z);
        re.w = fma(are.w, bre1.w, -aimbim.w);

        im.x = fma(aim.x, bre1.x, arebim.x);
        im.y = fma(aim.y, bre1.y, arebim.y);
        im.z = fma(aim.z, bre1.z, arebim.z);
        im.w = fma(aim.w, bre1.w, arebim.w);

        shared_data[idxi] = re;
        shared_data[idxi + ns8] = im;

        aimbim.x = aim.x * bim2.x;
        aimbim.y = aim.y * bim2.y;
        aimbim.z = aim.z * bim2.z;
        aimbim.w = aim.w * bim2.w;

        arebim.x = are.x * bim2.x;
        arebim.y = are.y * bim2.y;
        arebim.z = are.z * bim2.z;
        arebim.w = are.w * bim2.w;

        re.x = fma(are.x, bre2.x, -aimbim.x);
        re.y = fma(are.y, bre2.y, -aimbim.y);
        re.z = fma(are.z, bre2.z, -aimbim.z);
        re.w = fma(are.w, bre2.w, -aimbim.w);

        im.x = fma(aim.x, bre2.x, arebim.x);
        im.y = fma(aim.y, bre2.y, arebim.y);
        im.z = fma(aim.z, bre2.z, arebim.z);
        im.w = fma(aim.w, bre2.w, arebim.w);

        shared_data[idxi + ns4] = re;
        shared_data[idxi + ns4 + ns8] = im;

        //shared_data[2 * idx + i] = result1;
        //shared_data[2 * idx + ns8 + i] = result2;
      }
    }else {
	  #pragma unroll
      for (int i = 0; i < 2; i++) {
        int idxi = 2 * idx + i - ns8;
        double4 re = shared_data1[idxi];
        double4 im = shared_data1[idxi + ns8];

        double4 re1, im1;

        re1.x = re.x + re.z;
        re1.y = re.y + re.w;
        re1.z = re.x - re.z;
        re1.w = im.w - im.y;

        im1.x = im.x + im.z;
        im1.y = im.y + im.w;
        im1.z = im.x - im.z;
        im1.w = re.y - re.w;

        double4 are, aim;
        are.x = re1.x + re1.y;
        are.y = re1.x - re1.y;
        are.z = re1.z + re1.w;
        are.w = re1.z - re1.w;

        aim.x = im1.x + im1.y;
        aim.y = im1.x - im1.y;
        aim.z = im1.z + im1.w;
        aim.w = im1.z - im1.w;

        double4 bre1, bim1,bre2,bim2;
        
        bre1 = trgswffti1[idxi];
        bim1 = trgswffti1[idxi + ns8];
        bre2 = trgswffti1[idxi + ns4];
        bim2 = trgswffti1[idxi + ns8 + ns4];

        double4 aimbim, arebim;
        aimbim.x = aim.x * bim1.x;
        aimbim.y = aim.y * bim1.y;
        aimbim.z = aim.z * bim1.z;
        aimbim.w = aim.w * bim1.w;

        arebim.x = are.x * bim1.x;
        arebim.y = are.y * bim1.y;
        arebim.z = are.z * bim1.z;
        arebim.w = are.w * bim1.w;

        re.x = fma(are.x, bre1.x, -aimbim.x);
        re.y = fma(are.y, bre1.y, -aimbim.y);
        re.z = fma(are.z, bre1.z, -aimbim.z);
        re.w = fma(are.w, bre1.w, -aimbim.w);

        im.x = fma(aim.x, bre1.x, arebim.x);
        im.y = fma(aim.y, bre1.y, arebim.y);
        im.z = fma(aim.z, bre1.z, arebim.z);
        im.w = fma(aim.w, bre1.w, arebim.w);

        shared_trig_tables[idxi] = re;
        shared_trig_tables[idxi + ns8] = im;

        aimbim.x = aim.x * bim2.x;
        aimbim.y = aim.y * bim2.y;
        aimbim.z = aim.z * bim2.z;
        aimbim.w = aim.w * bim2.w;

        arebim.x = are.x * bim2.x;
        arebim.y = are.y * bim2.y;
        arebim.z = are.z * bim2.z;
        arebim.w = are.w * bim2.w;

        re.x = fma(are.x, bre2.x, -aimbim.x);
        re.y = fma(are.y, bre2.y, -aimbim.y);
        re.z = fma(are.z, bre2.z, -aimbim.z);
        re.w = fma(are.w, bre2.w, -aimbim.w);

        im.x = fma(aim.x, bre2.x, arebim.x);
        im.y = fma(aim.y, bre2.y, arebim.y);
        im.z = fma(aim.z, bre2.z, arebim.z);
        im.w = fma(aim.w, bre2.w, arebim.w);

        shared_trig_tables[idxi + ns4] = re;
        shared_trig_tables[idxi + ns4 + ns8] = im;

        //shared_data1[2 * idx - ns8 + i] = result1;
        //shared_data1[2 * idx + i] = result2;
      }
	}
    __syncthreads();

  if (idx < ns8) {
      int idx0 = 4 * idx;
      #pragma unroll
      for(int i =0;i<4;++i) {
        double4 x,y;
        x = shared_data[idx0 + i];
        y = shared_trig_tables[idx0 + i];
        x.x += y.x;
        x.y += y.y;
        x.z += y.z;
        x.w += y.w;
        out_rev_d4[idx0 + i] = x;
      }
      
  }
}

__device__ double4 reduce(double4 *arr, int idx) {
  double4 t1, t2, t3;

  t1 = arr[idx];
  t2 = arr[idx +  512];
  t3 = arr[idx + 1024];

  t1.x += t2.x + t3.x;
  t1.y += t2.y + t3.y;
  t1.z += t2.z + t3.z;
  t1.w += t2.w + t3.w;

  return t1;
}

__device__ void sum2trlwe(double4 *arr0, double4 *arr1, int idx0, int idx1) {
    double4 t1, t2;

    t2 = arr1[idx1];
    t1 = arr0[idx0];

    t1.x += uint32_t(int64_t(t2.x));
    t1.y += uint32_t(int64_t(t2.y));
    t1.z += uint32_t(int64_t(t2.z));
    t1.w += uint32_t(int64_t(t2.w));

    arr0[idx0] = t1;
}

__global__ void reduce_fft(double *trig_tables, double *inout_direct_d, double *inout_rev_i, const int32_t Ns2) {

    int ns8 = Ns2 / 4; // threads needed
    int ns16 = ns8 / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ns8)
        return;
    // convert to double4
    double4 *inout_direct_d4 = (double4 *)inout_direct_d;
    double4 *inout_rev_i4 = (double4 *)inout_rev_i;
    double4 *trig_tables4 = (double4 *)trig_tables;
    __shared__ double4 shared_data[512];
    double4 *shared_data1 = &shared_data[256]; // Second half of shared_data
    __shared__ double4 shared_trig_tables[512];

    // load to SM
    if (idx < ns16) {
      int idx0 = 2 * idx;
      shared_data[idx0] = reduce(inout_rev_i4, idx0);
      shared_data[idx0 + 1] = reduce(inout_rev_i4, idx0 + 1);
      shared_data[idx0 + ns8] = reduce(inout_rev_i4, idx0 + ns8);
      shared_data[idx0 + ns8 + 1] = reduce(inout_rev_i4, idx0 + ns8 + 1);

      int group = 4 * idx0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        shared_trig_tables[group + i] = trig_tables4[group + i];
      }
    } else {
      int idx0 = 2 * idx - ns8;
      int idx1 = 2 * idx + ns8;
      shared_data1[idx0] = reduce(inout_rev_i4, idx1);
      shared_data1[idx0 + 1] = reduce(inout_rev_i4, idx1 + 1);
      shared_data1[idx0 + ns8] = reduce(inout_rev_i4, idx1 + ns8);
      shared_data1[idx0 + ns8 + 1] = reduce(inout_rev_i4, idx1 + ns8 + 1);

      int group = 4 * idx0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        shared_trig_tables[group + i] = trig_tables4[group + i];
      }
    }

    __syncthreads();

    // size2 & size4
    if (idx < ns16) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            double4 re = shared_data[2 * idx + i];
            double4 im = shared_data[2 * idx + ns8 + i];

            double4 re1, im1;
            re1.x = re.x + re.y;
            re1.y = re.x - re.y;
            re1.z = re.z + re.w;
            re1.w = re.z - re.w;

            im1.x = im.x + im.y;
            im1.y = im.x - im.y;
            im1.z = im.z + im.w;
            im1.w = im.z - im.w;

            double4 result1, result2;

            result1.x = (re1.x + re1.z)/Ns2;
            result1.y = (re1.y + im1.w)/Ns2;
            result1.z = (re1.x - re1.z)/Ns2;
            result1.w = (re1.y - im1.w)/Ns2;

            result2.x = (im1.x + im1.z)/Ns2;
            result2.y = (im1.y - re1.w)/Ns2;
            result2.z = (im1.x - im1.z)/Ns2;
            result2.w = (im1.y + re1.w)/Ns2;

            shared_data[2 * idx + i] = result1;
            shared_data[2 * idx + ns8 + i] = result2;
        }
    }else {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        double4 re = shared_data1[2 * idx - ns8 + i];
        double4 im = shared_data1[2 * idx + i];

        double4 re1, im1;
        re1.x = re.x + re.y;
        re1.y = re.x - re.y;
        re1.z = re.z + re.w;
        re1.w = re.z - re.w;

        im1.x = im.x + im.y;
        im1.y = im.x - im.y;
        im1.z = im.z + im.w;
        im1.w = im.z - im.w;

        double4 result1, result2;

        result1.x = (re1.x + re1.z) / Ns2;
        result1.y = (re1.y + im1.w) / Ns2;
        result1.z = (re1.x - re1.z) / Ns2;
        result1.w = (re1.y - im1.w) / Ns2;

        result2.x = (im1.x + im1.z) / Ns2;
        result2.y = (im1.y - re1.w) / Ns2;
        result2.z = (im1.x - im1.z) / Ns2;
        result2.w = (im1.y + re1.w) / Ns2;

        shared_data1[2 * idx - ns8 + i] = result1;
        shared_data1[2 * idx + i] = result2;
      }
    }
    __syncthreads();

    // general loop
    if (idx < ns16) {
      #pragma unroll
      for (int32_t halfnn = 4; halfnn < Ns2; halfnn *= 2) {
        // ns4/8 iterations (N = 1024 is 64 iters)
        int32_t nns4 = halfnn / 2;
        int32_t nns8 = halfnn / 4;
        int i = idx / nns8;
        int j = idx % nns8;
        int inout_idx = i * nns4 + j;
        int table_idx = nns4 + 2 * j - 2;

        double4 tcs = shared_trig_tables[table_idx];
        double4 tsn = shared_trig_tables[table_idx + 1];

        double4 re0 = shared_data[inout_idx];
        double4 im0 = shared_data[inout_idx + ns8];
        double4 re1 = shared_data[inout_idx + nns8];
        double4 im1 = shared_data[inout_idx + ns8 + nns8];

        double4 tmp0, tmp1;
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
        tmp0.x = fma(-im1.x, tsn.x, tmp0.x);
        tmp0.y = fma(-im1.y, tsn.y, tmp0.y);
        tmp0.z = fma(-im1.z, tsn.z, tmp0.z);
        tmp0.w = fma(-im1.w, tsn.w, tmp0.w);
        // re1*sin+im1*cos
        tmp1.x = fma(im1.x, tcs.x, tmp1.x);
        tmp1.y = fma(im1.y, tcs.y, tmp1.y);
        tmp1.z = fma(im1.z, tcs.z, tmp1.z);
        tmp1.w = fma(im1.w, tcs.w, tmp1.w);

        // re1=re0-tmp0
        re1.x = re0.x - tmp0.x;
        re1.y = re0.y - tmp0.y;
        re1.z = re0.z - tmp0.z;
        re1.w = re0.w - tmp0.w;
        // re0=re0+tmp0
        re0.x = re0.x + tmp0.x;
        re0.y = re0.y + tmp0.y;
        re0.z = re0.z + tmp0.z;
        re0.w = re0.w + tmp0.w;
        // im1=im0-tmp1
        im1.x = im0.x - tmp1.x;
        im1.y = im0.y - tmp1.y;
        im1.z = im0.z - tmp1.z;
        im1.w = im0.w - tmp1.w;
        // im0=im0+tmp1;
        im0.x = im0.x + tmp1.x;
        im0.y = im0.y + tmp1.y;
        im0.z = im0.z + tmp1.z;
        im0.w = im0.w + tmp1.w;

        shared_data[inout_idx] = re0;
        shared_data[inout_idx + ns8] = im0;
        shared_data[inout_idx + nns8] = re1;
        shared_data[inout_idx + ns8 + nns8] = im1;

        __syncthreads();
      }
    } else {
      #pragma unroll
      for (int32_t halfnn = 4; halfnn < Ns2; halfnn *= 2) {
        // ns4/8 iterations (N = 1024 is 64 iters)
        int32_t nns4 = halfnn / 2;
        int32_t nns8 = halfnn / 4;
        int i = (idx-ns16) / nns8;
        int j = (idx-ns16) % nns8;
        int inout_idx = i * nns4 + j;
        int table_idx = nns4 + 2 * j - 2;

        double4 tcs = shared_trig_tables[table_idx];
        double4 tsn = shared_trig_tables[table_idx + 1];

        double4 re0 = shared_data1[inout_idx];
        double4 im0 = shared_data1[inout_idx + ns8];
        double4 re1 = shared_data1[inout_idx + nns8];
        double4 im1 = shared_data1[inout_idx + ns8 + nns8];

        double4 tmp0, tmp1;
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
        tmp0.x = fma(-im1.x, tsn.x, tmp0.x);
        tmp0.y = fma(-im1.y, tsn.y, tmp0.y);
        tmp0.z = fma(-im1.z, tsn.z, tmp0.z);
        tmp0.w = fma(-im1.w, tsn.w, tmp0.w);
        // re1*sin+im1*cos
        tmp1.x = fma(im1.x, tcs.x, tmp1.x);
        tmp1.y = fma(im1.y, tcs.y, tmp1.y);
        tmp1.z = fma(im1.z, tcs.z, tmp1.z);
        tmp1.w = fma(im1.w, tcs.w, tmp1.w);

        // re1=re0-tmp0
        re1.x = re0.x - tmp0.x;
        re1.y = re0.y - tmp0.y;
        re1.z = re0.z - tmp0.z;
        re1.w = re0.w - tmp0.w;
        // re0=re0+tmp0
        re0.x = re0.x + tmp0.x;
        re0.y = re0.y + tmp0.y;
        re0.z = re0.z + tmp0.z;
        re0.w = re0.w + tmp0.w;
        // im1=im0-tmp1
        im1.x = im0.x - tmp1.x;
        im1.y = im0.y - tmp1.y;
        im1.z = im0.z - tmp1.z;
        im1.w = im0.w - tmp1.w;
        // im0=im0+tmp1;
        im0.x = im0.x + tmp1.x;
        im0.y = im0.y + tmp1.y;
        im0.z = im0.z + tmp1.z;
        im0.w = im0.w + tmp1.w;

        shared_data1[inout_idx] = re0;
        shared_data1[inout_idx + ns8] = im0;
        shared_data1[inout_idx + nns8] = re1;
        shared_data1[inout_idx + ns8 + nns8] = im1;

        __syncthreads();
      }
    }

    // multiply by omb^j
    if (idx < ns16) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        int dataIdx = 2 * idx + i;
        int trigIdx = Ns2 / 2 - 2 + dataIdx * 2;

        double4 re = shared_data[dataIdx];
        double4 im = shared_data[dataIdx + ns8];
        double4 tcs = shared_trig_tables[trigIdx];
        double4 tsn = shared_trig_tables[trigIdx + 1];

        double4 tmp0, tmp1;

        // re*cos - im*sin
        tmp0.x = fma(re.x, tcs.x, -im.x * tsn.x);
        tmp0.y = fma(re.y, tcs.y, -im.y * tsn.y);
        tmp0.z = fma(re.z, tcs.z, -im.z * tsn.z);
        tmp0.w = fma(re.w, tcs.w, -im.w * tsn.w);

        // im*cos + re*sin
        tmp1.x = fma(im.x, tcs.x, re.x * tsn.x);
        tmp1.y = fma(im.y, tcs.y, re.y * tsn.y);
        tmp1.z = fma(im.z, tcs.z, re.z * tsn.z);
        tmp1.w = fma(im.w, tcs.w, re.w * tsn.w);

        //(re*cos-im*sin) + i (im*cos+re*sin)
        shared_data[dataIdx] = tmp0;
        shared_data[dataIdx + ns8] = tmp1;
      }
    }else {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        int dataIdx = 2 * idx - ns8 + i;
        int trigIdx = Ns2 / 2 - 2 + dataIdx * 2;

        double4 re = shared_data1[dataIdx];
        double4 im = shared_data1[dataIdx + ns8];
        double4 tcs = shared_trig_tables[trigIdx];
        double4 tsn = shared_trig_tables[trigIdx + 1];

        double4 tmp0, tmp1;

        // re*cos - im*sin
        tmp0.x = fma(re.x, tcs.x, -im.x * tsn.x);
        tmp0.y = fma(re.y, tcs.y, -im.y * tsn.y);
        tmp0.z = fma(re.z, tcs.z, -im.z * tsn.z);
        tmp0.w = fma(re.w, tcs.w, -im.w * tsn.w);

        // im*cos + re*sin
        tmp1.x = fma(im.x, tcs.x, re.x * tsn.x);
        tmp1.y = fma(im.y, tcs.y, re.y * tsn.y);
        tmp1.z = fma(im.z, tcs.z, re.z * tsn.z);
        tmp1.w = fma(im.w, tcs.w, re.w * tsn.w);

        //(re*cos-im*sin) + i (im*cos+re*sin)
        shared_data1[dataIdx] = tmp0;
        shared_data1[dataIdx + ns8] = tmp1;
      }
    }

    __syncthreads();

    if (idx < ns16) {
      sum2trlwe(inout_direct_d4, shared_data, 2 * idx, 2 * idx);
      sum2trlwe(inout_direct_d4, shared_data, 2*idx+1, 2*idx+1);
      sum2trlwe(inout_direct_d4, shared_data, 2 * idx + ns8, 2 * idx + ns8);
      sum2trlwe(inout_direct_d4, shared_data, 2 * idx + ns8 + 1, 2 * idx + ns8 + 1);
    }else {
      int idx0 = 2 * idx - ns8;
      int idx1 = 2 * idx + ns8;
      sum2trlwe(inout_direct_d4, shared_data1, idx1, idx0);
      sum2trlwe(inout_direct_d4, shared_data1, idx1+1, idx0+1);
      sum2trlwe(inout_direct_d4, shared_data1, idx1 + ns8, idx0 + ns8);
      sum2trlwe(inout_direct_d4, shared_data1, idx1 + ns8 + 1, idx0 + ns8 + 1);
    }
}

__global__ void PolynomialMulByXai_kernelv(double *res, const uint32_t poly, const uint32_t a, const int32_t n) {
  int idx = threadIdx.x; // Thread index
  int ns8 = n/8;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (idx < ns8) {
      int index = idx + j * ns8;
      double temp1, temp2;
      if (a == 0) {
        temp1 = 0;
        temp2 = -poly;
      } else if (a < n) {
        if (index < a) {
          temp1 = 0;
          temp2 = poly;
        } else {
          temp1 = 0;
          temp2 = -poly;
        }
      } else {
        const uint32_t aa = a - n;
        if (index < aa) {
          temp1 = 0;
          temp2 = -poly;
        } else {
          temp1 = 0;
          temp2 = poly;
        }
      }
      res[index] = temp1;
      res[index + n] = temp2;
    }
  }
}

__global__ void PolynomialMulByXai_kernelg(double *res, uint32_t scale_bits, const uint32_t a, const int32_t n) {
  constexpr uint32_t padding_bits = 6; // P :: nbit - plain_bits;
  uint32_t poly = 1ULL << scale_bits;
  int idx = threadIdx.x; // Thread index
  int ns8 = n/8;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (idx < ns8) {
      int index = idx + j * ns8;
      double temp2;
      if (a == 0) {
        temp2 = poly * (index >> padding_bits);
      } else if (a < n) {
        if (index < a) {
          temp2 = -poly * ((index - a + n) >> padding_bits);
        } else {
          temp2 = poly * ((index - a) >> padding_bits);
        }
      } else {
        const uint32_t aa = a - n;
        if (index < aa) {
          temp2 = poly * ((index - aa + n) >> padding_bits);
        } else {
          temp2 = -poly * ((index - aa) >> padding_bits);
        }
      }
      res[index] = 0;
      res[index + n] = temp2;
    }
  }
}

__global__ void SampleExtractIndex_Kernelv(uint32_t *tlwe_d, double *inout_direct_d, const int32_t N) {
  int idx = threadIdx.x;
  int ns8 = N/8;
  if (idx >= ns8) // 128
    return;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    int index = idx + j * ns8;
    if (index != 0) {
      tlwe_d[index] = -uint32_t(int64_t(inout_direct_d[N - index]));
    } else {
      tlwe_d[0] = uint32_t(int64_t(inout_direct_d[0]));
      tlwe_d[N] = uint32_t(int64_t(inout_direct_d[N]));
    }
  }
}

class CuFFT_TRLWE {
public:
  const int32_t _2N;
  const int32_t N;
  const int32_t Ns2;
  const int32_t trgswLen;
private:
  // device
  double *tables_direct_d;
  double *tables_reverse_d;
  double *trgswfft_d;

  double *inout_direct_ds;
  double *inout_rev_is;
  uint32_t *tlwe_ds;

  double accurate_cos(int32_t i, int32_t n) { // cos(2pi*i/n)
    i = ((i % n) + n) % n;
    if (i >= 3 * n / 4)
      return cos(2. * M_PI * (n - i) / double(n));
    if (i >= 2 * n / 4)
      return -cos(2. * M_PI * (i - n / 2) / double(n));
    if (i >= 1 * n / 4)
      return -cos(2. * M_PI * (n / 2 - i) / double(n));
    return cos(2. * M_PI * (i) / double(n));
  }

  double accurate_sin(int32_t i, int32_t n) { // sin(2pi*i/n)
    i = ((i % n) + n) % n;
    if (i >= 3 * n / 4)
      return -sin(2. * M_PI * (n - i) / double(n));
    if (i >= 2 * n / 4)
      return -sin(2. * M_PI * (i - n / 2) / double(n));
    if (i >= 1 * n / 4)
      return sin(2. * M_PI * (n / 2 - i) / double(n));
    return sin(2. * M_PI * (i) / double(n));
  }

public:
  CuFFT_TRLWE(const int32_t N) : _2N(2 * N), N(N), Ns2(N / 2), trgswLen(12 * N) {
    int32_t ns4 = _2N / 4;
    double *tables_direct_h;
    double *tables_reverse_h;
    cudaHostAlloc((void**)&tables_direct_h, _2N * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&tables_reverse_h, _2N * sizeof(double), cudaHostAllocDefault);

	  // direct table
    double *ptr_direct = tables_direct_h;
    for (int32_t halfnn = 4; halfnn < ns4; halfnn *= 2) {
      int32_t nn = 2 * halfnn;
      int32_t j = _2N / nn;
      for (int32_t i = 0; i < halfnn; i += 4) {
        for (int32_t k = 0; k < 4; k++)
          *(ptr_direct++) = accurate_cos(-j * (i + k), _2N);
        for (int32_t k = 0; k < 4; k++)
          *(ptr_direct++) = accurate_sin(-j * (i + k), _2N);
      }
    }
    // last iteration
    for (int32_t i = 0; i < ns4; i += 4) {
      for (int32_t k = 0; k < 4; k++)
        *(ptr_direct++) = accurate_cos(-(i + k), _2N);
      for (int32_t k = 0; k < 4; k++)
        *(ptr_direct++) = accurate_sin(-(i + k), _2N);
    }
	
	  // reverse table
    double *ptr_reverse = tables_reverse_h;
    for (int32_t j = 0; j < ns4; j += 4) {
      for (int32_t k = 0; k < 4; k++)
        *(ptr_reverse++) = accurate_cos(j + k, _2N);
      for (int32_t k = 0; k < 4; k++)
        *(ptr_reverse++) = accurate_sin(j + k, _2N);
    }
    // subsequent iterations
    for (int32_t nn = ns4; nn >= 8; nn /= 2) {
      int32_t halfnn = nn / 2;
      int32_t j = _2N / nn;
      for (int32_t i = 0; i < halfnn; i += 4) {
        for (int32_t k = 0; k < 4; k++)
          *(ptr_reverse++) = accurate_cos(j * (i + k), _2N);
        for (int32_t k = 0; k < 4; k++)
          *(ptr_reverse++) = accurate_sin(j * (i + k), _2N);
      }
    }

    cudaMalloc((void **)&tables_direct_d, _2N * sizeof(double));
    cudaMalloc((void **)&tables_reverse_d, _2N * sizeof(double));

    

    // move table to GPU
    cudaMemcpy(tables_direct_d, tables_direct_h, _2N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tables_reverse_d, tables_reverse_h, _2N * sizeof(double), cudaMemcpyHostToDevice);

    cudaFreeHost(tables_direct_h);
    cudaFreeHost(tables_reverse_h);
  }

  template <class P>
  void LoadBK(const BootstrappingKeyFFT_test<P> &bkfft, uint32_t num_idx) {
    cudaMalloc((void **)&inout_direct_ds, num_idx * _2N * sizeof(double));
    cudaMalloc((void **)&inout_rev_is, 3 * num_idx * _2N * sizeof(double));
    cudaMalloc((void **)&tlwe_ds, num_idx * 1025 * sizeof(uint32_t));

    cudaMalloc((void **)&trgswfft_d, 672 * trgswLen * sizeof(double));
    for (int k = 0; k < 672; ++k) {
      uint32_t idk = k * trgswLen;
      for (int i = 0; i < 3; ++i) {
        uint32_t idx = idk + 4 * i * N;
        cudaMemcpy(trgswfft_d + idx, bkfft[k][i][0].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(trgswfft_d + idx + N, bkfft[k][i][1].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(trgswfft_d + idx + 2 * N, bkfft[k][i+3][0].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(trgswfft_d + idx + 3 * N, bkfft[k][i+3][1].data(), N * sizeof(double), cudaMemcpyHostToDevice);
      }
    }

  }

template <class P>
void MSBGateBootstrappingTLWE2TLWEFFT_kernels(
    std::array<uint32_t, 1025> &res,
    const std::array<uint32_t, 673> &tlwe, // lvl0
    const uint32_t u,
    uint32_t idx) {

  constexpr uint32_t roundoffset = 1048576;

  const uint32_t b̄ = 2048 - (tlwe[672] >> 21);
  //acc[0]=X^b*testVector
  PolynomialMulByXai_kernelv<<<1,128 >>>(inout_direct_ds + _2N*idx, u, b̄, N);
  cudaDeviceSynchronize();

  #pragma unroll
  for (int i = 0; i < 672; i++) {
	
    
    const uint32_t ā = (tlwe[i] + roundoffset) >> 21;
	
    if (ā == 0)
      continue;

    double *trgswfft_i = &trgswfft_d[i * trgswLen];
    //kernel融合
    trgswfftFMA_kernel3v<P><<<3, 128 >>>(inout_rev_is+ 3*_2N*idx, tables_reverse_d, inout_direct_ds+ _2N*idx, trgswfft_i, ā, Ns2);//ifft + BK[i]*((x^a-1)*acc[i])
    cudaDeviceSynchronize();
    reduce_fft<<<2, 64 >>>(tables_direct_d, inout_direct_ds + _2N * idx, inout_rev_is + 3*_2N * idx, Ns2);//fft
  }
  //cudaDeviceSynchronize();
  SampleExtractIndex_Kernelv<<<1, 128>>>(tlwe_ds+ 1025*idx, inout_direct_ds + _2N * idx, N);// TRLWE->TLWE
  cudaDeviceSynchronize();

  cudaMemcpyAsync(res.data(), tlwe_ds + 1025*idx, sizeof(uint32_t) * 1025, cudaMemcpyDeviceToHost);
  }

template <class P>
void MSBGateBootstrappingTLWE2TLWEFFT_st(
    uint32_t *res,
    const std::array<uint32_t, 673> &tlwe, // lvl0
    const uint32_t u,
    uint32_t idx, const cudaStream_t st) {

  constexpr uint32_t roundoffset = 1048576;

  const uint32_t b̄ = 2048 - (tlwe[672] >> 21);
  //acc[0]=X^b*testVector
  PolynomialMulByXai_kernelv<<<1,128,0,st>>>(inout_direct_ds + _2N*idx, u, b̄, N);
  cudaStreamSynchronize(st);

  #pragma unroll
  for (int i = 0; i < 672; i++) {
	
    
    const uint32_t ā = (tlwe[i] + roundoffset) >> 21;
	
    if (ā == 0)
      continue;

    double *trgswfft_i = &trgswfft_d[i * trgswLen];
    //kernel融合
    trgswfftFMA_kernel3v<P><<<3, 128, 0, st>>>(inout_rev_is+ 3*_2N*idx, tables_reverse_d, inout_direct_ds+ _2N*idx, trgswfft_i, ā, Ns2);//ifft + BK[i]*((x^a-1)*acc[i])
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(st);
    reduce_fft<<<2, 64, 0, st>>>(tables_direct_d, inout_direct_ds + _2N * idx, inout_rev_is + 3*_2N * idx, Ns2);//fft
  }
  //cudaDeviceSynchronize();
  SampleExtractIndex_Kernelv<<<1, 128, 0, st>>>(tlwe_ds+ 1025*idx, inout_direct_ds + _2N * idx, N);// TRLWE->TLWE
  cudaStreamSynchronize(st);
  //cudaDeviceSynchronize();

  cudaMemcpyAsync(res, tlwe_ds + 1025*idx, sizeof(uint32_t) * 1025, cudaMemcpyDeviceToHost, st);
  }

template <class P>
void IdeGateBootstrappingTLWE2TLWEFFT_kernels(
    std::array<uint32_t, 1025> &res,
    const std::array<uint32_t, 673> &tlwe, // lvl0
    const uint32_t scale_bits,
    uint32_t idx) {

  constexpr uint32_t roundoffset = 1048576;

  const uint32_t b̄ = 2048 - (tlwe[672] >> 21);
  //acc[0]=X^b*testVector
  PolynomialMulByXai_kernelg<<<1,128 >>>(inout_direct_ds + _2N*idx, scale_bits, b̄, N);
  cudaDeviceSynchronize();

  #pragma unroll
  for (int i = 0; i < 672; i++) {
	
    
    const uint32_t ā = (tlwe[i] + roundoffset) >> 21;
	
    if (ā == 0)
      continue;

    double *trgswfft_i = &trgswfft_d[i * trgswLen];
    //kernel融合
    trgswfftFMA_kernel3v<P><<<3, 128 >>>(inout_rev_is+ 3*_2N*idx, tables_reverse_d, inout_direct_ds+ _2N*idx, trgswfft_i, ā, Ns2);//ifft + BK[i]*((x^a-1)*acc[i])
    cudaDeviceSynchronize();
    reduce_fft<<<2, 64 >>>(tables_direct_d, inout_direct_ds + _2N * idx, inout_rev_is + 3*_2N * idx, Ns2);//fft
  }
  //cudaDeviceSynchronize();
  SampleExtractIndex_Kernelv<<<1, 128>>>(tlwe_ds+ 1025*idx, inout_direct_ds + _2N * idx, N);// TRLWE->TLWE
  cudaDeviceSynchronize();

  cudaMemcpyAsync(res.data(), tlwe_ds + 1025*idx, sizeof(uint32_t) * 1025, cudaMemcpyDeviceToHost);
  }

  template <class P>
  void IdeGateBootstrappingTLWE2TLWEFFT_st(
      uint32_t *res,
      const std::array<uint32_t, 673> &tlwe, // lvl0
      const uint32_t scale_bits, uint32_t idx, const cudaStream_t st) {

    constexpr uint32_t roundoffset = 1048576;

    const uint32_t b̄ = 2048 - (tlwe[672] >> 21);
    // acc[0]=X^b*testVector
    PolynomialMulByXai_kernelg<<<1, 128, 0, st>>>(inout_direct_ds + _2N * idx, scale_bits, b̄, N);
    cudaStreamSynchronize(st);

  #pragma unroll
  for (int i = 0; i < 672; i++) {
	
    
    const uint32_t ā = (tlwe[i] + roundoffset) >> 21;
	
    if (ā == 0)
      continue;

    double *trgswfft_i = &trgswfft_d[i * trgswLen];
    //kernel融合
    trgswfftFMA_kernel3v<P><<<3, 128, 0, st>>>(inout_rev_is+ 3*_2N*idx, tables_reverse_d, inout_direct_ds+ _2N*idx, trgswfft_i, ā, Ns2);//ifft + BK[i]*((x^a-1)*acc[i])
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(st);
    reduce_fft<<<2, 64, 0, st>>>(tables_direct_d, inout_direct_ds + _2N * idx, inout_rev_is + 3*_2N * idx, Ns2);//fft
  }
  //cudaDeviceSynchronize();
  SampleExtractIndex_Kernelv<<<1, 128, 0, st>>>(tlwe_ds+ 1025*idx, inout_direct_ds + _2N * idx, N);// TRLWE->TLWE
  cudaStreamSynchronize(st);

  cudaMemcpyAsync(res, tlwe_ds + 1025*idx, sizeof(uint32_t) * 1025, cudaMemcpyDeviceToHost, st);
  }

  ~CuFFT_TRLWE() {
    // free
    cudaFree(tables_direct_d);
    cudaFree(tables_reverse_d);
    cudaFree(trgswfft_d);

    cudaFree(inout_direct_ds);
    cudaFree(inout_rev_is);
    cudaFree(tlwe_ds);
  }


};


__global__ void warmupKernel() {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

void warmupGPU() {
  warmupKernel<<<1, 128>>>();

  cudaDeviceSynchronize();

  void *temp;
  cudaMalloc(&temp, 128);
  cudaFree(temp);
}