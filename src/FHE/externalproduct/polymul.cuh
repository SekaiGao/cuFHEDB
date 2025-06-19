#pragma once
#include "utils.cuh"

namespace cufhedb {
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
  shared_pre[idx] = uint4ToDouble4(in_rev_dre[idx]);
  shared_pre[idx + 64] = uint4ToDouble4(in_rev_dre[idx + 64]);
  shared_pim[idx] = uint4ToDouble4(in_rev_dim[idx]);
  shared_pim[idx + 64] = uint4ToDouble4(in_rev_dim[idx + 64]);

  double4 *__restrict__ ifftb4 = (double4 *)(ifftb);

  // IFFT
  ifft1024(shared_pre, shared_pim, Ns2, idx);

  // MulInFD
  MulInFD(ifftb4, shared_pre, shared_pim, idx);

  // FFT
  fft1024(out_direct_dre, out_direct_dim, shared_pre, shared_pim, Ns2, idx);

}
};