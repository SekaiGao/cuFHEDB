#pragma once

#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "BlindRotate_gpu.cuh"

namespace cufft {

class CuFFT_Torus {
public:
  const int32_t _2N;
  const int32_t N;
  const int32_t Ns2;
private:
  // device
  uint32_t *uint_inout_d;
  uint32_t *tlwe_d;
  uint32_t *tlwe1_d;
  double *buf;
  double *BootstrappingKeyfft_d;
  volatile int *Syncin_d;
  volatile int *Syncout_d;
  uint32_t num_stream;
  std::vector<cudaStream_t> stream;

  size_t uint_pitch, tlwe_pitch, tlwe1_pitch, buf_pitch, in_pitch, out_pitch;

  __host__ __device__ inline double accurate_cos(int32_t i, int32_t n) { // cos(2pi*i/n)
    i = ((i % n) + n) % n;
    if (i >= 3 * n / 4)
      return cos(2. * M_PI * (n - i) / double(n));
    if (i >= 2 * n / 4)
      return -cos(2. * M_PI * (i - n / 2) / double(n));
    if (i >= 1 * n / 4)
      return -cos(2. * M_PI * (n / 2 - i) / double(n));
    return cos(2. * M_PI * (i) / double(n));
  }

  __host__ __device__ inline double accurate_sin(int32_t i, int32_t n) { // sin(2pi*i/n)
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
  __host__ inline void new_fft_table() {
    int32_t ns4 = _2N / 4;

    double *tables_direct_h;
    double *tables_reverse_h;

    // pinned memory
    cudaMallocHost(&tables_direct_h, sizeof(double) * _2N);
    cudaMallocHost(&tables_reverse_h, sizeof(double) * _2N);

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

    cudaMemcpyToSymbol(tables_direct_d, tables_direct_h, sizeof(double) * _2N);
    cudaMemcpyToSymbol(tables_reverse_d, tables_reverse_h, sizeof(double) * _2N);

    cudaFreeHost(tables_direct_h);
    cudaFreeHost(tables_reverse_h);
  }

  __host__ inline CuFFT_Torus(const int32_t N, const int num_stream = 10): _2N(2 * N), N(N), Ns2(N / 2), num_stream(num_stream) {
    
    // trig_table
    new_fft_table();
	  
    // inout_table
    cudaMallocPitch(&uint_inout_d, &uint_pitch, _2N * sizeof(uint32_t), num_stream);

    cudaMallocPitch(&tlwe_d, &tlwe_pitch, 673 * sizeof(uint32_t), num_stream);
    cudaMallocPitch(&tlwe1_d, &tlwe1_pitch, 1025 * sizeof(uint32_t), num_stream);

    cudaMallocPitch(&buf, &buf_pitch, 6 * _2N * sizeof(double), num_stream);
    cudaMalloc(&BootstrappingKeyfft_d, 672 * 6 * _2N * sizeof(double));

    cudaMallocPitch((void **)&Syncin_d, &in_pitch, 6 * sizeof(int), num_stream);
    cudaMemset2D((void *)Syncin_d, in_pitch, 0, 6 * sizeof(int), num_stream);
    cudaMallocPitch((void **)&Syncout_d, &out_pitch, 6 * sizeof(int), num_stream);
    cudaMemset2D((void *)Syncout_d, out_pitch, 0, 6 * sizeof(int), num_stream);

    // init stream
    stream.resize(num_stream);
    for (int i = 0; i < num_stream; ++i) {
      cudaStreamCreate(&stream[i]);
    }
  }

  __host__ inline void MoveIn(std::array<uint32_t, 673> &poly, const int row_idx = 0) {
    uint32_t *dest_ptr = (uint32_t *)((char *)tlwe_d + row_idx * tlwe_pitch);
    cudaMemcpy2D(dest_ptr, tlwe_pitch, poly.data(), sizeof(uint32_t) * 673, sizeof(uint32_t) * 673, 1, cudaMemcpyHostToDevice);
  }

  __host__ inline void MoveOut(std::array<uint32_t, 1025> &tlwe1, const int row_idx = 0) {
    uint32_t *dest_ptr = (uint32_t *)((char *)tlwe1_d + row_idx * tlwe1_pitch);
    cudaMemcpy2D(tlwe1.data(), tlwe1_pitch, dest_ptr, sizeof(uint32_t) * 1025, sizeof(uint32_t) * 1025, 1, cudaMemcpyDeviceToHost);
  }

  template<class P>
  __host__ inline void LoadBK(const std::array<std::array<std::array<std::array<double, P::n>, 2>, 6>, 672> &BootstrappingKeyfft) {
    int trgswlen = 6 * _2N;
    for (int k = 0; k < 672; ++k) {
      int offset = k * trgswlen;
      for (int i = 0; i < 6; ++i) {
        cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i, BootstrappingKeyfft[k][i][0].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i + N, BootstrappingKeyfft[k][i][1].data(), N * sizeof(double), cudaMemcpyHostToDevice);
      }
    }
  }

  template <class P>
  __host__ inline void GateBootstrappingTLWE2TLWEFFT_kernel(const uint32_t u, const int row_idx = 0, const int idx = 0) {
    uint32_t *tlwe1_i = (uint32_t *)((char *)tlwe1_d + idx * tlwe1_pitch);
    uint32_t *tlwe_i = (uint32_t *)((char *)tlwe_d + idx * tlwe_pitch);
    uint32_t *uint_inout_i = (uint32_t *)((char *)uint_inout_d + row_idx * uint_pitch);
    double *buf_i = (double *)((char *)buf + row_idx * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + idx * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + idx * out_pitch);

    cufft::GateBootstrappingTLWE2TLWEFFT<P><<<6, 64>>>(tlwe1_i, uint_inout_i, tlwe_i, BootstrappingKeyfft_d, u, buf_i, Ns2, SyncIn, SyncOut, false);
    cudaDeviceSynchronize();
#if 0
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
#endif
  }


  template <class P>
  __host__ inline void GateBootstrappingTLWE2TLWEFFT_st(std::array<uint32_t, 1025> &tlwe1, std::array<uint32_t, 673> &poly, const uint32_t u, const uint32_t stream_id) {

    uint32_t *tlwe_i = (uint32_t *)((char *)tlwe_d + stream_id * tlwe_pitch);
    uint32_t *tlwe1_i = (uint32_t *)((char *)tlwe1_d + stream_id * tlwe1_pitch);
    uint32_t *uint_inout_i = (uint32_t *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(tlwe_i, tlwe_pitch, poly.data(), sizeof(uint32_t) * 673, sizeof(uint32_t) * 673, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::GateBootstrappingTLWE2TLWEFFT<P><<<6, 64, 0, stream[stream_id]>>>(tlwe1_i, uint_inout_i, tlwe_i, BootstrappingKeyfft_d, u, buf_i, Ns2, SyncIn, SyncOut, false);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe1.data(), tlwe1_pitch, tlwe1_i, sizeof(uint32_t) * 1025, sizeof(uint32_t) * 1025, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
#if 0
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  template <class P>
  __host__ inline void IdeGateBootstrappingTLWE2TLWEFFT_kernel(const uint32_t scale_bits, const int row_idx = 0) {

    uint32_t *tlwe1_i = (uint32_t *)((char *)tlwe1_d + row_idx * tlwe1_pitch);
    uint32_t *tlwe_i = (uint32_t *)((char *)tlwe_d + row_idx * tlwe_pitch);
    uint32_t *uint_inout_i = (uint32_t *)((char *)uint_inout_d + row_idx * uint_pitch);
    double *buf_i = (double *)((char *)buf + row_idx * buf_pitch);
    volatile int *Syncin_i = (volatile int *)((char *)Syncin_d + row_idx * in_pitch);
    volatile int *Syncout_i = (volatile int *)((char *)Syncout_d + row_idx * out_pitch);

    cufft::GateBootstrappingTLWE2TLWEFFT<P><<<6, 64>>>(tlwe1_i, uint_inout_i, tlwe_i, BootstrappingKeyfft_d, scale_bits, buf_i, Ns2, Syncin_i, Syncout_i, true);
    cudaDeviceSynchronize();
#if 0
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
#endif
  }

  template <class P>
  __host__ inline void IdeGateBootstrappingTLWE2TLWEFFT_st(std::array<uint32_t, 1025> &tlwe1, std::array<uint32_t, 673> &poly, const uint32_t scale_bits, const uint32_t stream_id) {

    uint32_t *tlwe_i = (uint32_t *)((char *)tlwe_d + stream_id * tlwe_pitch);
    uint32_t *tlwe1_i = (uint32_t *)((char *)tlwe1_d + stream_id * tlwe1_pitch);
    uint32_t *uint_inout_i = (uint32_t *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(tlwe_i, tlwe_pitch, poly.data(), sizeof(uint32_t) * 673, sizeof(uint32_t) * 673, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::GateBootstrappingTLWE2TLWEFFT<P><<<6, 64, 0, stream[stream_id]>>>(tlwe1_i, uint_inout_i, tlwe_i, BootstrappingKeyfft_d, scale_bits, buf_i, Ns2, SyncIn, SyncOut, true);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe1.data(), tlwe1_pitch, tlwe1_i, sizeof(uint32_t) * 1025, sizeof(uint32_t) * 1025, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
#if 0  
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  __host__ inline ~CuFFT_Torus() {
    // free
    cudaFree(uint_inout_d);
    cudaFree(BootstrappingKeyfft_d);
    cudaFree(tlwe_d);
    cudaFree(tlwe1_d);
    cudaFree(buf);
    cudaFree((void *)Syncin_d);
    cudaFree((void *)Syncout_d);

    for (int i = 0; i < num_stream; ++i) {
      cudaStreamDestroy(stream[i]);
    }
  }
};

};

__global__ void warmupKernel() {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

void warmupGPU() {
  warmupKernel<<<6, 64>>>();

  cudaDeviceSynchronize();

  void *temp;
  cudaMalloc(&temp, 128);
  cudaFree(temp);
}