#pragma once

#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "BlindRotate_gpu.cuh"

namespace cufft {

template<class P>
class CuFFT_Torus {
public:
  const int32_t _2N;
  const int32_t N;
  const int32_t Ns2;
private:
  // device
  typename P::T *uint_inout_d;
  uint32_t *tlwe0_d;
  typename P::T *tlwe_d;
  double *buf;
  double *BootstrappingKeyfft_d;
  double *ifftb_d;
  double *trgswfft_d;
  volatile int *Syncin_d;
  volatile int *Syncout_d;
  uint32_t num_thread;
  uint32_t num_stream;
  std::vector<cudaStream_t> stream;

  size_t uint_pitch, tlwe0_pitch, tlwe_pitch, buf_pitch, in_pitch, out_pitch, trgswfft_pitch;

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
  // generate twiddle factor for FFT
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

    if constexpr (std::is_same_v<typename P::T, uint32_t>) {
      cudaMemcpyToSymbol(tables_direct_d, tables_direct_h, sizeof(double) * _2N);
      cudaMemcpyToSymbol(tables_reverse_d, tables_reverse_h, sizeof(double) * _2N);
    } else {
      cudaMemcpyToSymbol(tables_direct_d64, tables_direct_h, sizeof(double) * _2N);
      cudaMemcpyToSymbol(tables_reverse_d64, tables_reverse_h, sizeof(double) * _2N);
    }

    typename P::T *b_h;
    cudaMallocHost(&b_h, sizeof(typename P::T) * N);

    for (int i = 0; i < N; ++i) {
      b_h[i] = 1;
    }

    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpy(b_d, b_h, sizeof(typename P::T) * N, cudaMemcpyHostToDevice);

    cufft::ifft<<<1, num_thread, 0, stream[0]>>>(ifftb_d, b_d, Ns2);

    cudaFreeHost(tables_direct_h);
    cudaFreeHost(tables_reverse_h);
    cudaFreeHost(b_h);
    cudaFree(b_d);
  }

  __host__ inline CuFFT_Torus(const int num_stream = 20): _2N(2 * P::n), N(P::n), Ns2(P::n / 2), num_thread(P::n >> 4), num_stream(num_stream) {
	  
    // inout_table
    cudaMallocPitch(&uint_inout_d, &uint_pitch, _2N * sizeof(typename P::T), num_stream);

    cudaMallocPitch(&tlwe0_d, &tlwe0_pitch, 673 * sizeof(uint32_t), num_stream);
    cudaMallocPitch(&tlwe_d, &tlwe_pitch, (P::n + 1) * sizeof(typename P::T), num_stream);

    cudaMallocPitch(&buf, &buf_pitch, (P::k + 1) * P::l * _2N * sizeof(double), num_stream);
    cudaMallocPitch(&trgswfft_d, &trgswfft_pitch, (P::k + 1) * P::l * _2N * sizeof(double), num_stream);
    cudaMalloc(&BootstrappingKeyfft_d, 672 * (P::k + 1) * P::l * _2N * sizeof(double));
    cudaMalloc(&ifftb_d, N * sizeof(double));

    cudaMallocPitch((void **)&Syncin_d, &in_pitch, (P::k + 1) * P::l * sizeof(int), num_stream);
    cudaMemset2D((void *)Syncin_d, in_pitch, 0, (P::k + 1) * P::l * sizeof(int), num_stream);
    cudaMallocPitch((void **)&Syncout_d, &out_pitch, (P::k + 1) * P::l * sizeof(int), num_stream);
    cudaMemset2D((void *)Syncout_d, out_pitch, 0, (P::k + 1) * P::l * sizeof(int), num_stream);

    // init stream
    stream.resize(num_stream);
    for (int i = 0; i < num_stream; ++i) {
      cudaStreamCreate(&stream[i]);
      //cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }
    
    // trig_table
    new_fft_table();

  }

  __host__ inline void MoveBuf(std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, const int stream_id = 0) {
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    
    for (int i = 0; i < (P::k + 1) * P::l; ++i) {
      cudaMemcpy2D(trgswfft[i][0].data(), trgswfft_pitch, buf_i + _2N * i, N * sizeof(double), N * sizeof(double), 1, cudaMemcpyDeviceToHost);
      cudaMemcpy2D(trgswfft[i][1].data(), trgswfft_pitch, buf_i + _2N * i + N, N * sizeof(double),  N * sizeof(double), 1, cudaMemcpyDeviceToHost);
    }
  }

  // Pre-loading Bootstrapping Key to GMEM
  __host__ inline void LoadBK(const std::array<std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l>, 672> &BootstrappingKeyfft) {
    int trgswlen = (P::k + 1) * P::l * _2N;
    for (int k = 0; k < 672; ++k) {
      int offset = k * trgswlen;
      for (int i = 0; i < (P::k + 1) * P::l; ++i) {
        cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i, BootstrappingKeyfft[k][i][0].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i + N, BootstrappingKeyfft[k][i][1].data(), N * sizeof(double), cudaMemcpyHostToDevice);
      }
    }
  }

  // ifft
  __host__ inline void ifft_st(std::array<double, P::n> &res, std::array<typename P::T, P::n> &a, const uint32_t stream_id) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpyAsync(b_d, a.data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::ifft<<<1, num_thread, 0, stream[stream_id]>>>(ifftb_d, b_d, Ns2);
    cudaStreamSynchronize(stream[stream_id]);
    cudaMemcpyAsync(res.data(), ifftb_d, sizeof(double) * N, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaFree(b_d);

  }

  // fft
  __host__ inline void fft_st(std::array<typename P::T, P::n> &res, std::array<double, P::n> &a, const uint32_t stream_id) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));

    cudaMemcpyAsync(ifftb_d, a.data(), sizeof(double) * N, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::fft<<<1, num_thread, 0, stream[stream_id]>>>(b_d, ifftb_d, Ns2);
    cudaStreamSynchronize(stream[stream_id]);
    
    cudaMemcpyAsync(res.data(), b_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost, stream[stream_id]);

  }

  // polynomial multiplication
  __host__ inline void PolyMul_st(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const uint32_t stream_id) {

    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    
    cudaMemcpy2DAsync(uint_inout_i, uint_pitch, trlwe[0].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);
    cudaMemcpy2DAsync(uint_inout_i + N, uint_pitch, trlwe[1].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::PolyMul<P><<<2, num_thread, 0, stream[stream_id]>>>(uint_inout_i, uint_inout_i, ifftb_d, Ns2);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(res[0].data(), uint_pitch, uint_inout_i, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaMemcpy2DAsync(res[1].data(), uint_pitch, uint_inout_i + N, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
  }

  __host__ inline void ExternalProduct_st(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, const uint32_t stream_id) {
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    double *trgswfft_i = (double *)((char *)trgswfft_d + stream_id * trgswfft_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(uint_inout_i, uint_pitch, trlwe[0].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);
    cudaMemcpy2DAsync(uint_inout_i + N, uint_pitch, trlwe[1].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    for (int i = 0; i < (P::k + 1) * P::l; ++i) {
      cudaMemcpy2DAsync(trgswfft_i + _2N * i, trgswfft_pitch, trgswfft[i][0].data(), N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice, stream[stream_id]);
      cudaMemcpy2DAsync(trgswfft_i + _2N * i + N, trgswfft_pitch, trgswfft[i][1].data(), N * sizeof(double),  N * sizeof(double), 1, cudaMemcpyHostToDevice, stream[stream_id]);
    }

    cufft::ExternalProduct<P><<<(P::k + 1) * P::l, num_thread, 0, stream[stream_id]>>>(uint_inout_i, uint_inout_i, trgswfft_i, buf_i, Ns2, SyncIn, SyncOut);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(res[0].data(), uint_pitch, uint_inout_i, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaMemcpy2DAsync(res[1].data(), uint_pitch, uint_inout_i + N, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
  }

  __host__ inline void GateBootstrappingTLWE2TLWEFFT_st(std::array<typename P::T, P::n + 1> &tlwe, std::array<uint32_t, 673> &poly, const typename P::T u, const uint32_t stream_id) {

    uint32_t *tlwe0_i = (uint32_t *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(uint32_t) * 673, sizeof(uint32_t) * 673, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::GateBootstrappingTLWE2TLWEFFT<P><<<(P::k + 1) * P::l, num_thread, 0, stream[stream_id]>>>(tlwe_i, uint_inout_i, tlwe0_i, BootstrappingKeyfft_d, u, buf_i, Ns2, SyncIn, SyncOut, false);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);
#if 0
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  __host__ inline void IdeGateBootstrappingTLWE2TLWEFFT_st(std::array<typename P::T, P::n + 1> &tlwe, std::array<uint32_t, 673> &poly, const uint32_t scale_bits, const uint32_t stream_id) {

    uint32_t *tlwe0_i = (uint32_t *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(uint32_t) * 673, sizeof(uint32_t) * 673, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufft::GateBootstrappingTLWE2TLWEFFT<P><<<(P::k + 1) * P::l, num_thread, 0, stream[stream_id]>>>(tlwe_i, uint_inout_i, tlwe0_i, BootstrappingKeyfft_d, scale_bits, buf_i, Ns2, SyncIn, SyncOut, true);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);
#if 0  
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  // for test
  __host__ inline void ifft_test(std::array<double, P::n> &res, std::array<typename P::T, P::n> &a, uint32_t test_num, double &cost) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpy(b_d, a.data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufft::ifft<<<1, num_thread>>>(ifftb_d, b_d, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res.data(), ifftb_d, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(b_d);

  }

  __host__ inline void GateBootstrappingTLWE2TLWEFFT_test(std::array<typename P::T, P::n + 1> &tlwe, std::array<uint32_t, 673> &poly, const typename P::T u, uint32_t test_num, double &cost) {

    uint32_t *tlwe0_i = (uint32_t *)((char *)tlwe0_d);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d);
    double *buf_i = (double *)((char *)buf);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d);

    cudaMemcpy2D(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(uint32_t) * 673, sizeof(uint32_t) * 673, 1, cudaMemcpyHostToDevice);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufft::GateBootstrappingTLWE2TLWEFFT<P><<<(P::k + 1) * P::l, num_thread>>>(tlwe_i, uint_inout_i, tlwe0_i, BootstrappingKeyfft_d, u, buf_i, Ns2, SyncIn, SyncOut, false);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy2D(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost);
  }

  __host__ inline void ExternalProduct_test(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, uint32_t test_num, double &cost) {
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d);
    double *buf_i = (double *)((char *)buf);
    double *trgswfft_i = (double *)((char *)trgswfft_d);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d);

    typename P::T *out_d;
    cudaMalloc(&out_d, _2N * sizeof(typename P::T));

    cudaMemcpy2D(uint_inout_i, uint_pitch, trlwe[0].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice);
    cudaMemcpy2D(uint_inout_i + N, uint_pitch, trlwe[1].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice);

    for (int i = 0; i < (P::k + 1) * P::l; ++i) {
      cudaMemcpy2D(trgswfft_i + _2N * i, trgswfft_pitch, trgswfft[i][0].data(), N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice);
      cudaMemcpy2D(trgswfft_i + _2N * i + N, trgswfft_pitch, trgswfft[i][1].data(), N * sizeof(double),  N * sizeof(double), 1, cudaMemcpyHostToDevice);
    }

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufft::ExternalProduct<P><<<(P::k + 1) * P::l, num_thread>>>(out_d, uint_inout_i, trgswfft_i, buf_i, Ns2, SyncIn, SyncOut);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res[0].data(), out_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(res[1].data(), out_d + N, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
  }

  __host__ inline void ExternalProduct_withoutFusion_test(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, uint32_t test_num, double &cost) {
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d);
    double *buf_i = (double *)((char *)buf);
    double *trgswfft_i = (double *)((char *)trgswfft_d);
  
    typename P::T *out_d;
    cudaMalloc(&out_d, _2N * sizeof(typename P::T));

    cudaMemcpy2D(uint_inout_i, uint_pitch, trlwe[0].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice);
    cudaMemcpy2D(uint_inout_i + N, uint_pitch, trlwe[1].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice);

    for (int i = 0; i < (P::k + 1) * P::l; ++i) {
      cudaMemcpy2D(trgswfft_i + _2N * i, trgswfft_pitch, trgswfft[i][0].data(), N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice);
      cudaMemcpy2D(trgswfft_i + _2N * i + N, trgswfft_pitch, trgswfft[i][1].data(), N * sizeof(double),  N * sizeof(double), 1, cudaMemcpyHostToDevice);
    }

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufft::MulByTRGSWFFT<P><<<(P::k + 1) * P::l, num_thread>>>(uint_inout_i, trgswfft_i, buf_i, Ns2);
      cudaDeviceSynchronize();
      cufft::Reduction<P><<<(P::k + 1), num_thread>>>(out_d, buf_i, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res[0].data(), out_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(res[1].data(), out_d + N, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
  }

  __host__ inline void GateBootstrappingTLWE2TLWEFFT_withoutFusion_test(std::array<typename P::T, P::n + 1> &tlwe, std::array<uint32_t, 673> &poly, const typename P::T u, uint32_t test_num, double &cost) {

    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d);
    double *buf_i = (double *)((char *)buf);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    constexpr uint32_t roundoffset = 1048576;
    const uint32_t b = 2048 - (poly[672] >> 21);
    for (int i=0;i<test_num;++i) {
      cufft::PolynomialMulByXai<P><<<1, 128>>>(uint_inout_d, u, b);
      cudaDeviceSynchronize();
      #pragma unroll
      for (int i = 0; i < 672; ++i) {

        const uint32_t a = (poly[i] + roundoffset) >> 21;
        if (a == 0)
          continue;
        double *trgswfft_i = &BootstrappingKeyfft_d[i * (P::k + 1) * P::l * _2N];
        cufft::MulByTRGSWFFT<P><<<(P::k + 1) * P::l, num_thread>>>(uint_inout_i, trgswfft_i, buf_i, Ns2);
        cudaDeviceSynchronize();
        cufft::Reduction<P><<<(P::k + 1), num_thread>>>(uint_inout_i, buf_i, Ns2);
        cudaDeviceSynchronize();
      }
      cufft::SampleExtractIndex_Kernel<<<1, 128>>>(tlwe_i, uint_inout_i, N);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy2D(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost);
  }

  __host__ inline ~CuFFT_Torus() {
    // free
    cudaFree(uint_inout_d);
    cudaFree(BootstrappingKeyfft_d);
    cudaFree(ifftb_d);
    cudaFree(tlwe0_d);
    cudaFree(tlwe_d);
    cudaFree(buf);
    cudaFree(trgswfft_d);
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
  warmupKernel<<<8, 128>>>();

  cudaDeviceSynchronize();

  void *temp;
  cudaMalloc(&temp, 128);
  cudaFree(temp);
}