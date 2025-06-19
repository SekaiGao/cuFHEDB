#pragma once

#include "FHE/bootstrapping/GateBootstrapping.cuh"
#include "FHE/externalproduct/polymul.cuh"
#include <array>
#include <cuda_runtime.h>
#include <vector>
#include "utils.cuh"

#define Debug false

namespace cufhedb {

/*
 * A class responsible for managing FHE operations on the GPU.
 * It handles memory allocation on the GPU, initializes CUDA streams,
 * and provides APIs for performing various FHE operations.
 * 
 * Template:
 *    P: security level (Lvl1, Lvl2)
 *    UNROLL: whether enable bootstrapping unroll
 */

template<class P, bool UNROLL = false>
class cuCoreFHE {
public:
  const int32_t _2N;
  const int32_t N;
  const int32_t Ns2;
  const int32_t num_Addends = 2;
  bool is_cleaned = false;

private:
  // device
  typename P::T *uint_inout_d;
  Lvl0_T *tlwe0_d;
  typename P::T *tlwe_d;
  double *buf;
  double *BootstrappingKeyfft_d;
  double *ifftb_d;
  double *trgswfft_d;
  double *XaiInFD_d;
  double *KeyBundleKeyfft_d;
  double *onetrgsw_d;
  volatile int *Syncin_d;
  volatile int *Syncout_d;
  uint32_t num_thread;
  uint32_t num_stream;

  std::vector<cudaStream_t> stream;

  size_t uint_pitch, tlwe0_pitch, tlwe_pitch, buf_pitch, in_pitch, out_pitch, trgswfft_pitch, kbk_pitch;

  // cos(2pi*i/n)
  __host__ __device__ inline double accurate_cos(int32_t i, int32_t n) { 
    i = ((i % n) + n) % n;
    if (i >= 3 * n / 4)
      return cos(2. * M_PI * (n - i) / double(n));
    if (i >= 2 * n / 4)
      return -cos(2. * M_PI * (i - n / 2) / double(n));
    if (i >= 1 * n / 4)
      return -cos(2. * M_PI * (n / 2 - i) / double(n));
    return cos(2. * M_PI * (i) / double(n));
  }

  // sin(2pi*i/n)
  __host__ __device__ inline double accurate_sin(int32_t i, int32_t n) { 
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

    cufhedb::ifft<<<1, num_thread, 0, stream[0]>>>(ifftb_d, b_d, Ns2);

    cudaFreeHost(tables_direct_h);
    cudaFreeHost(tables_reverse_h);
    cudaFreeHost(b_h);
    cudaFree(b_d);
  }

  // Allocates device memory, initializes CUDA streams.
  __host__ inline cuCoreFHE(const int num_stream = 20): _2N(2 * P::n), N(P::n), Ns2(P::n / 2), num_thread(P::n >> 4), num_stream(num_stream) {
	  
    // inout_table
    cudaMallocPitch(&uint_inout_d, &uint_pitch, _2N * sizeof(typename P::T), num_stream);

    uint32_t lvl0_n = Lvl0_n;
    if (std::is_same<P, TFHEpp::lvl1param80>::value) {
      lvl0_n = Lvl0_n80;
    }

    cudaMallocPitch(&tlwe0_d, &tlwe0_pitch, (lvl0_n + 1) * sizeof(Lvl0_T), num_stream);
    cudaMallocPitch(&tlwe_d, &tlwe_pitch, (P::n + 1) * sizeof(typename P::T), num_stream);

    cudaMallocPitch(&buf, &buf_pitch, (P::k + 1) * P::l * _2N * sizeof(double), num_stream);
    cudaMallocPitch(&trgswfft_d, &trgswfft_pitch, (P::k + 1) * P::l * _2N * sizeof(double), num_stream);
    
    // unroll 2
    //#ifdef UNROLL
    if constexpr (UNROLL) {
      cudaMalloc(&BootstrappingKeyfft_d, (lvl0_n / num_Addends) * (2 * num_Addends - 1) * (P::k + 1) * P::l * _2N * sizeof(double));
      
      // trgsw(1)
      cudaMalloc(&onetrgsw_d, (P::k + 1) * P::l * _2N * sizeof(double));
      
      alignas(64) const TFHEpp::TRGSWFFT<P> onetrgsw = cufhedb::oneTRGSWFFTgen<P>();
      int trgswlen = (P::k + 1) * P::l * _2N;
      for (int j = 0; j < (P::k + 1) * P::l; ++j) {
        double temp_data[2 * N];
        std::memcpy(temp_data, onetrgsw[j][0].data(), N * sizeof(double));
        std::memcpy(temp_data + N, onetrgsw[j][1].data(), N * sizeof(double));
        cudaMemcpy(onetrgsw_d + _2N * j, temp_data, _2N * sizeof(double), cudaMemcpyHostToDevice);
      }

      // KeyBundle Key is huge
      cudaMallocPitch(&KeyBundleKeyfft_d, &kbk_pitch, (lvl0_n / num_Addends) * (P::k + 1) * P::l * _2N * sizeof(double), num_stream);

      // (X^ai - 1) in Fourier domain
      cudaMalloc(&XaiInFD_d, _2N * N * sizeof(double));

      alignas(64) const std::unique_ptr<const std::array<TFHEpp::PolynomialInFD<P>, 2 * P::n>> xaitt = TFHEpp::XaittGen<P>();
      for (int k = 0; k < 2 * P::n; ++k) {
          int offset = k * N;
          cudaMemcpy(XaiInFD_d + offset, (*xaitt)[k].data(), N * sizeof(double), cudaMemcpyHostToDevice);
      }
    } else
      cudaMalloc(&BootstrappingKeyfft_d, lvl0_n * (P::k + 1) * P::l * _2N * sizeof(double));

    cudaMalloc(&ifftb_d, N * sizeof(double));

    // memory for inter-block synchronization
    cudaMallocPitch((void **)&Syncin_d, &in_pitch, (P::k + 1) * P::l * sizeof(int), num_stream);
    cudaMemset2D((void *)Syncin_d, in_pitch, 10000, (P::k + 1) * P::l * sizeof(int), num_stream);
    cudaMallocPitch((void **)&Syncout_d, &out_pitch, (P::k + 1) * P::l * sizeof(int), num_stream);
    cudaMemset2D((void *)Syncout_d, out_pitch, 10000, (P::k + 1) * P::l * sizeof(int), num_stream);

    // init stream
    stream.resize(num_stream);
    for (int i = 0; i < num_stream; ++i) {
      cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }
    
    // trig_table
    new_fft_table();

  }

  // Pre-generating Bootstrapping Key to global memory
  __host__ inline void emplaceBK(TFHEpp::SecretKey &sk) {
    int trgswlen = (P::k + 1) * P::l * _2N;

    using bkP = bkP_T<P>;

    //#ifdef UNROLL
    if constexpr (UNROLL) {
    std::unique_ptr<TFHEpp::KBKFFT<bkP>> bkfft = std::make_unique<TFHEpp::KBKFFT<bkP>>();
    TFHEpp::KBKfftgen<bkP>(*bkfft, sk);

    int addends = 2 * num_Addends - 1;
    for (int k = 0; k < Lvl0_n / num_Addends; ++k) {
      for (int j = 0; j < addends; ++j) {
        int offset = (k * addends + j) * trgswlen;
        for (int i = 0; i < (P::k + 1) * P::l; ++i) {
          cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i, (*bkfft)[k][j][i][0].data(), N * sizeof(double), cudaMemcpyHostToDevice);
          cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i + N, (*bkfft)[k][j][i][1].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        }
      }
    }
    } 
    else {
    std::unique_ptr<TFHEpp::BKFFT<bkP>> bkfft = std::make_unique<TFHEpp::BKFFT<bkP>>();
    TFHEpp::BKfftgen<bkP>(*bkfft, sk);
    for (int k = 0; k < Lvl0_n; ++k) {
      int offset = k * trgswlen;
      for (int i = 0; i < (P::k + 1) * P::l; ++i) {
        cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i, (*bkfft)[k][0][i][0].data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(BootstrappingKeyfft_d + offset + _2N * i + N, (*bkfft)[k][0][i][1].data(), N * sizeof(double), cudaMemcpyHostToDevice);
      }
    }
    }
  }

  // ifft
  __host__ inline void ifft_st(std::array<double, P::n> &res, std::array<typename P::T, P::n> &a, const uint32_t stream_id) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpyAsync(b_d, a.data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice, stream[stream_id]);

    cufhedb::ifft<<<1, num_thread, 0, stream[stream_id]>>>(ifftb_d, b_d, Ns2);
    cudaStreamSynchronize(stream[stream_id]);
    cudaMemcpyAsync(res.data(), ifftb_d, sizeof(double) * N, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaFree(b_d);
  }

  // fft
  __host__ inline void fft_st(std::array<typename P::T, P::n> &res, std::array<double, P::n> &a, const uint32_t stream_id) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));

    cudaMemcpyAsync(ifftb_d, a.data(), sizeof(double) * N, cudaMemcpyHostToDevice, stream[stream_id]);

    cufhedb::fft<<<1, num_thread, 0, stream[stream_id]>>>(b_d, ifftb_d, Ns2);
    cudaStreamSynchronize(stream[stream_id]);
    
    cudaMemcpyAsync(res.data(), b_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost, stream[stream_id]);

  }

  // polynomial multiplication
  __host__ inline void PolyMul_st(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const uint32_t stream_id) {

    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    
    cudaMemcpy2DAsync(uint_inout_i, uint_pitch, trlwe[0].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);
    cudaMemcpy2DAsync(uint_inout_i + N, uint_pitch, trlwe[1].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufhedb::PolyMul<P><<<2, num_thread, 0, stream[stream_id]>>>(uint_inout_i, uint_inout_i, ifftb_d, Ns2);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(res[0].data(), uint_pitch, uint_inout_i, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaMemcpy2DAsync(res[1].data(), uint_pitch, uint_inout_i + N, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
  }

  // external product
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

    cufhedb::ExternalProduct<P><<<(P::k + 1) * P::l, num_thread, 0, stream[stream_id]>>>(uint_inout_i, uint_inout_i, trgswfft_i, buf_i, Ns2, SyncIn, SyncOut);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(res[0].data(), uint_pitch, uint_inout_i, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaMemcpy2DAsync(res[1].data(), uint_pitch, uint_inout_i + N, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
  }

  // external product use cooporative group
  __host__ inline void ExternalProductCG_st(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, const uint32_t stream_id) {
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    double *trgswfft_i = (double *)((char *)trgswfft_d + stream_id * trgswfft_pitch);
    cudaMemcpy2DAsync(uint_inout_i, uint_pitch, trlwe[0].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);
    cudaMemcpy2DAsync(uint_inout_i + N, uint_pitch, trlwe[1].data(), sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyHostToDevice, stream[stream_id]);

    for (int i = 0; i < (P::k + 1) * P::l; ++i) {
      cudaMemcpy2DAsync(trgswfft_i + _2N * i, trgswfft_pitch, trgswfft[i][0].data(), N * sizeof(double), N * sizeof(double), 1, cudaMemcpyHostToDevice, stream[stream_id]);
      cudaMemcpy2DAsync(trgswfft_i + _2N * i + N, trgswfft_pitch, trgswfft[i][1].data(), N * sizeof(double),  N * sizeof(double), 1, cudaMemcpyHostToDevice, stream[stream_id]);
    }

    void (*kernel_ptr)(typename P::T*, typename P::T*, double*, double*, const int32_t) = cufhedb::ExternalProductCG<P>;
    void* kernelArgs[] = {&uint_inout_i, &uint_inout_i, &trgswfft_i, &buf_i, const_cast<int32_t*>(&Ns2)};
    cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);

    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(res[0].data(), uint_pitch, uint_inout_i, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
    cudaMemcpy2DAsync(res[1].data(), uint_pitch, uint_inout_i + N, sizeof(typename P::T) * N, sizeof(typename P::T) * N, 1, cudaMemcpyDeviceToHost, stream[stream_id]);
  }

  // gate boootstrapping
  __host__ inline void GateBootstrappingTLWE2TLWEFFT_st(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n + 1> &poly, const typename P::T u, const uint32_t stream_id) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n + 1), sizeof(Lvl0_T) * (Lvl0_n + 1), 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufhedb::GateBootstrappingTLWE2TLWEFFT<P, false><<<(P::k + 1) * P::l, num_thread, 0, stream[stream_id]>>>(tlwe_i, uint_inout_i, tlwe0_i, BootstrappingKeyfft_d, u, buf_i, Ns2, SyncIn, SyncOut);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);
#if Debug
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  // gate boootstrapping (need OpenMP to assist concurrent schedule)
  __host__ inline void GateBootstrapping_st(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n + 1> &poly, const typename P::T u, const uint32_t stream_id) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n + 1), sizeof(Lvl0_T) * (Lvl0_n + 1), 1, cudaMemcpyHostToDevice, stream[stream_id]);
    
    if constexpr (UNROLL) {
      double *KBKfft_i = (double *)((char *)KeyBundleKeyfft_d + stream_id * kbk_pitch);
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, double*, double*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, false>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, &XaiInFD_d, &KBKfft_i, &onetrgsw_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    }
    else {
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, false>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    }
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);

#if Debug
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  // gate boootstrapping (need pinned memory for concurrent scheduling)
  __host__ inline void GateBootstrapping(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n + 1> &poly, const typename P::T u, const uint32_t stream_id) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n + 1), sizeof(Lvl0_T) * (Lvl0_n + 1), 1, cudaMemcpyHostToDevice, stream[stream_id]);
    
    if constexpr (UNROLL) {
      double *KBKfft_i = (double *)((char *)KeyBundleKeyfft_d + stream_id * kbk_pitch);
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, double*, double*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, false>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, &XaiInFD_d, &KBKfft_i, &onetrgsw_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    }
    else {
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, false>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    }

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);

#if Debug
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  // gate boootstrapping (need pinned memory for concurrent scheduling) 80-bit
  __host__ inline void GateBootstrapping(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n80 + 1> &poly, const typename P::T u, const uint32_t stream_id) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n80 + 1), sizeof(Lvl0_T) * (Lvl0_n80 + 1), 1, cudaMemcpyHostToDevice, stream[stream_id]);

    double *KBKfft_i = (double *)((char *)KeyBundleKeyfft_d + stream_id * kbk_pitch);
    void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, double*, double*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG80<P, false>;
    void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, &XaiInFD_d, &KBKfft_i, &onetrgsw_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
    cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    
    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);

  }

  // programmable bootstrapping for homomorphic right shift in HE3DB
  __host__ inline void IdeGateBootstrappingTLWE2TLWEFFT_st(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n + 1> &poly, const uint32_t scale_bits, const uint32_t stream_id) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d + stream_id * in_pitch);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d + stream_id * out_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n + 1), sizeof(Lvl0_T) * (Lvl0_n + 1), 1, cudaMemcpyHostToDevice, stream[stream_id]);

    cufhedb::GateBootstrappingTLWE2TLWEFFT<P, true><<<(P::k + 1) * P::l, num_thread, 0, stream[stream_id]>>>(tlwe_i, uint_inout_i, tlwe0_i, BootstrappingKeyfft_d, scale_bits, buf_i, Ns2, SyncIn, SyncOut);
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);
#if Debug
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  __host__ inline void IdeGateBootstrapping_st(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n + 1> &poly, const typename P::T u, const uint32_t stream_id) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d + stream_id * tlwe0_pitch);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d + stream_id * tlwe_pitch);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d + stream_id * uint_pitch);
    double *buf_i = (double *)((char *)buf + stream_id * buf_pitch);

    cudaMemcpy2DAsync(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n + 1), sizeof(Lvl0_T) * (Lvl0_n + 1), 1, cudaMemcpyHostToDevice, stream[stream_id]);

    if constexpr (UNROLL) {
      double *KBKfft_i = (double *)((char *)KeyBundleKeyfft_d + stream_id * kbk_pitch);
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, double*, double*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, true>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, &XaiInFD_d, &KBKfft_i, &onetrgsw_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    }
    else {
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, true>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs, 0, stream[stream_id]);
    
    }
    cudaStreamSynchronize(stream[stream_id]);

    cudaMemcpy2DAsync(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost, stream[stream_id]);

#if Debug
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: Stream %d, %s\n", stream_id, cudaGetErrorString(err));
    }
#endif
  }

  /**
  The following functions are for performance test
   */
  __host__ inline void ifft_test(std::array<double, P::n> &res, std::array<typename P::T, P::n> &a, uint32_t test_num, double &cost) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpy(b_d, a.data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufhedb::ifft<<<1, num_thread>>>(ifftb_d, b_d, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res.data(), ifftb_d, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(b_d);

  }

  #if 1
  __host__ inline void tfhe_rs_ifft_test(std::array<double, P::n> &res, std::array<typename P::T, P::n> &a, uint32_t test_num, double &cost) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpy(b_d, a.data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufhedb::negacyclic_ifft<<<1, num_thread>>>(ifftb_d, b_d, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res.data(), ifftb_d, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(b_d);

  }
  #endif

  __host__ inline void fft_test(std::array<typename P::T, P::n> &res, std::array<double, P::n> &a, uint32_t test_num, double &cost) {
    typename P::T *b_d;
    cudaMalloc(&b_d, N * sizeof(typename P::T));
    cudaMemcpy(ifftb_d, a.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i=0;i<test_num;++i) {
      cufhedb::fft<<<1, num_thread>>>(b_d, ifftb_d, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res.data(), b_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
    cudaFree(b_d);

  }

  template<int32_t batch_size>
  __host__ inline void ifft_th_test(std::vector<std::array<double, P::n>> &res, std::vector<std::array<typename P::T, P::n>> &a, uint32_t test_num, double &cost) {
    typename P::T *b_d;
    double *ib_d;
    cudaMalloc(&b_d, batch_size * N * sizeof(typename P::T));
    cudaMalloc(&ib_d, batch_size * N * sizeof(double));
    for (int i = 0; i < batch_size; ++i)
      cudaMemcpyAsync(b_d + i * N, a[i].data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice, stream[0]);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i) {
      cufhedb::batch_ifft_th<batch_size><<<1, batch_size * num_thread, 0, stream[0]>>>(ib_d, b_d, Ns2);
      cudaStreamSynchronize(stream[0]);
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    for (int i = 0; i < batch_size; ++i)
      cudaMemcpyAsync(res[i].data(), ib_d + i * N, sizeof(double) * N, cudaMemcpyDeviceToHost, stream[0]);
    cudaFree(b_d);
    cudaFree(ib_d);
  }

  template<int32_t batch_size>
  __host__ inline void ifft_blk_test(std::vector<std::array<double, P::n>> &res, std::vector<std::array<typename P::T, P::n>> &a, uint32_t test_num, double &cost) {
    typename P::T *b_d;
    double *ib_d;
    cudaMalloc(&b_d, batch_size * N * sizeof(typename P::T));
    cudaMalloc(&ib_d, batch_size * N * sizeof(double));
    for (int i = 0; i < batch_size; ++i)
        cudaMemcpyAsync(b_d + i * N, a[i].data(), sizeof(typename P::T) * N, cudaMemcpyHostToDevice, stream[0]);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    for (int i = 0; i< test_num;++i) {
      cufhedb::batch_ifft_blk<<<batch_size, num_thread, 0, stream[0]>>>(ib_d, b_d, Ns2);
      cudaStreamSynchronize(stream[0]);
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    for (int i = 0; i < batch_size; ++i)
      cudaMemcpyAsync(res[i].data(), ib_d + i * N, sizeof(double) * N, cudaMemcpyDeviceToHost, stream[0]);
    cudaFree(b_d);
    cudaFree(ib_d);

  }

  __host__ inline void GateBootstrapping_test(std::array<typename P::T, P::n + 1> &tlwe, const std::array<Lvl0_T, Lvl0_n + 1> &poly, const typename P::T u, uint32_t test_num, double &cost) {

    Lvl0_T *tlwe0_i = (Lvl0_T *)((char *)tlwe0_d);
    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d);
    double *buf_i = (double *)((char *)buf);
    volatile int *SyncIn = (volatile int *)((char *)Syncin_d);
    volatile int *SyncOut = (volatile int *)((char *)Syncout_d);

    cudaMemcpy2D(tlwe0_i, tlwe0_pitch, poly.data(), sizeof(Lvl0_T) * (Lvl0_n + 1), sizeof(Lvl0_T) * (Lvl0_n + 1), 1, cudaMemcpyHostToDevice);

    if constexpr (UNROLL) {
      double *KBKfft_i = (double *)((char *)KeyBundleKeyfft_d + kbk_pitch);
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, double*, double*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, false>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, &XaiInFD_d, &KBKfft_i, &onetrgsw_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      std::chrono::system_clock::time_point start, end;
      start = std::chrono::system_clock::now();
      for (int i=0;i<test_num;++i) {
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs);
        cudaDeviceSynchronize();
      }
      end = std::chrono::system_clock::now();
      cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    else {
      void (*kernel_ptr)(typename P::T*, typename P::T*, Lvl0_T*, double*, typename P::T, double*, int32_t) = cufhedb::GateBootstrappingCG<P, false>;
      void *kernelArgs[] = {&tlwe_i, &uint_inout_i, &tlwe0_i, &BootstrappingKeyfft_d, const_cast<typename P::T*>(&u), &buf_i, const_cast<int32_t*>(&Ns2)};
      std::chrono::system_clock::time_point start, end;
      start = std::chrono::system_clock::now();
      for (int i=0;i<test_num;++i) {
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs);
        cudaDeviceSynchronize();
      }
      end = std::chrono::system_clock::now();
      cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

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
      cufhedb::ExternalProduct<P><<<(P::k + 1) * P::l, num_thread>>>(out_d, uint_inout_i, trgswfft_i, buf_i, Ns2, SyncIn, SyncOut);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res[0].data(), out_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(res[1].data(), out_d + N, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
  }

  __host__ inline void ExternalProduct_th_test(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, uint32_t test_num, double &cost) {
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
      cufhedb::ExternalProduct_th<P><<<1, (P::k + 1) * P::l * num_thread>>>(out_d, uint_inout_i, trgswfft_i, buf_i, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res[0].data(), out_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(res[1].data(), out_d + N, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
  }

  __host__ inline void ExternalProductCG_test(std::array<std::array<typename P::T, P::n>, 2> &res, std::array<std::array<typename P::T, P::n>, 2> &trlwe, const std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft, uint32_t test_num, double &cost) {
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

    // Check if the device supports cooperative launch
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        printf("Device does not support cooperative launch\n");
        return;
    }

    void (*kernel_ptr)(typename P::T*, typename P::T*, double*, double*, const int32_t) = cufhedb::ExternalProductCG<P>;
    void* kernelArgs[] = {&out_d, &uint_inout_i, &trgswfft_i, &buf_i, const_cast<int32_t*>(&Ns2)};

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    for (int i = 0; i < test_num; ++i) {
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr), (P::k + 1) * P::l, num_thread, kernelArgs);
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
      cufhedb::MulByTRGSWFFT<P><<<(P::k + 1) * P::l, num_thread>>>(uint_inout_i, trgswfft_i, buf_i, Ns2);
      cudaDeviceSynchronize();
      cufhedb::Reduction<P><<<(P::k + 1), num_thread>>>(out_d, buf_i, Ns2);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy(res[0].data(), out_d, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(res[1].data(), out_d + N, sizeof(typename P::T) * N, cudaMemcpyDeviceToHost);
  }

  __host__ inline void GateBootstrapping_withoutFusion_test(std::array<typename P::T, P::n + 1> &tlwe, std::array<Lvl0_T, Lvl0_n + 1> &poly, const typename P::T u, uint32_t test_num, double &cost) {

    typename P::T *tlwe_i = (typename P::T *)((char *)tlwe_d);
    typename P::T *uint_inout_i = (typename P::T *)((char *)uint_inout_d);
    double *buf_i = (double *)((char *)buf);

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    constexpr uint32_t roundoffset = 16;
    const uint32_t b = 2048 - (poly[Lvl0_n] >> 5);
    for (int i=0;i<test_num;++i) {
      cufhedb::PolynomialMulByXai<P><<<1, 128>>>(uint_inout_d, u, b);
      cudaDeviceSynchronize();
      #pragma unroll
      for (int i = 0; i < Lvl0_n; ++i) {

        const uint32_t a = (poly[i] + roundoffset) >> 5;
        if (a == 0)
          continue;
        double *trgswfft_i = &BootstrappingKeyfft_d[i * (P::k + 1) * P::l * _2N];
        cufhedb::MulByTRGSWFFT<P><<<(P::k + 1) * P::l, num_thread>>>(uint_inout_i, trgswfft_i, buf_i, Ns2);
        cudaDeviceSynchronize();
        cufhedb::Reduction<P><<<(P::k + 1), num_thread>>>(uint_inout_i, buf_i, Ns2);
        cudaDeviceSynchronize();
      }
      cufhedb::SampleExtractIndex_Kernel<<<1, 128>>>(tlwe_i, uint_inout_i, N);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaMemcpy2D(tlwe.data(), tlwe_pitch, tlwe_i, sizeof(typename P::T) * (P::n + 1), sizeof(typename P::T) * (P::n + 1), 1, cudaMemcpyDeviceToHost);
  }
  
  __host__ inline ~cuCoreFHE() {
    if (is_cleaned)
      return;

    // free
    if (uint_inout_d)
      cudaFree(uint_inout_d);
    if (BootstrappingKeyfft_d)
      cudaFree(BootstrappingKeyfft_d);
    if (ifftb_d)
      cudaFree(ifftb_d);
    if (tlwe0_d)
      cudaFree(tlwe0_d);
    if (tlwe_d)
      cudaFree(tlwe_d);
    if (buf)
      cudaFree(buf);
    if (trgswfft_d)
      cudaFree(trgswfft_d);
    if (Syncin_d)
      cudaFree((void *)Syncin_d);
    if (Syncout_d)
      cudaFree((void *)Syncout_d);

    if constexpr (UNROLL) {
      if (onetrgsw_d)
        cudaFree(onetrgsw_d);
      if (XaiInFD_d)
        cudaFree(XaiInFD_d);
      if (KeyBundleKeyfft_d)
        cudaFree(KeyBundleKeyfft_d);
    }

    // destroy cuda stream
    for (int i = 0; i < num_stream; ++i) {
      if (stream[i]) {
        cudaStreamDestroy(stream[i]);
      }
    }

    is_cleaned = true;
  }
};

}; // namespace cufhedb