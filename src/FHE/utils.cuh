#pragma once

#include <params.hpp>

namespace cufhedb {

template <class P>
using bkP_T = typename std::conditional<(std::is_same_v<typename P::T, uint32_t>), TFHEpp::lvl01param, TFHEpp::lvl02param>::type;

// generate I=TRGSW(1)
template <class P>
constexpr TFHEpp::TRGSWFFT<P> oneTRGSWFFTgen()
{
    constexpr std::array<typename P::T, P::l> h = TFHEpp::hgen<P>();
    TFHEpp::TRGSW<P> trgsw;
    for (TFHEpp::TRLWE<P> &trlwe : trgsw)
      trlwe = {};
    for (int i = 0; i < P::l; i++) {
        for (int k = 0; k < P::k + 1; k++) {
            trgsw[i + k * P::l][k][0] = static_cast<typename P::T>(h[i]);
        }
    }
    return TFHEpp::ApplyFFT2trgsw<P>(trgsw);
}

};

int getSMCount() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

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