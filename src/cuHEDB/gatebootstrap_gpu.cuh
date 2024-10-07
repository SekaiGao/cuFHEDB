#pragma once

#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/utils/types.h"
#include "fft_gpu/cufft_gpu.cuh"
#include <limits>
#include <cstring>

using namespace HEDB;

int getSMCount() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

static int num_SMs = getSMCount();

int num_stream1 = 2 * (num_SMs / 6);
int num_stream2 = 2 * (num_SMs / 8);

int result = setenv("CUDA_DEVICE_MAX_CONNECTIONS", std::to_string(num_stream1).c_str(), 1);

cufft::CuFFT_Torus<Lvl1> cufftlvl1(num_stream1);
cufft::CuFFT_Torus<Lvl2> cufftlvl2(num_stream2);

namespace cuHEDB {

// Lvl1   

void MSBGateBootstrapping(TFHEpp::TLWE<Lvl1> &res,
                                 const TFHEpp::TLWE<Lvl1> &tlwe,
                                 const TFHEEvalKey &ek, bool result_type,
                                 uint32_t stream_id) {
  Lvl1::T u = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    u = u << 1;
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 6);
  TFHEpp::TLWE<Lvl1> tlweoffset = tlwe;
  tlweoffset[Lvl1::k * Lvl1::n] += offset;
  TFHEpp::TLWE<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlweoffset, *ek.iksklvl10);
  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, -u, stream_id);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += u;
}

void MSBGateBootstrapping(TFHEpp::TLWE<Lvl1> &res,
                          const TFHEpp::TLWE<Lvl2> &tlwe, const TFHEEvalKey &ek,
                          bool result_type, uint32_t stream_id) {
  Lvl1::T u = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    u = u << 1;
  constexpr uint64_t offset =
      1ULL << (std::numeric_limits<Lvl2::T>::digits - 6);
  TFHEpp::TLWE<Lvl2> tlweoffset = tlwe;
  tlweoffset[Lvl2::k * Lvl2::n] += offset;
  TFHEpp::TLWE<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl20>(tlwelvl0, tlweoffset, *ek.iksklvl20);
  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, -u, stream_id);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += u;
}

void MSBGateBootstrapping(TFHEpp::TLWE<Lvl2> &res,
                          const TFHEpp::TLWE<Lvl2> &tlwe, const TFHEEvalKey &ek,
                          bool result_type, uint32_t stream_id) {
  Lvl2::T u = Lvl2::μ;
  if (IS_ARITHMETIC(result_type))
    u = u << 1;
  constexpr uint64_t offset =
      1ULL << (std::numeric_limits<Lvl2::T>::digits - 7);
  TFHEpp::TLWE<Lvl2> tlweoffset = tlwe;
  tlweoffset[Lvl2::k * Lvl2::n] += offset;
  TFHEpp::TLWE<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl20>(tlwelvl0, tlweoffset, *ek.iksklvl20);
  cufftlvl2.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, -u, stream_id);
  if (IS_ARITHMETIC(result_type))
    res[Lvl2::k * Lvl2::n] += u;
}

void IdeGateBootstrapping(TFHEpp::TLWE<Lvl1> &res,
                                 const TFHEpp::TLWE<Lvl1> &tlwe,
                                 uint32_t scale_bits, const TFHEEvalKey &ek,
                                 uint32_t stream_id) {
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 6);
  TFHEpp::TLWE<Lvl1> tlweoffset = tlwe;
  tlweoffset[Lvl1::k * Lvl1::n] += offset;
  TFHEpp::TLWE<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlweoffset, *ek.iksklvl10);
  cufftlvl1.IdeGateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, scale_bits, stream_id);
}

void IdeGateBootstrapping(TFHEpp::TLWE<Lvl1> &res,
                          const TFHEpp::TLWE<Lvl2> &tlwe, uint32_t scale_bits,
                          const TFHEEvalKey &ek, uint32_t stream_id) {
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl2::T>::digits - 6);
  TFHEpp::TLWE<Lvl2> tlweoffset = tlwe;
  tlweoffset[Lvl2::k * Lvl2::n] += offset;
  TFHEpp::TLWE<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl20>(tlwelvl0, tlweoffset, *ek.iksklvl20);
  cufftlvl1.IdeGateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, scale_bits - 32, stream_id);
}

void IdeGateBootstrapping(TFHEpp::TLWE<Lvl2> &res,
                          const TFHEpp::TLWE<Lvl2> &tlwe, uint32_t scale_bits,
                          const TFHEEvalKey &ek, uint32_t stream_id) {
  constexpr uint64_t offset =  1ULL << (std::numeric_limits<Lvl2::T>::digits - 7);
  TFHEpp::TLWE<Lvl2> tlweoffset = tlwe;
  tlweoffset[Lvl2::k * Lvl2::n] += offset;
  constexpr uint32_t plain_bits = 5;
  TFHEpp::TLWE<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl20>(tlwelvl0, tlweoffset, *ek.iksklvl20);
  cufftlvl2.IdeGateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, scale_bits, stream_id);
}

template <typename P>
inline void HomNOT(TFHEpp::TLWE<P> &res, const TFHEpp::TLWE<P> &tlwe) {
  for (int i = 0; i <= P::k * P::n; i++)
    res[i] = -tlwe[i];
}

// c0 + c1 - 1/8
void HomAND(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb,
            const TFHEEvalKey &ek, bool result_type, uint32_t stream_id) {
  Lvl1::T offset = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    offset = (offset << 1);
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = ca[i] + cb[i];
  res[Lvl1::k * Lvl1::n] -= Lvl1::μ >> 1; // - 1/8
  TLWELvl0 tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, offset, stream_id);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += offset;
}

// c0 + c1 + 1/8
void HomOR(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb,
           const TFHEEvalKey &ek, bool result_type, uint32_t stream_id) {
  Lvl1::T offset = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    offset = (offset << 1);
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = ca[i] + cb[i];
  res[Lvl1::k * Lvl1::n] += (Lvl1::μ >> 1); // + 1/8
  TLWELvl0 tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, offset, stream_id);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += offset;
}

void LOG_to_ARI(TLWELvl1 &res, const TLWELvl1 &tlwe, const TFHEEvalKey &ek, uint32_t stream_id) {
  Lvl1::T μ = Lvl1::μ;
  μ = μ << 1;
  TLWELvl0 tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlwe, *ek.iksklvl10);
  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, -μ, stream_id);
  res[Lvl1::k * Lvl1::n] += Lvl1::μ;
}

void log_rescale(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t scale_bits, const TFHEEvalKey &ek, uint32_t stream_id) {
    Lvl1::T μ = 1ULL << (scale_bits - 1);
    TLWELvl0 tlwelvl0;
    TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlwe, *ek.iksklvl10);
    cufftlvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, -μ, stream_id);
    res[Lvl1::k * Lvl1::n] += μ;
}

};