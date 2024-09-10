#pragma once

#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/utils/types.h"
#include "cufft.cuh"
#include <limits>

using namespace HEDB;

CuFFT_TRLWE cufftplvl1(Lvl1::n);


namespace cuHEDB {

// Lvl1   

void MSBGateBootstrapping_kernel(TLWE_test<Lvl1> &res,
                                 const TLWE_test<Lvl1> &tlwe,
                                 const TFHEEvalKey &ek, bool result_type,
                                 uint32_t idx) {
  Lvl1::T u = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    u = u << 1;
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 6);
  TLWE_test<Lvl1> tlweoffset = tlwe;
  tlweoffset[Lvl1::k * Lvl1::n] += offset;
  TLWE_test<Lvl0> tlwelvl0;

  // t1
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlweoffset, *ek.iksklvl10);
  //t2
  cufftplvl1.MSBGateBootstrappingTLWE2TLWEFFT_kernels<Lvl1>(res, tlwelvl0, u, idx);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += u;
}

void IdeGateBootstrapping_kernel(TLWE_test<Lvl1> &res,
                                 const TLWE_test<Lvl1> &tlwe,
                                 uint32_t scale_bits, const TFHEEvalKey &ek,
                                 uint32_t idx) {
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 6);
  TLWE_test<Lvl1> tlweoffset = tlwe;
  tlweoffset[Lvl1::k * Lvl1::n] += offset;
  // constexpr uint32_t plain_bits = 4;
  TLWE_test<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlweoffset, *ek.iksklvl10);
  cufftplvl1.IdeGateBootstrappingTLWE2TLWEFFT_kernels<Lvl1>(res, tlwelvl0, scale_bits, idx);
}

template <typename P>
inline void HomNOT(TLWE_test<P> &res, const TLWE_test<P> &tlwe) {
  for (int i = 0; i <= P::k * P::n; i++)
    res[i] = -tlwe[i];
}

// c0 + c1 - 1/8
void HomAND(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb,
            const TFHEEvalKey &ek, bool result_type, uint32_t idx) {
  Lvl1::T offset = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    offset = (offset << 1);
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = ca[i] + cb[i];
  res[Lvl1::k * Lvl1::n] -= Lvl1::μ >> 1; // - 1/8
  TLWELvl0 tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
  cufftplvl1.MSBGateBootstrappingTLWE2TLWEFFT_kernels<Lvl1>(res, tlwelvl0, -offset, idx);
  // TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl01>(res, tlwelvl0, *ek.bkfftlvl01,
  // TFHEpp::μ_polygen<Lvl1>(-offset));
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += offset;
}

// c0 + c1 + 1/8
void HomOR(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb,
           const TFHEEvalKey &ek, bool result_type, uint32_t idx) {
  Lvl1::T offset = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    offset = (offset << 1);
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = ca[i] + cb[i];
  res[Lvl1::k * Lvl1::n] += (Lvl1::μ >> 1); // + 1/8
  TLWELvl0 tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
  cufftplvl1.MSBGateBootstrappingTLWE2TLWEFFT_kernels<Lvl1>(res, tlwelvl0, -offset, idx);
  // TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl01>(res, tlwelvl0, *ek.bkfftlvl01,
  // TFHEpp::μ_polygen<Lvl1>(-offset));
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += offset;
}


void MSBGateBootstrapping_kernel(Lvl1::T *res,
                                 const Lvl1::T *tlwe,
                                 const TFHEEvalKey &ek, bool result_type,
                                 uint32_t idx, const cudaStream_t st) {
  Lvl1::T u = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    u = u << 1;
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 6);
  TLWE_test<Lvl1> tlweoffset;
  std::copy(tlwe, tlwe + 1025, tlweoffset.begin());
  tlweoffset[Lvl1::k * Lvl1::n] += offset;
  TLWE_test<Lvl0> tlwelvl0;

  // t1
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlweoffset, *ek.iksklvl10);
  //t2
  cufftplvl1.MSBGateBootstrappingTLWE2TLWEFFT_st<Lvl1>(res, tlwelvl0, u, idx, st);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += u;
}

void IdeGateBootstrapping_kernel(Lvl1::T *res, const Lvl1::T *tlwe,
                                 uint32_t scale_bits, const TFHEEvalKey &ek,
                                 uint32_t idx, const cudaStream_t st) {
  constexpr uint64_t offset = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 6);
  TLWE_test<Lvl1> tlweoffset;
  std::copy(tlwe, tlwe + 1025, tlweoffset.begin());
  tlweoffset[Lvl1::k * Lvl1::n] += offset;
  // constexpr uint32_t plain_bits = 4;
  TLWE_test<Lvl0> tlwelvl0;
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlweoffset, *ek.iksklvl10);
  cufftplvl1.IdeGateBootstrappingTLWE2TLWEFFT_st<Lvl1>(res, tlwelvl0, scale_bits, idx, st);
}

// c0 + c1 - 1/8
void HomAND(Lvl1::T *res, const Lvl1::T *ca, const Lvl1::T *cb,
            const TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st) {
  Lvl1::T offset = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    offset = (offset << 1);
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = ca[i] + cb[i];
  res[Lvl1::k * Lvl1::n] -= Lvl1::μ >> 1; // - 1/8
  TLWELvl0 tlwelvl0;
  TLWELvl1 res0;
  std::copy(res, res + 1025, res0.begin());
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res0, *ek.iksklvl10);
  cufftplvl1.MSBGateBootstrappingTLWE2TLWEFFT_st<Lvl1>(res, tlwelvl0, -offset, idx, st);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += offset;
}

// c0 + c1 + 1/8
void HomOR(Lvl1::T *res, const Lvl1::T *ca, const Lvl1::T *cb,
           const TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st) {
  Lvl1::T offset = Lvl1::μ;
  if (IS_ARITHMETIC(result_type))
    offset = (offset << 1);
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = ca[i] + cb[i];
  res[Lvl1::k * Lvl1::n] += (Lvl1::μ >> 1); // + 1/8
  TLWELvl0 tlwelvl0;
  TLWELvl1 res0;
  std::copy(res, res + 1025, res0.begin());
  TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res0, *ek.iksklvl10);
  cufftplvl1.MSBGateBootstrappingTLWE2TLWEFFT_st<Lvl1>(res, tlwelvl0, -offset, idx, st);
  if (IS_ARITHMETIC(result_type))
    res[Lvl1::k * Lvl1::n] += offset;
}

inline void HomNOT(Lvl1::T *res, const Lvl1::T *tlwe) {
  for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
    res[i] = -tlwe[i];
}
};