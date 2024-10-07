#pragma once

#include "ARCEDB/utils/types.h"
#include "fft_gpu/cufft_gpu.cuh"
#include <limits>
#include <cstring>

using namespace arcedb;

int getSMCount() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

static int num_SMs = getSMCount();

int num_stream1 = 2 * (num_SMs / 6);
int num_stream2 = 2 * (num_SMs / 8);

int result = setenv("CUDA_DEVICE_MAX_CONNECTIONS", std::to_string(num_stream1).c_str(), 1);

cufft::CuFFT_Torus<Lvl1> cufftplvl1(num_stream1);
cufft::CuFFT_Torus<Lvl2> cufftplvl2(num_stream2);

namespace cuARCEDB {

// Lvl1   

    void HomAND(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb, const TFHEEvalKey &ek, uint32_t stream_id)
    {
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            res[i] = ca[i] + cb[i];
        Lvl1::T offset = -Lvl1::μ;
        res[Lvl1::k * Lvl1::n] += offset;
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
        cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, Lvl1::μ, stream_id);
    }

    void HomOR(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb, const TFHEEvalKey &ek, uint32_t stream_id)
    {
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            res[i] = ca[i] + cb[i];
        Lvl1::T offset = Lvl1::μ;
        res[Lvl1::k * Lvl1::n] += offset;
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
        cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, Lvl1::μ, stream_id);
    }

	void greater_than_tfhepp(TRLWELvl1 &cipher1, TRGSWLvl1 &cipher2, TLWELvl1 &res, TFHESecretKey &sk, uint32_t stream_id)
    {
        TRLWELvl1 trlwelvl1, trlwe_mul;
        TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, cipher1, cipher2);
        //cufftplvl1.ExternalProduct_st<Lvl1>(trlwelvl1, cipher1, cipher2, stream_id);
        cufftplvl1.PolyMul_st(trlwe_mul, trlwelvl1, stream_id);
        TFHEpp::SampleExtractIndex<Lvl1>(res, trlwe_mul, 0);
        for (size_t i = 0; i <= Lvl1::n; i++)
        {
            res[i] = -res[i];
        }
    }

    void greater_than_tfhepp(std::vector<TRLWELvl1> &ciphers1, std::vector<TRGSWLvl1> &ciphers2, size_t cipher_size, TLWELvl1 &res, 
                            TFHEEvalKey &ek, TFHESecretKey &sk, uint32_t stream_id)
    {
        if (cipher_size == 1)
        {
            greater_than_tfhepp(ciphers1[0], ciphers2[0], res, sk, stream_id);
        }
        else
        {
            TLWELvl1 low_res, high_res, equal_res;
            TRLWELvl1 trlwelvl1;
            greater_than_tfhepp(ciphers1, ciphers2, cipher_size - 1, low_res, ek, sk, stream_id);
            TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, ciphers1[cipher_size-1], ciphers2[cipher_size-1]);
            //cufftplvl1.ExternalProduct_st<Lvl1>(trlwelvl1, ciphers1[cipher_size-1], ciphers2[cipher_size-1], stream_id);
            TFHEpp::SampleExtractIndex<Lvl1>(equal_res, trlwelvl1, 0);
            greater_than_tfhepp(ciphers1[cipher_size-1], ciphers2[cipher_size-1], high_res, sk, stream_id);
            for (size_t i = 0; i <= Lvl1::n; i++)
            {
                high_res[i] = high_res[i] + high_res[i];
            }

            TLWELvl1 tlwelvl1;
            uint32_t offset = Lvl1::μ >> 1;
            for (size_t i = 0; i <= Lvl1::k * Lvl1::n; i++)
            {
                tlwelvl1[i] = equal_res[i] + high_res[i] + low_res[i];
            }
            tlwelvl1[Lvl1::n] += offset;
            TLWELvl0 tlwelvl0;
            TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlwelvl1, *ek.iksklvl10);
            cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, Lvl1::μ, stream_id);
        }
    }

    void equality_tfhepp(TRLWELvl1 &cipher1, TRGSWLvl1 &cipher2, TLWELvl1 &res, TFHESecretKey &sk, uint32_t stream_id)
    {
        TRLWELvl1 trlwe_mul;
        TFHEpp::trgswfftExternalProduct<Lvl1>(trlwe_mul, cipher1, cipher2);
        //cufftplvl1.ExternalProduct_st(trlwe_mul, cipher1, cipher2, stream_id);
        TFHEpp::SampleExtractIndex<Lvl1>(res, trlwe_mul, 0);
        for (size_t i = 0; i <= Lvl1::n; i++)
        {
            res[i] = 2 * res[i];
        }
        res[Lvl1::n] -= Lvl1::μ;
    }

    void equality_tfhepp(std::vector<TRLWELvl1> &ciphers1, std::vector<TRGSWLvl1> &ciphers2, size_t cipher_size, TLWELvl1 &res, 
                            TFHEEvalKey &ek, TFHESecretKey &sk, uint32_t stream_id)
    {
        if (cipher_size == 1)
        {
            equality_tfhepp(ciphers1[0], ciphers2[0], res, sk, stream_id);
        }
        else
        {
            TLWELvl1 low_res, high_res, equal_res;
            TRLWELvl1 trlwelvl1;
            equality_tfhepp(ciphers1, ciphers2, cipher_size - 1, low_res, ek, sk, stream_id);
            equality_tfhepp(ciphers1[cipher_size-1], ciphers2[cipher_size-1], high_res, sk, stream_id);
            HomAND(res, low_res, high_res, ek, stream_id);
        }
    }

    void less_than_tfhepp(TRLWELvl1 &cipher1, TRGSWLvl1 &cipher2, TLWELvl1 &res, TFHESecretKey &sk, uint32_t stream_id)
    {
        TRLWELvl1 trlwelvl1, trlwe_mul;
        TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, cipher1, cipher2);
        cufftplvl1.PolyMul_st(trlwe_mul, trlwelvl1, stream_id);
        TFHEpp::SampleExtractIndex<Lvl1>(res, trlwe_mul, 0);
    }

    void less_than_tfhepp(std::vector<TRLWELvl1> &ciphers1, std::vector<TRGSWLvl1> &ciphers2, size_t cipher_size, TLWELvl1 &res, 
                            TFHEEvalKey &ek, TFHESecretKey &sk, uint32_t stream_id)
    {
        if (cipher_size == 1)
        {
          less_than_tfhepp(ciphers1[0], ciphers2[0], res, sk, stream_id);
        }
        else
        {
            TLWELvl1 low_res, high_res, equal_res;
            TRLWELvl1 trlwelvl1;
            less_than_tfhepp(ciphers1, ciphers2, cipher_size - 1, low_res, ek, sk, stream_id);
            TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, ciphers1[cipher_size-1], ciphers2[cipher_size-1]);
            TFHEpp::SampleExtractIndex<Lvl1>(equal_res, trlwelvl1, 0);
            less_than_tfhepp(ciphers1[cipher_size-1], ciphers2[cipher_size-1], high_res, sk, stream_id);
            for (size_t i = 0; i <= Lvl1::n; i++)
            {
                high_res[i] = high_res[i] + high_res[i];
            }

            TLWELvl1 tlwelvl1;
            uint32_t offset = Lvl1::μ >> 1;
            for (size_t i = 0; i <= Lvl1::k * Lvl1::n; i++)
            {
                tlwelvl1[i] = equal_res[i] + high_res[i] + low_res[i];
            }
            tlwelvl1[Lvl1::n] += offset;
            TLWELvl0 tlwelvl0;
            TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, tlwelvl1, *ek.iksklvl10);
            cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, Lvl1::μ, stream_id);
        }
    }

    void lift_and_and(TLWELvl1 &cipher1, TLWELvl1 &cipher2, TLWELvl1 &res, uint32_t scale_bits, TFHEpp::EvalKey &ek, TFHEpp::SecretKey &sk, uint32_t stream_id)
    {
        using namespace TFHEpp;
        TLWELvl1 temp;
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            temp[i] = cipher1[i] + cipher2[i];
        temp[Lvl1::k * Lvl1::n] -= Lvl1::μ;
        Lvl1::T c = (1ULL << (scale_bits-1));
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, temp, ek.getiksk<Lvl10>());
        cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, c, stream_id);
        res[Lvl1::k * Lvl1::n] += c;
    }

    void lift_and_and(TLWELvl1 &cipher1, TLWELvl1 &cipher2, TLWELvl2 &res, uint32_t scale_bits, TFHEpp::EvalKey &ek, TFHEpp::SecretKey &sk, uint32_t stream_id)
    {
        using namespace TFHEpp;
        TLWELvl1 temp;
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            temp[i] = cipher1[i] + cipher2[i];
        temp[Lvl1::k * Lvl1::n] -= Lvl1::μ;
        Lvl2::T c = (1ULL << (scale_bits-1));
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, temp, ek.getiksk<Lvl10>());
        cufftplvl2.GateBootstrappingTLWE2TLWEFFT_st(res, tlwelvl0, c, stream_id);
        res[Lvl2::k * Lvl2::n] += c;
    }

};