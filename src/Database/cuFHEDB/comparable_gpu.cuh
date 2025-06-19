#pragma once

#include "ARCEDB/utils/types.h"
#include "FHE/coreFHE.cuh"
#include <limits>
#include <cstring>

/**
 * comparison method from ArcEDB
*/

// Enable bootstrapping unroll
#define UNROLL true

using namespace arcedb;

static int num_SMs = getSMCount();

int num_stream1 = num_SMs; // 2 * (num_SMs / 6);
int num_stream2 = num_SMs;     // 2 * (num_SMs / 8);

int result = setenv("CUDA_DEVICE_MAX_CONNECTIONS", std::to_string(num_stream1).c_str(), 1);

cufhedb::cuCoreFHE<Lvl1, UNROLL> cufftplvl1(num_stream1);
cufhedb::cuCoreFHE<Lvl2> cufftplvl2(num_stream2);

namespace cuFHEDB {

    // Logic gates

    /**
    * @brief Performs a homomorphic AND operation on two ciphertexts.
    * 
    * @param res The result ciphertext after the AND operation.
    * @param ca The first input ciphertext.
    * @param cb The second input ciphertext.
    * @param ek The evaluation key used for bootstrapping.
    * @param stream_id The CUDA stream identifier.
    */
    inline void HomAND(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb, const TFHEEvalKey &ek, uint32_t stream_id)
    {
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            res[i] = ca[i] + cb[i];
        const Lvl1::T offset = -Lvl1::μ;
        res[Lvl1::k * Lvl1::n] += offset;
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
        cufftplvl1.GateBootstrapping_st(res, tlwelvl0, Lvl1::μ, stream_id);
    }

    inline void HomOR(TLWELvl1 &res, const TLWELvl1 &ca, const TLWELvl1 &cb, const TFHEEvalKey &ek, uint32_t stream_id)
    {
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            res[i] = ca[i] + cb[i];
        Lvl1::T offset = Lvl1::μ;
        res[Lvl1::k * Lvl1::n] += offset;
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, res, *ek.iksklvl10);
        cufftplvl1.GateBootstrapping_st(res, tlwelvl0, Lvl1::μ, stream_id);
    }

    inline void HomNOT(TLWELvl1 &res, const TLWELvl1 &tlwe) {
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            res[i] = -tlwe[i];
    }

    // Homomorphic comparison

    /**
    * @brief Performs a homomorphic comparison (greater than) between two ciphertexts.
    * 
    * @param cipher1 The first RLWE ciphertext.
    * @param cipher2 The second RGSW ciphertext.
    * @param res The result LWE ciphertext after comparison.
    * @param stream_id The CUDA stream identifier.
    */
	void greater_than(TRLWELvl1 &cipher1, TRGSWLvl1 &cipher2, TLWELvl1 &res, uint32_t stream_id)
    {
        TRLWELvl1 trlwelvl1, trlwe_mul;
        TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, cipher1, cipher2);
        // cufftplvl1.PolyMul_st(trlwe_mul, trlwelvl1, stream_id);
        TFHEpp::Polynomial<Lvl1> test_plaintext;
        for (size_t i = 0; i < Lvl1::n; i++) {
            test_plaintext[i] = 1;
        }

        TFHEpp::PolyMul<Lvl1>(trlwe_mul[0], trlwelvl1[0], test_plaintext);
        TFHEpp::PolyMul<Lvl1>(trlwe_mul[1], trlwelvl1[1], test_plaintext);
        TFHEpp::SampleExtractIndex<Lvl1>(res, trlwe_mul, 0);
        for (size_t i = 0; i <= Lvl1::n; i++)
        {
            res[i] = -res[i];
        }
    }

    /**
    * @brief Performs a homomorphic comparison (greater than) recursively.
    * 
    * @param ciphers1 The first group of ciphertexts.
    * @param ciphers2 The second group of ciphertexts.
    * @param cipher_size The number of ciphertexts.
    * @param res The result LWE ciphertext after comparison.
    * @param ek The evaluation key used for bootstrapping.
    * @param stream_id The CUDA stream identifier.
    */
    void greater_than(std::vector<TRLWELvl1> &ciphers1, std::vector<TRGSWLvl1> &ciphers2, size_t cipher_size, TLWELvl1 &res, 
                            TFHEEvalKey &ek, uint32_t stream_id)
    {
        if (cipher_size == 1)
        {
            greater_than(ciphers1[0], ciphers2[0], res, stream_id);
        }
        else
        {
            TLWELvl1 low_res, high_res, equal_res;
            TRLWELvl1 trlwelvl1;
            greater_than(ciphers1, ciphers2, cipher_size - 1, low_res, ek, stream_id);
            TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, ciphers1[cipher_size-1], ciphers2[cipher_size-1]);
            //cufftplvl1.ExternalProduct_st<Lvl1>(trlwelvl1, ciphers1[cipher_size-1], ciphers2[cipher_size-1], stream_id);
            TFHEpp::SampleExtractIndex<Lvl1>(equal_res, trlwelvl1, 0);
            greater_than(ciphers1[cipher_size-1], ciphers2[cipher_size-1], high_res, stream_id);
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
            cufftplvl1.GateBootstrapping_st(res, tlwelvl0, Lvl1::μ, stream_id);
        }
    }

    void equality(TRLWELvl1 &cipher1, TRGSWLvl1 &cipher2, TLWELvl1 &res, uint32_t stream_id)
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

    void equality(std::vector<TRLWELvl1> &ciphers1, std::vector<TRGSWLvl1> &ciphers2, size_t cipher_size, TLWELvl1 &res, 
                            TFHEEvalKey &ek, uint32_t stream_id)
    {
        if (cipher_size == 1)
        {
            equality(ciphers1[0], ciphers2[0], res, stream_id);
        }
        else
        {
            TLWELvl1 low_res, high_res, equal_res;
            TRLWELvl1 trlwelvl1;
            equality(ciphers1, ciphers2, cipher_size - 1, low_res, ek, stream_id);
            equality(ciphers1[cipher_size-1], ciphers2[cipher_size-1], high_res, stream_id);
            HomAND(res, low_res, high_res, ek, stream_id);
        }
    }

    void less_than(TRLWELvl1 &cipher1, TRGSWLvl1 &cipher2, TLWELvl1 &res, uint32_t stream_id)
    {
        TRLWELvl1 trlwelvl1, trlwe_mul;
        TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, cipher1, cipher2);
        //cufftplvl1.PolyMul_st(trlwe_mul, trlwelvl1, stream_id);
        TFHEpp::Polynomial<Lvl1> test_plaintext;
        test_plaintext[0] = Lvl1::plain_modulus - 1;
        for (size_t i = 1; i < Lvl1::n; i++)
        {
            test_plaintext[i] = 1;
        }

        TFHEpp::PolyMul<Lvl1>(trlwe_mul[0], trlwelvl1[0], test_plaintext);
        TFHEpp::PolyMul<Lvl1>(trlwe_mul[1], trlwelvl1[1], test_plaintext);
        
        TFHEpp::SampleExtractIndex<Lvl1>(res, trlwe_mul, 0);
    }

    void less_than(std::vector<TRLWELvl1> &ciphers1, std::vector<TRGSWLvl1> &ciphers2, size_t cipher_size, TLWELvl1 &res, 
                            TFHEEvalKey &ek, uint32_t stream_id)
    {
        if (cipher_size == 1)
        {
            less_than(ciphers1[0], ciphers2[0], res, stream_id);
        }
        else
        {
            TLWELvl1 low_res, high_res, equal_res;
            TRLWELvl1 trlwelvl1;
            less_than(ciphers1, ciphers2, cipher_size - 1, low_res, ek, stream_id);
            TFHEpp::trgswfftExternalProduct<Lvl1>(trlwelvl1, ciphers1[cipher_size-1], ciphers2[cipher_size-1]);
            TFHEpp::SampleExtractIndex<Lvl1>(equal_res, trlwelvl1, 0);
            less_than(ciphers1[cipher_size-1], ciphers2[cipher_size-1], high_res, stream_id);
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
            cufftplvl1.GateBootstrapping_st(res, tlwelvl0, Lvl1::μ, stream_id);
        }
    }

    // lift the ciphertext to certain scale
    void lift_and_and(TLWELvl1 &cipher1, TLWELvl1 &cipher2, TLWELvl1 &res, uint32_t scale_bits, TFHEpp::EvalKey &ek, uint32_t stream_id)
    {
        TLWELvl1 temp;
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            temp[i] = cipher1[i] + cipher2[i];
        temp[Lvl1::k * Lvl1::n] -= Lvl1::μ;
        Lvl1::T c = (1ULL << (scale_bits-1));
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, temp, ek.getiksk<Lvl10>());
        cufftplvl1.GateBootstrapping_st(res, tlwelvl0, c, stream_id);
        res[Lvl1::k * Lvl1::n] += c;
    }

    void lift_and_and(TLWELvl1 &cipher1, TLWELvl1 &cipher2, TLWELvl2 &res, uint32_t scale_bits, TFHEpp::EvalKey &ek, uint32_t stream_id)
    {
        TLWELvl1 temp;
        for (int i = 0; i <= Lvl1::k * Lvl1::n; i++)
            temp[i] = cipher1[i] + cipher2[i];
        temp[Lvl1::k * Lvl1::n] -= Lvl1::μ;
        Lvl2::T c = (1ULL << (scale_bits-1));
        TLWELvl0 tlwelvl0;
        TFHEpp::IdentityKeySwitch<Lvl10>(tlwelvl0, temp, ek.getiksk<Lvl10>());
        cufftplvl2.GateBootstrapping_st(res, tlwelvl0, c, stream_id);
        res[Lvl2::k * Lvl2::n] += c;
    }

};