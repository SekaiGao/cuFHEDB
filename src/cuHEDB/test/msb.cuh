#pragma once

#include "gatebootstrap.cuh"

namespace cuHEDB
{
    // 结果, 密文, 评估密钥, 结果类型(Logic, Arithmetic)
    void ExtractMSB5(TLWELvl1 &res, const TLWELvl1 &tlwe, const TFHEEvalKey &ek, bool result_type, uint32_t idx = 0) {
        MSBGateBootstrapping_kernel(res, tlwe, ek, result_type, idx);
    }
    //测试各部分耗时
    // plain_bits: 明文bits数(6-9)
    void ExtractMSB9(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t idx = 0) {
        TLWELvl1 shift_tlwe, sign_tlwe5;
        // 32 - plain_bits
        uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - plain_bits;
        // ct1 = ct<<(plain_bits-5)
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = tlwe[i] << (plain_bits - 5);
        }
        // sign_tlwe5 = HMSB(ct1)
        MSBGateBootstrapping_kernel(sign_tlwe5, shift_tlwe, ek, ARITHMETIC, idx);
        // ct3 = ct1 - sign_tlwe5 
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = shift_tlwe[i] - sign_tlwe5[i];
        }
        // ct4 = ct3>>(plain_bits-5)
        IdeGateBootstrapping_kernel(shift_tlwe, shift_tlwe, scale_bits, ek, idx);
        // ct5 = ct-ct4
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            res[i] = tlwe[i] - shift_tlwe[i];
        }
        // ct5小于5bits, 直接提取
        ExtractMSB5(res, res, ek, result_type, idx);
    }
    
    void ExtractMSB10(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t idx = 0) {
        TLWELvl1 shift_tlwe, sign_tlwe5;
        uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - plain_bits;
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = tlwe[i] << (plain_bits - 5);
        }
        MSBGateBootstrapping_kernel(sign_tlwe5, shift_tlwe, ek, ARITHMETIC, idx);
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = shift_tlwe[i] - sign_tlwe5[i];
        }
        IdeGateBootstrapping_kernel(shift_tlwe, shift_tlwe, scale_bits, ek, idx);
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            res[i] = tlwe[i] - shift_tlwe[i];
        }
        // ct5为前5位数
        ExtractMSB9(res, res, plain_bits - 4, ek, result_type, idx); //plain_bits-5?
    }

    void HomMSB(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t idx) {
        if (plain_bits <= 5) ExtractMSB5(res, tlwe, ek, result_type, idx);
        else if (plain_bits <= 9) ExtractMSB9(res, tlwe, plain_bits, ek, result_type, idx);
        else if  (plain_bits == 10) ExtractMSB10(res, tlwe, plain_bits, ek, result_type, idx);
        else throw std::invalid_argument("Plain bits out. ");
    }
    

    void ExtractMSB5(Lvl1::T *res, const Lvl1::T *tlwe, const TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st) {
        MSBGateBootstrapping_kernel(res, tlwe, ek, result_type, idx, st);
    }
    //测试各部分耗时
    // plain_bits: 明文bits数(6-9)
    void ExtractMSB9(Lvl1::T *res, const Lvl1::T *tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st) {
        Lvl1::T *shift_tlwe, *sign_tlwe5;
        cudaMallocHost(&shift_tlwe, sizeof(uint32_t) * 1025);
        cudaMallocHost(&sign_tlwe5, sizeof(uint32_t) * 1025);
        // 32 - plain_bits
        uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - plain_bits;
        // ct1 = ct<<(plain_bits-5)
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = tlwe[i] << (plain_bits - 5);
        }
        // sign_tlwe5 = HMSB(ct1)
        MSBGateBootstrapping_kernel(sign_tlwe5, shift_tlwe, ek, ARITHMETIC, idx, st);
        cudaStreamSynchronize(st);
        // ct3 = ct1 - sign_tlwe5 
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = shift_tlwe[i] - sign_tlwe5[i];
        }
        // ct4 = ct3>>(plain_bits-5)
        IdeGateBootstrapping_kernel(shift_tlwe, shift_tlwe, scale_bits, ek, idx, st);
        cudaStreamSynchronize(st);
        // ct5 = ct-ct4
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            res[i] = tlwe[i] - shift_tlwe[i];
        }
        // ct5小于5bits, 直接提取
        ExtractMSB5(res, res, ek, result_type, idx, st);

        cudaFreeHost(shift_tlwe);
        cudaFreeHost(sign_tlwe5);
    }
    
    void ExtractMSB10(Lvl1::T *res, const Lvl1::T *tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t idx , const cudaStream_t st) {
        Lvl1::T *shift_tlwe, *sign_tlwe5;
        cudaMallocHost(&shift_tlwe, sizeof(uint32_t) * 1025);
        cudaMallocHost(&sign_tlwe5, sizeof(uint32_t) * 1025);
        uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - plain_bits;
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = tlwe[i] << (plain_bits - 5);
        }
        MSBGateBootstrapping_kernel(sign_tlwe5, shift_tlwe, ek, ARITHMETIC, idx, st);
        cudaStreamSynchronize(st);
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = shift_tlwe[i] - sign_tlwe5[i];
        }
        IdeGateBootstrapping_kernel(shift_tlwe, shift_tlwe, scale_bits, ek, idx, st);
        cudaStreamSynchronize(st);
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            res[i] = tlwe[i] - shift_tlwe[i];
        }
        // ct5为前5位数
        ExtractMSB9(res, res, plain_bits - 4, ek, result_type, idx, st); //plain_bits-5?

        cudaFreeHost(shift_tlwe);
        cudaFreeHost(sign_tlwe5);
    }

	void HomMSB(Lvl1::T *res, const Lvl1::T *tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t idx , const cudaStream_t st) {
        if (plain_bits <= 5) ExtractMSB5(res, tlwe, ek, result_type, idx, st);
        else if (plain_bits <= 9) ExtractMSB9(res, tlwe, plain_bits, ek, result_type, idx, st);
        else if  (plain_bits == 10) ExtractMSB10(res, tlwe, plain_bits, ek, result_type, idx, st);
        else throw std::invalid_argument("Plain bits out. ");
    }
  
}; // namespace cuHEDB
