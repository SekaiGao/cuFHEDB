#pragma once

#include "gatebootstrap_gpu.cuh"

namespace cuHEDB
{
    void ExtractMSB5(TLWELvl1 &res, const TLWELvl1 &tlwe, const TFHEEvalKey &ek, bool result_type, uint32_t stream_id = 0) {
        MSBGateBootstrapping(res, tlwe, ek, result_type, stream_id);
    }
    void ExtractMSB9(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t stream_id = 0) {
        TLWELvl1 shift_tlwe, sign_tlwe5;
        // 32 - plain_bits
        uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - plain_bits;
        // ct1 = ct<<(plain_bits-5)
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = tlwe[i] << (plain_bits - 5);
        }
        // sign_tlwe5 = HMSB(ct1)
        MSBGateBootstrapping(sign_tlwe5, shift_tlwe, ek, ARITHMETIC, stream_id);
        // ct3 = ct1 - sign_tlwe5 
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = shift_tlwe[i] - sign_tlwe5[i];
        }
        // ct4 = ct3>>(plain_bits-5)
        IdeGateBootstrapping(shift_tlwe, shift_tlwe, scale_bits, ek, stream_id);
        // ct5 = ct-ct4
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            res[i] = tlwe[i] - shift_tlwe[i];
        }
        ExtractMSB5(res, res, ek, result_type, stream_id);
    }
    
    void ExtractMSB10(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t stream_id = 0) {
        TLWELvl1 shift_tlwe, sign_tlwe5;
        uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - plain_bits;
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = tlwe[i] << (plain_bits - 5);
        }
        MSBGateBootstrapping(sign_tlwe5, shift_tlwe, ek, ARITHMETIC, stream_id);
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            shift_tlwe[i] = shift_tlwe[i] - sign_tlwe5[i];
        }
        IdeGateBootstrapping(shift_tlwe, shift_tlwe, scale_bits, ek, stream_id);
        for (size_t i = 0; i <= Lvl1 :: n; i++)
        {
            res[i] = tlwe[i] - shift_tlwe[i];
        }
    
        ExtractMSB9(res, res, plain_bits - 4, ek, result_type, stream_id); //plain_bits-5?
    }

    void HomMSB(TLWELvl1 &res, const TLWELvl1 &tlwe, uint32_t plain_bits, const TFHEEvalKey &ek, bool result_type, uint32_t stream_id) {
        if (plain_bits <= 5) ExtractMSB5(res, tlwe, ek, result_type, stream_id);
        else if (plain_bits <= 9) ExtractMSB9(res, tlwe, plain_bits, ek, result_type, stream_id);
        else if  (plain_bits == 10) ExtractMSB10(res, tlwe, plain_bits, ek, result_type, stream_id);
        else throw std::invalid_argument("Plain bits out. ");
    }
    
};