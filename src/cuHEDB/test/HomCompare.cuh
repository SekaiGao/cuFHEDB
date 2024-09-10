#pragma once
#include "msb.cuh"

/***
 * A Greater than B
 * A Greater than or equal to B
 * A less than B
 * A less than or equal to B
 * A equal to B
***/
namespace cuHEDB
{
    // cipher1 > cipher2 <=> msb(cipher2 - cipher1) == 1
    template <typename P>
    void greater_than(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, TLWELvl1 &res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx)
    {
        TFHEpp::TLWE<P> sub_tlwe; 
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher2[i] - cipher1[i];
        }
        //减法需要plain_bits+1位保存结果(正负号)
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, result_type, idx);
    }

    // cipher1 >= cipher2 <=> NOT(msb(cipher1 - cipher2)) == 1
    template <typename P>
    void greater_than_equal(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, TLWELvl1 &res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx)
    {
        TFHEpp::TLWE<P> sub_tlwe; 
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher1[i] - cipher2[i];
        }
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, LOGIC, idx);
        // res每位取负
        HomNOT<Lvl1>(res, res);
        if (IS_ARITHMETIC(result_type)) TFHEpp::LOG_to_ARI(res, res, ek);
    }


    // cipher1 < cipher2 <=> msb(cipher1 - cipher2) == 1
    template <typename P>
    void less_than(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, TLWELvl1 &res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx)
    {
        TFHEpp::TLWE<P> sub_tlwe; 
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher1[i] - cipher2[i];
        }
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, result_type, idx);
    }
    
    // cipher1 <= cipher2 <=> NOT(msb(cipher2 - cipher1)) == 1
    template <typename P>
    void less_than_equal(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, TLWELvl1 &res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx)
    {
        TFHEpp::TLWE<P> sub_tlwe; 
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher2[i] - cipher1[i];
        }
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, LOGIC, idx);
        HomNOT<Lvl1>(res, res);
        if (IS_ARITHMETIC(result_type)) TFHEpp::LOG_to_ARI(res, res, ek);
    }

    // equal是否有改进空间?
    template <typename P>
    void equal(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, TLWELvl1 &res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx)
    {
        TLWELvl1 greater_tlwe, less_tlwe;
        greater_than_equal<P>(cipher1, cipher2, greater_tlwe, plain_bits, ek, LOGIC, idx);
        less_than_equal<P>(cipher1, cipher2, less_tlwe, plain_bits, ek, LOGIC, idx);
        // >= && <=
        HomAND(res, greater_tlwe, less_tlwe, ek, result_type, idx);
    }



    // cipher1 > cipher2 <=> msb(cipher2 - cipher1) == 1
    template <typename P>
    void greater_than(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, Lvl1::T *res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st)
    {
        Lvl1::T *sub_tlwe;
        cudaMallocHost((void **)&sub_tlwe, sizeof(uint32_t) * 1025);
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher2[i] - cipher1[i];
        }
        //减法需要plain_bits+1位保存结果(正负号)
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, result_type, idx, st);
        cudaStreamSynchronize(st);
        cudaFreeHost(sub_tlwe);
    }

    // cipher1 >= cipher2 <=> NOT(msb(cipher1 - cipher2)) == 1
    template <typename P>
    void greater_than_equal(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, Lvl1::T *res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st)
    {
        Lvl1::T *sub_tlwe;
        cudaMallocHost((void **)&sub_tlwe, sizeof(uint32_t) * 1025);
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher1[i] - cipher2[i];
        }
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, LOGIC, idx, st);
        cudaStreamSynchronize(st);
        // res每位取负
        HomNOT(res, res);
        if (IS_ARITHMETIC(result_type)) 
            printf("111\n");
        //TFHEpp::LOG_to_ARI(res, res, ek);

        cudaFreeHost(sub_tlwe);
    }


    // cipher1 < cipher2 <=> msb(cipher1 - cipher2) == 1
    template <typename P>
    void less_than(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, Lvl1::T *res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st)
    {
        Lvl1::T *sub_tlwe;
        cudaMallocHost((void **)&sub_tlwe, sizeof(uint32_t) * 1025);
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher1[i] - cipher2[i];
        }
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, result_type, idx, st);
        cudaStreamSynchronize(st);
        cudaFreeHost(sub_tlwe);
    }
    
    // cipher1 <= cipher2 <=> NOT(msb(cipher2 - cipher1)) == 1
    template <typename P>
    void less_than_equal(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, Lvl1::T *res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st)
    {
        Lvl1::T *sub_tlwe;
        cudaMallocHost((void **)&sub_tlwe, sizeof(uint32_t) * 1025);
        for (size_t i = 0; i <= P::k * P::n; i++)
        {
            sub_tlwe[i] = cipher2[i] - cipher1[i];
        }
        HomMSB(res, sub_tlwe, plain_bits + 1, ek, LOGIC, idx, st);
        cudaStreamSynchronize(st);
        HomNOT(res, res);
        if (IS_ARITHMETIC(result_type))
          printf("111\n");
        //TFHEpp::LOG_to_ARI(res, res, ek);

        cudaFreeHost(sub_tlwe);
    }

    // equal是否有改进空间?
    template <typename P>
    void equal(TFHEpp::TLWE<P> &cipher1, TFHEpp::TLWE<P> &cipher2, Lvl1::T *res, uint32_t plain_bits, TFHEEvalKey &ek, bool result_type, uint32_t idx, const cudaStream_t st)
    {
        Lvl1::T *greater_tlwe, *less_tlwe;
        cudaMallocHost((void **)&greater_tlwe, sizeof(uint32_t) * 1025);
        cudaMallocHost((void **)&less_tlwe, sizeof(uint32_t) * 1025);
        greater_than_equal<P>(cipher1, cipher2, greater_tlwe, plain_bits, ek, LOGIC, idx, st);
        less_than_equal<P>(cipher1, cipher2, less_tlwe, plain_bits, ek, LOGIC, idx, st);
        // >= && <=
        HomAND(res, greater_tlwe, less_tlwe, ek, result_type, idx, st);

        cudaFreeHost(greater_tlwe);
        cudaFreeHost(less_tlwe);
    }
} // namespace HEDB
