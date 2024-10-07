#include "ARCEDB/comparison/batch_bootstrap.h"
#include "ARCEDB/comparison/comparable.h"
#include "ARCEDB/comparison/rgsw_ciphertext.h"
#include "ARCEDB/conversion/packlwes.h"
#include "ARCEDB/conversion/repack.h"
#include "ARCEDB/utils/serialize.h"
#include "cuHEDB/comparable_gpu.cuh"
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace arcedb;
using namespace seal;

size_t num = 1 << 8;

/*
    select
        100.00 * sum(case
            when p_type like 'PROMO%'
                then l_extendedprice * (1 - l_discount)
            else 0
        end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
    from
        lineitem,
        part
    where
        l_partkey = p_partkey
        and l_shipdate >= date ':1'
        and l_shipdate < date ':1' + interval '1' month;
    Consider the joined table
*/
void relational_query14(size_t num)
{
    std::cout << "Relational SQL Query14 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Records: " << num << std::endl;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;
    std::uniform_int_distribution<uint32_t> shipdate_message(10000, 20000);
    std::uniform_int_distribution<uint32_t> revenue_message(0, 100);
    std::uniform_int_distribution<uint32_t> ptype_message(0, 100);
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);
    //ek.emplacebkfft<Lvl02>(sk);

    // load BK to device
    cufftplvl1.LoadBK(*ek.bkfftlvl01);

    // Filtering
    std::vector<uint64_t> ship_date(num);
    std::vector<uint64_t> ptype(num);
    std::vector<ComparableLvl1> shipdate_ciphers(num);
    std::vector<TRLWELvl1> ptype_ciphers(num);


    std::vector<TRGSWLvl1> predicate1_cipher(2), predicate2_cipher(2);
    TRGSWLvl1 predicate3_cipher, predicate4_cipher;
    uint64_t predicate1_value = 10592, predicate2_value = 10957;
    uint64_t predicate3_value = 30, predicate4_value = 70;
    exponent_encrypt_rgsw<P>(predicate1_value, 16, predicate1_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate2_value, 16, predicate2_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate3_value, predicate3_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate4_value, predicate4_cipher, sk, true);


    // Start sql evaluation
    std::vector<TLWELvl1> filter_res(num), filter_case_res(num);
    std::vector<TLWELvl2> aggregation_res(num);
    TLWELvl2 count_res;

    std::vector<double> revenue(num);

    for (size_t i = 0; i < num; i++)
    {
        revenue[i] = revenue_message(engine);
    }
    
    for (size_t i = 0; i < num; i++)
    {
        // Generate data
        ship_date[i] = shipdate_message(engine);
        ptype[i] = ptype_message(engine);
        exponent_encrypt<P>(ship_date[i], 16, shipdate_ciphers[i], sk);
        exponent_encrypt<P>(ptype[i], ptype_ciphers[i], sk);
    }

    std::chrono::system_clock::time_point start, end;
    double filtering_time = 0, aggregation_time;

	start = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < num; i++)
    {
        
        TLWELvl1 pre_res;
        greater_than_tfhepp(shipdate_ciphers[i], predicate1_cipher, shipdate_ciphers[i].size(), filter_res[i], ek, sk);
        less_than_tfhepp(shipdate_ciphers[i], predicate2_cipher, shipdate_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res[i], pre_res, filter_res[i], ek);
        greater_than_tfhepp(ptype_ciphers[i], predicate3_cipher, pre_res, sk);
        TFHEpp::HomAND(filter_case_res[i], pre_res, filter_res[i], ek);
        less_than_tfhepp(ptype_ciphers[i], predicate4_cipher, pre_res, sk);
        lift_and_and(filter_case_res[i], pre_res, filter_case_res[i], 29, ek, sk);
        lift_and_and(filter_res[i], filter_res[i], filter_res[i], 29, ek, sk);
        
    }
    end = std::chrono::system_clock::now();
    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Filter Time on CPU: " << filtering_time << "ms." << std::endl;

	uint64_t plain_agg_res = 0, plain_agg_case_res = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (tlweSymInt32Decrypt<Lvl1>(filter_case_res[i], std::pow(2.,29), sk.key.get<Lvl1>()))
        {
			plain_agg_case_res += revenue[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res[i], std::pow(2.,29), sk.key.get<Lvl1>()))
        {
          plain_agg_res += revenue[i];
        }
    }

	std::cout << "Cipher query result(CPU): " << std::endl;
    std::cout << std::setw(12) <<"promo_revenue" << std::endl;
    std::cout << std::setw(12) << (plain_agg_case_res + 0.) / plain_agg_res << std::endl;


    start = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < num; i++)
    {
        uint32_t stream_id = omp_get_thread_num();

        TLWELvl1 pre_res;
        
        cuARCEDB::greater_than_tfhepp(shipdate_ciphers[i], predicate1_cipher, shipdate_ciphers[i].size(), filter_res[i], ek, sk, stream_id);
        cuARCEDB::less_than_tfhepp(shipdate_ciphers[i], predicate2_cipher, shipdate_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res[i], pre_res, filter_res[i], ek, stream_id);
        cuARCEDB::greater_than_tfhepp(ptype_ciphers[i], predicate3_cipher, pre_res, sk, stream_id);
        cuARCEDB::HomAND(filter_case_res[i], pre_res, filter_res[i], ek, stream_id);
        cuARCEDB::less_than_tfhepp(ptype_ciphers[i], predicate4_cipher, pre_res, sk, stream_id);
        cuARCEDB::lift_and_and(filter_case_res[i], pre_res, filter_case_res[i], 29, ek, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res[i], filter_res[i], filter_res[i], 29, ek, sk, stream_id);
        
        cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Filter Time on GPU: " << filtering_time << "ms." << std::endl;
    
    plain_agg_res = 0, plain_agg_case_res = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (tlweSymInt32Decrypt<Lvl1>(filter_case_res[i], std::pow(2.,29), sk.key.get<Lvl1>()))
        {
			plain_agg_case_res += revenue[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res[i], std::pow(2.,29), sk.key.get<Lvl1>()))
        {
          plain_agg_res += revenue[i];
        }
    }

	std::cout << "Cipher query result(GPU): " << std::endl;
    std::cout << std::setw(12) <<"promo_revenue" << std::endl;
    std::cout << std::setw(12) << (plain_agg_case_res + 0.) / plain_agg_res << std::endl;

	std::vector<uint64_t> plain_filter_res(num), plain_filter_case_res(num);
    plain_agg_res = 0, plain_agg_case_res = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (ship_date[i] > predicate1_value && ship_date[i] < predicate2_value)
        {
            plain_filter_res[i] = 1;
            plain_agg_res += revenue[i];
            if (ptype[i] > predicate3_value && ptype[i] < predicate4_value)
            {
                plain_filter_case_res[i] = 1;
                plain_agg_case_res += revenue[i];
            }
            else
            {
                plain_filter_case_res[i] = 0;
            }
            
        }
        else
        {
            plain_filter_res[i] = 0;
            plain_filter_case_res[i] = 0;
        }

    }

    std::cout << "Filtering finish" << std::endl;

#if 0
    std::cout << "Aggregation :" << std::endl;
    uint64_t scale_bits = 29;
    uint64_t modq_bits = 32;
    uint64_t modulus_bits = 45;
    uint64_t repack_scale_bits = modulus_bits + scale_bits - modq_bits;
    uint64_t slots_count = filter_res.size();
    std::cout << "Generating Parameters..." << std::endl;
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, {59, 42, 42, 42, 42, 42, 42, 42, 42, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 59}));
    double scale = std::pow(2.0, scale_bits);

    //context instance
    seal::SEALContext context(parms, true, seal::sec_level_type::none);

    //key generation
    seal::KeyGenerator keygen(context);
    seal::SecretKey seal_secret_key = keygen.secret_key();
    seal::RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    seal::GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);
    

    //utils
    seal::Encryptor encryptor(context, seal_secret_key);
    seal::Evaluator evaluator(context);
    seal::Decryptor decryptor(context, seal_secret_key);

    //encoder
    seal::CKKSEncoder ckks_encoder(context);

    

    //generate evaluation key
    std::cout << "Generating Conversion Key..." << std::endl;
    LTPreKey pre_key;
    LWEsToRLWEKeyGen(pre_key, std::pow(2., modulus_bits), seal_secret_key, sk, P::n, ckks_encoder, encryptor, context);


    // conversion
    std::cout << "Starting Conversion..." << std::endl;
    seal::Ciphertext result, result_case;
    start = std::chrono::system_clock::now();
    LWEsToRLWE(result, filter_res, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(result, result.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

    LWEsToRLWE(result_case, filter_case_res, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(result_case, result_case.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);
    end = std::chrono::system_clock::now();
    aggregation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    seal::Plaintext plain;
    std::vector<double> computed(slots_count), computed_case(slots_count);
    decryptor.decrypt(result, plain);
    seal::pack_decode(computed, plain, ckks_encoder);

    decryptor.decrypt(result_case, plain);
    seal::pack_decode(computed_case, plain, ckks_encoder);

    double err1 = 0., err2 = 0.;
    
    for (size_t i = 0; i < slots_count; ++i)
    {
        err1 += std::abs(computed[i] - plain_filter_res[i]);
        err2 += std::abs(computed_case[i] - plain_filter_case_res[i]);
    }

    printf("Repack average error = %f ~ 2^%.1f\n", err1 / slots_count, std::log2(err1 / slots_count));
    printf("Repack average error = %f ~ 2^%.1f\n", err2 / slots_count, std::log2(err2 / slots_count));


    // Filter result * data
    seal::Ciphertext revenue_cipher;
    double qd = parms.coeff_modulus()[result.coeff_modulus_size() - 1].value();
    seal::pack_encode(revenue, qd, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, revenue_cipher);

    std::cout << "Aggregating price and discount .." << std::endl;
    start = std::chrono::system_clock::now();
    seal::multiply_and_relinearize(result, revenue_cipher, result, evaluator, relin_keys);
    seal::multiply_and_relinearize(result_case, revenue_cipher, result_case, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(result);
    evaluator.rescale_to_next_inplace(result_case);
    std::cout << "Remian modulus: " << result.coeff_modulus_size() << std::endl;
    int logrow = log2(num);
    
    seal::Ciphertext temp;
    size_t step;
    for (size_t i = 0; i < logrow; i++)
    {
        temp = result;
        step = 1 << (logrow - i - 1);
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(result, temp);

        temp = result_case;
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(result_case, temp);
    }
    end = std::chrono::system_clock::now();
    aggregation_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::vector<double> agg_result(slots_count), agg_case_result(slots_count);
    decryptor.decrypt(result, plain);
    seal::pack_decode(agg_result, plain, ckks_encoder);

    decryptor.decrypt(result_case, plain);
    seal::pack_decode(agg_case_result, plain, ckks_encoder);

    std::cout << "Query Evaluation Time: " << filtering_time + aggregation_time << " ms" << std::endl;

    std::cout << "Encrypted query result: " << std::endl;
    std::cout << std::setw(12) <<"promo_revenue" << std::endl;
    std::cout << std::setw(12) << agg_case_result[0] / agg_result[0] << std::endl;
#endif

	std::cout << "Plain query result: " << std::endl;
    std::cout << std::setw(12) <<"promo_revenue" << std::endl;
    std::cout << std::setw(12) << (plain_agg_case_res + 0.) / plain_agg_res << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    
}


int main()
{
  omp_set_num_threads(num_stream2);
  warmupGPU();

  relational_query14(num);

  return 0;
}


