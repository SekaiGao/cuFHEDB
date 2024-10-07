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

/***
 * TPC-H Query 6
 * select
        sum(l_extendedprice * l_discount) as revenue
    from
	    lineitem
    where
        l_shipdate >= date ':1'
        and l_shipdate < date ':1' + interval '1' year
        and l_discount between :2 - 0.01 and :2 + 0.01
        and l_quantity < :3;
    
    consider data \in [10592~10957]
*/

void relational_query6(size_t num)
{
    std::cout << "Relational SQL Query6 Test: "<< std::endl;
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
    std::uniform_int_distribution<uint32_t> discount_message(19000, 21000);
    std::uniform_int_distribution<uint32_t> quantity_message(20000, 40000);
    std::uniform_int_distribution<uint64_t> revenue_message(0, 100);
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);

    // load BK to device
    cufftplvl1.LoadBK(*ek.bkfftlvl01);

    // Filtering
    std::vector<uint64_t> ship_date(num);
    std::vector<uint64_t> discount(num), quantity(num);
    std::vector<ComparableLvl1> shipdate_ciphers(num), discount_ciphers(num), quantity_ciphers(num);


    std::vector<TRGSWLvl1> predicate1_cipher(2), predicate2_cipher(2), predicate3_cipher(2), predicate4_cipher(2), predicate5_cipher(2);
    uint64_t predicate1_value = 10592, predicate2_value = 10957;
    uint64_t predicate3_value = 19900, predicate4_value = 20100, predicate5_value = 30000;
    exponent_encrypt_rgsw<P>(predicate1_value, 16, predicate1_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate2_value, 16, predicate2_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate3_value, 16, predicate3_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate4_value, 16, predicate4_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate5_value, 16, predicate5_cipher, sk, true);


    // Start sql evaluation
    std::vector<TLWELvl1> filter_res(num), filter_res1(num);
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
        discount[i] = discount_message(engine);
        quantity[i] = quantity_message(engine);
        exponent_encrypt<P>(ship_date[i], 16, shipdate_ciphers[i], sk);
        exponent_encrypt<P>(discount[i], 16, discount_ciphers[i], sk);
        exponent_encrypt<P>(quantity[i], 16, quantity_ciphers[i], sk);
    }

    std::chrono::system_clock::time_point start, end;
    double filtering_time = 0, aggregation_time;

	start = std::chrono::system_clock::now();

	#pragma omp parallel for
    for (size_t i = 0; i < num; i++)
    {
        uint32_t stream_id = omp_get_thread_num();

        TLWELvl1 pre_res;
        cuARCEDB::greater_than_tfhepp(shipdate_ciphers[i], predicate1_cipher, shipdate_ciphers[i].size(), filter_res[i], ek, sk, stream_id);
        cuARCEDB::less_than_tfhepp(shipdate_ciphers[i], predicate2_cipher, shipdate_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res[i], pre_res, filter_res[i], ek, stream_id);
        cuARCEDB::greater_than_tfhepp(discount_ciphers[i], predicate3_cipher, discount_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res[i], pre_res, filter_res[i], ek, stream_id);
        cuARCEDB::less_than_tfhepp(discount_ciphers[i], predicate4_cipher, discount_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res[i], pre_res, filter_res[i], ek, stream_id);
        cuARCEDB::less_than_tfhepp(quantity_ciphers[i], predicate5_cipher, quantity_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res[i], pre_res, filter_res[i], 29, ek, sk, stream_id);
        
		cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();

    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Filter Time on GPU: " << filtering_time << "ms." << std::endl;


    start = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < num; i++)
    {
        
        TLWELvl1 pre_res;
        greater_than_tfhepp(shipdate_ciphers[i], predicate1_cipher, shipdate_ciphers[i].size(), filter_res1[i], ek, sk);
        less_than_tfhepp(shipdate_ciphers[i], predicate2_cipher, shipdate_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res1[i], pre_res, filter_res1[i], ek);
        greater_than_tfhepp(discount_ciphers[i], predicate3_cipher, discount_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res1[i], pre_res, filter_res1[i], ek);
        less_than_tfhepp(discount_ciphers[i], predicate4_cipher, discount_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res1[i], pre_res, filter_res1[i], ek);
        less_than_tfhepp(quantity_ciphers[i], predicate5_cipher, quantity_ciphers[i].size(), pre_res, ek, sk);
        lift_and_and(filter_res1[i], pre_res, filter_res1[i], 29, ek, sk);
        
    }

    end = std::chrono::system_clock::now();

    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Filter Time on CPU: " << filtering_time << "ms." << std::endl;

    std::vector<uint64_t> plain_filter_res(num);
    uint64_t plain_agg_res = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (ship_date[i] > predicate1_value && ship_date[i] < predicate2_value && discount[i] > predicate3_value && discount[i] < predicate4_value && quantity[i] < predicate5_value)
        {
            plain_filter_res[i] = 1.;
            plain_agg_res += revenue[i];
        }
        else
        {
            plain_filter_res[i] = 0.;
        }

    }

    std::cout << "Filtering finish." << std::endl;

    std::cout << "Plaintext query result: " << plain_agg_res<<std::endl;

    uint64_t cipher_agg_res = 0;
    for (size_t i = 0; i < num; i++) {
      if (tlweSymInt32Decrypt<Lvl1>(filter_res[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
        cipher_agg_res += revenue[i];
        }
    }
    std::cout << "Ciphertext query result(GPU): " << cipher_agg_res << std::endl;

    cipher_agg_res = 0;
    for (size_t i = 0; i < num; i++) {
      if (tlweSymInt32Decrypt<Lvl1>(filter_res1[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
        cipher_agg_res += revenue[i];
        }
    }
    std::cout << "Ciphertext query result(CPU): " << cipher_agg_res << std::endl;

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
    seal::Ciphertext result;
    start = std::chrono::system_clock::now();
    LWEsToRLWE(result, filter_res, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(result, result.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);
    end = std::chrono::system_clock::now();
    aggregation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    seal::Plaintext plain;
    std::vector<double> computed(slots_count);
    decryptor.decrypt(result, plain);
    seal::pack_decode(computed, plain, ckks_encoder);

    double err = 0.;
    
    for (size_t i = 0; i < slots_count; ++i)
    {
        err += std::abs(computed[i] - plain_filter_res[i]);
    }

    printf("Repack average error = %f ~ 2^%.1f\n", err / slots_count, std::log2(err / slots_count));


    // Filter result * data
    seal::Ciphertext revenue_cipher;
    double qd = parms.coeff_modulus()[result.coeff_modulus_size() - 1].value();
    seal::pack_encode(revenue, qd, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, revenue_cipher);

    std::cout << "Aggregating price and discount .." << std::endl;
    start = std::chrono::system_clock::now();
    seal::multiply_and_relinearize(result, revenue_cipher, result, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(result);
    std::cout << "Remian modulus: " << result.coeff_modulus_size() << std::endl;
    int logrow = log2(num);
    
    seal::Ciphertext temp;
    for (size_t i = 0; i < logrow; i++)
    {
        temp = result;
        size_t step = 1 << (logrow - i - 1);
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(result, temp);
    }
    end = std::chrono::system_clock::now();
    aggregation_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::vector<double> agg_result(slots_count);
    decryptor.decrypt(result, plain);
    seal::pack_decode(agg_result, plain, ckks_encoder);

    std::cout << "Aggregation Time: " << aggregation_time << " ms" << std::endl;

    std::cout << "Encrypted query result: " << std::endl;
    std::cout << std::setw(12) <<"revenue" << std::endl;
    std::cout << std::setw(12) << std::round(agg_result[0]) << std::endl;
    std::cout << "Plain query result: " << std::endl;
    std::cout << std::setw(12) <<"revenue" << std::endl;
    std::cout << std::setw(12) << plain_agg_res << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
#endif
}




int main()
{
  omp_set_num_threads(num_stream2);
  warmupGPU();

  relational_query6(num);

  return 0;
}


