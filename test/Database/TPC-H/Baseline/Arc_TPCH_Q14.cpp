#include "ARCEDB/comparison/batch_bootstrap.h"
#include "ARCEDB/comparison/comparable.h"
#include "ARCEDB/comparison/rgsw_ciphertext.h"
#include "ARCEDB/conversion/packlwes.h"
#include "ARCEDB/conversion/repack.h"
#include "ARCEDB/utils/serialize.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <random>
#include <string>
#include <unistd.h>
#include <unordered_map>

using namespace arcedb;
using namespace seal;

/*
    TPC-H Query 14
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
*/

/***
 * AGGREGATE:
 *      Controls whether ciphertext aggregation is enabled. Set to true to combine ciphertexts; 
 *      set to false for fast filtering without aggregation.
 *
 * num:
 *      Specifies the number of rows in the dataset to be processed.
 ***/

#define AGGREGATE true

size_t l_num = 1 << 4;
size_t p_num = 1 << 4;

constexpr int num_thread = 96; // setting multi threads

void tpch_query14(size_t l_num, size_t p_num)
{
	omp_set_num_threads(num_thread);
	
    std::cout << "ArcEDB TPC-H Query14 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Records: " << l_num * p_num << std::endl;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;

    // generate strings for "LIKE"
    int max_like_len = 3;
    std::vector<int> numbers = {0,1,2,3,4,5,6,7,8,9};
    std::vector<std::string> ptype_words = { "PROMO", "COPPER", "LARGE", "BRASS", "POLISHED", "STANDARD", "BRUSHED", "TIN", "ECONOMY", "STEEL" };
    std::uniform_int_distribution<uint32_t> dis(0, ptype_words.size() - 1);
    std::vector<std::vector<std::string>> part_types(p_num);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i=0;i<p_num;++i) {
        std::shuffle(numbers.begin(), numbers.end(), gen);
        for (int k = 0; k < max_like_len; ++k)
            part_types[i].emplace_back(ptype_words[numbers[k]]);
    }
    
    // build hash map
    std::unordered_map<std::string, uint32_t> string2int_ptype;  
    std::unordered_map<uint32_t, std::string> int2string_ptype;
    uint32_t current_id = 0;  

    for (const auto& word : ptype_words) {
        if (string2int_ptype.find(word) == string2int_ptype.end()) {
            string2int_ptype[word] = current_id;
            int2string_ptype[current_id] = word;
            ++current_id;
        }
    }

    std::uniform_int_distribution<uint32_t> shipdate_message(10000, 20000);
    std::uniform_int_distribution<uint32_t> revenue_message(0, 100);
    std::uniform_int_distribution<uint32_t> pkey_message(0, p_num);
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);

    // Filtering
    std::vector<uint64_t> ship_date(l_num);
    std::vector<uint64_t> ptype(p_num), p_pkey(p_num), l_pkey(l_num);
    std::vector<ComparableLvl1> shipdate_ciphers(l_num), l_pkey_ciphers(l_num);
    std::vector<ComparbleRGSWLvl1> p_pkey_ciphers(p_num);
    std::vector<std::vector<TRLWELvl1>> ptype_ciphers(p_num);
    // TLWELvl1 lwe0 = TFHEpp::tlweSymEncrypt<Lvl1>(0, sk.key.get<Lvl1>());
    // TLWELvl1 lwe1 = TFHEpp::tlweSymEncrypt<Lvl1>(1, sk.key.get<Lvl1>());

    // encode and encrypt p_type
    for (int i=0;i<p_num;++i) {
        ptype_ciphers[i].resize(max_like_len);
        for(int k=0;k<max_like_len;++k){
            exponent_encrypt<P>(string2int_ptype[part_types[i][k]], ptype_ciphers[i][k], sk);
        }
    }
    
    // LIKE "PROMO"
    uint64_t like_type = string2int_ptype["PROMO"];

    std::vector<TRGSWLvl1> predicate1_cipher(2), predicate2_cipher(2);
    TRGSWLvl1 like_type_cipher;
    uint64_t predicate1_value = 10592, predicate2_value = 11957;
    exponent_encrypt_rgsw<P>(predicate1_value, 16, predicate1_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate2_value, 16, predicate2_cipher, sk, true);
    exponent_encrypt_rgsw<P>(like_type, like_type_cipher, sk, true);

    int pkey_bit = static_cast<int>(std::log2(p_num)) + 1;

    // Start sql evaluation
    std::vector<TLWELvl1> filter_res(l_num), filter_case_res(l_num);
    std::vector<TLWELvl2> aggregation_res(l_num);
    TLWELvl2 count_res;

    std::vector<double> revenue(l_num);

    for (size_t i = 0; i < l_num; i++)
    {
        revenue[i] = revenue_message(engine);
    }
    
    for (size_t i = 0; i < l_num; i++)
    {
        // Generate data
        ship_date[i] = shipdate_message(engine);
        l_pkey[i] = pkey_message(engine);
        exponent_encrypt<P>(ship_date[i], 16, shipdate_ciphers[i], sk);
        exponent_encrypt<P>(l_pkey[i], pkey_bit, l_pkey_ciphers[i], sk);
    }

    for (size_t i = 0; i < p_num; i++)
    {
        p_pkey[i] = i; // primary key
        exponent_encrypt_rgsw<P>(p_pkey[i], pkey_bit, p_pkey_ciphers[i], sk, true);
    }

    std::chrono::system_clock::time_point start, end;
    double filtering_time_d = 0, filtering_time_h = 0, aggregation_time = 0;
    uint64_t plain_agg_res = 0, plain_agg_case_res = 0;

    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Filtering..." << std::endl;
    start = std::chrono::system_clock::now();
    double stept;

    std::vector<TLWELvl1> ptype_res(p_num);
    // filtering part table
	#pragma omp parallel for
    for (size_t i = 0; i < p_num; i++)
    {
        TLWELvl1 pre_res;
        equality_tfhepp(ptype_ciphers[i][0], like_type_cipher, ptype_res[i], sk);

        // word-wise matching for LIKE
        for (int j=1;j<max_like_len;++j){
            equality_tfhepp(ptype_ciphers[i][j], like_type_cipher, pre_res, sk);
            TFHEpp::HomOR(ptype_res[i], pre_res, ptype_res[i], ek);
        }
    }
    end = std::chrono::system_clock::now();
    stept = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "LIKE time: " << stept << "ms." << std::endl;
    filtering_time_d += stept;

    start = std::chrono::system_clock::now();
    //filtering lineitem table
	#pragma omp parallel for
    for (size_t i = 0; i < l_num; i++)
    {
        TLWELvl1 pre_res;
        
        greater_than_tfhepp(shipdate_ciphers[i], predicate1_cipher, shipdate_ciphers[i].size(), filter_res[i], ek, sk);
        less_than_tfhepp(shipdate_ciphers[i], predicate2_cipher, shipdate_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res[i], pre_res, filter_res[i], ek);
    }
    end = std::chrono::system_clock::now();
    stept = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "filtering time: " << stept << "ms." << std::endl;

    filtering_time_d += stept;

    start = std::chrono::system_clock::now();

    std::vector<TLWELvl1> joined_res(p_num * l_num);
    //join table
	#pragma omp parallel for
    for (size_t i = 0; i < l_num; i++) {
        for(size_t j=0;j<p_num;++j) {
            TLWELvl1 pre_res;
            equality_tfhepp(l_pkey_ciphers[i], p_pkey_ciphers[j], l_pkey_ciphers[i].size(), pre_res, ek, sk);
            TFHEpp::HomAND(pre_res, pre_res, filter_res[i], ek);
            //cuFHEDB::HomAND(joined_res[i*p_num+j], pre_res, ptype_res[j], ek, stream_id);
            lift_and_and(pre_res, ptype_res[j], joined_res[i*p_num+j], 29, ek, sk);
        }
    }
    end = std::chrono::system_clock::now();
    stept = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "JOIN time: " << stept << "ms." << std::endl;

    filtering_time_d += stept;

	#pragma omp parallel for
    for (size_t i = 0; i < l_num; i++)
    {
        //filter_case_res[i] = joined_res[i * p_num];
        filter_case_res[i] = {};
        for(size_t j=0;j<p_num;++j) {
            for(int k=0;k<=Lvl1::n;++k)
                filter_case_res[i][k]+=joined_res[i*p_num+j][k];//HomAdd
        }
        // rescale
        lift_and_and(filter_res[i], filter_res[i], filter_res[i], 29, ek, sk);
    }
    
    
    plain_agg_res = 0, plain_agg_case_res = 0;
    for (size_t i = 0; i < l_num; i++)
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

    
	std::cout << "Cipher query result: " << std::endl;
    std::cout <<"promo_revenue: " << (plain_agg_case_res + 0.) / plain_agg_res << std::endl;

	std::vector<uint64_t> plain_filter_res(l_num), plain_filter_case_res(l_num);
    plain_agg_res = 0, plain_agg_case_res = 0;
    for (size_t i = 0; i < l_num; i++)
    {
        if (ship_date[i] > predicate1_value && ship_date[i] < predicate2_value)
        {
            plain_agg_res += revenue[i];
            for (int j=0;j<p_num;++j) {
                if(l_pkey[i] == p_pkey[j]) {
                    for(int k=0;k<max_like_len;++k) {
                        if (part_types[j][k] == "PROMO") {
                            plain_agg_case_res += revenue[i];
                        }
                    }
                }
            }
        }
    }

    std::cout << "Filtering finish" << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
#if AGGREGATE
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
    int logrow = log2(l_num);
    
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
    std::cout << "Aggregation Time: " << aggregation_time << " ms" << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
#endif
    std::cout << "Plain query result: " << std::endl;
    std::cout <<"promo_revenue: " << (plain_agg_case_res + 0.) / plain_agg_res << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    std::cout << "Query Evaluation Time: " << filtering_time_d + aggregation_time << " ms" << std::endl;
	
}

int main(int argc, char *argv[]) {

  if (argc > 1) {
    l_num = std::stoi(argv[1]);
  }
  if (argc > 2) {
    l_num = std::stoi(argv[1]);
    p_num = std::stoi(argv[2]);
  }

  tpch_query14(l_num, p_num);

  return 0;
}
