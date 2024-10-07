#include "HEDB/comparison/comparison.h"
#include "HEDB/conversion/repack.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "cuHEDB/HomCompare_gpu.cuh"
#include "fastR.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <unordered_set>
#include <vector>

using namespace HEDB;
using namespace TFHEpp;

const int rows = 1 << 10; // Number of plaintexts

/***
select
    sum(l_extendedprice) / 7.0 as avg_yearly 
from
    lineitem, part
where
    p_partkey = l_partkey
    and p_brand = '[BRAND]' // 0-A, 1-B, 2-C, 3-D, 4-E, 5-F, 6-G, 7-H
    and p_container = '[CONTAINER]' //0, 1, 2, 3 
    and l_quantity < ( 
        select
            0.2 * avg(l_quantity)
        from
            lineitem
        where
            l_partkey = p_partkey
    );
    Consider the joined table
*/

const uint32_t price_bits = 8;
const uint32_t quantity_bits = 9;
const uint32_t brand_bits = 3;
const uint32_t container_bits = 2;

//lineitem(l_extendedprice(8), l_quantity(9), l_brand(3), l_container(2))
void gen_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem, int lineitem_rows) {

    std::random_device seed_gen;
    std::default_random_engine generator(seed_gen());
    std::uniform_int_distribution<Lvl1::T> price_distribution(0, (1 << price_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> quantity_distribution(0, 50);
    std::uniform_int_distribution<Lvl1::T> brand_distribution(0, (1 << brand_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> container_distribution(0, (1 << container_bits) - 1);

    std::vector<Lvl1::T> prices, quantities, brands, containers;

    std::cout<<"Begin to generate database..."<<std::endl;

    for (int i = 0; i < lineitem_rows; ++i) {
        Lvl1::T price = price_distribution(generator);
        Lvl1::T quantity = quantity_distribution(generator);
        Lvl1::T brand = brand_distribution(generator);
        Lvl1::T container = container_distribution(generator);

        prices.emplace_back(price);
        quantities.emplace_back(quantity);
        brands.emplace_back(brand);
        containers.emplace_back(container);
    }

    plain_lineitem.emplace_back(prices);
    plain_lineitem.emplace_back(quantities);
    plain_lineitem.emplace_back(brands);
    plain_lineitem.emplace_back(containers);

    std::cout<<"Successfully generated database."<<std::endl;
}

void encrypt_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem,
                      std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                      TFHESecretKey &sk) {

    uint32_t scale_bits_lineitem[4];
    scale_bits_lineitem[0] = std::numeric_limits<Lvl1::T>::digits - price_bits - 1;
    scale_bits_lineitem[1] = std::numeric_limits<Lvl1::T>::digits - quantity_bits - 1;
    scale_bits_lineitem[2] = std::numeric_limits<Lvl1::T>::digits - brand_bits - 1;
    scale_bits_lineitem[3] = std::numeric_limits<Lvl1::T>::digits - container_bits - 1;

    std::cout << "Begin to encrypt database..." << std::endl;

    cipher_lineitem.resize(4);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < plain_lineitem[i].size(); ++j) {
            cipher_lineitem[i].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_lineitem[i][j], Lvl1::α, pow(2., scale_bits_lineitem[i]), sk.key.get<Lvl1>()));
        }
    }

    std::cout << "Success encrypt database." << std::endl;
}

void encrypt_Condition(std::vector<Lvl1::T> &plain_num, std::vector<TLWE<Lvl1>> &cipher_num, TFHESecretKey &sk) {
 
	uint32_t brand_scale_bits = std::numeric_limits<Lvl1::T>::digits - brand_bits - 1;
	uint32_t container_scale_bits = std::numeric_limits<Lvl1::T>::digits - container_bits - 1;
	
    cipher_num.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_num[0], Lvl1::α, pow(2., brand_scale_bits), sk.key.get<Lvl1>()));
	cipher_num.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_num[1], Lvl1::α, pow(2., container_scale_bits), sk.key.get<Lvl1>()));
}



void Plain_Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem, std::vector<Lvl1::T> &plain_num) {
    std::vector<Lvl1::T> &l_price = plain_lineitem[0];
    std::vector<Lvl1::T> &l_quantity = plain_lineitem[1];
    std::vector<Lvl1::T> &l_brand = plain_lineitem[2];
    std::vector<Lvl1::T> &l_container = plain_lineitem[3];

	Lvl1::T filter_brand = plain_num[0];
	Lvl1::T filter_container = plain_num[1];

    std::cout<<"Begin to filter plain database..."<<std::endl;

    std::vector<uint32_t> pres(l_price.size());
    double avg = 0;

    for (size_t i = 0; i < l_price.size(); ++i) {
        avg += l_quantity[i];
    }
    uint32_t avgu = uint32_t(0.2*avg/l_price.size());
    
    double avg_yearly = 0;

    for (size_t i = 0; i < l_price.size(); ++i) {
        pres[i] = 0;
        if (l_brand[i] == filter_brand && l_container[i] == filter_container && l_quantity[i] < avgu){
            pres[i] = 1;
            avg_yearly += l_price[i];
        }  
    }

    avg_yearly /= 7.0;
    std::cout << "\nPlaintext query result(avg_yearly):" << avg_yearly<<endl<< std::endl;

}

void Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem, std::vector<uint32_t> &pred_res) {
    std::vector<Lvl1::T> &l_price = plain_lineitem[0];
    std::vector<Lvl1::T> &l_quantity = plain_lineitem[1];

    std::vector<uint32_t> pres(l_price.size());
    double avg = 0;

    for (size_t i = 0; i < l_price.size(); ++i) {
        avg += l_quantity[i];
    }
    uint32_t avgu = uint32_t(0.2*avg/l_price.size());
    
    double avg_yearly = 0;

    for (size_t i = 0; i < l_price.size(); ++i) {
        if (pred_res[i]==1){
            avg_yearly += l_price[i];
        }  
    }

    avg_yearly /= 7.0;
    std::cout << "\nCiphertext query result(avg_yearly):" << avg_yearly<<endl<< std::endl;

}

void AggAVG(TLWELvl1 &cres, std::vector<TLWELvl1> &pred_cres, TFHESecretKey &sk) {
    TLWELvl1 partial_cres{}; 
    TLWELvl1 final_cres{};   
    size_t batch_size = 10;   
    size_t total_size = pred_cres.size();
    uint32_t final_res = 0;

    uint32_t quantity_scale_bits = std::numeric_limits<Lvl1::T>::digits - quantity_bits - 1;

    for (size_t i = 0; i < total_size; i += batch_size) {
        partial_cres = {};

        for (size_t k = i; k < std::min(i + batch_size, total_size); ++k) {
            for (size_t j = 0; j <= Lvl1::n; j++) {
                partial_cres[j] += pred_cres[k][j];
            }
        }

        final_res += TFHEpp::tlweSymInt32Decrypt<Lvl1>(partial_cres, pow(2., quantity_scale_bits), sk.key.lvl1);
    }

    cres = TFHEpp::tlweSymInt32Encrypt<Lvl1>(uint32_t(0.2*final_res/total_size), Lvl1::α, pow(2., quantity_scale_bits), sk.key.lvl1);
}


void Filter_Cipher_d(std::vector<TLWELvl1> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     std::vector<TLWE<Lvl1>> &cipher_num, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<uint32_t> &pred_res) {

  std::vector<TLWE<Lvl1>> &l_price = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_quantity = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_brand = cipher_lineitem[2];
  std::vector<TLWE<Lvl1>> &l_container = cipher_lineitem[3];

  TLWE<Lvl1> &brand = cipher_num[0];
  TLWE<Lvl1> &container = cipher_num[1];
  

  uint32_t lineitem_rows = l_price.size();

  std::vector<TLWELvl1> pred_cres1(lineitem_rows), pred_cres2(lineitem_rows),
      pred_cres3(lineitem_rows);

  cres.resize(lineitem_rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  TLWELvl1 Avg;
  AggAVG(Avg, l_quantity, sk);

    #pragma omp parallel for
    for (size_t i = 0; i < lineitem_rows; ++i) {
        uint32_t stream_id = omp_get_thread_num();
        cuHEDB::less_than<Lvl1>(l_quantity[i], Avg, pred_cres1[i], quantity_bits, ek, LOGIC, stream_id);
        cuHEDB::equal<Lvl1>(l_brand[i], brand, pred_cres2[i], brand_bits, ek, LOGIC, stream_id);
        cuHEDB::equal<Lvl1>(l_container[i], container, pred_cres3[i], brand_bits, ek, LOGIC, stream_id);
        cuHEDB::HomAND(cres[i], pred_cres1[i], pred_cres2[i], ek, LOGIC, stream_id);
        cuHEDB::HomAND(cres[i], pred_cres3[i], cres[i], ek, LOGIC, stream_id);

        cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();
    double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Filter Time on GPU: " << costs / 1000 << "ms" << std::endl;

    uint32_t rlwe_scale_bits = 29;
    #pragma omp parallel for
    for (size_t i = 0; i < lineitem_rows; ++i) {
        TFHEpp::log_rescale(cres[i], cres[i], rlwe_scale_bits, ek);
    }

    pred_res.resize(lineitem_rows);

    for (size_t i = 0; i < lineitem_rows; ++i) {
        pred_res[i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[i], pow(2., 29), sk.key.get<Lvl1>());
	}
}


void Filter_Cipher_h(std::vector<TLWELvl1> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     std::vector<TLWE<Lvl1>> &cipher_num, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<uint32_t> &pred_res) {

  std::vector<TLWE<Lvl1>> &l_price = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_quantity = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_brand = cipher_lineitem[2];
  std::vector<TLWE<Lvl1>> &l_container = cipher_lineitem[3];

  TLWE<Lvl1> &brand = cipher_num[0];
  TLWE<Lvl1> &container = cipher_num[1];
  

  uint32_t lineitem_rows = l_price.size();

  std::vector<TLWELvl1> pred_cres1(lineitem_rows), pred_cres2(lineitem_rows),
      pred_cres3(lineitem_rows);

  cres.resize(lineitem_rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  TLWELvl1 Avg;
  AggAVG(Avg, l_quantity, sk);

    for (size_t i = 0; i < lineitem_rows; ++i) {
        HEDB::less_than<Lvl1>(l_quantity[i], Avg, pred_cres1[i], quantity_bits, ek, LOGIC);
        HEDB::equal<Lvl1>(l_brand[i], brand, pred_cres2[i], brand_bits, ek, LOGIC);
        HEDB::equal<Lvl1>(l_container[i], container, pred_cres3[i], brand_bits, ek, LOGIC);
        HEDB::HomAND(cres[i], pred_cres1[i], pred_cres2[i], ek, LOGIC);
        HEDB::HomAND(cres[i], pred_cres3[i], cres[i], ek, LOGIC);
    }

    end = std::chrono::system_clock::now();
    double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Filter Time on CPU: " << costs / 1000 << "ms" << std::endl;

#if 0
    uint32_t rlwe_scale_bits = 29;
    for (size_t i = 0; i < lineitem_rows; ++i) {
        TFHEpp::log_rescale(cres[i], cres[i], rlwe_scale_bits, ek);
    }

   

    pred_res.resize(lineitem_rows);

    for (size_t i = 0; i < lineitem_rows; ++i) {
        pred_res[i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[i], pow(2., 29), sk.key.get<Lvl1>());
	}
#endif
}

void aggregation(std::vector<TLWELvl1> &pred_cres, std::vector<uint32_t> &pred_res, size_t tfhe_n,
            std::vector<Lvl1::T> &extendedprice_data,
             size_t rows, TFHESecretKey &sk) 
{
    double aggregation_time;
    std::cout << "Aggregation :" << std::endl;
    uint64_t scale_bits = 29;
    uint64_t modq_bits = 32;
    uint64_t modulus_bits = 45;
    uint64_t repack_scale_bits = modulus_bits + scale_bits - modq_bits;
    uint64_t slots_count = pred_cres.size();
    std::cout << "Generating Parameters..." << std::endl;
    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    size_t poly_modulus_degree = 65536;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, {59, 42, 42, 42, 42, 42, 42, 42, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 59}));
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
    LWEsToRLWEKeyGen(pre_key, std::pow(2., modulus_bits), seal_secret_key, sk, tfhe_n, ckks_encoder, encryptor, context);


    // conversion
    std::cout << "Starting Conversion..." << std::endl;
    // repack pred_cres(LWEs) -> result(RLWE)
    seal::Ciphertext result;
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    LWEsToRLWE(result, pred_cres, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
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
        err += std::abs(computed[i] - pred_res[i]);
    }

    printf("Repack average error = %f ~ 2^%.1f\n", err / slots_count, std::log2(err / slots_count));

    std::vector<double> price_discount(extendedprice_data.size());
    seal::Ciphertext price_discount_cipher;
    for (size_t i = 0; i < rows; i++)
    {
        price_discount[i] = extendedprice_data[i];
    }
    double qd = parms.coeff_modulus()[result.coeff_modulus_size() - 1].value();
    seal::pack_encode(price_discount, qd, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, price_discount_cipher);

    std::cout << "Aggregating price .." << std::endl;
    start = std::chrono::system_clock::now();
    seal::multiply_and_relinearize(result, price_discount_cipher, result, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(result);
    int logrow = log2(rows);

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

    
    cout << "\nEncrypted aggregation result(avg_yearly): " << agg_result[0] / 7.0 <<endl;
    std::cout<<"\nAggregation time: "<<aggregation_time<<"ms"<<std::endl;
}


int main() {

  omp_set_num_threads(num_stream1);
  // Lvl1
  std::cout << "Encrypting" << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  double costs;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());


  TFHEpp::BootstrappingKeyFFT<lvl01param> *bkfft = new TFHEpp::BootstrappingKeyFFT<lvl01param>;
  TFHEpp::KeySwitchingKey<TFHEpp::lvl10param> *isk = new TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>;  

  TFHESecretKey sk;
  TFHEEvalKey ek;
  // ek.emplacebkfft<Lvl01>(sk);
  // ek.emplaceiksk<Lvl10>(sk);
  readFromFile(path, sk.key.lvl1, *bkfft, *isk);
  ek.bkfftlvl01.reset(bkfft);
  ek.iksklvl10.reset(isk);


  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "generate Secret Key Time: " << costs/1000 << "ms" <<std::endl;

  std::cout << "Loading" << std::endl;

  warmupGPU();
  start = std::chrono::system_clock::now();

  //load BK to device
  cufftlvl1.LoadBK(*ek.bkfftlvl01);
  //cufftlvl2.LoadBK(*ek.bkfftlvl02);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<"Load Success."<<std::endl;
  std::cout << "Loading Time: " << costs/1000 << "ms" << std::endl;


  // DataBase init
  std::vector<std::vector<Lvl1::T>> plain_lineitem;
  std::vector<Lvl1::T> plain_num = {4, 2};
  std::vector<std::vector<TLWE<Lvl1>>> cipher_lineitem;
  std::vector<TLWE<Lvl1>> cipher_num;
  std::vector<TLWE<Lvl1>> cres;
  std::vector<Lvl1::T> pres;

  gen_DataBase(plain_lineitem, rows);

  encrypt_DataBase(plain_lineitem, cipher_lineitem, sk);

  encrypt_Condition(plain_num, cipher_num, sk);

  Plain_Query(plain_lineitem, plain_num);

  Filter_Cipher_d(cres, cipher_lineitem, cipher_num, ek, sk, pres);

  Query(plain_lineitem, pres);

#if 0
  aggregation(cres, pres, Lvl1::n, plain_lineitem[0], rows, sk);

  Filter_Cipher_h(cres, cipher_lineitem, cipher_num, ek, sk, pres);
#endif  
  return 0;
}
