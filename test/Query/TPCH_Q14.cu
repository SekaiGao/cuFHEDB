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

const int rows = 1 << 18; // Number of plaintexts

/***
select
100.00 * sum(case
when p_type like 'PROMO%' 
then l_extendedprice*(1-l_discount) 
else 0
end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
lineitem, part
where
l_partkey = p_partkey
and l_shipdate >= date '[DATE]' 
and l_shipdate < date '[DATE]' + interval '1' month;

Consider the joined table
*/

const uint32_t key_bits = 6;
const uint32_t type_bits = 3;
const uint32_t date_bits = 9;
const uint32_t price_bits = 8;
const uint32_t discount_bits = 8;

//lineitem(partkey(6), type(3), shipdate(9), price(8), discount(8))
void gen_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem, int lineitem_rows) {
  
    std::random_device seed_gen;
    std::default_random_engine generator(seed_gen());
    std::uniform_int_distribution<Lvl1::T> partkey_distribution(0, (1 << key_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> type_distribution(0, (1 << type_bits) - 1);
	  std::uniform_int_distribution<Lvl1::T> date_distribution(0, (1 << date_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> price_distribution(0, (1 << price_bits) - 1);
	  std::uniform_int_distribution<Lvl1::T> discount_distribution(0, 100);

    std::vector<Lvl1::T> partkeys, types, shipdates, prices, discounts;

    std::cout<<"Begin to generate database..."<<std::endl;

    for (int i = 0; i < lineitem_rows; ++i) {
        Lvl1::T partkey = partkey_distribution(generator);
        Lvl1::T type = type_distribution(generator);
        Lvl1::T shipdate = date_distribution(generator);
        Lvl1::T price = price_distribution(generator);
        Lvl1::T discount = discount_distribution(generator);

        partkeys.emplace_back(partkey);
        types.emplace_back(type);
        shipdates.emplace_back(shipdate);
        prices.emplace_back(price);
        discounts.emplace_back(discount);
    }

    plain_lineitem.emplace_back(partkeys);
    plain_lineitem.emplace_back(types);
    plain_lineitem.emplace_back(shipdates);
    plain_lineitem.emplace_back(prices);
    plain_lineitem.emplace_back(discounts);

    std::cout<<"Successfully generated database."<<std::endl;
}


void encrypt_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem,
                      std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                      TFHESecretKey &sk) {

  uint32_t scale_bits_lineitem[3];
  scale_bits_lineitem[0] = std::numeric_limits<Lvl1::T>::digits - key_bits - 1;
  scale_bits_lineitem[1] = std::numeric_limits<Lvl1::T>::digits - type_bits - 1;
  scale_bits_lineitem[2] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
  
  std::cout << "Begin to encrypt database..." << std::endl;

  cipher_lineitem.resize(3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < plain_lineitem[i].size(); ++j) {
      cipher_lineitem[i].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_lineitem[i][j], Lvl1::α, pow(2., scale_bits_lineitem[i]), sk.key.get<Lvl1>()));
      }
  }

  std::cout << "Success encrypt database." << std::endl;
}

void encrypt_Condition(std::vector<Lvl1::T> &plain_type, std::vector<Lvl1::T> &plain_date, std::vector<TLWE<Lvl1>> &cipher_type, std::vector<TLWE<Lvl1>> &cipher_date, TFHESecretKey &sk) {
 
	uint32_t date_scale_bits = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
	uint32_t type_scale_bits = std::numeric_limits<Lvl1::T>::digits - type_bits - 1;
	
	
  cipher_date.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date[0], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	cipher_date.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date[1], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	cipher_type.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_type[0], Lvl1::α, pow(2., type_scale_bits), sk.key.get<Lvl1>()));
  cipher_type.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_type[1], Lvl1::α, pow(2., type_scale_bits), sk.key.get<Lvl1>()));
}

void Plain_Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem,
                 std::vector<Lvl1::T> &filter_type,
                 std::vector<Lvl1::T> &filter_date, std::vector<Lvl1::T> &Tpres,
                 std::vector<Lvl1::T> &Ppres) {

  std::vector<Lvl1::T> &l_partkey = plain_lineitem[0];
  std::vector<Lvl1::T> &l_shipdate = plain_lineitem[2];
  std::vector<Lvl1::T> &l_type = plain_lineitem[1];
  std::vector<Lvl1::T> &l_extendedprice = plain_lineitem[3];
  std::vector<Lvl1::T> &l_discount = plain_lineitem[4];

  Lvl1::T date1 = filter_date[0];
  Lvl1::T date2 = filter_date[1];

  Lvl1::T type1 = filter_type[0];
  Lvl1::T type2 = filter_type[1];

  double promo_revenue = 0.0;
  double total_revenue = 0.0;

  Tpres.resize(l_partkey.size());
  Ppres.resize(l_partkey.size());

  std::cout<<"Begin to filter plain database..."<<std::endl;

  for (size_t i = 0; i < l_partkey.size(); ++i) {
    Tpres[i] = 0;
    Ppres[i] = 0;
    if (l_shipdate[i] >= date1 && l_shipdate[i] < date2) {
      Tpres[i] = 1;
      double revenue = l_extendedprice[i] * (1 - 0.01*l_discount[i]);
      total_revenue += revenue;
      if (l_type[i] >= type1 && l_type[i] < type2) {
        promo_revenue += revenue;
        Ppres[i] = 1;
      }
    }
  }

    double promo_revenue_percentage = 100.0 * promo_revenue / total_revenue;
    std::cout << "\nPlaintext query result:" << promo_revenue_percentage<<endl<< std::endl;
}

void Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem, std::vector<uint32_t> &pred_res, std::vector<uint32_t> &pred_res1) {

  std::vector<Lvl1::T> &l_extendedprice = plain_lineitem[3];
  std::vector<Lvl1::T> &l_discount = plain_lineitem[4];

  double promo_revenue = 0.0;
  double total_revenue = 0.0;

  std::cout<<"Begin to filter plain database..."<<std::endl;

  for (size_t i = 0; i < rows; ++i) {
    if (pred_res[i]==1) {
      double revenue = l_extendedprice[i] * (1 - 0.01*l_discount[i]);
      total_revenue += revenue;
      if (pred_res1[i]==1) {
        promo_revenue += revenue;
      }
    }
  }

    double promo_revenue_percentage = 100.0 * promo_revenue / total_revenue;
    std::cout << "\nCiphertext query result:" << promo_revenue_percentage<<endl<< std::endl;
}

void Filter_Cipher_d(std::vector<TLWELvl1> &pred_cres,
                     std::vector<TLWELvl1> &pred_cres1,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     std::vector<TLWE<Lvl1>> &cipher_type,
                     std::vector<TLWE<Lvl1>> &cipher_date, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<uint32_t> &pred_res,
                     std::vector<uint32_t> &pred_res1) {

  std::vector<TLWE<Lvl1>> &l_partkey = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_type = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_shipdate = cipher_lineitem[2];


  TLWE<Lvl1> &date1 = cipher_date[0];
  TLWE<Lvl1> &date2 = cipher_date[1];
  TLWE<Lvl1> &type1 = cipher_type[0];
  TLWE<Lvl1> &type2 = cipher_type[1];

  uint32_t lineitem_rows = l_partkey.size();

  std::vector<TLWELvl1> pred_cres2(lineitem_rows), pred_cres3(lineitem_rows),
      pred_cres4(lineitem_rows), pred_cres5(lineitem_rows);

  pred_cres.resize(lineitem_rows), pred_cres1.resize(lineitem_rows),

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for (size_t i = 0; i < lineitem_rows; ++i) {
    uint32_t stream_id = omp_get_thread_num();

	  cuHEDB::greater_than_equal<Lvl1>(l_shipdate[i], date1, pred_cres2[i], date_bits, ek, LOGIC, stream_id);
	  cuHEDB::less_than<Lvl1>(l_shipdate[i], date2, pred_cres3[i], date_bits, ek, LOGIC, stream_id);
	  cuHEDB::HomAND(pred_cres[i], pred_cres2[i], pred_cres3[i], ek, LOGIC, stream_id);
    cuHEDB::greater_than_equal<Lvl1>(l_type[i], type1, pred_cres4[i], key_bits, ek, LOGIC, stream_id);
    cuHEDB::HomAND(pred_cres1[i], pred_cres[i], pred_cres4[i], ek, LOGIC, stream_id);
    cuHEDB::less_than<Lvl1>(l_type[i], type2, pred_cres5[i], key_bits, ek, LOGIC, stream_id);
    cuHEDB::HomAND(pred_cres1[i], pred_cres1[i], pred_cres5[i], ek, LOGIC, stream_id);

    cudaDeviceSynchronize();
  }
    
  end = std::chrono::system_clock::now();
  double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout<<"Filter Time on GPU: "<< costs/1000 <<"ms"<< std::endl;

  uint32_t rlwe_scale_bits = 29;
  #pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    TFHEpp::log_rescale(pred_cres1[i], pred_cres1[i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(pred_cres[i], pred_cres[i], rlwe_scale_bits, ek);
  }

  std::vector<Lvl1::T> cpres;
    
    pred_res.resize(lineitem_rows);
    pred_res1.resize(lineitem_rows);
    for (size_t i = 0; i < lineitem_rows;++i) {
      pred_res[i]=TFHEpp::tlweSymInt32Decrypt<Lvl1>(pred_cres[i], pow(2., 29), sk.key.get<Lvl1>());
      pred_res1[i]=TFHEpp::tlweSymInt32Decrypt<Lvl1>(pred_cres1[i], pow(2., 29), sk.key.get<Lvl1>());
    }

}

void Filter_Cipher_h(std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     std::vector<TLWE<Lvl1>> &cipher_type,
                     std::vector<TLWE<Lvl1>> &cipher_date, TFHEEvalKey &ek,
                     TFHESecretKey &sk) {

  std::vector<TLWE<Lvl1>> &l_partkey = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_type = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_shipdate = cipher_lineitem[2];


  TLWE<Lvl1> &date1 = cipher_date[0];
  TLWE<Lvl1> &date2 = cipher_date[1];
  TLWE<Lvl1> &type1 = cipher_type[0];
  TLWE<Lvl1> &type2 = cipher_type[1];

  uint32_t lineitem_rows = l_partkey.size();

  std::vector<TLWELvl1> pred_cres(lineitem_rows), pred_cres1(lineitem_rows),
      pred_cres2(lineitem_rows), pred_cres3(lineitem_rows),
      pred_cres4(lineitem_rows), pred_cres5(lineitem_rows);


  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  for (size_t i = 0; i < lineitem_rows; ++i) {
	  greater_than_equal<Lvl1>(l_shipdate[i], date1, pred_cres2[i], date_bits, ek, LOGIC);
	  less_than<Lvl1>(l_shipdate[i], date2, pred_cres3[i], date_bits, ek, LOGIC);
	  HomAND(pred_cres[i], pred_cres2[i], pred_cres3[i], ek, LOGIC);
    greater_than_equal<Lvl1>(l_type[i], type1, pred_cres4[i], key_bits, ek, LOGIC);
    HomAND(pred_cres1[i], pred_cres[i], pred_cres4[i], ek, LOGIC);
    less_than<Lvl1>(l_type[i], type2, pred_cres5[i], key_bits, ek, LOGIC);
    HomAND(pred_cres1[i], pred_cres1[i], pred_cres5[i], ek, LOGIC);
  }
    
  end = std::chrono::system_clock::now();
  double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	
  std::cout<<"Filter Time on CPU: "<< costs/1000 <<"ms"<< std::endl;

}

void aggregation(std::vector<TLWELvl1> &pred_cres,
                 std::vector<TLWELvl1> &pred_cres1,
                 std::vector<uint32_t> &pred_res,
                 std::vector<uint32_t> &pred_res1, size_t tfhe_n,
                 std::vector<Lvl1::T> &extendedprice_data,
                 std::vector<Lvl1::T> &discount_data, size_t rows,
                 TFHESecretKey &sk) {
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
  parms.set_coeff_modulus(seal::CoeffModulus::Create(
      poly_modulus_degree, {59, 42, 42, 42, 42, 42, 42, 42, 45, 45,
                            45, 45, 45, 45, 45, 45, 45, 45, 45, 59}));
  double scale = std::pow(2.0, scale_bits);

  // context instance
  seal::SEALContext context(parms, true, seal::sec_level_type::none);

  // key generation
  seal::KeyGenerator keygen(context);
  seal::SecretKey seal_secret_key = keygen.secret_key();
  seal::RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  seal::GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  // utils
  seal::Encryptor encryptor(context, seal_secret_key);
  seal::Evaluator evaluator(context);
  seal::Decryptor decryptor(context, seal_secret_key);

  // encoder
  seal::CKKSEncoder ckks_encoder(context);

  // generate evaluation key
  std::cout << "Generating Conversion Key..." << std::endl;
  LTPreKey pre_key;
  LWEsToRLWEKeyGen(pre_key, std::pow(2., modulus_bits), seal_secret_key, sk,
                   tfhe_n, ckks_encoder, encryptor, context);

  // conversion
  std::cout << "Starting Conversion..." << std::endl;
  // repack pred_cres(LWEs) -> result(RLWE)
  seal::Ciphertext result, result1;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  LWEsToRLWE(result, pred_cres, pre_key, scale, std::pow(2., modq_bits),
             std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys,
             relin_keys, evaluator, context);
  HomRound(result, result.scale(), ckks_encoder, relin_keys, evaluator,
           decryptor, context);

  LWEsToRLWE(result1, pred_cres1, pre_key, scale, std::pow(2., modq_bits),
             std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys,
             relin_keys, evaluator, context);
  HomRound(result1, result1.scale(), ckks_encoder, relin_keys, evaluator,
           decryptor, context);
  end = std::chrono::system_clock::now();
  aggregation_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  seal::Plaintext plain;
  std::vector<double> computed(slots_count), computed1(slots_count);
  decryptor.decrypt(result, plain);
  seal::pack_decode(computed, plain, ckks_encoder);
  decryptor.decrypt(result1, plain);
  seal::pack_decode(computed1, plain, ckks_encoder);

  double err1 = 0., err2 = 0.;

  for (size_t i = 0; i < slots_count; ++i) {
    err1 += std::abs(computed[i] - pred_res[i]);
    err2 += std::abs(computed1[i] - pred_res1[i]);
  }

  printf("Repack average error1 = %f ~ 2^%.1f\n", err1 / slots_count,
         std::log2(err1 / slots_count));
  printf("Repack average error2 = %f ~ 2^%.1f\n", err2 / slots_count,
         std::log2(err2 / slots_count));

  // Filter result * data
  std::vector<double> price_discount(extendedprice_data.size());
  seal::Ciphertext price_discount_cipher;
  for (size_t i = 0; i < rows; i++) {
    price_discount[i] = extendedprice_data[i] * (1-discount_data[i]* 0.01) ;
  }
  double qd = parms.coeff_modulus()[result.coeff_modulus_size() - 1].value();
  seal::pack_encode(price_discount, qd, plain, ckks_encoder);
  encryptor.encrypt_symmetric(plain, price_discount_cipher);

  std::cout << "Aggregating price and discount .." << std::endl;
  start = std::chrono::system_clock::now();
  // result * price_discount_cipher
  seal::multiply_and_relinearize(result, price_discount_cipher, result, evaluator, relin_keys);
  seal::multiply_and_relinearize(result1, price_discount_cipher, result1, evaluator, relin_keys);
  evaluator.rescale_to_next_inplace(result);
  evaluator.rescale_to_next_inplace(result1);
  int logrow = log2(rows);

  seal::Ciphertext temp;
  for (size_t i = 0; i < logrow; i++) {
    temp = result;
    size_t step = 1 << (logrow - i - 1);
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result, temp);

    temp = result1;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result1, temp);
  }
  end = std::chrono::system_clock::now();
  aggregation_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::vector<double> agg_result(slots_count), agg_result1(slots_count);
  decryptor.decrypt(result, plain);
  seal::pack_decode(agg_result, plain, ckks_encoder);

  decryptor.decrypt(result1, plain);
  seal::pack_decode(agg_result1, plain, ckks_encoder);

  cout << "\nEncrypted aggregation result: " << 100.*agg_result1[0]/agg_result[0] << endl;
  std::cout << "\nAggregation time: " << aggregation_time << "ms" << std::endl;
}

int main() {
  omp_set_num_threads(num_stream);
  warmupGPU();
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
  // writeToFile(path, sk.key.lvl1, *ek.bkfftlvl01, *ek.iksklvl10);
  readFromFile(path, sk.key.lvl1, *bkfft, *isk);
  ek.bkfftlvl01.reset(bkfft);
  ek.iksklvl10.reset(isk);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "generate Secret Key Time: " << costs/1000 << "ms" <<std::endl;

  std::cout << "Loading" << std::endl;
  start = std::chrono::system_clock::now();

  //load BK to device
  cufftplvl.LoadBK<lvl1param>(*ek.bkfftlvl01);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<"Load Success."<<std::endl;
  std::cout << "Loading Time: " << costs/1000 << "ms" << std::endl;


  // DataBase init
  std::vector<std::vector<Lvl1::T>> plain_lineitem;
  std::vector<Lvl1::T> plain_date{102, 430}, plain_type{3, 7};
  std::vector<std::vector<TLWE<Lvl1>>> cipher_lineitem;
  std::vector<TLWE<Lvl1>> cipher_date, cipher_type;
  std::vector<Lvl1::T>  Ppres, Tpres;
  std::vector<TLWE<Lvl1>> cres1, cres;
  std::vector<Lvl1::T> pres, pres1;

  gen_DataBase(plain_lineitem, rows);

  encrypt_DataBase(plain_lineitem, cipher_lineitem, sk);

  encrypt_Condition(plain_type, plain_date, cipher_type, cipher_date, sk);

  Plain_Query(plain_lineitem, plain_type, plain_date, Tpres, Ppres);

  Filter_Cipher_d(cres, cres1, cipher_lineitem, cipher_type, cipher_date, ek, sk, pres, pres1);

  Query(plain_lineitem, pres, pres1);

#if 0
  std::vector<Lvl1::T> &discounts = plain_lineitem[4];
  std::vector<Lvl1::T> &prices = plain_lineitem[3];

  aggregation(cres, cres1, pres, pres1, Lvl1::n, prices, discounts, rows, sk);

  Filter_Cipher_h(cipher_lineitem, cipher_type, cipher_date, ek, sk);
#endif
  return 0;
}
