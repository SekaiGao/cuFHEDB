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
#include <unordered_map>
#include <vector>

using namespace HEDB;
using namespace TFHEpp;

const int rows = 1 << 18; // Number of plaintexts

/***
select
o_orderpriority, count(*) as order_count
from orders
where
o_orderdate >= date '[DATE]'
and o_orderdate < date '[DATE]' + interval '3' month
and exists (
select * from lineitem
where
l_orderkey = o_orderkey
and l_commitdate < l_receiptdate
)
group by 
o_orderpriority
order by 
o_orderpriority;

consider joined table

orderpriority \in ('0-URGENT', '1-HIGH', '2-MEDIUM', '3-LOW')
*/

const uint32_t orderkey_bits = 8;
const uint32_t orderpriority_bits = 2;
const uint32_t date_bits = 9;


//order(orderkey(8), orderpriority(2), orderdate(9), commitdate(9), receiptdate(9))
void gen_DataBase(std::vector<std::vector<Lvl1::T>> &plain_order, int order_rows) {


    std::random_device seed_gen;
    std::default_random_engine generator(seed_gen());
    std::uniform_int_distribution<Lvl1::T> orderkey_distribution(0, (1 << orderkey_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> orderpriority_distribution(0, (1 << orderpriority_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> date_distribution(0, (1 << date_bits) - 1);

    std::vector<Lvl1::T> orderkeys, orderpriorities, commitdates, receiptdates, orderdates;

    std::cout<<"Begin to generate database..."<<std::endl;
    for (int i = 0; i < order_rows; ++i) {
        Lvl1::T orderkey = orderkey_distribution(generator);
        Lvl1::T orderpriority = orderpriority_distribution(generator);
        Lvl1::T orderdate = date_distribution(generator);
        Lvl1::T commitdate = date_distribution(generator);
        Lvl1::T receiptdate = date_distribution(generator);

        orderkeys.emplace_back(orderkey);
        orderpriorities.emplace_back(orderpriority);
        orderdates.emplace_back(orderdate);
        commitdates.emplace_back(commitdate);
        receiptdates.emplace_back(receiptdate);
    }

    plain_order.emplace_back(orderkeys);
    plain_order.emplace_back(orderpriorities);
    plain_order.emplace_back(orderdates);
    plain_order.emplace_back(commitdates);
    plain_order.emplace_back(receiptdates);

    std::cout<<"Successfully generated database."<<std::endl;
}

void encrypt_DataBase(std::vector<std::vector<Lvl1::T>> &plain_order,
                      std::vector<std::vector<TLWE<Lvl1>>> &cipher_order,
                      TFHESecretKey &sk) {

  uint32_t scale_bits_order[5];
  scale_bits_order[0] = std::numeric_limits<Lvl1::T>::digits - orderkey_bits - 1;
  scale_bits_order[1] = std::numeric_limits<Lvl1::T>::digits - orderpriority_bits - 1;
  scale_bits_order[2] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
  scale_bits_order[3] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
  scale_bits_order[4] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;

  std::cout << "Begin to encrypt database..." << std::endl;

  cipher_order.resize(plain_order.size());
  for (int i = 0; i < plain_order.size(); ++i) {
    for (int j = 0; j < plain_order[i].size(); ++j) {
        cipher_order[i].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_order[i][j], Lvl1::α, pow(2., scale_bits_order[i]), sk.key.get<Lvl1>()));
      }
  }

  std::cout << "Success encrypt database." << std::endl;
}

void encrypt_Condition(std::vector<Lvl1::T> &plain_date,std::vector<TLWE<Lvl1>> &cipher_date, std::vector<TLWE<Lvl1>> &cipher_num, TFHESecretKey &sk) {
 
	uint32_t date_scale_bits = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
  uint32_t orderpriority_scale_bits = std::numeric_limits<Lvl1::T>::digits - orderpriority_bits - 1;
	
  cipher_date.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date[0], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	cipher_date.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date[1], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));

  Lvl1::T plain_num[4] = {0, 1, 2, 3};

  for(int i = 0; i < 4; ++i){
    cipher_num.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_num[i], Lvl1::α, pow(2., orderpriority_scale_bits), sk.key.get<Lvl1>()));
  }
}

void Plain_Query(std::vector<std::vector<Lvl1::T>> &plain_order, std::vector<Lvl1::T> &filter_date) {
    std::vector<Lvl1::T> &o_orderkey = plain_order[0];
    std::vector<Lvl1::T> &o_orderpriority = plain_order[1];
    std::vector<Lvl1::T> &o_orderdate = plain_order[2];
    std::vector<Lvl1::T> &o_commitdate = plain_order[3];
    std::vector<Lvl1::T> &o_receiptdate = plain_order[4];
    
    Lvl1::T date1 = filter_date[0];
    Lvl1::T date2 = filter_date[1];

    std::vector<std::vector<Lvl1::T>>res(4);
    for(int i=0;i<4;++i){
      res[i].resize(rows);
      res[i].assign(rows, 0);
    }

    std::cout<<"Begin to filter plain database."<<std::endl;
    std::map<Lvl1::T, Lvl1::T> order_count;

    for (size_t i = 0; i < o_orderkey.size(); ++i) {
      if (o_orderdate[i] >= date1 && o_orderdate[i] < date2 &&
          o_commitdate[i] < o_receiptdate[i]) {
        order_count[o_orderpriority[i]]++;
        
      }
      res[o_orderpriority[i]][i] = 1;
    }


    string op[4] = {"URGENT", "HIGH", "MEDIUM", "LOW"};
    std::cout << "\nPlaintext query result:" << std::endl;
    cout << "order_priority, count" << std::endl;
    for (int i=0;i<4;++i) {
      cout << op[i] << ", " << order_count[i] << std::endl;
    }

    cout<<endl;
}

void Query(std::vector<std::vector<uint32_t>> &pred_res) {

    std::map<Lvl1::T, Lvl1::T> order_count;

    for (size_t i = 0; i < rows; ++i) {
      if(pred_res[0][i]==1) {
        order_count[0]++;}
      else if(pred_res[1][i]==1) {
        order_count[1]++; }
      else if(pred_res[2][i]==1) {
        order_count[2]++; }
      else if(pred_res[3][i]==1){
        order_count[3]++; }
    }


    string op[4] = {"URGENT", "HIGH", "MEDIUM", "LOW"};
    std::cout << "\nCiphertext query result:" << std::endl;
    cout << "order_priority, count" << std::endl;
    for (int i=0;i<4;++i) {
      cout << op[i] << ", " << order_count[i] << std::endl;
    }

    cout<<endl;
}

void Filter_Cipher_d(std::vector<std::vector<TLWE<Lvl1>>> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_order,
                     std::vector<TLWE<Lvl1>> &cipher_date,
                     std::vector<TLWE<Lvl1>> &cipher_num, TFHEEvalKey &ek,
                     TFHESecretKey &sk,
                     std::vector<std::vector<Lvl1::T>> & pred_res) {
  std::vector<TLWE<Lvl1>> &o_orderkey = cipher_order[0];
  std::vector<TLWE<Lvl1>> &o_orderpriority = cipher_order[1];
  std::vector<TLWE<Lvl1>> &o_orderdate = cipher_order[2];
  std::vector<TLWE<Lvl1>> &o_commitdate = cipher_order[3];
  std::vector<TLWE<Lvl1>> &o_receiptdate = cipher_order[4];

  TLWE<Lvl1> &date1 = cipher_date[0];
  TLWE<Lvl1> &date2 = cipher_date[1];

  uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - 1 - 1;

  std::vector<TLWELvl1> pred_cres(rows), pred_cres1(rows), pred_cres2(rows),
      pred_cres3(rows), pred_cres4(rows), pred_cres5(rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  cres.resize(4);
  for(int i=0;i<4;++i){
    cres[i].resize(rows);
  }

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for (size_t i = 0; i < rows; ++i) {
    uint32_t stream_id = omp_get_thread_num();

    cuHEDB::equal<Lvl1>(o_orderpriority[i], cipher_num[0], cres[0][i], orderpriority_bits, ek, LOGIC, stream_id);
    cuHEDB::equal<Lvl1>(o_orderpriority[i], cipher_num[1], cres[1][i], orderpriority_bits, ek, LOGIC, stream_id);
    cuHEDB::equal<Lvl1>(o_orderpriority[i], cipher_num[2], cres[2][i], orderpriority_bits, ek, LOGIC, stream_id);
    cuHEDB::equal<Lvl1>(o_orderpriority[i], cipher_num[3], cres[3][i], orderpriority_bits, ek, LOGIC, stream_id);

    cuHEDB::greater_than_equal<Lvl1>(o_orderdate[i], date1, pred_cres1[i], date_bits, ek, LOGIC, stream_id);
	  cuHEDB::less_than_equal<Lvl1>(o_orderdate[i], date2, pred_cres2[i], date_bits, ek, LOGIC, stream_id);
	  cuHEDB::HomAND(pred_cres1[i], pred_cres1[i], pred_cres2[i], ek, LOGIC, stream_id);
    cuHEDB::less_than<Lvl1>(o_commitdate[i], o_receiptdate[i], pred_cres[i], date_bits, ek, LOGIC, stream_id);
    cuHEDB::HomAND(pred_cres[i], pred_cres[i], pred_cres1[i], ek, LOGIC, stream_id);

    cuHEDB::HomAND(cres[0][i], cres[0][i], pred_cres[i], ek, LOGIC, stream_id);
    cuHEDB::HomAND(cres[1][i], cres[1][i], pred_cres[i], ek, LOGIC, stream_id);
    cuHEDB::HomAND(cres[2][i], cres[2][i], pred_cres[i], ek, LOGIC, stream_id);
    cuHEDB::HomAND(cres[3][i], cres[3][i], pred_cres[i], ek, LOGIC, stream_id);

    cudaDeviceSynchronize();
  }
    

	end = std::chrono::system_clock::now();
  double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	
	std::cout<<"Filter Time on GPU: "<< costs/1000 <<"ms"<< std::endl;

  uint32_t rlwe_scale_bits = 29;
  #pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    TFHEpp::log_rescale(cres[0][i], cres[0][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[1][i], cres[1][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[2][i], cres[2][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[3][i], cres[3][i], rlwe_scale_bits, ek);
  }

  std::vector<Lvl1::T> cpres;


  pred_res.resize(4);
  pred_res[0].resize(rows);
  pred_res[1].resize(rows);
  pred_res[2].resize(rows);
  pred_res[3].resize(rows);

  for (size_t i = 0; i < rows; ++i) {
    pred_res[0][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[0][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[1][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[1][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[2][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[2][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[3][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[3][i], pow(2., 29),sk.key.get<Lvl1>());
  }

}

void Filter_Cipher_h(std::vector<std::vector<TLWE<Lvl1>>> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_order,
                     std::vector<TLWE<Lvl1>> &cipher_date,
                     std::vector<TLWE<Lvl1>> &cipher_num, TFHEEvalKey &ek,
                     TFHESecretKey &sk,
                     std::vector<std::vector<Lvl1::T>> & pred_res) {
  std::vector<TLWE<Lvl1>> &o_orderkey = cipher_order[0];
  std::vector<TLWE<Lvl1>> &o_orderpriority = cipher_order[1];
  std::vector<TLWE<Lvl1>> &o_orderdate = cipher_order[2];
  std::vector<TLWE<Lvl1>> &o_commitdate = cipher_order[3];
  std::vector<TLWE<Lvl1>> &o_receiptdate = cipher_order[4];

  TLWE<Lvl1> &date1 = cipher_date[0];
  TLWE<Lvl1> &date2 = cipher_date[1];

  uint32_t scale_bits = std::numeric_limits<Lvl1::T>::digits - 1 - 1;

  std::vector<TLWELvl1> pred_cres(rows), pred_cres1(rows), pred_cres2(rows),
      pred_cres3(rows), pred_cres4(rows), pred_cres5(rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  cres.resize(4);
  for(int i=0;i<4;++i){
    cres[i].resize(rows);
  }

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  for (size_t i = 0; i < rows; ++i) {
    equal<Lvl1>(o_orderpriority[i], cipher_num[0], cres[0][i], orderpriority_bits, ek, LOGIC);
    equal<Lvl1>(o_orderpriority[i], cipher_num[1], cres[1][i], orderpriority_bits, ek, LOGIC);
    equal<Lvl1>(o_orderpriority[i], cipher_num[2], cres[2][i], orderpriority_bits, ek, LOGIC);
    equal<Lvl1>(o_orderpriority[i], cipher_num[3], cres[3][i], orderpriority_bits, ek, LOGIC);

    greater_than_equal<Lvl1>(o_orderdate[i], date1, pred_cres1[i], date_bits, ek, LOGIC);
	  less_than_equal<Lvl1>(o_orderdate[i], date2, pred_cres2[i], date_bits, ek, LOGIC);
	  HomAND(pred_cres1[i], pred_cres1[i], pred_cres2[i], ek, LOGIC);
    less_than<Lvl1>(o_commitdate[i], o_receiptdate[i], pred_cres[i], date_bits, ek, LOGIC);
    HomAND(pred_cres[i], pred_cres[i], pred_cres1[i], ek, LOGIC);

    HomAND(cres[0][i], cres[0][i], pred_cres[i], ek, LOGIC);
    HomAND(cres[1][i], cres[1][i], pred_cres[i], ek, LOGIC);
    HomAND(cres[2][i], cres[2][i], pred_cres[i], ek, LOGIC);
    HomAND(cres[3][i], cres[3][i], pred_cres[i], ek, LOGIC);
  }
    

	end = std::chrono::system_clock::now();
  	double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	
	std::cout<<"Filter Time on CPU: "<< costs/1000 <<"ms"<< std::endl;
#if 0
  uint32_t rlwe_scale_bits = 29;
  for (size_t i = 0; i < rows; i++) {
    TFHEpp::log_rescale(cres[0][i], cres[0][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[1][i], cres[1][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[2][i], cres[2][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[3][i], cres[3][i], rlwe_scale_bits, ek);
  }

  std::vector<Lvl1::T> cpres;


  pred_res.resize(4);
  pred_res[0].resize(rows);
  pred_res[1].resize(rows);
  pred_res[2].resize(rows);
  pred_res[3].resize(rows);

  for (size_t i = 0; i < rows; ++i) {
    pred_res[0][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[0][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[1][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[1][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[2][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[2][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[3][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[3][i], pow(2., 29),sk.key.get<Lvl1>());
  }
#endif
}

void aggregation(std::vector<std::vector<TLWELvl1>> &pred_cres,
                 std::vector<std::vector<uint32_t>> &pred_res,
                 size_t tfhe_n, size_t rows,
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
  seal::Ciphertext result0, result1, result2, result3;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  
  LWEsToRLWE(result0, pred_cres[0], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result0,   result0.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  LWEsToRLWE(result1, pred_cres[1], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result1, result1.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  LWEsToRLWE(result2, pred_cres[2], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result2, result2.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  LWEsToRLWE( result3, pred_cres[3], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound( result3,  result3.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  end = std::chrono::system_clock::now();
  aggregation_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  seal::Plaintext plain;
  std::vector<std::vector<double>> computed(4);
  for(int i=0;i<4;++i){
    computed[i].resize(slots_count);
  }
  decryptor.decrypt(result0, plain);
  seal::pack_decode(computed[0], plain, ckks_encoder);

  decryptor.decrypt(result1, plain);
  seal::pack_decode(computed[1], plain, ckks_encoder);

  decryptor.decrypt(result2, plain);
  seal::pack_decode(computed[2], plain, ckks_encoder);

  decryptor.decrypt(result3, plain);
  seal::pack_decode(computed[3], plain, ckks_encoder);

  double err[4] = {0.,0.,0.,0.};
  for(int j=0;j<4;++j)
  for (size_t i = 0; i < slots_count; ++i) {
    err[j] += std::abs(computed[j][i] - pred_res[j][i]);
  }
  for (int j = 0; j < 4; ++j)
    printf("Repack average error%d = %f ~ 2^%.1f\n",j, err[j] / slots_count,
           std::log2(err[j] / slots_count));


  std::cout << "Aggregating price and discount .." << std::endl;
  start = std::chrono::system_clock::now();
  

  int logrow = log2(rows);

  seal::Ciphertext temp;
  for (size_t i = 0; i < logrow; i++) {
	size_t step = 1 << (logrow - i - 1);
    temp = result0;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result0, temp);

    temp = result1;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result1, temp);

    temp = result2;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result2, temp);

    temp = result3;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result3, temp);
  }

  end = std::chrono::system_clock::now();
  aggregation_time +=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::vector<std::vector<double>> agg_result(4);
  for (int j = 0; j < 4; ++j) {
	agg_result[j].resize(slots_count);
  }
  decryptor.decrypt( result0, plain);
  seal::pack_decode(agg_result[0], plain, ckks_encoder);
  decryptor.decrypt( result1, plain);
  seal::pack_decode(agg_result[1], plain, ckks_encoder);
  decryptor.decrypt( result2, plain);
  seal::pack_decode(agg_result[2], plain, ckks_encoder);
  decryptor.decrypt( result3, plain);
  seal::pack_decode(agg_result[3], plain, ckks_encoder);

  cout << "\nEncrypted aggregation result1: " << endl;

  cout << "order_priority,\t count" << std::endl;

  cout << "URGENT" << ",\t " << abs(std::round(agg_result[0][0])) << std::endl;
  cout << "HIGH" << ",\t " << abs(std::round(agg_result[1][0])) << std::endl;
  cout << "MEDIUM" << ",\t " << abs(std::round(agg_result[2][0])) << std::endl;
  cout << "LOW" << ",\t " << abs(std::round(agg_result[3][0])) << std::endl;

#if 0
  std::vector<TLWELvl1> crst(4);
  std::vector<uint32_t> rst(4);

  for (size_t i = 0; i < rows; i++)
    {
        if (i == 0)
        {
            crst[0] = pred_cres[0][i];
            crst[1] = pred_cres[1][i];
            crst[2] = pred_cres[2][i];
            crst[3] = pred_cres[3][i];
        }
        else
        {
            for (size_t j = 0; j <= Lvl1::n; j++)
            {
              crst[0][j] += pred_cres[0][i][j];
              crst[1][j] += pred_cres[1][i][j];
              crst[2][j] += pred_cres[2][i][j];
              crst[3][j] += pred_cres[3][i][j];
            }
        }
    }

  for(int i=0;i<4;++i){
	rst[i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(crst[i], pow(2., scale_bits), sk.key.lvl1);
  }
  
  cout << "\nEncrypted aggregation result2: " << endl;

  cout << "order_priority,\t count" << std::endl;

  cout << "URGENT" << ",\t " <<rst[0] << std::endl;
  cout << "HIGH" << ",\t " << rst[1] << std::endl;
  cout << "MEDIUM" << ",\t " << rst[2] << std::endl;
  cout << "LOW" << ",\t " << rst[3] << std::endl;
#endif

  std::cout << "\nAggregation time: " << aggregation_time << "ms" << std::endl;


}

int main() {

  omp_set_num_threads(num_stream);

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

  warmupGPU();

  std::cout << "Loading" << std::endl;
  start = std::chrono::system_clock::now();

  //load BK to device
  cufftplvl.LoadBK<lvl1param>(*ek.bkfftlvl01);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<"Load Success."<<std::endl;
  std::cout << "Loading Time: " << costs/1000 << "ms" << std::endl;


  // DataBase init
  std::vector<std::vector<Lvl1::T>> plain_order;
  std::vector<Lvl1::T> plain_date{102, 430};
  std::vector<std::vector<TLWE<Lvl1>>> cipher_order;
  std::vector<TLWE<Lvl1>> cipher_date, cipher_num;
  std::vector<std::vector<TLWE<Lvl1>>> cres;
  std::vector<std::vector<Lvl1::T>> pres;

  gen_DataBase(plain_order, rows);

  encrypt_DataBase(plain_order, cipher_order, sk);

  encrypt_Condition(plain_date, cipher_date, cipher_num, sk);

  Plain_Query(plain_order, plain_date);

  Filter_Cipher_d(cres, cipher_order, cipher_date, cipher_num, ek, sk, pres);

  Query(pres);

#if 0
  aggregation(cres, pres, Lvl1::n, rows, sk);

  Filter_Cipher_h(cres, cipher_order, cipher_date, cipher_num, ek, sk, pres);
#endif
  return 0;
}
