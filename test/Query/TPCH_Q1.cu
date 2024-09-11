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

const int rows = 1<<10; // Number of plaintexts

/***
TPC-H Query 1
    select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
    from
        lineitem
    where
        l_shipdate <= date '1998-12-01' - interval ':1' day (3)
    group by
        l_returnflag,
        l_linestatus
    order by
        l_returnflag,
        l_linestatus;
*/

const uint32_t bin_bits = 1;
const uint32_t quantity_bits = 8;
const uint32_t date_bits = 9;

//Generate database lineitem(returnflag(1), linestatus(1), shipdate(9), quantity(8))
void gen_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem, int lineitem_rows) {

    std::random_device seed_gen;
    std::default_random_engine generator(seed_gen());
    std::uniform_int_distribution<Lvl1::T> bin_distribution(0, (1 << bin_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> quantity_distribution(0, (1 << quantity_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> date_distribution(0, (1 << date_bits) - 1);

    std::vector<Lvl1::T> returnflags, linestatuses, shipdates, quantities;

    std::cout << "Begin to generate database..." << std::endl;

    for (int i = 0; i < lineitem_rows; ++i) {
        Lvl1::T returnflag = bin_distribution(generator);
        Lvl1::T linestatus = bin_distribution(generator);
        Lvl1::T shipdate = date_distribution(generator);
        Lvl1::T quantity = quantity_distribution(generator);

        returnflags.emplace_back(returnflag);
        linestatuses.emplace_back(linestatus);
        shipdates.emplace_back(shipdate);
        quantities.emplace_back(quantity);
    }

    plain_lineitem.emplace_back(returnflags);
    plain_lineitem.emplace_back(linestatuses);
    plain_lineitem.emplace_back(shipdates);
    plain_lineitem.emplace_back(quantities);

    std::cout<<"Successfully generated database."<<std::endl;
}

//Encrypt database
void encrypt_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem,
                      std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                      TFHESecretKey &sk) {

  uint32_t scale_bits_lineitem[4];
  scale_bits_lineitem[0] = std::numeric_limits<Lvl1::T>::digits - bin_bits - 1;
  scale_bits_lineitem[1] = std::numeric_limits<Lvl1::T>::digits - bin_bits - 1;
  scale_bits_lineitem[2] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
  scale_bits_lineitem[3] = std::numeric_limits<Lvl1::T>::digits - quantity_bits - 1;

  std::cout << "Begin to encrypt database..." << std::endl;

  cipher_lineitem.resize(4);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < plain_lineitem[i].size(); ++j) {
      cipher_lineitem[i].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_lineitem[i][j], Lvl1::α, pow(2., scale_bits_lineitem[i]), sk.key.get<Lvl1>()));
      }
  }

  std::cout << "Successfully encrypted database." << std::endl;
}

//Encrypt filter condition
void encrypt_Condition(Lvl1::T &plain_date, TLWE<Lvl1> &cipher_date, std::vector<TLWE<Lvl1>> &cipher_bool, TFHESecretKey &sk) {
 
	uint32_t date_scale_bits = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
	uint32_t bin_scale_bits = std::numeric_limits<Lvl1::T>::digits - bin_bits - 1;
	
  cipher_date=TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date, Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>());
	cipher_bool.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(0, Lvl1::α, pow(2., bin_scale_bits), sk.key.get<Lvl1>()));
	cipher_bool.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(1, Lvl1::α, pow(2., bin_scale_bits), sk.key.get<Lvl1>()));
}


// query over plaintext
void Plain_Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem,
              Lvl1::T &filter_date) {

    std::vector<Lvl1::T> &l_returnflag = plain_lineitem[0];
    std::vector<Lvl1::T> &l_linestatus = plain_lineitem[1];
    std::vector<Lvl1::T> &l_shipdate = plain_lineitem[2];
    std::vector<Lvl1::T> &l_quantity = plain_lineitem[3];

    Lvl1::T date1 = filter_date;

    Lvl1::T sum_qty_0_0 = 0;
    Lvl1::T sum_qty_0_1 = 0;
    Lvl1::T sum_qty_1_0 = 0;
    Lvl1::T sum_qty_1_1 = 0;

    std::cout<<"Begin to filter plain database..."<<std::endl;
    vector<vector<Lvl1::T>>pred(4);
    for(int i = 0; i<4;++i){
      pred[i].resize(rows);
      std::fill(pred[i].begin(), pred[i].end(), 0);
    }
    for (size_t i = 0; i < l_shipdate.size(); ++i) {
        if (l_shipdate[i] <= date1) {
            if (l_returnflag[i] == 0 && l_linestatus[i] == 0) {
                sum_qty_0_0 += l_quantity[i];
                pred[0][i] = 1;
            } else if (l_returnflag[i] == 0 && l_linestatus[i] == 1) {
                sum_qty_0_1 += l_quantity[i];
                pred[1][i] = 1;
            } else if (l_returnflag[i] == 1 && l_linestatus[i] == 0) {
                sum_qty_1_0 += l_quantity[i];
                pred[2][i] = 1;
            } else if (l_returnflag[i] == 1 && l_linestatus[i] == 1) {
                sum_qty_1_1 += l_quantity[i];
                pred[3][i] = 1;
            }
        }
    }
	std::vector<Lvl1::T> result_returnflag, result_linestatus, result_sum_qty;

    result_returnflag = {0, 0, 1, 1};
    result_linestatus = {0, 1, 0, 1};
    result_sum_qty = {sum_qty_0_0, sum_qty_0_1, sum_qty_1_0, sum_qty_1_1};

    std::cout<<"\nPlaintext query result:"<<std::endl;

    cout << "returnflag,\t linestatus,\t sum_qty" << std::endl;
    
    for (size_t i = 0; i < result_returnflag.size(); ++i) {
		cout << result_returnflag[i] << ",\t " << result_linestatus[i] << ",\t " << result_sum_qty[i] << std::endl;
    }

    // cout<<"plain:"<<endl;
    // for(int i=0;i<rows;++i){
    //   cout<<pred[0][i]<<'\t'<<pred[1][i]<<'\t'<<pred[2][i]<<'\t'<<pred[3][i]<<'\t'<<endl;
    // }

    cout<<std::endl;
}

void Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem, std::vector<std::vector<uint32_t>> &pred_res) {

    std::vector<Lvl1::T> &l_quantity = plain_lineitem[3];

    Lvl1::T sum_qty_0_0 = 0;
    Lvl1::T sum_qty_0_1 = 0;
    Lvl1::T sum_qty_1_0 = 0;
    Lvl1::T sum_qty_1_1 = 0;
    for (size_t i = 0; i < l_quantity.size(); ++i) {
            if (pred_res[0][i] == 1) {
                sum_qty_0_0 += l_quantity[i];
            } else if (pred_res[1][i] == 1) {
                sum_qty_0_1 += l_quantity[i];
            } else if (pred_res[2][i] == 1) {
                sum_qty_1_0 += l_quantity[i];
            } else if (pred_res[3][i] == 1) {
                sum_qty_1_1 += l_quantity[i];
            }
    }
	std::vector<Lvl1::T> result_returnflag, result_linestatus, result_sum_qty;

    result_returnflag = {0, 0, 1, 1};
    result_linestatus = {0, 1, 0, 1};
    result_sum_qty = {sum_qty_0_0, sum_qty_0_1, sum_qty_1_0, sum_qty_1_1};

    std::cout<<"\nCiphertext query result:"<<std::endl;

    cout << "returnflag,\t linestatus,\t sum_qty" << std::endl;
    
    for (size_t i = 0; i < result_returnflag.size(); ++i) {
		cout << result_returnflag[i] << ",\t " << result_linestatus[i] << ",\t " << result_sum_qty[i] << std::endl;
    }

    cout<<std::endl;
}

// Ciphertext filtering on the gpu
void Filter_Cipher_d(std::vector<std::vector<TLWELvl1>> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     TLWE<Lvl1> &cipher_date,
                     std::vector<TLWE<Lvl1>> &cipher_bool, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<std::vector<uint32_t>> &pred_res) {

  std::vector<TLWE<Lvl1>> &l_returnflag = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_linestatus = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_shipdate = cipher_lineitem[2];
  std::vector<TLWE<Lvl1>> &l_quantity = cipher_lineitem[3];

  TLWE<Lvl1> &date1 = cipher_date;
  TLWE<Lvl1> &cipher0 = cipher_bool[0];
  TLWE<Lvl1> &cipher1 = cipher_bool[1];

  uint32_t lineitem_rows = l_returnflag.size();
  cres.resize(4);
  cres[0].resize(lineitem_rows);
  cres[1].resize(lineitem_rows);
  cres[2].resize(lineitem_rows);
  cres[3].resize(lineitem_rows);

  std::vector<TLWELvl1> pred_cres1(lineitem_rows),
      pred_cres2(lineitem_rows), pred_cres3(lineitem_rows),
      pred_cres4(lineitem_rows), pred_cres5(lineitem_rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  #pragma omp parallel for //num_threads(18)
  for (size_t i = 0; i < lineitem_rows; ++i) {
    uint32_t stream_id = omp_get_thread_num();

    cuHEDB::less_than_equal<Lvl1>(l_shipdate[i], date1, pred_cres1[i], date_bits, ek, LOGIC, stream_id);
	  cuHEDB::less_than<Lvl1>(l_returnflag[i], cipher1, pred_cres2[i], bin_bits, ek, LOGIC, stream_id);
	  cuHEDB::less_than<Lvl1>(l_linestatus[i], cipher1, pred_cres3[i], bin_bits, ek, LOGIC, stream_id);
	  cuHEDB::HomNOT<Lvl1>(pred_cres4[i], pred_cres2[i]);
	  cuHEDB::HomNOT<Lvl1>(pred_cres5[i], pred_cres3[i]);

    cuHEDB::HomAND(cres[0][i], pred_cres1[i], pred_cres2[i], ek, LOGIC, stream_id);
	  cuHEDB::HomAND(cres[1][i], cres[0][i], pred_cres5[i], ek, LOGIC, stream_id); // 01
    cuHEDB::HomAND(cres[0][i], cres[0][i], pred_cres3[i], ek, LOGIC, stream_id); // 00

    cuHEDB::HomAND(cres[2][i], pred_cres1[i], pred_cres4[i], ek, LOGIC, stream_id);
    cuHEDB::HomAND(cres[3][i], cres[2][i], pred_cres5[i], ek, LOGIC, stream_id); // 11
    cuHEDB::HomAND(cres[2][i], cres[2][i], pred_cres3[i], ek, LOGIC, stream_id); // 10

    cudaDeviceSynchronize();
  }

  end = std::chrono::system_clock::now();
  double costs =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Filter Time on GPU: " << costs / 1000 << "ms" << std::endl;

  uint32_t rlwe_scale_bits = 29;
  #pragma omp parallel for 
  for (size_t i = 0; i < lineitem_rows; i++) {
    TFHEpp::log_rescale(cres[0][i], cres[0][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[1][i], cres[1][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[2][i], cres[2][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[3][i], cres[3][i], rlwe_scale_bits, ek);
  }

  
  pred_res.resize(4);
  pred_res[0].resize(lineitem_rows);
  pred_res[1].resize(lineitem_rows);
  pred_res[2].resize(lineitem_rows);
  pred_res[3].resize(lineitem_rows);

  for (size_t i = 0; i < lineitem_rows; ++i) {
    pred_res[0][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[0][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[1][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[1][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[2][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[2][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[3][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[3][i], pow(2., 29),sk.key.get<Lvl1>());
  }

}

// Ciphertext filtering on the cpu
void Filter_Cipher_h(std::vector<std::vector<TLWELvl1>> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     TLWE<Lvl1> &cipher_date,
                     std::vector<TLWE<Lvl1>> &cipher_bool, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<std::vector<uint32_t>> &pred_res) {

  std::vector<TLWE<Lvl1>> &l_returnflag = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_linestatus = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_shipdate = cipher_lineitem[2];
  std::vector<TLWE<Lvl1>> &l_quantity = cipher_lineitem[3];

  TLWE<Lvl1> &date1 = cipher_date;
  TLWE<Lvl1> &cipher0 = cipher_bool[0];
  TLWE<Lvl1> &cipher1 = cipher_bool[1];

  uint32_t lineitem_rows = l_returnflag.size();
  cres.resize(4);
  cres[0].resize(lineitem_rows);
  cres[1].resize(lineitem_rows);
  cres[2].resize(lineitem_rows);
  cres[3].resize(lineitem_rows);

  std::vector<TLWELvl1> pred_cres1(lineitem_rows),
      pred_cres2(lineitem_rows), pred_cres3(lineitem_rows),
      pred_cres4(lineitem_rows), pred_cres5(lineitem_rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  for (size_t i = 0; i < lineitem_rows; ++i) {
   less_than_equal<Lvl1>(l_shipdate[i], date1, pred_cres1[i], date_bits, ek, LOGIC);
	 less_than<Lvl1>(l_returnflag[i], cipher1, pred_cres2[i], bin_bits, ek, LOGIC);
	 less_than<Lvl1>(l_linestatus[i], cipher1, pred_cres3[i], bin_bits, ek, LOGIC);
	 HEDB::HomNOT<Lvl1>(pred_cres4[i], pred_cres2[i]);
	 HEDB::HomNOT<Lvl1>(pred_cres5[i], pred_cres3[i]);

   HomAND(cres[0][i], pred_cres1[i], pred_cres2[i], ek, LOGIC);
	 HomAND(cres[1][i], cres[0][i], pred_cres5[i], ek, LOGIC); // 01
   HomAND(cres[0][i], cres[0][i], pred_cres3[i], ek, LOGIC); // 00

   HomAND(cres[2][i], pred_cres1[i], pred_cres4[i], ek, LOGIC);
   HomAND(cres[3][i], cres[2][i], pred_cres5[i], ek, LOGIC); // 11
   HomAND(cres[2][i], cres[2][i], pred_cres3[i], ek, LOGIC); // 10
  }

  end = std::chrono::system_clock::now();
  double costs =std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Filter Time on CPU: " << costs / 1000 << "ms" << std::endl;

#if 0
  uint32_t rlwe_scale_bits = 29;
  for (size_t i = 0; i < lineitem_rows; i++) {
    TFHEpp::log_rescale(cres[0][i], cres[0][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[1][i], cres[1][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[2][i], cres[2][i], rlwe_scale_bits, ek);
    TFHEpp::log_rescale(cres[3][i], cres[3][i], rlwe_scale_bits, ek);
  }


  pred_res.resize(4);
  pred_res[0].resize(lineitem_rows);
  pred_res[1].resize(lineitem_rows);
  pred_res[2].resize(lineitem_rows);
  pred_res[3].resize(lineitem_rows);

  for (size_t i = 0; i < lineitem_rows; ++i) {
    pred_res[0][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[0][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[1][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[1][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[2][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[2][i], pow(2., 29),sk.key.get<Lvl1>());
	  pred_res[3][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[3][i], pow(2., 29),sk.key.get<Lvl1>());
  }
#endif
}

void aggregation(std::vector<std::vector<TLWELvl1>> &pred_cres,
                 std::vector<std::vector<uint32_t>> &pred_res,
                 size_t tfhe_n,
                 std::vector<Lvl1::T> &quantity_data, size_t rows,
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
  seal::Ciphertext result00, result01, result10, result11;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  
  LWEsToRLWE(result00, pred_cres[0], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result00, result00.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  LWEsToRLWE(result01, pred_cres[1], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result01, result01.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  LWEsToRLWE(result10, pred_cres[2], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result10, result10.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  LWEsToRLWE(result11, pred_cres[3], pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
  HomRound(result11, result11.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

  end = std::chrono::system_clock::now();
  aggregation_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  seal::Plaintext plain;
  std::vector<std::vector<double>> computed(4);
  for(int i=0;i<4;++i){
    computed[i].resize(slots_count);
  }
  decryptor.decrypt(result00, plain);
  seal::pack_decode(computed[0], plain, ckks_encoder);

  decryptor.decrypt(result01, plain);
  seal::pack_decode(computed[1], plain, ckks_encoder);

  decryptor.decrypt(result10, plain);
  seal::pack_decode(computed[2], plain, ckks_encoder);

  decryptor.decrypt(result11, plain);
  seal::pack_decode(computed[3], plain, ckks_encoder);

  double err[4] = {0.,0.,0.,0.};
  for(int j=0;j<4;++j)
  for (size_t i = 0; i < slots_count; ++i) {
    err[j] += std::abs(computed[j][i] - pred_res[j][i]);
  }
  for (int j = 0; j < 4; ++j)
    printf("Repack average error%d = %f ~ 2^%.1f\n",j, err[j] / slots_count,
           std::log2(err[j] / slots_count));

  // Filter
  std::vector<double> quantity(quantity_data.size());
  seal::Ciphertext quantity_cipher;
  for (size_t i = 0; i < rows; i++) {
    quantity[i] = quantity_data[i];
  }

  double qd = parms.coeff_modulus()[result00.coeff_modulus_size() - 1].value();
  seal::pack_encode(quantity, qd, plain, ckks_encoder);
  encryptor.encrypt_symmetric(plain, quantity_cipher);


  std::cout << "Aggregating price and discount .." << std::endl;
  start = std::chrono::system_clock::now();
  // result * quantity_cipher
  
  seal::multiply_and_relinearize(result00, quantity_cipher, result00, evaluator, relin_keys);
  evaluator.rescale_to_next_inplace(result00);

  seal::multiply_and_relinearize(result01, quantity_cipher, result01, evaluator, relin_keys);
  evaluator.rescale_to_next_inplace(result01);

  seal::multiply_and_relinearize(result10, quantity_cipher, result10, evaluator, relin_keys);
  evaluator.rescale_to_next_inplace(result10);

  seal::multiply_and_relinearize(result11, quantity_cipher, result11, evaluator, relin_keys);
  evaluator.rescale_to_next_inplace(result11);
  

  int logrow = log2(rows);
  // Reduction
  seal::Ciphertext temp;
  for (size_t i = 0; i < logrow; i++) {
	size_t step = 1 << (logrow - i - 1);
    temp = result00;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result00, temp);

    temp = result01;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result01, temp);

    temp = result10;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result10, temp);

    temp = result11;
    evaluator.rotate_vector_inplace(temp, step, galois_keys);
    evaluator.add_inplace(result11, temp);
  }

  end = std::chrono::system_clock::now();
  aggregation_time +=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::vector<std::vector<double>> agg_result(4);
  for (int j = 0; j < 4; ++j) {
	agg_result[j].resize(slots_count);
  }
  decryptor.decrypt(result00, plain);
  seal::pack_decode(agg_result[0], plain, ckks_encoder);
  decryptor.decrypt(result01, plain);
  seal::pack_decode(agg_result[1], plain, ckks_encoder);
  decryptor.decrypt(result10, plain);
  seal::pack_decode(agg_result[2], plain, ckks_encoder);
  decryptor.decrypt(result11, plain);
  seal::pack_decode(agg_result[3], plain, ckks_encoder);

  cout << "\nEncrypted aggregation result: " << endl;

  cout << "returnflag,\t linestatus,\t sum_qty" << std::endl;

  cout << 0 << ",\t " << 0 << ",\t " << std::round(agg_result[0][0]) << std::endl;
  cout << 0 << ",\t " << 1 << ",\t " << std::round(agg_result[1][0]) << std::endl;
  cout << 1 << ",\t " << 0 << ",\t " << std::round(agg_result[2][0]) << std::endl;
  cout << 1 << ",\t " << 1 << ",\t " << std::round(agg_result[3][0]) << std::endl;

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
  Lvl1::T plain_date = 430;
  std::vector<std::vector<TLWE<Lvl1>>> cipher_lineitem;
  std::vector<TLWE<Lvl1>> cipher_bool;
  TLWE<Lvl1> cipher_date;
  std::vector<Lvl1::T>  Ppres, Tpres;
  std::vector<std::vector<TLWE<Lvl1>>> cres;
  std::vector<std::vector<Lvl1::T>> pres;

  gen_DataBase(plain_lineitem, rows);

  encrypt_DataBase(plain_lineitem, cipher_lineitem, sk);

  encrypt_Condition(plain_date, cipher_date, cipher_bool, sk);

  Plain_Query(plain_lineitem, plain_date);

  Filter_Cipher_d(cres, cipher_lineitem, cipher_date, cipher_bool, ek, sk, pres);

  Query(plain_lineitem, pres);
#if 0
  std::vector<Lvl1::T> &quantity = plain_lineitem[3];

  aggregation(cres, pres, Lvl1::n, quantity, rows, sk);

  Filter_Cipher_h(cres, cipher_lineitem, cipher_date, cipher_bool, ek, sk, pres);
#endif
  return 0;
}
