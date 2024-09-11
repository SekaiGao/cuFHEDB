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
    TPC-H Query 12
    select
        l_shipmode,
        sum(case
            when o_orderpriority = '0-URGENT'
                or o_orderpriority = '1-HIGH'
                then 1
            else 0
        end) as high_line_count,
        sum(case
            when o_orderpriority <> '0-URGENT'
                and o_orderpriority <> '1-HIGH'
                then 1
            else 0
        end) as low_line_count
    from
        orders,
        lineitem
    where
        o_orderkey = l_orderkey
        and l_shipmode in ('SHIP', 'MAIL')
        and l_commitdate < l_receiptdate
        and l_shipdate < l_commitdate
        and l_receiptdate >= date ':3'
        and l_receiptdate < date ':3' + interval '1' year
    group by
        l_shipmode
    order by
        l_shipmode;
    Consider the joined table
*/

const uint32_t mode_bits = 2;
const uint32_t priority_bits = 2;
const uint32_t date_bits = 9;

//lineitem(shipmode(2), orderpriority(2), shipdate(9), commitdate(9), receiptdate(9))
void gen_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem, int lineitem_rows) {

    std::random_device seed_gen;
    std::default_random_engine generator(seed_gen());
    std::uniform_int_distribution<Lvl1::T> mode_distribution(0, (1 << mode_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> priority_distribution(0, (1 << priority_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> date_distribution(0, (1 << date_bits) - 1);

    std::vector<Lvl1::T> shipmodes, orderpriorities, shipdates, commitdates, receiptdates;

    std::cout<<"Begin to generate database..."<<std::endl;

    for (int i = 0; i < lineitem_rows; ++i) {
        Lvl1::T shipmode = mode_distribution(generator);
        Lvl1::T orderpriority = priority_distribution(generator);
        Lvl1::T shipdate = date_distribution(generator);
        Lvl1::T commitdate = date_distribution(generator);
        Lvl1::T receiptdate = date_distribution(generator);

        shipmodes.emplace_back(shipmode);
        orderpriorities.emplace_back(orderpriority);
        shipdates.emplace_back(shipdate);
        commitdates.emplace_back(commitdate);
        receiptdates.emplace_back(receiptdate);
    }

    plain_lineitem.emplace_back(shipmodes);
    plain_lineitem.emplace_back(orderpriorities);
    plain_lineitem.emplace_back(shipdates);
    plain_lineitem.emplace_back(commitdates);
    plain_lineitem.emplace_back(receiptdates);

    std::cout<<"Successfully generated database."<<std::endl;
}

void encrypt_DataBase(std::vector<std::vector<Lvl1::T>> &plain_lineitem,
                      std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                      TFHESecretKey &sk) {

    uint32_t scale_bits_lineitem[5];
    scale_bits_lineitem[0] = std::numeric_limits<Lvl1::T>::digits - mode_bits - 1;
    scale_bits_lineitem[1] = std::numeric_limits<Lvl1::T>::digits - priority_bits - 1;
    scale_bits_lineitem[2] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
    scale_bits_lineitem[3] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
    scale_bits_lineitem[4] = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;

    std::cout << "Begin to encrypt database..." << std::endl;

    cipher_lineitem.resize(5);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < plain_lineitem[i].size(); ++j) {
            cipher_lineitem[i].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_lineitem[i][j], Lvl1::α, pow(2., scale_bits_lineitem[i]), sk.key.get<Lvl1>()));
        }
    }

    std::cout << "Success encrypt database." << std::endl;
}

void encrypt_Condition(std::vector<Lvl1::T> &plain_date, std::vector<TLWE<Lvl1>> &cipher_date, std::vector<TLWE<Lvl1>> &cipher_num, TFHESecretKey &sk) {
 
	uint32_t date_scale_bits = std::numeric_limits<Lvl1::T>::digits - date_bits - 1;
	uint32_t bin_scale_bits = std::numeric_limits<Lvl1::T>::digits - mode_bits - 1;
	
    cipher_date.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date[0], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	cipher_date.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_date[1], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	cipher_num.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(0, Lvl1::α, pow(2., bin_scale_bits), sk.key.get<Lvl1>()));
	cipher_num.emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(1, Lvl1::α, pow(2., bin_scale_bits), sk.key.get<Lvl1>()));

}



void Plain_Query(std::vector<std::vector<Lvl1::T>> &plain_lineitem, std::vector<Lvl1::T> &plain_date) {
    std::vector<Lvl1::T> &l_shipmode = plain_lineitem[0];
    std::vector<Lvl1::T> &l_orderpriority = plain_lineitem[1];
    std::vector<Lvl1::T> &l_shipdate = plain_lineitem[2];
    std::vector<Lvl1::T> &l_commitdate = plain_lineitem[3];
    std::vector<Lvl1::T> &l_receiptdate = plain_lineitem[4];

	Lvl1::T filter_date_start = plain_date[0];
	Lvl1::T filter_date_end = plain_date[1];

    std::cout<<"Begin to filter database..."<<std::endl;

    std::map<Lvl1::T, Lvl1::T> high_line_count;
    std::map<Lvl1::T, Lvl1::T> low_line_count;

    for (Lvl1::T mode = 0; mode < 4; ++mode) {
        high_line_count[mode] = 0;
        low_line_count[mode] = 0;
    }

    for (size_t i = 0; i < l_shipdate.size(); ++i) {
        if (l_receiptdate[i] >= filter_date_start && l_receiptdate[i] < filter_date_end &&
            l_commitdate[i] < l_receiptdate[i] && l_shipdate[i] < l_commitdate[i]) {
            
            Lvl1::T shipmode = l_shipmode[i];
            Lvl1::T orderpriority = l_orderpriority[i];

			if(shipmode == 0 || shipmode == 1) {
            if (orderpriority == 0 || orderpriority == 1) { // 0-URGENT or 1-HIGH
                high_line_count[shipmode]++;
            } else {
                low_line_count[shipmode]++;
            }
			}
        }
    }

    std::cout << "\nPlaintext query result:" << std::endl;
    std::cout << "shipmode, high_line_count, low_line_count" << std::endl;
    
	string mode_str[2]={"SHIP", "MAIL"};

    for (Lvl1::T mode = 0; mode < 2; ++mode) {
        std::cout << mode_str[mode] << ", " << high_line_count[mode] << ", " << low_line_count[mode] << std::endl;
    }
    cout<<endl;
}

void Query(std::vector<std::vector<uint32_t>> &pred_res) {

    std::map<Lvl1::T, Lvl1::T> high_line_count;
    std::map<Lvl1::T, Lvl1::T> low_line_count;

    for (Lvl1::T mode = 0; mode < 4; ++mode) {
        high_line_count[mode] = 0;
        low_line_count[mode] = 0;
    }

    for (size_t i = 0; i < rows; ++i) {
            
            if(pred_res[0][i]==1)
                high_line_count[0]++;
            else if(pred_res[1][i]==1)
                low_line_count[0]++;
            if(pred_res[2][i]==1)
                high_line_count[1]++;
            else if(pred_res[3][i]==1)
                low_line_count[1]++;
    }

    std::cout << "\nCiphertext query result:" << std::endl;
    std::cout << "shipmode, high_line_count, low_line_count" << std::endl;
    
	string mode_str[2]={"SHIP", "MAIL"};

    for (Lvl1::T mode = 0; mode < 2; ++mode) {
        std::cout << mode_str[mode] << ", " << high_line_count[mode] << ", " << low_line_count[mode] << std::endl;
    }
    cout<<endl;
}

void Filter_Cipher_d(std::vector<std::vector<TLWELvl1>> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     std::vector<TLWE<Lvl1>> &cipher_date,
                     std::vector<TLWE<Lvl1>> &cipher_num, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<std::vector<uint32_t>> &pred_res) {

  std::vector<TLWE<Lvl1>> &l_shipmode = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_orderpriority = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_shipdate = cipher_lineitem[2];
  std::vector<TLWE<Lvl1>> &l_commitdate = cipher_lineitem[3];
  std::vector<TLWE<Lvl1>> &l_receiptdate = cipher_lineitem[4];

  TLWE<Lvl1> &date0 = cipher_date[0];
  TLWE<Lvl1> &date1 = cipher_date[1];
  TLWE<Lvl1> &cipher0 = cipher_num[0];//ciphertext zero
  TLWE<Lvl1> &cipher1 = cipher_num[1];// ciphertext one

  uint32_t lineitem_rows = l_shipmode.size();
  cres.resize(4);
  cres[0].resize(lineitem_rows);// mode0 high
  cres[1].resize(lineitem_rows);// mode0 low
  cres[2].resize(lineitem_rows); // mode1 high
  cres[3].resize(lineitem_rows); // mode1 low

  std::vector<TLWELvl1> pred_cres1(lineitem_rows), pred_cres2(lineitem_rows),
      pred_cres3(lineitem_rows), pred_cres4(lineitem_rows),
      pred_cres5(lineitem_rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < lineitem_rows; ++i) {
        uint32_t stream_id = omp_get_thread_num();

        cuHEDB::greater_than_equal<Lvl1>(l_receiptdate[i], date0, pred_cres1[i], date_bits, ek, LOGIC, stream_id);
        cuHEDB::less_than<Lvl1>(l_receiptdate[i], date1, pred_cres2[i], date_bits, ek, LOGIC, stream_id);
        cuHEDB::HomAND(pred_cres1[i], pred_cres1[i], pred_cres2[i], ek, LOGIC, stream_id);

        cuHEDB::less_than<Lvl1>(l_commitdate[i], l_receiptdate[i], pred_cres3[i], date_bits, ek, LOGIC, stream_id);
        cuHEDB::less_than<Lvl1>(l_shipdate[i], l_commitdate[i], pred_cres4[i], date_bits, ek, LOGIC, stream_id);
        cuHEDB::HomAND(pred_cres1[i], pred_cres1[i], pred_cres3[i], ek, LOGIC, stream_id);
        cuHEDB::HomAND(pred_cres1[i], pred_cres1[i], pred_cres4[i], ek, LOGIC, stream_id);

        cuHEDB::less_than_equal<Lvl1>(l_orderpriority[i], cipher1, pred_cres2[i], priority_bits, ek, LOGIC, stream_id);

		cuHEDB::HomNOT<Lvl1>(pred_cres3[i], pred_cres2[i]);

		cuHEDB::equal<Lvl1>(l_shipmode[i], cipher0, pred_cres4[i], mode_bits, ek, LOGIC, stream_id);// mode0
		cuHEDB::equal<Lvl1>(l_shipmode[i], cipher1, pred_cres5[i], mode_bits, ek, LOGIC, stream_id);// mode1

		cuHEDB::HomAND(pred_cres2[i], pred_cres2[i], pred_cres1[i], ek, LOGIC, stream_id);
		cuHEDB::HomAND(pred_cres3[i], pred_cres3[i], pred_cres1[i], ek, LOGIC, stream_id);
        
		cuHEDB::HomAND(cres[0][i], pred_cres2[i], pred_cres4[i], ek, LOGIC, stream_id);// 0 high
        cuHEDB::HomAND(cres[1][i], pred_cres3[i], pred_cres4[i], ek, LOGIC, stream_id);// 0 low
        cuHEDB::HomAND(cres[2][i], pred_cres2[i], pred_cres5[i], ek, LOGIC, stream_id);// 1 high
        cuHEDB::HomAND(cres[3][i], pred_cres3[i], pred_cres5[i], ek, LOGIC, stream_id);// 1 low

        cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();
    double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Filter Time on GPU: " << costs / 1000 << "ms" << std::endl;

    uint32_t rlwe_scale_bits = 29;
    #pragma omp parallel for
    for (size_t i = 0; i < lineitem_rows; ++i) {
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
        pred_res[0][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[0][i], pow(2., 29), sk.key.get<Lvl1>());
        pred_res[1][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[1][i], pow(2., 29), sk.key.get<Lvl1>());
		pred_res[2][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[2][i], pow(2., 29), sk.key.get<Lvl1>());
        pred_res[3][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[3][i], pow(2., 29), sk.key.get<Lvl1>());
    }
}

void Filter_Cipher_h(std::vector<std::vector<TLWELvl1>> &cres,
                     std::vector<std::vector<TLWE<Lvl1>>> &cipher_lineitem,
                     std::vector<TLWE<Lvl1>> &cipher_date,
                     std::vector<TLWE<Lvl1>> &cipher_num, TFHEEvalKey &ek,
                     TFHESecretKey &sk, std::vector<std::vector<uint32_t>> &pred_res) {

  std::vector<TLWE<Lvl1>> &l_shipmode = cipher_lineitem[0];
  std::vector<TLWE<Lvl1>> &l_orderpriority = cipher_lineitem[1];
  std::vector<TLWE<Lvl1>> &l_shipdate = cipher_lineitem[2];
  std::vector<TLWE<Lvl1>> &l_commitdate = cipher_lineitem[3];
  std::vector<TLWE<Lvl1>> &l_receiptdate = cipher_lineitem[4];

  TLWE<Lvl1> &date0 = cipher_date[0];
  TLWE<Lvl1> &date1 = cipher_date[1];
  TLWE<Lvl1> &cipher0 = cipher_num[0];//ciphertext zero
  TLWE<Lvl1> &cipher1 = cipher_num[1];// ciphertext one

  uint32_t lineitem_rows = l_shipmode.size();
  cres.resize(4);
  cres[0].resize(lineitem_rows);// mode0 high
  cres[1].resize(lineitem_rows);// mode0 low
  cres[2].resize(lineitem_rows); // mode1 high
  cres[3].resize(lineitem_rows); // mode1 low

  std::vector<TLWELvl1> pred_cres1(lineitem_rows), pred_cres2(lineitem_rows),
      pred_cres3(lineitem_rows), pred_cres4(lineitem_rows),
      pred_cres5(lineitem_rows);

  std::cout << "Begin to filter cipher database..." << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

    for (size_t i = 0; i < lineitem_rows; ++i) {
        greater_than_equal<Lvl1>(l_receiptdate[i], date0, pred_cres1[i], date_bits, ek, LOGIC);
        less_than<Lvl1>(l_receiptdate[i], date1, pred_cres2[i], date_bits, ek, LOGIC);
        HomAND(pred_cres1[i], pred_cres1[i], pred_cres2[i], ek, LOGIC);

        less_than<Lvl1>(l_commitdate[i], l_receiptdate[i], pred_cres3[i], date_bits, ek, LOGIC);
        less_than<Lvl1>(l_shipdate[i], l_commitdate[i], pred_cres4[i], date_bits, ek, LOGIC);
        HomAND(pred_cres1[i], pred_cres1[i], pred_cres3[i], ek, LOGIC);
        HomAND(pred_cres1[i], pred_cres1[i], pred_cres4[i], ek, LOGIC);

        less_than_equal<Lvl1>(l_orderpriority[i], cipher1, pred_cres2[i], priority_bits, ek, LOGIC);

		HEDB::HomNOT<Lvl1>(pred_cres3[i], pred_cres2[i]);

		equal<Lvl1>(l_shipmode[i], cipher0, pred_cres4[i], mode_bits, ek, LOGIC);// mode0
		equal<Lvl1>(l_shipmode[i], cipher1, pred_cres5[i], mode_bits, ek, LOGIC);// mode1

		HomAND(pred_cres2[i], pred_cres2[i], pred_cres1[i], ek, LOGIC);
		HomAND(pred_cres3[i], pred_cres3[i], pred_cres1[i], ek, LOGIC);
        
		HomAND(cres[0][i], pred_cres2[i], pred_cres4[i], ek, LOGIC);// 0 high
        HomAND(cres[1][i], pred_cres3[i], pred_cres4[i], ek, LOGIC);// 0 low
        HomAND(cres[2][i], pred_cres2[i], pred_cres5[i], ek, LOGIC);// 1 high
        HomAND(cres[3][i], pred_cres3[i], pred_cres5[i], ek, LOGIC);// 1 low
    }

    end = std::chrono::system_clock::now();
    double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Filter Time on CPU: " << costs / 1000 << "ms" << std::endl;

#if 0
    uint32_t rlwe_scale_bits = 29;
    for (size_t i = 0; i < lineitem_rows; ++i) {
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
        pred_res[0][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[0][i], pow(2., 29), sk.key.get<Lvl1>());
        pred_res[1][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[1][i], pow(2., 29), sk.key.get<Lvl1>());
		pred_res[2][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[2][i], pow(2., 29), sk.key.get<Lvl1>());
        pred_res[3][i] = TFHEpp::tlweSymInt32Decrypt<Lvl1>(cres[3][i], pow(2., 29), sk.key.get<Lvl1>());
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
  std::cout << "Aggregating price and discount .." << std::endl;
  start = std::chrono::system_clock::now();
  
  int logrow = log2(rows);

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

  cout << "\nEncrypted aggregation result1: " << endl;

  cout << "shipmode, high_line_count, low_line_count" << std::endl;

  cout << "SHIP" << ",\t " << abs(std::round(agg_result[0][0])) << ",\t " << abs(std::round(agg_result[1][0])) << std::endl;
  cout << "MAIL" << ",\t " << abs(std::round(agg_result[2][0])) << ",\t " << abs(std::round(agg_result[3][0])) << std::endl;

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

  cout << "shipmode, high_line_count, low_line_count" << std::endl;

  cout << "SHIP" << ",\t " << rst[0] << ",\t " << rst[1] << std::endl;
  cout << "MAIL" << ",\t " << rst[2] << ",\t " << rst[3] << std::endl;
  
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

  std::cout << "Loading" << std::endl;

  warmupGPU();
  start = std::chrono::system_clock::now();

  //load BK to device
  cufftplvl.LoadBK<lvl1param>(*ek.bkfftlvl01);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<"Load Success."<<std::endl;
  std::cout << "Loading Time: " << costs/1000 << "ms" << std::endl;


  // DataBase init
  std::vector<std::vector<Lvl1::T>> plain_lineitem;
  std::vector<Lvl1::T> plain_date = {234, 430};
  std::vector<std::vector<TLWE<Lvl1>>> cipher_lineitem;
  std::vector<TLWE<Lvl1>> cipher_date, cipher_num;
  std::vector<std::vector<TLWE<Lvl1>>> cres;
  std::vector<std::vector<Lvl1::T>> pres;

  gen_DataBase(plain_lineitem, rows);

  encrypt_DataBase(plain_lineitem, cipher_lineitem, sk);

  encrypt_Condition(plain_date, cipher_date, cipher_num, sk);

  Plain_Query(plain_lineitem, plain_date);

  Filter_Cipher_d(cres, cipher_lineitem, cipher_date, cipher_num, ek, sk, pres);

  Query(pres);
#if 0
  aggregation(cres, pres, Lvl1::n, rows, sk);

  Filter_Cipher_h(cres, cipher_lineitem, cipher_date, cipher_num, ek, sk, pres);
#endif
  return 0;
}
