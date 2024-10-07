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
#include <string>
#include <vector>

using namespace HEDB;
using namespace TFHEpp;

const int rows = 1 << 10; // Number of plaintexts

/***
 * TPC-H Query
 * select
        (price * discount) as revenue
    where
        shipdate >= date1
        and shipdate < date2
        and discount between 8 and 10
        and quantity < 3;

    consider date in [10101~10302]
*/

struct Plain_Bits {
  uint32_t shipdate_bits;
  uint32_t price_bits;
  uint32_t discount_bits;
  uint32_t quantity_bits;
};

//(shipdate, price, discount, quantity)
void gen_DataBase(std::vector<std::vector<Lvl1::T>> &plaintexts, int row, Plain_Bits bits) {

    std::random_device seed_gen;
    std::default_random_engine generator(seed_gen());
    std::uniform_int_distribution<Lvl1::T> shipdate_distribution(0, (1 << bits.shipdate_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> price_distribution(0, (1 << bits.price_bits) - 1); 
    std::uniform_int_distribution<Lvl1::T> discount_distribution(0, (1 << bits.discount_bits) - 1);
    std::uniform_int_distribution<Lvl1::T> quantity_distribution(0, (1 << bits.quantity_bits) - 1);

	std::vector<Lvl1::T> shipdates, prices, discounts, quantities;

    std::cout<<"Begin to generate database..."<<std::endl;

    for (int i = 0; i < row; ++i) {
        Lvl1::T shipdate = shipdate_distribution(generator);
        Lvl1::T price = price_distribution(generator);
        Lvl1::T discount = discount_distribution(generator);
        Lvl1::T quantity = quantity_distribution(generator);

		shipdates.emplace_back(shipdate);
		prices.emplace_back(price);
		discounts.emplace_back(discount);
		quantities.emplace_back(quantity);

    }

	plaintexts.emplace_back(shipdates);
	plaintexts.emplace_back(prices);
	plaintexts.emplace_back(discounts);
	plaintexts.emplace_back(quantities);
	

    std::cout<<"Successfully generated database."<<std::endl;
}

void encrypt_DataBase(std::vector<std::vector<Lvl1::T>> &plaintexts,std::vector<std::vector<TLWE<Lvl1>>> &ciphertexts, TFHESecretKey &sk, Plain_Bits bits) {
	uint32_t scale_bits[4]; 
	scale_bits[0] = std::numeric_limits<Lvl1::T>::digits - bits.shipdate_bits - 1;
	scale_bits[1] = std::numeric_limits<Lvl1::T>::digits - bits.price_bits - 1;
	scale_bits[2] = std::numeric_limits<Lvl1::T>::digits - bits.discount_bits - 1;
	scale_bits[3] = std::numeric_limits<Lvl1::T>::digits - bits.quantity_bits - 1;

    std::cout<< "Begin to encrypt database..." <<std::endl;
    
    ciphertexts.resize(plaintexts.size());
	for(int i = 0; i < plaintexts.size(); ++i) {
		for (int j = 0; j < plaintexts[i].size(); ++j) {
    		// Encrypt plaintext
    		ciphertexts[i].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plaintexts[i][j], Lvl1::α, pow(2., scale_bits[i]), sk.key.get<Lvl1>()));
  		}
	}

    std::cout << "Success encrypt database." << std::endl;
}	

void encrypt_Condition(std::vector<std::vector<Lvl1::T>> &plain_condition,std::vector<std::vector<TLWE<Lvl1>>> &cipher_condition, TFHESecretKey &sk, Plain_Bits bits) {
 
	uint32_t date_scale_bits = std::numeric_limits<Lvl1::T>::digits - bits.shipdate_bits - 1;
	uint32_t discount_scale_bits = std::numeric_limits<Lvl1::T>::digits - bits.discount_bits - 1;
	uint32_t quantity_scale_bits = std::numeric_limits<Lvl1::T>::digits - bits.quantity_bits - 1;

    cipher_condition.resize(plain_condition.size());
	// shipdate
    cipher_condition[0].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_condition[0][0], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	cipher_condition[0].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_condition[0][1], Lvl1::α, pow(2., date_scale_bits), sk.key.get<Lvl1>()));
	// discount
	cipher_condition[1].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_condition[1][0], Lvl1::α, pow(2., discount_scale_bits), sk.key.get<Lvl1>()));
	cipher_condition[1].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_condition[1][1], Lvl1::α, pow(2., discount_scale_bits), sk.key.get<Lvl1>()));
	//quantity
	cipher_condition[2].emplace_back(TFHEpp::tlweSymInt32Encrypt<Lvl1>(plain_condition[2][0], Lvl1::α, pow(2., quantity_scale_bits), sk.key.get<Lvl1>()));

}

void Plain_Query(std::vector<std::vector<Lvl1::T>> &plaintexts, std::vector<std::vector<Lvl1::T>> &filter_condition, std::vector<Lvl1::T> &pres) {
    std::vector<Lvl1::T> &shipdates = plaintexts[0];
    std::vector<Lvl1::T> &prices = plaintexts[1];
    std::vector<Lvl1::T> &discounts = plaintexts[2];
    std::vector<Lvl1::T> &quantities = plaintexts[3];

    
    Lvl1::T date1 = filter_condition[0][0];
    Lvl1::T date2 = filter_condition[0][1];
    Lvl1::T discount1 = filter_condition[1][0];
    Lvl1::T discount2 = filter_condition[1][1];
	Lvl1::T quantity1 = filter_condition[2][0];

    std::cout<<"Begin to filter plain database..."<<std::endl;

    // revenue
    for (size_t i = 0; i < shipdates.size(); ++i) {
        if (shipdates[i] >= date1 && shipdates[i] < date2 && discounts[i] >= discount1 && discounts[i] <= discount2 && quantities[i] < quantity1) {
            pres.emplace_back(1);
        }
        else pres.emplace_back(0);
    }

    double rev = 0;
    for (int i = 0; i < pres.size(); ++i) {
          rev += prices[i] * discounts[i] * pres[i];
    }
    rev*=0.01;
    std::cout << "\nPlaintext query result:" << rev << std::endl << std::endl;

}

void Query(std::vector<std::vector<Lvl1::T>> &plaintexts, std::vector<uint32_t> &pres) {
    std::vector<Lvl1::T> &prices = plaintexts[1];
    std::vector<Lvl1::T> &discounts = plaintexts[2];

    double rev = 0;
    for (int i = 0; i < pres.size(); ++i) {
          rev += prices[i] * discounts[i] * pres[i];
    }
    rev*=0.01;
    std::cout << "\nCiphertext query result:" << rev << std::endl << std::endl;
}

void Filter_Cipher_d(std::vector<std::vector<TLWE<Lvl1>>> &ciphertexts, std::vector<std::vector<TLWE<Lvl1>>> &cipher_condition, TFHEEvalKey &ek, TFHESecretKey &sk, std::vector<TLWE<Lvl1>> &pred_cres, std::vector<Lvl1::T> &cpres, Plain_Bits bits) {
	std::vector<TLWE<Lvl1>> &shipdates = ciphertexts[0];
    std::vector<TLWE<Lvl1>> &prices = ciphertexts[1];
    std::vector<TLWE<Lvl1>> &discounts = ciphertexts[2];
    std::vector<TLWE<Lvl1>> &quantities = ciphertexts[3];

    TLWE<Lvl1> &date1 = cipher_condition[0][0];
    TLWE<Lvl1> &date2 = cipher_condition[0][1];
    TLWE<Lvl1> &discount1 = cipher_condition[1][0];
    TLWE<Lvl1> &discount2 = cipher_condition[1][1];
    TLWE<Lvl1> &quantity1 = cipher_condition[2][0];

	int rows = shipdates.size();

	std::vector<TLWELvl1> pred_cres1(rows), pred_cres2(rows), pred_cres3(rows), pred_cres4(rows), pred_cres5(rows);
    pred_cres.resize(rows); 
    std::cout<< "Begin to filter cipher database..." <<std::endl;

	std::chrono::system_clock::time_point start, end;
  	start = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        uint32_t stream_id = omp_get_thread_num();

		    cuHEDB::greater_than_equal<Lvl1>(shipdates[i], date1, pred_cres1[i], bits.shipdate_bits, ek, LOGIC, stream_id);
		    cuHEDB::less_than<Lvl1>(shipdates[i], date2, pred_cres2[i], bits.shipdate_bits, ek, LOGIC, stream_id);
		    cuHEDB::HomAND(pred_cres[i], pred_cres1[i], pred_cres2[i], ek, LOGIC, stream_id);//AND -> shipdate result
		    cuHEDB::greater_than_equal<Lvl1>(discounts[i], discount1, pred_cres3[i], bits.discount_bits, ek, LOGIC, stream_id);
		    cuHEDB::HomAND(pred_cres[i], pred_cres[i], pred_cres3[i], ek, LOGIC, stream_id);
		    cuHEDB::less_than_equal<Lvl1>(discounts[i], discount2, pred_cres4[i], bits.discount_bits, ek, LOGIC, stream_id);
		    cuHEDB::HomAND(pred_cres[i], pred_cres[i], pred_cres4[i], ek, LOGIC, stream_id);
		    cuHEDB::less_than<Lvl1>(quantities[i], quantity1, pred_cres5[i], bits.quantity_bits, ek, LOGIC, stream_id);
		    cuHEDB::HomAND(pred_cres[i], pred_cres[i], pred_cres5[i], ek, ARITHMETIC, stream_id);
        //cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();
  	double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout<<"Filter Time on GPU: "<< costs/1000 <<"ms"<< std::endl;

    uint32_t rlwe_scale_bits = 29;
    for (size_t i = 0; i < rows; i++) {
      TFHEpp::ari_rescale(pred_cres[i], pred_cres[i], rlwe_scale_bits, ek);
    }

	for (size_t i = 0; i < rows; ++i) {
        cpres.emplace_back(TFHEpp::tlweSymInt32Decrypt<Lvl1>(pred_cres[i], pow(2., 29), sk.key.get<Lvl1>()));
    }

}

void Filter_Cipher_h(std::vector<std::vector<TLWE<Lvl1>>> &ciphertexts, std::vector<std::vector<TLWE<Lvl1>>> &cipher_condition, TFHEEvalKey &ek, TFHESecretKey &sk, std::vector<Lvl1::T> &cres, Plain_Bits bits) {
	std::vector<TLWE<Lvl1>> &shipdates = ciphertexts[0];
    std::vector<TLWE<Lvl1>> &prices = ciphertexts[1];
    std::vector<TLWE<Lvl1>> &discounts = ciphertexts[2];
    std::vector<TLWE<Lvl1>> &quantities = ciphertexts[3];

    TLWE<Lvl1> &date1 = cipher_condition[0][0];
    TLWE<Lvl1> &date2 = cipher_condition[0][1];
    TLWE<Lvl1> &discount1 = cipher_condition[1][0];
    TLWE<Lvl1> &discount2 = cipher_condition[1][1];
    TLWE<Lvl1> &quantity1 = cipher_condition[2][0];

	int rows = shipdates.size();

	std::vector<TLWELvl1> pred_cres(rows), pred_cres1(rows), pred_cres2(rows), pred_cres3(rows), pred_cres4(rows), pred_cres5(rows);

    std::cout<< "Begin to filter cipher database..." <<std::endl;

	std::chrono::system_clock::time_point start, end;
  	start = std::chrono::system_clock::now();

  for (size_t i = 0; i < rows; ++i) {
		greater_than_equal<Lvl1>(shipdates[i], date1, pred_cres1[i], bits.shipdate_bits, ek, LOGIC);
		less_than<Lvl1>(shipdates[i], date2, pred_cres2[i], bits.shipdate_bits, ek, LOGIC);
		HomAND(pred_cres[i], pred_cres1[i], pred_cres2[i], ek, LOGIC);
		greater_than_equal<Lvl1>(discounts[i], discount1, pred_cres3[i], bits.discount_bits, ek, LOGIC);
		HomAND(pred_cres[i], pred_cres[i], pred_cres3[i], ek, LOGIC);
		less_than_equal<Lvl1>(discounts[i], discount2, pred_cres4[i], bits.discount_bits, ek, LOGIC);
		HomAND(pred_cres[i], pred_cres[i], pred_cres4[i], ek, LOGIC);
		less_than<Lvl1>(quantities[i], quantity1, pred_cres5[i], bits.quantity_bits, ek, LOGIC);
		HomAND(pred_cres[i], pred_cres[i], pred_cres5[i], ek, LOGIC);
  }

	end = std::chrono::system_clock::now();
  	double costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	
	std::cout<<"Filter Time on CPU: "<< costs/1000 <<"ms"<< std::endl;

	for (size_t i = 0; i < rows; ++i) {
        cres.emplace_back(TFHEpp::tlweSymDecrypt<Lvl1>(pred_cres[i], sk.key.lvl1));
    }

}

void aggregation(std::vector<TLWELvl1> &pred_cres, std::vector<uint32_t> &pred_res, size_t tfhe_n,
            std::vector<Lvl1::T> &extendedprice_data, std::vector<Lvl1::T> &discount_data,
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


    // Filter result * data
    std::vector<double> price_discount(extendedprice_data.size());
    seal::Ciphertext price_discount_cipher;
    for (size_t i = 0; i < rows; i++)
    {
        price_discount[i] = extendedprice_data[i] * discount_data[i] *0.01;
    }
    double qd = parms.coeff_modulus()[result.coeff_modulus_size() - 1].value();
    seal::pack_encode(price_discount, qd, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, price_discount_cipher);

    std::cout << "Aggregating price and discount .." << std::endl;
    start = std::chrono::system_clock::now();
    //result * price_discount_cipher
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

    cout << "\nEncrypted aggregation result: " << agg_result[0] <<endl<<endl;
    std::cout<<"Aggregation time: "<<aggregation_time<<"ms"<<std::endl;
}

int main() {

  omp_set_num_threads(num_stream1);

  warmupGPU();
  // Lvl1
  std::cout << "Encrypting" << std::endl;

  double costs;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  
  TFHESecretKey sk;
  TFHEEvalKey ek;
  ek.emplacebkfft<Lvl01>(sk);
  ek.emplaceiksk<Lvl10>(sk);
  ek.emplacebkfft<Lvl02>(sk);
  ek.emplaceiksk<Lvl20>(sk);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "generate Secret Key Time: " << costs/1000 << "ms" <<std::endl;

  std::cout << "Loading" << std::endl;
  start = std::chrono::system_clock::now();

  //load BK to device
  cufftlvl1.LoadBK(*ek.bkfftlvl01);
  //cufftlvl2.LoadBK(*ek.bkfftlvl02);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<"Load Success."<<std::endl;
  std::cout << "Loading Time: " << costs/1000 << "ms" << std::endl;

  start = std::chrono::system_clock::now();
  
  // DataBase init
  std::vector<std::vector<Lvl1::T>> plaintexts, plain_condition{{101, 402}, {4, 10}, {50}};
  std::vector<std::vector<TLWE<Lvl1>>> ciphertexts, cipher_condition;
  std::vector<Lvl1::T>  pres, pres1, pres2;
  std::vector<TLWE<Lvl1>> cres;
  // bits needed
  Plain_Bits bits;
  bits.shipdate_bits = 9;
  bits.price_bits = 9;
  bits.discount_bits = 4;
  bits.quantity_bits = 6;

  gen_DataBase(plaintexts, rows, bits);

  encrypt_DataBase(plaintexts, ciphertexts, sk, bits);

  encrypt_Condition(plain_condition, cipher_condition, sk, bits);

  Plain_Query(plaintexts, plain_condition, pres);

  Filter_Cipher_d(ciphertexts, cipher_condition, ek, sk, cres, pres1, bits);

  Query(plaintexts, pres1);

#if 0
  std::vector<Lvl1::T> &discounts = plaintexts[2];
  std::vector<Lvl1::T> &prices = plaintexts[1];

  aggregation(cres, pres, Lvl1::n, prices, discounts, rows, sk);

  Filter_Cipher_h(ciphertexts, cipher_condition, ek, sk, pres2, bits);
  Query(plaintexts, pres2);
#endif
  return 0;
}