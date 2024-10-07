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
 * TPC-H Query 1
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

void relational_query1(size_t num)
{
    std::cout << "Relational SQL Query1 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Records: " << num << std::endl;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;
    std::uniform_int_distribution<uint32_t> shipdate_message(10592-100, 10592+100);
    std::uniform_int_distribution<uint64_t> quantity_message(0, 8);
    std::uniform_real_distribution<double> extendedprice_message(0., 8.);
    std::uniform_real_distribution<double> discount_message(0., 1.);
    std::uniform_real_distribution<double> tax_message(0., 1.);
    std::uniform_int_distribution<uint32_t> bianry_message(0, 1);
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);

    // load BK to device
    cufftplvl1.LoadBK(*ek.bkfftlvl01);

    // Filtering
    std::vector<uint64_t> ship_date(num);
    std::vector<uint64_t> returnflag(num);
    std::vector<uint64_t> linestatus(num);
    std::vector<ComparableLvl1> ship_data_ciphers(num);
    std::vector<TRLWELvl1> returnflag_ciphers(num);
    std::vector<TRLWELvl1> linestatus_ciphers(num);

    std::vector<TRGSWLvl1> shipdate_predicate(2);
    TRGSWLvl1 returnflag_predicate_Y, returnflag_predicate_N, linestatus_predicate_Y, linestatus_predicate_N;
    exponent_encrypt_rgsw<P>(10592, 20, shipdate_predicate, sk, true);
    exponent_encrypt_rgsw<P>(1, returnflag_predicate_Y, sk, true);
    exponent_encrypt_rgsw<P>(0, returnflag_predicate_N, sk, true);
    exponent_encrypt_rgsw<P>(1, linestatus_predicate_Y, sk, true);
    exponent_encrypt_rgsw<P>(0, linestatus_predicate_N, sk, true);


    std::vector<double> quantity(num), extendedprice(num), discount(num), tax(num);
    seal::Ciphertext quantity_cipher, extendedprice_cipher, discount_cipher, tax_cipher;
    for (size_t i = 0; i < num; i++)
    {
        quantity[i] = quantity_message(engine);
        extendedprice[i] = extendedprice_message(engine);
        discount[i] = discount_message(engine);
        tax[i] = tax_message(engine);
    }

    // Start sql evaluation
    std::vector<TLWELvl1> filter_res_YY(num), filter_res_YN(num), filter_res_NY(num), filter_res_NN(num);
    
    for (size_t i = 0; i < num; i++)
    {
        // Generate data
        ship_date[i] = shipdate_message(engine);
        returnflag[i] = bianry_message(engine);
        linestatus[i] = bianry_message(engine);
        
        // Encrypt
        exponent_encrypt<P>(ship_date[i], 20, ship_data_ciphers[i], sk);
        exponent_encrypt<P>(returnflag[i], returnflag_ciphers[i], sk);
        exponent_encrypt<P>(linestatus[i], linestatus_ciphers[i], sk);
    }

    std::chrono::system_clock::time_point start, end;
    double filtering_time = 0, aggregation_time;
    start = std::chrono::system_clock::now();

    #pragma omp parallel for 
    for (size_t i = 0; i < num; i++)
    {
        uint32_t stream_id = omp_get_thread_num();

        TLWELvl1 pre_res_YY, pre_res_YN, pre_res_NY, pre_res_NN;

        // returnflag = Y, linestatus = Y
        cuARCEDB::less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_YY[i], ek, sk, stream_id);
        cuARCEDB::equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_Y, pre_res_YY, sk, stream_id);
        cuARCEDB::HomAND(filter_res_YY[i], pre_res_YY, filter_res_YY[i], ek, stream_id);
        cuARCEDB::equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_Y, pre_res_YY, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_YY[i], pre_res_YY, filter_res_YY[i], 29, ek, sk, stream_id);

        // returnflag = Y, linestatus = N
        cuARCEDB::less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_YN[i], ek, sk, stream_id);
        cuARCEDB::equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_Y, pre_res_YN, sk, stream_id);
        cuARCEDB::HomAND(filter_res_YN[i], pre_res_YN, filter_res_YN[i], ek, stream_id);
        cuARCEDB::equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_N, pre_res_YN, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_YN[i], pre_res_YN, filter_res_YN[i], 29, ek, sk, stream_id);

        // returnflag = N, linestatus = Y
        cuARCEDB::less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_NY[i], ek, sk, stream_id);
        cuARCEDB::equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_N, pre_res_NY, sk, stream_id);
        cuARCEDB::HomAND(filter_res_NY[i], pre_res_NY, filter_res_NY[i], ek, stream_id);
        cuARCEDB::equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_Y, pre_res_NY, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_NY[i], pre_res_NY, filter_res_NY[i], 29, ek, sk, stream_id);

        // returnflag = N, linestatus = N
        cuARCEDB::less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_NN[i], ek, sk, stream_id);
        cuARCEDB::equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_N, pre_res_NN, sk, stream_id);
        cuARCEDB::HomAND(filter_res_NN[i], pre_res_NN, filter_res_NN[i], ek, stream_id);
        cuARCEDB::equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_N, pre_res_NN, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_NN[i], pre_res_NN, filter_res_NN[i], 29, ek, sk, stream_id);
        
		cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();

    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout<<"Filter Time on GPU: " << filtering_time << "ms." << std::endl;

    uint64_t cipher_agg_res_YY = 0, cipher_agg_res_YN = 0, cipher_agg_res_NY = 0, cipher_agg_res_NN = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_YY[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_YY += quantity[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_YN[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_YN += quantity[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_NY[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_NY += quantity[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_NN[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_NN += quantity[i];
        }
    }

    std::cout << "Cipher query result(GPU): " << std::endl;
    std::cout << std::setw(16) <<"returnfalg" << "|" << std::setw(16) << "linestatus" << "|" << std::setw(16) << "sum_qty" << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << cipher_agg_res_YY << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << cipher_agg_res_YN << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << cipher_agg_res_NY << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << cipher_agg_res_NN << std::endl;


    start = std::chrono::system_clock::now();
    for (size_t i = 0; i < num; i++)
    {
        
        TLWELvl1 pre_res_YY, pre_res_YN, pre_res_NY, pre_res_NN;

        // returnflag = Y, linestatus = Y
        less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_YY[i], ek, sk);
        equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_Y, pre_res_YY, sk);
        TFHEpp::HomAND(filter_res_YY[i], pre_res_YY, filter_res_YY[i], ek);
        equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_Y, pre_res_YY, sk);
        lift_and_and(filter_res_YY[i], pre_res_YY, filter_res_YY[i], 29, ek, sk);

        // returnflag = Y, linestatus = N
        less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_YN[i], ek, sk);
        equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_Y, pre_res_YN, sk);
        TFHEpp::HomAND(filter_res_YN[i], pre_res_YN, filter_res_YN[i], ek);
        equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_N, pre_res_YN, sk);
        lift_and_and(filter_res_YN[i], pre_res_YN, filter_res_YN[i], 29, ek, sk);

        // returnflag = N, linestatus = Y
        less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_NY[i], ek, sk);
        equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_N, pre_res_NY, sk);
        TFHEpp::HomAND(filter_res_NY[i], pre_res_NY, filter_res_NY[i], ek);
        equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_Y, pre_res_NY, sk);
        lift_and_and(filter_res_NY[i], pre_res_NY, filter_res_NY[i], 29, ek, sk);

        // returnflag = N, linestatus = N
        less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, filter_res_NN[i], ek, sk);
        equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_N, pre_res_NN, sk);
        TFHEpp::HomAND(filter_res_NN[i], pre_res_NN, filter_res_NN[i], ek);
        equality_tfhepp(linestatus_ciphers[i], linestatus_predicate_N, pre_res_NN, sk);
        lift_and_and(filter_res_NN[i], pre_res_NN, filter_res_NN[i], 29, ek, sk);
        
    }

    end = std::chrono::system_clock::now();

    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout<<"Filter Time on CPU: " << filtering_time << "ms." << std::endl;

    cipher_agg_res_YY = 0, cipher_agg_res_YN = 0, cipher_agg_res_NY = 0, cipher_agg_res_NN = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_YY[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_YY += quantity[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_YN[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_YN += quantity[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_NY[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_NY += quantity[i];
        }
        if (tlweSymInt32Decrypt<Lvl1>(filter_res_NN[i], std::pow(2.,29), sk.key.get<Lvl1>())) {
            cipher_agg_res_NN += quantity[i];
        }
    }

    std::cout << "Cipher query result(CPU): " << std::endl;
    std::cout << std::setw(16) <<"returnfalg" << "|" << std::setw(16) << "linestatus" << "|" << std::setw(16) << "sum_qty" << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << cipher_agg_res_YY << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << cipher_agg_res_YN << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << cipher_agg_res_NY << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << cipher_agg_res_NN << std::endl;



    std::vector<uint64_t> plain_filter_res_YY(num, 0), plain_filter_res_YN(num, 0), plain_filter_res_NY(num, 0), plain_filter_res_NN(num, 0);
    uint64_t plain_agg_res_YY = 0, plain_agg_res_YN = 0, plain_agg_res_NY = 0,plain_agg_res_NN = 0;
    for (size_t i = 0; i < num; i++)
    {
        if (ship_date[i] < 10592)
        {
            if (returnflag[i] == 1)
            {
                if (linestatus[i] == 1)
                {
                    plain_filter_res_YY[i] = 1;
                    plain_agg_res_YY += quantity[i];
                }
                else
                {
                    plain_filter_res_YN[i] = 1;
                    plain_agg_res_YN += quantity[i];
                }
            }
            else
            {
                if (linestatus[i] == 1)
                {
                    plain_filter_res_NY[i] = 1;
                    plain_agg_res_NY += quantity[i];
                }
                else
                {
                    plain_filter_res_NN[i] = 1;
                    plain_agg_res_NN += quantity[i];
                }
            }
        }

    }

    std::cout << "Plain query result: " << std::endl;
    std::cout << std::setw(16) <<"returnfalg" << "|" << std::setw(16) << "linestatus" << "|" << std::setw(16) << "sum_qty" << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << plain_agg_res_YY << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << plain_agg_res_YN << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << plain_agg_res_NY << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << plain_agg_res_NN << std::endl;\


    std::cout << "Filtering finish." << std::endl;
#if 0
    std::cout << "Aggregation :" << std::endl;
    uint64_t scale_bits = 29;
    uint64_t modq_bits = 32;
    uint64_t modulus_bits = 45;
    uint64_t repack_scale_bits = modulus_bits + scale_bits - modq_bits;
    uint64_t slots_count = filter_res_YY.size();
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
    seal::Ciphertext resultYY, resultYN, resultNY, resultNN;
    start = std::chrono::system_clock::now();

    LWEsToRLWE(resultYY, filter_res_YY, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(resultYY, resultYY.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

    LWEsToRLWE(resultYN, filter_res_YN, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(resultYN, resultYN.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);
    
    LWEsToRLWE(resultNY, filter_res_NY, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(resultNY, resultNY.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);

    LWEsToRLWE(resultNN, filter_res_NN, pre_key, scale, std::pow(2., modq_bits), std::pow(2., modulus_bits - modq_bits), ckks_encoder, galois_keys, relin_keys, evaluator, context);
    HomRound(resultNN, resultNN.scale(), ckks_encoder, relin_keys, evaluator, decryptor, context);
    end = std::chrono::system_clock::now();
    aggregation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    seal::Plaintext plainYY, plainYN, plainNY, plainNN;
    std::vector<double> computedYY(slots_count), computedYN(slots_count), computedNY(slots_count), computedNN(slots_count);
    decryptor.decrypt(resultYY, plainYY);
    seal::pack_decode(computedYY, plainYY, ckks_encoder);

    decryptor.decrypt(resultYN, plainYN);
    seal::pack_decode(computedYN, plainYN, ckks_encoder);

    decryptor.decrypt(resultNY, plainNY);
    seal::pack_decode(computedNY, plainNY, ckks_encoder);

    decryptor.decrypt(resultNN, plainNN);
    seal::pack_decode(computedNN, plainNN, ckks_encoder);

    double errYY = 0., errYN = 0., errNY = 0., errNN = 0.;
    
    for (size_t i = 0; i < slots_count; ++i)
    {
        errYY += std::abs(computedYY[i] - plain_filter_res_YY[i]);

        errYN += std::abs(computedYN[i] - plain_filter_res_YN[i]);

        errNY += std::abs(computedNY[i] - plain_filter_res_NY[i]);

        errNN += std::abs(computedNN[i] - plain_filter_res_NN[i]);
    }

    printf("Repack YY average error = %f ~ 2^%.1f\n", errYY / slots_count, std::log2(errYY / slots_count));
    printf("Repack YN average error = %f ~ 2^%.1f\n", errYN / slots_count, std::log2(errYN / slots_count));
    printf("Repack NY average error = %f ~ 2^%.1f\n", errNY / slots_count, std::log2(errNY / slots_count));
    printf("Repack NN average error = %f ~ 2^%.1f\n", errNN / slots_count, std::log2(errNN / slots_count));


    /*
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
    */
    
    
    Plaintext plain;
    double qd = parms.coeff_modulus()[resultYY.coeff_modulus_size() - 1].value();
    scale = std::pow(2., 42);
    seal::pack_encode(quantity, qd, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, quantity_cipher);

    seal::pack_encode(extendedprice, scale, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, extendedprice_cipher);

    seal::pack_encode(discount, scale, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, discount_cipher);

    seal::pack_encode(tax, scale, plain, ckks_encoder);
    encryptor.encrypt_symmetric(plain, tax_cipher);

    std::cout << "Aggregating quanlity, Taking SUM(quantlity) as an example.." << std::endl;

    Ciphertext sum_qty_cipher_YY, sum_qty_cipher_YN, sum_qty_cipher_NY, sum_qty_cipher_NN;
    start = std::chrono::system_clock::now();
    seal::multiply_and_relinearize(resultYY, quantity_cipher, sum_qty_cipher_YY, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(sum_qty_cipher_YY);

    seal::multiply_and_relinearize(resultYN, quantity_cipher, sum_qty_cipher_YN, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(sum_qty_cipher_YN);

    seal::multiply_and_relinearize(resultNY, quantity_cipher, sum_qty_cipher_NY, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(sum_qty_cipher_NY);

    seal::multiply_and_relinearize(resultNN, quantity_cipher, sum_qty_cipher_NN, evaluator, relin_keys);
    evaluator.rescale_to_next_inplace(sum_qty_cipher_NN);
    int logrow = log2(num);
    
    seal::Ciphertext temp;
    size_t step;
    for (size_t i = 0; i < logrow; i++)
    {
        temp = sum_qty_cipher_YY;
        step = 1 << (logrow - i - 1);
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(sum_qty_cipher_YY, temp);

        temp = sum_qty_cipher_YN;
        step = 1 << (logrow - i - 1);
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(sum_qty_cipher_YN, temp);

        temp = sum_qty_cipher_NY;
        step = 1 << (logrow - i - 1);
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(sum_qty_cipher_NY, temp);

        temp = sum_qty_cipher_NN;
        step = 1 << (logrow - i - 1);
        evaluator.rotate_vector_inplace(temp, step, galois_keys);
        evaluator.add_inplace(sum_qty_cipher_NN, temp);
    }
    end = std::chrono::system_clock::now();
    aggregation_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::vector<double> agg_resultYY(slots_count), agg_resultYN(slots_count), agg_resultNY(slots_count), agg_resultNN(slots_count);
    decryptor.decrypt(sum_qty_cipher_YY, plain);
    seal::pack_decode(agg_resultYY, plain, ckks_encoder);

    decryptor.decrypt(sum_qty_cipher_YN, plain);
    seal::pack_decode(agg_resultYN, plain, ckks_encoder);

    decryptor.decrypt(sum_qty_cipher_NY, plain);
    seal::pack_decode(agg_resultNY, plain, ckks_encoder);

    decryptor.decrypt(sum_qty_cipher_NN, plain);
    seal::pack_decode(agg_resultNN, plain, ckks_encoder);

    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Aggregation Time: " << aggregation_time << " ms" << std::endl;

    std::cout << "Encrypted query result: " << std::endl;
    std::cout << std::setw(16) <<"returnfalg" << "|" << std::setw(16) << "linestatus" << "|" << std::setw(16) << "sum_qty" << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << std::round(agg_resultYY[0]) << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << std::round(agg_resultYN[0]) << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << std::round(agg_resultNY[0]) << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << std::round(agg_resultNN[0]) << std::endl;

    std::cout << "Plain query result: " << std::endl;
    std::cout << std::setw(16) <<"returnfalg" << "|" << std::setw(16) << "linestatus" << "|" << std::setw(16) << "sum_qty" << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << plain_agg_res_YY << std::endl;
    std::cout << std::setw(16) <<"Y" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << plain_agg_res_YN << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "Y" << "|" << std::setw(16) << plain_agg_res_NY << std::endl;
    std::cout << std::setw(16) <<"N" << "|" << std::setw(16) << "N" << "|" << std::setw(16) << plain_agg_res_NN << std::endl;\

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

  relational_query1(num);

  return 0;
}


