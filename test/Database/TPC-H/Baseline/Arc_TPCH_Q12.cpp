#include "ARCEDB/comparison/batch_bootstrap.h"
#include "ARCEDB/comparison/comparable.h"
#include "ARCEDB/comparison/rgsw_ciphertext.h"
#include "ARCEDB/conversion/packlwes.h"
#include "ARCEDB/conversion/repack.h"
#include "ARCEDB/utils/serialize.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace arcedb;
using namespace seal;

/*
    TPC-H Query 12
    select
        l_shipmode,
        sum(case
            when o_orderpriority = '1-URGENT'
                or o_orderpriority = '2-HIGH'
                then 1
            else 0
        end) as high_line_count,
        sum(case
            when o_orderpriority <> '1-URGENT'
                and o_orderpriority <> '2-HIGH'
                then 1
            else 0
        end) as low_line_count
    from
        orders,
        lineitem
    where
        o_orderkey = l_orderkey
        and l_shipmode in (':1', ':2')
        and l_commitdate < l_receiptdate
        and l_shipdate < l_commitdate
        and l_receiptdate >= date ':3'
        and l_receiptdate < date ':3' + interval '1' year
    group by
        l_shipmode
    order by
        l_shipmode;
*/

/***
 * l_num:
 *      Specifies the number of rows in the plaintext/ciphertext lineitem table to be processed.
 * o_num:
 *      Specifies the number of rows in the plaintext/ciphertext order table to be processed.
 ***/

size_t l_num = 1 << 4;
size_t o_num = 1 << 4;

constexpr int num_thread = 96; // setting multi threads

void tpch_query12(size_t l_num, size_t o_num)
{
	omp_set_num_threads(num_thread);
	
    std::cout << "ArcEDB TPC-H Query12 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Records: " << l_num * o_num << std::endl;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;
    std::uniform_int_distribution<uint32_t> shipmode_message(1, 10);
    std::uniform_int_distribution<uint32_t> shipdate_message(15000, 20000);
    std::uniform_int_distribution<uint32_t> receiptdate_message(15000, 20000);
    std::uniform_int_distribution<uint32_t> commitdate_message(15000, 20000);
    std::uniform_int_distribution<uint32_t> orderkey_message(0, o_num);
    // orderpriority \in ('1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW')
    std::uniform_int_distribution<uint64_t> orderpriority_message(1, 5);
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);
    ek.emplacebkfft<Lvl02>(sk);

    // Filtering
    std::vector<uint64_t> shipmode(l_num), shipdate(l_num), commitdate(l_num), receiptdate(l_num), l_orderkey(l_num);
    std::vector<uint64_t> o_orderkey(o_num), orderpriority(o_num);
    std::vector<ComparableLvl1> shipdate_ciphers(l_num),l_orderkey_ciphers(l_num), commitdate_ciphers(l_num), receiptdate_ciphers(l_num);
    std::vector<TRLWELvl1> shipmode_ciphers(l_num), orderpriority_ciphers(o_num);
    std::vector<ComparbleRGSWLvl1> receiptdate_rgsw_ciphers(l_num), o_orderkey_ciphers(o_num), commitdate_rgsw_ciphers(l_num);

    TRGSWLvl1 predicate_mail_cipher, predicate_ship_cipher; // 'MAIL', 'SHIP'
    TRGSWLvl1 predicate_urgent_cipher, predicate_high_cipher, predicate_upper_bound_cipher; // '1-URGENT', '2-HIGH'
    std::vector<TRGSWLvl1> predicate_date_cipher1(2), predicate_date_cipher2(2);
    uint64_t predicate_mail = 1, predicate_ship= 2, predicate_urgent = 1, predicate_high = 2, predicate_upper_bound = 3;
    uint64_t predicate_date1 = 15500, predicate_date2 = 19000;
    exponent_encrypt_rgsw<P>(predicate_mail, predicate_mail_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_ship, predicate_ship_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_urgent, predicate_urgent_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_high, predicate_high_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_upper_bound, predicate_upper_bound_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_date1, 16, predicate_date_cipher1, sk, true);
    exponent_encrypt_rgsw<P>(predicate_date2, 16, predicate_date_cipher2, sk, true);

    int orderkey_bit = static_cast<int>(std::log2(o_num)) + 1;

    // Start sql evaluation
    std::vector<TLWELvl1> filter_res_mail(l_num), filter_res_ship(l_num), order_res(o_num);
    std::vector<TLWELvl1> res_mail_order(l_num), res_ship_order(l_num);
    std::vector<TLWELvl1> count_mail(l_num), count_ship(l_num);
    

    for (size_t i = 0; i < l_num; i++)
    {
        // Generate data
        shipdate[i] = shipdate_message(engine);
        commitdate[i] = commitdate_message(engine);
        receiptdate[i] = receiptdate_message(engine);
        shipmode[i] = shipmode_message(engine);
        l_orderkey[i] = orderkey_message(engine);

        exponent_encrypt<P>(shipdate[i], 16, shipdate_ciphers[i], sk);
        exponent_encrypt<P>(commitdate[i], 16, commitdate_ciphers[i], sk);
        exponent_encrypt<P>(receiptdate[i], 16, receiptdate_ciphers[i], sk);
        exponent_encrypt<P>(l_orderkey[i], orderkey_bit, l_orderkey_ciphers[i], sk);
        exponent_encrypt<P>(shipmode[i], shipmode_ciphers[i], sk);
        
        exponent_encrypt_rgsw<P>(receiptdate[i], 16, receiptdate_rgsw_ciphers[i], sk, true);
        exponent_encrypt_rgsw<P>(commitdate[i], 16, commitdate_rgsw_ciphers[i], sk, true);
    }

    for (size_t i = 0; i < o_num; i++) {
        o_orderkey[i] = i; // just unique is ok
        orderpriority[i] = orderpriority_message(engine);
        exponent_encrypt<P>(orderpriority[i], orderpriority_ciphers[i], sk);
        exponent_encrypt_rgsw<P>(o_orderkey[i], orderkey_bit, o_orderkey_ciphers[i], sk, true);
    }

    std::chrono::system_clock::time_point start, end;
    double filtering_time_d = 0, filtering_time_h = 0, aggregation_time;
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Filtering..." << std::endl;
    start = std::chrono::system_clock::now();

    // filtering order table
	#pragma omp parallel for
    for (size_t i = 0; i < o_num; i++) {
        less_than_tfhepp(orderpriority_ciphers[i], predicate_upper_bound_cipher, order_res[i],  sk);
    }

    // filtering lineitem table
	#pragma omp parallel for
    for (size_t i = 0; i < l_num; i++) {

        TLWELvl1 pre_res;
        
        less_than_tfhepp(commitdate_ciphers[i], receiptdate_rgsw_ciphers[i], commitdate_ciphers[i].size(), filter_res_mail[i], ek, sk);
        less_than_tfhepp(shipdate_ciphers[i], commitdate_rgsw_ciphers[i], shipdate_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek);
        greater_than_tfhepp(receiptdate_ciphers[i], predicate_date_cipher1, receiptdate_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek);
        less_than_tfhepp(receiptdate_ciphers[i], predicate_date_cipher2, receiptdate_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek);
        filter_res_ship[i] = filter_res_mail[i];
        equality_tfhepp(shipmode_ciphers[i], predicate_mail_cipher, pre_res,sk);
        TFHEpp::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek);

        equality_tfhepp(shipmode_ciphers[i], predicate_ship_cipher, pre_res,sk);
        TFHEpp::HomAND(filter_res_ship[i], pre_res, filter_res_ship[i], ek);

    }
    end = std::chrono::system_clock::now();
    filtering_time_d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Filter Time: " << filtering_time_d << "ms." << std::endl;

    
    std::vector<TLWELvl1> joined_res_mail(l_num * o_num), joined_res_ship(l_num * o_num), count_mail_order(l_num), count_ship_order(l_num);

    start = std::chrono::system_clock::now();
    // table join
	#pragma omp parallel for
    for (size_t i = 0; i < l_num; i++) {
        for (size_t j = 0; j < o_num; ++j) {
            TLWELvl1 pre_res;

            equality_tfhepp(l_orderkey_ciphers[i], o_orderkey_ciphers[j], l_orderkey_ciphers[i].size(), pre_res, ek, sk);
            TFHEpp::HomAND(joined_res_mail[i * o_num + j], pre_res, filter_res_mail[i], ek);
            TFHEpp::HomAND(joined_res_ship[i * o_num + j], pre_res, filter_res_ship[i], ek);

            lift_and_and(joined_res_mail[i * o_num + j], order_res[j], joined_res_mail[i * o_num + j], 29, ek, sk);
            lift_and_and(joined_res_ship[i * o_num + j], order_res[j], joined_res_ship[i * o_num + j], 29, ek, sk);
        }
    }

    end = std::chrono::system_clock::now();
    filtering_time_d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "JOIN time: " << filtering_time_d << "ms." << std::endl;

    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Aggregation :" << std::endl;

    int agg_scale_bits = 48;
    using aggP = Lvl2;
    TFHEpp::TLWE<aggP> agg_mail={}, agg_ship={}, agg_mail_order={}, agg_ship_order={};
    std::vector<TFHEpp::TLWE<aggP>> pre_res_ship_order(l_num), pre_res_mail_order(l_num), pre_res_ship(l_num), pre_res_mail(l_num);
    

    start = std::chrono::system_clock::now();
	#pragma omp parallel for
    for (size_t i = 0; i < l_num; i++) {
        count_mail_order[i] = {};
        count_ship_order[i] = {};
        // HomOR
        for (int j=0;j<o_num;++j){
            for (int k=0;k<=Lvl1::n;++k) {// HomAdd
              count_mail_order[i][k] += joined_res_mail[i * o_num + j][k];
              count_ship_order[i][k] += joined_res_ship[i * o_num + j][k];
            }
        }
        // lift to 16-bit precision
        lift_and_and(filter_res_ship[i], filter_res_ship[i], pre_res_ship[i], agg_scale_bits, ek, sk);
        lift_and_and(filter_res_mail[i], filter_res_mail[i], pre_res_mail[i], agg_scale_bits, ek, sk);
        lift_and_and(count_ship_order[i], count_ship_order[i], pre_res_ship_order[i], agg_scale_bits, ek, sk);
        lift_and_and(count_mail_order[i], count_mail_order[i], pre_res_mail_order[i], agg_scale_bits, ek, sk);

    }

    // aggregate
    for (size_t i = 0; i < l_num; i++) {
        // HomAdd
        for (int k=0; k <= aggP::n; ++k) {
            agg_ship[k] += pre_res_ship[i][k];
            agg_mail[k] += pre_res_mail[i][k];
            agg_ship_order[k] += pre_res_ship_order[i][k];
            agg_mail_order[k] += pre_res_mail_order[i][k];
        }
    }    
    end = std::chrono::system_clock::now();
    aggregation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Aggregation Time: " << aggregation_time << " ms" << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    uint64_t query_res_mail = 0;
    uint64_t query_res_ship = 0;
    uint64_t query_res_mail_order = 0;
    uint64_t query_res_ship_order = 0;

    // for (size_t i = 0; i < l_num; i++)
    // {
    //     query_res_mail += TFHEpp::tlweSymDecrypt<Lvl1>(filter_res_mail[i], sk.key.get<Lvl1>());
    //     query_res_ship += TFHEpp::tlweSymDecrypt<Lvl1>(filter_res_ship[i], sk.key.get<Lvl1>());
    // }

    // for (size_t i = 0; i < l_num; ++i)
    // {
    //     query_res_mail_order += tlweSymInt32Decrypt<Lvl1>(count_mail_order[i], std::pow(2.,29), sk.key.get<Lvl1>());
    //     query_res_ship_order += tlweSymInt32Decrypt<Lvl1>(count_ship_order[i], std::pow(2.,29), sk.key.get<Lvl1>());
    // }

    query_res_mail = tlweSymInt32Decrypt<aggP>(agg_mail, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    query_res_ship = tlweSymInt32Decrypt<aggP>(agg_ship, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    query_res_mail_order = tlweSymInt32Decrypt<aggP>(agg_mail_order, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    query_res_ship_order = tlweSymInt32Decrypt<aggP>(agg_ship_order, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    
    std::cout << "Encrypted result: " << std::endl;
    std::cout << std::setw(12) <<"shipmode" << "|" << std::setw(16) << "high_line_count" << "|" << std::setw(16) << "low_line_count" << std::endl;
    std::cout << std::setw(12) <<"MAIL" << "|" << std::setw(16) << query_res_mail_order << "|" << std::setw(16) << query_res_mail - query_res_mail_order << std::endl;
    std::cout << std::setw(12) <<"SHIP" << "|" << std::setw(16) << query_res_ship_order << "|" << std::setw(16) << query_res_ship - query_res_ship_order << std::endl;

    std::vector<uint64_t> plain_filter_res_mail(l_num, 0), plain_filter_res_ship(l_num, 0), plain_filter_order(l_num, 0);
    std::vector<uint64_t> plain_res_mail_order(l_num, 0), plain_res_ship_order(l_num, 0);
    uint64_t agg_mail_res = 0, agg_mail_order_res = 0, agg_ship_res = 0, agg_ship_order_res = 0;
    bool ress;
    for (size_t i = 0; i < l_num; i++)
    {
        if (commitdate[i] < receiptdate[i] && shipdate[i] < commitdate[i] && receiptdate[i] > predicate_date1 && receiptdate[i] < predicate_date2)
        {
            ress = true;
            
            if (shipmode[i] == 1) {
                agg_mail_res += 1;
                for (size_t j = 0; j < o_num; ++j) {
                    if (l_orderkey[i] == o_orderkey[j]) {
                        if (orderpriority[j] == 1 || orderpriority[j] == 2) {
                            agg_mail_order_res += 1;
                        }
                        break;
                    }
                }
            }

            if (shipmode[i] == 2) {
                agg_ship_res += 1;
                for (size_t j = 0; j < o_num; ++j) {
                    if (l_orderkey[i] == o_orderkey[j]) {
                        if (orderpriority[j] == 1 || orderpriority[j] == 2) {
                            agg_ship_order_res += 1;
                        }
                        break;
                    }
                }
            }
        }
    }

    std::cout << "Filtering finish" << std::endl;
    
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Plain result: " << std::endl;
    std::cout << std::setw(12) <<"shipmode" << "|" << std::setw(16) << "high_line_count" << "|" << std::setw(16) << "low_line_count" << std::endl;
    std::cout << std::setw(12) <<"MAIL" << "|" << std::setw(16) << agg_mail_order_res << "|" << std::setw(16) << agg_mail_res - agg_mail_order_res << std::endl;
    std::cout << std::setw(12) <<"SHIP" << "|" << std::setw(16) << agg_ship_order_res << "|" << std::setw(16) << agg_ship_res - agg_ship_order_res << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    std::cout << "Query Evaluation Time: " << filtering_time_d + aggregation_time << " ms" << std::endl;
	
}


int main(int argc, char *argv[]) {

  if (argc > 1) {
    l_num = std::stoi(argv[1]);
  }
  if (argc > 2) {
    l_num = std::stoi(argv[1]);
    o_num = std::stoi(argv[2]);
  }


  tpch_query12(l_num, o_num);

  return 0;
}


