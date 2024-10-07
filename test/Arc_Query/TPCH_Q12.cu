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
    Consider the joined table
*/
void relational_query12(size_t num)
{
    std::cout << "Relational SQL Query12 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Records: " << num << std::endl;
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;
    std::uniform_int_distribution<uint32_t> shipmode_message(1, 10);
    std::uniform_int_distribution<uint32_t> shipdate_message(10000, 20000);
    std::uniform_int_distribution<uint32_t> receiptdate_message(10000, 20000);
    std::uniform_int_distribution<uint32_t> commitdate_message(10000, 20000);
    // orderpriority \in ('1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW')
    std::uniform_int_distribution<uint64_t> orderpriority_message(1, 5);
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);
    ek.emplacebkfft<Lvl02>(sk);

    // load BK to device
    cufftplvl1.LoadBK(*ek.bkfftlvl01);
    cufftplvl2.LoadBK(*ek.bkfftlvl02);

    // Filtering
    std::vector<uint64_t> shipdate(num), commitdate(num), receiptdate(num);
    std::vector<uint64_t> orderpriority(num), shipmode(num);
    std::vector<ComparableLvl1> shipdate_ciphers(num), commitdate_ciphers(num), receiptdate_ciphers(num);
    std::vector<TRLWELvl1> shipmode_ciphers(num), orderpriority_ciphers(num);
    std::vector<ComparbleRGSWLvl1> receiptdate_rgsw_ciphers(num), commitdate_rgsw_ciphers(num);

    TRGSWLvl1 predicate_mail_cipher, predicate_ship_cipher; // 'MAIL', 'SHIP'
    TRGSWLvl1 predicate_urgent_cipher, predicate_high_cipher; // '1-URGENT', '2-HIGH'
    std::vector<TRGSWLvl1> predicate_date_cipher1(2), predicate_date_cipher2(2);
    uint64_t predicate_mail = 1, predicate_ship= 2, predicate_urgent = 1, predicate_high = 2;
    uint64_t predicate_date1 = 13000, predicate_date2 = 18000;
    exponent_encrypt_rgsw<P>(predicate_mail, predicate_mail_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_ship, predicate_ship_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_urgent, predicate_urgent_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_high, predicate_high_cipher, sk, true);
    exponent_encrypt_rgsw<P>(predicate_date1, 16, predicate_date_cipher1, sk, true);
    exponent_encrypt_rgsw<P>(predicate_date2, 16, predicate_date_cipher2, sk, true);


    // Start sql evaluation
    std::vector<TLWELvl1> filter_res_mail(num), filter_res_ship(num), order_res(num);
    std::vector<TLWELvl1> res_mail_order(num), res_ship_order(num);
    std::vector<TLWELvl1> count_mail(num), count_ship(num), count_mail_order(num), count_ship_order(num);
    TLWELvl1 agg_mail, agg_ship, agg_mail_order, agg_ship_order;
    

    for (size_t i = 0; i < num; i++)
    {
        // Generate data
        shipdate[i] = shipdate_message(engine);
        commitdate[i] = commitdate_message(engine);
        receiptdate[i] = receiptdate_message(engine);
        shipmode[i] = shipmode_message(engine);
        orderpriority[i] = orderpriority_message(engine);
        exponent_encrypt<P>(shipdate[i], 16, shipdate_ciphers[i], sk);
        exponent_encrypt<P>(commitdate[i], 16, commitdate_ciphers[i], sk);
        exponent_encrypt<P>(receiptdate[i], 16, receiptdate_ciphers[i], sk);
        exponent_encrypt<P>(shipmode[i], shipmode_ciphers[i], sk);
        exponent_encrypt<P>(orderpriority[i], orderpriority_ciphers[i], sk);
        exponent_encrypt_rgsw<P>(receiptdate[i], 16, receiptdate_rgsw_ciphers[i], sk, true);
        exponent_encrypt_rgsw<P>(commitdate[i], 16, commitdate_rgsw_ciphers[i], sk, true);
    }

    std::chrono::system_clock::time_point start, end;
    double filtering_time = 0, aggregation_time;
    start = std::chrono::system_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        
        uint32_t stream_id = omp_get_thread_num();

        TLWELvl1 pre_res;
        
        cuARCEDB::less_than_tfhepp(commitdate_ciphers[i], receiptdate_rgsw_ciphers[i], commitdate_ciphers[i].size(), filter_res_mail[i], ek, sk, stream_id);
        cuARCEDB::less_than_tfhepp(shipdate_ciphers[i], commitdate_rgsw_ciphers[i], shipdate_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek, stream_id);
        cuARCEDB::greater_than_tfhepp(receiptdate_ciphers[i], predicate_date_cipher1, receiptdate_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek, stream_id);
        cuARCEDB::less_than_tfhepp(receiptdate_ciphers[i], predicate_date_cipher2, receiptdate_ciphers[i].size(), pre_res, ek, sk, stream_id);
        cuARCEDB::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek, stream_id);
        filter_res_ship[i] = filter_res_mail[i];
        cuARCEDB::equality_tfhepp(shipmode_ciphers[i], predicate_mail_cipher, pre_res,sk, stream_id);
        cuARCEDB::HomAND(filter_res_mail[i], pre_res, filter_res_mail[i], ek, stream_id);

        cuARCEDB::equality_tfhepp(shipmode_ciphers[i], predicate_ship_cipher, pre_res,sk, stream_id);
        cuARCEDB::HomAND(filter_res_ship[i], pre_res, filter_res_ship[i], ek, stream_id);

        cuARCEDB::equality_tfhepp(orderpriority_ciphers[i], predicate_urgent_cipher, order_res[i], sk, stream_id);
        cuARCEDB::equality_tfhepp(orderpriority_ciphers[i], predicate_high_cipher, pre_res, sk, stream_id);
        cuARCEDB::HomOR(order_res[i], pre_res, order_res[i], ek, stream_id);

        cuARCEDB::lift_and_and(filter_res_mail[i], order_res[i], count_mail_order[i], 22, ek, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_ship[i], order_res[i], count_ship_order[i], 22, ek, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_mail[i], filter_res_mail[i], count_mail[i], 22, ek, sk, stream_id);
        cuARCEDB::lift_and_and(filter_res_ship[i], filter_res_ship[i], count_ship[i], 22, ek, sk, stream_id);
        cudaDeviceSynchronize();
    }

    end = std::chrono::system_clock::now();
    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Filter Time on GPU: " << filtering_time << "ms." << std::endl;

    for (size_t i = 0; i < num; i++)
    {
        if (i == 0)
        {
            agg_mail = count_mail[0];
            agg_ship = count_ship[0];
            agg_mail_order = count_mail_order[0];
            agg_ship_order = count_ship_order[0];
        }
        else
        {
            for (size_t j = 0; j <= Lvl1::n; j++)
            {
                agg_mail[j] += count_mail[i][j];
                agg_ship[j] += count_ship[i][j];
                agg_mail_order[j] += count_mail_order[i][j];
                agg_ship_order[j] += count_ship_order[i][j];
            }
            
        }
    }

    uint64_t query_res_mail = tlweSymInt32Decrypt<Lvl1>(agg_mail, std::pow(2.,22), sk.key.get<Lvl1>());
    uint64_t query_res_ship = tlweSymInt32Decrypt<Lvl1>(agg_ship, std::pow(2.,22), sk.key.get<Lvl1>());
    uint64_t query_res_mail_order = tlweSymInt32Decrypt<Lvl1>(agg_mail_order, std::pow(2.,22), sk.key.get<Lvl1>());
    uint64_t query_res_ship_order = tlweSymInt32Decrypt<Lvl1>(agg_ship_order, std::pow(2.,22), sk.key.get<Lvl1>());

    std::cout << "Encrypted result(GPU): " << std::endl;
    std::cout << std::setw(12) <<"shipmode" << "|" << std::setw(16) << "high_line_count" << "|" << std::setw(16) << "low_line_count" << std::endl;
    std::cout << std::setw(12) <<"MAIL" << "|" << std::setw(16) << query_res_mail_order << "|" << std::setw(16) << query_res_mail - query_res_mail_order << std::endl;
    std::cout << std::setw(12) <<"SHIP" << "|" << std::setw(16) << query_res_ship_order << "|" << std::setw(16) << query_res_ship - query_res_ship_order << std::endl;


    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        
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

        equality_tfhepp(orderpriority_ciphers[i], predicate_urgent_cipher, order_res[i], sk);
        equality_tfhepp(orderpriority_ciphers[i], predicate_high_cipher, pre_res, sk);
        TFHEpp::HomOR(order_res[i], pre_res, order_res[i], ek);

        lift_and_and(filter_res_mail[i], order_res[i], count_mail_order[i], 22, ek, sk);
        lift_and_and(filter_res_ship[i], order_res[i], count_ship_order[i], 22, ek, sk);
        lift_and_and(filter_res_mail[i], filter_res_mail[i], count_mail[i], 22, ek, sk);
        lift_and_and(filter_res_ship[i], filter_res_ship[i], count_ship[i], 22, ek, sk);
        
    }
    end = std::chrono::system_clock::now();
    filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Filter Time on CPU: " << filtering_time << "ms." << std::endl;

    for (size_t i = 0; i < num; i++)
    {
        if (i == 0)
        {
            agg_mail = count_mail[0];
            agg_ship = count_ship[0];
            agg_mail_order = count_mail_order[0];
            agg_ship_order = count_ship_order[0];
        }
        else
        {
            for (size_t j = 0; j <= Lvl1::n; j++)
            {
                agg_mail[j] += count_mail[i][j];
                agg_ship[j] += count_ship[i][j];
                agg_mail_order[j] += count_mail_order[i][j];
                agg_ship_order[j] += count_ship_order[i][j];
            }
            
        }
    }

    query_res_mail = tlweSymInt32Decrypt<Lvl1>(agg_mail, std::pow(2.,22), sk.key.get<Lvl1>());
    query_res_ship = tlweSymInt32Decrypt<Lvl1>(agg_ship, std::pow(2.,22), sk.key.get<Lvl1>());
    query_res_mail_order = tlweSymInt32Decrypt<Lvl1>(agg_mail_order, std::pow(2.,22), sk.key.get<Lvl1>());
    query_res_ship_order = tlweSymInt32Decrypt<Lvl1>(agg_ship_order, std::pow(2.,22), sk.key.get<Lvl1>());

    std::cout << "Encrypted result(CPU): " << std::endl;
    std::cout << std::setw(12) <<"shipmode" << "|" << std::setw(16) << "high_line_count" << "|" << std::setw(16) << "low_line_count" << std::endl;
    std::cout << std::setw(12) <<"MAIL" << "|" << std::setw(16) << query_res_mail_order << "|" << std::setw(16) << query_res_mail - query_res_mail_order << std::endl;
    std::cout << std::setw(12) <<"SHIP" << "|" << std::setw(16) << query_res_ship_order << "|" << std::setw(16) << query_res_ship - query_res_ship_order << std::endl;


    std::vector<uint64_t> plain_filter_res_mail(num, 0), plain_filter_res_ship(num, 0), plain_filter_order(num, 0);
    std::vector<uint64_t> plain_res_mail_order(num, 0), plain_res_ship_order(num, 0);
    uint64_t agg_mail_res = 0, agg_mail_order_res = 0, agg_ship_res = 0, agg_ship_order_res = 0;
    bool ress;
    for (size_t i = 0; i < num; i++)
    {
        if (commitdate[i] < receiptdate[i] && shipdate[i] < commitdate[i] && receiptdate[i] > predicate_date1 && receiptdate[i] < predicate_date2)
        {
            ress = true;
        }
        else
        {
            ress = false;
        }

        if (orderpriority[i] == 1 || orderpriority[i] == 2)
        {
            plain_filter_order[i] = 1;
        }

        if (ress && shipmode[i] == 1)
        {
            plain_filter_res_mail[i] = 1;
            agg_mail_res += 1;
            if (plain_filter_order[i] == 1)
            {
                plain_res_mail_order[i] = 1;
                agg_mail_order_res += 1;
            }
        }
        if (ress && shipmode[i] == 2)
        {
            plain_filter_res_ship[i] = 1;
            agg_ship_res += 1;
            if (plain_filter_order[i] == 1)
            {
                plain_res_ship_order[i] = 1;
                agg_ship_order_res += 1;
            }
        }
        

    }

    std::cout << "Filtering finish" << std::endl;
    
    std::cout << "Plain result: " << std::endl;
    std::cout << std::setw(12) <<"shipmode" << "|" << std::setw(16) << "high_line_count" << "|" << std::setw(16) << "low_line_count" << std::endl;
    std::cout << std::setw(12) <<"MAIL" << "|" << std::setw(16) << agg_mail_order_res << "|" << std::setw(16) << agg_mail_res - agg_mail_order_res << std::endl;
    std::cout << std::setw(12) <<"SHIP" << "|" << std::setw(16) << agg_ship_order_res << "|" << std::setw(16) << agg_ship_res - agg_ship_order_res << std::endl;

}


int main()
{
  omp_set_num_threads(num_stream2);
  warmupGPU();

  relational_query12(num);

  return 0;
}


