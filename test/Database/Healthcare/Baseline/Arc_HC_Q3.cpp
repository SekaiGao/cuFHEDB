#include "ARCEDB/comparison/comparable.h"
#include "Database/DatabaseParser.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace arcedb;

/***
 *  HealthCare Query 3

        SELECT Gender, 
            COUNT(*) AS Patient_Count,
            AVG(Age) AS Avg_Age
        FROM healthcare
        WHERE Test_Results = 'Abnormal'
            AND Billing_Amount < 10000.0
        GROUP BY Gender
        ORDER BY Gender;
 *
 *  Describe: This query counts the number of patients with abnormal test results 
 *  and a billing amount under 10,000, grouped by gender. It also calculates the average 
 *  age of patients in each gender group.
 * 
 *  Privacy Concern: The query involves sensitive health information such as abnormal 
 *  test results and patient age, raising privacy risks.
 */


struct PlainTable {
    std::vector<int> Age;
    std::vector<int> Gender;
    std::vector<float> BillingAmount;
    std::vector<int> TestResults;

    std::vector<std::string> i2sGender, i2sTestResults;
    std::unordered_map<std::string, int> s2iGender, s2iTestResults;
};

struct CipherTable {
    std::vector<TLWELvl2> age_ciphers;
    std::vector<TRLWELvl1> gender_ciphers;
    std::vector<ComparableLvl1> bill_ciphers;
    std::vector<TRLWELvl1> test_results_ciphers;


    // predicate
    std::vector<TRGSWLvl1> bill_predicate;
    TRGSWLvl1 gender_male, gender_female, test_results_predicate;

    CipherTable(size_t num)
        : bill_ciphers(num), age_ciphers(num), gender_ciphers(num),
        test_results_ciphers(num), bill_predicate(2) {}
};

void plaintext_query(const HealthCare& records) {
    int count_male = 0, count_female = 0;
    float sum_male_age = 0.0f, sum_female_age = 0.0f;

    for (size_t i = 0; i < records.Age.size(); ++i) {
        // TestResults = 'Abnormal' AND BillingAmount < 500
        if (records.TestResults[i] == "Abnormal" && records.BillingAmount[i] < 3000.0) {
            if (records.Gender[i] == "Male") {  // Male
                count_male++;
                sum_male_age += records.Age[i];
            } else {  // Female
                count_female++;
                sum_female_age += records.Age[i];
            }
        }
    }

    std::cout << "Plaintext query result:\n";
    std::cout << "Gender\t|\t" << "Patient_Count\t|\t" << "Avg_Age\n";
    std::cout << "Male" << "\t|\t" << count_male << "\t|\t" << sum_male_age / count_male << std::endl;
    std::cout << "Female" << "\t|\t" << count_female << "\t|\t" << sum_female_age / count_female << std::endl;
}

/*
    This function executes the Healthcare Query3.
    It performs the following operations:
    - Loads the healthcare dataset from the provided file path.
    - Encodes and encrypts the records.
    - Filters the data based on certain conditions (TestResults and BillingAmount).
    - Performs group-by operations and aggregates the data (Patient Count and Avg Age).
    - Decrypts the aggregated results and compares them with the plaintext results.

    Parameters:
        - `filePath`: The path to the healthcare dataset file.

    Key Operations:
        - Loading and encoding the dataset.
        - Encrypting the input data using homomorphic encryption.
        - Filtering and group-by operations.
        - Aggregating the data using homomorphic addition and multiplication.
        - Decrypting the final results.

    Output:
        - Prints the query results.
*/
void healthcare_query3(const std::string &filePath) {
    std::cout << "ArcEDB Healthcare Query3 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    std::chrono::system_clock::time_point start, end;
    HealthCare records;
	PlainTable encode_records;

	std::cout<<"Loading database...\n";
    
    processCSV(filePath, records);
    double eval_time = 0;


	int rows = records.Age.size();
    std::cout << "Records: " << rows << std::endl;

	// Encode
	std::cout<<"Start encoding...\n";

    encode_records.Age = records.Age;
    encode_records.BillingAmount = records.BillingAmount;
    Encode(records.Gender, encode_records.Gender, encode_records.i2sGender, encode_records.s2iGender);
    Encode(records.TestResults, encode_records.TestResults, encode_records.i2sTestResults, encode_records.s2iTestResults);
    

	// Encrypt
	std::cout<<"Start encrypting...\n";
    
    CipherTable cr(rows);
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);
    ek.emplacebkfft<Lvl02>(sk);

    start = std::chrono::system_clock::now();

    exponent_encrypt_rgsw<P>(3000, 20, cr.bill_predicate, sk, true);
    exponent_encrypt_rgsw<P>(1, cr.gender_male, sk, true);
    exponent_encrypt_rgsw<P>(0, cr.gender_female, sk, true);
    exponent_encrypt_rgsw<P>(encode_records.s2iTestResults["Abnormal"], cr.test_results_predicate, sk, true);

	for (size_t i = 0; i < rows; i++) {
        arcedb::tlweSymInt32Encrypt<Lvl2>(encode_records.Age[i], cr.age_ciphers[i], std::pow(2.,48), sk.key.get<Lvl2>());
        exponent_encrypt<P>(encode_records.Gender[i], cr.gender_ciphers[i], sk);
        exponent_encrypt<P>(std::ceil(encode_records.BillingAmount[i]), 20, cr.bill_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.TestResults[i], cr.test_results_ciphers[i], sk);
    }
    end = std::chrono::system_clock::now();
    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<TLWELvl1> filter_res0(rows);
    int agg_scale_bits = 48;
    using aggP = Lvl2;
    std::vector<TFHEpp::TLWE<P>> filter_res1(rows), filter_res2(rows);
    std::vector<TFHEpp::TLWE<aggP>> agg_filter_res1(rows), agg_filter_res2(rows);

    // Filtering
    /*
    WHERE Test_Results = 'Abnormal'
        AND Billing_Amount < 10000.0
    */
    std::cout << "Filtering..." << std::endl;
    double filtering_time_d = 0, aggregation_time = 0;
    start = std::chrono::system_clock::now(); 
    for (size_t i = 0; i < rows; i++) {
        TLWELvl1 pre_res;
        equality_tfhepp(cr.test_results_ciphers[i], cr.test_results_predicate, pre_res, sk);
        less_than_tfhepp(cr.bill_ciphers[i], cr.bill_predicate, 2, filter_res0[i], ek, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);
    }
    end = std::chrono::system_clock::now();

    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Group By
    std::cout << "Group By..." << std::endl;
    start = std::chrono::system_clock::now();
    // GROUP BY Gender
    for (size_t i = 0; i < rows; i++)
    {
        TLWELvl1 pre_res;
    
		equality_tfhepp(cr.gender_ciphers[i], cr.gender_male, pre_res, sk);
        TFHEpp::HomAND(filter_res1[i], pre_res, filter_res0[i], ek);
        TFHEpp::HomNOT(pre_res, pre_res);
        TFHEpp::HomAND(filter_res2[i], pre_res, filter_res0[i], ek);
        
    }
    end = std::chrono::system_clock::now();

    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // aggregate
    TFHEpp::TLWE<aggP> agg_res1={}, agg_res2={}, agg_res3={}, agg_res4={}, mul_res;
    for (size_t i = 0; i < rows; i++) {
        //lift to 15 bit precision for homomorphic addition
        lift_and_and(filter_res1[i], filter_res1[i], agg_filter_res1[i], agg_scale_bits, ek, sk);
        lift_and_and(filter_res2[i], filter_res2[i], agg_filter_res2[i], agg_scale_bits, ek, sk);
    }
    
    std::cout << "COUNT..." << std::endl;
    start = std::chrono::system_clock::now();
    // COUNT(*)
    for (size_t i = 0; i < rows; i++) {
        // homomorphic addition
        for (int k=0; k <= aggP::n; ++k) {
            agg_res1[k] += agg_filter_res1[i][k];
            agg_res2[k] += agg_filter_res2[i][k];
        }
    }
    // AVG(Age)
    std::cout << "AVG..." << std::endl;
    for (size_t i = 0; i < rows; i++) {
        multiply<aggP>(filter_res1[i], cr.age_ciphers[i], mul_res, ek, sk);
        // homomorphic addition
		for (int k=0; k <= aggP::n; ++k) {
            agg_res3[k] += mul_res[k];
        }
        multiply<aggP>(filter_res2[i], cr.age_ciphers[i], mul_res, ek, sk);
        // homomorphic addition
		for (int k=0; k <= aggP::n; ++k) {
            agg_res4[k] += mul_res[k];
        }
	}
    end = std::chrono::system_clock::now();
    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();



    uint64_t cipher_agg_res0 = 0, count0 = 0, cipher_agg_res1 = 0, count1 = 0;

    std::cout << "Decrypting..." << std::endl;
    count1 = tlweSymInt32Decrypt<aggP>(agg_res1, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    count0 = tlweSymInt32Decrypt<aggP>(agg_res2, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    cipher_agg_res1 = tlweSymInt32Decrypt<aggP>(agg_res3, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    cipher_agg_res0 = tlweSymInt32Decrypt<aggP>(agg_res4, std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
    
	// Query results
    std::cout << "--------------------------------------------------------"<< std::endl;
	std::cout << "Cipher query result:\n";
	std::cout << "Gender\t|\t" << "Patient_Count\t|\t" << "Avg_Age\n";
	std::cout << encode_records.i2sGender[1] <<"\t|\t" << count1 <<"\t|\t" << float(cipher_agg_res1)/count1<<std::endl;
	std::cout << encode_records.i2sGender[0] <<"\t|\t" << count0 <<"\t|\t" << float(cipher_agg_res0)/count0<<std::endl;

    std::cout << "--------------------------------------------------------"<< std::endl;
    plaintext_query(records);
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Query Evaluation Time: "<<eval_time<<" ms\n" << std::endl; 
}

int main() {

    std::string filePath = "../data/healthcare_dataset.csv";

    healthcare_query3(filePath);

    return 0;
}