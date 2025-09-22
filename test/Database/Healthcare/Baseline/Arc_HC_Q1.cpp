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
 *  HealthCare Query 1

        SELECT Name, Age, Medication, Date_of_Admission
        FROM healthcare
        WHERE Medical_Condition = 'Diabetes'
            AND Gender = 'Female'
            AND Age BETWEEN 60 AND 80
            AND Test_Results = 'Normal'
            AND Date_of_Admission > '2023/12/31';
 *
 *  Describe: This query retrieves the names, ages, medications, and dates of admission 
 *  for female patients aged between 60 and 80 with normal test results and diagnosed 
 *  with diabetes, who were admitted after 2023/12/31.
 * 
 *  Privacy Concern: The query returns sensitive personal details, including names, ages, 
 *  and medical conditions, potentially compromising patient privacy if not handled securely.
 */

constexpr int num_thread = 96; // setting multi threads

struct PlainTable {
    std::vector<std::string> Name;
    std::vector<int> Age;
    std::vector<int> Gender;
    std::vector<int> TestResults;
    std::vector<int> Medication;
    std::vector<int> MedicalCondition;
    std::vector<int> DateOfAdmission;

    std::vector<std::string> i2sGender, i2sMedicalCondition, i2sTestResults;
    std::unordered_map<std::string, int> s2iGender, s2iMedicalCondition, s2iTestResults;
};

struct CipherTable {
    std::vector<TRLWELvl1> name_ciphers;
    std::vector<TRLWELvl1> age_ciphers;
    std::vector<TRLWELvl1> gender_ciphers;
    std::vector<TRLWELvl1> test_results_ciphers;
    std::vector<TRLWELvl1> medical_condition_ciphers;
    std::vector<ComparableLvl1> date_of_admission_ciphers;

    ComparbleRGSWLvl1 date_predicate;
    TRGSWLvl1 medical_condition_predicate, test_results_predicate, age_predicate0, age_predicate1, gender_predicate;

    CipherTable(size_t num)
        : name_ciphers(num), age_ciphers(num), gender_ciphers(num), 
        test_results_ciphers(num), medical_condition_ciphers(num),
        date_of_admission_ciphers(num) {}
};

void plaintext_query(const HealthCare& records) {
    std::cout << "Plain query results: \n";
    std::cout << "Name\tAge\tGender\tMedication\tDate of Admission\n";
    for (size_t i = 0; i < records.Age.size(); ++i) {
        if (records.MedicalCondition[i] == "Diabetes" && records.Gender[i] == "Female" && records.Age[i] > 60 && records.Age[i] < 80) {
            if (records.TestResults[i] == "Normal" && convertDateToInt(records.DateOfAdmission[i]) > convertDateToInt("2023/12/31")) {
                std::cout << records.Name[i] << "\t"
                        << records.Age[i] << "\t"
                        << records.Gender[i] << "\t"
                        << records.Medication[i] << "\t"
                        << records.DateOfAdmission[i] << std::endl;
            }
        }
    }
}

/*
    Executes Healthcare Query1 to retrieve records based on the following conditions:
    - Medical Condition = 'Diabetes'
    - Gender = 'Female'
    - Age between 60 and 80
    - Test Results = 'Normal'
    - Date of Admission > '2023/12/31'

    The function performs the following steps:
        1. Loads and encodes the healthcare dataset.
        2. Encrypts the records.
        3. Filters the records based on the specified conditions.
        4. Decrypts the filtered results.
        5. Compares the query results with the plaintext query.

    Parameters:
        - `filePath`: Path to the healthcare dataset file.

    Output:
        - Prints the filtered records and the query evaluation time.
*/
void healthcare_query1(const std::string &filePath) {

	omp_set_num_threads(num_thread);
	
    std::cout << "ArcEDB Healthcare Query1 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    HealthCare records;
    PlainTable encode_records;

    // Load database
    std::chrono::system_clock::time_point start, end;
    double eval_time = 0;
	std::cout<<"Loading database...\n";
    
    
    processCSV(filePath, records);

    int rows = records.Age.size();
    std::cout << "Records: " << rows << std::endl;

    // Encode
    std::cout << "Start encoding...\n";
    encode_records.Name = records.Name;
    encode_records.Age = records.Age;
    processDate(records.DateOfAdmission, encode_records.DateOfAdmission);

    Encode(records.Gender, encode_records.Gender, encode_records.i2sGender, encode_records.s2iGender);
    Encode(records.TestResults, encode_records.TestResults, encode_records.i2sTestResults, encode_records.s2iTestResults);
    Encode(records.MedicalCondition, encode_records.MedicalCondition, encode_records.i2sMedicalCondition, encode_records.s2iMedicalCondition);
    

    // Encrypt
    std::cout << "Start encrypting...\n";
    start = std::chrono::system_clock::now();
    CipherTable cr(rows);
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using P = Lvl1;
    using bkP = Lvl01;
    using iksP = Lvl10;
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);

    int timestamp_bit = static_cast<int>(std::log2(convertDateToInt("2025/12/31"))) + 1;

    // Encryption of predicates
    exponent_encrypt_rgsw<P>(encode_records.s2iMedicalCondition["Diabetes"], cr.medical_condition_predicate, sk, true);
    exponent_encrypt_rgsw<P>(encode_records.s2iTestResults["Normal"], cr.test_results_predicate, sk, true);
    exponent_encrypt_rgsw<P>(60, cr.age_predicate0, sk, true);
    exponent_encrypt_rgsw<P>(80, cr.age_predicate1, sk, true);
    exponent_encrypt_rgsw<P>(convertDateToInt("2023/12/31"), timestamp_bit, cr.date_predicate, sk, true);
    exponent_encrypt_rgsw<P>(encode_records.s2iGender["Female"], cr.gender_predicate, sk, true);

    // Encrypt records
    for (size_t i = 0; i < records.Age.size(); ++i) {
        exponent_encrypt<P>(encode_records.Age[i], cr.age_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.Gender[i], cr.gender_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.TestResults[i], cr.test_results_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.MedicalCondition[i], cr.medical_condition_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.DateOfAdmission[i], timestamp_bit, cr.date_of_admission_ciphers[i], sk);
    }
    end = std::chrono::system_clock::now();
    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    

    // Filtering
    /*
    WHERE Medical_Condition = 'Diabetes'
        AND Gender = 'Female'
        AND Age BETWEEN 60 AND 80
        AND Test_Results = 'Normal'
        AND Date_of_Admission > '2023/12/31'
     */
    std::cout << "Filtering...\n";
    std::vector<TLWELvl1> filter_res0(records.Age.size());

    double filtering_time_d = 0;
    start = std::chrono::system_clock::now();
	#pragma omp parallel for 
    for (size_t i = 0; i < records.Age.size(); i++) {
        TLWELvl1 pre_res;

        // Filtering by medical condition
        equality_tfhepp(cr.medical_condition_ciphers[i], cr.medical_condition_predicate, filter_res0[i], sk);

        // Filtering by age
        greater_than_tfhepp(cr.age_ciphers[i], cr.age_predicate0, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);
        less_than_tfhepp(cr.age_ciphers[i], cr.age_predicate1, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);

        // Filtering by Gender
        equality_tfhepp(cr.gender_ciphers[i], cr.gender_predicate, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);

        // Filtering by test results
        equality_tfhepp(cr.test_results_ciphers[i], cr.test_results_predicate, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);

        // Filtering by date of admission
        greater_than_tfhepp(cr.date_of_admission_ciphers[i], cr.date_predicate,cr.date_of_admission_ciphers[i].size(), pre_res, ek, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);
    }
    end = std::chrono::system_clock::now();
    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Decrypting..." << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    // Query results
    std::cout << "Cipher query results: \n";
    std::cout << "Name\tAge\tGender\tMedication\tDate of Admission\n";
    
    for (size_t i = 0; i < records.Age.size(); ++i) {
        if (TFHEpp::tlweSymDecrypt<Lvl1>(filter_res0[i], sk.key.get<Lvl1>())) {
            std::cout << records.Name[i] << "\t"
                      << records.Age[i] << "\t"
                      << records.Gender[i] << "\t"
                      << records.Medication[i] << "\t"
                      << records.DateOfAdmission[i] << std::endl;
        }
    }
    
    std::cout << "--------------------------------------------------------"<< std::endl;
    plaintext_query(records);
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Query Evaluation Time: " << eval_time << " ms\n" << std::endl;
}

int main() {

    std::string filePath = "../data/healthcare_dataset.csv";

    healthcare_query1(filePath);
    
    return 0;
}
