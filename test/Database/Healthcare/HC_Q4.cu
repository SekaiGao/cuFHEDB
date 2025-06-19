#include "ARCEDB/comparison/comparable.h"
#include "Database/DatabaseParser.h"
#include "Database/cuFHEDB/comparable_gpu.cuh"
#include "HEDB/comparison/comparison.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace HEDB;
using namespace arcedb;
/***
 *  HealthCare Query 4
 *
 *      SELECT Admission_Type,
 *          COUNT(Name) AS Patient_Count
 *      FROM healthcare
 *      WHERE Medical_Condition = "Cancer"
 *          AND Date_of_Admission BETWEEN '2023/12/25' AND '2024/04/01'
 *      GROUP BY Admission_Type
 *      ORDER BY Patient_Count DESC;
 *
 *  Describe: This query counts the number of cancer patients admitted between
 *  2023/12/25 and 2024/04/01, grouped by admission type (Emergency, Elective, Urgent).
 * 
 *  Privacy Concern: The query may risk exposing sensitive information related to
 *  cancer diagnoses and admission types, which could potentially lead to re-identification
 *  of individuals if combined with other identifiable data.
 */

struct PlainTable {
    std::vector<std::string> Name;
    std::vector<int> Age;
    std::vector<int> MedicalCondition;
    std::vector<int> AdmissionType;
    std::vector<int> DateOfAdmission;

    std::unordered_map<std::string, int> s2iAdmissionType;
    std::unordered_map<std::string, int> s2iMedicalCondition;
    std::vector<std::string> i2sAdmissionType;
    std::vector<std::string> i2sMedicalCondition;
};

struct CipherTable {
    std::vector<TRLWELvl1> name_ciphers;
    std::vector<TRLWELvl1> age_ciphers;
    std::vector<TRLWELvl1> medical_condition_ciphers;
    std::vector<TRLWELvl1> admission_type_ciphers;
    std::vector<ComparableLvl1> date_of_admission_ciphers;

    ComparbleRGSWLvl1 date_predicate0, date_predicate1;
    TRGSWLvl1 medical_condition_predicate;
    std::vector<TRGSWLvl1> admission_type_attr_cipher;

    CipherTable(size_t num)
        : name_ciphers(num), age_ciphers(num), 
          medical_condition_ciphers(num), admission_type_ciphers(num),
          date_of_admission_ciphers(num) {}
};

void plaintext_query(const HealthCare& records) {
    std::cout << "Plain query results: \n";
    std::cout << "Admission Type\tPatient Count\n";

    int emergency_count = 0, elective_count = 0, urgent_count = 0;

    // Count the number of patients for each Admission Type
    for (int i = 0; i < records.Age.size(); ++i) {
        if (convertDateToInt(records.DateOfAdmission[i]) > convertDateToInt("2023/12/25") &&
            convertDateToInt(records.DateOfAdmission[i]) < convertDateToInt("2024/04/01")) {
            
            if (records.MedicalCondition[i] == "Cancer") {
                
                if (records.AdmissionType[i] == "Emergency") {
                    emergency_count++;
                } else if (records.AdmissionType[i] == "Elective") {
                    elective_count++;
                } else if (records.AdmissionType[i] == "Urgent") {
                    urgent_count++;
                }
            }
        }
    }

    std::vector<std::pair<std::string, int>> results = {
        {"Emergency", emergency_count},
        {"Elective", elective_count},
        {"Urgent", urgent_count}
    };

    // Sort the results by Patient Count in descending order
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return b.second < a.second; 
    });

    // Print the sorted results
    for (const auto& result : results) {
        std::cout << result.first << "\t" << result.second << std::endl;
    }
}

/*
    Executes Healthcare Query4 to retrieve the count of patients grouped by admission type based on the following conditions:
    - Medical Condition = "Cancer"
    - Date of Admission between '2023/12/25' and '2024/04/01'
    - Group by Admission Type
    - Order by Patient Count (Descending)

    The function performs the following steps:
        1. Loads and encodes the healthcare dataset.
        2. Encrypts the records.
        3. Filters the records based on the specified conditions.
        4. Groups and aggregates the results by Admission Type.
        5. Orders the results by Patient Count.
        6. Decrypts and prints the final results.

    Parameters:
        - `filePath`: Path to the healthcare dataset file.

    Output:
        - Prints the sorted patient counts per admission type, along with the evaluation time.
*/
void healthcare_query4(const std::string &filePath) {
    std::cout << "Healthcare Query4 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    HealthCare records;
    PlainTable encode_records;

    std::chrono::system_clock::time_point start, end;
    double eval_time = 0;
	std::cout<<"Loading database...\n";
    

    processCSV(filePath, records);

    int rows = records.Age.size();
    std::cout << "Records: " << rows << std::endl;

    // Encode
    std::cout << "Start encoding...\n";
    
    Encode(records.MedicalCondition, encode_records.MedicalCondition, encode_records.i2sMedicalCondition, encode_records.s2iMedicalCondition);
    Encode(records.AdmissionType, encode_records.AdmissionType, encode_records.i2sAdmissionType, encode_records.s2iAdmissionType);
    processDate(records.DateOfAdmission, encode_records.DateOfAdmission);
    

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
    ek.emplacebkfft<Lvl02>(sk);
    ek.emplaceiksk<iksP>(sk);
    ek.emplaceiksk<Lvl20>(sk);
    ek.emplaceiksk<Lvl21>(sk);

    // load BK to device
    cufftplvl1.emplaceBK(sk);
    cufftplvl2.emplaceBK(sk);

    int timestamp_bit = static_cast<int>(std::log2(convertDateToInt("2025/12/31"))) + 1;

    // Encrypt predicates
    exponent_encrypt_rgsw<P>(convertDateToInt("2023/12/25"), timestamp_bit, cr.date_predicate0, sk, true);
    exponent_encrypt_rgsw<P>(convertDateToInt("2024/04/01"), timestamp_bit, cr.date_predicate1, sk, true);

    int admission_type_attr_size = encode_records.i2sAdmissionType.size();
    cr.admission_type_attr_cipher.resize(admission_type_attr_size);
    for (int i = 0; i < admission_type_attr_size; ++i)
        exponent_encrypt_rgsw<P>(i, cr.admission_type_attr_cipher[i], sk, true);

    int cancer_index = encode_records.s2iMedicalCondition["Cancer"];
    exponent_encrypt_rgsw<P>(cancer_index, cr.medical_condition_predicate, sk, true);

    // Encrypt records
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        exponent_encrypt<P>(encode_records.DateOfAdmission[i], timestamp_bit, cr.date_of_admission_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.AdmissionType[i], cr.admission_type_ciphers[i], sk);
        exponent_encrypt<P>(encode_records.MedicalCondition[i], cr.medical_condition_ciphers[i], sk);
    }
    end = std::chrono::system_clock::now();
    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    int agg_bits = 15;
    using aggP = Lvl2;
    int agg_scale_bits = std::numeric_limits<aggP::T>::digits - agg_bits - 1;

    // Filtering
    /*
    WHERE Medical_Condition = "Cancer"
 *  AND Date_of_Admission BETWEEN '2023/12/25' AND '2024/04/01'
    */
    std::cout << "Filtering...\n";
    std::vector<std::vector<TFHEpp::TLWE<P>>> filter_res(admission_type_attr_size);
    std::vector<std::vector<TFHEpp::TLWE<aggP>>> agg_filter_res(admission_type_attr_size);
    for (int i = 0; i < admission_type_attr_size; ++i) {
        filter_res[i].resize(rows);
        agg_filter_res[i].resize(rows);
    }

    std::vector<TFHEpp::TLWE<P>> filter_res1(rows);
    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        uint32_t stream_id = omp_get_thread_num();
        TLWELvl1 pre_res0, pre_res1;

        // Filtering by date of admission
        cuFHEDB::greater_than(cr.date_of_admission_ciphers[i], cr.date_predicate0, cr.date_of_admission_ciphers[i].size(), pre_res0, ek,  stream_id);
        cuFHEDB::less_than(cr.date_of_admission_ciphers[i], cr.date_predicate1, cr.date_of_admission_ciphers[i].size(), pre_res1, ek,  stream_id);
        cuFHEDB::HomAND(pre_res0, pre_res1, pre_res0, ek, stream_id);

        // Filtering by medical condition 
        cuFHEDB::equality(cr.medical_condition_ciphers[i], cr.medical_condition_predicate, pre_res1,  stream_id);
        cuFHEDB::HomAND(filter_res1[i], pre_res1, pre_res0, ek, stream_id);
    }
    end = std::chrono::system_clock::now();
    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Group By
    // GROUP BY Admission_Type
    std::cout << "Group By...\n";
    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        uint32_t stream_id = omp_get_thread_num();
        TLWELvl1 pre_res0, pre_res1;
        // Group By Admission Type
        for (int j = 0; j < admission_type_attr_size; ++j) {
            cuFHEDB::equality(cr.admission_type_ciphers[i], cr.admission_type_attr_cipher[j], pre_res1,  stream_id);
            cuFHEDB::HomAND(filter_res[j][i], pre_res1, filter_res1[i], ek, stream_id);
        }
    }
    end = std::chrono::system_clock::now();
    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    // Aggregation
    int aggregation_time=0;
    std::vector<TFHEpp::TLWE<aggP>> agg_res(admission_type_attr_size);
    for (int j = 0; j < admission_type_attr_size; ++j) {
		agg_res[j] = {};
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        uint32_t stream_id = omp_get_thread_num();
        for (int j = 0; j < admission_type_attr_size; ++j) {
            //lift to 15 bit precision to perform homomorphic addition
            lift_and_and(filter_res[j][i], filter_res[j][i], agg_filter_res[j][i], agg_scale_bits, ek, sk);
        }
    }

    std::cout << "COUNT...\n";
    // COUNT(Name) AS Patient_Count
    start = std::chrono::system_clock::now();
	// COUNT(name)
    for (size_t i = 0; i < rows; i++) {
        // Homomorphic Addition
        for (int j = 0; j < admission_type_attr_size; ++j)
			for (int k = 0; k <= aggP::n; ++k) {
				agg_res[j][k] += agg_filter_res[j][i][k];
			}
    }  
    end = std::chrono::system_clock::now();
    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    

	// Order By 
    // ORDER BY Patient_Count DESC
	TLWELvl1 tmp_cres;
	std::vector<TFHEpp::TLWE<aggP>> sort_idx(admission_type_attr_size);
	for (int j = 0; j < admission_type_attr_size; ++j) {
		sort_idx[j] = {};
    }

    std::cout << "Order By...\n";
    start = std::chrono::system_clock::now();
	// Homomorphic Sort
	for (int i=0; i < admission_type_attr_size; ++i) {
		for (int j = 0; j < admission_type_attr_size; ++j) {
			if (i!=j) {
				TFHEpp::TLWE<aggP> tmp_cres1;
                // DESC
				less_than<aggP>(agg_res[i], agg_res[j], tmp_cres, agg_bits, ek, LOGIC);
				lift_and_and(tmp_cres, tmp_cres, tmp_cres1, agg_scale_bits, ek, sk);
				for (int k = 0; k <= aggP::n; ++k) {
				sort_idx[i][k] += tmp_cres1[k];
				}
			}
		}
	}
    end = std::chrono::system_clock::now();
    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// Decryption
    std::cout<<"Decrypting..."<<std::endl;
	int pidx[admission_type_attr_size];
	for (int j = 0; j < admission_type_attr_size; ++j)
		pidx[arcedb::tlweSymInt32Decrypt<aggP>(sort_idx[j], std::pow(2.,agg_scale_bits), sk.key.get<aggP>())] = j;

	uint64_t Patient_Count[admission_type_attr_size] = {0};
	for (int j = 0; j < admission_type_attr_size; ++j)
		Patient_Count[j] = arcedb::tlweSymInt32Decrypt<aggP>(agg_res[j], std::pow(2.,agg_scale_bits), sk.key.get<aggP>());
	

    std::cout << "--------------------------------------------------------"<< std::endl;
    // Query results
    std::cout << "Cipher query results: \n";
    std::cout << "Admission Type\tPatient Count\n";
    for (int j = 0; j < admission_type_attr_size; ++j) {
		int sorted_idx = pidx[j];
        std::cout << encode_records.i2sAdmissionType[sorted_idx] << "\t" << Patient_Count[sorted_idx] << std::endl;
    }

    std::cout << "--------------------------------------------------------"<< std::endl;
    plaintext_query(records);
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Query Evaluation Time: " << eval_time << " ms\n" << std::endl;

}

int main() {

    std::string filePath = "../data/healthcare_dataset.csv";

    healthcare_query4(filePath);

    return 0;
}
