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
 *  HealthCare Query 2

        SELECT Name, Age, Gender, Insurance_Provider, Medication
        FROM healthcare
        WHERE Insurance_Provider LIKE '%care%'
            AND Age BETWEEN 10 AND 17
            AND Test_Results = 'Normal';
 *
 *  Describe: This query retrieves the names, ages, genders, insurance providers, 
 *  and medications of patients aged between 10 and 17 who have normal test results 
 *  and whose insurance provider related to "care".
 * 
 *  Privacy Concern: The query includes identifiable personal information such as 
 *  names, ages, and insurance providers, which may expose sensitive data if mishandled.
 */


struct PlainTable {
	std::vector<std::string> Name;
    std::vector<int> Age;
    std::vector<std::string> Gender;
    std::vector<std::vector<int>> InsuranceProvider;
    std::vector<int> TestResults;
    std::vector<std::string> Medication;

    std::vector<std::string> i2sPattern, i2sTestResults;
    std::unordered_map<std::string, int> s2iPattern, s2iTestResults;
};



struct CipherTable {
	std::vector<TRLWELvl1> name_ciphers;
    std::vector<TRLWELvl1> age_ciphers;
    std::vector<TRLWELvl1> gender_ciphers;
    std::vector<std::vector<TRLWELvl1>> InsuranceProvider_ciphers;
    std::vector<TRLWELvl1> test_results_ciphers;
    std::vector<TRLWELvl1> medication_ciphers;

    TRGSWLvl1 pattern_predicate, test_results_predicate, age_predicate0, age_predicate1;

    CipherTable(size_t num):
        name_ciphers(num), age_ciphers(num), gender_ciphers(num), InsuranceProvider_ciphers(num),
        test_results_ciphers(num), medication_ciphers(num) {}
};


void plaintext_query(const HealthCare& records) {
    std::cout << "Plain query results: \n";
    std::cout << "Name\tAge\tGender\tInsurance Provider\tMedication\n";
    std::string likepattern = "care";
    for (size_t i = 0; i < records.Age.size(); ++i) {
        if (records.TestResults[i] == "Normal" && records.Age[i] >= 10 && records.Age[i] <= 17) {
            bool insuranceMatch = false;

            if (is_match(records.InsuranceProvider[i], likepattern)) {
                insuranceMatch = true; 
            }
            
            if (insuranceMatch) 
            {
                std::cout << records.Name[i] << "\t"
                        << records.Age[i] << "\t"
                        << records.Gender[i] << "\t"
                        << records.InsuranceProvider[i] << "\t"
                        << records.Medication[i] << std::endl;
            }
        }
    }
}

/*
    This function executes the Healthcare Query2.
    It performs the following operations:
    - Loads the healthcare dataset from the provided file path.
    - Encodes and encrypts the records.
    - Applies the `LIKE` pattern matching on certain field, filters the data based on the conditions.
    - Decrypts the results and compares them with the plaintext results.

    Parameters:
        - `filePath`: The path to the healthcare dataset file.

    Key Operations:
        - Loading and encoding the dataset.
        - Encrypting the input data using homomorphic encryption.
        - Performing pattern matching for the `Insurance_Provider` using the `LIKE` operator.
        - Filtering data based on the `Age` and `Test_Results` conditions.
        - Decrypting the final results and printing them.

    Output:
        - Prints the query results.
*/
void healthcare_query2(const std::string &filePath) {
    std::cout << "ArcEDB Healthcare Query2 Test: "<< std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    HealthCare records;
	PlainTable encode_records;

    std::chrono::system_clock::time_point start, end;
    double eval_time = 0;

	std::cout<<"Loading database...\n";

    processCSV(filePath, records);
    
    // match pattern
	std::string LikePattern = "care";

    //displayData(records);

	int rows = records.Age.size();
	std::cout << "Records: " << rows << std::endl;


	// Encode
	std::cout<<"Start encoding...\n";
	encode_records.Name = records.Name;
    encode_records.Age = records.Age;
    encode_records.Gender = records.Gender;
    encode_records.Medication = records.Medication;
    Encode(records.TestResults, encode_records.TestResults, encode_records.i2sTestResults, encode_records.s2iTestResults);
	genLike(records.InsuranceProvider, encode_records.InsuranceProvider, LikePattern, encode_records.s2iPattern, encode_records.i2sPattern);
    

	// Encrypt
	std::cout<<"Start encrypting...\n";
    start = std::chrono::system_clock::now();
    CipherTable cr(rows);
    using P = Lvl1;
    TFHESecretKey sk;
    TFHEEvalKey ek;
    using bkP = Lvl01;
    using iksP = Lvl10;
    ek.emplacebkfft<bkP>(sk);
    ek.emplaceiksk<iksP>(sk);
    ek.emplacebkfft<Lvl02>(sk);

    exponent_encrypt_rgsw<P>(10, cr.age_predicate0, sk, true);
    exponent_encrypt_rgsw<P>(18, cr.age_predicate1, sk, true);
    exponent_encrypt_rgsw<P>(encode_records.s2iPattern[LikePattern], cr.pattern_predicate, sk, true);
    exponent_encrypt_rgsw<P>(encode_records.s2iTestResults["Normal"], cr.test_results_predicate, sk, true);

	for (size_t i = 0; i < rows; i++) {
		exponent_encrypt<P>(encode_records.Age[i], cr.age_ciphers[i], sk);
		int sizeIP = encode_records.InsuranceProvider[i].size();
		cr.InsuranceProvider_ciphers[i].resize(sizeIP);
		for(int j=0;j<sizeIP;++j)
			exponent_encrypt<P>(encode_records.InsuranceProvider[i][j], cr.InsuranceProvider_ciphers[i][j], sk);
        exponent_encrypt<P>(encode_records.TestResults[i], cr.test_results_ciphers[i], sk);
    }
    end = std::chrono::system_clock::now();
    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<TLWELvl1> filter_res0(rows);

    
    // Filtering
    /*
    WHERE Insurance_Provider LIKE '%care%'
     */
    std::cout << "LIKE..." << std::endl;
    double filtering_time_d = 0;
    start = std::chrono::system_clock::now();

    for (size_t i = 0; i < rows; i++)
    {
        TLWELvl1 pre_res;

		equality_tfhepp(cr.InsuranceProvider_ciphers[i][0], cr.pattern_predicate, filter_res0[i], sk);

        // word-wise matching for LIKE
        for (int j=1;j<cr.InsuranceProvider_ciphers[i].size();++j){
            equality_tfhepp(cr.InsuranceProvider_ciphers[i][j], cr.pattern_predicate, pre_res, sk);
            TFHEpp::HomOR(filter_res0[i], pre_res, filter_res0[i], ek);
        }
    }

    end = std::chrono::system_clock::now();

    eval_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Filtering..." << std::endl;
    /**
    WHERE AND Age BETWEEN 10 AND 17
        AND Test_Results = 'Normal'
    */
    filtering_time_d = 0;
    start = std::chrono::system_clock::now();

    for (size_t i = 0; i < rows; i++)
    {
        TLWELvl1 pre_res;
        // age
        less_than_tfhepp(cr.age_ciphers[i], cr.age_predicate1, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);
		greater_than_tfhepp(cr.age_ciphers[i], cr.age_predicate0, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);

        // test results
        equality_tfhepp(cr.test_results_ciphers[i], cr.test_results_predicate, pre_res, sk);
        TFHEpp::HomAND(filter_res0[i], pre_res, filter_res0[i], ek);
    }
    end = std::chrono::system_clock::now();

    eval_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Decrypting..." << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;
    
    // Query results decrypt
    std::cout << "Cipher query results: \n";
    std::cout << "Name\tAge\tGender\tInsurance Provider\tMedication\n";
    for (size_t i = 0; i < rows; i++)
    {
        if (TFHEpp::tlweSymDecrypt<Lvl1>(filter_res0[i], sk.key.get<Lvl1>())) {
			std::cout << records.Name[i] << "\t"
                        << records.Age[i] << "\t"
                        << records.Gender[i] << "\t"
                        << records.InsuranceProvider[i] << "\t"
                        << records.Medication[i] << std::endl;
        }
	}
    std::cout << "--------------------------------------------------------"<< std::endl;
    plaintext_query(records);
    std::cout << "--------------------------------------------------------"<< std::endl;
    std::cout << "Query Evaluation Time: " << eval_time << " ms\n" << std::endl;
}

int main() {

    std::string filePath = "../data/healthcare_dataset.csv";
    
    healthcare_query2(filePath);

    return 0;
}
