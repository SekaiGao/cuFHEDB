#include "ARCEDB/comparison/comparable.h"
#include "Database/cuFHEDB/comparable_gpu.cuh"
#include "HEDB/comparison/comparison.h"
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace HEDB;
using namespace TFHEpp;

template <int bits>
using T = typename std::conditional<(bits < 10), Lvl1, Lvl2>::type;

/*
    Benchmarks the performance of homomorphic comparison operators, such as "greater than" and "equality,"
    across various FHE-based database schemes. This function evaluates and compares the performance of 
    cuFHEDB against state-of-the-art (SOTA) FHE-based database implementations, including ArcEDB and HE3DB.

    Parameters:
        - `test_num`: Number of test cases to evaluate for homomorphic comparisons.

    Template:
        - `bits`: Determines the plaintext precision in bits (e.g., 8-bit, 10-bit, or 16-bit).

    Output:
        - Prints performance metrics, including:
            1. Latency (in milliseconds per operation) for each FHE implementation.
            2. Error rates for "greater than" and "equality" operations.

    Key Operations:
        - Key Generation: Generates secret and evaluation keys for encryption and decryption.
        - Data Encryption: Encrypts randomly generated plaintexts into ciphertexts for testing.
        - Comparison Operators: Benchmarks homomorphic "greater than" and "equality" operations.
*/
template<int bits>
void homomorphic_comparison(int test_num) {

    std::cout<<"plaintext precision: "<<bits<<"bit.\n";
    using P = T<bits>;

    omp_set_num_threads(omp_get_max_threads());

    std::cout << "Generate Secret Key..." << std::endl;

    double costs;
    std::chrono::system_clock::time_point start, end;
    
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    TFHESecretKey sk;
    TFHEEvalKey ek;
    ek.emplacebkfft<Lvl01>(sk);
    ek.emplaceiksk<Lvl10>(sk);
    if (bits >= 10) {
        ek.emplacebkfft<Lvl02>(sk);
        ek.emplaceiksk<Lvl20>(sk);
        ek.emplaceiksk<Lvl21>(sk);
    }

    std::cout << "Loading..." << std::endl;
    start = std::chrono::system_clock::now();

    // load BK to device
    cufftplvl1.emplaceBK(sk);

    std::cout << "Load Success." << std::endl;
    

    start = std::chrono::system_clock::now();
    typename Lvl2::T pt3;
    std::vector<typename Lvl2::T> pt1(test_num);
    std::vector<uint32_t> resh1(test_num), resh2(test_num), resd(test_num);
    std::vector<ComparableLvl1> ct1(test_num);
    std::vector<TLWE<P>> ct2(test_num);
    ComparbleRGSWLvl1 ct3;
    TLWE<P> ct4;
    std::vector<TLWE<TFHEpp::lvl1param>> cresh11(test_num), cresh21(test_num), cresd1(test_num);
    std::vector<TLWE<TFHEpp::lvl1param>> cresh12(test_num), cresh22(test_num), cresd2(test_num);

    std::uniform_int_distribution<typename Lvl1::T> data_distribution(0, (1 << bits) - 1);

    for (int i = 0; i < test_num; ++i) {
        pt1[i] = data_distribution(engine);
    }
    pt3 = data_distribution(engine);

    uint32_t scale_bits = std::numeric_limits<typename P::T>::digits - bits - 1;

    std::cout << "Encrypting..." << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < test_num; ++i) {
        exponent_encrypt<Lvl1>(pt1[i], bits, ct1[i], sk);
        ct2[i] = TFHEpp::tlweSymInt32Encrypt<P>(pt1[i], P::α, pow(2., scale_bits), sk.key.get<P>());
    }
    exponent_encrypt_rgsw<Lvl1>(pt3, bits, ct3, sk, true);
    ct4 = TFHEpp::tlweSymInt32Encrypt<P>(pt3, P::α, pow(2., scale_bits), sk.key.get<P>());

    for (int i = 0; i < test_num; ++i) {
        warmupGPU();
        cudaDeviceSynchronize();
    }

    std::cout << "Benchmarking Homomorphic Comparisons..." << std::endl;

    std::cout << "cuFHEDB HCMP..." << std::endl;
    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < test_num; ++i) {
        cuFHEDB::greater_than(ct1[i], ct3, ct1[i].size(), cresd1[i], ek, omp_get_thread_num());
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "homomorphic greater than: " << costs / (1000 * test_num) << "ms" << std::endl;
    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < test_num; ++i) {
        cuFHEDB::equality(ct1[i], ct3, ct1[i].size(), cresd2[i], ek, omp_get_thread_num());
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end -
    start).count(); std::cout << "homomorphic equality: " << costs /
    (1000 * test_num) << "ms" << std::endl;
    
    test_num /= 10;

    std::cout << "ArcEDB HCMP..." << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i) {
        greater_than_tfhepp(ct1[i], ct3, ct1[i].size(), cresh11[i], ek, sk);
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "homomorphic greater than: " << costs / (1000 * test_num) << "ms" << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i) {
        equality_tfhepp(ct1[i], ct3, ct1[i].size(), cresh12[i], ek, sk);
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); 
    std::cout << "homomorphic equality: " << costs /(1000 * test_num) << "ms" << std::endl;

    test_num /= 10;

    std::cout << "HE3DB HCMP..." << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i) {
        greater_than<P>(ct2[i], ct4, cresh21[i], bits, ek, LOGIC);
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "homomorphic greater than: " << costs / (1000 * test_num) << "ms" << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i) {
        equal<P>(ct2[i], ct4, cresh22[i], bits, ek, LOGIC);
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "homomorphic greater than: " << costs / (1000 * test_num) << "ms" << std::endl;


    // Error checking
    double times = 0;
    for (int i = 0; i < test_num; ++i) {
        resd[i] = tlweSymDecrypt<TFHEpp::lvl1param>(cresd1[i], sk.key.get<TFHEpp::lvl1param>());
        resh1[i] = tlweSymDecrypt<TFHEpp::lvl1param>(cresh11[i], sk.key.get<TFHEpp::lvl1param>());
        resh2[i] = TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(cresh21[i], sk.key.get<TFHEpp::lvl1param>());

        uint32_t res00 = (pt1[i] > pt3);
        if (resd[i] != res00 || resh1[i] != res00 || resh2[i] != res00) {
            std::cout<<pt1[i] <<'>' <<pt3<<", "<<resd[i]<<", " <<resh1[i]<<", " <<resh2[i]<<std::endl;
            times++;
        }
    }
    std::cout << "greater than error rate: " << 100 * times / test_num << "%." << std::endl;

    times = 0;
    for (int i = 0; i < test_num; ++i) {
        resd[i] = tlweSymDecrypt<TFHEpp::lvl1param>(cresd2[i], sk.key.get<TFHEpp::lvl1param>());
        resh1[i] = tlweSymDecrypt<TFHEpp::lvl1param>(cresh12[i], sk.key.get<TFHEpp::lvl1param>());
        resh2[i] = TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(cresh22[i], sk.key.get<TFHEpp::lvl1param>());

        uint32_t res00 = (pt1[i] == pt3);
        if (resd[i] != res00 || resh1[i] != res00 || resh2[i] != res00) {
            std::cout<<pt1[i] <<'==' <<pt3<<", "<<resd[i]<<", " <<resh1[i]<<", " <<resh2[i]<<std::endl;
            times++;
        }
    }
    std::cout << "equality error rate: " << 100 * times / test_num << "%." << std::endl;
}

int main() {
    int test_num = 10000;
    std::cout << "--------------------------------------------------------"<< std::endl;
    homomorphic_comparison<4>(test_num);
    std::cout << "--------------------------------------------------------\n\n\n\n"<< std::endl;
    homomorphic_comparison<8>(test_num);
    std::cout << "--------------------------------------------------------\n\n\n\n"<< std::endl;
    homomorphic_comparison<16>(test_num);
    std::cout << "--------------------------------------------------------\n\n\n\n"<< std::endl;
    homomorphic_comparison<32>(test_num);
    std::cout << "--------------------------------------------------------"<< std::endl;
    return 0;
}
