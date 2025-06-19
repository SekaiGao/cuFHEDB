#include "HEDB/comparison/comparison.h"
#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "FHE/coreFHE.cuh"
#include <chrono>
#include <cufft.h>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace HEDB;

template<class P>
void generateData(std::array<std::array<typename P::T, P::n>, 2> &trlwe) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<typename P::T> dis(0, 58983720);

    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < P::n; i++) {
            uint32_t value = dis(gen);
            trlwe[k][i] = value;
        }
}

template<class P>
void generateData(std::array<std::array<std::array<double, P::n>, 2>, (P::k + 1) * P::l> &trgswfft) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1844677.0, 18446747.0);

    for (int i = 0; i < (P::k + 1) * P::l; i++) {
        for (int j = 0; j < 2 * P::n; j++) {
            double value = dis(gen);
            if (j < P::n) {
                trgswfft[i][0][j] = value;
            } else {
                trgswfft[i][1][j - P::n] = value;
            }
        }
    }
}

template<class P>
void generateData(TFHEpp::TRGSWNTT<P> &trgswntt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, 5898372418983290);

    for (int i = 0; i < (P::k + 1) * P::l; i++) {
        for (int j = 0; j < 2 * P::n; j++) {
            double value = dis(gen);
            if (j < P::n) {
                trgswntt[i][0][j] = value;
            } else {
                trgswntt[i][1][j - P::n] = value;
            }
        }
    }
}

__global__ void emptyKernel() {}


/*
    This function tests the latency of the external product operation.

    Parameters:
        - `test_num`: Number of iterations to run for each test.

    Template:
        - The function is templated on `P`, supporting `Lvl01` (parameter sets II) and `Lvl02` (parameter sets III).

    Description:
        - Measures kernel launch overhead on the GPU.
        - Benchmarks GPU-based inter-block external product.
        - Benchmarks CPU-based inter-thread external product from cuFHE and nuFHE.
*/
template <typename P>
void ExternalProduct_test(uint32_t test_num) {

    cufhedb::cuCoreFHE<P> ep(1);

    TFHEpp::TRLWE<P> trlwefft, resh, resd, resntt;
    TFHEpp::TRGSWFFT<P> trgswfft;
    TFHEpp::TRGSWNTT<P> trgswntt;

    generateData<P>(trlwefft);
    generateData<P>(trgswfft);
    generateData<P>(trgswntt);

    double costs;
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    costs = 0;

    // Kernel launch overhead test
    for (int i = 0; i < test_num; ++i) {
        emptyKernel<<<6, 64>>>();
        cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "kernel launch overhead: " << costs / test_num << "μs." << std::endl;
    
    for (int i = 0; i < test_num; ++i)
        warmupGPU();
    
    // External Product tests
    ep.ExternalProduct_withoutFusion_test(resd, trlwefft, trgswfft, test_num, costs);
    std::cout << "Inter-block external product latency (without kernel fusion): " << costs / test_num << "μs." << std::endl;

    // Inter-Block External Product
    ep.ExternalProduct_test(resd, trlwefft, trgswfft, test_num, costs);
    std::cout << "Inter-block external product latency: " << costs / test_num << "μs." << std::endl;

    // Inter-Thread External Product
    ep.ExternalProduct_th_test(resd, trlwefft, trgswfft, test_num, costs);
    std::cout << "Inter-thread external product latency: " << costs / test_num << "μs." << std::endl;

}

int main() {
    uint32_t test_num = 100000; 
    using P = Lvl1;
    ExternalProduct_test<P>(test_num);
    return 0;
}
