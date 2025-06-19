#include "HEDB/comparison/comparison.h"
#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "FHE/coreFHE.cuh"
#include <chrono>
#include <cufft.h>
#include <fstream> 
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace HEDB;

template<class P>
void generateData(std::array<typename P::T, P::n> &trlwe) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<typename P::T> dis(0, 184467440737);

    for (int i = 0; i < P::n; i++) {
        uint32_t value = dis(gen);
        trlwe[i] = value;
    }
}

__global__ void emptyKernel() {}


/*
    This function benchmarks the performance of batch FFT operations in FHE computations. 
    Specifically, given a batch size, it evaluates the inter-thread batch FFT inter-block batch FFT performance.

    Parameters:
        - `test_num`: Number of iterations to execute for each benchmark.
        - `batch_size`: Number of FFTs performed simultaneously in a batch.

    Key Operations:
        - Batch IFFT (Inter-Thread): Benchmarks the inter-thread FFT performance.
        - Batch IFFT (Inter-Block): Benchmarks the inter-block FFT performance.

    Output:
        - Benchmark results, including:
            1. Kernel launch overhead.
            2. Average execution time for inter-thread and inter-block FFTs.
            3. Numerical error rates for inter-thread and inter-block FFTs.
*/
int main() {
    // Batch size for Batch FFT 
    constexpr int32_t batch_size = 4;
    uint32_t test_num = 100000; 
    using P = Lvl1;

    cufhedb::cuCoreFHE<P> fft(1);

    warmupGPU();
	
    // Prepare data
    std::vector<TFHEpp::Polynomial<P>> tlwefft(batch_size);
    std::vector<std::array<double, P::n>> fftb0(batch_size), fftb1(batch_size), fftb2(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        generateData<P>(tlwefft[i]);
    }

    double costs;
    std::chrono::system_clock::time_point start, end;

    // Kernel launch overhead benchmark
    start = std::chrono::system_clock::now();
    costs = 0;
    for (int i = 0; i < test_num; ++i) {
        emptyKernel<<<1, 128>>>();
        cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Kernel launch overhead: " << costs / test_num << "μs." << std::endl;

    // Inter-thread FFT benchmark
    fft.ifft_th_test<batch_size>(fftb1, tlwefft, test_num, costs);
    std::cout << "Batch IFFT (Inter-Thread): " << costs / test_num << "μs." << std::endl;

    // Inter-block FFT benchmark
    fft.ifft_blk_test<batch_size>(fftb2, tlwefft, test_num, costs);
    std::cout << "Batch IFFT (Inter-Block): " << costs / test_num << "μs." << std::endl;

    // Error checking
    std::cout << std::fixed << std::setprecision(6) << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    for (int k = 0; k < batch_size; ++k) {
        TFHEpp::TwistIFFT<P>(fftb0[k], tlwefft[k]); 

        double errr1 = 0, errr2 = 0;
        for (int j = 0; j < P::n; ++j) {
            double ref = fftb0[k][j];
            double thread_result = fftb1[k][j];
            double block_result = fftb2[k][j];

            errr1 += std::fabs(ref - thread_result);
            errr2 += std::fabs(ref - block_result);
        }
        
        std::cout << "Batch ID: " << k 
                << ", Inter-Thread Error: " << errr1 
                << ", Inter-Block Error: " << errr2 << std::endl;
    }

    return 0;
}
