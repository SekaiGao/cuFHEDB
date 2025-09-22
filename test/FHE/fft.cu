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
    This function tests the latency of the FFT operation.

    Parameters:
        - `test_num`: Number of iterations to run for each test.

    Description:
        - Measures kernel launch overhead on the GPU.
        - Benchmarks custom-optimized folding FFT.
        - Benchmarks FFT method from TFHE-rs.
*/

void FFT_test(uint32_t test_num) {
	using P = Lvl1; 

    cufhedb::cuCoreFHE<P> folding_fft(1);
    cufhedb::cuCoreFHE<Lvl2> negacyclic_fft(1);

    TFHEpp::Polynomial<P> tlwefft1, ffta1, ffta2;
    std::array<double, P::n> fftb1;

    TFHEpp::Polynomial<Lvl2> tlwefft2;
    std::array<double, Lvl2::n> fftb2;

    generateData<P>(tlwefft1);
    generateData<Lvl2>(tlwefft2);

    double costs;
    std::chrono::system_clock::time_point start, end;
    
    warmupGPU();
    // Kernel launch overhead test
    start = std::chrono::system_clock::now();
    costs = 0;
    for (int i = 0; i < test_num; ++i) {
        emptyKernel<<<1, 128>>>();
        cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "kernel launch overhead: " << costs / test_num << "μs." << std::endl;

    // GPU IFFT latency test
    folding_fft.ifft_test(fftb1, tlwefft1, test_num, costs);
    std::cout << "folding ifft latency: " << costs / test_num << "μs." << std::endl;

    // GPU FFT latency test
    negacyclic_fft.ifft_test(fftb2, tlwefft2, test_num, costs);
    std::cout << "TFHE-rs ifft latency: " << costs / test_num << "μs." << std::endl;

	// GPU FFT latency test
    folding_fft.fft_test(ffta1, fftb1, test_num, costs);
    std::cout << "folding fft latency: " << costs / test_num << "μs." << std::endl;
	
	// GPU FFT latency test
    folding_fft.fft_shuffle_test(ffta2, fftb1, test_num, costs);
    std::cout << "folding fft (4x threads + warp shuffle) latency: " << costs / test_num << "μs." << std::endl;
}

int main() {
    uint32_t test_num = 100000; 
    
    FFT_test(test_num); 
    return 0;
}
