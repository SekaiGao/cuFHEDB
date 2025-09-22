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
#include <thread> 
#include <unistd.h>

using namespace HEDB;

// determine whether test 80 bit parameter
#define Test80bitParam 1

template<class P>
void generateData(std::array<typename P::T, P::n + 1> &tlwe) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<typename P::T> dis((1UL << 10), (1UL << 16) - 1);

    for (int i = 0; i <= P::n; i++) {
      typename P::T value = dis(gen);
      tlwe[i] = value;
    }
}

__global__ void emptyKernel() {}

/*
    This function tests the bootstrapping latency of cuFHEDB.
    
    Parameters:
    - `test_num`: The number of bootstrapping operations to be tested for benchmarking.

    Functionality:
    - Evaluates the performance of the bootstrapping process in cuFHEDB under different configurations:
        1. With and without bootstrapping unroll optimization.
        2. With and without CUDA stream concurrency.
        3. Different security level.
    
*/
void Bootstrapping_test(uint32_t test_num) {
    using bkP = Lvl01;
    using P = typename bkP::targetP;
    using bkP2 = Lvl02;
    using P2 = typename bkP2::targetP;

    int max_stream = getSMCount();

    cufhedb::cuCoreFHE<P, false> bs(max_stream);
    cufhedb::cuCoreFHE<P, true> bs2(max_stream);

    cufhedb::cuCoreFHE<P2, false> bs153(max_stream);
    cufhedb::cuCoreFHE<P2, true> bs153_2(max_stream);

    int test_num_st = 5 * test_num;
    std::vector<TFHEpp::TLWE<Lvl0>> tlwe(test_num_st);
    std::vector<TFHEpp::TLWE<P>> resh(test_num_st), resd(test_num_st);
    std::vector<TFHEpp::TLWE<P2>> resh2(test_num_st), resd2(test_num_st);

    TFHESecretKey sk;
    TFHEEvalKey ek;

    // pre-load
    double costs;
    std::chrono::system_clock::time_point start, end;

    ek.emplacebkfft<bkP>(sk);
    ek.emplacebkfft<bkP2>(sk);
    
    bs.emplaceBK(sk);
    bs2.emplaceBK(sk);
    bs153.emplaceBK(sk);
    bs153_2.emplaceBK(sk);

    for (int i = 0; i < test_num_st; ++i) {
      generateData<Lvl0>(tlwe[i]);
    }

    typename P::T u = 1ULL << (std::numeric_limits<typename P::T>::digits - 3);

    // pinned meomory
    for (int i = 0; i < test_num_st; ++i) {
        cudaHostRegister(tlwe[i].data(), (Lvl0::n + 1) * sizeof(typename Lvl0::T), cudaHostRegisterDefault);
        cudaHostRegister(resd[i].data(), (P::n + 1) * sizeof(typename P::T), cudaHostRegisterDefault);
        cudaHostRegister(resd2[i].data(), (P2::n + 1) * sizeof(typename P2::T), cudaHostRegisterDefault);

    }

    // GPU warm up
    for (int i = 0; i < test_num; ++i) {
      warmupGPU();
      cudaDeviceSynchronize();
    }

    #if Test80bitParam
    // 80-bit
    std::cout << "cuFHEDB 80-bit key bundle bootstrapping: " << std::endl;
    cufhedb::cuCoreFHE<TFHEpp::lvl1param80, true> bs80(max_stream);
    std::vector<TFHEpp::TLWE<TFHEpp::lvl0param80>> tlwe80(test_num_st);
    std::vector<TFHEpp::TLWE<TFHEpp::lvl1param80>> resd80(test_num_st);
  
    // generate data
    for (int i = 0; i < test_num_st; ++i)
      generateData<TFHEpp::lvl0param80>(tlwe80[i]);
    // pinned meomory
    for (int i = 0; i < test_num_st; ++i) {
        cudaHostRegister(tlwe80[i].data(), (TFHEpp::lvl0param80::n + 1) * sizeof(typename TFHEpp::lvl0param80::T), cudaHostRegisterDefault);
        cudaHostRegister(resd80[i].data(), (TFHEpp::lvl1param80::n + 1) * sizeof(typename TFHEpp::lvl1param80::T), cudaHostRegisterDefault);
    }

    // bootstrapping
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num_st; ++i)
      bs80.GateBootstrapping(resd80[i], tlwe80[i], -u, i % max_stream);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency (stream): " << costs / test_num_st << "μs." << std::endl;

    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num_st; ++i){
      bs80.GateBootstrapping(resd80[i], tlwe80[i], -u, i % max_stream);
      cudaDeviceSynchronize();
    }
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency: " << costs / test_num_st << "μs." << std::endl;
    
    std::cout << "--------------------------------------------------------"<< std::endl;

    // unpinned
    for (int i = 0; i < test_num; ++i) {
      cudaHostUnregister(tlwe80[i].data());
      cudaHostUnregister(resd80[i].data());
    }
#endif

    std::cout<<"cuFHEDB 128-bit bootstrapping: "<<std::endl;
    start = std::chrono::system_clock::now();
    // employ CUDA stream
    for (int i = 0; i < test_num_st; ++i)
      bs.GateBootstrapping(resd[i], tlwe[i], -u, i % max_stream);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency (stream): " << costs / test_num_st << "μs." << std::endl;

    TFHEpp::TLWE<P> rest;
    TFHEpp::TLWE<P2> rest2;

    // without stream
    bs.GateBootstrapping_test(rest, tlwe[0], -u, test_num, costs);
    std::cout << "Bootstrapping latency: " << costs / test_num << "μs." << std::endl;
    bs.GateBootstrapping_withoutFusion_test(rest, tlwe[0], -u, test_num, costs);
    std::cout << "Bootstrapping latency (without kernel fusion): " << costs / test_num << "μs." << std::endl;
    std::cout << "--------------------------------------------------------"<< std::endl;

    for (int i = 0; i < test_num; ++i) {
      warmupGPU();
      cudaDeviceSynchronize();
    }

    std::cout << "cuFHEDB 128-bit key bundle bootstrapping: " << std::endl;
    start = std::chrono::system_clock::now();
    // employ both bootstrapping unroll and CUDA stream
    for (int i = 0; i < test_num_st; ++i)
      bs2.GateBootstrapping(resd[i], tlwe[i], -u, i % max_stream);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency (stream): " << costs / test_num_st << "μs." << std::endl;

    // only bootstrapping unroll 
    bs2.GateBootstrapping_test(rest, tlwe[0], -u, test_num, costs);
    std::cout << "Bootstrapping latency: " << costs / test_num << "μs." << std::endl;
    
    std::cout << "--------------------------------------------------------"<< std::endl;

    std::cout << "cuFHEDB 153-bit key bundle bootstrapping: " << std::endl;
    start = std::chrono::system_clock::now();
    // employ both bootstrapping unroll and CUDA stream
    for (int i = 0; i < test_num_st; ++i)
      bs153.GateBootstrapping(resd2[i], tlwe[i], -u, i % max_stream);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency (stream): " << costs / test_num_st << "μs." << std::endl;

    // only bootstrapping unroll 
    bs153_2.GateBootstrapping_test(rest2, tlwe[0], -u, test_num, costs);
    std::cout << "Bootstrapping latency: " << costs / test_num << "μs." << std::endl;
    
    std::cout << "--------------------------------------------------------"<< std::endl;

    // CPU baseline
    std::cout << "TFHEpp 128-bit key bundle bootstrapping: " << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i)
        TFHEpp::GateBootstrappingTLWE2TLWEFFT<bkP>(resh[i], tlwe[i], *ek.bkfftlvl01, TFHEpp::μ_polygen<P>(u));
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency: " << costs / test_num << "μs." << std::endl;

    std::cout << "TFHEpp 128-bit key bundle bootstrapping (multi-thread): " << std::endl;
    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < test_num; ++i)
        TFHEpp::GateBootstrappingTLWE2TLWEFFT<bkP>(resh[i], tlwe[i], *ek.bkfftlvl01, TFHEpp::μ_polygen<P>(u));
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping multi-threaded throughput: " << 1000000 * test_num / costs << "TP/s." << std::endl;
    
    std::cout << "--------------------------------------------------------"<< std::endl;

    std::cout << "TFHEpp 153-bit key bundle bootstrapping: " << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < test_num; ++i)
        TFHEpp::GateBootstrappingTLWE2TLWEFFT<bkP2>(resh2[i], tlwe[i], *ek.bkfftlvl02, TFHEpp::μ_polygen<P2>(u));
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping latency: " << costs / test_num << "μs." << std::endl;

    std::cout << "TFHEpp 153-bit key bundle bootstrapping (multi-thread): " << std::endl;
    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < test_num; ++i)
        TFHEpp::GateBootstrappingTLWE2TLWEFFT<bkP2>(resh2[i], tlwe[i], *ek.bkfftlvl02, TFHEpp::μ_polygen<P2>(u));
    end = std::chrono::system_clock::now();
    costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Bootstrapping multi-threaded throughput: " << 1000000 * test_num / costs << "TP/s." << std::endl;

    std::cout << "--------------------------------------------------------"<< std::endl;

    // error check
    double err = 0;
    for (size_t i = 0; i < test_num; ++i) {
        uint32_t tp0, tp1;
        tp0=TFHEpp::tlweSymDecrypt<P>(resh[i], sk.key.get<P>());
        tp1=TFHEpp::tlweSymDecrypt<P>(resd[i], sk.key.get<P>());
        if(tp0 != tp1)
            err++;
    }

    std::cout<<"error rate: "<<100*err/test_num<<"%\n";

    // unpinned
    for (int i = 0; i < test_num; ++i) {
      cudaHostUnregister(tlwe[i].data());
      cudaHostUnregister(resd[i].data());
    }
}

int main() {
    const uint32_t test_num = 1000;
    
    Bootstrapping_test(test_num);

    return 0;
}
