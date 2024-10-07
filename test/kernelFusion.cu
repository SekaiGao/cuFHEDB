#include "HEDB/comparison/comparison.h"
#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/conversion/repack.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "cuHEDB/HomCompare_gpu.cuh"
#include <fstream>
#include <chrono>
#include <cufft.h>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace HEDB;
using namespace seal;

template<class P>
void generateData(std::array<std::array<typename P::T, P::n>, 2> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<typename P::T> dis(0, 58983724);//1898329

  for (int k = 0; k < 2; ++k)
    for (int i = 0; i < P::n; i++) {
      uint32_t value = dis(gen);
      trlwe[k][i] = value;
    }
}

void generateData(std::array<uint32_t, Lvl0::n + 1> &tlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis((1UL<<10), (1UL<<30)-1);

  for (int i = 0; i <= Lvl0::n; i++) {
    uint32_t value = dis(gen);
    tlwe[i] = value;
  }
}

template<class P>
void generateData(TFHEpp::TRGSWFFT<P> &trgswfft1) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-181234560.0, 181234560.0);

  for (int i = 0; i < (P::k+1) * P::l; i++) {
    for (int j = 0; j < 2 * P::n; j++) {
      double value = dis(gen);
      if (j < P::n) {
        trgswfft1[i][0][j] = value;
      } else {
        trgswfft1[i][1][j - P::n] = value;
      }
    }
  }
}

template<class P>
void generateData(TFHEpp::BootstrappingKeyFFT<P> &bkfft) {
  for (int k = 0; k < P::domainP::n; ++k) {
    generateData<typename P::targetP>(bkfft[k]);
  }
}

template<class P>
void generateData(TFHEpp::TRGSWNTT<Lvl1> &trgswntt) {
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

__global__ void emptyKernel() {
}

int main()
{
  uint32_t test_num = 100000;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  TFHEpp::TLWE<Lvl0> tlwe;
  TRLWELvl1 trlwefft, resd2, resd1;


  TFHEpp::TRGSWFFT<Lvl1> trgswfft;
  generateData<Lvl1>(trlwefft);
  generateData<Lvl1>(trgswfft);

  // for external product
  double costs;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  costs = 0;
  for (int i = 0; i < test_num; ++i) {
    emptyKernel<<<6, 64>>>();
    cudaDeviceSynchronize();
  }
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "kernel launch overhead: " << costs / test_num << "μs." << std::endl;

  
  cufftlvl1.ExternalProduct_withoutFusion_test(resd2, trlwefft, trgswfft, test_num, costs);

  std::cout << "external product cpu(without kernel fusion): " << costs / test_num << "μs." << std::endl;

 
  cufftlvl1.ExternalProduct_test(resd1, trlwefft, trgswfft, test_num, costs);

  std::cout << "external product gpu: " << costs / test_num << "μs." << std::endl;


  test_num = 1000;
  // for bootstrapping
  TLWELvl1 res1, res2;
  TFHEpp::BootstrappingKeyFFT<Lvl01> *bkfft = new TFHEpp::BootstrappingKeyFFT<Lvl01>;

  generateData<Lvl01>(*bkfft);

  // pre-load
  cufftlvl1.LoadBK(*bkfft);

  generateData(tlwe);

  Lvl1::T u = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 3);

  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_withoutFusion_test(res1, tlwe, -u, test_num, costs);
  std::cout << "Bootstrapping gpu(without kernel fusion): " << costs / test_num << "μs." << std::endl;

  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_test(res2, tlwe, -u, test_num, costs);
  std::cout << "Bootstrapping gpu: " << costs / test_num << "μs." << std::endl;

  delete bkfft;

  return 0;
}
