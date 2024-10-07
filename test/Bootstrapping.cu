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

__global__ void emptyKernel() {
}

int main()
{
  uint32_t test_num = 1000;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  TFHEpp::TLWE<Lvl0> tlwe;
  TFHEpp::TLWE<Lvl1> resh, resd;

  TFHEpp::BootstrappingKeyFFT<Lvl01> *bkfft = new TFHEpp::BootstrappingKeyFFT<Lvl01>;

  generateData<Lvl01>(*bkfft);

  // pre-load
  cufftlvl1.LoadBK(*bkfft);

  generateData(tlwe);

  Lvl1::T u = 1ULL << (std::numeric_limits<Lvl1::T>::digits - 3);

  double costs;
  std::chrono::system_clock::time_point start, end;

  cufftlvl1.GateBootstrappingTLWE2TLWEFFT_test(resd, tlwe, -u, test_num, costs);

  std::cout << "Bootstrapping gpu: " << costs / test_num << "μs." << std::endl;

  
  start = std::chrono::system_clock::now();
  
  for (int i=0;i<test_num;++i)
    TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl01>(resh, tlwe, *bkfft, TFHEpp::μ_polygen<Lvl1>(u));
  
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Bootstrapping cpu: " << costs / test_num << "μs." << std::endl;

  //ntt
  
  double err =0;
  for(int i=0;i<Lvl0::n;++i) {
	  err+=fabs(double(resh[i])-double(resd[i]));
  }
  std::cout<<"error: "<<err<<std::endl;

  delete bkfft;

  return 0;
}
