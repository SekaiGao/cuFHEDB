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
void generateData(TFHEpp::TRGSWNTT<Lvl2> &trgswntt) {
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
  uint32_t test_num = 10000;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  TRLWELvl2 trlwefft, resh, resd, resntt;


  TFHEpp::TRGSWFFT<Lvl2> trgswfft;
  TFHEpp::TRGSWNTT<Lvl2> trgswntt;
  generateData<Lvl2>(trlwefft);
  generateData<Lvl2>(trgswfft);
  generateData<Lvl2>(trgswntt);

  double costs;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  costs = 0;
  for (int i = 0; i < test_num; ++i) {
    emptyKernel<<<8, 128>>>();
    cudaDeviceSynchronize();
  }
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "kernel launch overhead: " << costs / test_num << "μs." << std::endl;

  
  cufftlvl2.ExternalProduct_test(resd, trlwefft, trgswfft, test_num, costs);


  std::cout << "external product gpu(fft): " << costs / test_num << "μs." << std::endl;

  
  start = std::chrono::system_clock::now();
  
  for (int i=0;i<test_num;++i)
        TFHEpp::trgswfftExternalProduct<Lvl2>(resh, trlwefft, trgswfft);
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "external product cpu(fft): " << costs / test_num << "μs." << std::endl;

  //ntt
  start = std::chrono::system_clock::now();
  
  for (int i=0;i<test_num;++i)
    TFHEpp::trgswnttExternalProduct<Lvl2>(resntt, trlwefft, trgswntt);
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "external product cpu(ntt): " << costs / test_num << "μs." << std::endl;
 
  double err =0;
  for(int k=0;k<2;++k)
  for(int i=0;i<Lvl2::n;++i) {
	  err+=fabs(double(resh[k][i])-double(resd[k][i]));
  }
  std::cout<<"error: "<<err<<std::endl;

  return 0;
}
