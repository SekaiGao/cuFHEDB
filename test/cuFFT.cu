#include "HEDB/comparison/comparison.h"
#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/conversion/repack.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "cuHEDB/HomCompare_gpu.cuh"
//#include "fastR.h"
#include <chrono>
#include <cufft.h>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace HEDB;
using namespace seal;

template<class P>
void generateData(std::array<typename P::T, P::n> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<typename P::T> dis(0, 18446744073700);

  for (int i = 0; i < P::n; i++) {
    uint32_t value = dis(gen);
    trlwe[i] = value;
  }
}

template<class P>
void computeIFFT(const std::array<typename P::T, P::n>& input, std::array<double, P::n>& output,uint32_t test_num, double&cost) {
    std::vector<double> input_double(P::n);  
    for (int i = 0; i < P::n; i++) {
        input_double[i] = static_cast<double>(input[i]);
    }

    std::chrono::system_clock::time_point start, end;

    cufftDoubleComplex* d_input;
    cufftDoubleComplex* d_output;
    cudaMalloc(&d_input, sizeof(cufftDoubleComplex) * P::n);
    cudaMalloc(&d_output, sizeof(cufftDoubleComplex) * P::n);
    
    cudaMemcpy(d_input, input_double.data(), sizeof(cufftDoubleComplex) * P::n, cudaMemcpyHostToDevice);


    cufftHandle plan;

	  start = std::chrono::system_clock::now();

	  for(int i=0;i<test_num;++i) {
	  	cufftPlan1d(&plan, P::n, CUFFT_Z2Z, 1);
      cufftExecZ2Z(plan, d_input, d_output, CUFFT_INVERSE);
	  	cudaDeviceSynchronize();
	  }

	  end = std::chrono::system_clock::now();
    cost = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


    std::vector<cufftDoubleComplex> host_output(P::n);
    cudaMemcpy(host_output.data(), d_output, sizeof(cufftDoubleComplex) * P::n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < P::n; i++) {
        output[i] = host_output[i].x / P::n; 
    }

    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void emptyKernel() {
}

int main()
{
  std::cout<<"1\n";
  #if 0
  uint32_t test_num = 1000;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  TFHEpp::Polynomial<Lvl1> tlwefft;
  std::array<double, Lvl1::n> fftb1, fftb2;

  generateData<Lvl1>(tlwefft);

  double costs;
  std::chrono::system_clock::time_point start, end;

  start = std::chrono::system_clock::now();
  costs = 0;
  for (int i = 0; i < test_num; ++i) {
    emptyKernel<<<1, 64>>>();
    cudaDeviceSynchronize();
  }
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "kernel launch overhead: " << costs / test_num << "μs." << std::endl;

  
  cufftlvl1.ifft_test(fftb2, tlwefft, test_num, costs);

  std::cout << "negacyclic fft gpu: " << costs / test_num << "μs." << std::endl;
  computeIFFT<Lvl1>(tlwefft, fftb1, test_num, costs);
  std::cout << "cufft: " << costs / test_num << "μs." << std::endl;

  start = std::chrono::system_clock::now();
  costs = 0;
  for (int i = 0; i < test_num; ++i) {
    TFHEpp::TwistIFFT<Lvl1>(fftb1, tlwefft);
  }
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "negacyclic fft cpu: " << costs / test_num << "μs." << std::endl;


  double err =0;
  for(int i=0;i<Lvl1::n;++i) {
	err+=fabs(fftb1[i]-fftb2[i]);
  }
  std::cout<<"error: "<<err<<std::endl;
  #endif
  return 0;
}
