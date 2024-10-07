#include "ARCEDB/comparison/batch_bootstrap.h"
#include "ARCEDB/comparison/comparable.h"
#include "ARCEDB/comparison/rgsw_ciphertext.h"
#include "ARCEDB/conversion/packlwes.h"
#include "ARCEDB/conversion/repack.h"
#include "ARCEDB/utils/serialize.h"
#include "cuHEDB/comparable_gpu.cuh"
#include "fastR.h"
#include <chrono>
#include <cufft.h>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace arcedb;
using namespace seal;

void generateData(std::array<uint32_t, 673> &tlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 10000000);

  for (int i = 0; i < 673; i++) {
    uint32_t value = dis(gen);
    tlwe[i] = value;
  }
}

void generateData(std::array<uint32_t, 1025> &tlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 100000);

  for (int i = 0; i < 1025; i++) {
    uint32_t value = dis(gen);
    tlwe[i] = value;
  }
}

void generateData(std::array<uint64_t, 2049> &tlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, 10000);

  for (int i = 0; i < 2049; i++) {
    uint32_t value = dis(gen);
    tlwe[i] = value;
  }
}

void generateData(std::array<uint32_t, 1024> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(440467440, 484467440);

  for (int i = 0; i < 1024; i++) {
      uint32_t value = dis(gen);
      trlwe[i] = value;
  }
}

void generateData(std::array<double, 2048> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 100000100149967295);

  for (int i = 0; i < 2048; i++) {
    uint32_t value = dis(gen);
    trlwe[i] = value;
  }
}

void generateData(std::array<uint64_t, 2048> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(0, 18446744073709551615);

  for (int i = 0; i < 2048; i++) {
    uint32_t value = dis(gen);
    trlwe[i] = value;
  }
}

void generateData(std::array<std::array<uint32_t, 1024>, 2> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1898329, 58983724);

  for (int k=0;k<2;++k)
  for (int i = 0; i < 2048; i++) {
    uint32_t value = dis(gen);
    trlwe[k][i] = value;
  }
}

void generateData(TRGSWLvl1 &trgswfft1) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-18446770.0, 184467470.0);

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2 * 1024; j++) {
      double value = dis(gen);
      if (j < 1024) {
        trgswfft1[i][0][j] = value;
      } else {
        trgswfft1[i][1][j - 1024] = value;
      }
    }
  }
}

void generateData(TFHEpp::BootstrappingKeyFFT<Lvl01> &bkfft) {
  for (int k = 0; k < Lvl01::domainP::n; ++k) {
    generateData(bkfft[k]);
  }
}

void generateData(TRGSWLvl2 &trgswfft1) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(58983907294246.0, 5898390729424560.0);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 2 * 2048; j++) {
      double value = dis(gen);
      if (j < 2048) {
        trgswfft1[i][0][j] = value;
      } else {
        trgswfft1[i][1][j - 2048] = value;
      }
    }
  }
}

void generateData(TFHEpp::BootstrappingKeyFFT<Lvl02> &bkfft) {
  for (int k = 0; k < Lvl02::domainP::n; ++k) {
    generateData(bkfft[k]);
  }
}

template <class P> 
typename P::T offsetgen_test() {
  typename P::T offset = 0;
  for (int i = 1; i <= P::l; i++)
    offset += P::Bg / 2 *
              (1ULL << (std::numeric_limits<typename P::T>::digits - i * P::Bgbit));
  return offset;
}

template <uint32_t data>
int bits_needed_test()
{
    uint32_t value = data;
    int bits = 0;
    for (int bit_test = 16; bit_test > 0; bit_test >>= 1) {
        if (value >> bit_test != 0) {
            bits += bit_test;
            value >>= bit_test;
        }
    }
    return bits + value;
}

template<class P>
void test() {
   typename P::T offset = offsetgen_test<P>();
   typename P::T roundoffset = 1ULL << (std::numeric_limits<typename P::T>::digits - P::l * P::Bgbit - 1);
   typename P::T digits = std::numeric_limits<typename P::T>::digits;
   typename P::T mask = static_cast<typename P::T>((1ULL << P::Bgbit) - 1);
   typename P::T halfBg = (1ULL << (P::Bgbit - 1));
   typename P::T Bgbit = P::Bgbit;

   std::cout << "offset: " << offset << std::endl;
   std::cout << "roundoffset: " << roundoffset << std::endl;
   std::cout << "digits: " << digits << std::endl;
   std::cout << "mask: " << mask << std::endl;
   std::cout << "halfBg: " << halfBg << std::endl;
   std::cout << "Bgbit: " << Bgbit << std::endl;
  }

  template <class P> void test1() {
    // 计算 bitwidth
    uint32_t bitwidth = bits_needed_test<0>();
    std::cout << "bitwidth: " << bitwidth << std::endl;

    // 计算 b
    uint32_t b = (((std::numeric_limits<typename P::domainP::T>::digits - 1 -
                    P::targetP::nbit + bitwidth))
                  << bitwidth);
    std::cout << "b: " << b << std::endl;

    // 计算 roundoffset
    typename P::domainP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                 P::targetP::nbit + bitwidth);
    std::cout << "roundoffset: " << roundoffset << std::endl;

    // 计算 a
    uint32_t a = (std::numeric_limits<typename P::domainP::T>::digits - 1 -
                  P::targetP::nbit + bitwidth)
                 << bitwidth;
    std::cout << "a: " << a << std::endl;
  }

  template <class P>
void DecompositionPolynomialFFT_test(
    TFHEpp::DecomposedPolynomialInFD<P> &decpolyfft,
    const TFHEpp::Polynomial<P> &poly, const int digit) {
  TFHEpp::Polynomial<P> decpoly;
  TFHEpp::DecompositionPolynomial<P>(decpoly, poly, digit);
  //printf("CPU: %u\n", poly[0]);
  TFHEpp::TwistIFFT<P>(decpolyfft, decpoly);
}

template <class P>
void trgswfftExternalProduct_test(TFHEpp::TRLWE<P> &res,
                                  const TFHEpp::TRLWE<P> &trlwe,
                                  const TFHEpp::TRGSWFFT<P> &trgswfft,
                                  TFHEpp::TRGSWFFT<P> &trgswfftres) {
  TFHEpp::PolynomialInFD<P> decpolyfft{};
  TFHEpp::TRLWEInFD<P> restrlwefft{};
  for (int i = 0; i < 4; i++) {
    __builtin_prefetch(trgswfft[i].data());
    DecompositionPolynomialFFT_test<P>(decpolyfft, trlwe[0], i);
    TFHEpp::FMAInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[i][0]);
    TFHEpp::FMAInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[i][1]);
    // TFHEpp::MulInFD<P::n>(trgswfftres[i][0], decpolyfft, trgswfft[i][0]);
    // TFHEpp::MulInFD<P::n>(trgswfftres[i][1], decpolyfft, trgswfft[i][1]);
  }
  for (int i = 0; i < 4; i++) {
    __builtin_prefetch(trgswfft[i + 4].data());
    DecompositionPolynomialFFT_test<P>(decpolyfft, trlwe[1], i);
    TFHEpp::FMAInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[i + 4][0]);
    TFHEpp::FMAInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[i + 4][1]);
    // TFHEpp::MulInFD<P::n>(trgswfftres[i + 4][0], decpolyfft, trgswfft[i + 4][0]);
    // TFHEpp::MulInFD<P::n>(trgswfftres[i + 4][1], decpolyfft, trgswfft[i + 4][1]);
  }

  // for(int i=0;i<2048;++i){
  //   res[0][i] = int64_t(restrlwefft[0][i]);
  //   res[1][i] = int64_t(restrlwefft[1][i]);
  // }
  TFHEpp::TwistFFT<P>(res[0], restrlwefft[0]);
  TFHEpp::TwistFFT<P>(res[1], restrlwefft[1]);
}

template <class P>
TFHEpp::Polynomial<P> gpolygen(uint32_t plain_bits, uint32_t scale_bits) {
  TFHEpp::Polynomial<P> poly;
  uint32_t padding_bits = P ::nbit - plain_bits;
  for (int i = 0; i < P::n; i++)
    poly[i] = (1ULL << scale_bits) * (i >> padding_bits);
  return poly;
}

template <class P, uint32_t num_out = 1>
void BlindRotate_test(
    TFHEpp::TRLWE<typename P::targetP> &res,
    const TFHEpp::TLWE<typename P::domainP> &tlwe,
    const TFHEpp::BootstrappingKeyFFT<P> &bkfft,
    const TFHEpp::Polynomial<typename P::targetP> &testvector) {
  constexpr uint32_t bitwidth = TFHEpp::bits_needed<num_out - 1>();
  const uint32_t b̄ = 2 * P::targetP::n -
                     ((tlwe[P::domainP::k * P::domainP::n] >>
                       (std::numeric_limits<typename P::domainP::T>::digits -
                        1 - P::targetP::nbit + bitwidth))
                      << bitwidth);
  res = {};
  TFHEpp::PolynomialMulByXai<typename P::targetP>(res[P::targetP::k],
                                                  testvector, b̄);
  for (int i = 0; i < 672; i++) {
      constexpr typename P::domainP::T roundoffset =
          1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                   P::targetP::nbit + bitwidth);
      const uint32_t ā =
          (tlwe[i] + roundoffset) >>
          (std::numeric_limits<typename P::domainP::T>::digits - 1 -
           P::targetP::nbit + bitwidth)
              << bitwidth;
      if (ā == 0) continue;
      // Do not use CMUXFFT to avoid unnecessary copy.
      TFHEpp::CMUXFFTwithPolynomialMulByXaiMinusOne<typename P::targetP>(res,
                                                                 bkfft[i],
                                                                 ā);
  }
}

void GateBootstrappingTLWE2TLWEFFT_test(
    std::array<uint64_t, 2049> &res, const std::array<uint32_t, 673> &tlwe,
    const TFHEpp::BootstrappingKeyFFT<Lvl02> &bkfft,
    const TFHEpp::Polynomial<Lvl2> &testvector) {
  TRLWELvl2 acc;
  BlindRotate_test<Lvl02>(acc, tlwe, bkfft, testvector);
  TFHEpp::SampleExtractIndex<Lvl2>(res, acc, 0);
}

void GateBootstrappingTLWE2TLWEFFT_test(
    std::array<uint32_t, 1025> &res, const std::array<uint32_t, 673> &tlwe,
    const TFHEpp::BootstrappingKeyFFT<Lvl01> &bkfft,
    const TFHEpp::Polynomial<Lvl1> &testvector) {
  TRLWELvl1 acc;
  BlindRotate_test<Lvl01>(acc, tlwe, bkfft, testvector);
  TFHEpp::SampleExtractIndex<Lvl1>(res, acc, 0);
}

constexpr int N = 2048;

void computeIFFT(const std::array<uint64_t, N>& input, std::array<double, N>& output, double&cost) {
    // 1. 分配和转换输入数据
    std::vector<double> input_double(N);  // 将输入整数转换为双精度浮点数
    for (int i = 0; i < N; i++) {
        input_double[i] = static_cast<double>(input[i]);
    }
  
    // 2. 分配CUDA内存
    cufftDoubleComplex* d_input;
    cufftDoubleComplex* d_output;
    cudaMalloc(&d_input, sizeof(cufftDoubleComplex) * N);
    cudaMalloc(&d_output, sizeof(cufftDoubleComplex) * N);
    


    // 3. 将输入数据从主机传输到设备
    cudaMemcpy(d_input, input_double.data(), sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice);

     // 4. 创建CUFFT计划（使用双精度）
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);  // CUFFT_Z2Z 用于双精度复数变换

    // 5. 执行逆FFT（CUFFT执行的是正向的，所以需要指定CUFFT_INVERSE进行逆向变换）
    cufftExecZ2Z(plan, d_input, d_output, CUFFT_INVERSE);

    // 6. 将结果传回主机
    std::vector<cufftDoubleComplex> host_output(N);
    cudaMemcpy(host_output.data(), d_output, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost);

    // 7. 将复数结果的实部转换为双精度浮点数，存储到输出数组中，并进行归一化
    for (int i = 0; i < N; i++) {
        output[i] = host_output[i].x / N;  // 取实部并进行归一化
    }

    // 8. 清理资源
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);

    
}

int main()
{
  uint32_t test_num = 1;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  std::array<uint64_t, 2048> tlwefft;
  std::array<double, 2048> fftb1, fftb2;
  std::array<std::array<uint32_t, 1024>, 2> trlwefft, res3, res4, res5;
  std::array<std::array<double, 2048>, 2>resfft{};

  TLWELvl0 tlwe;
  generateData(tlwe);
  TLWELvl2 resh, resd;

  TRGSWLvl2 trgswfft, buf1, buf2;
  generateData(trlwefft);
  generateData(trgswfft);

  //TFHEpp::BootstrappingKeyFFT<TFHEpp::lvl02param> *bkfft = new TFHEpp::BootstrappingKeyFFT<TFHEpp::lvl02param>;
  //TFHEpp::KeySwitchingKey<TFHEpp::lvl10param> *isk = new TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>;  

  //generateData(*bkfft);

  uint64_t u = Lvl2::μ;

  TFHESecretKey sk;
  TFHEEvalKey ek;
  //ek.emplacebkfft<Lvl01>(sk);
  //ek.emplacebkntt<Lvl01>(sk);
  // //ek.emplaceiksk<Lvl10>(sk);
  // ek.emplacebkfft<Lvl02>(sk);

  //cufftplvl2.LoadBK(*bkfft);

  uint32_t scale_bits = 29, plain_bits = 5;

  //TRGSWLvl2 &buf1 = (*ek.bkfftlvl02)[100];

  //GateBootstrappingTLWE2TLWEFFT_test(resh, tlwe, *bkfft, μ_polygen<Lvl2>(u));
  //cufftplvl2.GateBootstrappingTLWE2TLWEFFT_st(resd, tlwe, -u, 0);
  //TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl02>(resh, tlwe, *bkfft, gpolygen<Lvl2>(plain_bits, scale_bits));
  //cufftplvl2.IdeGateBootstrappingTLWE2TLWEFFT_st(resd, tlwe, scale_bits, 0);

  //TFHEpp::TwistIFFT<Lvl1>(fftb1, tlwefft);
  //cufftplvl1.ifft_st(fftb2, tlwefft, 0);

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  double costs;
  for (int i=0;i<1000;++i)
    //cufftplvl1.ExternalProduct_st(res3, trlwefft, (*ek.bkfftlvl01)[10], 0);
   //TFHEpp::trgswfftExternalProduct<Lvl1>(res3, trlwefft, (*ek.bkfftlvl01)[10]);
   cufftplvl2.ifft_st(fftb2, tlwefft, 0);
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout<<"fft: "<<costs/1000<<std::endl;
  start = std::chrono::system_clock::now();
  costs = 0;
  for (int i = 0; i < 1000; ++i)
    //TFHEpp::trgswnttExternalProduct<Lvl1>(res4, trlwefft, (*ek.bknttlvl01)[10]);
    computeIFFT(tlwefft, fftb1, costs);
    //TFHEpp::TwistIFFT<Lvl1>(fftb1, tlwefft);
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "cufft: " << costs / 1000 << std::endl;
  //trgswfftExternalProduct_test<Lvl2>(res3, trlwefft, trgswfft, buf1);
  //cufftplvl2.ExternalProduct_st(res4, trlwefft, trgswfft, 0);
#if 0
  double toldf = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2048; j++) {
      double diff = double(res3[i][j]) - double(res4[i][j]);
      toldf += fabs(diff);
    }
  std::cout << "i = " << 0 << " ,err = " << toldf << std::endl;

  for(int i=0;i<1;++i){
    trgswfftExternalProduct_test<Lvl2>(res3, res3, trgswfft, buf1);
    cufftplvl2.ExternalProduct_st(res4, res4, trgswfft, 0);

    double toldf = 0;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2048; j++) {
        double diff = double(res3[i][j]) - double(res4[i][j]);
        toldf += fabs(diff);
      }
      if (toldf > 1e7)
        break;
    std::cout<<"i = "<<i+1<<" ,err = "<<toldf<<std::endl;
  }
  cudaDeviceSynchronize();
  cufftplvl2.MoveBuf(buf2, 0);

  for(int i=1;i<8;++i){
    for(int j=0;j<2048;++j){
      resfft[0][j] += buf2[i][0][j];
      resfft[1][j] += buf2[i][1][j];
    }
  }
  TFHEpp::TwistFFT<Lvl2>(res5[0], resfft[0]);
  TFHEpp::TwistFFT<Lvl2>(res5[1], resfft[1]);
#endif
  // for (int j = 0; j < 2048; ++j) {
  //   res5[0][j] = int64_t(resfft[0][j]);
  //   res5[1][j] = int64_t(resfft[1][j]);
  // }

  //test1<Lvl01>();
  //test1<Lvl02>();

#if 0
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  double costs[4]={0,0,0,0};

  uint32_t u = uint32_t(1ULL << 15);

  // CPU
  start = std::chrono::system_clock::now();
  //#pragma omp parallel for 
  for (int i = 0; i < test_num; ++i) {
    //TFHEpp::HomAND(resh, cipher1, cipher2, ek);
    TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl01>(resh, tlwe, *ek.bkfftlvl01, μ_polygen<Lvl1>(u));
    // TFHEpp::PolyMul<Lvl1>(res1[0], trlwea[0], trlweb);
    // TFHEpp::PolyMul<Lvl1>(res1[1], trlwea[1], trlweb);
    //TFHEpp::TwistIFFT<Lvl1>(fftb1, trlweb);
	//
    //trgswfftExternalProduct1_test<Lvl1>(buf1, trlwea, cipher);
  }
  end = std::chrono::system_clock::now();
  costs[0] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  start = std::chrono::system_clock::now();
  //#pragma omp parallel for
  for (int i = 0; i < test_num; ++i) {
    //int stream_id = omp_get_thread_num();
    //cuARCEDB::HomAND(resd, cipher1, cipher2, ek, stream_id);
    //GateBootstrappingTLWE2TLWEFFT_test(resd, tlwe, *ek.bkfftlvl01, μ_polygen<Lvl1>(u));
    cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st<Lvl1>(resd, tlwe, u, 0);
    //cufftplvl1.PolyMul_st<Lvl1>(res2, trlwea, 0);
    //
    //trgswfftExternalProduct_test<Lvl1>(res2, trlwea, cipher);
    // cufftplvl1.ifft_st(fftb2, trlweb,0);
  }

  end = std::chrono::system_clock::now();
  costs[3] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

  std::cout<<"CPU: "<<costs[0]/test_num<<"us"<<std::endl;
  std::cout << "GPU Torus4: " << costs[3] / test_num << "us" << std::endl;
#endif
#if 0
  // 比较CPU和GPU结果
  std::ofstream file("/home/gaoshijie/cuHEDB/test/Arc_Query/fft_res.txt");
  file << std::fixed;

  file << std::setprecision(6);

  if (file.is_open()) {
    double toldf = 0;
    // for (int j = 0; j < 2049; j++) {
    //   double diff = double(resh[j]) - double(resd[j]);
    //   toldf += fabs(diff);
    //   file << "CPU = " << resh[j] << ", GPU = " << resd[j]
    //        << ", Diff = " << diff << std::endl;
    // }
    // for (int j = 0; j < 2048; j++) {
    //   double diff = fftb1[j] - fftb2[j];
    //   toldf += fabs(diff);
    //   file << "CPU = " << fftb1[j] << ", GPU = " << fftb2[j]
    //        << ", Diff = " << diff << std::endl;
    // }

	for(int i=0;i<2;++i)
    for (int j = 0; j < 2048; j++) {
      long double diff = (long double)(res3[i][j]) - (long double)(res4[i][j]);
      toldf += fabs(diff);
      file << "CPU = " << res3[i][j] << ", GPU = " << res4[i][j]<<", GPU2 = " <<res5[i][j]
           << ", Diff = " << diff << std::endl;
    }

	// for(int k=0;k<8;++k) {
  //   for (int i = 0; i < 2; ++i){
	// 	file<<"k = "<<k<<", i = "<< i<<std::endl;
  //     for (int j = 0; j < 2048; j++) {
  //       double diff = int64_t(buf1[k][i][j]) - int64_t(buf2[k][i][j]);
  //       toldf += fabs(diff);
  //       file << "CPU = " << buf1[k][i][j] << ", GPU = " << buf2[k][i][j]<< ", Diff = " << diff 
  //       << std::endl;
  //     }
	// }
	// }

   std::cout << "err: " << toldf << std::endl;
  }

  file.close();

  //delete bkfft;
#endif

  return 0;
}
