#include "ARCEDB/comparison/batch_bootstrap.h"
#include "ARCEDB/comparison/comparable.h"
#include "ARCEDB/comparison/rgsw_ciphertext.h"
#include "ARCEDB/conversion/packlwes.h"
#include "ARCEDB/conversion/repack.h"
#include "ARCEDB/utils/serialize.h"
#include "cuHEDB/comparable_gpu.cuh"
#include "fastR.h"
#include <chrono>
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

void generateData(std::array<uint32_t, 1024> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 100000);//2149967295);

  for (int i = 0; i < 1024; i++) {
      uint32_t value = dis(gen);
      trlwe[i] = value;
  }
}

void generateData(std::array<double, 2048> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 1000000);

  for (int i = 0; i < 2048; i++) {
    uint32_t value = dis(gen);
    trlwe[i] = value;
  }
}

void generateData(std::array<std::array<uint32_t, 1024>, 2> &trlwe) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 4139967295);

  for(int k=0;k<2;++k)
  for (int i = 0; i < 1024; i++) {
    uint32_t value = dis(gen);
    trlwe[k][i] = value;
  }
}

void generateData(TRGSWLvl1 &trgswfft1) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1000, 1000);

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

template <class P>
void trgswfftExternalProduct_test(TFHEpp::TRLWE<P> &res,
                                  const TFHEpp::TRLWE<P> &trlwe,
                                  const TFHEpp::TRGSWFFT<P> &trgswfft) {
  TFHEpp::PolynomialInFD<P> decpolyfft{};
  __builtin_prefetch(trgswfft[0].data());
  TFHEpp::DecompositionPolynomialFFT<P>(decpolyfft, trlwe[0], 0);
  TFHEpp::TRLWEInFD<P> restrlwefft{};
  TFHEpp::MulInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[0][0]);
  TFHEpp::MulInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[0][1]);
  for (int i = 1; i < 3; i++) {
    __builtin_prefetch(trgswfft[i].data());
    TFHEpp::DecompositionPolynomialFFT<P>(decpolyfft, trlwe[0], i);
    TFHEpp::FMAInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[i][0]);
    TFHEpp::FMAInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[i][1]);
  }
  for (int i = 0; i < 3; i++) {
    __builtin_prefetch(trgswfft[i + 3].data());
    TFHEpp::DecompositionPolynomialFFT<P>(decpolyfft, trlwe[1], i);
    TFHEpp::FMAInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[i + 3][0]);
    TFHEpp::FMAInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[i + 3][1]);
  }
  TFHEpp::TwistFFT<P>(res[0], restrlwefft[0]);
  TFHEpp::TwistFFT<P>(res[1], restrlwefft[1]);
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
void trgswfftExternalProduct_test(std::array<std::array<double, 1024>, 2> &res,
                                  const TFHEpp::TRLWE<P> &trlwe,
                                  const TFHEpp::TRGSWFFT<P> &trgswfft) {
  TFHEpp::PolynomialInFD<P> decpolyfft{};
  __builtin_prefetch(trgswfft[0].data());
  DecompositionPolynomialFFT_test<P>(decpolyfft, trlwe[0], 0);
  TFHEpp::TRLWEInFD<P> restrlwefft{};
  TFHEpp::MulInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[0][0]);
  TFHEpp::MulInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[0][1]);
  for (int i = 1; i < 3; i++) {
    __builtin_prefetch(trgswfft[i].data());
    DecompositionPolynomialFFT_test<P>(decpolyfft, trlwe[0], i);
    TFHEpp::FMAInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[i][0]);
    TFHEpp::FMAInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[i][1]);
  }
  for (int i = 0; i < 3; i++) {
    __builtin_prefetch(trgswfft[i + 3].data());
    DecompositionPolynomialFFT_test<P>(decpolyfft, trlwe[1], i);
    TFHEpp::FMAInFD<P::n>(restrlwefft[0], decpolyfft, trgswfft[i + 3][0]);
    TFHEpp::FMAInFD<P::n>(restrlwefft[1], decpolyfft, trgswfft[i + 3][1]);
  }

  TFHEpp::TwistFFT<P>(res[0], restrlwefft[0]);
  TFHEpp::TwistFFT<P>(res[1], restrlwefft[1]);
}

#if 0
template <class P>
void CMUXFFTwithPolynomialMulByXaiMinusOne(TRLWE<P> &acc, const TRGSWFFT<P> &cs,
                                           const typename P::T a) {
  TRLWE<P> temp;
  for (int k = 0; k < P::k + 1; k++)
    PolynomialMulByXaiMinusOne<P>(temp[k], acc[k], a);
  trgswfftExternalProduct<P>(temp, temp, cs);
  for (int k = 0; k < P::k + 1; k++)
    for (int i = 0; i < P::n; i++)
      acc[k][i] += temp[k][i];
}
#endif

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
    std::array<uint32_t, 1025> &res, const std::array<uint32_t, 673> &tlwe,
    const TFHEpp::BootstrappingKeyFFT<Lvl01> &bkfft,
    const TFHEpp::Polynomial<Lvl1> &testvector) {
  TRLWELvl1 acc;
  TFHEpp::BlindRotate<Lvl01>(acc, tlwe, bkfft, testvector);
  TFHEpp::SampleExtractIndex<Lvl1>(res, acc, 0);
}


int main()
{
  uint32_t test_num = 1;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  

  TFHESecretKey sk;
  TFHEEvalKey ek;
  // ek.emplacebkfft<Lvl01>(sk);
  // ek.emplaceiksk<Lvl10>(sk);

  TFHEpp::BootstrappingKeyFFT<TFHEpp::lvl01param> *bkfft = new TFHEpp::BootstrappingKeyFFT<TFHEpp::lvl01param>;
  TFHEpp::KeySwitchingKey<TFHEpp::lvl10param> *isk = new TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>;  

  readFromFile(path, sk.key.lvl1, *bkfft, *isk);
  //generateData(*bkfft);
  ek.bkfftlvl01.reset(bkfft);
  ek.iksklvl10.reset(isk);
  



  TFHEpp::TLWE<Lvl0> tlwe;
  TFHEpp::TLWE<Lvl1> cipher1, cipher2;
  TFHEpp::TLWE<Lvl1> resh, resd;

  TFHEpp::TRLWE<Lvl1> res1, res2, trlwea;
  TRGSWLvl1 cipher, buf1, buf2;

  std::vector<TFHEpp::TRLWE<Lvl1>> trlwea2(2);

  std::array<uint32_t, 1024> trlweb;

  std::array<double, 2048> tlwefft, tlweres1, tlweres2;
  generateData(tlwefft);

  generateData(cipher);
  generateData(trlwea);
  generateData(trlweb);
  generateData(trlwea2[0]);
  generateData(trlwea2[1]);
  generateData(tlwe);
  generateData(cipher1);
  generateData(cipher2);

  cufftplvl1.LoadBK(*ek.bkfftlvl01);


  std::array<double, 1024> fftb1, fftb2;

  int rows = 10;

  // Filtering
  std::vector<uint64_t> ship_date(rows);
  std::vector<uint64_t> returnflag(rows);
  std::vector<ComparableLvl1> ship_data_ciphers(rows);
  std::vector<TRLWELvl1> returnflag_ciphers(rows);

  std::vector<TRGSWLvl1> shipdate_predicate(2);
  TRGSWLvl1 returnflag_predicate_Y, returnflag_predicate_N,
      linestatus_predicate_Y, linestatus_predicate_N;
  exponent_encrypt_rgsw<Lvl1>(10592, 20, shipdate_predicate, sk, true);
  exponent_encrypt_rgsw<Lvl1>(1, returnflag_predicate_Y, sk, true);
  exponent_encrypt_rgsw<Lvl1>(0, returnflag_predicate_N, sk, true);

  // Start sql evaluation
  //TLWELvl1 filter_res_YY, filter_res_YN, filter_res_NY, filter_res_NN;
  std::vector<TLWELvl1> resh1(rows), resd1(rows), resh2(rows), resd2(rows);
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_int_distribution<uint32_t> shipdate_message(10592-100, 10592+100);
  std::uniform_int_distribution<uint32_t> bianry_message(0, 1);

  // Generate data
  for(int i=0;i<rows;++i){
  ship_date[i] = shipdate_message(engine);
  returnflag[i] = bianry_message(engine);

  // Encrypt
  exponent_encrypt<Lvl1>(ship_date[i], 20, ship_data_ciphers[i], sk);
  exponent_encrypt<Lvl1>(returnflag[i], returnflag_ciphers[i], sk);

  std::cout << "i = " << i << ", " << ship_date[i] << ", " << returnflag[i]<<std::endl;
  }

  //#pragma omp parallel for 

        //TLWELvl1 pre_res_YY, pre_res_YN, pre_res_NY, pre_res_NN;

        // returnflag = Y, linestatus = Y
  std::cout<<"CPU:"<<std::endl;
  for(int i=0;i<rows;++i){
  arcedb::less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, resh1[i], ek, sk);
  arcedb::equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_Y, resh2[i], sk);


  int result[4];
  result[0] = TFHEpp::tlweSymDecrypt<Lvl1>(resh1[i], sk.key.get<Lvl1>());
  result[1] = TFHEpp::tlweSymDecrypt<Lvl1>(resh2[i], sk.key.get<Lvl1>());

  arcedb::lift_and_and(resh1[i], resh2[i], resh, 29, ek, sk);

  result[2] = tlweSymInt32Decrypt<Lvl1>(resh, std::pow(2.,29), sk.key.get<Lvl1>());

  TFHEpp::HomAND(resh, resh1[i], resh2[i], ek);

  result[3] = TFHEpp::tlweSymDecrypt<Lvl1>(resh, sk.key.get<Lvl1>());

  std::cout<<"i = "<< i;
  for(int j=0;j<4;++j){
    std::cout<<", "<<result[j];
  }
  std::cout<<std::endl;
  }

  std::cout<<"GPU:"<<std::endl;
  for(int i=0;i<rows;++i){
  arcedb::less_than_tfhepp(ship_data_ciphers[i], shipdate_predicate, 2, resd1[i], ek, sk);
  arcedb::equality_tfhepp(returnflag_ciphers[i], returnflag_predicate_Y, resd2[i], sk);


  int result[4];
  result[0] = TFHEpp::tlweSymDecrypt<Lvl1>(resd1[i], sk.key.get<Lvl1>());
  result[1] = TFHEpp::tlweSymDecrypt<Lvl1>(resd2[i], sk.key.get<Lvl1>());

  arcedb::lift_and_and(resd1[i], resd2[i], resd, 29, ek, sk);

  result[2] = tlweSymInt32Decrypt<Lvl1>(resd, std::pow(2.,29), sk.key.get<Lvl1>());

  TFHEpp::HomAND(resd, resd1[i], resd2[i], ek);

  result[3] = TFHEpp::tlweSymDecrypt<Lvl1>(resd, sk.key.get<Lvl1>());

  std::cout<<"i = "<< i;
  for(int j=0;j<4;++j){
    std::cout<<", "<<result[j];
  }
  std::cout<<std::endl;
  }

  //cuARCEDB::less_than_tfhepp(ship_data_ciphers, shipdate_predicate, 2, resd, ek, sk, 0);
  //arcedb::greater_than_tfhepp(ship_data_ciphers, shipdate_predicate, 2, resh, ek, sk);
  //cuARCEDB::greater_than_tfhepp(ship_data_ciphers, shipdate_predicate, 2, resd, ek, sk, 0);

  //TFHEpp::HomAND(resh, resh1, resh2, ek);
  //cuARCEDB::HomAND(resd, resd1, resd2, ek, 0);
  //arcedb::lift_and_and(resh1, resh2, resh, 45, ek, sk);
  //cuARCEDB::lift_and_and(resh1, resh2, resd, 45, ek, sk, 0);
  uint32_t u = -Lvl1::μ;
  //printf("u=%d\n", u);
  //TFHEpp::IdentityKeySwitch<Lvl10>(tlwe, cipher1, *ek.iksklvl10);

  //TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl01>(resh, tlwelvl0, *ek.bkfftlvl01, TFHEpp::μpolygen<Lvl1, Lvl1::μ>());
  //GateBootstrappingTLWE2TLWEFFT_test(resd, tlwe, *ek.bkfftlvl01, μ_polygen<Lvl1>(u));
  //BlindRotate_test<Lvl01>(res1, tlwe, *ek.bkfftlvl01, μ_polygen<Lvl1>(u));
  //cufftplvl1.GateBootstrappingTLWE2TLWEFFT_st(resd, tlwelvl0, -u, 0);

  std::array<std::array<uint32_t, 1024>, 2> res3, res4;

  //trgswfftExternalProduct_test<Lvl1>(res3, returnflag_ciphers, returnflag_predicate_Y);
  //cufftplvl1.ExternalProduct_st(res4, returnflag_ciphers, returnflag_predicate_Y, 0);
  // TFHEpp::PolyMul<Lvl1>(res1[0], ship_data_ciphers[0][0], trlweb);
  // TFHEpp::PolyMul<Lvl1>(res1[1], ship_data_ciphers[0][1], trlweb);

  // cufftplvl1.PolyMul_st<Lvl1>(res2, ship_data_ciphers[0], 0);

  // TFHEpp::TwistIFFT<Lvl1>(fftb1, trlweb);
  // cufftplvl1.ifft_st(fftb2, trlweb, 0);


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
  // 比较CPU和GPU结果
  std::ofstream file("/home/gaoshijie/cuHEDB/test/Arc_Query/fft_res.txt");
  file << std::fixed;

  file << std::setprecision(6);

  if (file.is_open()) {
    double toldf = 0;
    #if 1
    for (int j = 0; j < 1025; j++) {
      double diff = int64_t(resh[j]) - int64_t(resd[j]);
      toldf += fabs(diff);
      file << "CPU = " << resh[j] << ", GPU = " << resd[j]
           << ", Diff = " << diff << std::endl;
    }
    #endif
    // for (int j = 0; j < 1024; j++) {
    //   double diff = int64_t(fftb1[j]) - int64_t(fftb2[j]);
    //   toldf += fabs(diff);
    //   file << "CPU = " << fftb1[j] << ", GPU = " << fftb2[j]
    //        << ", Diff = " << diff << std::endl;
    // }
    #if 0
	for(int i=0;i<2;++i)
    for (int j = 0; j < 1024; j++) {
      double diff = double(res3[i][j]) - double(res4[i][j]);
      toldf += fabs(diff);
      file << "CPU = " << res3[i][j] << ", GPU = " << res4[i][j]
           << ", Diff = " << diff << std::endl;
    }
    #endif
	// for(int k=0;k<6;++k) {
    // for (int i = 0; i < 2; ++i){
	// 	file<<"k = "<<k<<", i = "<< i<<std::endl;
    //   for (int j = 0; j < 1024; j++) {
    //     double diff = int64_t(buf1[k][i][j]) - int64_t(buf2[k][i][j]);
    //     toldf += fabs(diff);
    //     file << "CPU = " << buf1[k][i][j] << ", GPU = " << buf2[k][i][j]
    //          << ", Diff = " << diff << std::endl;
    //   }
	// }
	// }
    std::cout << "err: " << toldf << std::endl;
    file.close();
  }

  file.close();

  return 0;
}
