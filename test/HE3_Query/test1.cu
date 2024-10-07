#include "HEDB/comparison/comparison.h"
#include "HEDB/comparison/tfhepp_utils.h"
#include "HEDB/conversion/repack.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "cuHEDB/HomCompare_gpu.cuh"
#include "fastR.h"
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <random>
#include <unistd.h>

using namespace HEDB;
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
  std::uniform_int_distribution<uint64_t> dis(0, 100000);

  for (int i = 0; i < 2049; i++) {
    uint32_t value = dis(gen);
    tlwe[i] = value;
  }
}
// template <typename lvl1param::T μ = lvl1param::μ>
// void GateBootstrapping(TLWE<lvl0param> &res, const TLWE<lvl0param> &tlwe,
//                        const EvalKey &ek) {
//   TLWE<lvl1param> tlwelvl1;
//   GateBootstrappingTLWE2TLWEFFT<lvl01param>(tlwelvl1, tlwe, *ek.bkfftlvl01,
//                                             μpolygen<lvl1param, μ>());
//   IdentityKeySwitch<lvl10param>(res, tlwelvl1, *ek.iksklvl10);
// }

// template <class P, int casign, int cbsign, typename P::T offset>
// inline void HomGate(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
//                     const EvalKey &ek) {
//   for (int i = 0; i <= P::k * P::n; i++)
//     res[i] = casign * ca[i] + cbsign * cb[i];
//   res[P::k * P::n] += offset;
//   GateBootstrapping(res, res, ek);
// }

// template <class P>
// void HomAND(TLWE<P> &res, const TLWE<P> &ca, const TLWE<P> &cb,
//             const EvalKey &ek) {
//   HomGate<P, 1, 1, -lvl1param::μ>(res, ca, cb, ek);
// }

int main()
{
  uint32_t test_num = 1;
  omp_set_num_threads(num_stream1);
  warmupGPU();

  // TFHEpp::BootstrappingKeyFFT<TFHEpp::lvl01param> *bkfft = new TFHEpp::BootstrappingKeyFFT<TFHEpp::lvl01param>;
  // TFHEpp::KeySwitchingKey<TFHEpp::lvl10param> *isk = new TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>;  

  TFHESecretKey sk;
  TFHEEvalKey ek;
  //ek.emplacebkfft<Lvl01>(sk);
 // ek.emplaceiksk<Lvl10>(sk);
  ek.emplacebkfft<Lvl02>(sk);
  ek.emplaceiksk<Lvl20>(sk);

  // readFromFile(path, sk.key.lvl1, *bkfft, *isk);
  // ek.bkfftlvl01.reset(bkfft);
  // ek.iksklvl10.reset(isk);

  TFHEpp::TLWE<Lvl0> tlwe;
  TFHEpp::TLWE<Lvl2> resh, resd, cipher1, cipher2;


  generateData(tlwe);
  generateData(cipher1);
  //generateData(cipher2);

  //cufftlvl1.LoadBK(*ek.bkfftlvl01);
  cufftlvl2.LoadBK(*ek.bkfftlvl02);

  //   cudaError_t err = cudaGetLastError();
  //   if (err != cudaSuccess) {
  //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
  //   }

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();
  double costs[4]={0,0,0,0};

  uint32_t u = uint32_t(1ULL << 15);

#if 1
  // CPU
  start = std::chrono::system_clock::now();
  //#pragma omp parallel for 
  for (int i = 0; i < test_num; ++i) {
    //TFHEpp::HomAND(resh, cipher1, cipher2, ek);
    //TFHEpp::GateBootstrappingTLWE2TLWEFFT<Lvl01>(resh, tlwe, *ek.bkfftlvl01, TFHEpp::μ_polygen<Lvl1>(u));
    TFHEpp::MSBGateBootstrapping(resh, cipher1, ek, LOGIC);
    //HEDB::HomAND(resh, cipher1, cipher2, ek, LOGIC);
  }
  end = std::chrono::system_clock::now();
  costs[0] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif
  start = std::chrono::system_clock::now();
  //#pragma omp parallel for
  for (int i = 0; i < test_num; ++i) {
    int stream_id = omp_get_thread_num();
    //cuARCEDB::HomAND(resd, cipher1, cipher2, ek, stream_id);
	//cufftplvl.GateBootstrappingTLWE2TLWEFFT_st<Lvl1>(resd, tlwe, -u, 0);
    cuHEDB::MSBGateBootstrapping(resd, cipher1, ek, LOGIC, stream_id);
    //cuHEDB::HomAND(resd, cipher1, cipher2, ek, LOGIC, stream_id);
  }

  end = std::chrono::system_clock::now();
  costs[3] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout<<"CPU: "<<costs[0]/test_num<<"us"<<std::endl;
  std::cout << "GPU Torus4: " << costs[3] / test_num << "us" << std::endl;

  // 比较CPU和GPU结果
  std::ofstream file("/home/gaoshijie/cuHEDB/test/HE3_Query/fft_res.txt");
  file << std::fixed;

  file << std::setprecision(6);

  if (file.is_open()) {
    double toldf = 0;
    for (int j = 0; j < 2049; j++) {
      double diff = int64_t(resh[j]) - int64_t(resd[j]);
      toldf += fabs(diff);
      file << "CPU = " << resh[j] << ", GPU = " << resd[j]
           << ", Diff = " << diff << std::endl;
    }
    std::cout << "err: " << toldf << std::endl;
    file.close();
  }

  file.close();


  return 0;
}


