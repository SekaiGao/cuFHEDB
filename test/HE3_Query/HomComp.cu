#include "HEDB/comparison/comparison.h"
#include "HEDB/conversion/repack.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "cuHEDB/HomCompare_gpu.cuh"
#include "fastR.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

using namespace HEDB;
using namespace TFHEpp;

int main() {

  const int test_num = 100;
  uint32_t bits = 30;

  warmupGPU();

  // Lvl2
  std::cout << "Encrypting" << std::endl;

  double costs;
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  
  TFHESecretKey sk;
  TFHEEvalKey ek;
  ek.emplacebkfft<Lvl01>(sk);
  ek.emplaceiksk<Lvl10>(sk);
  ek.emplacebkfft<Lvl02>(sk);
  ek.emplaceiksk<Lvl20>(sk);
  ek.emplaceiksk<Lvl21>(sk);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "generate Secret Key Time: " << costs/1000 << "ms" <<std::endl;

  std::cout << "Loading" << std::endl;
  start = std::chrono::system_clock::now();

  //load BK to device
  cufftlvl1.LoadBK(*ek.bkfftlvl01);
  cufftlvl2.LoadBK(*ek.bkfftlvl02);

  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<"Load Success."<<std::endl;
  std::cout << "Loading Time: " << costs/1000 << "ms" << std::endl;

  start = std::chrono::system_clock::now();

  Lvl2::T pt1, pt2;
  uint32_t resh, resd;
  TLWE<Lvl2> ct1, ct2;
  TLWE<TFHEpp::lvl1param> cresh, cresd;

  std::uniform_int_distribution<Lvl2::T> data_distribution(0, (1 << bits) - 1);

  pt1 = data_distribution(generator);
  pt2 = data_distribution(generator);

  std::cout<<pt1<<", "<<pt2<<std::endl;

  uint32_t scale_bits = std::numeric_limits<Lvl2::T>::digits - bits - 1;

  std::cout << "Encrypt..." <<std::endl;

  ct1 = TFHEpp::tlweSymInt32Encrypt<Lvl2>(pt1, Lvl2::α, pow(2., scale_bits), sk.key.get<Lvl2>());
  ct2 = TFHEpp::tlweSymInt32Encrypt<Lvl2>(pt2, Lvl2::α, pow(2., scale_bits), sk.key.get<Lvl2>());

  std::cout << "HomCompare on gpu..." << std::endl;
  start = std::chrono::system_clock::now();
  for(int i=0;i<test_num;++i){
	cuHEDB::greater_than_equal<Lvl2>(ct1, ct2, cresd, bits, ek, LOGIC, 0);
  }
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout<<"homomophic greater than(gpu): "<< costs/(1000*test_num) <<"ms"<< std::endl;

  std::cout << "HomCompare on cpu..." << std::endl;
  start = std::chrono::system_clock::now();
  for(int i=0;i<test_num;++i){
	greater_than_equal<Lvl2>(ct1, ct2, cresh, bits, ek, LOGIC);
  }
  end = std::chrono::system_clock::now();
  costs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout<<"homomophic greater than(cpu): "<< costs/(1000*test_num) <<"ms"<< std::endl;

  resd = TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(cresd, sk.key.get<TFHEpp::lvl1param>());
  resh = TFHEpp::tlweSymDecrypt<TFHEpp::lvl1param>(cresh, sk.key.get<TFHEpp::lvl1param>());

  std::cout<<"result: "<<"plain: "<<(pt1 >= pt2)<<", cipher(cpu): "<<resh<<", cipher(gpu): "<<resd<<std::endl;

  return 0;
}