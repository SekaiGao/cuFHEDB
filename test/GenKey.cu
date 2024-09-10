#include "HEDB/comparison/comparison.h"
#include "HEDB/conversion/repack.h"
#include "HEDB/utils/types.h"
#include "HEDB/utils/utils.h"
#include "cuHEDB/HomCompare_gpu.cuh"
#include "Query/fastR.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <unordered_set>
#include <vector>

using namespace HEDB;
using namespace TFHEpp;

int main() {
	TFHESecretKey sk;
  	TFHEEvalKey ek;
  	ek.emplacebkfft<Lvl01>(sk);
  	ek.emplaceiksk<Lvl10>(sk);

	writeToFile(path, sk.key.lvl1, *ek.bkfftlvl01, *ek.iksklvl10);
	return 0;
}