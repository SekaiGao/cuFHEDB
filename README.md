# cuFHEDB: GPU-Accelerated Fully Homomorphic Encryption Database

## Introduction
This is the open-source implementation of **cuFHEDB**: the first GPU-accelerated Fully Homomorphic Encryption (FHE) database specifically designed to address the computational challenges of executing encrypted SQL queries. By harnessing GPU parallelism and advanced optimization techniques, **cuFHEDB** delivers substantial performance improvements in SQL query evaluation, making FHE-based databases a more viable solution for real-world privacy-preserving applications.

## Code Structure Overview
We provide a detailed overview of the code structure, which is organized to ensure clarity in functionality, modularity for easier development and maintenance, and streamlined access to datasets, scripts, source code, and benchmark tests.
```
cuFHEDB/
│
├── data/                       # Contains dataset for healthcare queries
│
├── scripts/                    # Scripts for evaluating benchmarks
│
├── src/                        # Source code
│   ├── Database/               # Modules for GPU-accelerated FHE-based database operators 
│   │   ├── cuFHEDB/            # Homomorphic comparison and SQL operations
│   │   └── cuHEDB/             # Other homomorphic comparison algorithm
│   │
│   └── FHE/                    # Modules for GPU-accelerated FHE operations
│       ├── fft/                # Custom-Optimized Folding FFT 
│       ├── externalproduct/    # Inter-Block External Product
│       ├── blindrotate/        # Inter-Block Blind Rotate
│       └── bootstrapping/      # GPU-Aware Key Bundle Bootstrapping
│
├── test/                       # Test files for various benchmarks
│   ├── Database/				        # Database benchmark tests
│	  │  	├── Healthcare/         # Healthcare benchmark tests
│   │	  └── TPC-H/              # TPC-H benchmark tests
│   └── FHE/                    # FHE benchmark tests
│
└── thirdparty/                 # Third-party dependencies               
```

## Dataset
The [Healthcare dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) is utilized in this project, simulating a realistic scale for privacy-preserving queries in hospital scenarios. It has been downloaded and placed in the `data/` directory for running the healthcare benchmark. Ensure the dataset is correctly placed in this directory before execution.



## Requirements

To build and run **cuFHEDB**, ensure the following dependencies are installed:

```
git 
gcc >= 11
cmake >= 3.18
GMP 6.2.0
CUDA >= 12.1
OpenMP
```

## Building cuFHEDB

You can follow these steps to build **cuFHEDB**:

```
mkdir build && cd build
cmake ..
make -j
```

After building, the compiled binaries will be available in the `build/bin/` directory for running examples and benchmarks.

#### 1. FFT Benchmark
- **Source**: `test/FHE/fft.cu`
- **Binary**: `build/bin/fft_test`
- **Description**: Evaluates and compares the efficiency of the custom-optimized folding FFT on GPU, a fundamental operation in FHE computations.
- **Execution Script**:
  ```bash
  bash scripts/fft_benchmark.sh
  ```
- **Expected Output**: Results will be saved in `results/fft_benchmark.log`.

#### 2. External Product Benchmark
- **Source**: `test/FHE/externalproduct.cu`
- **Binary**: `build/bin/externalproduct_test`
- **Description**: Benchmarks the efficiency of the External Product operation, which is crucial for homomorphic ciphertext multiplication.
- **Execution Script**:
  ```bash
  bash scripts/externalproduct_benchmark.sh
  ```
- **Expected Output**: Results will be saved in `results/externalproduct_benchmark.log`.

#### 3. Bootstrapping Benchmark
- **Source**: `test/FHE/bootstrapping.cu`
- **Binary**: `build/bin/bootstrap_test`
- **Description**: Measures GPU-accelerated bootstrapping efficiency, essential for logic gates and homomorphic comparisons.
- **Execution Script**:
  ```bash
  bash scripts/bootstrapping_benchmark.sh
  ```
- **Expected Output**: Results will be saved in `results/bootstrapping_benchmark.log`.



### Database Benchmarks

#### 1. Homomorphic Comparison Benchmark
- **Source**: `test/Database/comparison.cu`
- **Binary**: `build/bin/conmarison_test`
- **Description**: Demonstrates homomorphic comparison operations in cuFHEDB and compares its efficiency with [HE<sup>3</sup>DB](https://github.com/zhouzhangwalker/HE3DB) and [ArcEDB](https://github.com/zhouzhangwalker/ArcEDB).

- **Execution Script**:
  ```bash
  bash scripts/comparison_benchmark.sh
  ```
- **Expected Output**: Benchmark results will be saved in `results/comparison_benchmark.log`.

#### 2. Healthcare Benchmarks
- **Source**: `test/Database/Healthcare/HC_Q1.cu` (and related queries Q2-Q4)
- **Binary**: `build/bin/HC_Q1` (similarly, Q2-Q4)
- **Description**: Evaluates healthcare-related queries (Q1-Q4) using a 56K-row encrypted database, demonstrating practical performance in privacy-preserving scenarios.
- **Execution Scripts**:
  - **cuFHEDB Queries**:
    ```bash
    bash scripts/healthcare_query.sh
    ```
  - **Baseline (ArcEDB)**:
    ```bash
    bash scripts/healthcare_baseline.sh
    ```
- **Expected Output**: The benchmark results of the four queries will be saved in the `results/` directory.

#### 3. TPC-H Benchmarks
- **Source**: `test/Database/TPC-H/TPCH_Q1.cu` (and related queries Q6, Q12, Q14)
- **Binary**: `build/bin/TPCH_Q1` (similarly, Q6, Q12, Q14)
- **Description**: Evaluates the performance of standard TPC-H analytical queries (Q1, Q6, Q12, Q14) using encrypted databases, showcasing cuFHEDB’s capabilities in analytical workloads.
- **Execution Scripts**:
  - **cuFHEDB Queries**:
    ```bash
    bash scripts/tpch_query.sh
    ```
  - **Baseline (ArcEDB and HE<sup>3</sup>DB)**:
    ```bash
    bash scripts/tpch_baseline.sh
    ```
- **Expected Output**: After execution, the benchmark results of the four queries will be saved in the `results/` directory.

## License

This project is licensed under the MIT License.

