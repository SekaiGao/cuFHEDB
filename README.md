# cuFHEDB
CUDA-accelerated Fully Homomorphic Encryption Database based on HE<sup>3</sup>DB

## Requirements
To build and run cuFHEDB, ensure the following dependencies are installed:

```
git 
gcc >= 10
cmake >= 3.16
GMP 6.2.0
```


## Building cuFHEDB
Follow these steps to build cuFHEDB:

```
mkdir build && cd build
cmake ..
make -j
```

After the build is complete, you can execute TPC-H queries located in the build/bin/ directory.


## Examples

### Query Evaluation

- Source code: test/TPCH_Q1.cu
- Output binary: build/bin/TPCH_Q1
- Description: This example demonstrates the evaluation of the TPC-H Q1 query on an encrypted database with 2<sup>10</sup> rows.
