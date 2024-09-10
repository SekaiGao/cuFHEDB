# cuFHEDB: 
CUDA-accelerated Fully Homomorphic Encryption DataBase
## Requirements

```
git 
gcc >= 10
cmake >= 3.16
GMP 6.2.0
```

## Building cuFHEDB
You can build the cuFHEDB by executing the following commands:
```
mkdir build
cd build
cmake .. 
make -j
```
Then you can run the TPC-H query in 
the `build/bin/` directory.


## Examples

### Query Evaluation
- codes `test/TPCH_Q1.cu`
- output binary `build/bin/TPCH_Q1`
- This demo shows the evaluation of TPC-H Q1 over a 2^{10} rows of encrypted database.
