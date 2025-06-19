#!/bin/bash

base_dir="../build/bin/"

programs=(
    "TPCH_Q1"
    "TPCH_Q6"
    "TPCH_Q12"
	"TPCH_Q14"
)

# Parameters:
rows=(16384) # The default dataset size is set to 16 rows
# rows=(16384 65536 262144 1048576) # 16K, 64K, 256K, 1M (dataset rows for evaluation)

if [ ! -d "../results" ]; then
    mkdir -p "../results"
    echo "Created directory: ../results"
fi

log_file="../results/tpch_query.log"
> "$log_file"

for program in "${programs[@]}"; do

	
    for row in "${rows[@]}"; do
        cmd="$base_dir$program $row 16" # assued joined table row is 16
        
        echo "Running $cmd, output will be logged to $log_file"
        
        nohup $cmd >> "$log_file" 2>&1 &
        
        pid=$!
        
        wait $pid

		echo "" >> "$log_file"
		echo "" >> "$log_file"
		echo "" >> "$log_file"
		echo "" >> "$log_file"
    done
done

echo "All programs have finished executing."
