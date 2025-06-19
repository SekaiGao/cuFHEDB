#!/bin/bash

base_dir="../build/bin/"

programs=(
    "Arc_TPCH_Q1"
    "Arc_TPCH_Q6"
    "Arc_TPCH_Q12"
	"Arc_TPCH_Q14"

	"HE3_TPCH_Q1"
    "HE3_TPCH_Q6"
    "HE3_TPCH_Q12"
	"HE3_TPCH_Q14"
)

# Parameters:
rows=(16384) # dataset rows

if [ ! -d "../results" ]; then
    mkdir -p "../results"
    echo "Created directory: ../results"
fi

log_file="../results/tpch_baseline.log"
> "$log_file"

for program in "${programs[@]}"; do

    for row in "${rows[@]}"; do
        cmd="$base_dir$program  $row 16" # assumed joined table row is 16
        
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
