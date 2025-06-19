#!/bin/bash

base_dir="../build/bin/"

programs=(
	"Arc_HC_Q1"
	"Arc_HC_Q2"
	"Arc_HC_Q3"
	"Arc_HC_Q4"
)

if [ ! -d "../results" ]; then
    mkdir -p "../results"
    echo "Created directory: ../results"
fi

log_file="../results/healthcare_baseline.log"
> "$log_file"

for program in "${programs[@]}"; do

	cmd="$base_dir$program"
	
	echo "Running $cmd, output will be logged to $log_file"
	
	nohup $cmd >> "$log_file" 2>&1 &
	
	pid=$!
	
	wait $pid

	echo "" >> "$log_file"
	echo "" >> "$log_file"
	echo "" >> "$log_file"
	echo "" >> "$log_file"
done

echo "All programs have finished executing."
