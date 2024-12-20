#!/bin/bash

# Parse arguments from flags
while getopts m:c: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;              # Model checkpoint path
        c) CWE=${OPTARG};;                # CWE code
    esac
done

echo "Running test with Model Checkpoint: $MODEL"
echo "CWE: $CWE"

# Load the new module and activate the environment
module load Anaconda3/2023.09-0
conda activate vc

# Run the test.py script with the specified model checkpoint and CWE
python test.py "$CWE" "$MODEL"
