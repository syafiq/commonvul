#!/bin/bash

MODEL_DIR="../../selective_model_RQ1_PV"
EPOCH=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
CWES=("119" "125" "787" "20" "476" "200" "416" "703" "190" "399")

codet5="Salesforce/codet5-small"
unixcoder="microsoft/unixcoder-base"
codebert="microsoft/codebert-base"
starcoder="bigcode/starcoder2-7b"
codegen="Salesforce/codegen25-7b-multi_P"

MODELS=($unixcoder)

# Loop through each CWE, each model, and each epoch to find the matching model checkpoints in MODEL_DIR
for CWE in "${CWES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for EPOCH_NUM in "${EPOCH[@]}"; do
      # Construct checkpoint path
      CHECKPOINT_PATH="${MODEL_DIR}/${MODEL}_${CWE}_${EPOCH_NUM}"
      echo "$CHECKPOINT_PATH"
	  # Only proceed if the checkpoint directory exists
      if [ -d "$CHECKPOINT_PATH" ]; then
        CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
        OUTPUT="output/pv_test_${CHECKPOINT_NAME}.out"
        
        echo "Processing checkpoint: $CHECKPOINT_NAME"
        echo "Output file: $OUTPUT"

        # Uncomment the line below to submit the test job using cos.sh for each model checkpoint
        sbatch --gres=gpu:1 -t 1-00:00:00 -A xxxxxx-x-xx -p gpua40i -o $OUTPUT cos.sh -m "$CHECKPOINT_PATH" -c "$CWE"

        echo "Submitted test job for model checkpoint $CHECKPOINT_NAME with CWE $CWE and Epoch $EPOCH_NUM"
      fi
    done
  done
done
