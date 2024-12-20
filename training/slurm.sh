#!/bin/bash

while getopts m:c:d:z:g:t:e:x:s: flag
do
    case "${flag}" in
        m) MODEL=${OPTARG};;            # Model checkpoint
        c) NTC=${OPTARG};;              # CWE code or TOP x
        d) DATADIR=${OPTARG};;          # Dataset directory
        z) CHARACTER=${OPTARG};;        # Character ("b" or "m")
        g) GRADIENT=${OPTARG};;         # Gradient accumulation steps
	t) TRAIN_BATCH=${OPTARG};;	# Train batch
	e) EVAL_BATCH=${OPTARG};;	# Eval batch
	x) TC=${OPTARG};;		# string either "cwe" or "top" or "all"
	s) SLURM=${OPTARG};;		# the slurm server
    esac
done

if [[ "$SLURM" = "alvis" ]]; then
	if [ -n "$SLURM_GPUS_ON_NODE" ]; then
		NUM_GPUS=$SLURM_GPUS_ON_NODE
	elif [ -n "$SLURM_GPUS_PER_NODE" ]; then
		NUM_GPUS=$SLURM_GPUS_PER_NODE
	elif [ -n "$SLURM_JOB_GPUS" ]; then
		NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
	elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
		NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
	else
		echo "Unable to determine the number of GPUs allocated."
		exit 1
	fi
	
	echo "Number of GPUs allocated: $NUM_GPUS"
	
	apptainer exec --nv \
	conda_env.sif accelerate launch --config_file config_cos.yaml finetune_no_lora.py \
	"$MODEL" "$NTC" "$DATADIR" "$CHARACTER" "$GRADIENT" "$TRAIN_BATCH" "$EVAL_BATCH" "$TC"

elif [[ "$SLURM" = "cosmos" ]]; then
	module load Anaconda3/2023.09-0
	conda activate vc

	accelerate launch --config_file config_cos.yaml finetune_no_lora.py \
	"$MODEL" "$NTC" "$DATADIR" "$CHARACTER" "$GRADIENT" "$TRAIN_BATCH" "$EVAL_BATCH" "$TC"
else
	echo "not valid slurm server"
fi

