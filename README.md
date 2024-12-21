# Catching Common Vulnerabilities with Code Language Models

This repository contains the code for the paper "Catching Common Vulnerabilities with Code Language Models" by Anonymous Authors.

## Research Questions

This paper addresses two main research questions:

1. RQ1: How do different code-LMs perform for each of the most common vulnerability types?
2. RQ2: How does code-LM performance change when incrementally including more of the most common vulnerability types in the dataset?

## Repository Structure

- `training/`: Contains the code for fine-tuning the language models on the vulnerability datasets.
- `inference/`: Includes the code for performing inference using the fine-tuned models.

**Note:** Due to size limitations on GitHub, the pre-trained models and datasets used in this project are provided separately. Please refer to the instructions below for accessing and using these resources.

## Pre-trained models and datasets:
- Fine-tuned code LMs are available from [here](https://drive.google.com/drive/folders/1wBVmRHghSiFXBKS2KU-CnA6wZezuFmMF?usp=sharing).
- datasets are available from [here](https://drive.google.com/drive/folders/1ki37wIXczktydHkW3Wmih852FAefatki?usp=sharing).

The dataset formatting for both RQs: 
- $`D_{i,train}`$: pv_train_{i}.json
- $`D_{i,test}`$: pv_test_{i}.json
- $`D_{test,i}^{comp}`$: pv_test_all_{i}.json
- $`D_{train,n}^{cum}`$: pv_train_top_{n}.json
- $`D_{test,n}^{cum}`$: pv_test_top_{n}.json
- $`D_{test,all,n}^{cum}`$: pv_TOP_test_all_{n}.json

## Fine-tune the language models:
To fine-tune the model, navigate to the `training/` directory. Run the fine-tuning script: `bash run.sh`, adapt the script according to your requirements.

## Perform inference and evaluation:
Once the model has been fine-tuned (or using [our](https://drive.google.com/drive/folders/1wBVmRHghSiFXBKS2KU-CnA6wZezuFmMF?usp=sharing) fine-tuned model), once can run the inference script (under `inference/` directory) to perform both the inference and evaluation: `bash run.sh` (one might need to adapt the script accordingly for the correct location of the dataset and model). The evaluation results will be generated automatically for each model.

## Output Analysis
The pt files can be analyzed using [this](https://github.com/syafiq/commonvul/blob/main/inference/calculate.ipynb) jupyter notebook, i.e., the one that is used in the paper.
