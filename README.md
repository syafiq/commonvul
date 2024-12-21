# Catching Common Vulnerabilities with Code Language Models

This repository contains the code for the paper "Catching Common Vulnerabilities with Code Language Models" by Anonymous Authors.

## Abstract

Code Language Model (code-LM)-based vulnerability detection for C/C++ faces a substantial challenge. Previous research has shown that even though code-LM-based is better than any prior machine learning approach, they still struggle to generalize well, as shown by the low F1 score. This paper investigates the problem from a different angle, focusing on how models perform when classifying whether code is vulnerable to a specific vulnerability type. We use the recently released PrimeVul dataset to investigate the ability to correctly classify different types of vulnerabilities. Our findings show that it is challenging to correctly identify a specific vulnerability class in a dataset containing all types, but if the task is modified to identify the most common vulnerabilities, the cumulative model outperforms previous binary classification results on the dataset. This result shows a promising path to make code-LMs practical in assisting developers with vulnerability detection tasks in C/C++ code.

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

## Fine-tune the language models:
To fine-tune the model, navigate to the `training/` directory. Run the fine-tuning script: `bash run.sh`, adapt the script according to your requirements.

## Perform inference and evaluation:
Once the model has been fine-tuned (or using [our](https://drive.google.com/drive/folders/1wBVmRHghSiFXBKS2KU-CnA6wZezuFmMF?usp=sharing) fine-tuned model), once can run the inference script (under `inference/` directory) to perform both the inference and evaluation: `bash run.sh` The evaluation results will be generated automatically for each model.

## Output Analysis
The pt files can be analyzed using [this](https://github.com/syafiq/commonvul/blob/main/inference/calculate.ipynb) jupyter notebook, i.e., the one that is used in the paper.
