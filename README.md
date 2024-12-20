# Catching Common Vulnerabilities with Code Language Models

This repository contains the code for the paper "Catching Common Vulnerabilities with Code Language Models" by Anonymous Authors.

## Abstract

Code Language Model (code-LM)-based vulnerability detection for C/C++ faces a substantial challenge. Previous research has shown that even though code-LM-based is better than any prior machine learning approach, they still struggle to generalize well, as shown by the low F1 score. This paper investigates the problem from a different angle, focusing on how models perform when classifying whether code is vulnerable to a specific vulnerability type. We use the recently released PrimeVul dataset to investigate the ability to correctly classify different types of vulnerabilities. Our findings show that it is challenging to correctly identify a specific vulnerability class in a dataset containing all types, but if the task is modified to identify the most common vulnerabilities, the cumulative model outperforms previous binary classification results on the dataset. This result shows a promising path to make code-LMs practical in assisting developers with vulnerability detection tasks in C/C++ code.

## Repository Structure

- `training/`: Contains the code for fine-tuning the language models on the vulnerability datasets.
- `inference/`: Includes the code for performing inference using the fine-tuned models.

**Note:** Due to size limitations on GitHub, the pre-trained models and datasets used in this project are provided separately. Please refer to the instructions below for accessing and using these resources.

## Pre-trained models and datasets:
- Download the pre-trained language models from [here](https://drive.google.com/drive/folders/1wBVmRHghSiFXBKS2KU-CnA6wZezuFmMF?usp=sharing).
- Download the vulnerability datasets from [here](https://drive.google.com/drive/folders/1ki37wIXczktydHkW3Wmih852FAefatki?usp=sharing).
- Place the downloaded models and datasets in the appropriate directories.

## Fine-tune the language models:
- If you want to fine-tune the models on your own, navigate to the `training/` directory.
- Run the fine-tuning script: `bash run.sh`
- Adapt the script according to your requirements.

## Perform inference and evaluation:
- Navigate to the `inference/` directory.
- If you fine-tuned your own models, place them in the appropriate locations.
- Otherwise, use the pre-trained models provided by the authors.
- Run the inference script: `bash run.sh`
- The evaluation results will be generated automatically.
