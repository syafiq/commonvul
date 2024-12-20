import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import sys
import os

# Import your custom DefectModel
from model_t5 import DefectModel

if len(sys.argv) != 3:
    print("Usage: python test_t5.py <CWE> <model_checkpoint>")
    sys.exit(1)

# Parse arguments
cwe = str(sys.argv[1])
model_checkpoint = str(sys.argv[2])
model_name = os.path.basename(model_checkpoint)
print(f"Running test with Model Checkpoint: {model_checkpoint}")
print(f"CWE: {cwe}")

# Load test data
test_df = pd.read_json(f"../dataset/test_{cwe}.json")
test_df = test_df.drop(['year', 'cwe', 'source', 'hash'], axis=1)
test_df = test_df.reset_index(drop=True)

# Convert test data to Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

# Load model configuration and tokenizer
config = AutoConfig.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Adjust max length according to model limits
model_max_length = getattr(config, 'n_positions', None)
if model_max_length is None:
    model_max_length = getattr(config, 'max_position_embeddings', 512)
max_seq_length = min(512, model_max_length)
print(f"Using max_seq_length: {max_seq_length}")

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.pad_token = '<pad>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

# Resize token embeddings if new tokens were added
if len(tokenizer) > config.vocab_size:
    config.vocab_size = len(tokenizer)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

# Tokenize test dataset
tokenized_datasets = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Prepare DataLoader
test_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)

# Load the fine-tuned T5 model with DefectModel
if not os.path.isdir(model_checkpoint):
    raise FileNotFoundError(f"Model checkpoint directory '{model_checkpoint}' not found.")

# Load the base T5 model
base_model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, config=config)

# Resize token embeddings on the base model before wrapping
base_model.resize_token_embeddings(len(tokenizer))

# Wrap the base model with your custom DefectModel
model = DefectModel(base_model, config, tokenizer, args=None)  # Adjust args if necessary

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Define label mapping (adjust based on your training labels)
label_map = {0: "non-vulnerable", 1: "vulnerable"}
inverse_label_map = {v: k for k, v in label_map.items()}

# Evaluate model
all_predictions = []
all_labels = []
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]

        # Forward pass through DefectModel
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

        # For binary classification
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).long().squeeze()
        else:
            # For multi-class classification
            probs = torch.softmax(logits, dim=-1)
            predictions = probs.argmax(dim=-1)

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

# Concatenate predictions and labels
all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average='binary')
precision = precision_score(all_labels, all_predictions, average='binary')
recall = recall_score(all_labels, all_predictions, average='binary')
tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# Print results
print(f"Model: {model_checkpoint}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")

# Optionally, save predictions and labels
os.makedirs("pred_label", exist_ok=True)
torch.save(all_predictions, f"pred_label/{model_name}_predictions.pt")
torch.save(all_labels, f"pred_label/{model_name}_labels.pt")

