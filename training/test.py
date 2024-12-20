import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BitsAndBytesConfig
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import sys
import os
from peft import PeftModel  # Import PeftModel to load LoRA adapters

if len(sys.argv) != 3:
    print("Usage: python test.py <CWE> <model_checkpoint>")
    sys.exit(1)

# Parse arguments
cwe = str(sys.argv[1])
model_checkpoint = str(sys.argv[2])
model_name = os.path.basename(model_checkpoint)  # Extract the base name
print(model_name)
print(f"Running test with Model Checkpoint: {model_checkpoint}")
print(f"CWE: {cwe}")

# Load test data
# all
# c_test = pd.read_json(f"../dataset/rest/test_all_{cwe}.json")
# specific_cwe
c_test = pd.read_json(f"../dataset/pv_test_{cwe}.json")
c_test = c_test.drop(['cwe', 'hash'], axis=1)
c_test = c_test.reset_index(drop=True)

# Convert test data to Hugging Face Dataset
test_dataset = Dataset.from_pandas(c_test)

# Determine if the model is StarCoder
is_starcoder = 'starcoder' in model_checkpoint.lower()

# Set the base model checkpoint
if is_starcoder:
    base_model_checkpoint = "bigcode/starcoder2-7b"
else:
    base_model_checkpoint = model_checkpoint

# Load model configuration from the base model
config = AutoConfig.from_pretrained(base_model_checkpoint, trust_remote_code=True)

# Load the tokenizer from the fine-tuned model checkpoint to include any added tokens
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, trust_remote_code=True)

# Adjust max length according to model limits
model_max_length = getattr(config, 'max_position_embeddings', 512)
if 'roberta' in config.model_type or 'bert' in config.model_type:
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    model_max_length -= num_special_tokens
max_seq_length = min(4020, model_max_length)
print(f"Using max_seq_length: {max_seq_length}")

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

# Tokenize test dataset
tokenized_datasets = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
#tokenized_datasets = tokenized_datasets.rename_column("new_label", "labels")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Prepare DataLoader
test_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)

# Load model
if not os.path.isdir(model_checkpoint):
    raise FileNotFoundError(f"Model checkpoint directory '{model_checkpoint}' not found.")

if is_starcoder:
    # Load the base model with quantization (remove llm_int8_skip_modules if not used during fine-tuning)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_checkpoint,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    # Resize token embeddings to match the tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    # Load the LoRA adapter
    trained_model = PeftModel.from_pretrained(base_model, model_checkpoint)
else:
    trained_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    trained_model.resize_token_embeddings(len(tokenizer))
    trained_model = trained_model.cuda()

trained_model.eval()

# Determine the device
if is_starcoder:
    device = next(iter(trained_model.parameters())).device
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate model
all_predictions = []
all_labels = []
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.inference_mode():
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = trained_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["labels"].cpu())

# Concatenate predictions and labels
all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)

# Save predictions and labels
os.makedirs("pred_label", exist_ok=True)
model_name = os.path.basename(model_checkpoint)  # Extract the base name
torch.save(all_predictions, f"pred_label/{model_name}_predictions.pt")
torch.save(all_labels, f"pred_label/{model_name}_labels.pt")

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
fpr = fp / (fp + tn)

# Print results
print(f"Model: {model_checkpoint}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"False Positive Rate (FPR): {fpr}")

