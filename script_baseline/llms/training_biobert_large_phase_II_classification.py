import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
import time

# 1. Configuration
MODEL_NAME = "dmis-lab/biobert-large-cased-v1.1"
TRAIN_DATA_PATH = "pqa_artificial.csv" 
VAL_DATA_PATH = "ori_pqal_flat - ori_pqal_flat.csv"
OUTPUT_DIR = "./biobert_large_stage1_output"
LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}
VRAM_LOG_FILE = "vram_usage_large_stage1.csv"

# 2. VRAM Logging Callback
class VRAMLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = time.time()
        pd.DataFrame(columns=["step", "vram_allocated_mb", "vram_reserved_mb", "elapsed_time_sec"]).to_csv(self.log_file, index=False)

    def on_log(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            elapsed = time.time() - self.start_time
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            new_data = {
                "step": state.global_step, 
                "vram_allocated_mb": allocated, 
                "vram_reserved_mb": reserved,
                "elapsed_time_sec": round(elapsed, 2)
            }
            pd.DataFrame([new_data]).to_csv(self.log_file, mode='a', header=False, index=False)

# 3. Data Loading
def load_and_preprocess(file_path, is_artificial=False, limit=None):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    if limit:
        df = df.head(limit)
    
    if is_artificial:
        df = df[['question', 'final_decision', 'context_flat']]
        df.columns = ['question', 'label', 'context']
    else:
        df = df[['question', 'final_decision', 'long_answer']]
        df.columns = ['question', 'label', 'context']
    
    df = df[df['label'].isin(LABEL_MAP.keys())].reset_index(drop=True)
    df['labels'] = df['label'].map(LABEL_MAP)
    df['text'] = df['question'] + " [SEP] " + df['context'].fillna("")
    
    return df[['text', 'labels']]

# 4. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1_macro}

class PQALDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def main():
    # To save time/memory, we can start with full but keep an eye on VRAM
    train_df = load_and_preprocess(TRAIN_DATA_PATH, is_artificial=True) 
    val_df = load_and_preprocess(VAL_DATA_PATH, is_artificial=False)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # Using max_length 256 for BioBERT-Large on 8GB VRAM to avoid OOM
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=256)

    train_dataset = PQALDataset(train_encodings, train_df['labels'].tolist())
    val_dataset = PQALDataset(val_encodings, val_df['labels'].tolist())

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=16,      # MINIMUM for 8GB VRAM
        gradient_accumulation_steps=16,     # INCREASED to maintain batch size 32
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,        # VRAM SAVER: Trades compute for memory
        fp16=True,                          # Required for memory efficiency
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            VRAMLoggerCallback(VRAM_LOG_FILE),
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    print(f"Starting BioBERT-Large Stage 1 training on {len(train_df)} samples...")
    trainer.train()

    print(f"Saving Best Large Model to {OUTPUT_DIR}...")
    trainer.save_model("./biobert_large_stage1_pqa_a_best")
    tokenizer.save_pretrained("./biobert_large_stage1_pqa_a_best")

if __name__ == "__main__":
    main()