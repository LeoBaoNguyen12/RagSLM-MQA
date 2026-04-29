import os
import pandas as pd
import torch
import time
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)

# 1. Configuration
MODEL_NAME = "./biobert_large_stage1_pqa_a_best" 
DATA_PATH = "ori_pqal_flat - ori_pqal_flat.csv"
OUTPUT_DIR = "./biobert_large_stage2_output"
LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}
VRAM_LOG_FILE = "vram_usage_large_stage2.csv"
VAL_RESULTS_FILE = "stage2_large_val_results_full.csv"
METRICS_FILE = "stage2_large_metrics.csv"

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
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df['final_decision'].isin(LABEL_MAP.keys())].reset_index(drop=True)
    df['labels'] = df['final_decision'].map(LABEL_MAP)
    df['bert_text'] = df['question'] + " [SEP] " + df['long_answer'].fillna("")
    return df

# 4. Manual Weighted Trainer for Class Imbalance
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 5. Metrics
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
    if not os.path.exists(MODEL_NAME):
        print(f"Error: Phase 1 model not found at {MODEL_NAME}. Please run Stage 1 first.")
        return

    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.5, stratify=df['labels'], random_state=42)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_df['bert_text'].tolist(), truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_df['bert_text'].tolist(), truncation=True, padding=True, max_length=256)

    train_dataset = PQALDataset(train_encodings, train_df['labels'].tolist())
    val_dataset = PQALDataset(val_encodings, val_df['labels'].tolist())

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,             # Gold set is small, 10 epochs is usually enough
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        gradient_checkpointing=True,        # Memory efficiency
        fp16=True,
        learning_rate=2e-5,              # Lower LR for refinement
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        label_smoothing_factor=0.05, 
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,                # Only keep best model
        save_only_model=True,              # Only save model weights
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none"
    )

    # Weights for yes, no, maybe (adjusting for extreme rarity of maybe)
    class_weights = torch.tensor([1.0, 1.2, 10.0], dtype=torch.float)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            VRAMLoggerCallback(VRAM_LOG_FILE),
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    print("Starting BioBERT-Large Stage 2 Final Fine-Tuning...")
    trainer.train()

    # Final Inference
    print("\nProcessing Final Inference...")
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    predictions = np.argmax(logits, axis=-1)
    
    LABEL_REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}
    val_df['predicted_label'] = [LABEL_REVERSE_MAP[p] for p in predictions]
    val_df['prediction_is_correct'] = val_df['final_decision'] == val_df['predicted_label']
    
    val_df.to_csv(VAL_RESULTS_FILE, index=False)
    print(f"Full results saved to: {VAL_RESULTS_FILE}")

    # Metrics
    acc = accuracy_score(val_df['labels'], predictions)
    f1_m = f1_score(val_df['labels'], predictions, average='macro')
    
    metrics_data = {
        "metric": ["accuracy", "macro_f1"],
        "value": [acc, f1_m]
    }
    pd.DataFrame(metrics_data).to_csv(METRICS_FILE, index=False)
    
    print("\n" + "="*30)
    print(f"FINAL BIOBERT-LARGE METRICS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_m:.4f}")
    print("="*30 + "\n")

    trainer.save_model("./biobert_large_final_best")
    print("Saved optimized Large model to ./biobert_large_final_best")

if __name__ == "__main__":
    main()