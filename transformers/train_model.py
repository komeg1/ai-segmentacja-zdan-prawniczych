from datasets import load_from_disk
import evaluate
import mlflow
import numpy as np
import os
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import torch

os.environ["MLFLOW_TRACKING_AUTH"] = "mlflow_oauth2_client.MlFlowAuthProvider"
os.environ["MLFLOW_SSO_USER"] = ""
os.environ["MLFLOW_SSO_PASSWORD"] = ""

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

MODEL_NAME = ""
DATASET_PATH = ""
GOLD_PATH = ""
OUTPUT_DIR = ""

LEARNING_RATE = 4e-5
BATCH_SIZE = 64
GRAD_ACCUM = 1
EPOCHS = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EVAL_STEPS = 500
BF16 = True

id2label = {0: "O", 1: "B-SENT_END"}
label2id = {"O": 0, "B-SENT_END": 1}

print("Loading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_from_disk(DATASET_PATH)
gold = load_from_disk(GOLD_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Train samples: {len(dataset['train'])} | Test samples: {len(dataset['test'])}")
print(f"Gold samples: {len(gold)}")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
).to(device)


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )
    labels = []
    for i, label in enumerate(examples["sbd_labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


print("Tokenizing training dataset...")
tokenized_ds = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4,
    desc="Tokenizing main dataset",
)

print("Tokenizing gold standard dataset...")
gold_cols = (
    gold.column_names
    if isinstance(gold.column_names, list)
    else gold["train"].column_names
)
tokenized_gold = gold.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=gold_cols,
    num_proc=4,
    desc="Tokenizing gold dataset",
)
print("Tokenization completed.")

metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_preds = [
        [id2label[pred] for pred, label in zip(row_p, row_l) if label != -100]
        for row_p, row_l in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[label] for label in row_l if label != -100]
        for row_l in labels
    ]
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "f1": results["overall_f1"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    fp16=False,
    bf16=BF16,
    dataloader_num_workers=4,
    logging_steps=100,
    report_to="mlflow",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

mlflow.set_tracking_uri("https://mlflow.caise.apl.task.gda.pl/pl0103-05")
mlflow.set_experiment("Master Thesis Project")
mlflow.transformers.autolog()

with mlflow.start_run(run_name=""):
    mlflow.log_params({
        "model": MODEL_NAME,
        "dataset": DATASET_PATH,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE * GRAD_ACCUM,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "class_weights": "none",
        "dropout": 0.1,
        "train_samples": len(tokenized_ds["train"]),
        "test_samples": len(tokenized_ds["test"]),
        "gold_samples": len(tokenized_gold),
    })

    print("\nStarting training...")
    torch.cuda.empty_cache()
    trainer.train()

    print("\nEvaluating on silver test set (2025)...")
    silver_results = trainer.evaluate(metric_key_prefix="silver")
    print(silver_results)

    mlflow.log_metrics({
        "silver_f1": silver_results["silver_f1"],
        "silver_precision": silver_results["silver_precision"],
        "silver_recall": silver_results["silver_recall"],
        "silver_loss": silver_results["silver_loss"],
    })

    print("\nEvaluating on gold standard dataset...")
    gold_results = trainer.evaluate(
        eval_dataset=tokenized_gold, metric_key_prefix="gold"
    )
    print(gold_results)

    mlflow.log_metrics({
        "gold_f1": gold_results["gold_f1"],
        "gold_precision": gold_results["gold_precision"],
        "gold_recall": gold_results["gold_recall"],
        "gold_loss": gold_results["gold_loss"],
    })

    SAVE_PATH = "./models/"
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"\nModel saved to: {SAVE_PATH}")
    print("\n=== SILVER EVALUATION RESULTS ===")
    print(f"F1:        {silver_results['silver_f1']:.4f}")
    print(f"Precision: {silver_results['silver_precision']:.4f}")
    print(f"Recall:    {silver_results['silver_recall']:.4f}")
    print("\n=== GOLD EVALUATION RESULTS ===")
    print(f"F1:        {gold_results['gold_f1']:.4f}")
    print(f"Precision: {gold_results['gold_precision']:.4f}")
    print(f"Recall:    {gold_results['gold_recall']:.4f}")