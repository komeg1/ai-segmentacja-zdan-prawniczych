import os
import time
import regex as re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = ""
FILES = [
    "../legal-text-downloader/data/acts/2026/act_2026_732_clean.txt",
    "../legal-text-downloader/data/acts/2026/act_2026_656_clean.txt",
    "../legal-text-downloader/data/acts/2026/act_2026_720_clean.txt",
]

WINDOW_SIZE = 350
STRIDE = 175
OUTPUT_DIR = "benchmark_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def custom_tokenize(text):
    text = re.sub(r"\s+", " ", text).strip()
    return re.findall(r"\p{L}+|\d+|[^\s\p{L}\d]", text, flags=re.UNICODE)


def reconstruct(tokens):
    result = ""
    for i, tok in enumerate(tokens):
        if i == 0 or tok in {".", ",", ")", ";", ":", "]", "}", "%"}:
            result += tok
        elif result and result[-1] in {"(", "[", "{"}:
            result += tok
        else:
            result += " " + tok
    return result.strip()


print("Loading PolBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Device: {device}\n")

results = []

for file_path in FILES:
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        found = None
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if basename.replace("_clean.txt", "") in f:
                    found = os.path.join(dir_path, f)
                    break
        if found:
            file_path = found
        else:
            print(f"File not found: {file_path}")
            continue

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_tokens = custom_tokenize(raw_text)
    total_tokens = len(raw_tokens)
    logit_sum = np.zeros(total_tokens, dtype=np.float32)
    logit_cnt = np.zeros(total_tokens, dtype=np.int32)

    if device.type == "cuda":
        dummy = tokenizer(["test"], return_tensors="pt").to(device)
        with torch.no_grad():
            model(**dummy)
        torch.cuda.synchronize()

    start = time.perf_counter()

    for start_idx in range(0, total_tokens, STRIDE):
        end_idx = min(start_idx + WINDOW_SIZE, total_tokens)
        chunk_tokens = raw_tokens[start_idx:end_idx]
        if not chunk_tokens:
            break

        inputs = tokenizer(
            chunk_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        logits_class1 = logits[0, :, 1].cpu().numpy()
        word_ids = inputs.word_ids(batch_index=0)
        prev_word_idx = None

        for sub_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                global_idx = start_idx + word_idx
                if global_idx < total_tokens:
                    logit_sum[global_idx] += logits_class1[sub_idx]
                    logit_cnt[global_idx] += 1
            prev_word_idx = word_idx

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_logits = np.where(logit_cnt > 0, logit_sum / logit_cnt, -999)
    labels = (avg_logits > 0).astype(int)
    num_sentences = int(labels.sum())

    fname = os.path.basename(file_path).replace("_clean.txt", "")
    output_path = os.path.join(OUTPUT_DIR, f"{fname}_polbert.txt")

    with open(output_path, "w", encoding="utf-8") as out_f:
        line_buffer = []
        for i, token in enumerate(raw_tokens):
            line_buffer.append(token)
            if labels[i] == 1 or i == total_tokens - 1:
                out_f.write(reconstruct(line_buffer) + " [SBD_END]\n")
                line_buffer = []

    results.append((fname, total_tokens, num_sentences, elapsed_ms, output_path))
    print(f"{fname}")
    print(f"  Tokens:     {total_tokens:,}")
    print(f"  Sentences:  {num_sentences}")
    print(f"  Time:       {elapsed_ms:.1f} ms")
    print(f"  Output:     {output_path}\n")

print("=" * 60)
print(f"{'File':<40} {'Sentences':>9} {'Time':>10}")
print("-" * 60)
for fname, tokens, sents, ms, _ in results:
    print(f"{fname:<40} {sents:>9} {ms:>9.1f} ms")
