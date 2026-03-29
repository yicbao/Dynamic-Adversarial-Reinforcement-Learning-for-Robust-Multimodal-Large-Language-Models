# -*- coding: utf-8 -*-
"""
Evaluate a VLM on POPE using strict multi-trial evaluation.

For each sample, the model is queried up to NUM_TRIALS times. Evaluation
short-circuits on the first wrong answer. Reports accuracy, precision,
recall, and F1 per category and overall.
"""
import argparse
import base64
import concurrent.futures
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import toml
from tqdm import tqdm

VLLM_API_URL = "http://127.0.0.1:25556/v1/chat/completions"
MODEL_IDENTIFIER = "your_model_name"
BENCHMARK_NAME = "POPE"
POPE_TSV_PATH = "../LMUData/POPE.tsv"
NUM_TRIALS = 3
MAX_WORKERS = 64
MAX_TOKENS = 4096
TEMPERATURE = 0.0
TIMEOUT = 1800

try:
    config = toml.load("config/prompts.toml")
    PROMPTS_CONFIG = config['prompts']
except FileNotFoundError:
    print("Error: prompts.toml not found. Ensure it is at 'config/prompts.toml'.")
    exit(1)
except KeyError:
    print("Error: prompts.toml is malformed. Ensure it has a [prompts] section with a 'yorn' key.")
    exit(1)

PROMPT_YES_NO_TEMPLATE = PROMPTS_CONFIG['yorn']
OUTPUT_BASE_DIR = f"output/{BENCHMARK_NAME}"


def load_pope_dataset(tsv_path: str) -> list[dict]:
    path = Path(tsv_path)
    if not path.is_file():
        print(f"Error: dataset file '{tsv_path}' not found.")
        return []
    try:
        df = pd.read_csv(path, sep="\t", keep_default_na=False)
        required_columns = {'index', 'question', 'answer', 'category', 'image'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            print(f"Error: TSV missing columns: {', '.join(missing)}")
            return []
        df['index'] = df['index'].astype(str)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading TSV '{tsv_path}': {e}")
        return []


def extract_yes_no(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    match = re.search(r'(?:final answer|the final answer is|the answer is)\s*:?\s*\b(Yes|No)\b', s, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    match = re.search(r'\\?boxed\s*[\{\(]\s*\b(Yes|No)\b\s*[\}\)]', s, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    match = re.search(r'\b(Yes|No)\b', s, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "NO_ANSWER_FOUND"


def call_vllm_api(messages: list):
    payload = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "logprobs": False,
    }
    try:
        response = requests.post(VLLM_API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def eval_model_row(sample_item: dict):
    sample_id = sample_item["index"]
    ground_truth = sample_item["answer"].strip().capitalize()

    try:
        image_base64 = sample_item["image"]
        if not image_base64 or not isinstance(image_base64, str):
            raise ValueError("Image base64 string is missing or invalid.")
        base64.b64decode(image_base64)
    except KeyError:
        print(f"Warning: sample {sample_id} missing 'image' field, skipping.")
        return None
    except (ValueError, Exception) as e:
        print(f"Warning: error processing image for sample {sample_id}: {e}, skipping.")
        return None

    prompt_text = PROMPT_YES_NO_TEMPLATE.format(question=sample_item['question'])
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]}]

    trial_results = []
    is_final_correct = True

    for _ in range(NUM_TRIALS):
        response_data = call_vllm_api(messages)
        prediction = "API_ERROR"
        raw_output = "API call failed."

        if response_data:
            try:
                raw_output = response_data.get("choices", [{}])[0].get("message", {}).get("content", "PARSE_ERROR")
                prediction = extract_yes_no(raw_output)
            except (IndexError, TypeError, KeyError) as e:
                prediction = "PARSE_ERROR"
                raw_output = f"Error parsing response: {e}"

        is_trial_correct = (prediction == ground_truth)
        trial_results.append({"prediction": prediction, "raw_output": raw_output, "is_correct": is_trial_correct})

        if not is_trial_correct:
            is_final_correct = False
            break

    return {
        "index": sample_id,
        "question": sample_item['question'],
        "answer": sample_item['answer'],
        "category": sample_item['category'],
        "vlm_prompt": prompt_text,
        "correct": is_final_correct,
        "trials": trial_results,
    }


def calculate_pope_metrics(processed_data: list[dict]) -> tuple[pd.DataFrame, dict]:
    if not processed_data:
        return pd.DataFrame(), {}

    def cal_f1_score(y_true, y_pred):
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score, precision, recall

    df = pd.DataFrame(processed_data)
    df['score'] = df['correct']
    df['extracted'] = df['trials'].apply(lambda x: x[-1]['prediction'] if x else "NO_TRIAL")

    df_exploded = df.assign(category=df['category'].str.split(',')).explode('category')
    df_exploded['category'] = df_exploded['category'].str.strip()

    results_summary = {}
    report_data = {'split': [], 'F1-Score': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

    y_true_overall = np.array([1 if i == 'Yes' else 0 for i in df['answer']])
    y_pred_overall = np.array([1 if i == 'Yes' else 0 for i in df['extracted']])
    f1, precision, recall = cal_f1_score(y_true_overall, y_pred_overall)
    accuracy = np.mean(df['score']) if not df.empty else 0

    results_summary['overall'] = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
    }
    report_data['split'].append('Overall')
    report_data['F1-Score'].append(f1 * 100)
    report_data['Accuracy'].append(accuracy * 100)
    report_data['Precision'].append(precision * 100)
    report_data['Recall'].append(recall * 100)

    category_metrics = {}
    categories = sorted([c for c in df_exploded['category'].unique() if pd.notna(c) and c])
    for cat in categories:
        sub_df = df_exploded[df_exploded['category'] == cat]
        y_true_cat = np.array([1 if i == 'Yes' else 0 for i in sub_df['answer']])
        y_pred_cat = np.array([1 if i == 'Yes' else 0 for i in sub_df['extracted']])
        f1_cat, precision_cat, recall_cat = cal_f1_score(y_true_cat, y_pred_cat)
        acc_cat = np.mean(sub_df['score']) if not sub_df.empty else 0

        category_metrics[cat] = {
            'accuracy': round(acc_cat, 4),
            'precision': round(precision_cat, 4),
            'recall': round(recall_cat, 4),
            'f1_score': round(f1_cat, 4),
        }
        report_data['split'].append(cat)
        report_data['F1-Score'].append(f1_cat * 100)
        report_data['Accuracy'].append(acc_cat * 100)
        report_data['Precision'].append(precision_cat * 100)
        report_data['Recall'].append(recall_cat * 100)

    results_summary['by_category'] = category_metrics
    return pd.DataFrame(report_data), results_summary


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"Loading dataset from {POPE_TSV_PATH}...")
    items_to_process = load_pope_dataset(POPE_TSV_PATH)
    if not items_to_process:
        print("No data loaded. Exiting.")
        exit(1)

    print(f"Loaded {len(items_to_process)} samples.")
    processed_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(eval_model_row, items_to_process), total=len(items_to_process)):
            if result is not None:
                processed_data.append(result)

    total_processed = len(processed_data)
    if total_processed == 0:
        print("No samples processed.")
        exit(0)

    report_df, metrics_summary = calculate_pope_metrics(processed_data)
    print(report_df.to_string(index=False, float_format="%.2f"))

    output_dict = {
        "evaluation_summary": {
            "model_identifier": MODEL_IDENTIFIER,
            "benchmark_name": BENCHMARK_NAME,
            "evaluation_strategy": f"Strict mode: max {NUM_TRIALS} trials per sample, short-circuited on first failure.",
            "total_samples_processed": total_processed,
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": TEMPERATURE,
            **metrics_summary,
        },
        'samples': processed_data,
    }

    overall_accuracy = metrics_summary.get('overall', {}).get('accuracy', 0)
    final_toml_filename = f"{MODEL_IDENTIFIER}_acc_{overall_accuracy * 100:.2f}.toml"
    final_toml_path = os.path.join(OUTPUT_BASE_DIR, final_toml_filename)
    try:
        with open(final_toml_path, "w", encoding="utf-8") as f:
            toml.dump(output_dict, f)
        print(f"Results saved to {final_toml_path}")
    except Exception as e:
        print(f"Error saving TOML: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description=f"Evaluate VLM on {BENCHMARK_NAME}.")
    parser.add_argument('--vllm_api_url', type=str, default=VLLM_API_URL)
    parser.add_argument('--model_identifier', type=str, default=MODEL_IDENTIFIER)
    parser.add_argument('--dataset_path', type=str, default=POPE_TSV_PATH)
    parser.add_argument('--num_trials', type=int, default=NUM_TRIALS)
    parser.add_argument('--max_workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--timeout', type=int, default=TIMEOUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    VLLM_API_URL = args.vllm_api_url
    MODEL_IDENTIFIER = args.model_identifier
    POPE_TSV_PATH = args.dataset_path
    NUM_TRIALS = args.num_trials
    MAX_WORKERS = args.max_workers
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    TIMEOUT = args.timeout
    OUTPUT_BASE_DIR = f"output/{BENCHMARK_NAME}"
    main()
