# -*- coding: utf-8 -*-
"""
Evaluate a VLM on HallusionBench using strict multi-trial evaluation.

For each sample, the model is queried up to NUM_TRIALS times. Evaluation
short-circuits on the first wrong answer; only samples passing all trials
are counted as correct. Reports aAcc, fAcc, and qAcc metrics.
"""
import argparse
import base64
import concurrent.futures
import os
import re
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import toml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

VLLM_API_URL = "http://127.0.0.1:25556/v1/chat/completions"
MODEL_IDENTIFIER = "your_model_name"
BENCHMARK_NAME = "HallusionBench"
HALLUSIONBENCH_TSV_PATH = "../LMUData/HallusionBench.tsv"
BASE_IMAGE_PATH = "../LMUData/images/HallusionBench"
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


def load_hallusionbench_dataset(tsv_path: str) -> list[dict]:
    path = Path(tsv_path)
    if not path.is_file():
        print(f"Error: dataset file '{tsv_path}' not found.")
        return []
    try:
        df = pd.read_csv(path, sep="\t", keep_default_na=False)
        required_columns = {'index', 'question', 'answer', 'category', 'l2-category', 'image_path'}
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
    payload = {"messages": messages, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "logprobs": False}
    try:
        response = requests.post(VLLM_API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def eval_model_row(sample_item: dict):
    sample_id = sample_item["index"]
    ground_truth = sample_item["answer"].strip().capitalize()

    try:
        original_image = None
        if sample_item.get('image') and isinstance(sample_item['image'], str) and len(sample_item['image']) > 100:
            try:
                img_data = base64.b64decode(sample_item['image'])
                original_image = Image.open(BytesIO(img_data))
            except (ValueError, UnidentifiedImageError, TypeError):
                original_image = None

        if original_image is None:
            image_path = Path(BASE_IMAGE_PATH) / sample_item['image_path']
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            original_image = Image.open(image_path)

        original_image = original_image.convert("RGB")
        buffered = BytesIO()
        original_image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Warning: skipping sample {sample_id}: {e}")
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
                prediction, raw_output = "PARSE_ERROR", f"Error parsing response: {e}"

        is_trial_correct = (prediction == ground_truth)
        trial_results.append({"prediction": prediction, "raw_output": raw_output, "is_correct": is_trial_correct})

        if not is_trial_correct:
            is_final_correct = False
            break

    result_item = sample_item.copy()
    result_item.pop('image', None)
    result_item.update({"vlm_prompt": prompt_text, "correct": is_final_correct, "trials": trial_results})
    return result_item


def calculate_hallusionbench_metrics(results_df: pd.DataFrame):
    if 'correct' not in results_df.columns or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df['score'] = df['correct'].astype(int)

    def extract_id(x, part):
        try:
            return x.split('_')[part]
        except IndexError:
            return 'unknown'

    df['set_id'] = df['index'].apply(lambda x: extract_id(x, 3))
    df['figure_id'] = df['index'].apply(lambda x: extract_id(x, 4))
    df['question_id'] = df['index'].apply(lambda x: extract_id(x, 5))

    def calc_aAcc(data):
        return np.mean(data['score']) * 100 if not data.empty else 0

    def calc_fAcc(data):
        if data.empty:
            return 0
        res = defaultdict(list)
        for _, line in data.iterrows():
            res[f"{line['l2-category']}_{line['set_id']}_{line['figure_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100 if res else 0

    def calc_qAcc(data):
        if data.empty:
            return 0
        res = defaultdict(list)
        for _, line in data.iterrows():
            res[f"{line['l2-category']}_{line['set_id']}_{line['question_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100 if res else 0

    metrics_results = {'split': [], 'aAcc': [], 'fAcc': [], 'qAcc': [], 'count': []}

    metrics_results['split'].append('Overall')
    metrics_results['aAcc'].append(calc_aAcc(df))
    metrics_results['fAcc'].append(calc_fAcc(df))
    metrics_results['qAcc'].append(calc_qAcc(df))
    metrics_results['count'].append(len(df))

    for cat in sorted(df['category'].unique()):
        sub_df = df[df['category'] == cat]
        metrics_results['split'].append(cat)
        metrics_results['aAcc'].append(calc_aAcc(sub_df))
        metrics_results['fAcc'].append(calc_fAcc(sub_df))
        metrics_results['qAcc'].append(calc_qAcc(sub_df))
        metrics_results['count'].append(len(sub_df))

    for l2_cat in sorted(df['l2-category'].unique()):
        sub_df = df[df['l2-category'] == l2_cat]
        metrics_results['split'].append(l2_cat)
        metrics_results['aAcc'].append(calc_aAcc(sub_df))
        metrics_results['fAcc'].append(calc_fAcc(sub_df))
        metrics_results['qAcc'].append(calc_qAcc(sub_df))
        metrics_results['count'].append(len(sub_df))

    return pd.DataFrame(metrics_results)


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"Loading dataset from {HALLUSIONBENCH_TSV_PATH}...")
    items_to_process = load_hallusionbench_dataset(HALLUSIONBENCH_TSV_PATH)
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

    results_df = pd.DataFrame(processed_data)
    metrics_df = calculate_hallusionbench_metrics(results_df)

    print("\n--- HallusionBench Metrics ---")
    print(metrics_df.to_string(index=False, float_format="%.2f"))

    overall_metrics = metrics_df[metrics_df['split'] == 'Overall'].iloc[0]
    overall_aAcc = overall_metrics['aAcc']
    correct_count = int(np.sum(results_df['correct']))

    output_dict = {
        "evaluation_summary": {
            "model_identifier": MODEL_IDENTIFIER,
            "benchmark_name": BENCHMARK_NAME,
            "evaluation_strategy": f"Strict mode: max {NUM_TRIALS} trials per sample, stops on first failure.",
            "aAcc_overall": round(overall_aAcc, 2),
            "fAcc_overall": round(overall_metrics['fAcc'], 2),
            "qAcc_overall": round(overall_metrics['qAcc'], 2),
            "total_samples_processed": total_processed,
            "correct_samples": correct_count,
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": TEMPERATURE,
        },
        "detailed_accuracies": metrics_df.to_dict('records'),
        'samples': processed_data,
    }

    final_toml_filename = f"{MODEL_IDENTIFIER}_aAcc_{overall_aAcc:.2f}.toml"
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
    parser.add_argument('--dataset_path', type=str, default=HALLUSIONBENCH_TSV_PATH)
    parser.add_argument('--images_dir', type=str, default=BASE_IMAGE_PATH)
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
    HALLUSIONBENCH_TSV_PATH = args.dataset_path
    BASE_IMAGE_PATH = args.images_dir
    NUM_TRIALS = args.num_trials
    MAX_WORKERS = args.max_workers
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    TIMEOUT = args.timeout
    OUTPUT_BASE_DIR = f"output/{BENCHMARK_NAME}"
    main()
