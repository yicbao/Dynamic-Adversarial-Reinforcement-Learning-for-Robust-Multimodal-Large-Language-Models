# -*- coding: utf-8 -*-
"""
Evaluate a VLM on HRBench-4K using strict multi-trial evaluation.

For each sample, the model is queried up to NUM_TRIALS times. Evaluation
short-circuits on the first wrong answer; only samples passing all trials
are counted as correct. Reports per-category and overall accuracy.
"""
import argparse
import base64
import concurrent.futures
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import toml
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

VLLM_API_URL = "http://127.0.0.1:25556/v1/chat/completions"
MODEL_IDENTIFIER = "your_model_name"
BENCHMARK_NAME = "HRBench4K"
HRBENCH_ANNOTATIONS_PATH = "../LMUData/hr_bench_4k_local.tsv"
NUM_TRIALS = 3
MAX_WORKERS = 200
MAX_TOKENS = 4096
TEMPERATURE = 0.0
TIMEOUT = 300

try:
    config = toml.load("config/prompts.toml")
    PROMPTS_CONFIG = config['prompts']
except FileNotFoundError:
    print("Error: prompts.toml not found. Ensure it is at 'config/prompts.toml'.")
    exit(1)
except KeyError:
    print("Error: prompts.toml is malformed. Ensure it has a [prompts] section with a 'mcq' key.")
    exit(1)

PROMPT_MCQ_TEMPLATE = PROMPTS_CONFIG['mcq']
OUTPUT_BASE_DIR = f"output/{BENCHMARK_NAME}"


def extract_answer_option(s: str) -> str:
    if not isinstance(s, str):
        return ""
    match = re.search(r'\\?boxed\s*[\{\(]\s*([A-D])\s*[\}\)]', s, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'(?:final answer|the final answer is|the answer is)\s*:?\s*([A-D])', s, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'[A-D]', s)
    if match:
        return match.group(0).upper()
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


def eval_model_row(item_tuple):
    index, item = item_tuple
    image_path = item['image_path']
    ground_truth = item["answer"].strip().upper()

    try:
        if not os.path.exists(image_path):
            print(f"Warning: image not found for index {index}, skipping.")
            return None
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image for index {index}: {e}")
        return None

    options_str = f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
    full_question_block = f"{item['question']}\nOptions:\n{options_str}"
    prompt_text = PROMPT_MCQ_TEMPLATE.format(question=full_question_block)

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]}]

    trial_results = []
    correct_count = 0
    is_final_correct = True

    for _ in range(NUM_TRIALS):
        response_data = call_vllm_api(messages)
        prediction = "API_ERROR"
        raw_output = "API call failed."

        if response_data:
            try:
                raw_output = response_data.get("choices", [{}])[0].get("message", {}).get("content", "PARSE_ERROR")
                prediction = extract_answer_option(raw_output)
            except (IndexError, TypeError, KeyError) as e:
                prediction = "PARSE_ERROR"
                raw_output = f"Error parsing response: {e}"

        is_trial_correct = (prediction == ground_truth)
        trial_results.append({"prediction": prediction, "raw_output": raw_output, "is_correct": is_trial_correct})

        if is_trial_correct:
            correct_count += 1
        else:
            is_final_correct = False
            break

    return {
        "index": index,
        "question": item['question'],
        "multi_choice_options": {'A': item['A'], 'B': item['B'], 'C': item['C'], 'D': item['D']},
        "answer": item['answer'],
        "category": item['category'],
        "vlm_prompt": prompt_text,
        "correct": is_final_correct,
        "correct_trial_count": correct_count,
        "trials": trial_results,
    }


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"Loading dataset from {HRBENCH_ANNOTATIONS_PATH}...")
    try:
        df = pd.read_csv(HRBENCH_ANNOTATIONS_PATH, sep="\t", keep_default_na=False)
    except FileNotFoundError:
        print(f"Error: dataset file not found at {HRBENCH_ANNOTATIONS_PATH}")
        exit(1)

    items_to_process = list(df.iterrows())
    print(f"Loaded {len(items_to_process)} samples.")
    processed_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(eval_model_row, items_to_process), total=len(items_to_process)):
            if result is not None:
                processed_data.append(result)

    total = len(processed_data)
    if total == 0:
        print("No samples processed.")
        exit(0)

    df_results = pd.DataFrame(processed_data)
    correct = df_results['correct'].sum()
    overall_acc = correct / total if total > 0 else 0

    category_stats = {}
    for category, group in df_results.groupby('category'):
        cat_total = len(group)
        cat_correct = group['correct'].sum()
        cat_acc = cat_correct / cat_total if cat_total > 0 else 0
        category_stats[category] = {"accuracy": round(cat_acc, 4), "correct": int(cat_correct), "total": int(cat_total)}
        print(f"  {category:<25}: {cat_correct:>4}/{cat_total:>4} = {cat_acc:.2%}")

    print(f"==> Overall accuracy: {correct}/{total} = {overall_acc:.2%}")

    output_dict = {
        "evaluation_summary": {
            "model_identifier": MODEL_IDENTIFIER,
            "benchmark_name": BENCHMARK_NAME,
            "evaluation_strategy": f"Strict mode: all {NUM_TRIALS} trials must be correct per sample.",
            "overall_accuracy": round(overall_acc, 4),
            "total_samples": total,
            "correct_predictions": int(correct),
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": TEMPERATURE,
            "category_accuracies": category_stats,
        },
        'samples': processed_data,
    }

    final_toml_filename = f"{MODEL_IDENTIFIER}_acc_{overall_acc * 100:.2f}.toml"
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
    parser.add_argument('--dataset_path', type=str, default=HRBENCH_ANNOTATIONS_PATH)
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
    HRBENCH_ANNOTATIONS_PATH = args.dataset_path
    NUM_TRIALS = args.num_trials
    MAX_WORKERS = args.max_workers
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    TIMEOUT = args.timeout
    OUTPUT_BASE_DIR = f"output/{BENCHMARK_NAME}"
    main()
