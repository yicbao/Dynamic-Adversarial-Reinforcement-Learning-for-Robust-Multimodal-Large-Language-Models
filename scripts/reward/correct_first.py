# -*- coding: utf-8 -*-
"""
Reward function for multiple-choice questions (MCQ) used in DAPO/GRPO training.

Scoring (total 1.0):
  - Accuracy: 0.8 if extracted answer matches ground truth, else 0.0
  - Format bonus: 0.2 if the answer is in perfect \\boxed{LETTER} format (only awarded when correct)
"""
import re
import json
import logging
from datetime import datetime
from typing import Any, List, Dict, Tuple

REWARD_WEIGHT_ACCURACY = 0.8
REWARD_WEIGHT_FORMAT = 0.2

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"mcq_reward_log_{timestamp}.jsonl"
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)


def extract_mcq_answer(response: str) -> Tuple[str, bool]:
    """
    Extract the MCQ answer letter from a model response.

    Returns:
        (letter, is_perfect_format): letter is the extracted answer (e.g. 'A'),
            is_perfect_format is True only when the response uses \\boxed{LETTER}.
    """
    if not isinstance(response, str):
        return "EXTRACTION_FAILURE", False

    # Perfect format: \boxed{A}
    perfect_match = re.search(r'\\boxed\{([A-Z])\}', response)
    if perfect_match:
        return perfect_match.group(1), True

    # Acceptable fallback: boxed variants
    robust_match = re.search(r'\\?boxed\s*[\{\(\[]\s*([A-Z])\s*[\}\)\]]', response, re.IGNORECASE)
    if robust_match:
        return robust_match.group(1).upper(), False

    return "EXTRACTION_FAILURE", False


def compute_mcq_reward(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Compute rewards for a batch of MCQ responses.

    Args:
        reward_inputs: list of dicts, each with keys:
            - 'response': model output text
            - 'ground_truth': correct answer letter (e.g. 'A')
            - 'step': training step (for logging)

    Returns:
        List of dicts with keys: 'overall', 'accuracy_score', 'format_score'
    """
    if not isinstance(reward_inputs, list):
        raise TypeError(f"Expected list, got {type(reward_inputs)}")

    scores_list = []
    for reward_input in reward_inputs:
        response = reward_input.get("response")
        ground_truth = reward_input.get("ground_truth")
        step = reward_input.get("step", -1)

        accuracy_score = 0.0
        format_score = 0.0

        if response is None or ground_truth is None:
            extracted_letter, is_perfect_format = "INPUT_ERROR", False
        else:
            extracted_letter, is_perfect_format = extract_mcq_answer(response)

        if extracted_letter == ground_truth:
            accuracy_score = REWARD_WEIGHT_ACCURACY
            if is_perfect_format:
                format_score = REWARD_WEIGHT_FORMAT

        overall_score = accuracy_score + format_score
        current_scores = {
            "overall": overall_score,
            "accuracy_score": accuracy_score,
            "format_score": format_score,
        }
        scores_list.append(current_scores)

        log_data = {
            "step": step,
            "inputs": {"response": response, "ground_truth": ground_truth},
            "extraction_details": {
                "extracted_letter": extracted_letter,
                "is_perfect_format": is_perfect_format,
            },
            "reward_scores": current_scores,
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))

    return scores_list
