from __future__ import annotations

import re
from typing import Iterable, List, Optional

from sklearn.metrics import accuracy_score


ACTION_PATTERN = r"I_(?:JUMP|LOOK|RUN|TURN_LEFT|TURN_RIGHT|WALK)"


class Metrics:
    @staticmethod
    def extract_answer_from_response(response_texts: Iterable[str]) -> List[str]:
        extracted: List[str] = []
        for response_text in response_texts:
            text = "" if response_text is None else str(response_text)
            boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
            if boxed_match:
                extracted.append(boxed_match.group(1).strip())
                continue
            answer_match = re.search(r"Answer:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            if answer_match:
                extracted.append(answer_match.group(1).strip())
            else:
                extracted.append(text.strip())
        return extracted

    @staticmethod
    def calculate_accuracy(ground_truth: List[str], predictions: List[str]) -> float:
        if len(ground_truth) != len(predictions):
            raise ValueError("The number of ground truth answers and predictions do not match")
        return float(accuracy_score(ground_truth, predictions))

    @staticmethod
    def normalize_answers(answers: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        for answer in answers:
            norm = "" if answer is None else str(answer).lower().strip()
            text_match = re.search(r"\\text\{([^}]*)\}", norm)
            if text_match:
                norm = text_match.group(1)
            boxed_match = re.search(r"\\boxed\{([^}]*)\}", norm)
            if boxed_match:
                norm = boxed_match.group(1)
            norm = norm.replace("$", "")
            norm = re.sub(r"\\left\s*\(", "(", norm)
            norm = re.sub(r"\\right\s*\)", ")", norm)
            norm = re.sub(r"[\[\]{}]", "", norm)
            norm = norm.replace(" ", "")
            norm = norm.replace("\\\\frac", "frac")
            norm = norm.replace("\\frac", "frac")
            if norm.startswith("(") and norm.endswith(")"):
                norm = norm[1:-1]
            normalized.append(norm)
        return normalized

    @staticmethod
    def evaluate_results(ground_truth: List[str], predictions: List[str], dataset_name: Optional[str]) -> dict:
        if dataset_name and dataset_name.lower().startswith("math500"):
            return Metrics._evaluate_math_results(ground_truth, predictions)
        return Metrics._evaluate_string_results(ground_truth, predictions)

    @staticmethod
    def _evaluate_math_results(ground_truth: List[str], predictions: List[str]) -> dict:
        correct = 0
        judge: List[bool] = []
        tolerance = 1e-6

        for gt, pred in zip(ground_truth, predictions):
            gt_decimal = fraction_to_decimal(str(gt))
            pred_decimal = fraction_to_decimal(str(pred))
            is_correct = False
            if gt_decimal is not None and pred_decimal is not None:
                is_correct = abs(gt_decimal - pred_decimal) <= tolerance
            else:
                gt_norm = Metrics.normalize_answers([str(gt)])[0]
                pred_norm = Metrics.normalize_answers([str(pred)])[0]
                is_correct = gt_norm == pred_norm
            judge.append(is_correct)
            correct += int(is_correct)

        total = len(ground_truth)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "sample_count": total,
            "judge": judge,
        }

    @staticmethod
    def _evaluate_string_results(ground_truth: List[str], predictions: List[str]) -> dict:
        extracted_predictions = Metrics.extract_answer_from_response(predictions)
        judge: List[bool] = []
        correct = 0

        for gt, pred in zip(ground_truth, extracted_predictions):
            gt_norm = Metrics.normalize_answers([str(gt)])[0]
            pred_norm = Metrics.normalize_answers([str(pred)])[0]
            is_correct = gt_norm == pred_norm
            judge.append(is_correct)
            correct += int(is_correct)

        total = len(ground_truth)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "sample_count": total,
            "judge": judge,
        }

    @staticmethod
    def evaluate_symbolic_results(ground_truth: List[str], predictions: List[str]) -> dict:
        judge: List[bool] = []
        correct = 0
        total = len(ground_truth)

        for gt, pred in zip(ground_truth, predictions):
            gt_actions = Metrics.extract_symbolic_actions(str(gt))
            pred_actions = Metrics.extract_symbolic_actions(str(pred))
            is_correct = gt_actions == pred_actions
            judge.append(is_correct)
            correct += int(is_correct)

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "sample_count": total,
            "judge": judge,
        }

    @staticmethod
    def extract_symbolic_actions(text: str) -> List[str]:
        cleaned = text.replace("\\", "") if text else ""
        return re.findall(ACTION_PATTERN, cleaned)


def fraction_to_decimal(fraction_str: str) -> Optional[float]:
    if not isinstance(fraction_str, str):
        return None

    fraction_str = fraction_str.strip()
    try:
        latex_match = re.search(r"\\frac\{([^}]+)\}\{([^}]+)\}", fraction_str)
        if latex_match:
            numerator = float(latex_match.group(1))
            denominator = float(latex_match.group(2))
            return numerator / denominator if denominator != 0 else None

        if "/" in fraction_str:
            parts = fraction_str.split("/")
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                return numerator / denominator if denominator != 0 else None

        return float(fraction_str)
    except ValueError:
        return None
