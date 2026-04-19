from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from api import LLMApiClient
from metrics import Metrics

LOGGER = logging.getLogger(__name__)


DATA_PATHS = {
    "commonsense_qa": "./data/commonsense_qa_test_full.json",
    "math500": "./data/math500_test_full.json",
    "gsm8k": "./data/gsm8k_test_full.json"
}

# These error types are used to guide the generation of contrastive CoTs that reflect specific reasoning errors.
# Here only show one error type as an example, more error types can be added to enrich the contrastive examples and better evaluate the faithfulness of the original CoT.
ERROR_TYPES = [
    "Operation Error: Change an operation or mathematical step.",
    "Conceptual Swap: Swap different mathematical or logical concepts.",
    "Misgeneralization: Incorrectly generalize from a concept to a broader rule.",
    "Reordered Logic: Change the order of reasoning steps.",
    "Contradiction: Introduce a contradiction with known facts or conclusions.",
]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def load_data(data_path: str):
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported file format: {data_path}")


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_results(results: dict, model_name: str, dataset_name: str, iteration: int, level: int) -> Tuple[str, str]:
    json_path = f"./output/{model_name}_main_results/iteration{iteration}/{model_name}_{dataset_name}_cot_results_level{level}.json"
    csv_path = f"./output/{model_name}_main_results/iteration{iteration}/{model_name}_{dataset_name}_cot_results_level{level}.csv"

    ensure_parent_dir(json_path)
    ensure_parent_dir(csv_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = build_flat_rows(results)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def build_flat_rows(results: dict) -> List[dict]:
    row_count = len(results.get("questions", []))
    rows: List[dict] = []
    for idx in range(row_count):
        rows.append({
            "question": _safe_index(results.get("questions", []), idx),
            "ground_truth": _safe_index(results.get("ground_truth", []), idx),
            "best_answer": _safe_index(results.get("final_answer", []), idx),
            "best_score": _safe_index(results.get("final_answer_probability", []), idx),
            "cot_best": _safe_index(results.get("cot_best", []), idx),
            "all_answers": _safe_index(results.get("answers", []), idx),
            "all_predictions": _safe_index(results.get("predictions", []), idx),
            "all_probabilities": _safe_index(results.get("probabilities", []), idx),
            "judge": _safe_index(results.get("metrics", {}).get("judge", []), idx),
        })
    return rows


def _safe_index(values: Sequence, idx: int):
    return values[idx] if idx < len(values) else None


def split_steps(cot: str) -> List[str]:
    if not cot:
        return []
    lines = [line.strip() for line in cot.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cot) if part.strip()]


def build_generation_prompt(dataset_name: str, item: dict) -> str:
    if dataset_name == "math500":
        question = item.get("question", "")
        return (
            "Think step by step to solve the following question. "
            "First provide the reasoning process (CoT), then give the final answer.\n\n"
            "Format:\nCoT: <reasoning>\nAnswer: <final numeric answer>\n\n"
            f"Question: {question}"
        )

    if dataset_name == "commonsense_qa":
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer_choice = ", ".join([f"{k}: {v}" for k, v in choices.items()])
        return (
            "Think step by step to solve the following question. "
            "First provide the reasoning process (CoT), then give the final answer from the answer choices.\n\n"
            "Format:\nCoT: <reasoning>\nAnswer: <choice letter and text>\n\n"
            f"Question: {question}\n"
            f"Answer Choices: {answer_choice}"
        )

    if dataset_name == "gsm8k":
        question = item.get("question", "")
        return (
            "Think step by step to solve the following question. "
            "End the response with the result in 'Answer: \\boxed{result}'.\n\n"
            f"Question: {question}"
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_ground_truth(dataset_name: str, item: dict) -> Tuple[str, Optional[int]]:
    if dataset_name == "math500":
        return str(item.get("answer", "")), item.get("level")
    if dataset_name in {"commonsense_qa", "gsm8k"}:
        return str(item.get("answer", "")), None
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def calculate_p_ans(
    client: LLMApiClient,
    model_name: str,
    question: str,
    cot: str,
    answer: str,
    temperature: float,
) -> Tuple[float, str]:
    prompt = (
        f"Question: {question}\n"
        f"CoT: {cot}\n"
        f"Answer: {answer}\n\n"
        "Whether the given CoT can deduce to the Answer of the Question?\n"
        "If only 'True' or 'False' can be used, provide the probability that the answer is 'True'.\n"
        "Output format:\nJudge: <True/False>\nProbability: <number between 0 and 1>"
    )
    result = client.call_model_with_probability(model_name=model_name, prompt=prompt, temperature=temperature)
    print(f"Consistency check response: {result.text}, prob_true: {result.prob_true}, source: {result.source}")
    if not result.valid or result.prob_true is None:
        return 0.0, "False"

    text_upper = result.text.lower()
    judge = "True" if "true" in text_upper and "false" not in text_upper[: text_upper.find("true") + 5] else "False"
    if judge == "False" and result.prob_true >= 0.5:
        judge = "True"
    return float(result.prob_true), judge


def estimate_consistency_score(
    client: LLMApiClient,
    model_name: str,
    question: str,
    cot: str,
    answer: str,
    temperature: float,
    n_trials: int,
) -> float:
    scores: List[float] = []
    for _ in range(max(1, n_trials)):
        p_true, judge = calculate_p_ans(client, model_name, question, cot, answer, temperature)
        scores.append(p_true if judge == "True" else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def generate_contrastive_cots(
    client: LLMApiClient,
    model_name: str,
    question: str,
    before_steps: str,
    after_steps: str,
    temperature: float,
) -> List[dict]:
    contrastive_cots: List[dict] = []
    for error in ERROR_TYPES:
        prompt = (
            "You are given a question and a reasoning chain split at step t.\n"
            "Generate an alternative continuation after step t that explicitly reflects the assigned error type.\n\n"
            f"Question: {question}\n"
            f"Chain before step t:\n{before_steps}\n\n"
            f"Chain after step t:\n{after_steps}\n\n"
            f"Error type: {error}\n\n"
            "Output only the contrastive continuation after step t."
        )
        text = client.call_model(model_name=model_name, prompt=prompt, temperature=temperature)
        text = text.strip()
        if not text:
            continue
        prefix_match = re.search(r"contrastive\s+chain\s+after\s+step\s+t\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if prefix_match:
            text = prefix_match.group(1).strip()
        contrastive_cots.append({"error": error, "cot": text})
    return contrastive_cots


def calculate_p_ord(
    results_contrastive_cots: List[dict],
    client: LLMApiClient,
    model_name: str,
    cot: str,
    question: str,
    temperature: float,
    flip_indices: Sequence[int],
) -> float:
    steps = split_steps(cot)
    if len(steps) <= 1:
        return 0.0

    p_ord_values: List[float] = []
    print(f"Evaluating faithfulness with {len(flip_indices)} flip positions...")
    for t in flip_indices:
        before_step_flip = "\n".join(steps[:t])
        after_step_flip = "\n".join(steps[t:])
        contrastive_cots = generate_contrastive_cots(
            client, model_name, question, before_step_flip, after_step_flip, temperature
        )

        print(f"\nFlip position {t}: Generated {len(contrastive_cots)} contrastive CoTs")

        correct_preference = 0
        valid_pair_count = 0
        preference_trace: List[str] = []

        for contrastive_cot_entry in contrastive_cots:
            pair_prompt = (
                "Choose the better option directly, without explanation.\n\n"
                f"Question: {question}\n"
                f"Previous reasoning:\n{before_step_flip}\n\n"
                f"Option A:\n{after_step_flip}\n\n"
                f"Option B:\n{contrastive_cot_entry['cot']}\n\n"
                "Answer Choice: [A/B/NA]"
            )
            pair_response = client.call_model(model_name=model_name, prompt=pair_prompt, temperature=temperature).strip()
            if pair_response in {"A", "Option A"}:
                correct_preference += 1
                valid_pair_count += 1
                preference_trace.append("A")
            elif pair_response in {"B", "Option B"}:
                valid_pair_count += 1
                preference_trace.append("B")
            else:
                preference_trace.append("NA")

            print(f"Pairwise comparison response: {pair_response}, current preference trace: {preference_trace}")

        results_contrastive_cots.append(
            {
                "flip_index": t,
                "before_step_flip": before_step_flip,
                "after_step_flip": after_step_flip,
                "contrastive_cot_error": contrastive_cots,
                "preference": preference_trace,
                "valid_pair_count": valid_pair_count,
                "correct_preference": correct_preference,
            }
        )

        if valid_pair_count > 0:
            p_ord_values.append(correct_preference / valid_pair_count)

    return sum(p_ord_values) / len(p_ord_values) if p_ord_values else 0.0


def estimate_faithfulness_score(
    results_contrastive_cots: List[dict],
    client: LLMApiClient,
    model_name: str,
    cot: str,
    question: str,
    temperature: float,
    n_checkpoints: int,
) -> float:
    steps = split_steps(cot)
    if len(steps) <= 1:
        return 0.0
    candidate_positions = list(range(1, len(steps)))
    # Here we sample a subset of flip positions to evaluate faithfulness, 
    # which can be adjusted based on the desired trade-off between evaluation thoroughness and cost.
    sampled_positions = random.sample(candidate_positions, min(n_checkpoints, len(candidate_positions)))[:1]

    print(f"Sampled flip positions for faithfulness estimation: {len(sampled_positions)}")

    return calculate_p_ord(results_contrastive_cots, client, model_name, cot, question, temperature, sampled_positions)


def select_best_cot_for_question(
    client: LLMApiClient,
    model_name: str,
    question: str,
    ground_truth: str,
    prompt: str,
    temperature: float,
    reasoning_budget: int,
    consistency_trials: int,
    random_flips: int,
) -> dict:
    best_answer = ""
    best_cot = ""
    best_score = -1.0
    predictions = []
    probabilities = []
    cot_process = []
    answers = []
    contrastive_cots = []

    print(f"Selecting best CoT for question with reasoning budget {reasoning_budget} and consistency trials {consistency_trials}")

    for idx in range(reasoning_budget):
        response = client.call_model(model_name=model_name, prompt=prompt, temperature=temperature)
        # print(f"\n===Candidate {idx + 1} response:\n{response}\n")
        cot, answer = client.extract_cot_and_answer(response)
        answer = answer.strip()

        answers.append({idx: answer})
        predictions.append({idx: answer})
        cot_process.append({idx: cot})

        if not cot or not answer:
            probabilities.append({idx: 0.0})
            contrastive_cots.append({"candidate_index": idx, "details": []})
            if not best_answer and answer:
                best_answer = answer
            continue

        candidate_contrastive_logs: List[dict] = []
        consistency_score = estimate_consistency_score(
            client, model_name, question, cot, answer, temperature, consistency_trials
        )
        if consistency_score == 0.0:
            reliability_score = 0.0
        else:
            faithfulness_score = estimate_faithfulness_score(
                candidate_contrastive_logs,
                client,
                model_name,
                cot,
                question,
                temperature,
                random_flips,
            )
            reliability_score = consistency_score * faithfulness_score

        probabilities.append({idx: reliability_score})
        contrastive_cots.append({"candidate_index": idx, "details": candidate_contrastive_logs})

        if reliability_score > best_score:
            best_score = reliability_score
            best_answer = answer
            best_cot = cot

    if not best_answer:
        for answer_dict in answers:
            value = list(answer_dict.values())[-1]
            if value:
                best_answer = value
                break

    if best_score < 0:
        best_score = 0.0

    return {
        "best_answer": best_answer,
        "best_cot": best_cot,
        "best_score": best_score,
        "predictions": predictions,
        "probabilities": probabilities,
        "cot_process": cot_process,
        "answers": answers,
        "contrastive_cots": contrastive_cots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACT-E candidate selection runner")
    parser.add_argument("--dataset", type=str, default="math500", choices=sorted(DATA_PATHS.keys()))
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning_budget", type=int, default=1, help="Number of candidate CoTs to generate")
    parser.add_argument("--random_flips", type=int, default=3, help="Number of sampled checkpoints for faithfulness estimation")
    parser.add_argument("--consistency_trials", type=int, default=3, help="Number of repeated consistency judgments")
    parser.add_argument("--iteration", type=int, default=1, help="Experiment iteration identifier for output folders")
    parser.add_argument("--level", type=int, default=2, help="Optional math difficulty filter")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    random.seed(args.seed)

    dataset_name = args.dataset
    data_path = DATA_PATHS[dataset_name]
    data = load_data(data_path)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    LOGGER.info("Data loading complete, %d records found", len(data))
    client = LLMApiClient()

    results = {
        "config": vars(args),
        "questions": [],
        "ground_truth": [],
        "level": [],
        "predictions": [],
        "probabilities": [],
        "contrastive_cots": [],
        "cot_best": [],
        "cot_process": [],
        "final_answer": [],
        "answers": [],
        "final_answer_probability": [],
    }

    for item in tqdm(data[:10], desc=f"Processing questions with {args.model_name}"):
        ground_truth, level = extract_ground_truth(dataset_name, item)
        if dataset_name == "math500" and level is not None and level != args.level:
            continue

        question = item.get("question", item.get("problem", ""))
        print(f"\n=== Question ===\n{question}\nGround Truth: {ground_truth}\n")
        prompt = build_generation_prompt(dataset_name, item)
        selection = select_best_cot_for_question(
            client=client,
            model_name=args.model_name,
            question=question,
            ground_truth=ground_truth,
            prompt=prompt,
            temperature=args.temperature,
            reasoning_budget=args.reasoning_budget,
            consistency_trials=args.consistency_trials,
            random_flips=args.random_flips,
        )

        results["questions"].append(question)
        results["ground_truth"].append(ground_truth)
        results["predictions"].append(selection["predictions"])
        results["probabilities"].append(selection["probabilities"])
        results["contrastive_cots"].append(selection["contrastive_cots"])
        results["cot_best"].append(selection["best_cot"])
        results["cot_process"].append(selection["cot_process"])
        results["final_answer"].append(selection["best_answer"])
        results["answers"].append(selection["answers"])
        results["final_answer_probability"].append(selection["best_score"])
        if dataset_name == "math500":
            results["level"].append(level)

        if len(results["questions"]) % 50 == 0:
            if dataset_name.startswith("sym"):
                metrics_result = Metrics.evaluate_symbolic_results(results["ground_truth"], results["final_answer"])
            else:
                metrics_result = Metrics.evaluate_results(results["ground_truth"], results["final_answer"], dataset_name)
            results["metrics"] = metrics_result
            json_path, csv_path = save_results(results, args.model_name, dataset_name, args.iteration, args.level)
            LOGGER.info("Intermediate save. Accuracy=%.4f json=%s csv=%s", metrics_result["accuracy"], json_path, csv_path)

    if dataset_name.startswith("sym"):
        metrics_result = Metrics.evaluate_symbolic_results(results["ground_truth"], results["final_answer"])
    else:
        metrics_result = Metrics.evaluate_results(results["ground_truth"], results["final_answer"], dataset_name)
    results["metrics"] = metrics_result

    json_path, csv_path = save_results(results, args.model_name, dataset_name, args.iteration, args.level)
    LOGGER.info("Evaluation complete! Accuracy: %.4f", metrics_result["accuracy"])
    LOGGER.info("Results saved to: %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
