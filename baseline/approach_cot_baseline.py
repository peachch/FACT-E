import json
import argparse
import os
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tqdm import tqdm
import pandas as pd
import  csv
from api import LLMApiClient
from metrics import Metrics
# import config  # Import global configuration

def cot_baseline(data_path, model_name, temperature, api_key=None):
    """
    Evaluate the performance of LLM using Chain of Thought (CoT) method

    Args:
        data_path: Path to the dataset
        model_name: Name of the model to use
        temperature: Generation temperature
        api_key: OpenAI API key
    """
    # Load dataset
    print(f"Loading dataset: {data_path}")
    fixed_files = {
        "commonsense_qa": "./data/commonsense_qa_test_full.json",
        "math500": "./data/math500_test_full.json",
        "gsm8k": "./data/gsm8k_test_full.json"
    }
    if data_path in fixed_files:
        data_path = fixed_files[data_path]

    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    elif data_path.endswith('.csv'):

        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    print(f"Dataset loaded successfully, total {len(data)} records")

    # Initialize API client
    client = LLMApiClient()

    # Store results
    results = {
        "questions": [],
        "ground_truth": [],
        "level": [],
        "predictions": [],
        "cot_best": [],
        "raw_responses": [],
        "answer_choice": []
    }
    dataset_name = os.path.basename(data_path).split(".")[0]  # Extract dataset name from file path

    print(f"===:{dataset_name}")

    if dataset_name == "math500_test_full":
    # Process each question
        for item in tqdm(data[:1], desc=f"Processing questions using {model_name}"):
            try:
                level = item.get("level", "")
                if level == 5:
                    print('------------------------break--------------------------')
                    continue
            except Exception as e:
                level = 0
            question = item.get("question", "")
            ground_truth = item.get("answer", "")
            
            prompt = f"""Think step to solve the following question. End the response with the result in "Answer: \\boxed{{result}}".
                    First provide the reasoning process (CoT), then give the numeric final answer.
                    Please format your response as follows:

                    CoT: Step-by-stepAnswer: Final numeric answer

                    Example:
                    CoT: Step 1: Calculate sugar for 8 batches of suckers: 8×30 = 240 ounces
                    Step 2: Add the sugar for 1 batch of fudge: 240 +  310 ounces
                    Answer: \\boxed{{310}}

                    Question: {question}
                    End the response with the result in "Answer: \\boxed{{result}}".
                    """

            try:
                # Call API
                response = client.call_model(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature
                )

                # Extract CoT and answer
                cot, answer = client.extract_cot_and_answer(response)

                # Store results
                results["questions"].append(question)
                results["ground_truth"].append(ground_truth)
                results["level"].append(level)
                results["predictions"].append(answer)
                results["cot_best"].append(cot)
                results["raw_responses"].append(response)

            except Exception as e:
                print(f"Error occurred while processing the question: {e}")
                results["questions"].append(question)
                results["ground_truth"].append(ground_truth)
                results["level"].append(level)
                results["predictions"].append("")
                results["cot_best"].append("")
                results["raw_responses"].append(f"Error: {str(e)}")

    elif dataset_name == "gsm8k_test_full":
        for item in tqdm(data[:3], desc=f"Processing questions using {model_name}"):
            question = item.get("question", "")
            ground_truth = item.get("answer", "")
            level = 0
            prompt = f"""Think step to solve the following question. End the response with the result in "Answer: \\boxed{{result}}".
                    First provide the reasoning process (CoT), then give the numeric final answer.
                    Please format your response as follows:

                    CoT: Step-by-stepAnswer: Final numeric answer

                    Example:
                    CoT: Step 1: Calculate sugar for 8 batches of suckers: 8×30 = 240 ounces
                    Step 2: Add the sugar for 1 batch of fudge: 240 +  310 ounces
                    Answer: \\boxed{{310}}

                    Question: {question}
                    End the response with the result in "Answer: \\boxed{{result}}".
                    """
            try:
                # Call API
                response = client.call_model(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature
                )

                # Extract CoT and answer
                cot, answer = client.extract_cot_and_answer(response)

                # Store results
                results["questions"].append(question)
                results["ground_truth"].append(ground_truth)
                results["level"].append(level)
                results["predictions"].append(answer)
                results["cot_best"].append(cot)
                results["raw_responses"].append(response)

            except Exception as e:
                print(f"Error occurred while processing the question: {e}")
                results["questions"].append(question)
                results["ground_truth"].append(ground_truth)
                results["level"].append(level)
                results["predictions"].append("")
                results["cot_best"].append("")
                results["raw_responses"].append(f"Error: {str(e)}")
        
    elif dataset_name == "commonsense_qa_test_full":
        for item in tqdm(data[:3], desc=f"Processing questions using {model_name}"):
            question = item.get("question", "")
            ground_truth = item.get("answer", "")
            choices = item.get("choices", [])
            answer_choice = ', '.join([f"{k}:{v}" for k, v in choices.items()])
            
            # Construct prompt
            prompt = (
                f"Think step by step to solve the following question. "
                f"First provide the reasoning process (CoT), then give the final answer from answer choice. "
                f"Please format your response as follows:\n\n"
                f"CoT: Step-by-step reasoning\n"
                f"Answer: Choice answer\n\n"
                f"Example:\n"
                f"Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? \n Answer Choices: A. Bank, B. Library, C. Department store, D. New York, E. Mall \n"
                f"CoT: Revolving doors allow people to enter and exit simultaneously, which is the “two direction travel” mentioned. The key point in the second half of the sentence is “serves as a security measure.” Where is security especially important among the options? Library: Usually does not require high security, not common for revolving doors to be a security measure here. Department store / Mall: While both control flow, security is not the primary reason for revolving doors. New York: This is a place, not a building type, and doesn’t fit the question as worded. Bank: Security is paramount in a bank. Controlling who enters and leaves, and preventing unauthorized access, is a main reason for having a revolving door as a security measure."
                f"Answer: A. \n\n"
                f"Question: {question}\n"
                f"Answer Choices: {answer_choice}"
            )

            try:
                # Call API
                response = client.call_model(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature
                )

                # Extract CoT and answer
                cot, answer = client.extract_cot_and_answer(response)

                # Store results
                results["questions"].append(question)
                results["answer_choice"].append(answer_choice)
                results["ground_truth"].append(ground_truth)
                results["predictions"].append(answer)
                results["cot_best"].append(cot)
                results["raw_responses"].append(response)

            except Exception as e:
                print(f"Error occurred while processing the question: {e}")
                results["questions"].append(question)
                results["answer_choice"].append(answer_choice)
                results["ground_truth"].append(ground_truth)
                results["predictions"].append("")
                results["cot_best"].append("")
                results["raw_responses"].append(f"Error")

    elif dataset_name == "gsm8k":
        print("todo")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Calculate accuracy
    try:
        metrics_result = Metrics.evaluate_results(
            results["ground_truth"],
            results["predictions"],
            dataset_name=dataset_name,
        )
    except:
        metrics_result = {"accuracy": 0.0}
        print("calculate acc wrong!")

    # Save results
    # dataset_name = os.path.basename(data_path).split(".")[0]  # Extract dataset name from file path
    output_path = f"baseline/results/cot_baseline/{model_name}_{dataset_name}_cot_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results["metrics"] = metrics_result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation completed! Accuracy: {metrics_result['accuracy']:.4f}")
    print(f"Results saved to: {output_path}")

    judge = results.get('metrics', {}).get('judge', [''] * len(results.get('questions', [])))
    csv_path = f"baseline/results/cot_baseline/{args.model_name}_{dataset_name}_cot_results.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            'question', 'level', 'cot_best',
            'ground_truth', 'final_answer', 'judge'
        ])
        for i in range(len(results.get('questions', []))):
            writer.writerow([
                results.get('questions', [])[i] if i < len(results.get('questions', [])) else '',
                results.get('level', [])[i] if i < len(results.get('level', [])) else '',
                results.get('answer_choice', [])[i] if i < len(results.get('answer_choice', [])) else '',
                results.get('cot_best', [])[i] if i < len(results.get('cot_best', [])) else '',
                # results.get('cot_process', [])[i] if i < len(results.get('cot_process', [])) else '',
                results.get('ground_truth', [])[i] if i < len(results.get('ground_truth', [])) else '',
                results.get('predictions', [])[i] if i < len(results.get('predictions', [])) else '',
                # results.get('final_answer_probability', [])[i] if i < len(results.get('final_answer_probability', [])) else '',
                judge[i] if i < len(judge) else ''
            ])
    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoT Baseline Evaluation")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (optional, default uses API_KEY from config.py)")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the model to use (refer to api.py)")
    parser.add_argument("--data_path", type=str, default="./data/", help="Directory containing the dataset (fixed to ./data)")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature")
    args = parser.parse_args()

    # Ensure output directory exists
    data_paths = {
        "commonsense_qa": "./data/commonsense_qa_test_full.json",
        "math500": "./data/math500_test_full.json",
        "gsm8k": "./data/gsm8k_test_full.json"
    }
    db = list(data_paths.keys())[0]

    for data_path in data_paths:
        cot_baseline(
            data_path,
            args.model_name,
            args.temperature,
            api_key=args.api_key
            # db = db
        )
