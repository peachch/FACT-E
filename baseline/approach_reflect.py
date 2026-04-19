import json
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from metrics import Metrics
from api import LLMApiClient
import csv

if __name__ == "__main__":

    # 1、Analyze parameters
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", default="gpt-4o-mini", help="which language model to use") # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", default="gsm8k", help="which dataset") # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-o", "--portion", default = 0.01, help="portion of the dataset to use")
    argParser.add_argument("-l", "--level", default=2, type=int, help="only level x questions")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    portion = args.portion

    # initialize variable
    question_list = []
    model_output = []
    groud_truth = []
    original_prompt_list = []
    original_response = []
    reflect_response_list = []
    cot_output = []
    levels = []

    #2、loading model
    client = LLMApiClient()
    file_name = ''

    dataset_map = {
        'math500': 'math500_test_full',
        'commonsense': 'commonsense_qa_test_full',
        'gsm8k': 'gsm8k_test_full'
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset}")

    file_name = dataset_map[dataset]
        
    with open("./data/"+ file_name + ".json", "r", encoding='utf-8') as f:
        data = json.load(f)

        # Take the specified proportion of data
        data = data[:int(len(data)*float(portion))]
        print(f"data loaded done ! total records:{len(data)}")

        # Request LLM
        for d in tqdm(data[:50]):
            if dataset == 'math500':
                question = d["question"]
                question_list.append(question)
                
                original_prompt = ( 
                    f"Think step by step to solve the following question. "
                    f"First provide the reasoning process (CoT), then give the numeric final answer. "
                    f"Please format your response as follows:\n\n"
                    f"CoT: Step-by-step reasoning\n"
                    f"Answer: Final numeric answer\n\n"
                    f"Example:\n"
                    f"CoT: $f(-2)+f(-1)+f(0)=\\frac{{3(-2)-2}}{{-2-2}}+\\frac{{3(-1)-2}}{{-1-2}}+\\frac{{3(0)-2}}{{0-2}}=\\frac{{-8}}{{-4}}+\\frac{{-5}}{{-3}}+\\frac{{-2}}{{-2}}=2+\\frac{{5}}{{3}}+1=\\boxed{{\\frac{{14}}{{3}}}}$ \n"
                    f"Answer: \\frac{{14}}{{3}} \n\n"
                    f"Question: {question}"
                )
            elif dataset == 'commonsense':
                level = "na"
                predictions = []
                this_cots = []
                final_probability = []
                question = d.get("question", "")
                ground_truth = d.get("answer", "")
                question_list.append(question)
                choices = d.get("choices", [])
                answer_choice = ', '.join([f"{k}:{v}" for k, v in choices.items()])
                original_prompt = (
                    f"Think step by step to solve the following question. "
                    f"First provide the reasoning process (CoT), then give the final answer from answer choice. "
                    f"Please format your response as follows:\n\n"
                    f"CoT: Step-by-step reasoning\n"
                    f"Answer: Choice answer\n\n"
                    f"Example:\n"
                    f"Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? \n Answer Choices: A. Bank, B. Library, C. Department store, D. New York, E. Mall \n"
                    f"CoT: Revolving doors allow people to enter and exit simultaneously, which is the “two direction travel” mentioned. \n"
                    f"The key point in the second half of the sentence is “serves as a security measure.\n"
                    f"Where is security especially important among the options?\n"
                    f"Library: Usually does not require high security, not common for revolving doors to be a security measure here.\n"
                    f"Department store / Mall: While both control flow, security is not the primary reason for revolving doors.\n"
                    f"New York: This is a place, not a building type, and doesn’t fit the question as worded.\n"
                    f"Bank: Security is paramount in a bank. Controlling who enters and leaves, and preventing unauthorized access, is a main reason for having a revolving door as a security measure.\n"
                    f"Answer: A. \n\n"
                    f"Question: {question}\n"
                    f"Answer Choices: {answer_choice}"
                    )
            elif dataset == 'gsm8k':
                predictions = []
                this_cots = []
                
                final_probability = []
                question = d.get("question", "")
                ground_truth = d.get("answer", "")
                question_list.append(question)
                original_prompt = f"""Think step to solve the following question. End the response with the result in "Answer: \\boxed{{result}}".
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
            else:
                raise ValueError(f"No such dataset: {dataset}")

            original_prompt_list.append(original_prompt)

            # Request LLM
            response = client.call_model(model_name, original_prompt, 0, 2048)
            original_response.append(response)

            if dataset == 'math500':
                reflect_prompt = f"Based on the Chain-of-Thought (COT) reasoning and the answer you just provided, please reconsider the " \
                "following question. Confirm the correctness of your prior answer, and then answer it again, also using the " \
                "Chain-of-Thought (COT) format followed by the final answer.\n Previous question: {question}\n Previous response: {response}\n" \
            
            if dataset  == "commonsense":
                reflect_prompt = f"Based on the Chain-of-Thought (COT) reasoning and the answer you just provided, please reconsider the " \
                "following question. Confirm the correctness of your prior answer, and then answer it again, also using the " \
                "Chain-of-Thought (COT) format followed by the final answer.\n Previous question: {question} Answer Choice: {answer_choice}\n Previous response: {response}\n" \

            if dataset  == "gsm8k":
                reflect_prompt = f"Based on the Chain-of-Thought (COT) reasoning and the answer you just provided, please reconsider the " \
                "following question. Confirm the correctness of your prior answer, and then answer it again, also using the " \
                "Chain-of-Thought (COT) format followed by the final answer.\n Previous question: {question}\n Previous response: {response}\n" \
                
            reflect_response = client.call_model(model_name, reflect_prompt, 0, 2048)
            reflect_response_list.append(reflect_response)

            # Dealing with cot and answer
            cot, answer = client.extract_cot_and_answer(response)
            model_output.append(answer)
            cot_output.append(cot)
            groud_truth.append(d["answer"])

    # evaluate results
    evaluate_results = Metrics.evaluate_results(groud_truth, model_output, dataset)
    judge_list = evaluate_results["judge"]

    # Save output content
    print(f"evaluate_results {evaluate_results}")

    # Put all lists into one list
    all_lists = [question_list, original_prompt_list, original_response, reflect_response_list, 
                 cot_output, model_output, groud_truth, judge_list, levels]
    
    data = {
        "questions": question_list,
        "original_prompts": original_prompt_list,
        "original_responses": original_response,
        "reflect_responses": reflect_response_list,
        "cot_outputs": cot_output,
        "model_outputs": model_output,
        "ground_truths": groud_truth,
        "judgments": judge_list,
        "levels": levels
    }

    # Create results directory if it doesn't exist
    os.makedirs("baseline/results/self_reflect", exist_ok=True)

    with open("baseline/results/self_reflect/" + model_name + "_" + dataset + f"_reflect_{args.level}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("saved json")

    with open("baseline/results/self_reflect/" + model_name + "_" + dataset + f"_reflect_{args.level}.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Column name
        writer.writerow(['question', 'original_prompt', 'original_response', 'reflect_response', 'cot', 
                         'model_output', 'groud_truth', 'judge', 'level'])
        # Row Data
        for row in zip(*all_lists):
            try:
                writer.writerow(row)
            except Exception as e:
                print(f"error {e}")

    print("The data has been successfully written")
