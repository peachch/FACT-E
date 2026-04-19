import argparse
import json
import csv
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tqdm import tqdm
from collections import Counter
from metrics import Metrics
from api import LLMApiClient

class SelfConsistency:
    def __init__(self, model_name, dataset, portion=1.0, sample_size=5, level = 1):
        self.model_name = model_name
        self.dataset = dataset
        self.portion = portion
        self.level = level
        self.sample_size = sample_size  # Number of samples to generate
        self.client = LLMApiClient()  # Assuming this is your API client
        
        # Initialize data containers
        self.question_list = []
        self.original_prompts = []
        self.all_responses = []  # List of all responses for each question (sample_size per question)
        self.all_cots = []       # List of all CoTs for each question
        self.all_answers = []    # List of all answers for each question
        self.final_cots = []     # CoT from the most common answer
        self.final_answers = []  # Majority vote answer
        self.ground_truths = []
        self.levels = []
        self.agreement_rates = []  # Percentage of answers that agreed with the majority
        self.judge = []
        
    
    def load_data(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data[:int(len(data)*float(self.portion))]
    
    def generate_prompt(self, question, answer_choice):
        if self.dataset == "math500":
            return (
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
        if self.dataset == "commonsense":
            return (
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
        if self.dataset == "gsm8k":
            return f"""Think step to solve the following question. End the response with the result in "Answer: \\boxed{{result}}".
                    First provide the reasoning process (CoT), then give the numeric final answer.
                    Please format your response as follows:

                    CoT: Step-by-step reasoning
                    Answer: Final numeric answer

                    Example:
                    Question: If there are 3 cars and each car has 4 wheels, how many wheels are there in total?
                    CoT: Each car has 4 wheels, so for 3 cars, we multiply the number of cars by the number of wheels per car: 3 cars * 4 wheels/car = 12 wheels.
                    Answer: \\boxed{12}

                    Question: {question}
                    """
        
    def extract_cot_and_answer(self, response):
        """Extracts both the CoT reasoning and final answer from the response"""
        # Implement your extraction logic here
        # This is just a placeholder - replace with your actual extraction method
        if "CoT:" in response and "Answer:" in response:
            cot_part = response.split("CoT:")[1].split("Answer:")[0].strip()
            answer_part = response.split("Answer:")[1].strip()
            return cot_part, answer_part
        else:
            # Fallback if the format isn't perfect
            lines = response.split('\n')
            cot_lines = []
            answer_line = ""
            for line in lines:
                if line.startswith("Answer:"):
                    answer_line = line.replace("Answer:", "").strip()
                else:
                    cot_lines.append(line)
            return '\n'.join(cot_lines).strip(), answer_line
    
    def majority_vote(self, answers):
        """Perform majority voting on the answers"""
        if not answers:
            return None, 0.0
        
        counter = Counter(answers)
        most_common = counter.most_common(1)
        majority_answer, count = most_common[0]
        agreement_rate = count / len(answers)
        
        return majority_answer, agreement_rate
    
    def get_cot_for_answer(self, answers, cots, target_answer):
        """Get the first CoT that matches the majority answer"""
        for answer, cot in zip(answers, cots):
            if answer == target_answer:
                return cot
        return cots[0]  # Fallback to first CoT if no match found
    
    def process_question(self, question, ground_truth, level, answer_choice):
        self.question_list.append(question)
        self.ground_truths.append(ground_truth)
        self.levels.append(level)
        
        if self.dataset == "commonsense":
            prompt = self.generate_prompt(question, answer_choice)
        if self.dataset == "math500":
            prompt = self.generate_prompt(question, None)
        if self.dataset == "gsm8k":
            prompt = self.generate_prompt(question, None)

        self.original_prompts.append(prompt)
        
        responses = []
        cots = []
        answers = []
        
        # Generate multiple samples
        for _ in range(self.sample_size):
            try:
                response = self.client.call_model(self.model_name, prompt, 0, 2048)
                cot, answer = self.extract_cot_and_answer(response)
                responses.append(response)
                cots.append(cot)
                answers.append(answer)
            except Exception as e:
                print(f"error q:{question}")
                print(f"error:{e}")
                continue
        
        self.all_responses.append(responses)
        self.all_cots.append(cots)
        self.all_answers.append(answers)
        
        # Perform majority voting
        majority_answer, agreement_rate = self.majority_vote(answers)
        self.agreement_rates.append(agreement_rate)
        self.final_answers.append(majority_answer)
    
    def run(self, data_path):
        data = self.load_data(data_path)
        print(f"Data loaded! Total records: {len(data)}")
        
        for d in tqdm(data[:1]):
            level = d.get("level", None)

            question = d["question"]
            ground_truth = d["answer"]
            answer_choice = None
            if self.dataset == "commonsense":
                choices = d.get("choices", [])
                answer_choice = ', '.join([f"{k}:{v}" for k, v in choices.items()])
            
            self.process_question(question, ground_truth, level, answer_choice)
        
        self.evaluate_results()
        self.save_results()
    
    def save_results(self):
        # Prepare data for saving
        data = {
            "questions": self.question_list,
            "original_prompts": self.original_prompts,
            "all_responses": self.all_responses,
            "all_cots": self.all_cots,
            "all_answers": self.all_answers,
            "final_cots": self.final_cots,
            "final_answers": self.final_answers,
            "ground_truths": self.ground_truths,
            "levels": self.levels,
            "agreement_rates": self.agreement_rates,
            "sample_size": self.sample_size,
            "judge":self.judge
        }
        
        # Create results directory if it doesn't exist
        os.makedirs("baseline/results/self_consistency", exist_ok=True)

        # Save JSON
        json_filename = f"baseline/results/self_consistency/{self.model_name}_{self.dataset}_selfconsistency_{args.level}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Save CSV
        csv_filename = f"baseline/results/self_consistency/{self.model_name}_{self.dataset}_selfconsistency_{args.level}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            headers = ['question', 'original_prompt']
            for i in range(1, self.sample_size + 1):
                headers.append(f'response_{i}')
            for i in range(1, self.sample_size + 1):
                headers.append(f'cot_{i}')
            for i in range(1, self.sample_size + 1):
                headers.append(f'answer_{i}')
            headers.extend([
                'final_answer', 
                'ground_truth', 'level', 'agreement_rate', 'judge'
            ])
            
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for i in range(len(self.question_list)):
                row = [
                    self.question_list[i],
                    self.original_prompts[i]
                ]
                row.extend(self.all_responses[i][:self.sample_size])
                row.extend(self.all_cots[i][:self.sample_size])
                row.extend(self.all_answers[i][:self.sample_size])
                row.extend([
                    self.final_answers[i],
                    self.ground_truths[i],
                    self.levels[i],
                    f"{self.agreement_rates[i]:.2f}",
                    self.judge[i]
                ])
                
                writer.writerow(row)
        
        print("Results saved successfully!")
    
    def evaluate_results(self):
        # Evaluate final answers against ground truth
        evaluate_results = Metrics.evaluate_results(
            self.ground_truths, 
            self.final_answers, 
            self.dataset
        )
        print(f"Evaluation results: {evaluate_results}")

        self.judge = evaluate_results["judge"]
        
        # Calculate average agreement rate
        avg_agreement = sum(self.agreement_rates) / len(self.agreement_rates)
        print(f"Average agreement rate among samples: {avg_agreement:.2%}")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", default="gpt-4o-mini", help="which language model to use")  # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", default="gsm8k", help="which dataset")  # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-o", "--portion", default=0.01, help="portion of the dataset to use")
    argParser.add_argument("-s", "--samples", default=2, type=int, help="number of samples for self-consistency")
    argParser.add_argument("-l", "--level", default=2, type=int, help="only level x questions")
    
    args = argParser.parse_args()
    
    # Dataset file mapping
    dataset_files = {
        'math500': 'math500_test_full',
        'commonsense': 'commonsense_qa_test_full',
        'gsm8k': 'gsm8k_test_full'
    }
    
    file_name = dataset_files.get(args.dataset, args.dataset)

    if args.dataset not in dataset_files:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    data_path = f"./data/{file_name}.json"
    
    # Run self-consistency process
    sc = SelfConsistency(
        model_name=args.model,
        dataset=args.dataset,
        portion=args.portion,
        sample_size=args.samples,
        level=args.level
    )
    
    sc.run(data_path)