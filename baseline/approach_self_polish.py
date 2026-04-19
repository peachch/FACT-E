import argparse
import json
import csv
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tqdm import tqdm
from metrics import Metrics
from api import LLMApiClient

class SelfPolish:
    def __init__(self, model_name, dataset, portion=1.0, max_attempts=3):
        self.model_name = model_name
        self.dataset = dataset
        self.portion = portion
        self.max_attempts = max_attempts
        self.client = LLMApiClient()  # Assuming this is your API client
        
        # Initialize data containers
        self.question_list = []
        self.original_prompts = []
        self.responses = []  # List of response sequences for each question
        self.cot_sequences = []  # List of CoT sequences for each question
        self.answer_sequences = []  # List of answer sequences for each question
        self.final_cots = []  # Final CoT for each question
        self.final_answers = []
        self.ground_truths = []
        self.levels = []
        self.attempt_counts = []  # Track how many attempts each question took
        self.judges = []
        self.answer_choices = []

    def load_data(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data[:int(len(data)*float(self.portion))]
    
    def generate_initial_prompt(self, question, answer_choice):
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
            f"First provide the reasoning process (CoT), then give the numeric final answer. "
            f"Please format your response as follows:\n\n"
            f"CoT: Step-by-step reasoning\n"
            f"Answer: Final numeric answer\n\n"
            f"Example:\n"
            f"Q: Where would you find a tuxedo?\n"
            f"A: A tuxedo is a type of formal suit typically worn at special events such as weddings, galas, or formal dinners. Therefore, you would most likely find a tuxedo in places associated with such events or in stores that sell formal wear.\n"
            f"CoT: A tuxedo is a formal suit. It is worn at special events like weddings and galas. These events are held in places like banquet halls, hotels, or event venues. Formal wear is also sold in clothing stores that specialize in suits and formal attire.\n"
            f"Answer: Clothing stores or event venues.\n"
            f"Question: {question}, Answer choice: {answer_choice}"
            )
        if self.dataset == "gsm8k":
            return (f"""Think step to solve the following question. End the response with the result in "Answer: \\boxed{{result}}".
                    First provide the reasoning process (CoT), then give the numeric final answer.
                    Please format your response as follows:

                    CoT: Step-by-stepAnswer: Final numeric answer

                    Example:
                    CoT: Step 1: Calculate sugar for 8 batches of suckers: 8×30 = 240 ounces
                    Step 2: Add the sugar for 1 batch of fudge: 240 +  310 ounces
                    Answer: \\boxed{{310}}

                    Question: {question}
                    End the response with the result in "Answer: \\boxed{{result}}".
                    """)
        
    def generate_reflection_prompt(self, question, CoT, answer, answer_choice):
        if self.dataset == "math500":
            return (
            f"Question: {question}\n Original CoT: {CoT}\n Original Answer: {answer}\n"
            "Based on your previous answer and CoT to this question, please rewrite new versions of the CoT to be more understandable "
            "and more relevant to the question. Don't omit any useful information, especially the numbers, and please maintain "
            "their original meaning when polysemous words appear."
            f"Please format your response as follows:\n\n"
            f"CoT: Step-by-step reasoning\n"
            f"Answer: Final numeric answer\n\n"
            f"Example:\n"
            f"CoT: $f(-2)+f(-1)+f(0)=\\frac{{3(-2)-2}}{{-2-2}}+\\frac{{3(-1)-2}}{{-1-2}}+\\frac{{3(0)-2}}{{0-2}}=\\frac{{-8}}{{-4}}+\\frac{{-5}}{{-3}}+\\frac{{-2}}{{-2}}=2+\\frac{{5}}{{3}}+1=\\boxed{{\\frac{{14}}{{3}}}}$ \n"
            f"Answer: \\frac{{14}}{{3}} \n\n"
            f"Question: {question}"
        )
        if self.dataset == "commonsense":
            return (f"Question: {question}\nAnswer choice: {answer_choice}\n Original CoT: {CoT}\n Original Answer: {answer}\n"
            "Based on your previous answer and CoT to this question, please rewrite new versions of the CoT to be more understandable "
            "and more relevant to the question. Don't omit any useful information, especially the numbers, and please maintain "
            "their original meaning when polysemous words appear."
            f"Please format your response as follows:\n\n"
            f"CoT: Step-by-step reasoning\n"
            f"Answer: Final answer choice\n\n"
            f"Example:\n"
            f"Q: Where would you find a tuxedo?\n"
            f"A: A tuxedo is a type of formal suit typically worn at special events such as weddings, galas, or formal dinners. Therefore, you would most likely find a tuxedo in places associated with such events or in stores that sell formal wear.\n"
            f"CoT: A tuxedo is a formal suit. It is worn at special events like weddings and galas. These events are held in places like banquet halls, hotels, or event venues. Formal wear is also sold in clothing stores that specialize in suits and formal attire.\n"
            f"Answer: Clothing stores or event venues.\n"
            f"Question: {question}, Answer choice: {answer_choice}"
            )
        if self.dataset == "gsm8k":
            return ( f"Question: {question}\nAnswer choice: {answer_choice}\n Original CoT: {CoT}\n Original Answer: {answer}\n"
                "Based on your previous answer and CoT to this question, please rewrite new versions of the CoT to be more understandable "
                "and more relevant to the question. Don't omit any useful information, especially the numbers, and please maintain "
                "their original meaning when polysemous words appear."
                f"Please format your response as follows:\n\n"

                """CoT: Step-by-stepAnswer: Final numeric answer

                Example:
                CoT: Step 1: Calculate sugar for 8 batches of suckers: 8×30 = 240 ounces
                Step 2: Add the sugar for 1 batch of fudge: 240 +  310 ounces
                Answer: \\boxed{{310}} """
                f"Question: {question}"
                """End the response with the result in "Answer: \\boxed{{result}}".""" )
    
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
    
    def is_converged(self, response1, response2):
        # Compare the extracted answers only
        _, answer1 = self.extract_cot_and_answer(response1)
        _, answer2 = self.extract_cot_and_answer(response2)
        return answer1 == answer2
    
    def process_question(self, question, ground_truth, level, answer_choice):
        self.question_list.append(question)
        self.ground_truths.append(ground_truth)
        self.levels.append(level)
        self.answer_choices.append(answer_choice)

        initial_prompt = self.generate_initial_prompt(question, answer_choice)
        self.original_prompts.append(initial_prompt)
        
        response_sequence = []
        cot_sequence = []
        answer_sequence = []
        attempts = 0
        converged = False
        
        # Get initial response
        current_response = self.client.call_model(self.model_name, initial_prompt, 0, 2048)
        response_sequence.append(current_response)
        current_cot, current_answer = self.extract_cot_and_answer(current_response)
        cot_sequence.append(current_cot)
        answer_sequence.append(current_answer)
        attempts += 1
        
        # Self-polish loop
        while attempts < self.max_attempts and not converged:
            reflect_prompt = self.generate_reflection_prompt(question, current_cot, current_answer, answer_choice)
            new_response = self.client.call_model(self.model_name, reflect_prompt, 0, 2048)
            
            new_cot, new_answer = self.extract_cot_and_answer(new_response)
            
            if self.is_converged(current_response, new_response):
                converged = True
            
            response_sequence.append(new_response)
            cot_sequence.append(new_cot)
            answer_sequence.append(new_answer)
            current_response = new_response
            current_cot, current_answer = new_cot, new_answer
            attempts += 1
        
        self.responses.append(response_sequence)
        self.cot_sequences.append(cot_sequence)
        self.answer_sequences.append(answer_sequence)
        self.final_cots.append(current_cot)
        self.final_answers.append(current_answer)
        self.attempt_counts.append(attempts)
        
        return response_sequence
    
    def run(self, data_path):
        data = self.load_data(data_path)
        print(f"Data loaded! Total records: {len(data)}")
        
        for d in tqdm(data[:100]):
            
            ground_truth = d["answer"]
            
            
            if self.dataset == "commonsense":
                question = d["question"]
                choices = d.get("choices", [])
                answer_choice = ', '.join([f"{k}:{v}" for k, v in choices.items()])
                self.process_question(question, ground_truth, None, answer_choice)
            
            if self.dataset == "math500":
                question = d["question"]
                level = d.get("level", None)
                self.process_question(question, ground_truth, level, None)
            
            if self.dataset == "gsm8k":
                question = d["question"]
                self.process_question(question, ground_truth, None, None)
                    
        
        self.evaluate_results()
        self.save_results()
    
    def save_results(self):
        # Prepare data for saving
        data = {
            "questions": self.question_list,
            "original_prompts": self.original_prompts,
            "response_sequences": self.responses,
            "cot_sequences": self.cot_sequences,
            "answer_sequences": self.answer_sequences,
            "final_cots": self.final_cots,
            "final_answers": self.final_answers,
            "ground_truths": self.ground_truths,
            "levels": self.levels,
            "attempt_counts": self.attempt_counts,
            "answer_choice": self.answer_choices,
            "judge": self.judges
        }

        # Create results directory if it doesn't exist
        os.makedirs("baseline/results/self_polish", exist_ok=True)
        
        # Save JSON
        json_filename = f"baseline/results/self_polish/{self.model_name}_{self.dataset}_selfpolish.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Save CSV
        csv_filename = f"baseline/results/self_polish/{self.model_name}_{self.dataset}_selfpolish.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Generate dynamic headers based on max_attempts
            headers = ['question', 'original_prompt']
            
            # Add response sequence headers
            headers.extend([f'response_{i+1}' for i in range(self.max_attempts)])
            # Add cot sequence headers
            headers.extend([f'cot_{i+1}' for i in range(self.max_attempts)])
            # Add answer sequence headers
            headers.extend([f'answer_{i+1}' for i in range(self.max_attempts)])
            
            # Add remaining headers
            headers.extend([
                'final_cot', 'final_answer', 'ground_truth', 
                'level', 'attempts', 'judge', 'answer_choice'
            ])
            
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for i in range(len(self.question_list)):
                # Pad sequences with empty strings to match max_attempts
                responses = (self.responses[i] + [''] * (self.max_attempts - len(self.responses[i])))[:self.max_attempts]
                cots = (self.cot_sequences[i] + [''] * (self.max_attempts - len(self.cot_sequences[i])))[:self.max_attempts]
                answers = (self.answer_sequences[i] + [''] * (self.max_attempts - len(self.answer_sequences[i])))[:self.max_attempts]
                
                # Build the row dynamically
                row = [
                    self.question_list[i],
                    self.original_prompts[i],
                    *responses,  # Unpack the response sequence
                    *cots,      # Unpack the cot sequence
                    *answers,    # Unpack the answer sequence
                    self.final_cots[i],
                    self.final_answers[i],
                    self.ground_truths[i],
                    self.levels[i],
                    self.attempt_counts[i],
                    self.judges[i],
                    self.answer_choices[i]  
                ]
                
                writer.writerow(row)
        
        print("Results saved successfully!")
    
    def evaluate_results(self):
        # Evaluate final answers against ground truth
        evaluate_results = Metrics.evaluate_results(
            self.ground_truths, 
            self.final_answers, 
            self.dataset
        )
        self.judges = evaluate_results["judge"]
        print(f"Evaluation results: {evaluate_results}")
        
        

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", default="gpt-4o-mini", help="which language model to use") # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", default="gsm8k", help="which dataset") # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-o", "--portion", default = 0.01, help="portion of the dataset to use")
    argParser.add_argument("-a", "--attempts", default=2, type=int, help="max self-polish attempts")
    
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
    
    # Run self-polish process
    polish = SelfPolish(
        model_name=args.model,
        dataset=args.dataset,
        portion=args.portion,
        max_attempts=args.attempts
    )
    
    polish.run(data_path)