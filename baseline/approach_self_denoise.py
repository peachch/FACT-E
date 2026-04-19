import random
import re
from typing import List, Dict, Tuple
import argparse
import json
import csv
import random
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tqdm import tqdm
from metrics import Metrics
from api import LLMApiClient

class SelfMasking:
    def __init__(self, model_name: str, dataset: str, portion: float = 1.0, num_iterations: int = 5):
        self.model_name = model_name
        self.dataset = dataset
        self.portion = portion
        self.num_iterations = num_iterations
        self.client = LLMApiClient()  # Assuming this is your API client
        
        # Data containers
        self.question_list = []
        self.masked_prompts = []  # List of lists (masked prompts for each iteration)
        self.reconstructed_prompts = []  # List of lists (reconstructed prompts)
        self.responses = []  # List of lists (responses for each iteration)
        self.cot_sequences = []  # List of lists (CoTs for each iteration)
        self.answer_sequences = []  # List of lists (answers for each iteration)
        self.final_answers = []  # Voted final answers
        self.ground_truths = []
        self.levels = []
        self.vote_distributions = []  # Track answer distribution across iterations
        self.judges = []
    
    def load_data(self, file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data[:int(len(data)*float(self.portion))]
    
    def generate_base_prompt(self, question: str, cot: str = None, answer_choice: str = None) -> str:
        if cot is None:
            if self.dataset == "math500":
                return (
                    f"Answer the following multiple-choice question with detailed reasoning. "
                    f"First provide the reasoning process (CoT), then give the final answer. "
                    f"Please format your response as follows:\n"
                    f"CoT: Step-by-step reasoning\n"
                    f"Answer: Final answer\n"
                    f"Example:\n"
                    f"Q: Where would you find a tuxedo?\n"
                    f"A: A tuxedo is a type of formal suit typically worn at special events such as weddings, galas, or formal dinners. Therefore, you would most likely find a tuxedo in places associated with such events or in stores that sell formal wear.\n"
                    f"CoT: A tuxedo is a formal suit. It is worn at special events like weddings and galas. These events are held in places like banquet halls, hotels, or event venues. Formal wear is also sold in clothing stores that specialize in suits and formal attire.\n"
                    f"Answer: Clothing stores or event venues.\n"
                    f"Question: {question}"
                )
            elif self.dataset == "commonsense":
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
            elif self.dataset == "gsm8k":
                return f"""Think step to solve the following question. End the response with the result in "Answer: \\boxed{{result}}".
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
            else: pass
        else:
            
            # Subsequent iterations - use previous CoT with masking
            if self.dataset == "commonsense":
                masked_cot = self.apply_random_masking(cot)
                return (
                    f"Please reconstruct and improve the following reasoning, then solve the question. \n"
                    f"Question: {question}\n"
                    f"Answer Choices: {answer_choice}\n"
                    f"Partial Reasoning: {masked_cot}\n"
                    f"Complete the reasoning by filling in the masked parts ([MASK]), "
                    f"then provide the final answer.\n"
                    f"Format your response as:\n\n"
                    f"CoT: Step-by-step reasoning\n"
                    f"Answer: Final answer choice\n"
                    )
            elif self.dataset == "math500" or self.dataset == "gsm8k":
                masked_cot = self.apply_random_masking(cot)
                return (
                f"Please reconstruct and improve the following reasoning, then solve the question.\n"
                f"Question: {question}\n"
                f"Partial Reasoning: {masked_cot}\n"
                f"Complete the reasoning by filling in the masked parts ([MASK]), "
                f"then provide the final answer.\n"
                f"Format your response as:\n\n"
                f"CoT: Step-by-step reasoning\n"
                f"Answer: Final numeric answer\n"
                """End the response with the result in "Answer: \\boxed{{result}}"""
                )
            else:
                pass
    
    def apply_random_masking(self, text: str, mask_ratio: float = 0.3) -> str:
        """Apply random word-level masking only to CoT portions"""
        # Split into sentences to preserve structure
        sentences = re.split(r'(?<=[.!?])\s+', text)
        masked_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
                
            # Determine how many words to mask in this sentence
            num_to_mask = max(1, int(len(words) * mask_ratio))
            mask_indices = random.sample(range(len(words)), min(num_to_mask, len(words)))
            
            # Apply masking
            masked_words = []
            for idx, word in enumerate(words):
                if idx in mask_indices:
                    mask_token = '[MASK]'
                    masked_words.append(mask_token)
                else:
                    masked_words.append(word)
            
            masked_sentences.append(' '.join(masked_words))
        
        return ' '.join(masked_sentences)
    
    def extract_reconstructed_reasoning(self, response: str) -> Tuple[str, str]:
        """Extract both the reconstructed reasoning and final answer from the response"""
        reasoning_pattern = r"Reconstructed Reasoning:(.*?)Answer:"
        answer_pattern = r"Answer:(.*?)$"
        
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        
        return reasoning, answer
    
    def get_voted_answer(self, answer_sequence: List[str]) -> str:
        """Select the most common answer from all iterations"""
        answer_counts = {}
        for answer in answer_sequence:
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
        
        # Get answer with highest count
        voted_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        self.vote_distributions.append(answer_counts)
        return voted_answer
    
    def process_question(self, question: str, ground_truth: str, level: str, answer_choice) -> List[str]:
        self.question_list.append(question)
        self.ground_truths.append(ground_truth)
        self.levels.append(level)
        
        response_sequence = []
        cot_sequence = []
        answer_sequence = []
        masked_prompts = []
        reconstructed_prompts = []
        previous_cot = None
        
        for i in range(self.num_iterations):
            # Generate prompt based on iteration
            if self.dataset == "commonsense":
                if i == 0:
                    # First iteration - no masking
                    prompt = self.generate_base_prompt(question, None, answer_choice)
                    masked_prompt = prompt
                else:
                    # Subsequent iterations - mask previous CoT
                    prompt = self.generate_base_prompt(question, previous_cot, answer_choice)                  
            if self.dataset == "math500":
                if i == 0:
                    # First iteration - no masking
                    prompt = self.generate_base_prompt(question, None, answer_choice)
                    masked_prompt = prompt
                else:
                    # Subsequent iterations - mask previous CoT
                    prompt = self.generate_base_prompt(question, previous_cot, answer_choice)
            if self.dataset == "gsm8k":
                if i == 0:
                    # First iteration - no masking
                    prompt = self.generate_base_prompt(question, None, None)
                    masked_prompt = prompt
                else:
                    # Subsequent iterations - mask previous CoT
                    prompt = self.generate_base_prompt(question, previous_cot, None)

            masked_prompt = prompt  # Store the masked version
            # Get model response
            response = self.client.call_model(self.model_name, prompt, 0, 2048)
            response_sequence.append(response)
            
            # Extract reasoning and answer
            reasoning, answer = self.client.extract_cot_and_answer(response)
            cot_sequence.append(reasoning)
            answer_sequence.append(answer)
            
            # Store for next iteration
            previous_cot = reasoning
            
            # Track prompts
            masked_prompts.append(masked_prompt)
            if self.dataset == "commonsense":
                reconstructed_prompts.append(
                    f"Question: {question}\nAnswer Choices: {answer_choice}\nReasoning: {reasoning}\nAnswer: {answer}"
                )
            elif self.dataset == "math500" or self.dataset == "gsm8k":
                reconstructed_prompts.append(
                f"Question: {question}\nReasoning: {reasoning}\nAnswer: {answer}"
             )
            elif self.dataset == "gsm8k":
                reconstructed_prompts.append(
                f"Question: {question}\nReasoning: {reasoning}\nAnswer: {answer}"
             )
        
        # Store all data
        self.masked_prompts.append(masked_prompts)
        self.reconstructed_prompts.append(reconstructed_prompts)
        self.responses.append(response_sequence)
        self.cot_sequences.append(cot_sequence)
        self.answer_sequences.append(answer_sequence)
        
        # Get voted answer
        voted_answer = self.get_voted_answer(answer_sequence)
        self.final_answers.append(voted_answer)
        
        return response_sequence
    
    def run(self, data_path: str):
        data = self.load_data(data_path)
        print(f"Data loaded! Total records: {len(data)}")
        
        for d in tqdm(data[:500]):
            
            ground_truth = d["answer"]
            
            if self.dataset == "commonsense":
                question = d["question"]
                choices = d.get("choices", [])
                answer_choice = ', '.join([f"{k}:{v}" for k, v in choices.items()])
            
                self.process_question(question, ground_truth, None, answer_choice)                
            if self.dataset == "math500" :
                question = d["question"]
                level = d.get("level", None)
                self.process_question(question, ground_truth, level, None)  
            if self.dataset  == 'gsm8k':
                question = d["question"]
                self.process_question(question, ground_truth, None, None)

            
        self.evaluate_results()
        self.save_results()
    
    def save_results(self):
        # Prepare data for saving
        data = {
            "questions": self.question_list,
            "responses": self.responses,
            "cot_sequences": self.cot_sequences,
            "answer_sequences": self.answer_sequences,
            "final_answers": self.final_answers,
            "ground_truths": self.ground_truths,
            "levels": self.levels,
            "judges": self.judges
        }
        
        # Create results directory if it doesn't exist
        os.makedirs("baseline/results/self_denoise", exist_ok=True)

        # Save JSON
        json_filename = f"baseline/results/self_denoise/{self.model_name}_{self.dataset}_self_denoisie.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Save CSV
        csv_filename = f"baseline/results/self_denoise/{self.model_name}_{self.dataset}_self_denoisie.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Generate dynamic headers based on num_iterations
            headers = ['question']
            
            # Add perturbed prompt headers
            headers.extend([f'masked_prompt_{i+1}' for i in range(self.num_iterations)])
            # Add response headers
            headers.extend([f'response_{i+1}' for i in range(self.num_iterations)])
            # Add cot headers
            headers.extend([f'cot_{i+1}' for i in range(self.num_iterations)])
            # Add answer headers
            headers.extend([f'answer_{i+1}' for i in range(self.num_iterations)])
            
            # Add remaining headers
            headers.extend([
                'final_answer', 'ground_truth', 'level', 'judge'
            ])
            
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for i in range(len(self.question_list)):
                # Build the row dynamically
                row = [self.question_list[i]]
                
                # Add perturbed prompts
                row.extend(self.masked_prompts[i])
                # Add responses
                row.extend(self.responses[i])
                # Add cots
                row.extend(self.cot_sequences[i])
                # Add answers
                row.extend(self.answer_sequences[i])
                
                # Add remaining fields
                row.extend([
                    self.final_answers[i],
                    self.ground_truths[i],
                    self.levels[i],
                    self.judges[i]
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
        self.judges = evaluate_results["judge"]
        print(f"Evaluation results: {evaluate_results}")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", default="gpt-4o-mini", help="which language model to use") # "aya_13b", "chatgpt", "gpt4"
    argParser.add_argument("-d", "--dataset", default="gsm8k", help="which dataset") # "mmlu", "hellaswag", "belebele"
    argParser.add_argument("-o", "--portion", default = 0.01, help="portion of the dataset to use")
    argParser.add_argument("-n", "--iterations", default=2, type=int, help="number of masking iterations")
    
    args = argParser.parse_args()
    
    # Initialize and run the self-masking process
    masker = SelfMasking(
        model_name=args.model,
        dataset=args.dataset,
        portion=args.portion,
        num_iterations=args.iterations
    )
    
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
    
    masker.run(data_path)