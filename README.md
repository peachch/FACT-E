# FACT-E: Causality-Inspired Evaluation for Trustworthy Chain-of-Thought Reasoning
<p align="center">
  <a href="https://arxiv.org/abs/2604.10693">
    <img src="https://img.shields.io/badge/arXiv-arxiv:2604.10693-b31b1b.svg" alt="arXiv"> 
  </a>
  <!-- <img src="https://komarev.com/ghpvc/?username=peachch&label=Page%20Views&color=00FF00" alt="Page Views"> -->
  <img src="https://img.shields.io/github/stars/peachch/FACT-E?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/peachch/FACT-E?style=social" alt="GitHub forks">
</p>

<p align="center">
  <img src="https://github.com/peachch/FACT-E/master/imgs/intro.png" alt="Logo" width="300"/>
</p>

#### This repository is for FACT-E. For more details, please refer to our [paper]([https://arxiv.org/abs/2604.10693]).

## Abstract
Chain-of-Thought (CoT) prompting has improved LLM reasoning, but models often generate explanations that appear coherent while containing unfaithful intermediate steps. Existing self-evaluation approaches are prone to inherent biases: the model may confidently endorse coherence even when the step-to-step implication is not valid, leading to unreliable faithfulness evaluation. We propose FACT-E, a causality-inspired framework for evaluating CoT quality. FACT-E uses controlled perturbations as an instrumental signal to separate genuine step-to-step dependence from bias-driven artifacts, producing more reliable faithfulness estimates (\textit{intra-chain faithfulness}). To select trustworthy trajectories, FACT-E jointly considers \textit{intra-chain faithfulness} and \textit{CoT-to-answer consistency}, ensuring that selected chains are both faithful internally and supportive of the correct final answer. Experiments on GSM8K, MATH, and CommonsenseQA show that FACT-E improves reasoning-trajectory selection and yields stronger in-context learning exemplars. FACT-E also reliably detects flawed reasoning under noisy conditions, providing a robust metric for trustworthy LLM reasoning.


## Features

- **Multiple Evaluation Methods**: Supports Self-Reflection, Self-Consistency, Self-Polish, and Self-Denoise baselines
- **Faithfulness Scoring**: Evaluates whether CoT reasoning actually leads to the answer (consistency × faithfulness)
- **Contrastive CoT Generation**: Generates alternative reasoning paths with specific error types
- **Multi-Dataset Support**: Works with Math500, GSM8K, and CommonsenseQA datasets
- **LLM API Integration**: Unified interface for OpenAI, Qwen, and other compatible APIs

## Project Structure

```
cot_causal/
├── api.py                            # LLM API client wrapper
├── metrics.py                        # Evaluation metrics (accuracy, normalization, etc.)
├── main.py                           # Main FACT-E candidate selection runner
├── baseline/                         # Baseline methods
│   ├── approach_reflect.py           # Self-reflection baseline
│   ├── approach_self_consistency.py  # Self-consistency voting
│   ├── approach_self_polish.py       # Self-polish refinement
│   └── approach_self_denoise.py      # Self-denoise cleaning
├── data/                             # Dataset directory 
│   ├── math500_test_full.json
│   ├── gsm8k_test_full.json
│   └── commonsense_qa_test_full.json
└── README.md
```

## Module Descriptions

### api.py
LLM API client with the following features:
- Supports OpenAI-compatible APIs and Qwen models
- Handles thinking mode for Qwen models
- Implements probability extraction via logprobs or self-reporting
- Provides CoT and answer extraction from responses

**Key Classes:**
- `LLMApiClient`: Main API wrapper
- `ProbabilityResult`: Dataclass for probability extraction results

### metrics.py
Evaluation utilities for CoT reasoning:
- Answer extraction from boxed format or "Answer:" prefix
- Answer normalization (LaTeX, math symbols, case insensitivity)
- Accuracy calculation with tolerance for math problems
- Math-specific evaluation supporting fraction comparison

**Key Methods:**
- `Metrics.evaluate_results()`: Main evaluation entry point
- `Metrics.normalize_answers()`: Normalize answer strings
- `Metrics.extract_answer_from_response()`: Extract answer from model output

### main.py
FACT-E (Faithful And Consistent Evaluation) candidate selection:
- Generates multiple CoT candidates
- Scores candidates by consistency × faithfulness
- Generates contrastive CoTs for faithfulness evaluation
- Saves results to JSON and CSV

**Key Functions:**
- `select_best_cot_for_question()`: Main selection logic
- `estimate_consistency_score()`: Calculate consistency via repeated judgments
- `estimate_faithfulness_score()`: Calculate faithfulness via contrastive CoTs
- `generate_contrastive_cots()`: Generate alternative reasoning paths

### baseline/
Baseline approaches for comparison:
- `approach_reflect.py`: Self-reflection (initial answer + reflection + revised answer)
- `approach_self_consistency.py`: Self-consistency voting (multiple samples + majority vote)
- `approach_self_polish.py`: Self-polish (iterative refinement)
- `approach_self_denoise.py`: Self-denoise (error detection and correction)

## Installation

### Requirements

```bash
pip install openai python-dotenv tqdm pandas scikit-learn numpy
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, for custom endpoints
```

## Usage

### Running Baseline Methods

#### Self-Reflection
```bash
python baseline/approach_reflect.py -m gpt-4o-mini -d math500 -o 0.1 -l 2
```

Arguments:
- `-m, --model`: Model name (default: gpt-4o-mini)
- `-d, --dataset`: Dataset name (math500, commonsense, gsm8k)
- `-o, --portion`: Portion of dataset to use (default: 0.1)
- `-l, --level`: Math difficulty level filter (default: 2)

#### Self-Consistency
```bash
python baseline/approach_self_consistency.py -m qwen3-14b -d math500 -o 1.0 -s 5
```

Arguments:
- `-s, --samples`: Number of samples for self-consistency voting (default: 3)

### Running FACT-E Main Method

```bash
python main.py --dataset math500 --model_name gpt-4o-mini --reasoning_budget 3 --consistency_trials 3 --random_flips 3
```

Arguments:
- `--dataset`: Dataset name (math500, commonsense_qa, gsm8k)
- `--model_name`: Model name for API calls
- `--reasoning_budget`: Number of CoT candidates to generate
- `--consistency_trials`: Number of repeated judgments for consistency scoring
- `--random_flips`: Number of sampled checkpoints for faithfulness estimation
- `--temperature`: Sampling temperature (default: 0.0)
- `--iteration`: Experiment iteration identifier
- `--level`: Math difficulty level filter
- `--max_samples`: Limit number of samples to process
- `--seed`: Random seed (default: 42)

## Output Structure

Results are saved to `output/` directories:

```
output/
└── {model_name}_main_results/
    └── iteration{iteration}/
        ├── {model_name}_{dataset}_cot_results_level{level}.json
        └── {model_name}_{dataset}_cot_results_level{level}.csv
```

### JSON Output Fields

- `config`: Experiment configuration
- `questions`: List of questions processed
- `ground_truth`: Ground truth answers
- `final_answer`: Model's final answers
- `cot_best`: Selected best CoT reasoning
- `predictions`: All candidate answers
- `probabilities`: Reliability scores for each candidate
- `contrastive_cots`: Contrastive CoT generation logs
- `metrics`: Evaluation metrics (accuracy, judge)

## Supported Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| math500 | Mathematical reasoning problems | MATH dataset test split |
| gsm8k | Grade school math word problems | GSM8K dataset |
| commonsense_qa | Commonsense reasoning questions | CommonsenseQA dataset |

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sun2026fact,
  title={FACT-E: Causality-Inspired Evaluation for Trustworthy Chain-of-Thought Reasoning},
  author={Sun, Yuxi and Zuo, Aoqi and Xie, Haotian and Gao, Wei and Gong, Mingming and Ma, Jing},
  journal={arXiv preprint arXiv:2604.10693},
  year={2026}
}
```
