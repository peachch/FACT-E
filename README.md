# FACT-E: Causality-Inspired Evaluation for Trustworthy Chain-of-Thought Reasoning
<p align="center">
  <a href="https://arxiv.org/abs/2506.00519">
    <img src="https://img.shields.io/badge/arXiv-arxiv:2506.00519-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://komarev.com/ghpvc/?username=peachch&label=Page%20Views&color=00FF00" alt="Page Views">
  <img src="https://img.shields.io/github/stars/peachch/CausalAbstain?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/peachch/CausalAbstain?style=social" alt="GitHub forks">
</p>

<p align="center">
  <img src="https://github.com/peachch/CausalAbstain/blob/master/imgs/multilingual_combine%20(1).png" alt="Logo" width="300"/>
</p>

#### This repository is for FACT-E. For more details, please refer to our [paper]([https://arxiv.org/abs/2506.00519](https://arxiv.org/abs/2604.10693)).

## Abstract
Chain-of-Thought (CoT) prompting has improved LLM reasoning, but models often generate explanations that appear coherent while containing unfaithful intermediate steps. Existing self-evaluation approaches are prone to inherent biases: the model may confidently endorse coherence even when the step-to-step implication is not valid, leading to unreliable faithfulness evaluation. We propose FACT-E, a causality-inspired framework for evaluating CoT quality. FACT-E uses controlled perturbations as an instrumental signal to separate genuine step-to-step dependence from bias-driven artifacts, producing more reliable faithfulness estimates (\textit{intra-chain faithfulness}). To select trustworthy trajectories, FACT-E jointly considers \textit{intra-chain faithfulness} and \textit{CoT-to-answer consistency}, ensuring that selected chains are both faithful internally and supportive of the correct final answer. Experiments on GSM8K, MATH, and CommonsenseQA show that FACT-E improves reasoning-trajectory selection and yields stronger in-context learning exemplars. FACT-E also reliably detects flawed reasoning under noisy conditions, providing a robust metric for trustworthy LLM reasoning.
