from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

@dataclass
class ProbabilityResult:
    text: str
    prob_true: Optional[float]
    source: str  # logprob | self_reported | none
    valid: bool


class LLMApiClient:
    """Thin wrapper around the chat completion API."""

    QWEN_MODELS = {"qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"}

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, sleep_seconds: float = 0.1):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.sleep_seconds = sleep_seconds

    def call_model(self, model_name: str, prompt: str, temperature: float = 0.0, max_new_tokens: int = 2048) -> str:
        try:
            kwargs = {
                "model": model_name,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if model_name in self.QWEN_MODELS:
                kwargs["extra_body"] = {"enable_thinking": False}
            else:
                kwargs["max_tokens"] = max_new_tokens

            response = self.client.chat.completions.create(**kwargs)
            time.sleep(self.sleep_seconds)
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            print(f"Error calling model {model_name}: {exc}")
            return ""

    def call_model_with_probability(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
    ) -> ProbabilityResult:
        """
        Returns a unified probability result.

        If the model supports logprobs, use them.
        Otherwise, the prompt should ask the model to emit a probability which is parsed from text.
        """
        if model_name in {"deepseek-v3", *self.QWEN_MODELS}:
            text = self.call_model(model_name, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            prob_true = self._extract_probability_number(text)
            return ProbabilityResult(
                text=text,
                prob_true=prob_true,
                source="self_reported",
                valid=prob_true is not None,
            )

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                logprobs=True,
            )
            time.sleep(self.sleep_seconds)
            content = (response.choices[0].message.content or "").strip()

            token_probs: Dict[str, float] = {}
            logprob_content = getattr(response.choices[0], "logprobs", None)
            if logprob_content and getattr(logprob_content, "content", None):
                for item in logprob_content.content:
                    try:
                        token_probs[item.token] = float(np.exp(item.logprob))
                    except Exception:
                        continue

            prob_true = token_probs.get("True")
            if prob_true is None:
                prob_true = self._extract_probability_number(content)
                source = "self_reported"
                valid = prob_true is not None
            else:
                source = "logprob"
                valid = True

            return ProbabilityResult(text=content, prob_true=prob_true, source=source, valid=valid)
        except Exception as exc:
            print(f"Error calling model with probability {model_name}: {exc}")
            return ProbabilityResult(text="", prob_true=None, source="none", valid=False)

    @staticmethod
    def extract_cot_and_answer(response: str) -> Tuple[str, str]:
        cot = ""
        answer = ""
        if not response:
            return cot, answer

        if "CoT:" in response and "Answer:" in response:
            try:
                cot_start = response.index("CoT:") + len("CoT:")
                answer_start = response.index("Answer:") + len("Answer:")
                cot = response[cot_start:response.index("Answer:")].strip()
                answer = response[answer_start:].strip()
                return cot, answer
            except ValueError:
                pass

        if "Judge:" in response and "Probability:" in response:
            try:
                judge_start = response.index("Judge:") + len("Judge:")
                prob_start = response.index("Probability:") + len("Probability:")
                cot = response[judge_start:response.index("Probability:")].strip()
                answer = response[prob_start:].strip()
                return cot, answer
            except ValueError:
                pass

        return response.strip(), ""

    @staticmethod
    def extract_answer(response: str) -> str:
        if not response:
            return ""
        if "Answer:" in response:
            try:
                answer_start = response.index("Answer:") + len("Answer:")
                return response[answer_start:].strip()
            except ValueError:
                return ""
        return response.strip()

    @staticmethod
    def _extract_probability_number(text: str) -> Optional[float]:
        if not text:
            return None
        matches = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)", text)
        for match in matches:
            try:
                value = float(match)
            except ValueError:
                continue
            if 0.0 <= value <= 1.0:
                return value
        return None
