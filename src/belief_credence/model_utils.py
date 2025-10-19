"""Utilities for loading models and extracting activations/logits."""

from __future__ import annotations

import os
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class ModelWrapper:
    """Wrapper for HuggingFace models with activation extraction.

    Supports extracting hidden states and logits from specified layers.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        token: str | None = None,
    ):
        """Load a HuggingFace model.

        Args:
            model_name: Model identifier (e.g., "meta-llama/Llama-2-70b-hf")
            device: Device to load model on
            load_in_8bit: Use 8-bit quantization to save memory
            token: HuggingFace API token (if None, loads from HF_TOKEN env var)
        """
        self.model_name = model_name
        self.device = device

        if token is None:
            token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "device_map": "auto" if load_in_8bit else None,
            "load_in_8bit": load_in_8bit,
            "token": token,
        }
        if not load_in_8bit:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not load_in_8bit:
            self.model = self.model.to(device)

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.0) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated text (without prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(prompt) :].strip()

    def get_next_token_logits(self, prompt: str) -> torch.Tensor:
        """Get logits for next token prediction.

        Args:
            prompt: Input prompt

        Returns:
            Logits tensor of shape (vocab_size,)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

        return logits

    def get_token_probabilities(self, prompt: str, tokens: list[str]) -> dict[str, float]:
        """Get probabilities for specific tokens.

        Args:
            prompt: Input prompt
            tokens: List of token strings to get probabilities for

        Returns:
            Dict mapping token strings to probabilities
        """
        logits = self.get_next_token_logits(prompt)
        probs = torch.softmax(logits, dim=-1)

        result = {}
        for token in tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
            result[token] = probs[token_id].item()

        return result

    def get_hidden_states(self, text: str, layer: int = -1) -> torch.Tensor:
        """Extract hidden states from a specific layer.

        Args:
            text: Input text
            layer: Layer index (-1 for last layer)

        Returns:
            Hidden states tensor of shape (seq_len, hidden_dim)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer][0]

        return hidden_states

    def get_contrast_activations(
        self, statement: str, negation: str, layer: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hidden states for a statement and its negation.

        Args:
            statement: Original statement
            negation: Negated statement
            layer: Layer index

        Returns:
            Tuple of (statement_activations, negation_activations)
        """
        pos_hidden = self.get_hidden_states(statement, layer)
        neg_hidden = self.get_hidden_states(negation, layer)

        return pos_hidden, neg_hidden
