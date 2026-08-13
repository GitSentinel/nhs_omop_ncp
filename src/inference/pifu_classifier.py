"""
Reusable FastPIFU Classifier Module

Loads the fine-tuned FastPIFU LoRA adapter and provides deterministic
three-class PIFU eligibility prediction for clinic letters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# Data and Model Libraries
import numpy as np
import torch

# PEFT Model Loading
from peft import PeftModel

# Transformer Components
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# PIFU Configuration
from src.config.pifu_settings import (
    PIFU_BASE_MODEL,
    PIFU_EVAL_BATCH_SIZE,
    PIFU_ID_TO_LABEL,
    PIFU_LABEL_IDS,
    PIFU_MAX_LENGTH,
    PIFU_OUTPUT_DIR,
    PIFU_PROMPT_TEMPLATE,
)


@dataclass(frozen=True)
class PIFUClassifierResult:
    """One deterministic PIFU classifier result."""

    predicted_label: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    def model_dict(self) -> dict:
        """Return a JSON-friendly dictionary."""

        # Dataclass Serialisation
        return asdict(self)


def setup_tokenizer(
    model_path: str | Path,
):
    """Load and configure the PIFU tokenizer."""

    # Tokeniser Loading
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Padding Token Setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    return tokenizer


def get_model_device(
    model,
) -> torch.device:
    """Return the first real model device."""

    # Device Detection
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    return torch.device("cuda:0")


def load_base_model():
    """Load the unfine-tuned PIFU base model."""

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for PIFU inference.")

    # Tokeniser Loading
    tokenizer = setup_tokenizer(PIFU_BASE_MODEL)

    # Base Model Loading
    model = AutoModelForCausalLM.from_pretrained(
        PIFU_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    return model, tokenizer


def load_finetuned_model():
    """Load the base model with the PIFU LoRA adapter."""

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for PIFU inference.")

    # Adapter Validation
    adapter_config_path = PIFU_OUTPUT_DIR / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"PIFU adapter config was not found: {adapter_config_path}"
        )

    # Tokeniser Loading
    try:
        tokenizer = setup_tokenizer(PIFU_OUTPUT_DIR)

    except OSError:
        tokenizer = setup_tokenizer(PIFU_BASE_MODEL)

    # Base Model Loading
    base_model = AutoModelForCausalLM.from_pretrained(
        PIFU_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    # Adapter Loading
    model = PeftModel.from_pretrained(
        base_model,
        str(PIFU_OUTPUT_DIR),
        is_trainable=False,
    )

    model.eval()

    return model, tokenizer


def label_token_sequences(
    tokenizer,
) -> dict[int, list[int]]:
    """Return the exact token sequences for labels 0, 1 and 2."""

    # Label Token Construction
    sequences = {
        label: tokenizer.encode(
            f" {label}",
            add_special_tokens=False,
        )
        for label in PIFU_LABEL_IDS
    }

    # Label Token Validation
    for label, token_ids in sequences.items():
        if not token_ids:
            raise ValueError(f"Label {label} produced no token IDs.")

    unique_sequences = {
        tuple(value)
        for value in sequences.values()
    }

    if len(unique_sequences) != len(PIFU_LABEL_IDS):
        raise ValueError(
            f"PIFU label token sequences are not unique: {sequences}"
        )

    return sequences


def build_prompt_ids(
    tokenizer,
    text: str,
    max_label_length: int,
) -> list[int]:
    """Format and tokenise one clinic letter."""

    # Prompt Formatting
    prompt = PIFU_PROMPT_TEMPLATE.format(
        text=str(text),
    )

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    # Prompt Length Control
    max_prompt_length = PIFU_MAX_LENGTH - max_label_length

    if max_prompt_length < 1:
        raise ValueError(
            "PIFU_MAX_LENGTH is too small for the label token sequences."
        )

    prompt_ids = prompt_ids[:max_prompt_length]

    if not prompt_ids:
        raise ValueError("The PIFU prompt produced no tokens.")

    return prompt_ids


def score_probabilities(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = PIFU_EVAL_BATCH_SIZE,
    verbose: bool = True,
) -> np.ndarray:
    """Calculate probabilities for the three PIFU classes."""

    # Input Validation
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not texts:
        return np.empty(
            (
                0,
                len(PIFU_LABEL_IDS),
            ),
            dtype=float,
        )

    # Scoring Setup
    model.eval()

    device = get_model_device(model)
    token_map = label_token_sequences(tokenizer)

    max_label_length = max(
        len(tokens)
        for tokens in token_map.values()
    )

    all_probabilities: list[list[float]] = []

    if verbose:
        print(
            "PIFU label token sequences: "
            + ", ".join(
                f"{label}={token_map[label]}"
                for label in PIFU_LABEL_IDS
            )
        )

    # Batch Scoring Loop
    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch_texts = texts[start:start + batch_size]

        sequences: list[list[int]] = []
        metadata: list[tuple[int, list[int]]] = []

        # Candidate Sequence Construction
        for text in batch_texts:
            prompt_ids = build_prompt_ids(
                tokenizer,
                text,
                max_label_length,
            )

            for label in PIFU_LABEL_IDS:
                candidate_ids = token_map[label]

                sequences.append(
                    prompt_ids
                    + candidate_ids
                )

                metadata.append(
                    (
                        len(prompt_ids),
                        candidate_ids,
                    )
                )

        # Tensor Construction
        max_sequence_length = max(
            len(sequence)
            for sequence in sequences
        )

        input_ids = torch.full(
            (
                len(sequences),
                max_sequence_length,
            ),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.zeros_like(input_ids)

        for row_index, sequence in enumerate(sequences):
            sequence_length = len(sequence)

            input_ids[
                row_index,
                :sequence_length,
            ] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )

            attention_mask[
                row_index,
                :sequence_length,
            ] = 1

        # Model Forward Pass
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        candidate_scores = []

        # Candidate Log-Probability Calculation
        for row_index, (prompt_length, candidate_ids) in enumerate(metadata):
            score = torch.zeros(
                (),
                dtype=torch.float32,
                device=device,
            )

            for offset, token_id in enumerate(candidate_ids):
                logits_position = prompt_length + offset - 1

                logits = outputs.logits[
                    row_index,
                    logits_position,
                    :,
                ].float()

                token_log_probability = torch.log_softmax(
                    logits,
                    dim=-1,
                )[token_id]

                score = score + token_log_probability

            candidate_scores.append(score)

        # Class Probability Calculation
        score_matrix = torch.stack(candidate_scores).reshape(
            len(batch_texts),
            len(PIFU_LABEL_IDS),
        )

        probabilities = torch.softmax(
            score_matrix,
            dim=-1,
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy().tolist()
        )

        if verbose:
            completed = min(
                start + len(batch_texts),
                len(texts),
            )

            print(f"{completed:,}/{len(texts):,} evaluated")

        # Batch Memory Cleanup
        del outputs
        del input_ids
        del attention_mask
        del score_matrix
        del probabilities

    return np.asarray(
        all_probabilities,
        dtype=float,
    )


def probability_dict(
    row: np.ndarray,
) -> dict[str, float]:
    """Convert one probability row into a class-name dictionary."""

    # Probability Mapping
    return {
        PIFU_ID_TO_LABEL[label].lower(): float(row[index])
        for index, label in enumerate(PIFU_LABEL_IDS)
    }


class PIFUClassifier:
    """Reusable fine-tuned PIFU classifier."""

    def __init__(self) -> None:
        """Load the fine-tuned PIFU model and tokenizer."""

        # Model Loading
        self.model, self.tokenizer = load_finetuned_model()

    def predict(
        self,
        text: str,
    ) -> PIFUClassifierResult:
        """Predict PIFU suitability for one clinic letter."""

        # Input Validation
        cleaned_text = str(text).strip()

        if not cleaned_text:
            raise ValueError("Clinic letter must not be empty.")

        # Probability Scoring
        matrix = score_probabilities(
            self.model,
            self.tokenizer,
            [cleaned_text],
            batch_size=1,
            verbose=False,
        )

        row = matrix[0]

        # Prediction Selection
        predicted_index = int(np.argmax(row))
        predicted_label = int(PIFU_LABEL_IDS[predicted_index])

        probabilities = probability_dict(row)

        return PIFUClassifierResult(
            predicted_label=predicted_label,
            predicted_class=PIFU_ID_TO_LABEL[predicted_label],
            confidence=float(row[predicted_index]),
            probabilities=probabilities,
        )