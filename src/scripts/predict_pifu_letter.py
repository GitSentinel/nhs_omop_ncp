"""
FastPIFU Cardiology Single-Letter Inference

Loads the fine-tuned PIFU LoRA adapter and predicts PIFU eligibility for one clinical letter stored as a UTF-8 text file.

Run:
uv run --extra finetune python src/scripts/predict_pifu.py letter.txt
"""

import argparse
import sys
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Model Libraries
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# PIFU Configuration
from src.config.pifu_settings import (
    PIFU_BASE_MODEL,
    PIFU_ID_TO_LABEL,
    PIFU_MAX_LENGTH,
    PIFU_OUTPUT_DIR,
    PIFU_PROMPT_TEMPLATE,
)

# Label Configuration
LABEL_IDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # CLI Argument Setup
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "letter",
        type=Path,
        help="UTF-8 text file containing one clinical letter.",
    )

    return parser.parse_args()


def load_letter(path: Path) -> str:
    """Load one clinical letter from disk."""

    # Letter Path Validation
    if not path.exists():
        raise FileNotFoundError(f"Clinical letter not found: {path}")

    if not path.is_file():
        raise ValueError(f"Clinical letter path is not a file: {path}")

    # Letter Loading
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Clinical letter is empty: {path}")

    return text


def setup_tokenizer():
    """Load the adapter tokenizer, falling back to the base tokenizer."""

    # Tokeniser Loading
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            PIFU_OUTPUT_DIR,
            trust_remote_code=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            PIFU_BASE_MODEL,
            trust_remote_code=True,
        )

    # Padding Token Setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    return tokenizer


def get_model_device(model) -> torch.device:
    """Return the first real device used by the model."""

    # Device Detection
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    return torch.device("cuda:0")


def load_pifu_model():
    """Load the base model and attach the PIFU LoRA adapter."""

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for PIFU inference.")

    # Adapter Check
    adapter_config_path = PIFU_OUTPUT_DIR / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"PIFU adapter not found at {PIFU_OUTPUT_DIR}. "
            "Run the PIFU fine-tuning script first."
        )

    # Tokeniser Setup
    tokenizer = setup_tokenizer()

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


def get_label_token_ids(tokenizer) -> dict[int, list[int]]:
    """Return token IDs for each valid PIFU class label."""

    # Label Token Construction
    label_token_ids = {
        label: tokenizer.encode(
            f" {label}",
            add_special_tokens=False,
        )
        for label in LABEL_IDS
    }

    # Label Token Validation
    for label, token_ids in label_token_ids.items():
        if not token_ids:
            raise ValueError(f"Label {label} produced no token IDs.")

    if len({tuple(value) for value in label_token_ids.values()}) != len(LABEL_IDS):
        raise ValueError(f"Non-unique label token sequences: {label_token_ids}")

    return label_token_ids


def build_prompt_ids(
    tokenizer,
    text: str,
    max_label_length: int,
) -> list[int]:
    """Build prompt token IDs while preserving space for label tokens."""

    # Prompt Formatting
    prompt = PIFU_PROMPT_TEMPLATE.format(text=text)

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    # Prompt Length Control
    max_prompt_length = PIFU_MAX_LENGTH - max_label_length

    if max_prompt_length < 1:
        raise ValueError(
            f"PIFU_MAX_LENGTH={PIFU_MAX_LENGTH} is too small for the label tokens."
        )

    prompt_ids = prompt_ids[:max_prompt_length]

    if not prompt_ids:
        raise ValueError("The formatted PIFU prompt produced no token IDs.")

    return prompt_ids


def score_label_probabilities(
    model,
    tokenizer,
    text: str,
) -> list[float]:
    """Score labels 0, 1 and 2 using exact label log probabilities."""

    # Scoring Setup
    device = get_model_device(model)
    label_token_ids = get_label_token_ids(tokenizer)

    max_label_length = max(len(token_ids) for token_ids in label_token_ids.values())

    prompt_ids = build_prompt_ids(
        tokenizer,
        text,
        max_label_length,
    )

    sequences = []
    metadata = []

    # Candidate Sequence Construction
    for label in LABEL_IDS:
        candidate_ids = label_token_ids[label]

        sequences.append(prompt_ids + candidate_ids)
        metadata.append((len(prompt_ids), candidate_ids))

    # Tensor Construction
    max_sequence_length = max(len(sequence) for sequence in sequences)

    input_ids = torch.full(
        (len(sequences), max_sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros_like(input_ids)

    for row_index, sequence in enumerate(sequences):
        sequence_length = len(sequence)

        input_ids[row_index, :sequence_length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )

        attention_mask[row_index, :sequence_length] = 1

    # Model Forward Pass
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    scores = []

    # Candidate Log-Probability Calculation
    for row_index, (prompt_length, candidate_ids) in enumerate(metadata):
        score = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        for offset, token_id in enumerate(candidate_ids):
            position = prompt_length + offset - 1

            score = (
                score
                + torch.log_softmax(
                    outputs.logits[
                        row_index,
                        position,
                        :,
                    ].float(),
                    dim=-1,
                )[token_id]
            )

        scores.append(score)

    probabilities = torch.softmax(
        torch.stack(scores),
        dim=0,
    )

    return probabilities.detach().cpu().tolist()


def print_prediction(probabilities: list[float]) -> None:
    """Print the predicted PIFU class and class probabilities."""

    # Prediction Selection
    prediction = int(
        max(
            LABEL_IDS,
            key=lambda label: probabilities[label],
        )
    )

    # Console Output
    print("=" * 60)
    print("PIFU ELIGIBILITY PREDICTION")
    print("=" * 60)
    print(f"Prediction: {PIFU_ID_TO_LABEL[prediction]}")
    print()

    for label in LABEL_IDS:
        print(f"{PIFU_ID_TO_LABEL[label]:<14}: {probabilities[label]:.4f}")


def main() -> None:
    """Run single-letter PIFU inference."""

    # CLI Setup
    args = parse_args()

    # Input Loading
    text = load_letter(args.letter)

    # Model Loading
    model, tokenizer = load_pifu_model()

    # Probability Scoring
    probabilities = score_label_probabilities(
        model,
        tokenizer,
        text,
    )

    # Result Output
    print_prediction(probabilities)


if __name__ == "__main__":
    main()
