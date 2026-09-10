# NHS OMOP NCP

**Parameter-Efficient Fine-Tuning of Large Language Models for RTT Pathway Validation in the NHS**

Research prototype for working with synthetic OMOP CDM data and synthetic outpatient letters using MCP-based data access, agent orchestration, parameter-efficient fine-tuning, and reproducible evaluation.

> **Research use only.** This repository uses synthetic data. The outputs are intended to support human review and are not designed for automated clinical or operational decision-making.

## Overview

NHS Referral to Treatment (RTT) pathway information can be spread across structured records and unstructured clinical correspondence. This project explores whether large language models can help extract and classify pathway-relevant information while keeping the workflow auditable and suitable for human review.

The repository contains two connected strands:

1. **OMOP agent workflow**: a FastMCP server exposes synthetic OMOP CDM v5.4 data to a LangChain Deep Agents orchestrator. A second MCP server provides specialty-specific FastPIFU guidance.
2. **PEFT / LoRA experiments**: Qwen3.5-9B is adapted with LoRA for outpatient-letter classification, including a three-class PIFU suitability task with held-out and challenge-set evaluation.

MLflow records model runs and artefacts. The repository also contains the frozen tables, figures, statistical analyses, and run metadata used for the dissertation evaluation.

## Architecture

```text
Synthetic OMOP CDM v5.4
        |
        v
+-----------------------+       +--------------------------+
| OMOP FastMCP Server   |       | FastPIFU Skills Server   |
| 9 patient-data tools  |       | specialty guidance      |
+-----------+-----------+       +------------+-------------+
            \                              /
             \                            /
              v                          v
             +----------------------------+
             | LangChain Deep Agent       |
             | tool-based orchestration   |
             +-------------+--------------+
                           |
                           v
                 Structured review output
                           |
                           v
                        MLflow

Synthetic outpatient letters
        |
        v
Qwen3.5-9B base model
        |
        +---- LoRA fine-tuning ----> PEFT-adapted model
        |                              |
        +---------- evaluation <-------+
                       |
                       v
      metrics, confusion matrices, bootstrap CIs,
      McNemar tests, error analysis and figures
```

## Main Components

### OMOP MCP server

[`src/mcp_server/server.py`](src/mcp_server/server.py) exposes nine tools over stdio:

- `get_patient_summary`
- `get_patient_conditions`
- `get_patient_medications`
- `get_patient_visits`
- `get_patient_measurements`
- `get_patient_observations`
- `get_patient_notes`
- `get_patient_procedures`
- `list_available_patients`

The server opens the configured DuckDB database in read-only mode.

### FastPIFU skills server

[`src/mcp_server/skills_server.py`](src/mcp_server/skills_server.py) provides specialty guidance for cardiology, dermatology, ENT, gastroenterology, general surgery, gynaecology, OMFS, ophthalmology, orthopaedics, spinal care, and urology, together with an OMOP clinical-reasoning guide.

The server supports:

- listing available skills;
- loading a skill by specialty;
- matching a condition to a specialty skill; and
- loading the OMOP reasoning guide.

### Deep Agent orchestration

[`src/agents/omop_agent.py`](src/agents/omop_agent.py) connects both MCP servers to a LangChain Deep Agent. The current entry point uses an OpenAI-compatible Azure endpoint configured through environment variables and logs each run to MLflow.

### PIFU classifier

The PIFU workflow uses `Qwen/Qwen3.5-9B` with a three-class label space:

| ID | Label | Meaning |
|---:|---|---|
| 0 | `NOT_ELIGIBLE` | An exclusion criterion or requirement for active/timed specialist management is present. |
| 1 | `BORDERLINE` | Evidence is incomplete, conflicting, or requires clinician review. |
| 2 | `ELIGIBLE` | The letter supports PIFU suitability under the project protocol. |

LoRA is configured with rank `r=8`, alpha `16`, dropout `0.05`, and projection/feed-forward target modules including `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

## Dataset

The project uses synthetic data only.

The processed FastPIFU cardiology dataset currently records:

| Split | Cases |
|---|---:|
| Training | 4,365 |
| Validation | 485 |
| Held-out external evaluation | 150 |
| Challenge set | 24 |

The source development corpus contains 5,000 rows. The 150 held-out evaluation cases are removed before the 4,850-case training pool is split into training and validation data.

Processed FastPIFU files are stored under [`data/external/fastpifu_cardiology/processed/`](data/external/fastpifu_cardiology/processed/).

Raw archives, extracted OMOP files, DuckDB databases, model weights, checkpoints, local MLflow stores, and inference runs are excluded by `.gitignore`.

## Project Structure

```text
nhs_omop_ncp/
├── config.py                         # Shared project and model configuration
├── data/
│   ├── external/fastpifu_cardiology # Processed FastPIFU dataset
│   ├── evaluations/                  # Evaluation outputs
│   ├── finetune/                     # Labelled fine-tuning data
│   └── pipeline_runs/                # Saved pipeline reports
├── notebooks/
│   └── 01_eda_omop.ipynb            # OMOP exploratory analysis
├── results/dissertation_results/     # Frozen dissertation result set
├── scripts/
│   └── extract_omop.py               # Build OMOP DuckDB from synthetic Parquet data
├── src/
│   ├── agents/                       # Deep Agent, PIFU workflows, safety and judge logic
│   ├── config/                       # Runtime and PIFU settings
│   ├── data_access/                  # DuckDB / Ibis OMOP queries
│   ├── inference/                    # PIFU classifier inference
│   ├── mcp_server/                   # OMOP and skills MCP servers
│   └── scripts/                      # Training, evaluation and pipeline CLIs
└── tests/
    └── integration/test_mcp_tools.py # MCP integration tests
```

## Requirements

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- NVIDIA CUDA GPU(s) for the configured PIFU fine-tuning workflow
- A synthetic OMOP CDM v5.4 DuckDB database for the OMOP agent workflow
- An API key for the configured OpenAI-compatible endpoint when using LLM-backed agent or explanation stages

## Installation

Clone the repository:

```bash
git clone https://github.com/GitSentinel/nhs_omop_ncp.git
cd nhs_omop_ncp
```

Install the core environment:

```bash
uv sync
```

Install fine-tuning and development dependencies when required:

```bash
uv sync --extra finetune --extra dev
```

## Configuration

Runtime settings can be supplied in a local `.env` file. `.env` is excluded from version control.

```dotenv
DUCKDB_PATH=data/processed/omop_v54.duckdb

AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.example.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=your-model-deployment

MLFLOW_TRACKING_URI=sqlite:///mlflow_runs/mlflow.db
```

The defaults and training parameters are defined in [`config.py`](config.py) and [`src/config/pifu_settings.py`](src/config/pifu_settings.py).

## Prepare the Synthetic OMOP Database

The OMOP DuckDB file is not committed to the repository. [`scripts/extract_omop.py`](scripts/extract_omop.py) builds `data/processed/omop_v54.duckdb` from a local synthetic `delphi-100k` OMOP v5.4 archive.

The current script contains a machine-specific `ZIP_PATH`. Change that value to the location of your local synthetic archive, then run:

```bash
uv run python scripts/extract_omop.py
```

## Run the OMOP Agent

The Deep Agent starts both MCP servers through stdio when it runs.

```bash
uv run python src/agents/omop_agent.py \
  --pid 17247 \
  --query "Summarise this patient's medical history."
```

You can also start either MCP server directly for development or debugging:

```bash
uv run python src/mcp_server/server.py
uv run python src/mcp_server/skills_server.py
```

## FastPIFU Workflow

### Prepare the dataset

To rebuild the processed PIFU splits, place the source archive at:

```text
data/external/fastpifu_cardiology/raw/fastpifu-cardiology.zip
```

Then run:

```bash
uv run --extra finetune python src/scripts/prepare_fastpifu_dataset.py --force
```

### Fine-tune the PIFU model

```bash
uv run --extra finetune python src/scripts/finetune_pifu.py
```

The current training configuration uses three epochs, BF16, gradient checkpointing, a target global batch size of 16, and automatic multi-GPU selection when enabled.

### Evaluate base and fine-tuned models

```bash
uv run --extra finetune python src/scripts/evaluate_pifu.py
```

The evaluator compares the base model and LoRA adapter on the held-out 150-case set and the 24-case challenge set.

### Run single-letter PIFU inference

From a text file:

```bash
uv run --extra finetune python src/scripts/run_pifu_inference.py \
  --letter-file letter.txt
```

Or pass the letter directly:

```bash
uv run --extra finetune python src/scripts/run_pifu_inference.py \
  --text "Synthetic clinic letter text here"
```

The inference workflow returns a predicted class and confidence, marks the case for human review, and can run explanation and LLM-as-a-Judge stages. Use `--no-llm-explanation` or `--no-llm-judge` to disable those stages.

### Run the end-to-end pipeline

```bash
uv run --extra finetune python src/scripts/run_agentic_pipeline.py \
  --target pifu \
  --mode post-finetune \
  --no-llm-report
```

Available modes include `full`, `train-and-evaluate`, and `post-finetune`.

## Statistical Analysis

The frozen PIFU predictions can be analysed with class-stratified paired bootstrap confidence intervals, exact two-sided McNemar tests, and Wilson score intervals:

```bash
uv run --extra finetune python src/scripts/analyse_pifu_statistics.py
```

The held-out external evaluation is treated as the primary evaluation. Challenge-set results are exploratory.

## Frozen Dissertation Results

The final result snapshot is stored in [`results/dissertation_results/`](results/dissertation_results/). It includes run metadata, evaluation artefacts, dissertation tables and figures, error analysis, statistical tests, and SHA-256 checksums.

Selected results from the frozen snapshot are shown below.

| Experiment | Base | Fine-tuned |
|---|---:|---:|
| Binary task F1 | 0.783 | **0.897** |
| Binary task ROC-AUC | 0.621 | **0.955** |
| PIFU held-out accuracy | 0.353 | **0.947** |
| PIFU held-out macro F1 | 0.299 | **0.947** |

On the 150-case PIFU held-out set, the fine-tuned model recorded 0.96 recall for `NOT_ELIGIBLE`, 0.92 for `BORDERLINE`, and 0.96 for `ELIGIBLE`. The project-specific `unsafe_eligible_count` fell from 20 for the base model to 0 for the fine-tuned model on this evaluation set.

These results describe this synthetic evaluation protocol. They do not establish clinical effectiveness or deployment readiness.

## MLflow

The project uses MLflow for experiment tracking, model/evaluation metadata, agent traces, and saved artefacts. The default local tracking URI is:

```text
sqlite:///mlflow_runs/mlflow.db
```

To inspect local runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow_runs/mlflow.db
```

Then open the local MLflow address printed by the command.

## Tests

Run the test suite with:

```bash
uv run --extra dev pytest
```

The current integration tests cover the MCP data-access tools.

## Safety and Governance

This repository is an MSc research prototype built around synthetic data. Human review remains mandatory for PIFU predictions in the frozen evaluation workflow. Confidence values, margins, safety rules, and LLM-as-a-Judge outputs are research signals rather than autonomous decisions.

Do not treat this repository as a production NHS pathway-validation system. Connecting it to real patient data, source systems, or operational workflows would require separate information-governance, clinical-safety, security, validation, and deployment work.

## Academic Context

Developed as part of an MSc Artificial Intelligence placement project with **Lancashire Teaching Hospitals NHS Foundation Trust** and **Lancaster University**.

Project focus: **Parameter-Efficient Fine-Tuning of Large Language Models for RTT Pathway Validation in the NHS**.
