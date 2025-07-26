# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Kaggle competition solution for the Eedi Mining Misconceptions in Mathematics challenge. The codebase implements a two-stage pipeline for student misconception detection using retrieval and reranking models.

## Common Commands

### Data Generation
```bash
# Generate synthetic training data using Qwen2.5-72B-Instruct-AWQ
SEED=777
python src/gen/generate_train_72b_awq_100_example.py --seed ${SEED}
```

### Retriever Training
```bash
# Train 32B baseline models (exp010)
for FOLD in 0 1; do
    python src/exp/exp010_fold_${FOLD}.py
done

# Generate inference data for reranker training
for FOLD in 0 1; do
    python src/exp/exp010_infer_gen_fold_${FOLD}.py
done

# Train final submission retrieval models (exp012)
for FOLD in 0 1; do
    python src/exp/exp012_fold_${FOLD}.py
done
```

### Reranker Training
```bash
# Train reranking models with competition + generated data (exp015)
for FOLD in 0 1; do
    python src/exp/exp015_fold_${FOLD}.py
done
```

### Docker Environment
```bash
# Build training environment
cd docker && ./build.sh

# Build vLLM environment for data generation
cd docker_vllm && ./build.sh
```

## Architecture

The solution follows a two-stage retrieval-reranking pipeline:

### Stage 1: Retrieval Models (exp010/exp012)
- **Model**: Qwen2.5-32B-Instruct-GPTQ-Int4 with LoRA fine-tuning
- **Architecture**: BiEncoder model for embedding queries and misconceptions
- **Training**: Cross-entropy loss with negative sampling (96 negatives per batch)
- **Features**: 
  - Last token pooling for sequence embeddings
  - Task-specific instruction formatting
  - Gradient checkpointing for memory efficiency
  - Custom similarity scoring with temperature scaling (×20)

### Stage 2: Reranking Models (exp015)
- **Model**: Qwen2.5-32B-Instruct-GPTQ-Int4 with LoRA fine-tuning
- **Architecture**: Causal language model for listwise reranking
- **Training**: Uses TRL library with completion-only data collation
- **Input**: Candidate misconceptions from Stage 1 retrieval

## Data Processing Pipeline

### Input Data Format
- `train.csv`: Original competition data with questions and misconceptions
- `misconception_mapping.csv`: Maps misconception IDs to descriptions
- Data is pivoted to create individual (question, answer, misconception) triplets

### Synthetic Data Generation
- Uses vLLM with Qwen2.5-72B-Instruct-AWQ for synthetic question generation
- Generates 100 examples per unseen misconception
- Templates include construct/subject information and answer choices
- Output format: structured CSV with all required fields

### Text Formatting
- Questions formatted with instruction template: `"Given a math problem statement and an incorrect answer as a query, retrieve relevant passages that identify and explain the nature of the error."`
- Input structure: `<Question> ... <Correct Answer> ... <Incorrect Answer> ... <Construct> ... <Subject> ... <LLMOutput> ...`

## Key Implementation Details

### Model Configuration
- **Quantization**: GPTQ Int4 for memory efficiency
- **LoRA**: r=32, alpha=64, dropout=0.05
- **Target modules**: All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Pooling**: Last token pooling preferred over mean pooling
- **Optimization**: AdamW with fused operations, linear warmup scheduler

### Training Parameters
- **Batch size**: 16 for retrieval, varies for reranking
- **Learning rate**: 1e-4 base, 5e-4 for LoRA layers
- **Epochs**: 10 for retrieval, 2 for reranking
- **Sequence length**: 144 tokens for queries, 256 for misconceptions
- **Negative sampling**: 96 negatives per training batch

### Hardware Requirements
- CUDA-enabled GPUs required
- Models use gradient checkpointing for memory efficiency
- Mixed precision training with GradScaler
- Supports tensor parallelism for large models

## File Structure

- `src/exp/`: Experiment scripts for training different model configurations
- `src/gen/`: Synthetic data generation scripts using vLLM
- `docker/`: Training environment with PyTorch base image
- `docker_vllm/`: vLLM environment for inference and generation
- `results/`: Output directory for models, predictions, and generated data

## Environment Variables

- `HOME`: Required environment variable pointing to user home directory
- `TOKENIZERS_PARALLELISM`: Set to "true" for parallel tokenization
- Path structure assumes data in `$HOME/data/eedi-mining-misconceptions-in-mathematics/`