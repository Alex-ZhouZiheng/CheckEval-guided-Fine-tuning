"""
Shared configuration for the baseline experiments.

All paths, dataset settings, model defaults, and inference parameters live
here so every script in this folder stays aligned.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ────────────────────────── env ──────────────────────────────
# Walk up to find .env at the project root (thesis/)
_PROJECT_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_PROJECT_ROOT_ENV, override=False)

HF_TOKEN = os.getenv("HF_TOKEN")

# ────────────────────────── paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
WITH_REASON_DIR = DATA_DIR / "with_reason"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKLISTS_DIR = PROJECT_ROOT/ "checklists" / "filtered"

for _dir in [DATA_DIR, RAW_DIR, SPLITS_DIR, WITH_REASON_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────── dataset ──────────────────────────
HF_DATASET_ID = "nvidia/HelpSteer3"
KEEP_DOMAINS = {"general", "stem", "code"}
KEEP_PREFERENCES = {-3, -2, 2, 3}

# ────────────────────────── splits ───────────────────────────
DEV_RATIO = 0.10
SEED = 42

TIER_SIZES = {
    "tier_5k": 5_000,
    "tier_10k": 10_000,
    "tier_20k": 20_000,
}

# ────────────────────────── model / inference ────────────────
JUDGE_MODEL_ID = PROJECT_ROOT/"models"/"Qwen3.5-9B"
# Smaller backbone for the checklist-generator model (two-model pipeline).
GENERATOR_MODEL_ID = PROJECT_ROOT/"models"/"Qwen3.5-4B"

VLLM_ENGINE_KWARGS = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 16384,
    "dtype": "auto",
    "trust_remote_code": True,
    "language_model_only": True,
    "max_num_seqs": 16,
}

# Qwen3.5 thinks by default. Disable thinking for concise judge outputs.
VLLM_CHAT_KWARGS = {
    "chat_template_kwargs": {"enable_thinking": False},
}

GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
}

# ────────────────────────── evaluation ───────────────────────
TIE_THRESHOLD = 0

# ────────────────────────── training / DPO ────────────────────
DPO_DIR = DATA_DIR / "dpo"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
TENSORBOARD_DIR = RESULTS_DIR / "tb_logs"

for _dir in [DPO_DIR, CHECKPOINTS_DIR,TENSORBOARD_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# Training hyperparameters
LEARNING_RATE = 1e-6
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
WARMUP_RATIO = 0.05
MAX_LENGTH = 2048


# DPO
DPO_BETA = 0.1

# ────────────────────────── joint training (DPO + checklist SFT) ──
CHECKLIST_SFT_DIR = DATA_DIR / "checklist_sft"
CHECKLIST_SFT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────── two-model pipeline ────────────────────
# SFT data for the separate checklist-generator and checklist-conditioned judge.
GENERATOR_SFT_DIR = DATA_DIR / "generator_sft"
JUDGE_SFT_DIR = DATA_DIR / "judge_sft"
# Generator inference output: one checklist per sample_id for each split.
GENERATED_CHECKLIST_DIR = DATA_DIR / "generated_checklists"
for _dir in [GENERATOR_SFT_DIR, JUDGE_SFT_DIR, GENERATED_CHECKLIST_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

JOINT_LAMBDA = 0.1          # weight of checklist SFT loss
SFT_MAX_LENGTH = 2048       # max token length for SFT samples

# DeepSpeed ZeRO-2
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

# Logging
WANDB_PROJECT = "Thesis"

# ────────────────────────── dynamic checklist domains ────────────────────────────

DOMAINS = [
    "correctness_completeness",
    "clarity_communication",
    "helpfulness_usefulness",
    "coding_communication_conditional",
    'relevance_instruction_following'
]

DOMAIN_DESCRIPTIONS = {
    "correctness_completeness": (
        "Factual accuracy, logical consistency, coverage of the user's request, "
        "and faithful task execution. Use when the reasoning talks about errors, "
        "missing information, wrong facts, incomplete answers, or whether the "
        "response actually does what was asked."
    ),
    "clarity_communication": (
        "How clearly the response is written: structure, readability, tone, "
        "conciseness, formatting, absence of confusing or ambiguous phrasing. "
        "Use when the reasoning talks about wording, explanation quality, "
        "organization, verbosity, or style."
    ),
    "helpfulness_usefulness": (
        "Whether the response is genuinely useful and actionable for the user, "
        "including relevance, depth, practical value, safety of advice, and "
        "whether it anticipates the user's real need. Use when the reasoning "
        "discusses how useful, relevant, or on-point the answer is."
    ),
    "coding_communication_conditional": (
        "Code-specific quality criteria that only apply when the response "
        "contains code: syntax correctness, runnable examples, consistent "
        "variable/import usage, avoidance of placeholder/pseudo-code, and "
        "clear code comments/explanations. Use ONLY when the reasoning "
        "references code behavior, snippets, syntax, or implementation details."
    ),
    "relevance_instruction_following":(
        "This category evaluates the extent to which the model response directly "
        "addresses the user's query and strictly adheres to all explicit and implicit instructions in the prompt."
        "It assesses whether the content remains fully on-topic, fulfills specific requirements "
        "(such as format, length, style, tone, scope, constraints, or target keywords), and avoids including any irrelevant, extraneous, or off-topic"
    ),
    "coherence_logic":(
        "This category assesses the internal logical consistency, structural organization, and smooth progression of ideas within the response. "
        "It examines whether the content flows logically from one idea to the next, maintains internal consistency without contradictions, "
        "uses appropriate transitions, and presents arguments or information in a clear, well-structured, and easy-to-follow manner."
    )
}